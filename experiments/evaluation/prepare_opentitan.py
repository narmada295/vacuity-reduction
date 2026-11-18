"""Build an input-only OpenTitan evaluation set from original RTL and documentation."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET

from experiments.evaluation.generate import file_hash, inference_messages, write_json
from experiments.evaluation.opentitan_sources import CASES, RELEASE, REVISION, ROOT, UPSTREAM, code_only
from experiments.training.prepare_sft import SYSTEM_PROMPT


PRIOR = Path('output/fixme_svagen/evaluation_one_shot_v4/full_test_protocol.json')
DATA = ROOT / 'data'
REMOTE_DATA = 'data/opentitan_v4'


def metadata_signals(ip, folder, hjson_doc):
    xml = ET.parse(folder / 'design.xml')
    top = next(m for m in xml.findall('.//module') if m.get('topModule') == '1')
    declarations = [v.attrib for v in top.findall('var') if v.get('dir')]
    primary = 'clk_core_i' if ip == 'pwm' else 'clk_i'
    reset = 'rst_core_ni' if ip == 'pwm' else 'rst_ni'
    ports = {d['name']: d for d in declarations}
    if primary not in ports or reset not in ports:
        raise ValueError('Selected module is missing its declared clock or reset')
    pins = {r['name']: r['desc'] for key in ('available_input_list', 'available_output_list', 'available_inout_list')
            for r in hjson_doc.get(key, [])}
    interrupts = {r['name']: r['desc'] for r in hjson_doc.get('interrupt_list', [])}
    result = []
    for port in declarations:
        name = port['name']
        label = re.sub(r'^(?:cio_|intr_)', '', name)
        label = re.sub(r'_(?:i|o|en_o)$', '', label)
        author_description = interrupts.get(label) if name.startswith('intr_') else pins.get(label)
        provenance = 'upstream HJSON description' if author_description else 'upstream RTL port declaration'
        if name == primary:
            description = 'Clock input for this module. Sequential logic uses its rising edge.'
        elif name == reset:
            description = 'Active-low reset input, used asynchronously by the sequential logic.'
        elif name == 'reg2hw':
            description = 'Register-to-hardware configuration bundle; the original register definitions and field descriptions are included in the specification.'
        elif name == 'hw2reg':
            description = 'Hardware-to-register update bundle; its type and fields are declared in the included register package.'
        elif author_description:
            description = author_description
        else:
            description = '{} port {} of type {} in {}. Its connections and use are present in the supplied original RTL.'.format(
                port['dir'], name, port.get('vartype', 'logic'), top.get('origName'))
        result.append({'name': name, 'information': description, 'direction': port['dir'],
                       'declared_type': port.get('vartype'), 'description_source': provenance})
    return result, primary, reset


def make_task(case, signal):
    context = case['formal_context']
    user = ('Full specification:\n' + case['spec']
            + '\n\nFull RTL:\n```systemverilog\n' + case['rtl'] + '\n```'
            + '\n\nTarget signal: ' + signal['name']
            + '\nSaved specification-based signal description: ' + signal['information']
            + '\n\nFormal checking context:\nChecker clock declarations:\nclock ' + context['clock']
            + '\nReset policy: reset -none. No global reset sequence or extra input assumptions '
              'were imposed for final labeling. Assertion-local clock events and disable iff '
              'conditions retain their own meaning.'
            + '\nGenerate one assertion for this signal.')
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]
    digest = hashlib.sha256(json.dumps([case['case_id'], signal['name'], messages], sort_keys=True).encode()).hexdigest()
    return {'task_id': case['case_id'].replace('/', '__') + '__' + digest[:20],
            'case_id': case['case_id'], 'target_signal': signal['name'], 'messages': messages}


def prepare_data():
    import hjson
    DATA.mkdir(parents=True, exist_ok=True)
    previous = json.loads(PRIOR.read_text())
    reports = json.loads((ROOT / 'source_preparation.json').read_text())
    sources = {r['ip']: r for r in reports}
    tasks, designs, files = [], [], {}
    original_files = set()
    for ip, top in CASES:
        source = sources[ip]
        folder = ROOT / 'cases' / ip
        destination = DATA / 'cases' / ip
        destination.mkdir(parents=True, exist_ok=True)
        author_spec = UPSTREAM / 'hw/ip' / ip / 'doc/_index.md'
        registers = UPSTREAM / 'hw/ip' / ip / 'data' / (ip + '.hjson')
        hjson_doc = hjson.loads(registers.read_text())
        signals, clock, reset = metadata_signals(ip, folder, hjson_doc)
        # Functional requirements are the authors' complete specification and HJSON.
        # Only provenance and the selected module boundary are added above them.
        spec = ('OpenTitan source release: ' + RELEASE + '\nRevision: ' + REVISION
                + '\nSelected RTL module: ' + top + '\nEvaluation scope: ' + source['scope']
                + '\nThe following original specification describes the parent IP. Apply its '
                  'requirements only to behavior implemented within the selected module boundary.\n\n'
                + author_spec.read_text()
                + '\n\n## Original upstream register and interface definition\n\n```hjson\n'
                + registers.read_text() + '\n```\n')
        context = {'clock': clock, 'checker_clocks': [clock], 'reset_signal': reset,
                   'formal_reset_policy': 'unconstrained', 'formal_clocking': None}
        record = {'case_id': 'OpenTitan/' + ip, 'top': top, 'spec': spec,
                  'rtl': (folder / 'dut.sv').read_text(), 'signals': signals, 'formal_context': context,
                  'source': source, 'specification_source': 'original upstream doc/_index.md and data HJSON',
                  'target_policy': 'Every top-level port of the selected module; packed structures remain one target each.'}
        if re.search(r'\b(?:assert|assume|cover)\s*(?:property|\()', code_only(record['rtl'])):
            raise ValueError('Original verification statement leaked into inference RTL')
        for name, text in (('spec.md', spec), ('dut.sv', record['rtl'])):
            (destination / name).write_text(text)
        for name in ('source_manifest.json', 'top_original.sv', 'compile.log'):
            shutil.copyfile(folder / name, destination / name)
        shutil.copyfile(author_spec, destination / 'upstream_spec.md')
        shutil.copyfile(registers, destination / 'upstream_registers.hjson')
        write_json(destination / 'signals.json', signals)
        write_json(destination / 'case.json', {k: v for k, v in record.items() if k not in ('rtl', 'spec')}
                   | {'rtl_file': 'dut.sv', 'spec_file': 'spec.md'})
        designs.append(record)
        for signal in signals:
            tasks.append(make_task(record, signal))
        original_files.update((r['path'] for r in source['source_files'] if r['original_upstream_file']))
        original_files.update((str(author_spec.relative_to(UPSTREAM)), str(registers.relative_to(UPSTREAM))))
    # Verify selected module names are absent from the earlier fine-tuning/evaluation cohort.
    prior_names = set()
    old = Path('output/fixme_svagen/sft_before_reduction_v2_complete/examples.jsonl')
    seen = set()
    with old.open() as handle:
        for line in handle:
            item = json.loads(line)
            if item['case_id'] not in seen:
                seen.add(item['case_id'])
                prior_names.update(re.findall(r'\bmodule\s+(\w+)', code_only(item['rtl'])))
    if {top for _, top in CASES} & prior_names:
        raise ValueError('A selected OpenTitan module already appears in the earlier cohort')
    for name, values in (('prompts.jsonl', tasks), ('designs.jsonl', designs)):
        (DATA / name).write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))
    upstream_files = []
    original_files.update(str(p.relative_to(UPSTREAM)) for p in (UPSTREAM / 'hw/ip/prim/util').rglob('*')
                          if p.is_file() and p.suffix in ('.py', '.tpl'))
    original_files.add('util/check_tool_requirements.py')
    for relative in sorted(original_files | {'LICENSE', 'NOTICE'}):
        path = UPSTREAM / relative
        if not path.is_file():
            if relative == 'NOTICE':
                continue
            raise ValueError('Missing upstream source: ' + relative)
        destination = DATA / 'upstream' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        upstream_files.append({'path': relative, 'sha256': file_hash(path),
                               'url': 'https://github.com/lowRISC/opentitan/blob/' + REVISION + '/' + relative})
    generated = DATA / 'upstream_generated'
    shutil.copytree(ROOT / 'generated_primitives', generated, dirs_exist_ok=True)
    for path in sorted(DATA.rglob('*')):
        if path.is_file() and path.name not in ('manifest.json', 'protocol.json'):
            files[str(path.relative_to(DATA))] = file_hash(path)
    manifest = {'cohort': 'opentitan_six_v4', 'split': 'test', 'designs': len(designs),
                'revision': REVISION, 'release': RELEASE, 'signals_per_model': len(tasks),
                'signals_by_design': dict(Counter(t['case_id'] for t in tasks)),
                'models': ['base', 'sft', 'kto'], 'total_generation_requests': len(tasks) * 3,
                'no_reference_assertions_or_outcome_labels': True,
                'specification_policy': 'Original author-written IP specification and register HJSON; scope/provenance headings added, no new functional requirements.',
                'rtl_policy': 'Original RTL dependency closure, official generic primitive generation, SYNTHESIS preprocessing to suppress existing verification macros; no functional RTL edits.',
                'target_policy': designs[0]['target_policy'],
                'previous_35_designs_excluded': sorted(seen), 'previous_top_modules': sorted(prior_names),
                'selected_top_modules_disjoint_from_previous_cohort': True,
                'source_files': upstream_files, 'cases': [
                    {'case_id': d['case_id'], 'top': d['top'], 'signals': len(d['signals']),
                     'top_module_lines': d['source']['top_module_lines'],
                     'dependency_source_lines': d['source']['all_dependency_lines'],
                     'rtl_characters': len(d['rtl']), 'spec_characters': len(d['spec']),
                     'compile_status': 'PASS', 'formal_context': d['formal_context']} for d in designs],
                'files': files}
    write_json(DATA / 'manifest.json', manifest)
    for task in tasks:
        inference_messages(task['messages'], 12)
    return manifest


def package(manifest):
    previous = json.loads(PRIOR.read_text())
    protocol = {k: copy.deepcopy(previous[k]) for k in (
        'split', 'prompt_version', 'generator', 'training_config', 'training_config_sha256',
        'generation', 'system_prompt_sha256', 'expected_training', 'expected_packages', 'models')}
    protocol.update(protocol='frozen_opentitan_six_v4', cohort='opentitan_six_v4',
                    release=RELEASE, upstream_revision=REVISION,
                    signal_count=manifest['signals_per_model'], signals_by_design=manifest['signals_by_design'],
                    test_designs=list(manifest['signals_by_design']), excluded_designs=manifest['previous_35_designs_excluded'],
                    prompts=REMOTE_DATA + '/prompts.jsonl', output_root='outputs/eval_one_shot_v4/opentitan_test',
                    log_root='logs/eval_one_shot_v4/opentitan_test',
                    requests_per_signal=1, refinement=False,
                    reference_assertions_or_labels_in_inference_inputs=False,
                    selection=manifest['target_policy'])
    scripts = Path('experiments/evaluation')
    protocol['files'] = {}
    for name in ('generate.py', 'prompts.py', 'run_full.py'):
        key = 'scripts/eval_one_shot_v4/' + name
        if file_hash(scripts / name) != previous['files'][key]:
            raise ValueError('Established v4 generation code changed: ' + name)
        protocol['files'][key] = previous['files'][key]
    protocol['files']['scripts/eval_one_shot_v4/run_opentitan.py'] = file_hash(scripts / 'run_opentitan.py')
    for relative, digest in manifest['files'].items():
        protocol['files'][REMOTE_DATA + '/' + relative] = digest
    protocol['files'][REMOTE_DATA + '/manifest.json'] = file_hash(DATA / 'manifest.json')
    write_json(DATA / 'protocol.json', protocol)
    bundle = ROOT / 'remote_opentitan_six_v4.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for path in sorted(DATA.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=REMOTE_DATA + '/' + str(path.relative_to(DATA)))
        for name in ('generate.py', 'prompts.py', 'run_full.py', 'run_opentitan.py'):
            archive.add(scripts / name, arcname='scripts/eval_one_shot_v4/' + name)
        archive.add('experiments/evaluation/README.md', arcname='OPENTITAN_SIX_V4_README.md')
    bundle.with_name(bundle.name + '.sha256').write_text(file_hash(bundle) + '  ' + bundle.name + '\n')
    report = {'status': 'PASS', 'cohort': protocol['cohort'], 'designs': manifest['designs'],
              'signals_per_model': manifest['signals_per_model'], 'signals_by_design': manifest['signals_by_design'],
              'total_generation_requests': manifest['total_generation_requests'],
              'all_six_rtl_configurations_compiled': True,
              'input_only_dataset_verified': True, 'original_specification_files_preserved': True,
              'generation_code_and_prompt_match_previous_v4': True,
              'bundle': str(bundle), 'bundle_sha256': file_hash(bundle),
              'gpu_generation': 'Run the supplied instructions on black6000; no GPU run performed locally.'}
    write_json(ROOT / 'preparation_verification.json', report)
    print(json.dumps(report, indent=2))
    return report


def main():
    package(prepare_data())


if __name__ == '__main__':
    main()
