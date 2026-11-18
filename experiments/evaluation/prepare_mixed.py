"""Prepare four OpenTitan and two unused FIXME cases with documented internals."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import xml.etree.ElementTree as ET

from experiments.evaluation.generate import file_hash, inference_messages, write_json
from experiments.evaluation.opentitan_sources import ROOT as OT_ROOT, REVISION, code_only
from experiments.evaluation.prepare_opentitan import make_task
from fixme_svagen.generate import NebiusClient
from fixme_svagen.signal_mapping import POLICY, map_documented_signals


ROOT = Path('output/external_mixed_v4')
DATA = ROOT / 'data'
FIXME = Path('/home/narmada/btp/fixme')
COHORT = 'external_mixed_six_v4'
REMOTE_DATA = 'data/external_mixed_v4'
PRIOR = Path('output/fixme_svagen/evaluation_one_shot_v4/full_test_protocol.json')
MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
OPEN_CASES = ('gpio', 'pwm', 'uart', 'i2c')
FIXME_CASES = (('8', 'tessera_sdram_ctrl', ['clk'], 'rst'),
               ('9', 'eth_macstatus', ['MRxClk', 'MTxClk'], 'Reset'))


def candidates_from_xml(path):
    xml = ET.parse(path)
    top = next(m for m in xml.findall('.//module') if m.get('topModule') == '1')
    result = []
    for var in top.findall('var'):
        name = var.get('origName', var.get('name'))
        if (var.get('param') or var.get('localparam') or name.startswith('__')
                or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', name)):
            continue
        result.append({'name': name, 'kind': 'port' if var.get('dir') else 'internal',
                       'direction': var.get('dir'), 'declared_type': var.get('vartype')})
    if len(result) != len({v['name'] for v in result}):
        raise ValueError('Ambiguous top-module signal declarations')
    return result


def load_cases():
    cases = []
    prior = {item['case_id']: item for item in map(json.loads, (OT_ROOT / 'data/designs.jsonl').read_text().splitlines())}
    for ip in OPEN_CASES:
        record = copy.deepcopy(prior['OpenTitan/' + ip])
        record.pop('signals')
        record.pop('target_policy')
        record['_xml'] = str(OT_ROOT / 'cases' / ip / 'design.xml')
        record['_original_dir'] = str(OT_ROOT / 'data/cases' / ip)
        cases.append(record)
    revision = subprocess.check_output(['git', '-C', str(FIXME), 'rev-parse', 'HEAD'], text=True).strip()
    for number, top, clocks, reset in FIXME_CASES:
        original = FIXME / 'data/SVAGen/L3/Text' / number
        spec, rtl = next(original.glob('*.txt')), next(original.glob('*.v'))
        folder = ROOT / 'source_checks' / top
        folder.mkdir(parents=True, exist_ok=True)
        xml = folder / 'design.xml'
        command = ['verilator', '--xml-only', '--xml-output', str(xml), '--Mdir', str(folder / 'obj_dir'),
                   '-Wno-fatal', '--top-module', top, str(rtl)]
        run = subprocess.run(command, text=True, capture_output=True)
        (folder / 'compile.log').write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError('FIXME compilation failed: ' + top)
        source = {'repository': 'https://github.com/ChatDesignVerification/FIXME', 'revision': revision,
                  'compile_status': 'PASS', 'functional_rtl_edits': False,
                  'compile_tool': subprocess.check_output(['verilator', '--version'], text=True).strip(),
                  'top_module_lines': len(rtl.read_text().splitlines()),
                  'all_dependency_lines': len(rtl.read_text().splitlines()),
                  'source_files': [{'path': str(p.relative_to(FIXME)), 'sha256': file_hash(p),
                                    'url': 'https://github.com/ChatDesignVerification/FIXME/blob/' + revision + '/' + str(p.relative_to(FIXME))}
                                   for p in (rtl, spec)]}
        context = {'clock': clocks[0], 'checker_clocks': clocks, 'reset_signal': reset,
                   'formal_reset_policy': 'unconstrained', 'formal_clocking': None}
        cases.append({'case_id': 'L3/' + number, 'top': top, 'spec': spec.read_text(), 'rtl': rtl.read_text(),
                      'formal_context': context, 'source': source,
                      'specification_source': 'Original FIXME SVAGen Text specification, unchanged',
                      '_xml': str(xml), '_original_dir': str(original), '_compile_log': str(folder / 'compile.log')})
    split = json.loads(Path('output/fixme_svagen/sft_before_reduction_v2_complete/split_plan.json').read_text())
    excluded = set().union(*(split[k] for k in ('train', 'validation', 'test')))
    observed = {r['case_id'] for r in cases}
    if observed & excluded or len(observed) != 6:
        raise ValueError('New evaluation cohort overlaps the previous saved splits')
    old_modules = set()
    with Path('output/fixme_svagen/sft_before_reduction_v2_complete/examples.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            old_modules.update(re.findall(r'\bmodule\s+(\w+)', code_only(row['rtl'])))
    if {r['top'] for r in cases} & old_modules:
        raise ValueError('Selected top module occurs in the previous cohort')
    return cases, sorted(excluded)


def map_case(case):
    candidates = candidates_from_xml(case['_xml'])
    work = ROOT / 'mapping' / case['case_id'].replace('/', '__')
    work.mkdir(parents=True, exist_ok=True)
    write_json(work / 'candidates.json', candidates)
    client = NebiusClient(Path('.env'), model=MODEL, max_tokens=12288)
    entries = map_documented_signals(client, case['spec'], case['rtl'], candidates, work)
    write_json(work / 'selection_audit.json', entries)
    signals = [entry for entry in entries if entry['selected']]
    write_json(work / 'signals.json', signals)
    print('MAPPED', case['case_id'], len(signals), 'targets;',
          sum(s['kind'] == 'internal' for s in signals), 'internal', flush=True)
    return signals


def task_for_case(case, signal):
    # Keep the established SFT-format input and unchanged inference transformer.
    # make_task prepends "clock "; represent the complete two-clock command list.
    record = dict(case, formal_context=dict(case['formal_context']))
    record['formal_context']['clock'] = '\nclock '.join(case['formal_context']['checker_clocks'])
    return make_task(record, signal)


def prepare_data(cases, excluded, workers):
    DATA.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        mappings = list(pool.map(map_case, cases))
    tasks, records = [], []
    sources = set()
    for case, signals in zip(cases, mappings):
        cid = case['case_id']
        destination = DATA / 'cases' / cid.replace('/', '__')
        destination.mkdir(parents=True, exist_ok=True)
        record = {k: v for k, v in case.items() if not k.startswith('_')}
        record['signals'] = signals
        record['target_policy'] = POLICY
        if re.search(r'\b(?:assert|assume|cover)\s*(?:property|\()', code_only(record['rtl'])):
            raise ValueError('Existing verification statements in inference RTL: ' + cid)
        names = {s['name'] for s in signals}
        if not set(record['formal_context']['checker_clocks'] + [record['formal_context']['reset_signal']]) <= names:
            raise ValueError('Selected clock/reset missing from targets')
        (destination / 'dut.sv').write_text(case['rtl'])
        (destination / 'spec.md').write_text(case['spec'])
        shutil.copyfile(case['_xml'], destination / 'declarations.xml')
        write_json(destination / 'signals.json', signals)
        write_json(destination / 'case.json', {k: v for k, v in record.items() if k not in ('spec', 'rtl')}
                   | {'rtl_file': 'dut.sv', 'spec_file': 'spec.md'})
        shutil.copytree(ROOT / 'mapping' / cid.replace('/', '__'), destination / 'mapping_audit', dirs_exist_ok=True)
        if cid.startswith('OpenTitan/'):
            original = Path(case['_original_dir'])
            for name in ('top_original.sv', 'upstream_spec.md', 'upstream_registers.hjson', 'source_manifest.json', 'compile.log'):
                shutil.copyfile(original / name, destination / name)
            sources.update(r['path'] for r in case['source']['source_files'] if r['original_upstream_file'])
            ip = cid.split('/')[-1]
            sources.update(('hw/ip/' + ip + '/doc/_index.md', 'hw/ip/' + ip + '/data/' + ip + '.hjson'))
        else:
            original = Path(case['_original_dir'])
            for path in original.iterdir():
                if path.suffix in ('.v', '.txt'):
                    shutil.copyfile(path, destination / ('upstream_' + path.name))
            shutil.copyfile(case['_compile_log'], destination / 'compile.log')
        records.append(record)
        tasks.extend(task_for_case(record, signal) for signal in signals)
    # Include original dependency files and official primitive-generation evidence.
    for relative in sorted(sources | {'LICENSE'}):
        original = OT_ROOT / 'data/upstream' / relative
        destination = DATA / 'upstream/opentitan' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
    shutil.copytree(OT_ROOT / 'data/upstream_generated', DATA / 'upstream/opentitan_generated', dirs_exist_ok=True)
    for original in (OT_ROOT / 'data/upstream/hw/ip/prim/util').rglob('*'):
        if original.is_file():
            destination = DATA / 'upstream/opentitan/hw/ip/prim/util' / original.relative_to(OT_ROOT / 'data/upstream/hw/ip/prim/util')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
    for name in ('LICENSE', 'LICENSE.md', 'LICENSE.txt', 'NOTICE'):
        if (FIXME / name).is_file():
            destination = DATA / 'upstream/fixme' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(FIXME / name, destination)
    for name, rows in (('prompts.jsonl', tasks), ('designs.jsonl', records)):
        (DATA / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    for task in tasks:
        inference_messages(task['messages'], 12)
    files = {str(path.relative_to(DATA)): file_hash(path) for path in sorted(DATA.rglob('*'))
             if path.is_file() and path.name not in ('manifest.json', 'protocol.json')}
    manifest = {'cohort': COHORT, 'split': 'test', 'designs': len(records),
                'signals_per_model': len(tasks), 'total_generation_requests': len(tasks) * 3,
                'signals_by_design': dict(Counter(t['case_id'] for t in tasks)),
                'target_policy': POLICY, 'mapping_model': MODEL,
                'mapping_configuration': {'temperature': 0.0, 'max_output_tokens': 12288,
                                          'batch_size': 40,
                                          'mapper_script_sha256': file_hash('fixme_svagen/signal_mapping.py'),
                                          'client_script_sha256': file_hash('fixme_svagen/generate.py'),
                                          'unsupported_internal_proposals': 'Excluded with original proposal preserved in audit',
                                          'selected_internal_evidence': 'Exact normalized source substring, 8 to 400 characters'},
                'descriptions': 'DeepSeek mapping with exact declaration and source-quotation validation; semantic review is not a formal proof.',
                'reference_assertions_or_outcome_labels_in_inference_inputs': False,
                'previous_35_designs_excluded': excluded,
                'original_functional_rtl_and_specifications': True,
                'cases': [{'case_id': r['case_id'], 'top': r['top'],
                           'targets': len(r['signals']),
                           'ports': sum(s['kind'] == 'port' for s in r['signals']),
                           'internals': sum(s['kind'] == 'internal' for s in r['signals']),
                           'top_module_lines': r['source']['top_module_lines'],
                           'compile_status': r['source']['compile_status'],
                           'formal_context': r['formal_context']} for r in records],
                'files': files}
    write_json(DATA / 'manifest.json', manifest)
    return manifest


def package(manifest):
    previous = json.loads(PRIOR.read_text())
    protocol = {k: copy.deepcopy(previous[k]) for k in (
        'split', 'prompt_version', 'generator', 'training_config', 'training_config_sha256',
        'generation', 'system_prompt_sha256', 'expected_training', 'expected_packages', 'models')}
    protocol.update(protocol='frozen_external_mixed_six_v4', cohort=COHORT,
                    signal_count=manifest['signals_per_model'], signals_by_design=manifest['signals_by_design'],
                    test_designs=list(manifest['signals_by_design']), excluded_designs=manifest['previous_35_designs_excluded'],
                    prompts=REMOTE_DATA + '/prompts.jsonl', output_root='outputs/eval_one_shot_v4/external_mixed_test',
                    log_root='logs/eval_one_shot_v4/external_mixed_test', requests_per_signal=1, refinement=False,
                    reference_assertions_or_labels_in_inference_inputs=False, selection=POLICY)
    scripts = Path('experiments/evaluation')
    protocol['files'] = {}
    for name in ('generate.py', 'prompts.py', 'run_full.py'):
        key = 'scripts/eval_one_shot_v4/' + name
        if file_hash(scripts / name) != previous['files'][key]:
            raise ValueError('Established v4 generation file changed: ' + name)
        protocol['files'][key] = previous['files'][key]
    protocol['files']['scripts/eval_one_shot_v4/run_mixed.py'] = file_hash(scripts / 'run_mixed.py')
    protocol['files'].update({REMOTE_DATA + '/' + name: digest for name, digest in manifest['files'].items()})
    protocol['files'][REMOTE_DATA + '/manifest.json'] = file_hash(DATA / 'manifest.json')
    write_json(DATA / 'protocol.json', protocol)
    bundle = ROOT / 'remote_external_mixed_six_v4.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for path in sorted(DATA.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=REMOTE_DATA + '/' + str(path.relative_to(DATA)))
        for name in ('generate.py', 'prompts.py', 'run_full.py', 'run_mixed.py'):
            archive.add(scripts / name, arcname='scripts/eval_one_shot_v4/' + name)
        archive.add('experiments/evaluation/README.md', arcname='EXTERNAL_MIXED_SIX_V4_README.md')
    bundle.with_name(bundle.name + '.sha256').write_text(file_hash(bundle) + '  ' + bundle.name + '\n')
    report = {'status': 'PASS', 'cohort': COHORT, 'signals_per_model': manifest['signals_per_model'],
              'signals_by_design': manifest['signals_by_design'], 'cases': manifest['cases'],
              'total_generation_requests': manifest['total_generation_requests'],
              'all_six_rtl_configurations_compiled': True,
              'bundle': str(bundle), 'bundle_sha256': file_hash(bundle),
              'generation_code_and_prompt_match_previous_v4': True,
              'gpu_generation': 'Not started; run on black6000', 'formal_evaluation': 'Not run'}
    write_json(ROOT / 'preparation_verification.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sources-only', action='store_true')
    args = parser.parse_args()
    cases, excluded = load_cases()
    if args.sources_only:
        print(json.dumps([{'case_id': c['case_id'], 'candidates': len(candidates_from_xml(c['_xml']))} for c in cases], indent=2))
        return
    package(prepare_data(cases, excluded, args.workers))


if __name__ == '__main__':
    main()
