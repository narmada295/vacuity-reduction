"""Prepare original OpenTitan RTL using upstream generators and preprocessing."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REVISION = 'ed044fc9760bdf9fc075d0015ba1db07fa075355'
RELEASE = 'earlgrey_silver_release_v5'
ROOT = Path('output/opentitan_eval_v4')
UPSTREAM = ROOT / 'upstream'
CASES = (
    ('gpio', 'gpio'), ('rv_timer', 'rv_timer'), ('pwm', 'pwm_core'),
    ('uart', 'uart_core'), ('spi_host', 'spi_host_core'), ('i2c', 'i2c_core'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_only(source):
    return re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*',
                  lambda m: ' ' * len(m.group()), source, flags=re.S)


class Sources:
    def __init__(self, upstream=UPSTREAM, output=ROOT):
        self.upstream = upstream.resolve()
        self.output = output.resolve()
        actual = subprocess.check_output(['git', '-C', str(self.upstream), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != REVISION:
            raise ValueError('OpenTitan revision differs from the selected release')
        self.generated = self.output / 'generated_primitives'
        self.generated.mkdir(parents=True, exist_ok=True)
        self.original_files = sorted((self.upstream / 'hw/ip').glob('*/rtl/*.sv'))
        self.original_files += [self.upstream / 'hw/top_earlgrey/rtl/top_pkg.sv']
        self.headers = sorted({str(p.parent) for p in self.original_files})
        self.index = {}
        for path in self.original_files:
            self.add_index(path)
        self.gapi_cores = self.generic_cores()
        self.generate('prim_pkg')

    def add_index(self, path):
        for name in re.findall(r'\b(?:module|package)\s+(?:automatic\s+)?([A-Za-z_]\w*)',
                               code_only(path.read_text())):
            if name in self.index and self.index[name] != path:
                raise ValueError('Ambiguous RTL definition: ' + name)
            self.index[name] = path

    def generic_cores(self):
        import yaml
        result = {}
        for path in sorted((self.upstream / 'hw/ip/prim_generic').glob('*.core')):
            doc = yaml.safe_load(path.read_text())
            name = doc['name']
            if len(name.split(':')) == 3:
                name += ':0'
            files = doc.get('filesets', {}).get('files_rtl', {}).get('files', [])
            result[name] = {'core_root': str(path.parent),
                            'files': [f for f in files if isinstance(f, str)]}
        return result

    def generate(self, module):
        import yaml
        destination = self.generated / module
        destination.mkdir(exist_ok=True)
        arguments = {'cores': self.gapi_cores, 'parameters': (
            {'action': 'generate_prim_pkg'} if module == 'prim_pkg'
            else {'prim_name': module.removeprefix('prim_')})}
        recipe = destination / 'gapi.yaml'
        recipe.write_text(yaml.safe_dump(arguments))
        output = destination / (module + '.sv')
        if not output.exists():
            command = [sys.executable, str(self.upstream / 'hw/ip/prim/util/primgen.py'), str(recipe)]
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join(str(Path(p or '.').resolve())
                                               for p in env.get('PYTHONPATH', '').split(os.pathsep))
            run = subprocess.run(command, cwd=destination, text=True, capture_output=True, env=env)
            (destination / 'generator.log').write_text(run.stdout + run.stderr)
            if run.returncode or not output.exists():
                raise RuntimeError('Upstream primitive generation failed: ' + module + '\n' + run.stdout[-2000:] + run.stderr[-2000:])
        self.add_index(output)

    def references(self, path):
        text = code_only(path.read_text())
        names = set(re.findall(r'\b[A-Za-z_]\w*\b', text))
        for name in sorted(names):
            if (name.startswith('prim_') and name not in self.index
                    and 'prim_generic_' + name[5:] in self.index):
                self.generate(name)
        return {self.index[name] for name in names if name in self.index and self.index[name] != path}

    def ordered_sources(self, top):
        ordered, active, done = [], set(), set()

        def visit(path):
            if path in done:
                return
            if path in active:
                raise ValueError('Source dependency cycle at ' + str(path))
            active.add(path)
            for dependency in sorted(self.references(path)):
                visit(dependency)
            active.remove(path)
            done.add(path)
            ordered.append(path)

        visit(self.index[top])
        return ordered

    def prepare(self, ip, top):
        folder = self.output / 'cases' / ip
        folder.mkdir(parents=True, exist_ok=True)
        files = self.ordered_sources(top)
        includes = ['-I' + p for p in self.headers]
        source_args = [str(p) for p in files]
        command = ['verilator', '-E', '-P', '-DSYNTHESIS', *includes, *source_args]
        run = subprocess.run(command, text=True, capture_output=True)
        (folder / 'preprocess.log').write_text(run.stderr)
        if run.returncode:
            raise RuntimeError('OpenTitan preprocessing failed: ' + ip + '\n' + run.stderr[-3000:])
        text = run.stdout
        # The synthesis macro setting suppresses upstream verification macros.
        if re.search(r'\b(?:assert|assume|cover)\s*(?:property|\()', code_only(text)):
            raise ValueError('Upstream verification statement remains in inference RTL: ' + ip)
        (folder / 'dut.sv').write_text(text)
        top_source = self.index[top].read_text()
        (folder / 'top_original.sv').write_text(top_source)
        command = ['verilator', '--xml-only', '--xml-output', str(folder / 'design.xml'),
                   '--Mdir', str(folder / 'obj_dir'), '-Wno-fatal', '--top-module', top,
                   str(folder / 'dut.sv')]
        checked = subprocess.run(command, text=True, capture_output=True)
        (folder / 'compile.log').write_text(checked.stdout + checked.stderr)
        if checked.returncode:
            raise RuntimeError('OpenTitan compilation failed: ' + ip + '\n' + checked.stderr[-4000:])
        records = []
        for source in files:
            original = self.upstream in source.parents
            relative = str(source.relative_to(self.upstream if original else self.output))
            records.append({'path': relative, 'original_upstream_file': original,
                            'sha256': sha(source), 'lines': len(source.read_text().splitlines()),
                            'url': ('https://github.com/lowRISC/opentitan/blob/' + REVISION + '/' + relative) if original else None})
        record = {'ip': ip, 'top': top, 'release': RELEASE, 'revision': REVISION,
                  'top_original_sha256': sha(self.index[top]), 'rtl_sha256': sha(folder / 'dut.sv'),
                  'top_module_lines': len(top_source.splitlines()),
                  'all_dependency_lines': sum(r['lines'] for r in records),
                  'preprocessed_rtl_lines': len(text.splitlines()), 'rtl_characters': len(text),
                  'source_files': records, 'compile_status': 'PASS',
                  'compile_tool': subprocess.check_output(['verilator', '--version'], text=True).strip(),
                  'preprocessing_defines': ['SYNTHESIS'],
                  'functional_rtl_edits': False, 'upstream_assertions_in_inference': False,
                  'primitive_implementation': 'OpenTitan upstream primgen, generic implementation',
                  'scope': 'complete IP' if top == ip else 'functional core and its dependency closure'}
        (folder / 'source_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
        print(ip, 'PASS', record['top_module_lines'], 'top lines;', record['all_dependency_lines'], 'dependency lines', flush=True)
        return record


def main():
    sources = Sources()
    reports = [sources.prepare(ip, top) for ip, top in CASES]
    (ROOT / 'source_preparation.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
