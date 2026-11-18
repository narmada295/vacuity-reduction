#!/usr/bin/env python3
"""Independent end-to-end verification of a pruned SVA property set.

Python >= 3.6, standard library only. Uses the state produced by
formal_redundancy.py; emits a single FREE-TRACE JasperGold miter.

  python3 validate_reduced_set.py prepare --run redundancy_piezo_final --out final_validation
  (cd final_validation && jg -batch run.tcl >& jg.log)   # csh
  python3 validate_reduced_set.py summary --out final_validation

Proves that all retained safety assertions together imply each removed safety
assertion. The original RTL is never instantiated. Previously skipped liveness
assertions remain retained and are not part of the implication proof.
"""
import argparse
import json
import re
from pathlib import Path
try:
    from .formal_clocking import clock_commands
except ImportError:
    from formal_clocking import clock_commands


def read_json(path):
    with path.open('r', encoding='utf8') as f:
        return json.load(f)


def prepare(args):
    root = Path(args.run).resolve()
    state = read_json(root / 'state.json')
    original = set(str(x) for x in state['order'])
    retained = [str(x) for x in state['remaining']]
    removed = [str(x) for x in state['order'] if str(x) not in retained]
    props = state['properties']
    if set(retained) | set(removed) != original or set(retained) & set(removed):
        raise ValueError('Invalid retained/removed partition')
    if not removed:
        raise ValueError('No removed assertions to validate')
    removed_record = {str(p['target_id']): p for p in state['removed']}
    if set(removed_record) != set(removed):
        raise ValueError('Removed records do not match state remaining')
    for rid in removed:
        item = removed_record[rid]
        if (item.get('result') != 'PROVEN' or not item.get('removed') or
                item.get('statuses', {}).get('TARGET_CHECK') != 'proven'):
            raise ValueError('Unsafe removal record for target {}'.format(rid))
        if props[rid].get('excluded_liveness'):
            raise ValueError('Removed liveness property {}'.format(rid))
    safety_retained = [x for x in retained if not props[x]['excluded_liveness']]
    skip_liveness = [x for x in retained if props[x]['excluded_liveness']]
    decls = state['signal_types']
    clk = state['clock']
    if clk and clk not in decls:
        raise ValueError('Missing clock declaration')
    clocks = state.get('clocks', [clk] if clk else [])
    if any(name not in decls for name in clocks):
        raise ValueError('Missing declaration for one or more property clocks')
    for name, width in decls.items():
        if not re.match(r'^[A-Za-z_$][A-Za-z0-9_$]*$', name):
            raise ValueError('Unsupported identifier {!r}'.format(name))
        ranges = [width.get('packed', ''), width.get('unpacked', '')] if isinstance(width, dict) else [width]
        if any(value and not re.match(r'^\[\s*\d+\s*:\s*\d+\s*\]$', value) for value in ranges):
            raise ValueError('Unsupported width {!r}'.format(width))
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'validation_manifest.json').exists():
        raise ValueError('Output already initialized: {}'.format(output))

    ports = []
    for name, width in decls.items():
        if isinstance(width, dict):
            ports.append('  input logic {}{} {}'.format(
                (width['packed'] + ' ') if width['packed'] else '', name, width['unpacked']))
        else:
            ports.append('  input logic {}{}'.format((width + ' ') if width else '', name))
    lines = ['// Free-trace GLOBAL VALIDATION: no original RTL and no reset assumptions.',
             'module final_validation_miter (', ',\n'.join(ports), ');', '']
    for name, value in state.get('constants', {}).items():
        if not re.match(r'^[A-Za-z_$][A-Za-z0-9_$]*$', name) or not isinstance(value, int):
            raise ValueError('Unsupported parameter declaration')
        lines.append('  localparam {} = {};'.format(name, value))
    for pid in safety_retained:
        lines += ['  KEEP_{:04d}: assume property ('.format(int(pid)),
                  props[pid]['property_expression'], '  );', '']
    for pid in removed:
        lines += ['  RESTORE_{:04d}: assert property ('.format(int(pid)),
                  props[pid]['property_expression'], '  );', '']
    lines += ['endmodule', '']
    (output / 'final_validation_miter.sv').write_text('\n'.join(lines), encoding='utf8')

    tcl = '''# JasperGold 2022.12-compatible commands previously tested on Piezo_Driver.
analyze -sv12 final_validation_miter.sv
elaborate -top final_validation_miter
{clock_commands}
reset -none
prove -all
set fh [open "status.tsv" w]
foreach p [get_property_list] {{
  if {{[regexp {{\\.RESTORE_([0-9]+)$}} $p whole num]}} {{
    if {{[catch {{get_property_info -list status $p}} st]}} {{
      set st "QUERY_ERROR"
    }}
    puts $fh "RESTORE_$num\\t$st"
  }}
}}
close $fh
exit
'''.format(clock_commands=clock_commands(clocks,state.get('formal_clocking')))
    (output / 'run.tcl').write_text(tcl, encoding='utf8')
    manifest = {'source_run':str(root), 'retained_safety':safety_retained,
                'retained_liveness_not_used_as_premises':skip_liveness,
                'removed_checked':removed, 'design':state.get('design',''),
                'formal_clocking':state.get('formal_clocking')}
    (output / 'validation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    print('Prepared global validation: {} retained safety premises -> {} removed assertion targets'.format(
        len(safety_retained), len(removed)))
    print('Retained liveness assertions (untouched, not premises): {}'.format(skip_liveness))
    print('Generated: {}'.format(output))
    print('From csh: cd {}; jg -batch run.tcl >& jg.log; cd ..'.format(output))
    print('Then: python3 validate_reduced_set.py summary --out {}'.format(output))


def summary(args):
    root = Path(args.out).resolve()
    manifest = read_json(root / 'validation_manifest.json')
    names = set('RESTORE_{:04d}'.format(int(x)) for x in manifest['removed_checked'])
    path = root / 'status.tsv'
    if not path.exists():
        raise ValueError('Missing status.tsv; inspect jg.log')
    actual = {}
    for ln in path.read_text(encoding='utf8').splitlines():
        parts = ln.split('\t')
        if len(parts) != 2 or parts[0] not in names or parts[0] in actual:
            raise ValueError('Unexpected/duplicate status line {}'.format(repr(ln)))
        actual[parts[0]] = parts[1].strip().strip('{}').lower()
    missing = names - set(actual)
    if missing:
        print('INCOMPLETE: statuses missing for {} targets: {}'.format(len(missing), sorted(missing)))
        return
    counts = {}
    for value in actual.values():
        counts[value] = counts.get(value, 0) + 1
    print('Status counts: {}'.format(counts))
    if set(actual.values()) == {'proven'}:
        print('PASS: Retained safety assertions imply ALL {} removed assertions (free-trace check).'.format(len(names)))
        print('Original set is equivalent to reduced set for checked safety assertions.')
        print('Liveness property was kept unchanged and not used as a premise.')
    else:
        not_proven = sorted(x for x in actual if actual[x] != 'proven')
        print('NOT VALIDATED: {} targets not proven: {}'.format(len(not_proven),
              [(x, actual[x]) for x in not_proven]))
        print('Do not treat reduction as globally verified.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd')
    a = sub.add_parser('prepare')
    a.add_argument('--run', required=True)
    a.add_argument('--out', required=True)
    b = sub.add_parser('summary')
    b.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        if args.cmd == 'prepare':
            prepare(args)
        elif args.cmd == 'summary':
            summary(args)
        else:
            p.error('Choose prepare or summary')
    except (IOError, ValueError, KeyError) as exc:
        p.exit(2, 'ERROR: {}\n'.format(exc))


if __name__ == '__main__':
    main()
