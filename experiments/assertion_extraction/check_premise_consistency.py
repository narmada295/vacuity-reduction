#!/usr/bin/env python3
"""Generate a separate RTL-witness check for consistency of an SVA property set.

This does NOT prove property-to-property redundancy. It checks that the full
safety-property set holds simultaneously on the same executable RTL model,
providing a consistency witness for the independent free-trace implication checks.
Compatible with Python 3.6.

Usage:
  python3 check_premise_consistency.py prepare --input unique_assertions.jsonl --case case.json --out consistency_piezo
  (cd consistency_piezo && jg -batch run.tcl)
  python3 check_premise_consistency.py summary --out consistency_piezo
"""
import argparse
import json
import re
from pathlib import Path
try:
    from .formal_clocking import clock_commands
except ImportError:
    from formal_clocking import clock_commands

LIVENESS = re.compile(r'\b(s_eventually|eventually|s_always|until|s_until|until_with|s_until_with|strong|weak)\b', re.I)


def prepare(args):
    rows = [json.loads(s) for s in Path(args.input).read_text(encoding='utf8').splitlines() if s.strip()]
    if not rows:
        raise ValueError('No assertions in input')
    clocks = set(r.get('clock') for r in rows)
    contexts = set((r.get('case_id'),r.get('rtl_sha256')) for r in rows)
    if len(clocks) != 1 or len(contexts) != 1:
        raise ValueError('Input contains mixed clocks or designs')
    clock = next(iter(clocks))
    if clock and not re.match(r'^[A-Za-z_$][\w$]*$', clock):
        raise ValueError('Unsupported clock name')
    if not clock and any('@' in r['property_expression'] for r in rows):
        raise ValueError('Clockless case contains an explicit clock event')
    property_clocks = list(dict.fromkeys(
        name for row in rows for name in re.findall(
            r'@\s*\(\s*(?:posedge|negedge)\s+([A-Za-z_$][\w$]*)\s*\)',
            row['property_expression'])))
    needs_default = any(not re.search(
        r'@\s*\(\s*(?:posedge|negedge)\s+[A-Za-z_$][\w$]*\s*\)',
        row['property_expression']) for row in rows)
    if clock and needs_default and clock not in property_clocks:
        property_clocks.append(clock)
    doc = json.loads(Path(args.case).read_text(encoding='utf8'))
    case_rows = doc.get('rows') if isinstance(doc, dict) else doc
    if not isinstance(case_rows, list):
        raise ValueError('Expected case JSON with rows array')
    unique_rtls = set(r.get('rtl') for r in case_rows if r.get('rtl'))
    if len(unique_rtls) != 1:
        raise ValueError('Expected one unique RTL from case JSON')
    rtl = unique_rtls.pop()
    modules = re.findall(r'\bmodule\s+([A-Za-z_$][\w$]*)\b', rtl)
    if len(modules) != 1 or len(re.findall(r'\bendmodule\b', rtl)) != 1:
        raise ValueError('This checker supports a single-module RTL case')
    if any(not r.get('property_expression') for r in rows):
        raise ValueError('Missing property expressions')
    safety = [r for r in rows if not LIVENESS.search(r['property_expression'])]
    ignored = [r for r in rows if LIVENESS.search(r['property_expression'])]
    insertion = []
    labels = []
    for row in safety:
        label = 'RTL_CHECK_{:04d}'.format(int(row['row_number']))
        labels.append(label)
        insertion.append('  {}: assert property (\n{}\n  );'.format(label,row['property_expression']))
    trace_clock = property_clocks[0] if property_clocks else clock
    if trace_clock:
        insertion.append("  RTL_TRACE_EXISTS: cover property (@(posedge {}) 1'b1);".format(trace_clock))
    else:
        insertion.append("  RTL_TRACE_EXISTS: cover property (1'b1);")
    rtl = re.sub(r'\bendmodule\b', '\n'.join(insertion) + '\nendmodule', rtl, count=1)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'rtl_witness.sv').write_text(rtl, encoding='utf8')
    # Property list and status query tested in JasperGold 2022.12 according to user's audit.
    tcl = '''analyze -sv12 rtl_witness.sv
elaborate -top TOP_REPLACE
CLOCK_REPLACE
reset -none
prove -all
set plist [get_property_list]
set fh [open "status.tsv" w]
foreach label {LABELS_REPLACE RTL_TRACE_EXISTS} {
  set matches {}
  foreach p $plist {
    if {$p eq $label || [string match "*.$label" $p] || [string match "*/$label" $p]} {
      lappend matches $p
    }
  }
  if {[llength $matches] != 1} {
    set status "QUERY_ERROR"
  } elseif {[catch {get_property_info -list status [lindex $matches 0]} status]} {
    set status "QUERY_ERROR"
  }
  puts $fh "$label\\t$status"
}
close $fh
exit
'''
    context = doc.get('formal_clocking') if isinstance(doc, dict) else None
    commands = clock_commands(property_clocks, context)
    tcl = tcl.replace('TOP_REPLACE',modules[0]).replace('CLOCK_REPLACE',commands).replace('LABELS_REPLACE',' '.join(labels))
    (out/'run.tcl').write_text(tcl,encoding='utf8')
    (out/'metadata.json').write_text(json.dumps({'count':len(labels),'excluded_liveness':[r['row_number'] for r in ignored], 'module':modules[0], 'formal_clocking':context},indent=2),encoding='utf8')
    print('Prepared {} RTL checks (excluded {} liveness).'.format(len(labels),len(ignored)))
    print('Run: (cd "{}" && jg -batch run.tcl > jg.log 2>&1)'.format(str(out)))
    print('Then: python3 check_premise_consistency.py summary --out "{}"'.format(str(out)))


def summary(args):
    out=Path(args.out)
    status_file=out/'status.tsv'
    if not status_file.exists():
        raise ValueError('Missing {}, run JasperGold first'.format(status_file))
    meta=json.loads((out/'metadata.json').read_text(encoding='utf8'))
    statuses={}
    for line in status_file.read_text(encoding='utf8').splitlines():
        p=line.split('\t')
        if len(p)!=2 or p[0] in statuses:
            raise ValueError('Invalid status line {!r}'.format(line))
        statuses[p[0]]=p[1].strip().lower()
    vals=[v for k,v in statuses.items() if k.startswith('RTL_CHECK_')]
    covers=statuses.get('RTL_TRACE_EXISTS','missing')
    if len(vals)!=meta['count']:
        raise ValueError('Wrong count: expected {}, got {}'.format(meta['count'],len(vals)))
    from collections import Counter
    print('Assertion statuses:',dict(Counter(vals)))
    print('RTL trace cover:',covers)
    if all(v=='proven' for v in vals) and covers=='covered':
        print('PASS: Same RTL admits a trace and proves all checked safety assertions.')
        print('The full premise set is consistent under this RTL/formal interpretation.')
    else:
        print('NOT VALIDATED: Do not auto-prune. Investigate unproven properties or missing cover.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd')
    pp=sub.add_parser('prepare');pp.add_argument('--input',required=True);pp.add_argument('--case',required=True);pp.add_argument('--out',required=True)
    ss=sub.add_parser('summary');ss.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.cmd=='prepare':prepare(a)
    elif a.cmd=='summary':summary(a)
    else:ap.error('Choose prepare or summary')

if __name__=='__main__':main()
