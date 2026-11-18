#!/usr/bin/env python3
"""Sequential all-others SVA redundancy checker (conservative JasperGold driver).

Read JSONL from remove_exact_duplicates.py; generate free-trace SVA miters,
run JasperGold one target at a time, and only delete on an explicit complete
proof plus cover witnesses. RTL used solely to obtain signal declarations.

Commands:
  init   --input INPUT.jsonl --case CASE.json --out RUN_DIR
  next   --out RUN_DIR
  run    --out RUN_DIR --max-jobs 1 --auto-prune
  status --out RUN_DIR

The original assertions/RTL are NEVER edited.
"""
from typing import Dict, List, Optional

import argparse
import ast
import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
try:
    from .formal_clocking import clock_commands, context_from_case
except ImportError:
    from formal_clocking import clock_commands, context_from_case

UNSUPPORTED_LIVENESS = re.compile(r'\b(s_eventually|eventually|s_always|until|s_until|until_with|s_until_with|strong|weak)\b', re.I)
CLOCK_RE = re.compile(r'@\s*\(\s*(?:posedge|negedge)\s+([A-Za-z_$][\w$]*)\s*\)')
IMP_RE = re.compile(r'\|=>|\|->')
DECL_RE = re.compile(r'\b(?:input|output|inout|reg|wire|logic|bit)\b')
PRIMITIVE_TYPES = ('logic', 'wire', 'reg', 'bit', 'signed', 'unsigned', 'var')
COV_STATUS = {'covered', 'reachable'}
PROOF_STATUS = {'proven'}
SV_LITERAL_RE = re.compile(r"(?:(?:\d+)?'\s*[sS]?[bBoOdDhH]\s*[0-9a-fA-F_xXzZ?]+|'\s*[01xXzZ])")


def store_json(p: Path, data: dict) -> None:
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    tmp.replace(p)


def load_state(root: Path) -> dict:
    p = root / 'state.json'
    if not p.exists():
        raise SystemExit(f'No {p}; run init first')
    return json.loads(p.read_text(encoding='utf8'))


def write_jsonl(p: Path, rows: List[dict]) -> None:
    with p.open('w', encoding='utf8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')


def safe_ident(s: str) -> bool:
    return re.fullmatch(r'[a-zA-Z_$][a-zA-Z0-9_$]*', s) is not None


def evaluate_numeric(expression: str, values: Dict[str, int]) -> int:
    """Evaluate only arithmetic used by numeric Verilog parameters and ranges."""
    def literal(match):
        token = match.group(0)
        base_char = re.search(r"'\s*[sS]?([bBoOdDhH])", token).group(1).lower()
        digits = re.sub(r'\s|_', '', token.rsplit(base_char, 1)[1])
        if re.search(r'[xXzZ?]', digits):
            raise ValueError('Unknown-valued parameter')
        return str(int(digits, {'b': 2, 'o': 8, 'd': 10, 'h': 16}[base_char]))

    normalized = re.sub(r"(?:\d+)?'\s*[sS]?[bBoOdDhH]\s*[0-9a-fA-F_xXzZ?]+",
                        literal, expression)
    node = ast.parse(normalized, mode='eval').body

    def visit(part):
        if isinstance(part, ast.Num):
            return int(part.n)
        if isinstance(part, ast.Name) and part.id in values:
            return values[part.id]
        if isinstance(part, ast.UnaryOp) and isinstance(part.op, ast.USub):
            return -visit(part.operand)
        if isinstance(part, ast.BinOp):
            left, right = visit(part.left), visit(part.right)
            if isinstance(part.op, ast.Add):
                return left + right
            if isinstance(part.op, ast.Sub):
                return left - right
            if isinstance(part.op, ast.Mult):
                return left * right
            if isinstance(part.op, ast.LShift) and right >= 0:
                return left << right
            if isinstance(part.op, ast.RShift) and right >= 0:
                return left >> right
            if isinstance(part.op, (ast.Div, ast.FloorDiv)) and right:
                return left // right
        raise ValueError('Unsupported parameter expression')

    return visit(node)


def parameter_values(rtl: str) -> Dict[str, int]:
    """Resolve numeric Verilog parameters used by fixed-width declarations."""
    rtl = re.sub(r'/\*.*?\*/|//[^\n]*', ' ', rtl, flags=re.S)
    raw = dict(re.findall(
        r'\b(?:localparam|parameter)\s+(?:(?:integer|int|signed|unsigned)\s+)?'
        r'(?:\[[^\]]+\]\s*)?([A-Za-z_$][\w$]*)\s*=\s*([^,;]+)', rtl))
    for block in re.findall(r'\b(?:localparam|parameter)\s+([^;]+);', rtl):
        for part in block.split(','):
            match = re.fullmatch(
                r'\s*(?:(?:integer|int|signed|unsigned)\s+)?'
                r'(?:\[[^\]]+\]\s*)?([A-Za-z_$][\w$]*)\s*=\s*(.+?)\s*', part)
            if match:
                raw[match.group(1)] = match.group(2)
    values = {}

    pending = dict(raw)
    for _ in range(len(raw)):
        changed = False
        for name, expression in list(pending.items()):
            try:
                values[name] = evaluate_numeric(expression, values)
            except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
                continue
            del pending[name]
            changed = True
        if not changed:
            break
    return values


def width_from_range(s: str, parameters: Optional[Dict[str, int]] = None) -> str:
    s = s.strip()
    if not s:
        return ''
    if re.fullmatch(r'\[\s*\d+\s*:\s*\d+\s*\]', s):
        return s
    pieces = s[1:-1].split(':') if s.startswith('[') and s.endswith(']') else []
    if len(pieces) == 2:
        try:
            high, low = [evaluate_numeric(part, parameters or {}) for part in pieces]
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
            pass
        else:
            if high >= 0 and low >= 0:
                return '[{}:{}]'.format(high, low)
    raise ValueError(f'parameterized/non-numeric packed range {s!r}: provide --decls with resolved widths')


def declare_types_from_rtl(rtl: str) -> Dict[str, str]:
    """Conservative Verilog declaration reader for simple modules.

    Unsupported declarations MUST be supplied using --decls; never silently
    treat unknown signal sizes as one bit.
    """
    rtl = re.sub(r'/\*.*?\*/|//[^\n]*', ' ', rtl, flags=re.S)
    parameters = parameter_values(rtl)
    declarations = {}
    # Separate (possibly ANSI) module header and body.
    m = re.search(r'\bmodule\s+[A-Za-z_$][\w$]*\s*(?:#\s*\(.*?\)\s*)?(?:\((.*?)\))?\s*;', rtl, flags=re.S)
    if not m:
        raise ValueError('Could not parse RTL module declaration; use --decls FILE')
    header, body = m.group(1) or '', rtl[m.end():]
    current_width = ''
    for fragment in header.split(','):
        part = fragment.split('=')[0].strip()
        if not part:
            continue
        mt = re.match(r'(?:(input|output|inout)\s+)?(?:(?:wire|reg|logic|bit|signed|unsigned|var)\s+)*(\[[^\]]+\])?\s*([A-Za-z_$][\w$]*)\s*$', part)
        if not mt:
            # A complex declaration will be handled by the missing signal check.
            continue
        if mt.group(1):
            current_width = width_from_range(mt.group(2) or '', parameters)
        elif mt.group(2):
            current_width = width_from_range(mt.group(2), parameters)
        declarations[mt.group(3)] = current_width
    for mt in re.finditer(r'\b(?:input|output|inout|reg|wire|logic|bit)\b\s*(?:(?:signed|unsigned|reg|logic|wire|bit)\b\s*)*(\[[^\]]+\])?\s*([^;]+);', body, flags=re.S):
        try:
            width = width_from_range(mt.group(1) or '', parameters)
        except ValueError:
            continue
        for part in mt.group(2).split(','):
            name = part.split('=')[0].strip()
            if safe_ident(name):
                declarations[name] = width
            else:
                array = re.fullmatch(r'([A-Za-z_$][\w$]*)\s*(\[[^\]]+\])', name)
                if array:
                    try:
                        unpacked = width_from_range(array.group(2), parameters)
                    except ValueError:
                        continue
                    declarations[array.group(1)] = {'packed': width, 'unpacked': unpacked}
    return declarations


def read_decls_file(p: Path) -> Dict[str, str]:
    """Manual file syntax: one 'input logic [N:M] signal;' per line."""
    out = {}
    for n, line in enumerate(p.read_text(encoding='utf8').splitlines(), 1):
        line = line.split('//')[0].strip()
        if not line:
            continue
        m = re.fullmatch(r'(?:input\s+)?(?:logic|wire|reg|bit)\s+(\[[^\]]+\])?\s*([\w$]+)\s*;', line)
        if not m or not safe_ident(m.group(2)):
            raise ValueError(f'Invalid declaration at {p}:{n}: {line}')
        out[m.group(2)] = width_from_range(m.group(1) or '')
    return out


def all_decls(rows: List[dict], rtl_text: str, declfile: Optional[Path]) -> Dict[str, str]:
    parsed = declare_types_from_rtl(rtl_text) if rtl_text else {}
    constants = parameter_values(rtl_text) if rtl_text else {}
    if declfile:
        parsed.update(read_decls_file(declfile))
    clock = rows[0].get('clock')
    if clock and clock not in parsed:
        raise ValueError(f'Clock {clock} missing declarations. Supply --decls.')
    for r in rows:
        expr = r['property_expression']
        if '\nendproperty' in expr or not expr.strip():
            raise ValueError('Unexpected full property declaration or empty expression')
        clocks = CLOCK_RE.findall(expr)
        if any(name not in parsed for name in clocks):
            raise ValueError(f'Row {r["row_number"]}: undeclared clock expression {clocks}; provide --decls')
        if not clock and '@' in expr:
            raise ValueError(f'Row {r["row_number"]}: explicit event in clockless design')
        # Determine references to design signals by scanning all identifiers.
        without_literals = SV_LITERAL_RE.sub(' ', expr)
        tokens = set(re.findall(r'(?<!\$)\b[A-Za-z_][\w$]*\b', without_literals))
        # SystemVerilog built-in names and fragments do not count as net signals.
        keywords = {'posedge','negedge','disable','iff','past','stable','changed','rose','fell',
                    'bits','isunknown','countones','onehot','onehot0','isunknown','s_eventually',
                    'strong','weak','always','eventually','inside','dist','throughout','intersect',
                    'first_match','not','or','and','until','s_until','true','false','local',
                    'null','signed','unsigned','int','logic','bit','begin','end','if','else'}
        # Look only at plausible *actual* SV names that appear in the RTL.
        # More importantly, missing identifiers that look like signals are rejected.
        unknown = {tok for tok in tokens if tok not in parsed and tok not in constants and tok.lower() not in keywords}
        if unknown:
            raise ValueError(f'Row {r["row_number"]}: undeclared tokens {sorted(unknown)}; provide --decls')
    return parsed


def load_case_rtl(path: Path) -> str:
    doc = json.loads(path.read_text(encoding='utf8'))
    rows = doc['rows'] if isinstance(doc, dict) and isinstance(doc.get('rows'), list) else doc
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a JSON object with rows array containing rtl')
    rtls = {r['rtl'] for r in rows if r.get('rtl')}
    if len(rtls) != 1:
        raise ValueError('Expected exactly one unique RTL in case JSON')
    return next(iter(rtls))


def cmd_init(args):
    inp = Path(args.input).resolve()
    rows = [json.loads(line) for line in inp.read_text(encoding='utf8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('No assertions found')
    contexts = {(r.get('case_id'),r.get('design'),r.get('rtl_sha256'),r.get('formal_reset_policy'),r.get('clock')) for r in rows}
    if len(contexts) != 1:
        raise ValueError('Mixed designs/formal contexts found. Run one design at a time.')
    if any(r.get('extraction_status') != 'ok' or not r.get('property_expression','').strip() for r in rows):
        raise ValueError('Unresolved assertions in input. Inspect extraction_issues.csv first.')
    if any(r.get('selected') is not True for r in rows):
        raise ValueError('Input contains unselected assertions; use deduplicated/unique_assertions.jsonl')
    ids = [str(r['row_number']) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate row_number IDs. Process separate case.json files separately.')
    rtl = load_case_rtl(Path(args.case)) if args.case else Path(args.rtl).read_text() if args.rtl else ''
    context = context_from_case(args.case) if args.case else None
    decls = all_decls(rows, rtl, Path(args.decls) if args.decls else None)
    constants = parameter_values(rtl) if rtl else {}
    explicit_clocks = list(dict.fromkeys(
        name for row in rows for name in CLOCK_RE.findall(row['property_expression'])))
    default_clock = rows[0].get('clock')
    needs_default_clock = any(not CLOCK_RE.findall(row['property_expression']) for row in rows)
    clocks = explicit_clocks + ([default_clock] if default_clock and needs_default_clock
                                 and default_clock not in explicit_clocks else [])
    by_id = {}
    for r in rows:
        rid = str(r['row_number'])
        by_id[rid] = {
            'id':rid, 'property_name':r.get('property_name',''),
            'property_expression':r['property_expression'],
            'assertion':r.get('assertion',''),
            'source_file':r.get('source_file',''),
            'case_id':r.get('case_id',''),
            'jasper_log_original':r.get('jasper_log',''),
            'excluded_liveness': bool(UNSUPPORTED_LIVENESS.search(r['property_expression']))
        }
    root=Path(args.out).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if (root/'state.json').exists():
        raise ValueError(f'{root}/state.json exists. Use existing run dir or choose another.')
    state = {
        'version':1, 'design':rows[0].get('design',''), 'clock':default_clock,
        'clocks':clocks, 'constants':constants,
        'formal_clocking':context,
        'rtl_sha256':rows[0].get('rtl_sha256',''),
        'source_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
        'source_path':str(inp),'properties':by_id,'order':ids,'remaining':ids.copy(),
        'processed':[], 'removed':[], 'signal_types':decls,
        'notes':'All outputs are candidate checks, not JasperGold proofs. RTL is NOT in free-trace miter.'
    }
    # Preserve original input for final output, independent of source location.
    write_jsonl(root/'original_unique.jsonl',rows)
    store_json(root/'state.json',state)
    export_reports(root,state)
    skip=[x for x in ids if by_id[x]['excluded_liveness']]
    print(f'Initialized {root}\n{len(ids)} properties, {len(decls)} symbolic signals. Liveness retained/not assumed: {skip}')
    prepare_next(root,state)


def find_antecedent(expr: str, clockless: bool = False) -> Optional[str]:
    """Very conservative: only generate cover if a simple |=> or |-> property.

    Supports top-level @(posedge clk), clockless expressions, and optional disable iff(...).
    Does not alter the assert/assume property expression.
    """
    if clockless:
        after=expr.strip()
    else:
        event=CLOCK_RE.search(expr)
        if not event: return None
        after=expr[event.end():].strip()
    disable_condition=None
    if after.startswith('disable iff'):
        mt=re.match(r'disable\s+iff\s*\(',after)
        if not mt: return None
        i=mt.end(); depth=1
        while i < len(after) and depth:
            if after[i]=='(':depth+=1
            elif after[i]==')':depth-=1
            i+=1
        if depth:return None
        disable_condition=after[mt.end():i-1].strip()
        after=after[i:].strip()
    sep=IMP_RE.search(after)
    if not sep:return None
    ant=after[:sep.start()].strip()
    # A property may wrap the whole implication in parentheses. The split
    # leaves those opening parentheses on the antecedent side; remove only
    # unmatched wrappers so the trigger cover remains valid SystemVerilog.
    while ant.startswith('(') and ant.count('(')>ant.count(')'):
        ant=ant[1:].strip()
    if ant.count('(')!=ant.count(')'):
        return None
    if not ant or '##' in ant or '@' in ant or 'throughout' in ant:return None
    if disable_condition:
        return f'(!({disable_condition}) && ({ant}))'
    return ant


def job_fingerprint(target: str, premises: List[str], state: dict) -> str:
    fields = {'target':target, 'premises':premises,
                          'expressions':[state['properties'][x]['property_expression'] for x in premises+[target]],
                          'types':state['signal_types'], 'constants':state.get('constants', {}),
                          'clocks':state.get('clocks', [])}
    if state.get('formal_clocking') is not None:
        fields['formal_clocking'] = state['formal_clocking']
    payload = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def input_port(name: str, datatype) -> str:
    if isinstance(datatype, dict):
        return '    input logic {}{} {}'.format(
            (datatype['packed'] + ' ') if datatype['packed'] else '',
            name, datatype['unpacked'])
    return '    input logic {}{}'.format((datatype + ' ') if datatype else '', name)


def prepare_next(root: Path, state: dict):
    processed_ids={p['target_id'] for p in state['processed']}
    remaining=state['remaining']
    target=next((x for x in state['order'] if x in remaining and x not in processed_ids
                 and not state['properties'][x]['excluded_liveness']), None)
    if target is None:
        print('No more eligible unchecked assertions. Run status.')
        return None
    # Excluding all liveness assumptions conservatively avoids vacuous proofs
    # from potentially inconsistent fairness/liveness constraints.
    premises=[x for x in remaining if x!=target and not state['properties'][x]['excluded_liveness']]
    fp=job_fingerprint(target,premises,state)
    job=root/'jobs'/f'target_{int(target):04d}_{fp}'
    job.mkdir(parents=True,exist_ok=True)
    clk=state['clock']
    clocks=state.get('clocks', [clk] if clk else [])
    decls=[]
    for name,width in state['signal_types'].items():
        decls.append(input_port(name, width))
    lines=['// Automatically generated free-trace miter. NEVER instantiate the RTL here.',
           '// Assumptions = other retained SVA properties; TARGET_CHECK = candidate.',
           'module redundancy_free_trace_miter (',',\n'.join(decls),');','']
    for name, value in state.get('constants', {}).items():
        lines.append('  localparam {} = {};'.format(name, value))
    for x in premises:
        expr=state['properties'][x]['property_expression']
        lines.append(f'  PREMISE_{int(x):04d}: assume property (\n{expr}\n  );')
    expr=state['properties'][target]['property_expression']
    lines.append(f'  TARGET_CHECK: assert property (\n{expr}\n  );')
    target_clocks=CLOCK_RE.findall(expr)
    trigger_clock=target_clocks[0] if target_clocks else clk
    cover_event=f'@(posedge {trigger_clock}) ' if trigger_clock else ''
    lines.append(f"  ASSUMPTION_WITNESS: cover property ({cover_event}1'b1);")
    ant=find_antecedent(expr, clockless=not bool(target_clocks or clk))
    if ant is None:
        lines.append(f"  TARGET_TRIGGER: cover property ({cover_event}1'b1);")
    else:
        lines.append(f'  TARGET_TRIGGER: cover property ({cover_event}({ant}));')
    lines.append('endmodule\n')
    (job/'free_trace_miter.sv').write_text('\n'.join(lines), encoding='utf8')
    commands=clock_commands(clocks,state.get('formal_clocking'))
    (job/'run.tcl').write_text('''# Run with: jg -batch run.tcl (inside this job directory)
analyze -sv12 free_trace_miter.sv
elaborate -top redundancy_free_trace_miter
'''+commands+'''
reset -none
# Do not add RTL or environment reset constraints: implication over free traces.
prove -all
# Query machine-readable property states. If your JG version differs,
# script fails closed: nothing is automatically removed.
set fh [open "status.tsv" w]
set plist [get_property_list]
foreach label {TARGET_CHECK ASSUMPTION_WITNESS TARGET_TRIGGER} {
  set matches {}
  foreach p $plist {
    if {$p eq $label || [string match "*.$label" $p] || [string match "*/$label" $p]} {
      lappend matches $p
    }
  }
  if {[llength $matches] != 1} {
    puts $fh "$label\\tQUERY_ERROR"
  } else {
    set st "QUERY_ERROR"
    catch {set st [get_property_info -list status [lindex $matches 0]]}
    puts $fh "$label\\t$st"
  }
}
close $fh
catch {report -property -assert -file assertion_report.txt}
catch {report -property -cover -file cover_report.txt}
exit
''', encoding='utf8')
    store_json(job/'job.json',{'target_id':target,'premises':premises, 'fingerprint':fp,
                               'target':state['properties'][target], 'trigger_covered_expression':ant,
                               'clocks':clocks,'formal_clocking':state.get('formal_clocking'),
                               'no_original_rtl':True})
    print(f'Next: target {target} ({state["properties"][target]["property_name"] or "inline"}); '
          f'{len(premises)} premises; job={job}')
    return job


def normalize_status(value: str) -> str:
    return value.strip().strip('{}').lower()


def parse_status(path: Path):
    if not path.exists():
        return None, 'Missing status.tsv (or unsupported Jasper Tcl commands)'
    found={}
    for line in path.read_text(encoding='utf8').splitlines():
        parts=line.split('\t')
        if len(parts)!=2 or parts[0] not in {'TARGET_CHECK','ASSUMPTION_WITNESS','TARGET_TRIGGER'} or parts[0] in found:
            return None, f'Invalid status.tsv line: {line}'
        found[parts[0]]=normalize_status(parts[1])
    if set(found)!={'TARGET_CHECK','ASSUMPTION_WITNESS','TARGET_TRIGGER'}:
        return None,'status.tsv missing property statuses'
    return found, None


def cmd_run(args):
    root=Path(args.out).resolve()
    state=load_state(root)
    for _ in range(args.max_jobs):
        job=prepare_next(root,state)
        if job is None:break
        # Ensures no stale status from an earlier run is accidentally trusted.
        if (job/'status.tsv').exists():
            (job/'status.tsv').unlink()
        command=shlex.split(args.jg)+['-batch','run.tcl']
        print('Running', ' '.join(command), 'in', job, flush=True)
        started=time.monotonic()
        try:
            with (job/'jg.log').open('w', encoding='utf8') as log:
                result=subprocess.run(command,cwd=job,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            rc=result.returncode
        except (subprocess.TimeoutExpired,FileNotFoundError) as exc:
            rc=-1
            (job/'runner_error.txt').write_text(str(exc)+'\n')
        elapsed=round(time.monotonic()-started,2)
        statuses,err=parse_status(job/'status.tsv') if rc==0 else (None, f'Jasper exit code {rc}')
        # Never turn an unsupported Tcl status query into a normal 'retained' check.
        # Such a failure must be fixed and the exact SAME job rerun.
        valid_proof = {'proven', 'cex', 'falsified', 'unknown',
                       'undetermined', 'unprocessed', 'unreachable', 'timeout'}
        valid_cover = COV_STATUS | {'unreachable', 'unknown', 'undetermined',
                                    'unprocessed', 'timeout'}
        if statuses and (statuses['TARGET_CHECK'] not in valid_proof or
                         statuses['ASSUMPTION_WITNESS'] not in valid_cover or
                         statuses['TARGET_TRIGGER'] not in valid_cover):
            err = f'Unsupported property status values {statuses} (check Tcl queries against JG version)'
            statuses = None
        if not statuses:
            print(f'ERROR: {err}. No progress recorded; fix Tcl/Jasper and rerun this job.')
            break
        complete=(statuses['TARGET_CHECK'] in PROOF_STATUS)
        covered=(statuses['ASSUMPTION_WITNESS'] in COV_STATUS and
                 statuses['TARGET_TRIGGER'] in COV_STATUS)
        # A completed implication proof is the ONLY basis for removal.
        remove=bool(args.auto_prune and complete and covered)
        if complete:
            summary='PROVEN' if covered else 'PROVEN_BUT_COVER_UNVERIFIED'
        elif statuses['TARGET_CHECK'] in {'cex','falsified'}:
            summary='COUNTEREXAMPLE'
        else:
            summary='UNKNOWN'
        tid=json.loads((job/'job.json').read_text())['target_id']
        premises=json.loads((job/'job.json').read_text())['premises']
        item={'target_id':tid,'premises':premises,'result':summary,'statuses':statuses or {},
              'error':err or '','removed':remove,'log':str(job/'jg.log'),
              'fingerprint':json.loads((job/'job.json').read_text())['fingerprint'],
              'elapsed_seconds':elapsed}
        state['processed'].append(item)
        if remove:
            state['remaining'].remove(tid)
            state['removed'].append(item)
        store_json(root/'state.json',state)
        export_reports(root,state)
        print(f'Result for {tid}: {summary}; removed={remove}; detail={statuses or err}')


def export_reports(root:Path, state:dict):
    rows=[json.loads(l) for l in (root/'original_unique.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
    kept_ids=set(state['remaining'])
    write_jsonl(root/'retained_assertions.jsonl',[r for r in rows if str(r['row_number']) in kept_ids])
    outputs=[]
    result_by_id={r['target_id']:r for r in state['processed']}
    for prop in state['order']:
        p=state['properties'][prop]
        res=result_by_id.get(prop,{})
        outputs.append({'id':prop,'name':p['property_name'] or '(inline)',
                        'decision':'REMOVED' if prop not in kept_ids else 'RETAINED',
                        'check_result':res.get('result','SKIPPED_LIVENESS' if p['excluded_liveness'] else 'PENDING'),
                        'premise_ids':','.join(res.get('premises',[])) if res.get('removed') else '',
                        'target_proof':res.get('statuses',{}).get('TARGET_CHECK',''),
                        'witness_cover':res.get('statuses',{}).get('ASSUMPTION_WITNESS',''),
                        'trigger_cover':res.get('statuses',{}).get('TARGET_TRIGGER',''),
                        'elapsed_seconds':res.get('elapsed_seconds',''),
                        'evidence_log':res.get('log','')})
    with (root/'results.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(outputs[0]));writer.writeheader();writer.writerows(outputs)


def cmd_status(args):
    root=Path(args.out).resolve(); s=load_state(root)
    export_reports(root,s)
    total=len(s['order']);done=len(s['processed'])
    skip=sum(1 for x in s['remaining'] if s['properties'][x]['excluded_liveness'])
    print(f"{s['design']} | initial unique={total}, checked={done}, removed={len(s['removed'])}, "
          f"retained={len(s['remaining'])} (incl. {skip} skipped liveness)")
    print('Details:',root/'results.csv','\nRetained JSONL:',root/'retained_assertions.jsonl')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs=p.add_subparsers(dest='cmd')
    p0=subs.add_parser('init',help='Initialize a run and write the first miter')
    p0.add_argument('--input',required=True)
    source=p0.add_mutually_exclusive_group(required=False)
    source.add_argument('--case',help='Original case.json containing RTL in rows[*].rtl')
    source.add_argument('--rtl',help='RTL .v file, ONLY for net declarations')
    p0.add_argument('--decls', help='Optional declaration override file for more complex designs')
    p0.add_argument('--out',required=True)
    p0.set_defaults(func=cmd_init)
    pn=subs.add_parser('next',help='Generate next unprocessed job; does not run JG')
    pn.add_argument('--out',required=True)
    pn.set_defaults(func=lambda args:prepare_next(Path(args.out).resolve(),load_state(Path(args.out).resolve())))
    pr=subs.add_parser('run',help='Run sequential Jasper jobs; auto-prune only with validated proof+cover statuses')
    pr.add_argument('--out',required=True)
    pr.add_argument('--jg',default='jg',help='JasperGold executable (default jg)')
    pr.add_argument('--timeout',type=int,default=300,help='Wall-clock timeout per job in seconds')
    pr.add_argument('--max-jobs',type=int,default=1,help='Jobs to process sequentially this invocation')
    pr.add_argument('--auto-prune', action='store_true',help='Remove property after PROVEN proof + both COVERED covers')
    pr.set_defaults(func=cmd_run)
    ps=subs.add_parser('status')
    ps.add_argument('--out',required=True)
    ps.set_defaults(func=cmd_status)
    return p


def main():
    parser=build_parser()
    args=parser.parse_args()
    if not hasattr(args, 'func'):
        parser.error('a command is required')
    try:args.func(args)
    except (ValueError,OSError,KeyError) as e:
        print(f'ERROR: {e}',file=sys.stderr);raise SystemExit(2)

if __name__=='__main__': main()
