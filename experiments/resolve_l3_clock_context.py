#!/usr/bin/env python3
"""Audit and correct the L3/3 clock-context discrepancy, retaining old evidence."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shlex

from experiments.fixme_cohort import connect, command, remote_json, write_json, put
from experiments.assertion_extraction.extract_assertions import resolve_property
from fixme_svagen.jasper import parse_log
from fixme_svagen.sva import insert_in_module

ROOT = Path('output/fixme_svagen')
CASE = ROOT / 'cases/L3/3/case.json'
AUDIT = ROOT / 'audits/L3_3_clock_resolution'
BASE = '/storage/ckarfa/narmada/experiments'
REMOTE = BASE + '/results/L3/3/clock_resolution_v1'
CONTEXT = {'version': 1, 'declarations': [{'signal': 'forever_cpuclk', 'edges': 'both'}],
           'rationale': 'Model both main-clock edges; derived clock aliases follow RTL; reset remains unconstrained, not a periodic clock.'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def probe():
    raw = CASE.read_bytes()
    case = json.loads(raw)
    row = case['rows'][982]
    name, body, _, status, _ = resolve_property(row)
    if name != 'l2rm_clk_reset' or status != 'ok':
        raise ValueError('Unexpected source assertion')
    AUDIT.mkdir(parents=True, exist_ok=True)
    request = {'case_sha256': sha(raw), 'source_row': 983, 'property_name': name,
               'expression': body, 'formal_clocking': CONTEXT, 'remote': REMOTE}
    write_json(AUDIT / 'request.json', request)
    single = insert_in_module(row['rtl'], row['top'],
                              'sangam_assert: assert property (\n' + body + '\n);')
    ssh, sftp = connect()
    try:
        command(ssh, ['mkdir', '-p', REMOTE + '/combined', REMOTE + '/individual'])
        with sftp.open(BASE + '/results/L3/3/run_01/consistency/rtl_witness.sv', 'rb') as h:
            witness = h.read()
        with sftp.open(BASE + '/results/L3/3/run_01/consistency/run.tcl', 'rb') as h:
            tcl = h.read().decode()
        tcl = re.sub(r'^clock[^\n]*\n', '', tcl, flags=re.M)
        tcl = tcl.replace('reset -none', 'clock -both_edges forever_cpuclk\nreset -none')
        (AUDIT / 'combined.sv').write_bytes(witness)
        (AUDIT / 'combined.tcl').write_text(tcl)
        (AUDIT / 'individual.sv').write_text(single)
        (AUDIT / 'individual.tcl').write_text(
            'analyze -sv12 dut.sv\nelaborate -top ' + row['top']
            + '\nclock -both_edges forever_cpuclk\nreset -none\nprove -all\n'
              'report -summary -force -result\nexit -force\n')
        for local, remote in [('combined.sv', 'combined/rtl_witness.sv'),
                              ('combined.tcl', 'combined/run.tcl'),
                              ('individual.sv', 'individual/dut.sv'),
                              ('individual.tcl', 'individual/run.tcl'),
                              ('request.json', 'request.json')]:
            put(ssh, sftp, AUDIT / local, REMOTE + '/' + remote, True)
        driver = '''import fcntl,json,sys
from pathlib import Path
sys.path.insert(0, '/storage/ckarfa/narmada/experiments/scripts')
from run_pipeline import run_jg
p=Path(REMOTE_REPLACE)
lock=(p/'probe.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if not (p/'complete.json').exists():
 for stage in ('combined','individual'):
  run_jg(p/stage,'/home/Cadence_tools/jasper_2022.12/bin/jg',180)
  print('Completed '+stage,flush=True)
 (p/'complete.json').write_text(json.dumps({'status':'complete'}))
'''.replace('REMOTE_REPLACE', repr(REMOTE))
        (AUDIT / 'run_probe.py').write_text(driver)
        put(ssh, sftp, AUDIT / 'run_probe.py', REMOTE + '/run_probe.py')
        launch = '''import subprocess
from pathlib import Path
p=Path(REMOTE_REPLACE)
log=open(str(p/'probe.log'),'ab',buffering=0)
c=subprocess.Popen(['csh','-c',SHELL_REPLACE],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Started independent clock checks',c.pid,str(p))
'''.replace('REMOTE_REPLACE', repr(REMOTE)).replace('SHELL_REPLACE', repr(
            'source ~/cad_cshrc; exec python3 -u ' + shlex.quote(REMOTE + '/run_probe.py')))
        print(command(ssh, ['python3', '-c', launch]), flush=True)
    finally:
        sftp.close()
        ssh.close()


def inspect():
    ssh, sftp = connect()
    try:
        for remote, local in [('combined/status.tsv', 'combined_status.tsv'),
                              ('combined/jg.log', 'combined_jg.log'),
                              ('individual/jg.log', 'individual_jg.log'),
                              ('probe.log', 'probe.log')]:
            try:
                with sftp.open(REMOTE + '/' + remote, 'rb') as h:
                    data = h.read()
                (AUDIT / local).write_bytes(data)
            except IOError:
                pass
        complete = remote_json(sftp, REMOTE + '/complete.json')
        if complete.get('status') != 'complete':
            raise ValueError('Clock checks are not complete')
    finally:
        sftp.close()
        ssh.close()
    states = dict(line.split('\t') for line in (AUDIT / 'combined_status.tsv').read_text().splitlines())
    assertion_states = {k: v for k, v in states.items() if k.startswith('RTL_CHECK_')}
    if len(assertion_states) != 529 or states.get('RTL_TRACE_EXISTS') != 'covered':
        raise ValueError('Unexpected combined witness status/count')
    failures = {k: v for k, v in assertion_states.items() if v != 'proven'}
    if failures != {'RTL_CHECK_0983': 'cex'}:
        raise ValueError('Unexpected combined failure set: ' + repr(failures))
    single = parse_log((AUDIT / 'individual_jg.log').read_text(), 0)
    if single.assertion_status != 'cex':
        raise ValueError('Independent property check did not confirm the counterexample')
    evidence = {'status': 'confirmed', 'formal_clocking': CONTEXT,
                'combined_status_counts': dict(collections.Counter(assertion_states.values())),
                'trace_status': states['RTL_TRACE_EXISTS'], 'independent_status': single.assertion_status,
                'remote': REMOTE, 'log_sha256': sha((AUDIT / 'individual_jg.log').read_bytes())}
    write_json(AUDIT / 'evidence.json', evidence)
    print(json.dumps(evidence, indent=2), flush=True)
    return evidence


def apply():
    evidence = inspect()
    request = json.loads((AUDIT / 'request.json').read_text())
    raw = CASE.read_bytes()
    if sha(raw) != request['case_sha256']:
        raise ValueError('Source case changed since probe; refusing to overwrite')
    case = json.loads(raw)
    row = case['rows'][982]
    if row['assertion_status'] != 'proven' or resolve_property(row)[1] != request['expression']:
        raise ValueError('Unexpected source row')
    backup = AUDIT / 'case.pre_clock_recheck.json'
    if backup.exists() and backup.read_bytes() != raw:
        raise ValueError('Audit backup already differs')
    backup.write_bytes(raw)
    previous = {k: row.get(k) for k in ('assertion_status','vacuous','reason','jasper_log',
                                       'precondition_status','contrapositive_status')}
    corrected_log = CASE.parent / 'final_logs_reset_none/l2rm_clk_reset_clock_recheck.log'
    corrected_log.write_bytes((AUDIT / 'individual_jg.log').read_bytes())
    row.update(assertion_status='cex', vacuous=None, reason='outside_scope_non_implication',
               precondition_status=None, contrapositive_status=None,
               jasper_log=str(corrected_log.relative_to(ROOT)),
               formal_recheck={'method': 'independent_reset_event_check_with_both_clock_edges',
                   'previous_result': previous, 'evidence': str(AUDIT / 'evidence.json'),
                   'log_sha256': evidence['log_sha256'], 'formal_clocking': CONTEXT})
    case['formal_clocking'] = CONTEXT
    case['formal_clock_recheck'] = dict(evidence, corrected_rows=[983], original_case_sha256=sha(raw))
    write_json(CASE, case)
    write_json(AUDIT / 'application.json', {'original_sha256':sha(raw),
        'corrected_sha256':sha(CASE.read_bytes()), 'corrected_rows':[983],
        'assertion_count_unchanged':len(case['rows']), 'assertion_text_unchanged':True})
    print('Corrected row 983 to cex; preserved original result and assertion text.', flush=True)


def publish():
    application = json.loads((AUDIT / 'application.json').read_text())
    if sha(CASE.read_bytes()) != application['corrected_sha256']:
        raise ValueError('Corrected case changed since audit')
    ssh, sftp = connect()
    try:
        incoming = BASE + '/inputs/L3/3/case.clock_rechecked.' + application['corrected_sha256'][:12] + '.json'
        put(ssh, sftp, CASE, incoming, True)
        put(ssh, sftp, AUDIT / 'evidence.json', REMOTE + '/evidence.json')
        put(ssh, sftp, AUDIT / 'application.json', REMOTE + '/application.json')
        code = '''import fcntl,hashlib,json
from pathlib import Path
base=Path(BASE_REPLACE)
change=CHANGE_REPLACE
lock=(base/'postprocess_35.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=base/'inputs/L3/3/case.json'
original=hashlib.sha256(source.read_bytes()).hexdigest()
if original==change['corrected_sha256']:
 print('Remote correction already published')
 raise SystemExit(0)
if original!=change['original_sha256']:raise RuntimeError('Remote case changed unexpectedly')
out=base/'results/L3/3/run_01'
if (out/'summary.json').exists():raise RuntimeError('Refusing to replace a run with a summary')
archived=out.with_name('run_01_pre_clock_fix_'+original[:12])
backup=source.with_name('case.pre_clock_recheck.'+original[:12]+'.json')
if backup.exists() or archived.exists():raise RuntimeError('An archive already exists; inspect before retrying')
if out.exists():out.rename(archived)
source.rename(backup)
Path(INCOMING_REPLACE).replace(source)
print('Preserved original case:',backup)
print('Preserved original failed reduction:',archived)
print('Published audited corrected case:',source)
'''.replace('BASE_REPLACE', repr(BASE)).replace('CHANGE_REPLACE', repr(application)).replace('INCOMING_REPLACE', repr(incoming))
        print(command(ssh, ['python3','-c',code]), flush=True)
    finally:
        sftp.close()
        ssh.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('probe','inspect','apply','publish'))
    args = parser.parse_args()
    {'probe':probe, 'inspect':inspect, 'apply':apply, 'publish':publish}[args.action]()


if __name__ == '__main__':
    main()
