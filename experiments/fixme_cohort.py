#!/usr/bin/env python3
"""Extend the established cohort without regenerating completed assertions.

prepare: immutable-input dataset exports and a combined reduction plan.
deploy: hash-checked remote inputs, existing tools, and a detached worker.
watch: synchronize completed work, export retained rows, compare coverage,
       verify the combined artifacts, and publish reports when ready.
"""
import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from fixme_svagen.jasper import load_connection

ROOT = Path('output/fixme_svagen')
METRICS = ('coi_statement', 'coi_branch', 'default_statement', 'default_branch',
           'proof_core_statement', 'proof_core_branch')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def write_json(path, data):
    write(path, (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode())


def file_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def positive(r):
    return (r['assertion_status'] == 'proven' and r['vacuous'] is False
            and r['reason'] == 'neither_targeted_vacuity_reason')


def outside(r):
    return (r['assertion_status'] == 'proven' and r['vacuous'] is None
            and r['reason'] == 'outside_scope_non_implication')


def cohort_tag(plan):
    return plan.get('batch_name', str(len(plan['case_ids'])))


def prepare(args):
    original = read(args.base_plan)
    if args.split_plan:
        split = read(args.split_plan)
        ids = [cid for name in ('train', 'validation', 'test') for cid in split[name]]
        if len(ids) != len(set(ids)):
            raise ValueError('A design occurs in multiple split partitions')
        new = [cid for cid in ids if cid not in original['case_ids']]
    else:
        new = list(dict.fromkeys(args.add_case or []))
        if not new or set(new) & set(original['case_ids']):
            raise ValueError('Specify distinct new cases outside the baseline cohort')
        ids = original['case_ids'] + new
    tag = args.tag or str(len(ids))
    if not re.fullmatch(r'[A-Za-z0-9_]+', tag):
        raise ValueError('Cohort tag must contain letters, digits, or underscores only')
    if tag == cohort_tag(original):
        raise ValueError('Choose a distinct tag to preserve the baseline snapshot')
    plan = dict(original, case_ids=ids, cohort='fixed_' + tag, created_utc=now(),
                batch_name=tag, baseline_tag=cohort_tag(original),
                baseline_plan=str(args.base_plan),
                designs=[dict(item) for item in original['designs'] if item['case_id'] in ids],
                empty_cases={cid: reason for cid, reason in original['empty_cases'].items() if cid in ids},
                new_case_ids=new, status_name='reduction_' + tag + '_status.json')
    if args.split_plan:
        plan['split_plan'] = str(args.split_plan)
    summary = {'cohort': plan['cohort'], 'generated_utc': now(), 'case_ids': ids,
               'state': 'postprocessing_pending', 'files': {},
               'no_final_assertions': plan['empty_cases'],
               'reduction_plan': str(ROOT / ('reduction_' + tag + '_plan.json'))}
    handles = {}
    selections = {
        'dataset': ('dataset_' + tag + '_designs.jsonl', lambda r: True),
        'non_vacuous_before': ('non_vacuous_' + tag + '_designs_before_reduction.jsonl', positive),
        'vacuous': ('vacuous_' + tag + '_designs.jsonl',
                    lambda r: r['assertion_status'] == 'proven' and r['vacuous'] is True),
        'outside_scope': ('outside_scope_non_implication_' + tag + '_designs.jsonl', outside),
    }
    counts = {key: collections.Counter() for key in selections}
    try:
        for key, (filename, _) in selections.items():
            handles[key] = (ROOT / (filename + '.tmp')).open('w')
        for cid in ids:
            case_path = ROOT / 'cases' / cid / 'case.json'
            case = read(case_path)
            if case['case_id'] != cid or case['status'] not in ('done', 'no_final_assertions'):
                raise ValueError('Unfinished case: ' + cid)
            for row in case['rows']:
                if row['case_id'] != cid or digest(row['rtl'].encode()) != row['rtl_sha256']:
                    raise ValueError('Invalid case/RTL provenance: ' + cid)
                encoded = json.dumps(row, ensure_ascii=False) + '\n'
                for key, (_, select) in selections.items():
                    if select(row):
                        handles[key].write(encoded)
                        counts[key][cid] += 1
            if cid in new:
                if case['status'] != 'done' or not case['rows']:
                    raise ValueError('New case has no completed assertions: ' + cid)
                remote = plan['remote_base']
                plan['designs'].append({'case_id': cid, 'case': remote + '/inputs/' + cid + '/case.json',
                                        'out': remote + '/results/' + cid + '/run_01', 'run': 'run_01'})
        for stream in handles.values():
            stream.close()
        for key, (filename, _) in selections.items():
            path = ROOT / filename
            path.with_suffix(path.suffix + '.tmp').replace(path)
            info = {'path': str(path), 'rows': sum(counts[key].values()),
                    'rows_by_design': dict(counts[key]), 'sha256': digest(path.read_bytes())}
            summary['files'][key] = info
            write_json(path.with_suffix('.manifest.json'), dict(info, case_ids=ids,
                       design_count=len(ids), generated_utc=now(), selection=key))
    finally:
        for stream in handles.values():
            stream.close()
    summary['reduction'] = {'selected_before': summary['files']['non_vacuous_before']['rows']
                            + summary['files']['outside_scope']['rows']}
    path = ROOT / ('reduction_' + tag + '_plan.json')
    write_json(path, plan)
    write_json(ROOT / ('summary_' + tag + '_designs.json'), summary)
    print(json.dumps({'cohort': tag, 'files': {k: v['rows'] for k,v in summary['files'].items()},
                      'new_cases': new}), flush=True)
    return path


def connect():
    import paramiko
    v = load_connection(Path('.env'))
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(v['HOST_NAME'], username=v['USERNAME'], password=v['PASSWORD'],
                timeout=15, banner_timeout=15, auth_timeout=15)
    ssh.get_transport().set_keepalive(30)
    return ssh, ssh.open_sftp()


def mkdir(sftp, path):
    try:
        sftp.stat(path)
    except IOError:
        mkdir(sftp, str(Path(path).parent))
        sftp.mkdir(path)


def remote_json(sftp, path):
    with sftp.open(path, 'rb') as h:
        return json.loads(h.read())


def command(ssh, argv):
    _, out, err = ssh.exec_command(' '.join(shlex.quote(str(x)) for x in argv), timeout=40)
    data, error = out.read().decode(), err.read().decode()
    code = out.channel.recv_exit_status()
    if code:
        raise RuntimeError('Remote command exited {}: {}'.format(code, error[-1200:]))
    return data


def put(ssh, sftp, local, remote, refuse_changes=False):
    local = Path(local)
    expected = digest(local.read_bytes())
    try:
        sftp.stat(remote)
    except IOError:
        existing = None
    else:
        existing = command(ssh, ['sha256sum', remote]).split()[0]
    if existing == expected:
        return {'path': remote, 'sha256': expected, 'action': 'unchanged'}
    if refuse_changes and existing:
        raise ValueError('Refusing to replace a different existing case input: ' + remote)
    mkdir(sftp, str(Path(remote).parent))
    temp = remote + '.uploading'
    sftp.put(str(local), temp)
    if command(ssh, ['sha256sum', temp]).split()[0] != expected:
        raise IOError('Upload hash mismatch: ' + remote)
    if existing:
        command(ssh, ['cp', '-p', remote, remote + '.previous'])
        sftp.remove(remote)
    sftp.rename(temp, remote)
    print('Uploaded ' + remote, flush=True)
    return {'path': remote, 'sha256': expected, 'action': 'uploaded'}


def deploy(plan_path):
    plan = read(plan_path)
    base, tag = plan['remote_base'], cohort_tag(plan)
    ssh, sftp = connect()
    uploaded = []
    try:
        for cid in plan['new_case_ids']:
            case = read(ROOT / 'cases' / cid / 'case.json')
            rtls = list((Path('/home/narmada/btp/fixme/data/SVAGen') / cid.split('/')[0]
                         / 'Text' / cid.split('/')[1]).glob('*.v'))
            if len(rtls) != 1 or any(r['rtl'] != rtls[0].read_text(errors='replace') for r in case['rows']):
                raise ValueError('RTL differs from completed generation input: ' + cid)
            uploaded.append(put(ssh, sftp, ROOT / 'cases' / cid / 'case.json',
                                base + '/inputs/' + cid + '/case.json', True))
            uploaded.append(put(ssh, sftp, rtls[0], base + '/inputs/' + cid + '/' + rtls[0].name, True))
        for script in ('run_pipeline.py','extract_assertions.py','remove_exact_duplicates.py','formal_clocking.py',
                       'check_premise_consistency.py','formal_redundancy.py','validate_reduced_set.py'):
            uploaded.append(put(ssh,sftp,Path('experiments/assertion_extraction')/script,base+'/scripts/'+script))
        for local, remote in [
            (Path('experiments/assertion_extraction/run_batch.py'), base+'/results/run_batch.py'),
            (Path('experiments/checker_coverage/run_batch.py'),base+'/checker_coverage/run_batch.py'),
            (Path('experiments/run_cohort_postprocess.py'),base+'/run_cohort_postprocess.py'),
            (plan_path,base+'/'+Path(plan_path).name),
        ]:
            uploaded.append(put(ssh,sftp,local,remote))
        write_json(ROOT / ('remote_inputs_' + tag + '_sync.json'),
                   {'generated_utc': now(), 'new_case_ids': plan['new_case_ids'], 'files': uploaded})
        # Launch detached on the licensed host, independent of this laptop/session.
        launch = """import os,subprocess,json
from pathlib import Path
base={base!r}; tag={tag!r}
p=Path(base)/('postprocess_'+tag+'_status.json')
if p.exists():
 s=json.loads(p.read_text())
 if s.get('state')=='running':
  try:
   os.kill(s['pid'],0)
   print('Already running PID',s['pid']); raise SystemExit(0)
  except ProcessLookupError: pass
log=open(str(Path(base)/('postprocess_'+tag+'.log')),'ab',buffering=0)
cmd=['csh','-c', {shell!r}]
p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=os.path.expanduser('~'))
print('Started remote cohort worker PID',p.pid)
""".format(base=base, tag=tag, shell='source ~/cad_cshrc; exec python3 -u '
                   + shlex.quote(base+'/run_cohort_postprocess.py') + ' --plan '
                   + shlex.quote(base+'/'+Path(plan_path).name))
        print(command(ssh, ['python3','-c',launch]), flush=True)
    finally:
        sftp.close()
        ssh.close()


def download(sftp, remote, local, required=True):
    try:
        with sftp.open(remote,'rb') as h:
            data=h.read()
    except IOError:
        if required: raise
        return False
    write(local,data)
    return True


def summarize(plan, tag):
    summary_path=ROOT/('summary_'+tag+'_designs.json')
    summary=read(summary_path)
    from experiments.checker_coverage.compare_sets import aggregate
    for suffix, selection in (('', 'reduced'), ('_unreduced', 'unreduced')):
        folder=ROOT/('checker_coverage_'+tag+suffix)
        per_design={cid:read(folder/cid/'summary.json') for cid in
                    [item['case_id'] for item in plan['designs']]}
        for cid,reason in plan['empty_cases'].items():
            empty={'case_id':cid,'status':'no_final_assertions','assertion_count':0,
                   'coverage':{name:None for name in METRICS},'reason':reason}
            write_json(folder/cid/'summary.json',empty)
            write(folder/cid/'summary.md',('# '+cid+'\n\nNo final assertions: '+reason+'\n').encode())
            per_design[cid]=empty
        result={'generated_at_utc':now(),'assertion_set':selection,'designs':per_design,
                'counts':dict(collections.Counter(s['status'] for s in per_design.values())),
                'metrics':{name:aggregate(list(per_design.values()),name) for name in METRICS}}
        write_json(folder/('coverage_'+tag+'_'+selection+'_summary.json'),result)
    details={}
    for cid in plan['case_ids']:
        case=read(ROOT/'cases'/cid/'case.json')
        rows=case['rows']
        item={'case_id':cid,'status':case['status'],'signal_count':case['signal_count'],
              'final_assertions':len(rows),'status_counts':dict(collections.Counter(r['assertion_status'] for r in rows)),
              'non_vacuous':sum(positive(r) for r in rows),
              'vacuous':sum(r['assertion_status']=='proven' and r['vacuous'] is True for r in rows),
              'outside_scope':sum(outside(r) for r in rows)}
        if case['status']=='done':
            item['reduction']=read(ROOT/('reduction_'+tag)/cid/'summary.json')
            item['coverage_comparison']=read(ROOT/('checker_coverage_'+tag+'_comparison')/cid/'comparison.json')
        details[cid]=item
    comparison=read(ROOT/('checker_coverage_'+tag+'_comparison')/('coverage_'+tag+'_comparison.json'))
    summary['state']='complete'
    summary['coverage']=comparison['coverage']
    manifest=read(ROOT/('non_vacuous_'+tag+'_designs_after_reduction.manifest.json'))
    summary['reduction'].update(retained_after=manifest['export_rows'],
                              retained_non_vacuous=manifest['retained_non_vacuous'],
                              retained_outside_scope_non_implication=manifest['retained_outside_scope_non_implication'])
    retained=ROOT/('retained_'+tag+'_designs_after_reduction.jsonl')
    if not retained.exists():retained.symlink_to('non_vacuous_'+tag+'_designs_after_reduction.jsonl')
    records=file_rows(retained)
    summary['files']['retained_after']={'path':str(retained),'rows':len(records),
        'sha256':digest(retained.read_bytes()),'rows_by_design':dict(collections.Counter(r['case_id'] for r in records)),
        'label_counts':{'proved_non_vacuous':manifest['retained_non_vacuous'],
                        'proved_outside_scope_non_implication':manifest['retained_outside_scope_non_implication']}}
    summary['generated_utc']=now()
    write_json(summary_path,summary)
    write_json(ROOT/('results_'+tag+'_detailed.json'),dict(summary,designs=details))
    flat=[]
    for item in details.values():
        row={k:v for k,v in item.items() if k not in ('status_counts','reduction','coverage_comparison')}
        row.update({'proof_'+k:v for k,v in item['status_counts'].items()})
        for key,value in item.get('reduction',{}).items():
            if isinstance(value,(int,float)):row['reduction_'+key]=value
        comp=item.get('coverage_comparison',{}).get('coverage',{})
        for metric in METRICS:
            for selection in ('unreduced','reduced'):
                data=comp.get(metric,{}).get(selection)
                for field in ('covered','total','percent'):
                    row[metric+'_'+selection+'_'+field]=data.get(field) if data else None
            row[metric+'_delta_pp']=comp.get(metric,{}).get('delta_pp')
        flat.append(row)
    with (ROOT/('results_'+tag+'_per_design.csv')).open('w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader();writer.writerows(flat)
    lines=['# FIXME SVAGen Text: '+tag+' processed cases','',
           '{} cases have final assertion rows; {} produced none.'.format(len(plan['designs']),len(plan['empty_cases'])),
           '', '| Dataset | Rows |','|---|---:|']
    lines += ['| {} | {} |'.format(k,v['rows']) for k,v in summary['files'].items()]
    lines += ['', '## Per-design assertion and reduction results','',
              '| Design | Final | Non-vacuous | Vacuous | Non-implication | Selected | Exact duplicates removed | Formally removed | Retained |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for item in details.values():
        reduction=item.get('reduction',{})
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            item['case_id'],item['final_assertions'],item['non_vacuous'],item['vacuous'],item['outside_scope'],
            *[reduction.get(k,'N/A') for k in ('selected_proven_nonvacuous_or_out_of_scope',
              'exact_duplicate_copies_removed','formal_removed','retained')]))
    lines += ['', '## Coverage comparison','',
              (ROOT/('checker_coverage_'+tag+'_comparison')/('coverage_'+tag+'_comparison.md')).read_text(),
              '', 'Labels cover only the two targeted implication vacuity cases. '
              'Checker coverage is not requirement completeness. The retained dataset includes proven non-implications.']
    text='\n'.join(lines)+'\n'
    write(ROOT/('results_'+tag+'_detailed.md'),text.encode())
    write(ROOT/('summary_'+tag+'_designs.md'),text.encode())


def finalize(plan_path):
    plan=read(plan_path);tag=cohort_tag(plan);base=plan['remote_base']
    baseline_tag=plan.get('baseline_tag', '30')
    ssh,sftp=connect()
    try:
        state=remote_json(sftp,base+'/postprocess_'+tag+'_status.json')
        write_json(ROOT/('postprocess_'+tag+'_remote_status.json'),state)
        for item in plan['designs']:
            cid=item['case_id']
            if cid not in plan['new_case_ids']:
                for stem in ('reduction_','checker_coverage_'):
                    source=ROOT/(stem+baseline_tag)/cid;dest=ROOT/(stem+tag)/cid
                    shutil.copytree(source,dest,dirs_exist_ok=True)
                shutil.copytree(ROOT/('checker_coverage_'+baseline_tag+'_unreduced')/cid,
                                ROOT/('checker_coverage_'+tag+'_unreduced')/cid,dirs_exist_ok=True)
                continue
            try:
                remote_reduction=remote_json(sftp,item['out']+'/summary.json')
            except IOError:
                remote_reduction=None
            if remote_reduction and remote_reduction.get('status')=='VALIDATED':
                dest=ROOT/('reduction_'+tag)/cid
                if not (dest/'summary.json').exists() or read(dest/'summary.json')!=remote_reduction:
                    for filename in ('summary.json','retained_assertions.jsonl','retained_assertions.svh',
                                     'input_metadata.json','extracted/selected_assertions.jsonl',
                                     'deduplicated/unique_assertions.jsonl','deduplicated/duplicate_groups.csv',
                                     'pruning/results.csv','pruning/state.json','consistency/verified.json',
                                     'consistency/rtl_witness.sv','consistency/metadata.json',
                                     'consistency/status.tsv','validation/verified.json','validation/status.tsv'):
                        download(sftp,item['out']+'/'+filename,dest/filename,False)
            for selection,suffix in (('reduced',''),('unreduced','_unreduced')):
                for ext in ('json','md'):
                    download(sftp,base+'/checker_coverage/'+cid+'/run_'+tag+('_final/' if selection=='reduced' else '_unreduced/')+'summary.'+ext,
                             ROOT/('checker_coverage_'+tag+suffix)/cid/('summary.'+ext),False)
        if state['state']!='complete':
            print('Remote worker: '+state['state']+'; current '+state.get('current','none'),flush=True)
            return False
    finally:
        sftp.close();ssh.close()
    subprocess.run([sys.executable,'experiments/assertion_extraction/export_non_vacuous_after_reduction.py',
                    '--plan',str(plan_path),'--before-reduction',str(ROOT/('non_vacuous_'+tag+'_designs_before_reduction.jsonl')),
                    '--output',str(ROOT/('non_vacuous_'+tag+'_designs_after_reduction.jsonl')),'--include-outside-scope'],check=True)
    subprocess.run([sys.executable,'experiments/checker_coverage/compare_sets.py','--plan',str(plan_path),
                    '--reduced',str(ROOT/('checker_coverage_'+tag)),
                    '--unreduced',str(ROOT/('checker_coverage_'+tag+'_unreduced')),
                    '--out',str(ROOT/('checker_coverage_'+tag+'_comparison')),
                    '--report-stem','coverage_'+tag+'_comparison'],check=True)
    summarize(plan,tag)
    verify(plan_path)
    ssh,sftp=connect()
    try:
        for cid in plan['case_ids']:
            if cid in plan['empty_cases']:continue
            for ext in ('json','md'):
                put(ssh,sftp,ROOT/('checker_coverage_'+tag+'_comparison')/cid/('comparison.'+ext),
                    base+'/checker_coverage/'+cid+'/coverage_comparison.'+ext)
        for local in [ROOT/('summary_'+tag+'_designs.json'),ROOT/('summary_'+tag+'_designs.md'),
                      ROOT/('results_'+tag+'_detailed.json'),ROOT/('results_'+tag+'_detailed.md'),
                      ROOT/('results_'+tag+'_per_design.csv'),
                      ROOT/('checker_coverage_'+tag)/('coverage_'+tag+'_reduced_summary.json'),
                      ROOT/('checker_coverage_'+tag+'_unreduced')/('coverage_'+tag+'_unreduced_summary.json')]:
            put(ssh,sftp,local,base+'/checker_coverage/'+local.name)
        for ext in ('json','md'):
            local=ROOT/('checker_coverage_'+tag+'_comparison')/('coverage_'+tag+'_comparison.'+ext)
            put(ssh,sftp,local,base+'/checker_coverage/'+local.name)
    finally:sftp.close();ssh.close()
    print('Combined cohort datasets, reduction, coverage, and published reports complete.',flush=True)
    return True


def verify(plan_path):
    plan=read(plan_path);tag=cohort_tag(plan)
    summary=read(ROOT/('summary_'+tag+'_designs.json'))
    assert summary['state']=='complete'
    for key,info in summary['files'].items():
        path=Path(info['path']);assert digest(path.read_bytes())==info['sha256']
        rows=file_rows(path);assert len(rows)==info['rows']
        assert set(r['case_id'] for r in rows)<=set(plan['case_ids'])
        if key=='non_vacuous_before':assert all(positive(r) for r in rows)
        if key=='vacuous':assert all(r['assertion_status']=='proven' and r['vacuous'] is True for r in rows)
        if key=='outside_scope':assert all(outside(r) for r in rows)
        if key=='retained_after':assert all(positive(r) or outside(r) for r in rows)
    comparison=read(ROOT/('checker_coverage_'+tag+'_comparison')/('coverage_'+tag+'_comparison.json'))
    retained=collections.Counter()
    for item in plan['designs']:
        cid=item['case_id'];case=read(ROOT/'cases'/cid/'case.json')
        selected=[r for r in case['rows'] if positive(r) or outside(r)]
        rs=read(ROOT/('reduction_'+tag)/cid/'summary.json')
        keep=file_rows(ROOT/('reduction_'+tag)/cid/'retained_assertions.jsonl')
        assert rs['status']=='VALIDATED' and rs['retained']==len(keep)
        assert rs['selected_proven_nonvacuous_or_out_of_scope']==len(selected)
        assert rs['unique_after_deduplication']==len(selected)-rs['exact_duplicate_copies_removed']
        assert len(keep)==rs['unique_after_deduplication']-rs['formal_removed']
        for r in keep:
            source=case['rows'][r['row_number']-1]
            assert all(r.get(k)==source.get(k) for k in ('case_id','assertion','vacuous','reason','rtl_sha256'))
            retained[(cid,r['assertion'])]+=1
        assert comparison['designs'][cid]['assertion_counts']=={'unreduced':len(selected),'reduced':len(keep)}
        for suffix in ('','_unreduced'):
            cs=read(ROOT/('checker_coverage_'+tag+suffix)/cid/'summary.json')
            assert cs['status'] in ('complete','no_retained_assertions')
            assert cs['source_sha256']==case['rows'][0]['rtl_sha256']
    exported=file_rows(ROOT/('non_vacuous_'+tag+'_designs_after_reduction.jsonl'))
    assert retained==collections.Counter((r['case_id'],r['assertion']) for r in exported)
    print('Verified combined '+tag+'-case dataset and all per-design reduction/coverage sources.',flush=True)


def status(plan_path):
    plan=read(plan_path);tag=cohort_tag(plan);base=plan['remote_base']
    ssh,sftp=connect()
    try:
        worker=remote_json(sftp,base+'/postprocess_'+tag+'_status.json')
        progress={'checked_utc':now(),'worker':worker,'designs':{}}
        for item in plan['designs']:
            if item['case_id'] not in plan['new_case_ids']:continue
            entry={}
            try:
                entry['reduction']=remote_json(sftp,item['out']+'/summary.json')
            except IOError:
                try:
                    p=remote_json(sftp,item['out']+'/pruning/state.json')
                    entry['pruning']={'checked':len(p['processed']),'total':len(p['order']),
                                      'removed':len(p['removed']),'remaining':len(p['remaining'])}
                except IOError:entry['reduction']='preparing_or_checking_consistency'
            for selection,run in (('reduced','final'),('unreduced','unreduced')):
                try:
                    s=remote_json(sftp,base+'/checker_coverage/'+item['case_id']+'/run_'+tag+'_'+run+'/summary.json')
                    entry[selection+'_coverage']=s['status']
                except IOError:entry[selection+'_coverage']='pending'
            progress['designs'][item['case_id']]=entry
        write_json(ROOT/('postprocess_'+tag+'_progress.json'),progress)
        print(json.dumps(progress,indent=2),flush=True)
    finally:sftp.close();ssh.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('prepare','deploy','watch','sync','verify','status'))
    ap.add_argument('--add-case',action='append')
    ap.add_argument('--base-plan',type=Path,default=ROOT/'reduction_30_plan.json')
    ap.add_argument('--split-plan',type=Path,help='Restrict the cohort to these training/validation/test designs')
    ap.add_argument('--tag',help='Distinct output tag, for example training35')
    ap.add_argument('--plan',type=Path,default=ROOT/'reduction_35_plan.json')
    ap.add_argument('--poll',type=int,default=120)
    args=ap.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='deploy':deploy(args.plan)
    elif args.action=='verify':verify(args.plan)
    elif args.action=='status':status(args.plan)
    elif args.action=='sync':finalize(args.plan)
    else:
        while True:
            try:
                if finalize(args.plan):return
                state=read(ROOT/('postprocess_'+cohort_tag(read(args.plan))+'_remote_status.json'))
                if state['state']=='failed':raise RuntimeError('Remote worker failed: '+state.get('current','unknown'))
            except (OSError,EOFError) as error:
                print('Temporary sync failure: '+str(error),flush=True)
            time.sleep(args.poll)


if __name__=='__main__':main()
