#!/usr/bin/env python3
"""Export a fixed, full-record dataset of validated retained assertions.

The reduction pipeline also retains proved non-implication assertions. Those
have vacuous=null; include them with --include-outside-scope.
"""

import argparse
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fixme_svagen.jasper import load_connection


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('output/fixme_svagen/reduction_25_plan.json'))
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--cases-root', type=Path, default=Path('output/fixme_svagen/cases'))
    parser.add_argument('--before-reduction', type=Path,
                        default=Path('output/fixme_svagen/non_vacuous_25_designs_before_reduction.jsonl'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/fixme_svagen/non_vacuous_25_designs_after_reduction.jsonl'))
    parser.add_argument('--include-outside-scope', action='store_true',
                        help='Also export validated retained non-implications with vacuous=null')
    args = parser.parse_args()
    plan = read_json(args.plan)
    if not plan['designs'] or len({item['case_id'] for item in plan['designs']}) != len(plan['designs']):
        raise ValueError('Expected a nonempty reduction plan with distinct designs')
    before = collections.defaultdict(dict)
    for line in args.before_reduction.open(encoding='utf8'):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row['case_id'], row['assertion']
        if key[1] in before[key[0]]:
            raise ValueError('Duplicate before-reduction assertion {}'.format(key))
        before[key[0]][key[1]] = row
    if set(before) - {item['case_id'] for item in plan['designs']}:
        raise ValueError('Before-reduction snapshot includes designs outside the plan')

    import paramiko
    values = load_connection(args.env_file)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(values['HOST_NAME'], username=values['USERNAME'], password=values['PASSWORD'],
                timeout=10, banner_timeout=10, auth_timeout=10)
    sftp = ssh.open_sftp()
    records = []
    sources = {}
    try:
        for item in plan['designs']:
            case_id = item['case_id']
            local_case = args.cases_root / case_id / 'case.json'
            local_data = local_case.read_bytes()
            case = json.loads(local_data)
            if case['status'] != 'done' or case['case_id'] != case_id:
                raise ValueError('Local case is not completed: {}'.format(case_id))
            original = case['rows']
            remote = item['out'].rstrip('/')
            retained_path = remote + '/retained_assertions.jsonl'
            summary_path = remote + '/summary.json'
            with sftp.open(retained_path, 'rb') as stream:
                retained_data = stream.read()
            with sftp.open(summary_path, 'rb') as stream:
                summary_data = stream.read()
            retained = [json.loads(line) for line in retained_data.splitlines() if line.strip()]
            summary = json.loads(summary_data)
            if summary.get('status') != 'VALIDATED' or summary.get('retained') != len(retained):
                raise ValueError('Reduction is not validated or count differs for {}'.format(case_id))
            seen = set()
            retained_non_vacuous = []
            retained_outside_scope = []
            outside_scope = 0
            for selected in retained:
                number = selected.get('row_number')
                if not isinstance(number, int) or not 1 <= number <= len(original) or number in seen:
                    raise ValueError('Invalid/duplicate reduction row_number {} for {}'.format(number, case_id))
                seen.add(number)
                source = original[number - 1]
                for key in ('case_id', 'assertion', 'assertion_status', 'vacuous', 'reason', 'rtl_sha256'):
                    if selected.get(key) != source.get(key):
                        raise ValueError('Reduction row {} disagrees on {} for {}'.format(number, key, case_id))
                if selected.get('selected') is not True or source['assertion_status'] != 'proven':
                    raise ValueError('Retained row is unselected or unproven: {}:{}'.format(case_id, number))
                if source['vacuous'] is False:
                    if source['reason'] != 'neither_targeted_vacuity_reason':
                        raise ValueError('Unexpected non-vacuous reason for {}:{}'.format(case_id, number))
                    if source['assertion'] not in before[case_id] or before[case_id][source['assertion']] != source:
                        raise ValueError('Retained row not found in before-reduction snapshot: {}:{}'.format(case_id, number))
                    retained_non_vacuous.append(number)
                    records.append(source)
                elif source['vacuous'] is None and source['reason'] == 'outside_scope_non_implication':
                    outside_scope += 1
                    retained_outside_scope.append(number)
                    if args.include_outside_scope:
                        records.append(source)
                else:
                    raise ValueError('Unexpected retained label for {}:{}'.format(case_id, number))
            sources[case_id] = {
                'local_case': str(local_case), 'local_case_sha256': sha256(local_data),
                'remote_retained': retained_path, 'remote_retained_sha256': sha256(retained_data),
                'remote_summary': summary_path, 'remote_summary_sha256': sha256(summary_data),
                'retained_total': len(retained),
                'retained_non_vacuous': len(retained_non_vacuous),
                'retained_outside_scope_non_implication': outside_scope,
                'retained_non_vacuous_row_numbers': retained_non_vacuous,
                'retained_outside_scope_row_numbers': retained_outside_scope,
            }
    finally:
        sftp.close()
        ssh.close()
    expected = sum(source['retained_total' if args.include_outside_scope else 'retained_non_vacuous']
                   for source in sources.values())
    if len(records) != expected:
        raise AssertionError('Export count mismatch')
    output_data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records).encode('utf8')
    manifest = {
        'description': ('Fixed after-reduction snapshot of all validated retained final SANGAM assertions '
                        'from the designs in the reduction plan' if args.include_outside_scope else
                        'Fixed after-reduction snapshot of proved non-vacuous final SANGAM assertions '
                        'from the designs in the reduction plan'),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'selection': ('Validated retained assertion AND assertion_status=proven AND '
                      '(vacuous=false OR reason=outside_scope_non_implication)'
                      if args.include_outside_scope else
                      'Validated retained assertion AND assertion_status=proven AND vacuous=false'),
        'excluded_retained_assertions': (None if args.include_outside_scope else
                                         'Proved non-implication assertions with vacuous=null and '
                                         'reason=outside_scope_non_implication'),
        'plan': str(args.plan), 'plan_sha256': sha256(args.plan.read_bytes()),
        'before_reduction': str(args.before_reduction),
        'before_reduction_sha256': sha256(args.before_reduction.read_bytes()),
        'before_reduction_rows': sum(len(rows) for rows in before.values()),
        'export': str(args.output), 'export_sha256': sha256(output_data),
        'export_rows': len(records), 'design_count': len(plan['designs']),
        'empty_cases': plan.get('empty_cases', []),
        'designs_with_export_rows': sum(bool(source['retained_total' if args.include_outside_scope else
                                                    'retained_non_vacuous']) for source in sources.values()),
        'retained_total': sum(source['retained_total'] for source in sources.values()),
        'retained_non_vacuous': sum(source['retained_non_vacuous'] for source in sources.values()),
        'retained_outside_scope_non_implication': sum(source['retained_outside_scope_non_implication'] for source in sources.values()),
        'include_outside_scope': args.include_outside_scope,
        'sources_by_design': sources,
    }
    write_atomic(args.output, output_data)
    manifest_path = args.output.with_suffix('.manifest.json')
    write_atomic(manifest_path, (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode('utf8'))
    print('Exported {} validated retained assertions from {} designs to {} ({} non-vacuous, {} outside scope)'.format(
        len(records), len(sources), args.output.resolve(), manifest['retained_non_vacuous'],
        manifest['retained_outside_scope_non_implication'] if args.include_outside_scope else 0))


if __name__ == '__main__':
    main()
