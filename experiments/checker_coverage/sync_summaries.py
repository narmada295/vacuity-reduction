#!/usr/bin/env python3
"""Copy the 25 remote checker-coverage summaries into the local workspace."""

import argparse
import hashlib
import json
from pathlib import Path

from fixme_svagen.jasper import load_connection


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('output/fixme_svagen/reduction_25_plan.json'))
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--out', type=Path, default=Path('output/fixme_svagen/checker_coverage_25'))
    parser.add_argument('--remote-root', help='Remote checker coverage directory (defaults to the original 25-design directory)')
    parser.add_argument('--batch-name', default='25')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    values = load_connection(args.env_file)

    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(values['HOST_NAME'], username=values['USERNAME'],
                password=values['PASSWORD'], timeout=12)
    sftp = ssh.open_sftp()
    remote_root = args.remote_root or plan['remote_base'].rstrip('/') + '/checker_coverage'
    records = []
    try:
        for item in plan['designs']:
            case_id = item['case_id']
            for filename in ('summary.json', 'summary.md'):
                remote = remote_root + '/' + case_id + '/run_{}_final/'.format(args.batch_name) + filename
                local = args.out / case_id / filename
                local.parent.mkdir(parents=True, exist_ok=True)
                with sftp.open(remote, 'rb') as source:
                    data = source.read()
                local.write_bytes(data)
                records.append({'remote': remote, 'local': str(local.resolve()),
                                'sha256': sha256(data)})
        for filename in ('coverage_{}_summary.json'.format(args.batch_name),
                         'coverage_{}_status.json'.format(args.batch_name)):
            remote = remote_root + '/' + filename
            local = args.out / filename
            local.parent.mkdir(parents=True, exist_ok=True)
            with sftp.open(remote, 'rb') as source:
                data = source.read()
            local.write_bytes(data)
            records.append({'remote': remote, 'local': str(local.resolve()),
                            'sha256': sha256(data)})
    finally:
        sftp.close()
        ssh.close()

    aggregate = json.loads((args.out / 'coverage_{}_summary.json'.format(args.batch_name)).read_text(encoding='utf8'))
    if len(aggregate['designs']) != len(plan['designs']):
        raise RuntimeError('Remote aggregate does not include all planned designs')
    if any(value['status'] not in ('complete', 'no_retained_assertions')
           for value in aggregate['designs'].values()):
        raise RuntimeError('Remote aggregate contains an incomplete design')
    def cell(metric):
        if metric is None or metric['total'] == 0:
            return 'N/A'
        return '{}/{} ({:.2f}%)'.format(metric['covered'], metric['total'], metric['percent'])

    lines = ['# Checker coverage: {} reduced FIXME designs'.format(len(plan['designs'])), '',
             'Coverage percentages are item-weighted JasperGold checker coverage. '
             'The no-assertion design has undefined coverage.', '',
             '| Design | Retained | COI statement | COI branch | Proof statement | Proof branch |',
             '|---|---:|---:|---:|---:|---:|']
    status_counts = {}
    for item in plan['designs']:
        case_id = item['case_id']
        summary = json.loads((args.out / case_id / 'summary.json').read_text(encoding='utf8'))
        for status, count in summary.get('assertion_status_counts', {}).items():
            status_counts[status] = status_counts.get(status, 0) + count
        coverage = summary['coverage']
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            case_id, summary['retained_count'],
            cell(coverage['coi_statement']), cell(coverage['coi_branch']),
            cell(coverage['proof_core_statement']), cell(coverage['proof_core_branch'])))
    metrics = aggregate['metrics']
    lines += ['', '## Aggregate', '',
              '- Retained assertion results: {}'.format(', '.join(
                  '{} {}'.format(count, status) for status, count in sorted(status_counts.items()))),
              '- COI statement: {}'.format(cell(metrics['coi_statement'])),
              '- COI branch: {}'.format(cell(metrics['coi_branch'])),
              '- Proof-core statement: {}'.format(cell(metrics['proof_core_statement'])),
              '- Proof-core branch: {}'.format(cell(metrics['proof_core_branch'])), '']
    (args.out / 'coverage_{}_summary.md'.format(args.batch_name)).write_text('\n'.join(lines), encoding='utf8')
    manifest = {'design_count': len(plan['designs']), 'files': records}
    (args.out / 'sync_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf8')
    print('Synced {} summaries ({} files) to {}'.format(
        len(plan['designs']), len(records), args.out.resolve()))


if __name__ == '__main__':
    main()
