#!/usr/bin/env python3
"""Copy unreduced checker-coverage summaries from the JasperGold host."""

import argparse
import hashlib
import json
from pathlib import Path

from fixme_svagen.jasper import load_connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('output/fixme_svagen/reduction_25_plan.json'))
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--out', type=Path, default=Path('output/fixme_svagen/checker_coverage_25_unreduced'))
    parser.add_argument('--remote-root', help='Remote checker coverage directory (defaults to the original 25-design directory)')
    parser.add_argument('--batch-name', default='25')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    import paramiko
    values = load_connection(args.env_file)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(values['HOST_NAME'], username=values['USERNAME'], password=values['PASSWORD'],
                timeout=10, banner_timeout=10, auth_timeout=10)
    sftp = ssh.open_sftp()
    remote_root = args.remote_root or plan['remote_base'].rstrip('/') + '/checker_coverage'
    files = []
    try:
        for item in plan['designs']:
            case_id = item['case_id']
            for filename in ('summary.json', 'summary.md'):
                remote = '{}/{}/run_{}_unreduced/{}'.format(remote_root, case_id, args.batch_name, filename)
                local = args.out / case_id / filename
                local.parent.mkdir(parents=True, exist_ok=True)
                with sftp.open(remote, 'rb') as stream:
                    data = stream.read()
                local.write_bytes(data)
                files.append({'remote': remote, 'local': str(local.resolve()),
                              'sha256': hashlib.sha256(data).hexdigest()})
        for filename in ('coverage_{}_unreduced_summary.json'.format(args.batch_name),
                         'coverage_{}_unreduced_status.json'.format(args.batch_name)):
            remote = '{}/{}'.format(remote_root, filename)
            local = args.out / filename
            with sftp.open(remote, 'rb') as stream:
                data = stream.read()
            local.write_bytes(data)
            files.append({'remote': remote, 'local': str(local.resolve()),
                          'sha256': hashlib.sha256(data).hexdigest()})
    finally:
        sftp.close()
        ssh.close()
    summaries = [json.loads((args.out / item['case_id'] / 'summary.json').read_text(encoding='utf8'))
                 for item in plan['designs']]
    if any(summary['status'] != 'complete' for summary in summaries):
        raise RuntimeError('Unreduced coverage is incomplete')
    (args.out / 'sync_manifest.json').write_text(
        json.dumps({'design_count': len(summaries), 'files': files}, indent=2, sort_keys=True) + '\n',
        encoding='utf8')
    print('Synced {} unreduced coverage summaries to {}'.format(len(summaries), args.out.resolve()))


if __name__ == '__main__':
    main()
