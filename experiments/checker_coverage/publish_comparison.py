#!/usr/bin/env python3
"""Copy validated coverage comparison summaries to the JasperGold host."""

import argparse
import json
from pathlib import Path

from fixme_svagen.jasper import load_connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('output/fixme_svagen/reduction_25_plan.json'))
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--input', type=Path, default=Path('output/fixme_svagen/checker_coverage_25_comparison'))
    parser.add_argument('--remote-root', help='Remote checker coverage directory (defaults to the original 25-design directory)')
    parser.add_argument('--report-stem', default='coverage_25_comparison')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf8'))
    aggregate = json.loads((args.input / (args.report_stem + '.json')).read_text(encoding='utf8'))
    if len(plan['designs']) != aggregate['design_count'] or set(aggregate['designs']) != {
            item['case_id'] for item in plan['designs']}:
        raise ValueError('Comparison does not match the fixed 25-design plan')
    import paramiko
    values = load_connection(args.env_file)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(values['HOST_NAME'], username=values['USERNAME'], password=values['PASSWORD'],
                timeout=10, banner_timeout=10, auth_timeout=10)
    sftp = ssh.open_sftp()
    remote_root = args.remote_root or plan['remote_base'].rstrip('/') + '/checker_coverage'
    copied = 0
    try:
        for item in plan['designs']:
            case_id = item['case_id']
            for suffix in ('json', 'md'):
                local = args.input / case_id / ('comparison.' + suffix)
                remote = '{}/{}/coverage_comparison.{}'.format(remote_root, case_id, suffix)
                sftp.put(str(local), remote)
                copied += 1
        for suffix in ('json', 'md'):
            filename = args.report_stem + '.' + suffix
            sftp.put(str(args.input / filename), remote_root + '/' + filename)
            copied += 1
    finally:
        sftp.close()
        ssh.close()
    print('Published {} comparison files to {}'.format(copied, remote_root))


if __name__ == '__main__':
    main()
