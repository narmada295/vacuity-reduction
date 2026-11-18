#!/usr/bin/env python3
"""Publish the fixed 30-design reports to the common JasperGold host tree."""

import json
from pathlib import Path

from fixme_svagen.jasper import load_connection


ROOT = Path('output/fixme_svagen')


def main():
    plan = json.loads((ROOT / 'reduction_30_plan.json').read_text(encoding='utf8'))
    report = json.loads((ROOT / 'checker_coverage_30_comparison/coverage_30_comparison.json').read_text(encoding='utf8'))
    if set(plan['case_ids']) != set(report['designs']) or len(plan['case_ids']) != 30:
        raise ValueError('The combined report does not match the 30-design plan')
    import paramiko
    values = load_connection(Path('.env'))
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(values['HOST_NAME'], username=values['USERNAME'], password=values['PASSWORD'],
                timeout=10, banner_timeout=10, auth_timeout=10)
    sftp = ssh.open_sftp()
    base = plan['remote_base'].rstrip('/')
    remote = base + '/checker_coverage'

    def put(source, destination):
        source = Path(source)
        sftp.put(str(source), destination)
        if sftp.stat(destination).st_size != source.stat().st_size:
            raise IOError('Upload size mismatch: ' + destination)

    try:
        for cid in plan['case_ids']:
            directory = remote + '/' + cid
            sftp.stat(directory)
            for suffix in ('json', 'md'):
                put(ROOT / 'checker_coverage_30_comparison' / cid / ('comparison.' + suffix),
                    directory + '/coverage_comparison.' + suffix)
        for source, destination in (
            (ROOT / 'reduction_30_plan.json', base + '/reduction_30_plan.json'),
            (ROOT / 'summary_30_designs.json', remote + '/summary_30_designs.json'),
            (ROOT / 'summary_30_designs.md', remote + '/summary_30_designs.md'),
            (ROOT / 'checker_coverage_30/coverage_30_reduced_summary.json',
             remote + '/coverage_30_reduced_summary.json'),
            (ROOT / 'checker_coverage_30_unreduced/coverage_30_unreduced_summary.json',
             remote + '/coverage_30_unreduced_summary.json'),
            (ROOT / 'checker_coverage_30_comparison/coverage_30_comparison.json',
             remote + '/coverage_30_comparison.json'),
            (ROOT / 'checker_coverage_30_comparison/coverage_30_comparison.md',
             remote + '/coverage_30_comparison.md'),
        ):
            put(source, destination)
    finally:
        sftp.close()
        ssh.close()
    print('Published 30-design reports to ' + remote)


if __name__ == '__main__':
    main()
