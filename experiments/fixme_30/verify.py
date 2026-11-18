#!/usr/bin/env python3
"""Check the fixed 30-design assertion, reduction, and coverage snapshot."""

import collections
import hashlib
import json
from pathlib import Path


ROOT = Path('output/fixme_svagen')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def rows(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    summary = read(ROOT / 'summary_30_designs.json')
    plan = read(ROOT / 'reduction_30_plan.json')
    coverage = read(ROOT / 'checker_coverage_30_comparison/coverage_30_comparison.json')
    ids = set(summary['case_ids'])
    assert len(ids) == 30 and ids == set(plan['case_ids']) == set(coverage['designs'])
    assert len(plan['designs']) == 28 and len(plan['empty_cases']) == 2
    assert set(plan['empty_cases']) == set(summary['no_final_assertions'])

    data = {}
    for label, info in summary['files'].items():
        path = Path(info['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256'], path
        data[label] = rows(path)
        assert len(data[label]) == info['rows'], path
        assert all(row['case_id'] in ids for row in data[label]), path
    assert len(data['dataset']) == 11495
    assert len(data['non_vacuous_before']) == 3289
    assert len(data['vacuous']) == 672
    assert len(data['retained_after']) == 1194
    assert all(row['assertion_status'] == 'proven' and row['vacuous'] is False
               for row in data['non_vacuous_before'])
    assert all(row['assertion_status'] == 'proven' and
               ((row['vacuous'] is False and row['reason'] == 'neither_targeted_vacuity_reason') or
                (row['vacuous'] is None and row['reason'] == 'outside_scope_non_implication'))
               for row in data['retained_after'])
    assert all(row['assertion_status'] == 'proven' and row['vacuous'] is True
               for row in data['vacuous'])
    before = collections.Counter((row['case_id'], row['assertion'])
                                 for row in data['non_vacuous_before'])
    after_non_vacuous = collections.Counter((row['case_id'], row['assertion'])
                                             for row in data['retained_after'] if row['vacuous'] is False)
    after_outside_scope = collections.Counter((row['case_id'], row['assertion'])
                                               for row in data['retained_after'] if row['vacuous'] is None)
    outside_scope_before = collections.Counter((row['case_id'], row['assertion'])
                                                for row in data['outside_scope'])
    assert sum(after_non_vacuous.values()) == 973
    assert sum(after_outside_scope.values()) == 221
    assert not after_non_vacuous - before
    assert not after_outside_scope - outside_scope_before

    retained = 0
    retained_non_vacuous = 0
    retained_outside_scope = 0
    retained_keys = collections.Counter()
    for item in plan['designs']:
        cid = item['case_id']
        d = ROOT / 'reduction_30' / cid
        reduction = read(d / 'summary.json')
        selected = rows(d / 'retained_assertions.jsonl')
        assert reduction['status'] == 'VALIDATED' and len(selected) == reduction['retained']
        retained += len(selected)
        retained_non_vacuous += sum(row['vacuous'] is False for row in selected)
        retained_outside_scope += sum(row['reason'] == 'outside_scope_non_implication'
                                      for row in selected)
        retained_keys.update((row['case_id'], row['assertion']) for row in selected)
        for directory, stem in (('checker_coverage_30', 'summary'),
                                ('checker_coverage_30_unreduced', 'summary'),
                                ('checker_coverage_30_comparison', 'comparison')):
            assert (ROOT / directory / cid / (stem + '.json')).exists()
    assert retained == summary['reduction']['retained_after'] == coverage['assertion_counts']['reduced'] == 1194
    assert retained_non_vacuous == 973 and retained_outside_scope == 221
    assert retained_keys == after_non_vacuous + after_outside_scope
    assert (ROOT / 'non_vacuous_30_designs_after_reduction.jsonl').read_bytes() == (ROOT / 'retained_30_designs_after_reduction.jsonl').read_bytes()
    assert coverage['assertion_counts']['unreduced'] == 5008
    for cid in plan['empty_cases']:
        assert not any(row['case_id'] == cid for row in data['dataset'])
        assert coverage['designs'][cid]['status'] == 'no_final_assertions'
    print('Verified 30 designs, 11,495 final rows, 1,194 retained assertions, and all per-design summaries')


if __name__ == '__main__':
    main()
