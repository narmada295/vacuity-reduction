#!/usr/bin/env python3
"""Compare checker coverage of selected assertions before and after reduction."""

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path


METRICS = ('coi_statement', 'coi_branch', 'default_statement', 'default_branch',
           'proof_core_statement', 'proof_core_branch')
MAIN = ('coi_statement', 'coi_branch', 'proof_core_statement', 'proof_core_branch')


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf8')


def value(metric):
    if metric is None or not metric['total']:
        return 'N/A'
    return '{}/{} ({:.2f}%)'.format(metric['covered'], metric['total'], metric['percent'])


def delta_cell(change):
    return 'N/A' if change is None else '{:+.2f} pp'.format(change)


def compare_metric(before, after):
    if before is None or after is None:
        return {'unreduced': before, 'reduced': after, 'delta_pp': None,
                'denominators_match': None}
    comparable = before['total'] == after['total']
    delta = (round(after['percent'] - before['percent'], 2)
             if before['percent'] is not None and after['percent'] is not None else None)
    return {'unreduced': before, 'reduced': after, 'delta_pp': delta,
            'denominators_match': comparable}


def aggregate(summaries, key):
    values = [s['coverage'][key] for s in summaries if s['coverage'][key] is not None]
    covered = sum(v['covered'] for v in values)
    total = sum(v['total'] for v in values)
    return {'covered': covered, 'total': total,
            'percent': round(100 * covered / total, 2) if total else None,
            'designs_with_metric': len(values)}


def design_markdown(result):
    lines = ['# Checker coverage comparison: {}'.format(result['case_id']), '',
             '- Unreduced selected assertions: {}'.format(result['assertion_counts']['unreduced']),
             '- Reduced retained assertions: {}'.format(result['assertion_counts']['reduced']),
             '- Assertion sources: proved non-vacuous and proved non-implication assertions before reduction; validated retained assertions after reduction.',
             '- Both runs use the same source RTL: {}'.format('yes' if result['source_rtl_matches'] else 'no'), '',
             '| Metric | Unreduced | Reduced | Reduced minus unreduced | Same cover-item denominator? |',
             '|---|---:|---:|---:|:---:|']
    for name in METRICS:
        metric = result['coverage'][name]
        match = metric['denominators_match']
        lines.append('| {} | {} | {} | {} | {} |'.format(
            name, value(metric['unreduced']), value(metric['reduced']),
            delta_cell(metric['delta_pp']), 'N/A' if match is None else 'yes' if match else 'no'))
    lines += ['', '## Assertion proof results', '',
              '- Unreduced: {}'.format(result['assertion_status_counts']['unreduced']),
              '- Reduced: {}'.format(result['assertion_status_counts']['reduced']), '',
              'A percentage-point change with different denominators compares the two reported checker views; it is not an identical-item coverage delta. ',
              'N/A means a set has no cover items or no assertions.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('output/fixme_svagen/reduction_25_plan.json'))
    parser.add_argument('--reduced', type=Path, default=Path('output/fixme_svagen/checker_coverage_25'))
    parser.add_argument('--unreduced', type=Path, default=Path('output/fixme_svagen/checker_coverage_25_unreduced'))
    parser.add_argument('--out', type=Path, default=Path('output/fixme_svagen/checker_coverage_25_comparison'))
    parser.add_argument('--report-stem', default='coverage_25_comparison')
    args = parser.parse_args()
    plan = read_json(args.plan)
    designs = {}
    reduced_summaries = []
    unreduced_summaries = []
    for item in plan['designs']:
        case_id = item['case_id']
        before = read_json(args.unreduced / case_id / 'summary.json')
        after = read_json(args.reduced / case_id / 'summary.json')
        if before['status'] != 'complete' or after['status'] not in ('complete', 'no_retained_assertions'):
            raise ValueError('Incomplete coverage for {}: {}, {}'.format(case_id, before['status'], after['status']))
        if before['source_sha256'] != after['source_sha256']:
            raise ValueError('Source RTL differs for {}'.format(case_id))
        result = {
            'case_id': case_id,
            'source_rtl_matches': True,
            'assertion_counts': {'unreduced': before['assertion_count'],
                                 'reduced': after['retained_count']},
            'assertion_status_counts': {'unreduced': before['assertion_status_counts'],
                                        'reduced': after['assertion_status_counts']},
            'coverage': {name: compare_metric(before['coverage'][name], after['coverage'][name])
                         for name in METRICS},
        }
        designs[case_id] = result
        reduced_summaries.append(after)
        unreduced_summaries.append(before)
        destination = args.out / case_id
        write_json(destination / 'comparison.json', result)
        (destination / 'comparison.md').write_text(design_markdown(result), encoding='utf8')
    metrics = {name: compare_metric(aggregate(unreduced_summaries, name),
                                    aggregate(reduced_summaries, name)) for name in METRICS}
    for name in METRICS:
        metrics[name]['denominators_match'] = (
            metrics[name]['unreduced']['total'] == metrics[name]['reduced']['total']
            and all(result['coverage'][name]['denominators_match'] is not False
                    for result in designs.values()))
    totals = {'unreduced': sum(s['assertion_count'] for s in unreduced_summaries),
              'reduced': sum(s['retained_count'] for s in reduced_summaries)}
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'design_count': len(designs), 'assertion_counts': totals, 'coverage': metrics,
               'designs': designs,
               'assertion_status_counts': {
                   'unreduced': dict(sum((collections.Counter(s['assertion_status_counts']) for s in unreduced_summaries), collections.Counter())),
                   'reduced': dict(sum((collections.Counter(s['assertion_status_counts']) for s in reduced_summaries), collections.Counter()))}}
    write_json(args.out / (args.report_stem + '.json'), summary)
    lines = ['# Checker coverage: unreduced versus reduced assertions for {} FIXME designs'.format(len(designs)), '',
             'Unreduced = all selected proved non-vacuous and proved non-implication assertions before exact deduplication and formal pruning. '
             'Reduced = validated retained assertions after reduction. Both sets use the same RTL and JasperGold checker-coverage procedure.', '',
             '- Designs: {}'.format(len(designs)),
             '- Assertion counts: {} unreduced, {} reduced'.format(totals['unreduced'], totals['reduced']),
             '- Proof results: {} unreduced; {} reduced'.format(summary['assertion_status_counts']['unreduced'], summary['assertion_status_counts']['reduced']), '',
             '## Item-weighted aggregate', '',
             '| Metric | Unreduced | Reduced | Reduced minus unreduced | Same cover-item denominator? |',
             '|---|---:|---:|---:|:---:|']
    for name in METRICS:
        metric = metrics[name]
        lines.append('| {} | {} | {} | {} | {} |'.format(
            name, value(metric['unreduced']), value(metric['reduced']),
            delta_cell(metric['delta_pp']), 'yes' if metric['denominators_match'] else 'no'))
    lines += ['', '## Per design', '',
              '| Design | Unreduced assertions | Reduced assertions | Unreduced COI statement | Reduced COI statement | Unreduced proof statement | Reduced proof statement |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for case_id, result in designs.items():
        cov = result['coverage']
        lines.append('| [{}]({}/comparison.md) | {} | {} | {} | {} | {} | {} |'.format(
            case_id, case_id, result['assertion_counts']['unreduced'], result['assertion_counts']['reduced'],
            value(cov['coi_statement']['unreduced']), value(cov['coi_statement']['reduced']),
            value(cov['proof_core_statement']['unreduced']), value(cov['proof_core_statement']['reduced'])))
    lines += ['', 'Each per-design comparison reports all six coverage metrics, numerator/denominator counts, and percentage-point changes. '
              'Different cover-item denominators mean the change compares distinct checker views. N/A denotes undefined coverage for an empty set.', '']
    (args.out / (args.report_stem + '.md')).write_text('\n'.join(lines), encoding='utf8')
    print('Compared {} designs: {} unreduced -> {} reduced assertions'.format(
        len(designs), totals['unreduced'], totals['reduced']))


if __name__ == '__main__':
    main()
