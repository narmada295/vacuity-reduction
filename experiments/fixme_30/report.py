#!/usr/bin/env python3
"""Generate a detailed, reproducible report for the fixed 30-design snapshot."""

import collections
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('output/fixme_svagen')
METRICS = ('coi_statement', 'coi_branch', 'default_statement', 'default_branch',
           'proof_core_statement', 'proof_core_branch')
REASONS = ('antecedent_never_true', 'consequent_always_true',
           'antecedent_never_true; consequent_always_true')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def fmt(metric):
    if metric is None or not metric['total']:
        return 'N/A'
    return '{}/{} ({:.2f}%)'.format(metric['covered'], metric['total'], metric['percent'])


def pair(metric):
    if metric is None:
        return 'N/A'
    change = metric['delta_pp']
    return '{} → {}{}'.format(
        fmt(metric['unreduced']), fmt(metric['reduced']),
        '' if change is None else ' ({:+.2f} pp)'.format(change))


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else 0.0


def make_row(cid, cohort, rows, comparison):
    case = read(ROOT / 'cases' / cid / 'case.json')
    statuses = collections.Counter(row['assertion_status'] for row in rows)
    labels = collections.Counter((row['assertion_status'], row['vacuous'], row['reason'])
                                 for row in rows)
    reasons = {reason: labels[('proven', True, reason)] for reason in REASONS}
    result = {
        'case_id': cid, 'level': cid.split('/')[0], 'status': case['status'],
        'signal_count': case['signal_count'], 'processed_signals': case['processed_signals'],
        'final_assertions': len(rows), 'proven': statuses['proven'],
        'counterexamples': statuses['cex'], 'errors': statuses['error'],
        'proved_non_vacuous': labels[('proven', False, 'neither_targeted_vacuity_reason')],
        'proved_vacuous': sum(reasons.values()), 'vacuity_reasons': reasons,
        'proved_outside_scope_non_implication': labels[('proven', None, 'outside_scope_non_implication')],
        'proved_vacuity_inconclusive': labels[('proven', None, 'related_cover_inconclusive_or_unsupported')],
        'failure_reason': case.get('failure_reason'),
    }
    if case['status'] == 'no_final_assertions':
        result['reduction'] = None
        result['coverage'] = {name: None for name in METRICS}
        return result
    reduction = read(ROOT / 'reduction_30' / cid / 'summary.json')
    retained = [json.loads(line) for line in (ROOT / 'reduction_30' / cid / 'retained_assertions.jsonl').read_text().splitlines()
                if line.strip()]
    if reduction['status'] != 'VALIDATED' or len(retained) != reduction['retained']:
        raise ValueError('Unvalidated reduction for ' + cid)
    result['reduction'] = {
        'selected': reduction['selected_proven_nonvacuous_or_out_of_scope'],
        'exact_duplicate_copies_removed': reduction['exact_duplicate_copies_removed'],
        'unique_after_deduplication': reduction['unique_after_deduplication'],
        'formal_removed': reduction['formal_removed'],
        'retained': reduction['retained'],
        'retained_non_vacuous': sum(row['vacuous'] is False for row in retained),
        'retained_outside_scope': sum(row['reason'] == 'outside_scope_non_implication' for row in retained),
        'retained_liveness': reduction['retained_liveness'],
    }
    comp = comparison['designs'][cid]
    result['coverage'] = comp['coverage']
    result['checker_assertions'] = comp['assertion_counts']
    result['checker_assertion_status_counts'] = comp['assertion_status_counts']
    preconditions = {}
    for label, path in (('before', ROOT / 'checker_coverage_30_unreduced' / cid / 'summary.json'),
                        ('after', ROOT / 'checker_coverage_30' / cid / 'summary.json')):
        summary = read(path)
        preconditions[label] = summary['precondition_status_counts']
    result['checker_precondition_status_counts'] = preconditions
    assert result['reduction']['selected'] == result['checker_assertions']['unreduced']
    assert result['reduction']['retained'] == result['checker_assertions']['reduced']
    assert result['reduction']['selected'] == result['proved_non_vacuous'] + result['proved_outside_scope_non_implication']
    assert result['reduction']['selected'] - result['reduction']['exact_duplicate_copies_removed'] == result['reduction']['unique_after_deduplication']
    assert result['reduction']['unique_after_deduplication'] - result['reduction']['formal_removed'] == result['reduction']['retained']
    return result


def main():
    cohort = read(ROOT / 'summary_30_designs.json')
    comparison = read(ROOT / 'checker_coverage_30_comparison/coverage_30_comparison.json')
    by_case = collections.defaultdict(list)
    with (ROOT / 'dataset_30_designs.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                by_case[row['case_id']].append(row)
    details = {cid: make_row(cid, cohort, by_case[cid], comparison) for cid in cohort['case_ids']}
    values = list(details.values())
    done = [row for row in values if row['reduction'] is not None]
    total = lambda field: sum(row[field] for row in values)
    reduction_total = lambda field: sum(row['reduction'][field] for row in done)
    reasons = {reason: sum(row['vacuity_reasons'][reason] for row in values) for reason in REASONS}
    precondition = {}
    for stage in ('before', 'after'):
        precondition[stage] = dict(sum((collections.Counter(row['checker_precondition_status_counts'][stage])
                                        for row in done), collections.Counter()))
    aggregate = {
        'designs': len(values), 'designs_with_final_assertions': len(done),
        'signals': total('signal_count'), 'processed_signals': total('processed_signals'),
        'final_assertions': total('final_assertions'), 'proven': total('proven'),
        'counterexamples': total('counterexamples'), 'errors': total('errors'),
        'proved_non_vacuous': total('proved_non_vacuous'),
        'proved_vacuous': total('proved_vacuous'), 'vacuity_reasons': reasons,
        'proved_outside_scope_non_implication': total('proved_outside_scope_non_implication'),
        'proved_vacuity_inconclusive': total('proved_vacuity_inconclusive'),
        'reduction': {field: reduction_total(field) for field in
                      ('selected', 'exact_duplicate_copies_removed', 'unique_after_deduplication',
                       'formal_removed', 'retained', 'retained_non_vacuous',
                       'retained_outside_scope', 'retained_liveness')},
        'checker_precondition_status_counts': precondition,
        'coverage': comparison['coverage'],
    }
    assert aggregate['final_assertions'] == 11495
    assert aggregate['proven'] == sum(aggregate[key] for key in
                                      ('proved_non_vacuous', 'proved_vacuous',
                                       'proved_outside_scope_non_implication',
                                       'proved_vacuity_inconclusive'))
    assert aggregate['reduction']['selected'] == comparison['assertion_counts']['unreduced'] == 5008
    assert aggregate['reduction']['retained'] == comparison['assertion_counts']['reduced'] == 1194
    assert cohort['files']['retained_after']['rows'] == aggregate['reduction']['retained'] == 1194
    assert cohort['files']['retained_after']['label_counts'] == {
        'proved_non_vacuous': aggregate['reduction']['retained_non_vacuous'],
        'proved_outside_scope_non_implication': aggregate['reduction']['retained_outside_scope']}

    level_summary = {}
    for level in ('L1', 'L2', 'L3'):
        subset = [row for row in values if row['level'] == level]
        level_summary[level] = {
            'designs': len(subset), 'with_assertions': sum(row['status'] == 'done' for row in subset),
            'signals': sum(row['signal_count'] for row in subset),
            'final_assertions': sum(row['final_assertions'] for row in subset),
            'proven': sum(row['proven'] for row in subset),
            'counterexamples': sum(row['counterexamples'] for row in subset),
            'errors': sum(row['errors'] for row in subset),
            'proved_non_vacuous': sum(row['proved_non_vacuous'] for row in subset),
            'proved_vacuous': sum(row['proved_vacuous'] for row in subset),
            'selected': sum(row['reduction']['selected'] for row in subset if row['reduction']),
            'retained': sum(row['reduction']['retained'] for row in subset if row['reduction']),
        }

    report = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'source_dataset': str(ROOT / 'dataset_30_designs.jsonl'),
              'source_coverage_comparison': str(ROOT / 'checker_coverage_30_comparison/coverage_30_comparison.json'),
              'aggregate': aggregate, 'levels': level_summary, 'designs': details}
    (ROOT / 'results_30_detailed.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf8')

    flat = []
    for row in values:
        record = {key: value for key, value in row.items()
                  if key not in ('vacuity_reasons', 'reduction', 'coverage', 'checker_assertions',
                                 'checker_assertion_status_counts', 'checker_precondition_status_counts')}
        for key, value in row['vacuity_reasons'].items():
            record['vacuity_' + key.replace('; ', '_and_')] = value
        for key, value in (row['reduction'] or {}).items():
            record['reduction_' + key] = value
        for name in METRICS:
            item = row['coverage'][name]
            for stage, source in (('before', item['unreduced'] if item else None),
                                  ('after', item['reduced'] if item else None)):
                for field in ('covered', 'total', 'percent'):
                    record[name + '_' + stage + '_' + field] = None if source is None else source[field]
            record[name + '_delta_pp'] = None if item is None else item['delta_pp']
        flat.append(record)
    fields = list(dict.fromkeys(key for record in flat for key in record))
    with (ROOT / 'results_30_per_design.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)

    lines = ['# FIXME SVAGen Text: fixed 30-design results', '',
             'All figures below are from the fixed 30-design snapshot. The ongoing full-benchmark run is separate.', '',
             '## Experiment and outcome', '',
             '- SANGAM model: `deepseek-ai/DeepSeek-V4-Flash-0731` through Nebius for all final rows.',
             '- 4 refinement iterations and 180-second formal timeout per design; assertion generation used unconstrained reset.',
             '- Designs: 30; 28 produced final assertions and 2 produced none.',
             '- Signals processed: {}/{}.'.format(aggregate['processed_signals'], aggregate['signals']),
             '- Final assertions: {:,}; JasperGold proved {:,} ({:.2f}%), found counterexamples for {:,} ({:.2f}%), and returned errors for {:,} ({:.2f}%).'.format(
                 aggregate['final_assertions'], aggregate['proven'], pct(aggregate['proven'], aggregate['final_assertions']),
                 aggregate['counterexamples'], pct(aggregate['counterexamples'], aggregate['final_assertions']),
                 aggregate['errors'], pct(aggregate['errors'], aggregate['final_assertions'])), '',
             '### Proved assertions by vacuity label', '',
             '| Label | Count |', '|---|---:|',
             '| Non-vacuous implication under the two targeted checks | {:,} |'.format(aggregate['proved_non_vacuous']),
             '| Vacuous implication | {:,} |'.format(aggregate['proved_vacuous']),
             '| Proved non-implication, outside vacuity scope | {:,} |'.format(aggregate['proved_outside_scope_non_implication']),
             '| Related-cover check inconclusive or unsupported | {:,} |'.format(aggregate['proved_vacuity_inconclusive']),
             '| **Total proved** | **{:,}** |'.format(aggregate['proven']), '',
             'Vacuous reasons: antecedent never true = {}, consequent always true = {}, both = {}. '
             'Among proved implications with a definite targeted-vacuity decision, {:.2f}% ({}/{}) were vacuous.'.format(
                 reasons[REASONS[0]], reasons[REASONS[1]], reasons[REASONS[2]],
                 pct(aggregate['proved_vacuous'], aggregate['proved_vacuous'] + aggregate['proved_non_vacuous']),
                 aggregate['proved_vacuous'], aggregate['proved_vacuous'] + aggregate['proved_non_vacuous']), '',
             '## Reduction', '',
             '| Stage | Assertions | Change from preceding stage |', '|---|---:|---:|',
             '| Selected proved non-vacuous + proved non-implication | {:,} | — |'.format(aggregate['reduction']['selected']),
             '| Unique after exact deduplication | {:,} | −{:,} duplicate copies |'.format(
                 aggregate['reduction']['unique_after_deduplication'], aggregate['reduction']['exact_duplicate_copies_removed']),
             '| Validated after formal pruning | {:,} | −{:,} |'.format(
                 aggregate['reduction']['retained'], aggregate['reduction']['formal_removed']), '',
             'The retained set is {:.2f}% of the selected set. It contains {:,} proved non-vacuous assertions and {:,} proved non-implication assertions outside vacuity scope; {} retained assertions are marked liveness. '
             'The 1,194-row after-reduction dataset includes both classes. Its non-vacuous component is {:.2f}% of the {:,} non-vacuous assertions before reduction.'.format(
                 pct(aggregate['reduction']['retained'], aggregate['reduction']['selected']),
                 aggregate['reduction']['retained_non_vacuous'], aggregate['reduction']['retained_outside_scope'],
                 aggregate['reduction']['retained_liveness'],
                 pct(aggregate['reduction']['retained_non_vacuous'], aggregate['proved_non_vacuous']),
                 aggregate['proved_non_vacuous']), '',
             'The independent checker-coverage runs proved all {:,} selected assertions before reduction and all {:,} retained assertions after reduction. '
             'Checker preconditions were covered/unreachable: {}/{} before and {}/{} after. The three unreachable sub-preconditions all occur in `L3/12`; these checker results are separate from the targeted vacuity labels.'.format(
                 aggregate['reduction']['selected'], aggregate['reduction']['retained'],
                 precondition['before'].get('covered', 0), precondition['before'].get('unreachable', 0),
                 precondition['after'].get('covered', 0), precondition['after'].get('unreachable', 0)), '',
             '## Item-weighted checker coverage comparison', '',
             'The aggregate counts sum JasperGold covered and total items across designs. Checker runs include 28 designs before reduction and 27 after reduction; `L1/test_endian` retains zero assertions. '
             'A metric is measurable only when JasperGold reports at least one cover item: 24 designs for statement metrics and 26 for branch metrics in either set. '
             'The two designs with no final assertions have undefined coverage. Checker runs used reset policy `none`.', '',
             '| Metric | Before reduction | After reduction | Change | Same denominator? |',
             '|---|---:|---:|---:|:---:|']
    for name in METRICS:
        m = comparison['coverage'][name]
        lines.append('| {} | {} | {} | {:+.2f} pp | {} |'.format(
            name, fmt(m['unreduced']), fmt(m['reduced']), m['delta_pp'],
            'yes' if m['denominators_match'] else 'no'))
    lines += ['', '### Where coverage changed', '']
    for name in METRICS:
        measurable = [(row['case_id'], row['coverage'][name]['delta_pp']) for row in values
                      if row['coverage'][name] is not None and row['coverage'][name]['delta_pp'] is not None]
        same = sum(delta == 0 for _, delta in measurable)
        changed = ', '.join('`{}` {:+.2f} pp'.format(cid, delta)
                            for cid, delta in measurable if delta != 0)
        lines.append('- `{}`: unchanged for {}/{} measurable designs; {}.'.format(
            name, same, len(measurable), changed or 'no changes'))
    lines += ['', '## By benchmark level', '',
              '| Level | Designs | With assertions | Signals | Final | Proved | Counterexamples | Errors | Non-vacuous | Vacuous | Selected | Retained |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for level, x in level_summary.items():
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            level, x['designs'], x['with_assertions'], x['signals'], x['final_assertions'],
            x['proven'], x['counterexamples'], x['errors'], x['proved_non_vacuous'],
            x['proved_vacuous'], x['selected'], x['retained']))
    lines += ['', '## Per-design assertion outcomes', '',
              '| Design | Signals | Final | Proved | CEX | Error | Non-vacuous | Vacuous | Outside scope | Inconclusive |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in values:
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            row['case_id'], row['signal_count'], row['final_assertions'], row['proven'],
            row['counterexamples'], row['errors'], row['proved_non_vacuous'], row['proved_vacuous'],
            row['proved_outside_scope_non_implication'], row['proved_vacuity_inconclusive']))
    lines += ['', '### Per-design vacuity reasons', '',
              '| Design | Antecedent never true only | Consequent always true only | Both |',
              '|---|---:|---:|---:|']
    for row in values:
        rs = row['vacuity_reasons']
        lines.append('| {} | {} | {} | {} |'.format(row['case_id'], *(rs[name] for name in REASONS)))
    lines += ['', '## Per-design reduction', '',
              '| Design | Selected | Duplicate copies removed | Unique | Formal removed | Retained | Retained non-vacuous | Retained outside scope | Liveness |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in values:
        d = row['reduction']
        if d is None:
            lines.append('| {} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |'.format(row['case_id']))
        else:
            lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                row['case_id'], d['selected'], d['exact_duplicate_copies_removed'],
                d['unique_after_deduplication'], d['formal_removed'], d['retained'],
                d['retained_non_vacuous'], d['retained_outside_scope'], d['retained_liveness']))
    lines += ['', '## Per-design checker coverage', '',
              'Each cell shows covered/total items and percentage before → after reduction, followed by the percentage-point change. `N/A` means coverage is undefined for that assertion set.', '']
    for row in values:
        lines += ['### {}'.format(row['case_id']), '']
        if row['failure_reason']:
            lines += ['No final assertions: {}.'.format(row['failure_reason']), '']
            continue
        d = row['reduction']
        lines += ['Selected {} → retained {} assertions. Checker proof results: {} before, {} after. Checker preconditions: {} before, {} after.'.format(
            d['selected'], d['retained'], row['checker_assertion_status_counts']['unreduced'],
            row['checker_assertion_status_counts']['reduced'],
            row['checker_precondition_status_counts']['before'], row['checker_precondition_status_counts']['after']), '',
            '| Metric | Before → after | Same denominator? |', '|---|---:|:---:|']
        for name in METRICS:
            m = row['coverage'][name]
            same = m['denominators_match']
            lines.append('| {} | {} | {} |'.format(name, pair(m),
                         'N/A' if same is None else 'yes' if same else 'no'))
        lines.append('')
    lines += ['## Reading the labels', '',
              '- `Non-vacuous` and `vacuous` apply only to proved implications under the two targeted checks: unreachable antecedent and always-true consequent.',
              '- `Outside scope` means a proved non-implication. `Inconclusive` means the related-cover check did not resolve the targeted classification.',
              '- `CEX` is an assertion falsified by JasperGold; it is excluded from the reduction selection. `Error` is a formal error or inconclusive assertion result.',
              '- Coverage percentages are JasperGold checker-coverage item totals, not the fraction of assertions proved. All assertions in the two checker runs proved.',
              '- Per-design source files: `cases/<level>/<design>/case.json`, `reduction_30/<level>/<design>/summary.json`, `checker_coverage_30*/<level>/<design>/summary.json`.',
              '- The flat machine-readable table is `results_30_per_design.csv`; the complete structured data is `results_30_detailed.json`.', '']
    (ROOT / 'results_30_detailed.md').write_text('\n'.join(lines), encoding='utf8')
    print('Wrote detailed 30-design report, JSON, and CSV')


if __name__ == '__main__':
    main()
