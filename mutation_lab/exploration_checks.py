"""Ground semantic reviews and reject provably inconsistent predicate conjunctions.

Natural-language reviews remain model judgments. These checks do not claim formal
equivalence, legal-state reachability, or independent specification correctness.
"""

from __future__ import annotations

import re
from pathlib import Path

from .core import Config
from .exploration_schema import REFERENCE, integer, text


def _reference_width(name: str, widths: dict[str, int]) -> int:
    match = REFERENCE.fullmatch(name)
    return (int(match[2]) - int(match[3] or match[2]) + 1
            if match[2] is not None else widths[match[1]])


def predicate_conflicts(items: list[dict], widths: dict[str, int], inputs: set[str]) -> list[str]:
    """Sound but incomplete unsigned, per-reference conjunction checks; never eval."""
    bounds, equal_bits, issues = {}, {}, []
    for item in items:
        left, right, op = item['left'], item['right'], item['op']
        phase = 'input' if REFERENCE.fullmatch(left)[1] in inputs else item['sample']
        if isinstance(right, str) and right.isdigit():
            right = int(right)
        if not isinstance(right, int):
            if left == right and op in ('!=', '<', '>'):
                issues.append(f'{left} {op} itself is impossible')
            continue
        width = _reference_width(left, widths)
        key = (left, phase)
        low, high, excluded = bounds.setdefault(key, [0, (1 << width) - 1, set()])
        if op == '==':
            low, high = max(low, right), min(high, right)
            match = REFERENCE.fullmatch(left)
            offset = int(match[3] or match[2] or 0)
            mask = ((1 << width) - 1) << offset
            value = (right << offset) & mask
            bits_key = (match[1], phase)
            previous_mask, previous_value = equal_bits.get(bits_key, (0, 0))
            if (previous_value ^ value) & previous_mask & mask:
                issues.append(f'Conflicting equalities on {match[1]} ({phase})')
            equal_bits[bits_key] = (previous_mask | mask, previous_value | value)
        elif op == '!=':
            excluded.add(right)
        elif op == '<':
            high = min(high, right - 1)
        elif op == '<=':
            high = min(high, right)
        elif op == '>':
            low = max(low, right + 1)
        elif op == '>=':
            low = max(low, right)
        bounds[key] = [low, high, excluded]
    for (name, phase), (low, high, excluded) in bounds.items():
        if low > high or high - low + 1 <= len(excluded) and all(x in excluded for x in range(low, high + 1)):
            issues.append(f'Unsatisfiable bounds on {name} ({phase})')
    return list(dict.fromkeys(issues))


def static_plan_checks(plan: dict, widths: dict[str, int], inputs: set[str]) -> dict:
    issues = ['trigger: ' + x for x in predicate_conflicts(plan['trigger_predicates'], widths, inputs)]
    warnings = []
    canonical = lambda p: (p['left'], p['op'], str(p['right']),
                            'input' if REFERENCE.fullmatch(p['left'])[1] in inputs else p['sample'])
    trigger = {canonical(p) for p in plan['trigger_predicates']}
    for control in plan['controls']:
        issues.extend(control['id'] + ': ' + x for x in predicate_conflicts(control['predicates'], widths, inputs))
        if trigger <= {canonical(p) for p in control['predicates']}:
            warnings.append(f"{control['id']} contains every trigger predicate; explain any different history or scope, otherwise it is not a preserved control")
    if re.sub(r'\s+', '', plan['golden_behavior']) == re.sub(r'\s+', '', plan['faulty_behavior']):
        issues.append('golden_behavior and faulty_behavior describe the same change')
    return {'status': 'failed' if issues else 'passed', 'issues': issues, 'warnings': warnings,
            'reachability_proven': False}


def source_evidence(raw, config: Config, root: Path) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError('Semantic review needs nonempty RTL source_evidence')
    result = []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError('Source evidence must be an object')
        source = Path(text(row.get('source_file'), 'source_file'))
        if source not in config.sources:
            raise ValueError('Source evidence references an unconfigured RTL file')
        lines = (root / source).read_text().splitlines()
        first = integer(row.get('start_line'), 'RTL start_line', 1, len(lines))
        last = integer(row.get('end_line'), 'RTL end_line', first, len(lines))
        result.append({'source_file': str(source), 'start_line': first, 'end_line': last,
                       'excerpt': '\n'.join(lines[first - 1:last])})
    return result


def semantic_review(raw: dict, required: list[str], config: Config, root: Path,
                    plan: dict, edit: bool = False) -> dict:
    comparison = None
    if not edit and 'golden_behavior' in required:
        if raw.get('review_subject') != 'proposed_mutation':
            raise ValueError('Plan review must evaluate a proposed_mutation, not search for an existing bug in golden RTL')
        comparison = raw.get('comparison')
        if not isinstance(comparison, dict):
            raise ValueError('Compare the unmodified golden rule with the hypothetical proposed rule before judging the plan')
        for field, plan_field in [('golden_rule', 'golden_behavior'), ('proposed_rule', 'faulty_behavior')]:
            if comparison.get(field) != plan[plan_field]:
                raise ValueError(f'comparison.{field} must copy plan.{plan_field} exactly; do not swap baseline and proposed behavior')
        failing = comparison.get('failing_case')
        cases = comparison.get('preserved_cases')
        if not isinstance(failing, dict) or type(failing.get('different')) is not bool:
            raise ValueError('Proposed-variant comparison needs failing_case with different=true/false')
        expected = {c['id'] for c in plan['controls']}
        if (not isinstance(cases, list) or len(cases) != len(expected)
                or any(not isinstance(c, dict) for c in cases)
                or {c.get('control_id') for c in cases} != expected):
            raise ValueError('Compare golden and proposed results for each preserved control exactly once')
        for case in [failing, *cases]:
            for field in ('condition', 'golden_result', 'proposed_result'):
                text(case.get(field), 'comparison.' + field)
            if case is not failing and type(case.get('equal')) is not bool:
                raise ValueError('Preserved-case comparison requires equal=true/false')
        if raw.get('verdict') == 'supported' and (not failing['different'] or not all(c['equal'] for c in cases)):
            raise ValueError('Supported plan verdict contradicts its golden-versus-proposed comparison')
    if raw.get('verdict') not in ('supported', 'mismatch', 'uncertain'):
        raise ValueError('Semantic verdict must be supported, mismatch, or uncertain')
    text(raw.get('reason'), 'semantic review reason')
    checks = raw.get('checks')
    if not isinstance(checks, dict) or set(checks) != set(required):
        raise ValueError('Semantic checks must contain exactly: ' + ', '.join(required))
    for name, row in checks.items():
        if not isinstance(row, dict) or row.get('assessment') not in ('supported', 'contradicted', 'uncertain'):
            raise ValueError(f'{name} needs assessment=supported/contradicted/uncertain')
        text(row.get('reason'), name + ' reason')
    cases = raw.get('controls')
    if not isinstance(cases, list) or {c.get('id') for c in cases if isinstance(c, dict)} != {c['id'] for c in plan['controls']} or len(cases) != len(plan['controls']):
        raise ValueError('Semantic review must assess every plan control exactly once')
    for case in cases:
        if case.get('assessment') not in ('preserved', 'contradicted', 'uncertain'):
            raise ValueError('Control assessment must be preserved, contradicted, or uncertain')
        text(case.get('reason'), 'control assessment reason')
    supported = all(x['assessment'] == 'supported' for x in checks.values()) and all(x['assessment'] == 'preserved' for x in cases)
    if raw['verdict'] == 'supported' and not supported:
        raise ValueError('Supported verdict contradicts its own checks or preserved controls')
    result = {**raw, 'source_evidence': source_evidence(raw.get('source_evidence'), config, root),
              'evidence_type': 'model_review_with_source_references', 'formal_proof': False}
    if comparison is not None:
        result['comparison_evidence'] = 'model_predicted_counterfactual; not executed simulation'
    if edit:
        result['golden_source_evidence'] = source_evidence(raw.get('golden_source_evidence'), config, config.design_root)
    return result
