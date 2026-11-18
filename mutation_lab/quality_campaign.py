"""Specification-guided, feedback-refined RTL mutant generation.

The evaluation assertion set is deliberately absent from this module. Model
quality ratings are review aids, not proof that a mutant violates the spec.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .campaign import direct_config
from .core import (Config, Edit, apply_edits, digest, read_spec,
                   rtl_fingerprint, write_json)
from .formal import lint
from .planner import LLMPlanner
from .quality_validation import BehaviorAuditor, _reset_active


CATEGORIES = ("simple", "timing", "state_sequence", "interaction")
SCORE_FIELDS = ("spec_relevance", "realism", "selectivity", "plan_edit_match",
                "diversity")


@dataclass(frozen=True)
class Behavior:
    id: str
    spec_id: str | None
    spec_quote: str
    intended_behavior: str
    related_signals: tuple[str, ...]


@dataclass(frozen=True)
class FaultPlan:
    requirement_id: str
    category: str
    trigger: str
    golden_behavior: str
    faulty_behavior: str
    root_cause: str
    target_signal: str
    why_realistic: str
    trigger_predicates: tuple[dict[str, str], ...] = ()
    control_predicates: tuple[tuple[dict[str, str], ...], ...] = ()
    preserve_first_operation: bool = False
    trigger_window_cycles: int = 0


MAP_SYSTEM = """Select concrete functional behaviors from the numbered specification passages.
Return one JSON object: {"requirements": [{"spec_id": "S001",
"intended_behavior": "...", "related_signals": ["..."]}]}.
Select a spec_id from the supplied PASSAGES; do not write or paraphrase a quote.
The controller copies the passage text verbatim into the saved requirement.
related_signals must name identifiers in the supplied RTL. Cover different
behaviors and interactions, including repeated events and operation over
multiple cycles when supported. Do not use assertions or invent requirements.
Return at most the requested number.
"""

PLAN_SYSTEM = """Plan ONE plausible functional RTL fault for the supplied requirement.
Return one JSON object with requirement_id, category, trigger,
golden_behavior, faulty_behavior, root_cause, target_signal, why_realistic
(all strings), trigger_predicates (a list), and preserve_first_operation
(a boolean), trigger_window_cycles (an integer 0-8), and
control_predicates (a list of 1-2 lists of predicates). Each inner list is a
conjunction describing a case where the target behavior must still match the
golden RTL. Controls must be distinct from the trigger. For state_sequence,
give at least one control for the correct first operation. For interaction,
give two controls that exercise each condition separately.
Each trigger_predicates item is {"left": "signal_or_bit",
"op": "==", "right": "signal_or_integer"}; use ==, !=, <, <=, >, or >=.
List every condition needed for the fault to activate, including important
state and input conditions. Bit selects such as data[3] are allowed.
preserve_first_operation is true only when the first ordinary transaction
must work before a later transaction fails.
trigger_window_cycles is 0 when the first state difference occurs on the
triggering edge. Use a positive value only when the fault manifests that many
cycles after the trigger, and explain the delayed causal path.
Use the requested category and a target_signal present in RTL. State the
specific causal RTL mistake and the condition that activates it. Preserve
ordinary behavior where practical. The faulty behavior must contradict the
quoted requirement. Avoid constant-output failures, arbitrary extra delays,
and repeats of prior plans. For state_sequence, require a correct first
transaction followed by a fault on a later transaction or state transition;
set preserve_first_operation=true. For interaction, name the two conditions
that must coincide, and explain what happens when each occurs alone. Merely
being out of reset does not count as an independent interaction condition. A fault
on the first simultaneous event is allowed if the isolated controls work.
Keep proposed RTL code out of the trigger field so it describes the event.
Use an existing state signal where possible. If new history state is essential,
its implementation must include a declaration, reset, and update.
Do not propose an assertion or predict exact cycle-by-cycle values. Do not include RTL edits yet. If this category is not
supported, return {"skip_reason": "specific reason"}.
"""

EDIT_SYSTEM = """Implement exactly the planned RTL fault in the golden sources.
Return JSON with edits, a list of 1-3 objects, each with source_file, old, new.
Each old must be an exact unique substring of its configured RTL source; the
controller replaces it once in an isolated copy. Related edits may jointly
implement one fault. Keep module ports, clock/reset wiring, and unrelated
behavior intact, especially reset priority and any first-operation behavior
the plan explicitly preserves. Prefer modifying existing update logic. Do not edit
assertions, add an arbitrary delay, or force an output constant. The plan is a
hypothesis; do not invent measured behavior. Use the supplied tool/critic
feedback to repair the same fault when present.
"""

CRITIC_SYSTEM = """Review an RTL mutation against its quoted specification
and fault plan. Return JSON with verdict (accept, revise, or reject), scores
(object with integer 1-5 fields spec_relevance, realism, selectivity,
plan_edit_match, diversity), issues (list of specific strings), and reason
(short string).
Assess the actual old/new RTL edits, not the author's claims. Penalize faults
that break normal behavior broadly, rely on artificial new delay/history
logic without a plausible cause, or fail to implement the plan. A first
operation may fail if the plan explicitly targets that operation. A small
syntactically valid edit is not automatically realistic. For diversity, compare
the root cause and failure behavior with prior accepted fault plans, not just
the syntax. Do not see or infer
the held-out evaluation assertions. Scores are review judgments, not proofs.
The measured behavior audit comes from executing golden and mutant RTL on the
same finite input probes. Use its first differences and control results when
judging selectivity and plan/edit consistency. Do not claim an edit has a
syntax error when the controller reports that lint passed. The audit is
sampled evidence, not a specification proof. Read the complete measured trace
excerpt before describing the failure. Do not claim the first event fails if
the first measured difference is on a later event. Include an
observed_behavior string in your JSON summarizing the measured trajectory.
Copy the audit's witness_check object exactly as measured_first_difference.
The controller checks this against the simulation before accepting the review.
"""


def _normalized(value: str) -> str:
    return " ".join(value.split())


def _spec_passages(specification: str) -> dict[str, str]:
    """Number source-owned spec passages so the model never has to quote text."""
    passages: dict[str, str] = {}
    for block in re.split(r"\n\s*\n", specification):
        lines = [line for line in block.splitlines()
                 if not re.match(r"^\s*#{1,6}\s", line)]
        # Markdown lists often contain several independent requirements in one
        # blank-line block. Keep each item intact instead of cutting mid-item.
        chunks: list[str] = []
        current: list[str] = []
        for line in lines:
            if re.match(r"^\s*(?:[-*+]\s|\d+[.)]\s)", line) and current:
                chunks.append("\n".join(current).strip())
                current = []
            current.append(line)
        if current:
            chunks.append("\n".join(current).strip())
        for chunk in chunks:
            if len(chunk) < 12:
                continue
            while chunk:
                if len(chunk) <= 500:
                    part, chunk = chunk, ""
                else:
                    # Prefer a sentence boundary; fall back to a word boundary.
                    ends = [match.end() for match in re.finditer(r"[.!?](?=\s)", chunk[:500])]
                    cut = max((end for end in ends if end >= 100), default=0)
                    if not cut:
                        cut = chunk.rfind(" ", 0, 501)
                    cut = cut if cut >= 100 else 500
                    part, chunk = chunk[:cut].strip(), chunk[cut:].strip()
                if len(part) >= 12:
                    passages[f"S{len(passages) + 1:03d}"] = part
    if not passages:
        raise ValueError("Specification has no text passages of at least 12 characters")
    return passages


def _fallback_behaviors(config: Config, specification: str,
                        identifiers: set[str], maximum: int) -> list[Behavior]:
    """Ground a small requirement set directly in spec passages after model failure."""
    reserved = {"module", "endmodule", "input", "output", "inout", "wire", "reg",
                "logic", "always", "always_ff", "begin", "end", "if", "else",
                "posedge", "negedge", "assign", "parameter", config.top}
    signals = identifiers - reserved
    cues = ("when", "whenever", "after", "until", "must", "reset", "write",
            "update", "increment", "toggle", "hold", "assert", "disable", "enable")
    candidates: list[tuple[int, int, str, str, tuple[str, ...]]] = []
    for position, (spec_id, passage) in enumerate(_spec_passages(specification).items()):
        mentioned = tuple(signal for signal in sorted(signals)
                          if re.search(r"\b" + re.escape(signal) + r"\b", passage))
        if not mentioned:
            continue
        lower = passage.lower()
        score = sum(cue in lower for cue in cues) * 2 + min(len(mentioned), 5)
        candidates.append((score, -position, spec_id, passage, mentioned))
    candidates.sort(reverse=True)
    if not candidates:
        raise ValueError("No specification passage names an RTL identifier for fallback mapping")
    return [Behavior(f"R{index + 1:03d}", spec_id, passage, passage,
                     mentioned[:8])
            for index, (_, _, spec_id, passage, mentioned) in
            enumerate(candidates[:maximum])]


def _short(value: Any, limit: int = 1200) -> str:
    return str(value)[:limit]


def _model_request(client: LLMPlanner, system: str, prompt: str, limit: int,
                   array_key: str | None = None) -> dict:
    if len(prompt) > limit:
        raise ValueError(f"LLM context has {len(prompt)} characters; limit is {limit}. "
                         "Use --focus-rtl or raise --max-context-chars")
    return (client._request(system, prompt, array_key=array_key) if array_key
            else client._request(system, prompt))


def _source_context(config: Config) -> tuple[str, set[str]]:
    focus = config.planner.get("context_sources", [str(path) for path in config.sources])
    if not isinstance(focus, list) or not focus:
        raise ValueError("Planner context needs at least one RTL source")
    parts: list[str] = []
    identifiers: set[str] = set()
    for name in focus:
        path = Path(name)
        if path not in config.sources:
            raise ValueError(f"Focus source is not configured RTL: {name}")
        source = (config.design_root / path).read_text()
        parts.append(f"RTL FILE {name}:\n{source}")
        identifiers.update(re.findall(r"\b[A-Za-z_]\w*\b", source))
    return "\n\n".join(parts), identifiers


def _map_behaviors(client: LLMPlanner, config: Config, specification: str,
                   rtl: str, identifiers: set[str], maximum: int,
                   context_limit: int) -> list[Behavior]:
    passages = _spec_passages(specification)
    prompt = (f"MAX REQUIREMENTS: {maximum}\n\nPASSAGES:\n" +
              json.dumps(passages, ensure_ascii=False) +
              f"\n\nGOLDEN RTL:\n{rtl}")
    feedback = ""
    for retry in range(3):
        try:
            response = _model_request(client, MAP_SYSTEM, prompt + feedback,
                                      context_limit, array_key="requirements")
            raw = response.get("requirements")
            if not isinstance(raw, list) or not raw:
                raise ValueError("requirements must be a nonempty list")
            behaviors: list[Behavior] = []
            seen: set[str] = set()
            rejected: list[str] = []
            for item in raw:
                if not isinstance(item, dict):
                    rejected.append("non-object requirement")
                    continue
                spec_id = item.get("spec_id")
                if isinstance(spec_id, str):
                    match = re.fullmatch(r"S0*(\d+)", spec_id.strip(), flags=re.IGNORECASE)
                    if match:
                        spec_id = f"S{int(match.group(1)):03d}"
                quote = passages.get(spec_id) if isinstance(spec_id, str) else None
                if quote is None:
                    # Accept exact legacy quotes, then anchor them to a source passage.
                    proposed = item.get("spec_quote")
                    if isinstance(proposed, str) and 12 <= len(proposed) <= 500:
                        match = next(((key, text) for key, text in passages.items()
                                      if _normalized(proposed) in _normalized(text)), None)
                        if match:
                            spec_id, quote = match
                if quote is None:
                    rejected.append("unknown spec_id or ungrounded spec_quote")
                    continue
                intended, signals = item.get("intended_behavior"), item.get("related_signals")
                if not isinstance(intended, str) or len(intended.strip()) < 12:
                    rejected.append("intended_behavior needs a concrete description")
                    continue
                if not isinstance(signals, list):
                    rejected.append("related_signals must be a list")
                    continue
                valid_signals = [name for signal in signals if isinstance(signal, str)
                                 for name in [signal.strip(" `").split("[", 1)[0]]
                                 if name in identifiers]
                if not valid_signals:
                    rejected.append("related_signals has no RTL identifiers")
                    continue
                key = str(spec_id) + ":" + _normalized(intended).casefold()
                if key in seen:
                    rejected.append("duplicate requirement behavior")
                    continue
                seen.add(key)
                behaviors.append(Behavior(f"R{len(behaviors) + 1:03d}", spec_id, quote,
                                          intended.strip(), tuple(dict.fromkeys(valid_signals))))
                if len(behaviors) >= maximum:
                    break
            if behaviors:
                return behaviors
            raise ValueError("No grounded requirements in model reply: " + "; ".join(rejected[:5]))
        except ValueError as exc:
            if retry == 2:
                raise
            feedback = "\n\nVALIDATION ERROR IN PREVIOUS RESPONSE: " + str(exc) + "\nReturn corrected JSON."
    raise AssertionError("unreachable")


def _plan(client: LLMPlanner, behavior: Behavior, category: str,
          rtl: str, identifiers: set[str], history: list[dict],
          context_limit: int, extra_feedback: str = "",
          reset: str | None = None) -> FaultPlan | str:
    avoided = list(dict.fromkeys(
        f"{item['plan']['target_signal']}: {item['plan']['root_cause'][:220]}"
        for item in history if isinstance(item.get("plan"), dict)
        and item["plan"].get("target_signal") and item["plan"].get("root_cause")))
    prompt = ("REQUIREMENT:\n" + json.dumps(asdict(behavior)) +
              "\n\nREQUESTED CATEGORY: " + category +
              "\n\nDO NOT REPEAT THESE FAULT MECHANISMS:\n" +
              json.dumps(avoided[-24:]) +
              "\n\nPRIOR PLANS AND OUTCOMES:\n" + json.dumps(history[-12:]) +
              "\n\nFEEDBACK:\n" + extra_feedback +
              "\n\nGOLDEN RTL:\n" + rtl)
    response = _model_request(client, PLAN_SYSTEM, prompt, context_limit)
    if "skip_reason" in response:
        reason = response["skip_reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("skip_reason must explain why the category does not apply")
        return reason.strip()
    fields = ("requirement_id", "category", "trigger", "golden_behavior",
              "faulty_behavior", "root_cause", "target_signal", "why_realistic")
    if any(not isinstance(response.get(name), str) or not response[name].strip()
           for name in fields):
        raise ValueError("Fault plan needs: " + ", ".join(fields))
    def parse_predicates(raw: Any, label: str) -> tuple[dict[str, str], ...]:
        if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
            raise ValueError(f"{label} needs one to eight predicates")
        parsed = []
        operand = re.compile(r"([A-Za-z_]\w*)(?:\[\d+\])?\Z")
        for item in raw:
            if not isinstance(item, dict) or set(item) != {"left", "op", "right"}:
                raise ValueError(f"Each {label} predicate needs left, op, right")
            left, op, right = item["left"], item["op"], item["right"]
            # JSON numbers and quoted decimal literals denote the same
            # predicate. Keep one canonical representation for the auditor.
            if type(right) is int and right >= 0:
                right = str(right)
            if not all(isinstance(value, str) for value in (left, op, right)) or op not in (
                    "==", "!=", ">=", "<=", ">", "<"):
                raise ValueError(f"{label} predicate operands/op are invalid")
            left_match, right_match = operand.fullmatch(left), operand.fullmatch(right)
            if (not left_match or left_match.group(1) not in identifiers or
                    not (right.isdigit() or right_match and right_match.group(1) in identifiers)):
                raise ValueError(f"{label} predicate must reference RTL identifiers or an integer")
            parsed.append({"left": left, "op": op, "right": right})
        return tuple(parsed)

    raw_predicates = response.get("trigger_predicates", [])
    if raw_predicates and not isinstance(raw_predicates, list):
        raise ValueError("trigger_predicates must be a list")
    predicates = parse_predicates(raw_predicates, "trigger") if raw_predicates else ()
    raw_controls = response.get("control_predicates", [])
    if not isinstance(raw_controls, list) or len(raw_controls) > 3:
        raise ValueError("control_predicates must be a list of up to three cases")
    controls = tuple(parse_predicates(item, "control") for item in raw_controls)
    preserve = response.get("preserve_first_operation", False)
    if type(preserve) is not bool:
        raise ValueError("preserve_first_operation must be a boolean")
    if category == "state_sequence" and not preserve:
        raise ValueError("state_sequence plan must preserve an ordinary first operation")
    if category == "state_sequence" and not controls:
        raise ValueError("state_sequence plan needs a first-operation control case")
    meaningful = [item for item in predicates if not (
        reset and item["left"] == reset and item["op"] == "==" and
        item["right"] == str(1 - _reset_active(reset)))]
    if category == "interaction" and len(meaningful) < 2:
        raise ValueError("interaction plan needs at least two explicit trigger predicates")
    if category == "interaction" and len(controls) < 2:
        raise ValueError("interaction plan needs two isolated-condition control cases")
    window = response.get("trigger_window_cycles", 0)
    if type(window) is not int or not 0 <= window <= 8:
        raise ValueError("trigger_window_cycles must be an integer from 0 to 8")
    plan = FaultPlan(**{name: response[name].strip() for name in fields},
                     trigger_predicates=predicates,
                     control_predicates=controls,
                     preserve_first_operation=preserve,
                     trigger_window_cycles=window)
    if plan.requirement_id != behavior.id or plan.category != category:
        raise ValueError("Fault plan requirement/category differs from requested focus")
    if plan.target_signal not in identifiers:
        raise ValueError("Fault plan target_signal is not an RTL identifier")
    if _normalized(plan.golden_behavior).casefold() == _normalized(plan.faulty_behavior).casefold():
        raise ValueError("Golden and faulty behavior must differ")
    return plan


def _request_plan(client: LLMPlanner, behavior: Behavior, category: str,
                  rtl: str, identifiers: set[str], history: list[dict],
                  context_limit: int, feedback: str = "",
                  reset: str | None = None) -> FaultPlan | str:
    for retry in range(2):
        try:
            return _plan(client, behavior, category, rtl, identifiers, history,
                         context_limit, feedback, reset)
        except (ValueError, json.JSONDecodeError) as exc:
            if retry:
                raise
            feedback += ("\nPrevious plan was invalid: " + _short(exc) +
                         ". Return a corrected JSON object with the requested fields.")
    raise AssertionError("unreachable")


def _parse_edits(response: dict, config: Config) -> tuple[Edit, ...]:
    raw = response.get("edits")
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict) and "edits" in raw[0]:
        raw = raw[0]["edits"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= 3:
        raise ValueError("Editor needs one to three RTL edits")
    return tuple(Edit.parse(item, config) for item in raw)


def _request_edits(client: LLMPlanner, prompt: str, context_limit: int,
                   config: Config) -> tuple[Edit, ...]:
    for retry in range(2):
        try:
            response = _model_request(client, EDIT_SYSTEM, prompt, context_limit,
                                      array_key="edits")
            return _parse_edits(response, config)
        except (ValueError, json.JSONDecodeError) as exc:
            if retry:
                raise
            prompt += ("\n\nFORMAT ERROR: " + _short(exc) +
                       "\nReturn only valid JSON with 1-3 edits. "
                       "Use {\"edits\":[{\"source_file\":\"...\",\"old\":\"...\",\"new\":\"...\"}]}.")
    raise AssertionError("unreachable")


def _top_header(config: Config, root: Path) -> str | None:
    pattern = re.compile(r"\bmodule\s+" + re.escape(config.top) + r"\b[^;]*;", re.DOTALL)
    for source in config.sources:
        match = pattern.search((root / source).read_text())
        if match:
            return _normalized(match.group())
    return None


def _critique(client: LLMPlanner, behavior: Behavior, plan: FaultPlan,
              edits: tuple[Edit, ...], rtl: str, accepted_plans: list[dict],
              context_limit: int, audit: dict | None = None,
              feedback: str = "") -> dict:
    audit_excerpt = ({key: value for key, value in audit.items()
                      if key not in ("behavior_signature", "witness_inputs")}
                     if audit else {"status": "disabled"})
    prompt = ("SPECIFICATION EXCERPT AND REQUIREMENT:\n" + json.dumps(asdict(behavior)) +
              "\n\nFAULT PLAN:\n" + json.dumps(asdict(plan)) +
              "\n\nACTUAL RTL EDITS:\n" + json.dumps([asdict(edit) for edit in edits]) +
              "\n\nCONTROLLER CHECK: VERILATOR LINT PASSED\n" +
              "\n\nMEASURED BEHAVIOR AUDIT:\n" + json.dumps(audit_excerpt) +
              "\n\nPRIOR ACCEPTED FAULT PLANS:\n" + json.dumps(accepted_plans[-12:]) +
              "\n\nREVIEW FEEDBACK:\n" + feedback +
              "\n\nGOLDEN RTL:\n" + rtl)
    raw = _model_request(client, CRITIC_SYSTEM, prompt, context_limit)
    expected = audit.get("witness_check") if audit else None
    if expected and raw.get("measured_first_difference") != expected:
        raise ValueError("Critic must copy measured_first_difference exactly from "
                         "the audit's witness_check: " + json.dumps(expected))
    verdict, scores, issues, reason = (raw.get("verdict"), raw.get("scores"),
                                       raw.get("issues"), raw.get("reason"))
    if verdict not in ("accept", "revise", "reject"):
        raise ValueError("Critic verdict must be accept, revise, or reject")
    if (not isinstance(scores, dict) or any(type(scores.get(name)) is not int or
                                             not 1 <= scores[name] <= 5
                                             for name in SCORE_FIELDS)):
        raise ValueError("Critic needs five integer quality scores from 1 to 5")
    if (not isinstance(issues, list) or any(not isinstance(issue, str) for issue in issues)
            or not isinstance(reason, str)):
        raise ValueError("Critic needs issues list and reason")
    observed = raw.get("observed_behavior", "")
    if not isinstance(observed, str):
        raise ValueError("Critic observed_behavior must be a string")
    return {"verdict": verdict, "scores": {name: scores[name] for name in SCORE_FIELDS},
            "total": sum(scores[name] for name in SCORE_FIELDS),
            "issues": issues, "reason": reason,
            "observed_behavior": observed,
            "measured_first_difference": raw.get("measured_first_difference")}


def _request_critique(client: LLMPlanner, behavior: Behavior, plan: FaultPlan,
                      edits: tuple[Edit, ...], rtl: str, accepted_plans: list[dict],
                      context_limit: int, audit: dict | None) -> dict:
    feedback = ""
    for retry in range(2):
        try:
            return _critique(client, behavior, plan, edits, rtl, accepted_plans,
                             context_limit, audit, feedback)
        except (ValueError, json.JSONDecodeError) as exc:
            if retry:
                raise
            feedback = "Previous review failed a format or trace fact check: " + _short(exc)
    raise AssertionError("unreachable")


def _concept_key(plan: FaultPlan) -> str:
    # The same RTL mechanism remains a duplicate when the model attaches it
    # to a different overlapping spec passage or fault category.
    return digest(json.dumps([plan.target_signal.casefold(),
                              _normalized(plan.root_cause).casefold()]).encode())


def run_quality_campaign(args: Any) -> None:
    """Plan, implement, critique, and save candidates without evaluation SVAs."""
    output = args.out.resolve()
    raw_config = direct_config(args)
    existing = {item.name for item in output.iterdir()} if output.exists() else set()
    if existing:
        saved = output / "config.direct.json"
        allowed = {saved.name, "setup_error.json", "spec_passages.json",
                   "requirements.json", "mapping_warning.json", "model_requests.jsonl"}
        if not existing.issubset(allowed) or not saved.is_file() or json.loads(saved.read_text()) != raw_config:
            raise ValueError(f"Use a fresh --out directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.direct.json", raw_config)
    config = Config.load(output / "config.direct.json")
    specification = read_spec(config.spec_file)
    write_json(output / "spec_passages.json", _spec_passages(specification))
    rtl, identifiers = _source_context(config)
    client = LLMPlanner(config, require_witness=False)
    mapping_mode = "llm"
    try:
        behaviors = _map_behaviors(client, config, specification, rtl, identifiers,
                                   args.max_requirements, args.max_context_chars)
    except RuntimeError as exc:
        write_json(output / "setup_error.json", {"stage": "requirement_mapping",
                                                "reason": _short(exc),
                                                "hint": "Check model endpoint/network access before retrying"})
        raise
    except (ValueError, json.JSONDecodeError) as exc:
        try:
            behaviors = _fallback_behaviors(config, specification, identifiers,
                                             args.max_requirements)
            mapping_mode = "deterministic_fallback"
            write_json(output / "mapping_warning.json", {"reason": _short(exc),
                                                        "fallback_requirements": len(behaviors)})
        except ValueError as fallback_exc:
            write_json(output / "setup_error.json", {"stage": "requirement_mapping",
                                                    "reason": _short(exc),
                                                    "fallback_reason": _short(fallback_exc),
                                                    "hint": "Review spec_passages.json; rerun with the same --out if only setup files exist"})
            raise RuntimeError(f"Requirement mapping failed after retries and fallback: {exc}; "
                               f"see {output / 'setup_error.json'}") from exc
    write_json(output / "requirements.json", [asdict(item) for item in behaviors])
    (output / "setup_error.json").unlink(missing_ok=True)
    categories = tuple(part.strip() for part in args.categories.split(","))
    golden_fingerprint = rtl_fingerprint(config, config.design_root)
    golden_header = _top_header(config, config.design_root)
    seen_rtl = {golden_fingerprint: "golden"}
    seen_concepts: dict[str, str] = {}
    history: list[dict] = []
    records: list[dict] = []
    accepted = 0
    audit_mode = getattr(args, "behavior_check", "required")
    if audit_mode not in ("required", "warn", "off"):
        raise ValueError("behavior_check must be required, warn, or off")
    auditor = (BehaviorAuditor(config, getattr(args, "max_probes", 48))
               if audit_mode != "off" else None)
    for index in range(1, args.max_attempts + 1):
        if accepted >= args.count:
            break
        requirement_index = (index - 1) % len(behaviors)
        requirement_round = (index - 1) // len(behaviors)
        behavior = behaviors[requirement_index]
        category = categories[(requirement_round + requirement_index) % len(categories)]
        identifier = f"m{index:04d}"
        directory = output / "mutants" / identifier
        directory.mkdir(parents=True)
        record: dict[str, Any] = {"id": identifier, "category": category,
                                  "requirement": asdict(behavior), "status": "pending",
                                  "revisions": []}
        try:
            plan = _request_plan(client, behavior, category, rtl, identifiers, history,
                                 args.max_context_chars, reset=config.reset)
            if isinstance(plan, FaultPlan) and _concept_key(plan) in seen_concepts:
                duplicate_of = seen_concepts[_concept_key(plan)]
                plan = _request_plan(client, behavior, category, rtl, identifiers, history,
                                     args.max_context_chars,
                                     f"Your previous proposal repeats {duplicate_of}. Choose a different "
                                     "RTL root cause and failure behavior for this requirement.",
                                     reset=config.reset)
            if isinstance(plan, str):
                record.update(status="unsupported_category",
                              reason=plan)
            else:
                record["plan"] = asdict(plan)
                concept = _concept_key(plan)
                if concept in seen_concepts:
                    record.update(status="duplicate_plan",
                                  reason=f"Same target and RTL root cause as {seen_concepts[concept]}")
                else:
                    seen_concepts[concept] = identifier
                    feedback = ""
                    previous_edits: list[dict] = []
                    for revision in range(args.max_repairs + 1):
                        revision_record: dict[str, Any] = {"revision": revision}
                        record["revisions"].append(revision_record)
                        prompt = ("FAULT PLAN:\n" + json.dumps(asdict(plan)) +
                                  "\n\nSPECIFICATION EXCERPT:\n" + behavior.spec_quote +
                                  "\n\nGOLDEN RTL:\n" + rtl +
                                  "\n\nPREVIOUS EDITS:\n" + json.dumps(previous_edits) +
                                  "\n\nFEEDBACK ON PREVIOUS EDIT:\n" + feedback)
                        try:
                            edits = _request_edits(client, prompt, args.max_context_chars,
                                                   config)
                            revision_record["edits"] = [asdict(item) for item in edits]
                            previous_edits = revision_record["edits"]
                            candidate = directory / "attempts" / f"a{revision + 1}"
                            candidate.mkdir(parents=True)
                            edit_result = apply_edits(config, edits, candidate)
                            if golden_header and _top_header(config, candidate / "rtl") != golden_header:
                                raise ValueError("The top module declaration or ports changed")
                            fingerprint = rtl_fingerprint(config, candidate / "rtl")
                            if fingerprint in seen_rtl:
                                raise ValueError(f"Resulting RTL duplicates {seen_rtl[fingerprint]}")
                            lint_result = lint(config, candidate)
                            revision_record["lint"] = lint_result
                            if lint_result["status"] != "passed":
                                raise ValueError("Verilator lint: " + _short(lint_result.get("detail", "")))
                            audit = auditor.assess(candidate / "rtl", plan, edits) if auditor else None
                            if audit:
                                revision_record["behavior_audit"] = audit
                                if audit_mode == "required" and audit["status"] != "passed":
                                    feedback = ("Behavior audit " + audit["status"] + ": " +
                                                audit.get("reason", "see measured evidence") +
                                                "\nMeasured first differences: " +
                                                json.dumps(audit.get("evidence", [])[:2]) +
                                                "\nTrigger conditions: " +
                                                json.dumps(audit.get("trigger_conditions", [])))
                                    revision_record.update(status="behavior_rejected", reason=feedback)
                                    continue
                            critique = _request_critique(client, behavior, plan, edits, rtl,
                                                         [item["plan"] for item in history
                                                          if item["status"] == "generated_unqualified" and item.get("plan")],
                                                         args.max_context_chars, audit)
                            revision_record["critique"] = critique
                            acceptable = (critique["verdict"] == "accept" and
                                          critique["total"] >= args.quality_threshold and
                                          all(value >= 3 for value in critique["scores"].values()))
                            if acceptable:
                                shutil.copytree(candidate / "rtl", directory / "rtl")
                                record.update(status="generated_unqualified",
                                              edit=edit_result, edits=[asdict(item) for item in edits],
                                              rtl_fingerprint=fingerprint, lint=lint_result,
                                              quality=critique, behavior_audit=audit)
                                seen_rtl[fingerprint] = identifier
                                if auditor and audit and audit["status"] == "passed":
                                    auditor.accept(audit["behavior_signature"], identifier)
                                accepted += 1
                                revision_record["status"] = "accepted"
                                break
                            reason = ("Critic " + critique["verdict"] + ": " +
                                      "; ".join(critique["issues"] or [critique["reason"]]))
                            if critique["total"] < args.quality_threshold or any(
                                    value < 3 for value in critique["scores"].values()):
                                reason += f"; scores={critique['scores']}"
                            revision_record.update(status="quality_rejected", reason=reason)
                            feedback = reason
                            if critique["verdict"] == "reject":
                                break
                        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                            feedback = _short(exc)
                            revision_record.update(status="edit_error", reason=feedback)
                    if record["status"] == "pending":
                        status = ("quality_rejected" if any(
                            item.get("status") == "quality_rejected" for item in record["revisions"])
                                  else "behavior_rejected" if any(
                            item.get("status") == "behavior_rejected" for item in record["revisions"])
                                  else "edit_rejected")
                        record.update(status=status, reason=feedback)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            record.update(status="plan_error", reason=_short(exc))
        write_json(directory / "record.json", record)
        records.append(record)
        history.append({"requirement_id": behavior.id, "category": category,
                        "status": record["status"], "plan": record.get("plan"),
                        "reason": record.get("reason", "")})
        print(f"{identifier}: {record['status']} ({behavior.id}/{category})")
    counts = {status: sum(item["status"] == status for item in records)
              for status in sorted({item["status"] for item in records})}
    by_category = {category: {"attempts": sum(item["category"] == category for item in records),
                              "accepted": sum(item["category"] == category and
                                              item["status"] == "generated_unqualified"
                                              for item in records)}
                   for category in categories}
    accepted_records = [item for item in records
                        if item["status"] == "generated_unqualified"]
    evidence_profile = {
        "with_output_witness": sum(bool((item.get("behavior_audit") or {}).get("output_witness"))
                                   for item in accepted_records),
        "with_first_operation_difference": sum(bool((item.get("behavior_audit") or {}).get(
            "ordinary_difference")) for item in accepted_records),
        "with_unobserved_controls": sum(bool((item.get("behavior_audit") or {}).get(
            "unobserved_controls")) for item in accepted_records),
        "without_behavior_audit": sum(not item.get("behavior_audit")
                                      for item in accepted_records),
    }
    write_json(output / "summary.json", {"planner_mode": "spec_refine",
                                        "requirement_mapping": mapping_mode,
                                        "config": str(output / "config.direct.json"),
                                        "requirements": len(behaviors),
                                        "requested": args.count,
                                        "attempts": len(records),
                                        "accepted": accepted,
                                        "counts": counts,
                                        "by_category": by_category,
                                        "evidence_profile": evidence_profile,
                                        "quality_note": "LLM critique is a review aid, not spec proof",
                                        "behavior_check": audit_mode,
                                        "behavior_probes": len(auditor.probes) if auditor else 0,
                                        "assertions_used_for_generation": False,
                                        "mutants": [item["id"] for item in records]})
    if auditor:
        auditor.close()
    print(f"Results: {output / 'summary.json'}")


def audit_saved_candidates(output: Path, max_probes: int = 48) -> dict:
    """Replay an existing refine campaign without editing its mutants or scores."""
    output = output.resolve()
    config = Config.load(output / "config.direct.json")
    records = sorted((output / "mutants").glob("*/record.json"))
    if not records:
        raise ValueError(f"No saved mutant records under {output / 'mutants'}")
    results: list[dict] = []
    with BehaviorAuditor(config, max_probes) as auditor:
        for path in records:
            record = json.loads(path.read_text())
            if record.get("status") != "generated_unqualified":
                continue
            try:
                plan = FaultPlan(**record["plan"])
                edits = tuple(Edit(**item) for item in record["edits"])
                result = auditor.assess(path.parent / "rtl", plan, edits)
                if result["status"] == "passed":
                    auditor.accept(result["behavior_signature"], record["id"])
            except (KeyError, TypeError, OSError, ValueError, RuntimeError) as exc:
                result = {"status": "audit_error", "reason": _short(exc)}
            results.append({"id": record["id"], "category": record.get("category"),
                            "audit": result})
    counts = {status: sum(item["audit"]["status"] == status for item in results)
              for status in sorted({item["audit"]["status"] for item in results})}
    report = {"source_run": str(output), "candidate_mutants": len(results),
              "max_probes": max_probes, "counts": counts,
              "note": "Sampled behavior screen; passed does not prove a specification fault",
              "results": results}
    write_json(output / "behavior_audit.json", report)
    print(f"Behavior audit: {output / 'behavior_audit.json'}")
    for item in results:
        print(f"{item['id']}: {item['audit']['status']}")
    return report
