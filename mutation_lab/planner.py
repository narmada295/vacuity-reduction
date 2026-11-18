"""One-proposal-at-a-time LLM planner with an offline JSONL alternative."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .core import Config, Proposal, ScenarioMutation, ScenarioPlan, confined_relative, read_spec


SYSTEM = """You are planning one functional RTL mutation for assertion evaluation.
Use the specification as the source of intended behavior and the golden RTL only to locate
a plausible, single-edit fault. Return exactly one JSON object and no explanation outside it.
Fields (all strings): requirement_id, failure, source_file, old, new,
expected_observation, rationale. source_file must be one listed source. old must be an
EXACT, UNIQUE substring of that source; new replaces it once. Preserve module ports,
clock/reset wiring, and syntactic validity. Choose one local functional change. Do not
propose an assertion edit. Do not repeat a prior resulting behavior or target when another
behavior is available. Treat the intended failure as a hypothesis.
"""

SCENARIO_SYSTEM = """Plan ONE plausible temporal RTL fault from the supplied specification.
Return one JSON object with string fields category, requirement_id, spec_excerpt,
golden_behavior, faulty_behavior, observation, and a trigger_sequence list of 2-12 strings.
The category must be the requested category. spec_excerpt must quote a contiguous
12-500 character phrase from the supplied specification exactly. Describe the events needed to
activate the fault and when the resulting behavior becomes observable. Do not invent
requirements or include the evaluation assertions. Avoid artificial delay or unrelated
faults. For timing, target an incorrect cycle or interval; for state_sequence,
target behavior dependent on earlier state or events; for interaction, target
concurrent or overlapping operations. Work through the RTL's clocked next-state updates
from reset before claiming a cycle number or period. In particular, do not claim that a
counter reaches a value after a condition that resets it. For state_sequence, include at
least two activations of the relevant event; for interaction, identify both operations.
Do not repeat a prior behavior or target when another behavior is available. Do not write
any RTL edit yet. Return JSON only. Include a witness object with cycles and observations.
cycles is a list of 2-128 objects mapping input port names (other than the clock) to
nonnegative integer values or 0x-prefixed hex strings, sampled at successive rising
clock edges. The first cycle must set reset when the design has a reset. Inputs omitted
in later cycles hold their previous value. observations is a list of at least two objects
with zero-based cycle, signal, golden, and mutant fields. signal must name a top-level
output or a direct internal register; golden and mutant are expected integer values after
that cycle's rising edge. At least one observation must differ between golden and mutant.
For timing, check both the missed/early event and the next expected event. For state_sequence,
check the first activation and a later activation. For interaction, check before and after
the second operation. Choose a short, reachable sequence that tests the specific faulty
behavior, not merely any difference from golden.
The fault must be selective: do not simply disable an output, force a constant, or break
the ordinary first operation. Include witness.controls: one independent normal-operation
trace for timing or state_sequence, and two traces for interaction (each operation alone).
Each control has the same cycles/observations shape as the main witness, but its golden
and mutant values must agree and at least one top-level output must change during the
control trace. The main trace must include an observation where golden and mutant agree
before the first disagreement. For state_sequence, the first activation must work normally
and the later activation must fail. For interaction, neither operation alone may fail.
Use short, concrete input sequences; do not pad them with idle cycles to appear complex.
"""

SCENARIO_EDIT_SYSTEM = """Implement the ONE supplied fault scenario in the golden RTL.
Return JSON only, with an edits list of 1-3 objects; each object has string fields
source_file, old, new. old must be an EXACT UNIQUE substring of one listed RTL source.
All edits must be related parts of this single conceptual fault. Preserve module ports,
clock signal and syntactic validity. Do not edit assertions, add arbitrary delay,
or change unrelated behavior. Prefer one edit when possible. Mentally run the edited RTL
from the scenario's reset and input sequence. If the edit would instead make a signal
constant, make the fault occur immediately, or produce an unreachable condition, revise
the edit before returning it. A comment or formatting change is not a mutation.
Preserve the normal behavior shown in every control trace. If the planned fault cannot
be implemented without breaking a control, return {"edits": []} so the controller can
try another plan. Do not substitute an easy always-on fault for the planned scenario.
"""

JOINT_SCENARIO_SYSTEM = """Plan and implement ONE plausible RTL fault from the supplied
specification and golden RTL in a single response. Return exactly one JSON object with
two fields: scenario and edits. scenario must contain string fields category,
requirement_id, spec_excerpt, golden_behavior, faulty_behavior, observation, plus a
trigger_sequence list of 2-12 strings. The category must be the requested category.
spec_excerpt must quote a contiguous 12-500 character phrase from the specification.
For timing, target an incorrect cycle or interval; for state_sequence, behavior depending
on earlier state or events; for interaction, concurrent or overlapping operations.
edits must be a list of 1-3 related objects with string source_file, old, and new fields.
Each old must be an EXACT UNIQUE substring of a listed RTL source. The edits must
implement the scenario's faulty_behavior, not merely make an assertion fail. Check the
actual behavior of the proposed new RTL against the trigger_sequence and observation
before returning the JSON. Preserve module ports, clock signal, and syntactic validity.
Do not edit assertions, add arbitrary delays, or change unrelated behavior. Treat the
intended failure as a hypothesis; no evaluation assertions are available.
Include scenario.witness using the same cycles and observations schema as the two-call
scenario planner: input port values per rising-edge cycle, and at least two observations
with cycle, signal, golden, and mutant integer values. The first cycle sets reset when
present. The edited RTL must match every mutant observation and the golden RTL must match
every golden observation. Include an observation at the intended later activation.
Include scenario.witness.controls as independent normal-operation traces: one for timing
or state_sequence, two for interaction, each with cycles and observations in the same
format. Golden and mutant must agree throughout each control. Each control must exercise
at least one top-level output transition. The main trace must show agreement before its
first disagreement. Do not substitute a constant output or a fault that breaks the first
ordinary operation for the planned selective fault.
"""


FAULT_ANGLES = {
    "timing": (
        "an incorrect boundary cycle when a counter or interval expires",
        "a stale value used for one cycle after a configuration or state update",
        "an event handled early or late only after a restart or re-enable sequence",
    ),
    "state_sequence": (
        "the first transaction succeeds but a repeated transaction uses stale state",
        "state retained or cleared incorrectly across a reset and later operation",
        "an accumulated state value becomes wrong only after multiple events",
    ),
    "interaction": (
        "a new request arrives while an earlier operation is active",
        "two inputs or events in the same cycle get the wrong priority",
        "a configuration update and an active operation use an inconsistent old/new value",
    ),
}


def _env_values(path: Path | None) -> dict[str, str]:
    values = dict(os.environ)
    if path and path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            values.setdefault(name.strip(), value.strip().strip("\"'"))
    return values


def _json_object(text: str, array_key: str | None = None) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        object_span = (stripped.find("{"), stripped.rfind("}"))
        array_span = (stripped.find("["), stripped.rfind("]"))
        candidates = [object_span]
        if array_key:
            candidates = ([array_span, object_span] if array_span[0] >= 0 and
                          (object_span[0] < 0 or array_span[0] < object_span[0])
                          else [object_span, array_span])
        for start, end in candidates:
            if start < 0 or end < start:
                continue
            try:
                value = json.loads(stripped[start:end + 1])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError("Model returned no parseable JSON object") from None
    if isinstance(value, list) and array_key:
        return {array_key: value}
    if not isinstance(value, dict):
        raise ValueError(f"Model response must be a JSON object (got {type(value).__name__})")
    return value


class LLMPlanner:
    def __init__(self, config: Config, scenario_mode: str | None = None,
                 require_witness: bool | None = None):
        self.config = config
        settings = config.planner
        self.scenario_mode = scenario_mode or settings.get("scenario_mode", "two-call")
        self.require_witness = (bool(settings.get("require_witness", True))
                                if require_witness is None else require_witness)
        if self.scenario_mode not in ("one-call", "two-call"):
            raise ValueError("planner scenario_mode must be one-call or two-call")
        self.endpoint = settings.get("endpoint", "https://api.tokenfactory.nebius.com/v1/chat/completions")
        self.model = settings.get("model", "deepseek-ai/DeepSeek-V4-Flash-0731")
        env_file = settings.get("env_file")
        env_path = (config.path.parent / env_file).resolve() if env_file else None
        key_name = settings.get("key_env", "NEBIUS_API_KEY1")
        self.key = _env_values(env_path).get(key_name)
        if not self.key:
            raise RuntimeError(f"Missing {key_name} for LLM planning")

    def _context(self, feedback: list[dict], category: str) -> str:
        config = self.config
        heading = ("REVIEWED REQUIREMENTS:\n" if not config.oracle_file else
                   "REVIEWED REQUIREMENTS AND FORMAL CHECKER LABELS:\n")
        sections = ["SPECIFICATION:\n" + read_spec(config.spec_file),
                    heading + json.dumps([
                        {"id": r.id, "text": r.text} for r in config.requirements
                    ], indent=2)]
        focus = config.planner.get("context_sources", [str(source) for source in config.sources])
        if not isinstance(focus, list) or not focus:
            raise ValueError("planner.context_sources must be a nonempty list")
        for name in focus:
            source = confined_relative(name)
            if source not in config.sources:
                raise ValueError(f"Planner context source is not configured RTL: {name}")
            sections.append(f"GOLDEN RTL FILE {source}:\n" + (config.design_root / source).read_text())
        targets: list[str] = []
        assignment = re.compile(r"(?m)^\s*(?:assign\s+)?([A-Za-z_]\w*)(?:\s*\[[^\]\n]+\])?\s*(?:<=|=(?!=))")
        for name in focus:
            for target in assignment.findall((config.design_root / confined_relative(name)).read_text()):
                if target not in targets and target not in (config.clock, config.reset):
                    targets.append(target)
        if targets:
            counts = {target: sum(item.get("focus_target") == target and
                                  item.get("category") == category for item in feedback)
                      for target in targets}
            self.last_focus_target = min(targets, key=lambda target: counts[target])
            sections.append("UNDEREXPLORED RTL UPDATE: " + self.last_focus_target +
                            "\nPrefer a spec-supported fault whose root cause is in this "
                            "update. The requested category and selective behavior take "
                            "priority if this update cannot support them.")
        else:
            self.last_focus_target = None
        angles = FAULT_ANGLES.get(category)
        if angles:
            tried = sum(item.get("category") == category for item in feedback)
            self.last_fault_angle = angles[tried % len(angles)]
            sections.append("FAULT ANGLE TO EXPLORE: " + self.last_fault_angle +
                            "\nUse it only if the specification and RTL support it. "
                            "Otherwise choose another precise fault in this category. "
                            "Preserve normal operation in the control traces.")
        else:
            self.last_fault_angle = None
        same_category = [item for item in feedback if item.get("category") == category][-20:]
        recent_other = [item for item in feedback if item.get("category") != category][-4:]
        sections.append("REQUESTED CATEGORY HISTORY (avoid the same resulting RTL and "
                        "the same faulty behavior, even if wording or edit anchors differ):\n"
                        + json.dumps(same_category, indent=2))
        sections.append("RECENT OTHER-CATEGORY FEEDBACK:\n"
                        + json.dumps(recent_other, indent=2))
        prompt = "\n\n".join(sections)
        limit = int(config.planner.get("max_context_chars", 100000))
        if len(prompt) > limit:
            raise ValueError(f"Planner context is {len(prompt)} characters; limit is {limit}. "
                             "Use a smaller configured source set or raise max_context_chars.")
        return prompt

    def _request(self, system: str, prompt: str, array_key: str | None = None) -> dict:
        config = self.config
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": int(config.planner.get("max_tokens", 1800)),
            "stream": False,
            **config.planner.get("extra_body", {}),
        }).encode()
        request = Request(self.endpoint, data=body, headers={
            "Authorization": "Bearer " + self.key,
            "Content-Type": "application/json",
        })
        started = time.monotonic()
        try:
            with urlopen(request, timeout=int(config.planner.get("timeout_seconds", 120))) as response:
                payload = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM connection failed: {exc.reason}") from exc
        choices = payload.get("choices", [])
        choice = choices[0] if choices else {}
        log_file = config.planner.get("request_log")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"requested_model": self.model, "served_model": payload.get("model"),
                     "response_id": payload.get("id"), "created": payload.get("created"),
                     "elapsed_seconds": round(time.monotonic() - started, 3),
                     "finish_reason": choice.get("finish_reason"),
                     "usage": payload.get("usage", {})}
            with log_path.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
        if choice.get("finish_reason") == "length" and not config.planner.get("raw_response_file"):
            raise ValueError("Model response was truncated at its token limit; increase --planner-max-tokens")
        try:
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("LLM response has no message content") from exc
        if not isinstance(content, str):
            raise ValueError("LLM response content is not text")
        raw_response_file = config.planner.get("raw_response_file")
        if raw_response_file:
            Path(raw_response_file).write_text(content)
        if choice.get("finish_reason") == "length":
            raise ValueError("Model response was truncated at its token limit; increase --planner-max-tokens")
        return _json_object(content, array_key=array_key)

    def propose(self, feedback: list[dict], category: str = "simple") -> Proposal | ScenarioMutation:
        context = self._context(feedback, category)
        if category == "simple":
            return Proposal.parse(self._request(SYSTEM, context), self.config)
        if self.scenario_mode == "one-call":
            response = self._request(JOINT_SCENARIO_SYSTEM,
                                     context + "\n\nREQUESTED CATEGORY: " + category)
            mutation = ScenarioMutation.parse(response, self.config, category)
            if getattr(self, "require_witness", False) and mutation.scenario.witness is None:
                raise ValueError("LLM scenario needs a concrete witness")
            if getattr(self, "require_witness", False):
                controls = mutation.scenario.witness.get("controls")
                needed = 2 if category == "interaction" else 1
                if not isinstance(controls, list) or len(controls) < needed:
                    raise ValueError(f"LLM {category} scenario needs {needed} normal-operation control trace(s)")
            return mutation
        scenario = ScenarioPlan.parse(self._request(
            SCENARIO_SYSTEM, context + "\n\nREQUESTED CATEGORY: " + category),
            self.config, category)
        if getattr(self, "require_witness", False) and scenario.witness is None:
            raise ValueError("LLM scenario needs a concrete witness")
        if getattr(self, "require_witness", False):
            controls = scenario.witness.get("controls")
            needed = 2 if category == "interaction" else 1
            if not isinstance(controls, list) or len(controls) < needed:
                raise ValueError(f"LLM {category} scenario needs {needed} normal-operation control trace(s)")
        response = self._request(SCENARIO_EDIT_SYSTEM,
                                 context + "\n\nPROPOSED FAULT SCENARIO:\n"
                                 + json.dumps(asdict(scenario), indent=2))
        return ScenarioMutation.parse({"scenario": asdict(scenario),
                                       "edits": response.get("edits")}, self.config, category)


class FilePlanner:
    """Offline plans let a run be reproduced without calling an LLM."""

    def __init__(self, config: Config, path: Path):
        self.config = config
        self.lines = [line for line in path.read_text().splitlines() if line.strip()]
        self.position = 0

    def propose(self, feedback: list[dict], category: str = "simple") -> Proposal | ScenarioMutation:
        if self.position >= len(self.lines):
            raise StopIteration
        line = self.lines[self.position]
        self.position += 1
        data = json.loads(line)
        if category == "simple":
            return Proposal.parse(data, self.config)
        return ScenarioMutation.parse(data, self.config, category)
