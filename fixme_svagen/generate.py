"""Spec-first SANGAM-style generation using the selected Nebius DeepSeek model."""

from __future__ import annotations

import json
import hashlib
import math
import re
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prompts import COMBINER_PROMPT, CRITIC_SYSTEM_PROMPT, SVA_SYSTEM_PROMPT

from .jasper import JasperRunner, JGResult, load_connection
from .sva import Assertion, extract_assertions


MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
GENERATION_CONFIG = "deepseek_v4_chat_v1"
API_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"


class NebiusClient:
    def __init__(self, env_file: Path, model: str = MODEL, max_tokens: int = 8192):
        values = load_connection(env_file)
        self.keys = [values[f"NEBIUS_API_KEY{i}"] for i in range(1, 25) if values.get(f"NEBIUS_API_KEY{i}")]
        if not self.keys:
            raise RuntimeError("Set NEBIUS_API_KEY1 in .env to generate assertions")
        self.key_index = 0
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, system: str, user: str) -> str:
        last_error: Exception | None = None
        short_final = False
        for attempt in range(12):
            request_user = user
            if short_final:
                request_user += (
                    "\nYour previous response ran out of output tokens before giving a final answer. "
                    "Put the requested final artifact first. Omit analysis and keep the final answer "
                    "under 2000 tokens; for SVAs, return at most 12 assertions."
                )
            body = json.dumps({
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": request_user}],
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"thinking": False},
                "stream": False,
            }).encode()
            request = Request(API_URL, data=body, headers={
                "Authorization": "Bearer " + self.keys[self.key_index],
                "Content-Type": "application/json",
            })
            try:
                with urlopen(request, timeout=240) as response:
                    payload = json.load(response)
                choice = payload["choices"][0]
                content = choice["message"].get("content")
                if not content or not isinstance(content, str):
                    if choice.get("finish_reason") == "length":
                        short_final = True
                    raise ValueError(f"Model returned no final text (finish_reason={choice.get('finish_reason')})")
                return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
            except HTTPError as exc:
                last_error = RuntimeError(f"Nebius HTTP {exc.code}: {exc.read(500).decode(errors='replace')}")
                if exc.code in (401, 403, 402) and len(self.keys) > 1:
                    self.key_index = (self.key_index + 1) % len(self.keys)
                elif exc.code not in (408, 429, 500, 502, 503, 504):
                    break
            except URLError as exc:
                last_error = exc
                # The endpoint occasionally presents a certificate for another
                # hostname for a short period. Keep verification enabled and
                # retry after the endpoint recovers.
                if (isinstance(exc.reason, ssl.SSLCertVerificationError)
                        and "hostname mismatch" in str(exc).lower()):
                    if attempt < 11:
                        time.sleep(10)
                    continue
                if attempt >= 3:
                    break
            except (TimeoutError, RuntimeError, ValueError, KeyError, IndexError) as exc:
                last_error = exc
                if attempt >= 3:
                    break
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Nebius request failed: {last_error}")


def extract_json_object(text: str) -> dict:
    stripped = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.I).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        value = json.loads(stripped[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Signal mapper did not return a JSON object")
    return value


def map_signals(llm: NebiusClient, spec: str, rtl: str) -> list[dict[str, str]]:
    """Ask the model to map signals from the specification; RTL only validates names."""
    prompt = (
        "Extract the signals described in this design specification for per-signal SVA generation. "
        "Return only JSON: {\"signals\":[{\"name\":\"...\",\"information\":\"spec-based description\"}]}. "
        "Use exact signal identifiers from the specification and do not invent functionality. "
        "Include ports and internal signals when the specification describes them.\n\n" + spec
    )
    data = extract_json_object(llm.chat("You are SANGAM's specification signal mapper.", prompt))
    signals: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in data.get("signals", []):
        name = item.get("name", "") if isinstance(item, dict) else ""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) or name in seen:
            continue
        if not re.search(r"\b" + re.escape(name) + r"\b", rtl):
            continue
        seen.add(name)
        signals.append({"name": name, "information": str(item.get("information", ""))})
    if not signals:
        raise ValueError("No spec-mapped signal identifiers were found in RTL")
    return signals


@dataclass
class Node:
    answer: str
    parent: int | None
    score: float
    hint: str
    feedback: str
    all_proven: bool
    visits: int = 1


class SangamGenerator:
    def __init__(self, llm: NebiusClient, jasper: JasperRunner, rtl: str, top: str,
                 clock: str | None, reset: str | None, workdir: Path, iterations: int):
        self.llm = llm
        self.jasper = jasper
        self.rtl = rtl
        self.top = top
        self.clock = clock
        self.reset = reset
        self.workdir = workdir
        self.iterations = iterations
        self.cache: dict[str, JGResult] = {}
        (workdir / "candidate_logs").mkdir(parents=True, exist_ok=True)
        (workdir / "candidate_results").mkdir(parents=True, exist_ok=True)

    def check(self, assertion: Assertion) -> JGResult:
        key = assertion.support + assertion.text
        if key not in self.cache:
            digest = hashlib.sha256(key.encode()).hexdigest()[:16]
            log_path = self.workdir / "candidate_logs" / f"{digest}.log"
            result_path = self.workdir / "candidate_results" / f"{digest}.json"
            fingerprint = hashlib.sha256(json.dumps([
                self.rtl, self.top, self.clock, self.reset, getattr(self.jasper, "timeout", None),
                getattr(self.llm, "model", MODEL), GENERATION_CONFIG, assertion.text,
                assertion.support, assertion.implication,
            ], ensure_ascii=False).encode()).hexdigest()
            if result_path.exists() and log_path.exists():
                try:
                    saved = json.loads(result_path.read_text())
                    log = log_path.read_text(errors="replace")
                    if (isinstance(saved, dict) and saved.get("fingerprint") == fingerprint
                            and saved.get("log_sha256") == hashlib.sha256(log.encode()).hexdigest()):
                        result = JGResult(saved["assertion_status"], saved["precondition_status"],
                                          saved["contrapositive_status"], log, saved["exit_code"])
                        self.cache[key] = result
                        return result
                except (json.JSONDecodeError, KeyError):
                    # An interrupted checkpoint must be checked again, never treated as evidence.
                    pass
            result = self.jasper.check(self.rtl, self.top, assertion, self.clock, self.reset)
            if result.assertion_status == "license_unavailable":
                raise RuntimeError("JasperGold license checkout failed; stop generation until the license server is reachable")
            self.cache[key] = result
            log_path.write_text(result.log)
            payload = {
                "fingerprint": fingerprint,
                "log_sha256": hashlib.sha256(result.log.encode()).hexdigest(),
                "assertion_status": result.assertion_status,
                "precondition_status": result.precondition_status,
                "contrapositive_status": result.contrapositive_status,
                "exit_code": result.exit_code,
            }
            temp = result_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False))
            temp.replace(result_path)
        return self.cache[key]

    def evaluate(self, query: str, answer: str) -> tuple[float, str, str, bool]:
        assertions = extract_assertions(answer)
        if not assertions:
            feedback = "No parseable assert property statement was generated."
            all_proven = False
        else:
            details = []
            for assertion in assertions:
                result = self.check(assertion)
                detail = f"{assertion.text}: {result.assertion_status}"
                if result.assertion_status in ("error", "unknown", "timeout"):
                    detail += "; tool output: " + result.log[-1200:]
                details.append(detail)
            feedback = "\n".join(details)
            all_proven = all(self.check(a).assertion_status == "proven" for a in assertions)
        critic = self.llm.chat(
            CRITIC_SYSTEM_PROMPT,
            f"Specification-based task: {query}\nGenerated answer:\n{answer}\nJasperGold feedback:\n{feedback}\n"
            "Critique every issue. End with [Score] followed by one number from -100 to 100.",
        )
        matches = re.findall(r"(?:\[Score\]|\bScore\b)\s*:?\s*(-?\d+(?:\.\d+)?)", critic, re.I)
        score = float(matches[-1]) if matches else -50.0
        return max(-100.0, min(score, 100.0)), critic, feedback, all_proven

    def run_signal(self, signal: dict[str, str], spec: str) -> tuple[list[Assertion], dict]:
        name = signal["name"]
        query = f"Generate SVAs for {name}. {signal['information']}\nFull specification:\n{spec}"
        if self.clock is None:
            query += (
                "\nThis RTL is combinational and has no clock. Write clockless concurrent "
                "assertions, for example assert property (antecedent |-> consequent); "
                "Do not use @(posedge ...), default clocking, $past, |=>, or ##. "
                "Use only signal names present in the specification."
            )
        weak_file = self.workdir / "weak_response.txt"
        if weak_file.exists():
            weak = weak_file.read_text()
        else:
            weak = self.llm.chat(
                SVA_SYSTEM_PROMPT,
                query + "\nReturn at most 12 distinct, specification-grounded assertions in a "
                "systemverilog code fence. Put the code first and omit explanations.",
            )
            weak_file.write_text(weak)
        score, hint, feedback, all_proven = self.evaluate(query, weak)
        nodes = [Node(weak, None, score, hint, feedback, all_proven)]
        for iteration in range(self.iterations):
            if nodes[-1].all_proven:
                break
            total = sum(node.visits for node in nodes)
            selected = max(range(len(nodes)), key=lambda i: nodes[i].score + 1.4 * math.sqrt(math.log(total + 1) / nodes[i].visits))
            parent = nodes[selected]
            parent.visits += 1
            improved_file = self.workdir / f"improved_{iteration}.txt"
            if improved_file.exists():
                improved = improved_file.read_text()
            else:
                improved = self.llm.chat(
                    SVA_SYSTEM_PROMPT,
                    f"{query}\nPrevious assertions:\n{parent.answer}\nCritic feedback:\n{parent.hint}\n"
                    f"JasperGold feedback:\n{parent.feedback}\nRefine the assertions. "
                    "Return only a systemverilog code fence with assertions for this signal.",
                )
                improved_file.write_text(improved)
            score, hint, feedback, all_proven = self.evaluate(query, improved)
            nodes.append(Node(improved, selected, score, hint, feedback, all_proven))

        combined_file = self.workdir / "combined_response.txt"
        if combined_file.exists():
            combined = combined_file.read_text()
        else:
            combined = self.llm.chat(
                COMBINER_PROMPT,
                f"Specification-based task: {query}\nReasoning tree:\n"
                + "\n".join(f"Answer {i}: {node.answer}\nCritique: {node.hint}" for i, node in enumerate(nodes))
                + "\nReturn the combined assertions in a systemverilog code fence.",
            )
            combined_file.write_text(combined)
        parsed = [a for node in nodes for a in extract_assertions(node.answer)] + extract_assertions(combined)
        candidates = list(dict.fromkeys(a.text for a in parsed))
        unique = {a.text: a for a in parsed}
        valid = [unique[text] for text in candidates if self.check(unique[text]).syntax_ok()]
        invalid = [unique[text] for text in candidates if text not in {a.text for a in valid}]
        if invalid:
            repaired = self.llm.chat(
                SVA_SYSTEM_PROMPT,
                f"Using this specification, repair only syntax/elaboration errors in these assertions for {name}. "
                "Keep their intended meaning. Return only a systemverilog code fence.\n"
                + ("The RTL has no clock; all assertions must be clockless.\n" if self.clock is None else "")
                + spec + "\n" + "\n".join(a.text + "\n" + self.check(a).log[-600:] for a in invalid),
            )
            for assertion in extract_assertions(repaired):
                if self.check(assertion).syntax_ok():
                    valid.append(assertion)
        valid = list({a.text: a for a in valid}.values())
        if valid:
            candidate_groups: dict[str, list[str]] = {}
            for assertion in valid:
                candidate_groups.setdefault(assertion.support, []).append(assertion.text)
            # Shared declarations belong to a group, not to every assertion in the prompt.
            # Repeating a large property block per assertion can exhaust the model context.
            candidate_text = "\n\n".join(
                f"Candidate group {index}:\n{support}\n" + "\n".join(statements)
                for index, (support, statements) in enumerate(candidate_groups.items(), start=1)
            )
            final_prompt = (
                f"Specification:\n{spec}\nSignal: {name}\nCandidate assertions:\n"
                + candidate_text
                + "\nReturn every unique valid assertion without modifying it. "
                "Copy the candidate assertions exactly, including their clock syntax. "
                "Return only a systemverilog code fence. Do not add new assertions."
            )
            if self.clock is None:
                final_prompt += "\nThis RTL has no clock. Do not add @(posedge ...), clk, or default clocking."
            final_response = self.llm.chat(
                SVA_SYSTEM_PROMPT,
                final_prompt,
            )
            by_normalized = {re.sub(r"\s+", "", a.text): a for a in valid}

            def select_exact(response: str) -> list[Assertion]:
                selected = []
                for proposed in extract_assertions(response):
                    original = by_normalized.get(re.sub(r"\s+", "", proposed.text))
                    if original and original not in selected:
                        selected.append(original)
                return selected

            final_assertions = select_exact(final_response)
            if not final_assertions:
                retry_response = self.llm.chat(
                    SVA_SYSTEM_PROMPT,
                    final_prompt + "\nYour previous answer changed the candidate assertions. "
                    "Copy their exact text now; do not insert a clock or rewrite any expression.",
                )
                final_assertions = select_exact(retry_response)
                if final_assertions:
                    final_response = retry_response
        else:
            final_response = ""
            final_assertions = []
        trace = {
            "signal": signal,
            "nodes": [node.__dict__ for node in nodes],
            "candidate_count": len(candidates),
            "valid_candidate_count": len(valid),
            "combined_response": combined,
            "final_response": final_response,
            "final_assertions": [a.__dict__ for a in final_assertions],
        }
        return final_assertions, trace
