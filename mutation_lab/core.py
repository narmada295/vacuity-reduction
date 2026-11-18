"""Configuration, proposal validation, and isolated RTL generation."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
SV_TOKENS = re.compile(
    r'\s+|//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\\[^\s]+'
    r'|[A-Za-z_$][A-Za-z_0-9$]*|\d+(?:\s*\'[sS]?[bBoOdDhH][0-9a-fA-F_xXzZ?]+)?'
    r'|>>>=|<<=|>>=|===|!==|==\?|!=\?|<<<|>>>|<=|>=|==|!=|&&|\|\|'
    r'|<<|>>|\*\*|~&|~\||~\^|\^~|\+:|-:|::|->|=>|##|\+=|-='
    r'|\*=|/=|%=|&=|\|=|\^=|.', re.DOTALL)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rtl_tokens(source: str) -> tuple[str, ...]:
    """Conservative token view for rejecting comments and formatting as mutants.

    Files with preprocessor directives use exact text: whitespace can affect macros.
    This is a duplicate filter, not a behavioral-equivalence proof.
    """
    if "`" in source:
        return (source,)
    return tuple(token for token in SV_TOKENS.findall(source)
                 if not token.isspace() and not token.startswith(("//", "/*")))


def rtl_fingerprint(config: "Config", source_root: Path) -> str:
    """Hash configured RTL files, ignoring comments and spacing where safe."""
    return digest(json.dumps([
        [str(source), rtl_tokens((source_root / source).read_text())]
        for source in config.sources
    ], ensure_ascii=False).encode())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def confined_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in ("..", ".") for part in path.parts):
        raise ValueError(f"Expected a relative path inside the design root: {value!r}")
    return path


def resolve_config_path(config_dir: Path, value: str) -> Path:
    return (config_dir / value).resolve()


@dataclass(frozen=True)
class Requirement:
    id: str
    text: str
    oracle: str
    reach: str


@dataclass(frozen=True)
class Config:
    path: Path
    design_root: Path
    sources: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    top: str
    spec_file: Path
    requirements: tuple[Requirement, ...]
    oracle_file: Path | None
    environment_file: Path | None
    evaluation_file: Path | None
    evaluation_names: tuple[str, ...]
    planner: dict[str, Any]
    jasper: dict[str, Any]
    lint_settings: dict[str, Any]
    clock: str | None
    reset: str | None

    @classmethod
    def load(cls, path: Path) -> "Config":
        path = path.resolve()
        raw = json.loads(path.read_text())
        base = path.parent
        root = resolve_config_path(base, raw["design_root"])
        if not root.is_dir():
            raise ValueError(f"Design root does not exist: {root}")
        sources = tuple(confined_relative(item) for item in raw["sources"])
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("sources must be a nonempty list without duplicates")
        for source in sources:
            if not (root / source).is_file() or not (root / source).resolve().is_relative_to(root):
                raise ValueError(f"Source missing or outside design root: {source}")
        include_dirs = tuple(confined_relative(item) if item != "." else Path(".")
                             for item in raw.get("include_dirs", ["."]))
        for directory in include_dirs:
            if not (root / directory).is_dir() or not (root / directory).resolve().is_relative_to(root):
                raise ValueError(f"Include directory missing or outside design root: {directory}")
        top = raw["top"]
        if not IDENTIFIER.fullmatch(top):
            raise ValueError("top must be a Verilog identifier")
        requirements = tuple(Requirement(**item) for item in raw.get("requirements", []))
        ids = [item.id for item in requirements]
        if len(set(ids)) != len(ids):
            raise ValueError("Requirement IDs must be unique")
        for item in requirements:
            if not item.text.strip() or any(not IDENTIFIER.fullmatch(v)
                                           for v in (item.id, item.oracle, item.reach)):
                raise ValueError(f"Invalid requirement: {item}")
        eval_names = tuple(raw.get("evaluation_names", []))
        if any(not IDENTIFIER.fullmatch(name) for name in eval_names):
            raise ValueError("evaluation_names must contain Verilog identifiers")
        def optional_file(name: str) -> Path | None:
            value = raw.get(name)
            result = resolve_config_path(base, value) if value else None
            if result and not result.is_file():
                raise ValueError(f"{name} does not exist: {result}")
            return result
        spec_file = optional_file("spec_file")
        if spec_file is None:
            raise ValueError("spec_file is required")
        for name in ("clock", "reset"):
            value = raw.get(name)
            if value is not None and not IDENTIFIER.fullmatch(value):
                raise ValueError(f"{name} must be a Verilog identifier")
        return cls(path, root, sources, include_dirs, top, spec_file, requirements,
                   optional_file("oracle_file"), optional_file("environment_file"),
                   optional_file("evaluation_file"), eval_names,
                   raw.get("planner", {}), raw.get("jasper", {}), raw.get("lint", {}),
                   raw.get("clock"), raw.get("reset"))

    def source_snapshot(self) -> str:
        return digest(json.dumps([(str(p), digest((self.design_root / p).read_bytes()))
                                  for p in self.sources]).encode())

    def requirement(self, identifier: str) -> Requirement:
        for requirement in self.requirements:
            if requirement.id == identifier:
                return requirement
        raise ValueError(f"Unknown requirement_id: {identifier}")


@dataclass(frozen=True)
class Proposal:
    requirement_id: str
    failure: str
    source_file: str
    old: str
    new: str
    expected_observation: str
    rationale: str

    @classmethod
    def parse(cls, data: dict[str, Any], config: Config) -> "Proposal":
        if not isinstance(data, dict):
            raise ValueError("Proposal must be a JSON object")
        fields = cls.__dataclass_fields__
        if any(field not in data or not isinstance(data[field], str) for field in fields):
            raise ValueError("Proposal needs string fields: " + ", ".join(fields))
        proposal = cls(**{field: data[field] for field in fields})
        config.requirement(proposal.requirement_id)
        if confined_relative(proposal.source_file) not in config.sources:
            raise ValueError("source_file must be one of the configured source files")
        if (not proposal.old or proposal.old == proposal.new
                or len(proposal.old) > 4000 or len(proposal.new) > 4000):
            raise ValueError("Edit must replace a nonempty, distinct, at most 4000-character span")
        if not proposal.failure.strip() or not proposal.expected_observation.strip():
            raise ValueError("Failure and expected_observation are required")
        return proposal

    def key(self) -> str:
        return digest(json.dumps([[self.source_file, self.old, self.new]]).encode())


SCENARIO_CATEGORIES = ("timing", "state_sequence", "interaction")


@dataclass(frozen=True)
class Edit:
    source_file: str
    old: str
    new: str

    @classmethod
    def parse(cls, data: dict[str, Any], config: Config) -> "Edit":
        if not isinstance(data, dict) or any(not isinstance(data.get(name), str)
                                             for name in ("source_file", "old", "new")):
            raise ValueError("Each scenario edit needs string source_file, old, and new fields")
        edit = cls(**{name: data[name] for name in ("source_file", "old", "new")})
        if confined_relative(edit.source_file) not in config.sources:
            raise ValueError("Scenario edit source_file must be a configured RTL source")
        if not edit.old or edit.old == edit.new or max(len(edit.old), len(edit.new)) > 4000:
            raise ValueError("Scenario edit needs distinct, nonempty spans of at most 4000 characters")
        return edit


@dataclass(frozen=True)
class ScenarioPlan:
    category: str
    requirement_id: str
    spec_excerpt: str
    trigger_sequence: tuple[str, ...]
    golden_behavior: str
    faulty_behavior: str
    observation: str
    witness: dict[str, Any] | None = None

    @classmethod
    def parse(cls, data: dict[str, Any], config: Config,
              requested_category: str | None = None) -> "ScenarioPlan":
        if not isinstance(data, dict):
            raise ValueError("Scenario must be a JSON object")
        fields = ("category", "requirement_id", "spec_excerpt", "golden_behavior",
                  "faulty_behavior", "observation")
        if any(not isinstance(data.get(name), str) or not data[name].strip() for name in fields):
            raise ValueError("Scenario needs nonempty category, requirement_id, spec_excerpt, "
                             "golden_behavior, faulty_behavior, and observation")
        category = data["category"]
        if category not in SCENARIO_CATEGORIES or (requested_category and category != requested_category):
            raise ValueError(f"Scenario category must be {requested_category or SCENARIO_CATEGORIES}")
        config.requirement(data["requirement_id"])
        sequence = data.get("trigger_sequence")
        if (not isinstance(sequence, (list, tuple)) or len(sequence) < 2 or len(sequence) > 12
                or any(not isinstance(step, str) or not step.strip() for step in sequence)):
            raise ValueError("Scenario trigger_sequence needs 2 to 12 nonempty steps")
        # A quote from the actual spec limits unsupported LLM-generated fault goals.
        normalize = lambda value: " ".join(value.split())
        excerpt = normalize(data["spec_excerpt"])
        if len(excerpt) < 12 or len(excerpt) > 500 or excerpt not in normalize(read_spec(config.spec_file)):
            raise ValueError("Scenario spec_excerpt must quote the supplied specification")
        witness = data.get("witness")
        if witness is not None and not isinstance(witness, dict):
            raise ValueError("Scenario witness must be a JSON object")
        return cls(category, data["requirement_id"], data["spec_excerpt"], tuple(sequence),
                   data["golden_behavior"], data["faulty_behavior"], data["observation"],
                   witness)


@dataclass(frozen=True)
class ScenarioMutation:
    scenario: ScenarioPlan
    edits: tuple[Edit, ...]

    @classmethod
    def parse(cls, data: dict[str, Any], config: Config,
              requested_category: str | None = None) -> "ScenarioMutation":
        if not isinstance(data, dict):
            raise ValueError("Scenario mutation must be a JSON object")
        scenario = ScenarioPlan.parse(data.get("scenario"), config, requested_category)
        raw_edits = data.get("edits")
        if not isinstance(raw_edits, list) or not 1 <= len(raw_edits) <= 3:
            raise ValueError("One scenario needs 1 to 3 related RTL edits")
        edits = tuple(Edit.parse(item, config) for item in raw_edits)
        return cls(scenario, edits)

    def key(self) -> str:
        return digest(json.dumps(sorted((item.source_file, item.old, item.new)
                                        for item in self.edits)).encode())


def stage_sources(config: Config, destination: Path) -> None:
    """Copy configured RTL and include trees; never edit the golden source directory."""
    rtl = destination / "rtl"
    rtl.mkdir(parents=True, exist_ok=False)
    for directory in config.include_dirs:
        source_dir = config.design_root / directory
        target_dir = rtl / directory
        for source in source_dir.rglob("*"):
            if source.is_symlink() or not source.is_file():
                continue
            relative = source.relative_to(source_dir)
            target = target_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for source in config.sources:
        target = rtl / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.design_root / source, target)


def apply_proposal(config: Config, proposal: Proposal, destination: Path) -> dict[str, Any]:
    stage_sources(config, destination)
    file = destination / "rtl" / confined_relative(proposal.source_file)
    original = file.read_text()
    count = original.count(proposal.old)
    if count != 1:
        raise ValueError(f"Edit anchor must occur exactly once in {proposal.source_file}; found {count}")
    edited = original.replace(proposal.old, proposal.new, 1)
    file.write_text(edited)
    return {
        "source_file": proposal.source_file,
        "old_sha256": digest(original.encode()),
        "new_sha256": digest(edited.encode()),
        "line": original[:original.index(proposal.old)].count("\n") + 1,
        "source_snapshot": config.source_snapshot(),
    }


def apply_scenario(config: Config, mutation: ScenarioMutation, destination: Path) -> dict[str, Any]:
    """Validate all anchors against golden copies, then apply related edits atomically."""
    return apply_edits(config, mutation.edits, destination)


def apply_edits(config: Config, edits: tuple[Edit, ...], destination: Path) -> dict[str, Any]:
    """Apply a checked set of exact edits to an isolated golden copy."""
    stage_sources(config, destination)
    changes: dict[Path, list[tuple[int, int, Edit]]] = {}
    original_text: dict[Path, str] = {}
    for edit in edits:
        file = destination / "rtl" / confined_relative(edit.source_file)
        original = original_text.setdefault(file, file.read_text())
        count = original.count(edit.old)
        if count != 1:
            raise ValueError(f"Edit anchor must occur exactly once in {edit.source_file}; found {count}")
        start = original.index(edit.old)
        changes.setdefault(file, []).append((start, start + len(edit.old), edit))
    for file, spans in changes.items():
        ordered = sorted(spans, key=lambda span: span[0])
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"Scenario edits overlap in {file.name}")
    details = []
    for file, spans in changes.items():
        original = original_text[file]
        edited = original
        for start, end, edit in sorted(spans, key=lambda span: span[0], reverse=True):
            edited = edited[:start] + edit.new + edited[end:]
            details.append({"source_file": edit.source_file,
                            "line": original[:start].count("\n") + 1,
                            "old_sha256": digest(edit.old.encode()),
                            "new_sha256": digest(edit.new.encode())})
        file.write_text(edited)
    return {"source_snapshot": config.source_snapshot(),
            "changes": sorted(details, key=lambda item: (item["source_file"], item["line"]))}


def read_spec(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError:
            if not shutil.which("pdftotext"):
                raise RuntimeError("Install PyMuPDF or pdftotext to read PDF specifications") from None
            result = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                    capture_output=True, text=True, check=True)
            content = result.stdout
        else:
            with fitz.open(path) as document:
                content = "\n".join(page.get_text() for page in document)
    else:
        content = path.read_text(errors="replace")
    if not content.strip():
        raise ValueError(f"No text extracted from specification: {path}")
    return content
