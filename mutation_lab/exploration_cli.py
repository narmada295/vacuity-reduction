"""CLI registration isolated from the legacy campaign defaults."""

import argparse
from pathlib import Path


def register_exploration(subparsers) -> None:
    comparison = subparsers.add_parser("compare-probes", help="Compare legacy and targeted probes on saved explore candidates")
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--max-probes", type=int, default=48)
    command = subparsers.add_parser("explore-campaign", help="Full-context family exploration with adaptive LLM probes")
    command.add_argument("--spec", type=Path, required=True)
    command.add_argument("--rtl", type=Path, nargs="+", required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--count", type=int, default=0,
                         help="Accepted-mutant target; 0 (default) explores until campaign budgets are reached")
    command.add_argument("--max-attempts", type=int, default=100,
                         help="Maximum individual fault plans to implement")
    command.add_argument("--plans-per-batch", type=int, default=8,
                         help="Maximum ideas per planning call; NOT a total or per-region limit")
    command.add_argument("--max-planning-calls", type=int, default=100)
    command.add_argument("--max-llm-calls", type=int, default=600,
                         help="Campaign-wide attempted call budget, including retries and all stages")
    command.add_argument("--region-rounds", type=int, default=0,
                         help="Planning rounds per focus; 0 allows repeated rounds within global budgets")
    command.add_argument("--empty-rounds", type=int, default=3,
                         help="Pause a focus after this many batches without new valid ideas; 0 disables")
    command.add_argument("--max-repairs", type=int, default=1)
    command.add_argument("--response-retries", type=int, default=1)
    command.add_argument("--probes-per-round", type=int, default=6)
    command.add_argument("--probe-refinements", type=int, default=2)
    command.add_argument("--max-probe-cycles", type=int, default=512)
    command.add_argument("--replay-probes", type=int, default=8,
                         help="Maximum archived LLM-discovered probes replayed per mutant; 0 disables")
    command.add_argument("--cross-region", action=argparse.BooleanOptionalAction, default=True)
    command.add_argument("--global-exploration", action=argparse.BooleanOptionalAction, default=False)
    command.add_argument("--analysis-cache", type=Path,
                         help="Optional reusable analysis JSON; invalidated by input/model/implementation changes")
    command.add_argument("--synthesis-check", action="store_true",
                         help="Require Yosys frontend, hierarchy, proc, and check -assert passes")
    command.add_argument("--design-root", type=Path)
    command.add_argument("--include-dir", type=Path, action="append", default=[])
    command.add_argument("--top")
    command.add_argument("--clock", help="One rising-edge clock; inferred, or omitted for combinational RTL")
    command.add_argument("--reset")
    command.add_argument("--reset-active", choices=("auto", "0", "1"), default="auto",
                         help="Explicit polarity recommended; auto uses reset-name conventions")
    command.add_argument("--jg-mode", choices=("ssh", "local"), default="ssh")
    command.add_argument("--jg-timeout", type=int, default=180)
    command.add_argument("--jg-executable", default="jg")
    command.add_argument("--env-file", type=Path, default=Path(".env"))
    command.add_argument("--endpoint", default="https://api.tokenfactory.nebius.com/v1/chat/completions")
    command.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    command.add_argument("--key-env", default="NEBIUS_API_KEY1")
    command.add_argument("--max-context-chars", type=int, default=250000)
    command.add_argument("--planner-max-tokens", type=int, default=16384)
    command.add_argument("--planner-extra-body", type=Path)
    command.add_argument("--planner-timeout", type=int, default=300)
    command.add_argument("--lint-arg", action="append", default=[])
    command.set_defaults(focus_rtl=[])


def validate_arguments(args, parser) -> None:
    for field in ("max_attempts", "plans_per_batch", "max_planning_calls", "max_llm_calls",
                  "probes_per_round", "max_probe_cycles", "planner_max_tokens",
                  "planner_timeout", "max_context_chars"):
        if getattr(args, field) < 1:
            parser.error("--" + field.replace("_", "-") + " must be positive")
    for field in ("count", "region_rounds", "empty_rounds", "max_repairs", "response_retries",
                  "probe_refinements", "replay_probes"):
        if getattr(args, field) < 0:
            parser.error("--" + field.replace("_", "-") + " must be nonnegative")
