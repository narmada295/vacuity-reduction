"""Direct, checkpointed assertion generation for the base/SFT/KTO comparison.

Run on the allocated GPU workstation in the existing vacuity-qwen environment.
This script performs no filtering, repair, formal verification, or model training.
"""

import argparse
from collections import Counter
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

if __package__:
    from .prompts import ONE_SHOT_PROMPT_VERSION, one_shot_system_prompt
else:
    from prompts import ONE_SHOT_PROMPT_VERSION, one_shot_system_prompt


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_text_atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path, value):
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def saved_attempts(expected, fingerprint, label):
    saved = []
    for prompt, sample, path in expected:
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        identity = (row.get("task_id"), row.get("case_id"), row.get("target_signal"),
                    row.get("sample"), row.get("label"), row.get("generation_config_fingerprint"))
        if identity != (prompt["task_id"], prompt["case_id"], prompt["target_signal"], sample, label, fingerprint):
            raise ValueError("Saved attempt differs from the assigned task or generation configuration: " + str(path))
        if row.get("status") not in ("generated", "generation_error"):
            raise ValueError("Saved attempt has an unsupported status: " + str(path))
        saved.append(row)
    return saved


def save_progress(out, expected, fingerprint, label, status, **metrics):
    # Individual attempt files are authoritative, including after a hard kill
    # between saving an attempt and refreshing these derived views.
    saved = saved_attempts(expected, fingerprint, label)
    write_text_atomic(out / "generations.jsonl", "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in saved))
    summary = {"status": status, "label": label, "expected_attempts": len(expected),
               "saved_attempts": len(saved), "remaining_attempts": len(expected) - len(saved),
               "outcomes": dict(Counter(r["status"] for r in saved)),
               "formal_evaluation": "not run; preserve and check every saved output"}
    summary.update(metrics)
    write_json(out / "summary.json", summary)
    return saved


def sample_seed(task_id, sample, seed):
    task_seed = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)
    return (seed + task_seed + sample) % (2 ** 32)


def inference_messages(messages, assertion_cap=None):
    if assertion_cap is None:
        return messages
    transformed = [dict(m) for m in messages]
    transformed[0]["content"] = one_shot_system_prompt(assertion_cap)
    tail = "Generate one assertion for this signal."
    if not transformed[1]["content"].endswith(tail):
        raise ValueError("Unexpected source user prompt suffix")
    transformed[1]["content"] = transformed[1]["content"][:-len(tail)] + (
        "Generate up to {} distinct assertions for this signal in one response. "
        "Cover different specified behaviors where possible. Return fewer if appropriate; "
        "do not repeat assertions or invent requirements to reach the limit."
    ).format(assertion_cap)
    return transformed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--label", choices=("base", "sft", "kto"), required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--training-config", type=Path,
                    default=Path("outputs/qwen3_4b_sft_v2_train/run_config.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples-per-prompt", type=int)
    ap.add_argument("--assertion-cap", type=int, help="One-shot mode: request up to this many assertions in one response per signal")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-download", action="store_true", help="Otherwise use cached model/tokenizer files only")
    args = ap.parse_args()
    if args.samples_per_prompt is None:
        args.samples_per_prompt = 1 if args.assertion_cap is not None else 5
    if args.assertion_cap is not None and (args.assertion_cap < 1 or args.samples_per_prompt != 1):
        ap.error("One-shot mode requires a positive assertion cap and exactly one response per signal")
    if bool(args.adapter) != (args.label != "base"):
        ap.error("Specify --adapter for sft/kto, and omit it for base")
    if args.samples_per_prompt < 1 or args.max_new_tokens < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        ap.error("Invalid generation budget or sampling settings")
    training = json.loads(args.training_config.read_text())
    prompts = [json.loads(line) for line in args.prompts.open() if line.strip()]
    ids = [p["task_id"] for p in prompts]
    if not prompts or len(set(ids)) != len(ids):
        raise ValueError("Expected a nonempty set of unique input contexts")
    for prompt in prompts:
        if set(prompt) != {"task_id", "case_id", "target_signal", "messages"}:
            raise ValueError("Unexpected task fields; use prepared inference inputs")
        if [m["role"] for m in prompt["messages"]] != ["system", "user"]:
            raise ValueError("Inference must not receive an assistant/reference response")
        inference_messages(prompt["messages"], args.assertion_cap)
    if args.assertion_cap is not None:
        signals = [(p["case_id"], p["target_signal"]) for p in prompts]
        if len(signals) != len(set(signals)):
            raise ValueError("One-shot mode requires exactly one input context per design/signal")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".generation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expected one allocated GPU slice; check gpustat and CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("The established configuration requires BF16 support")
    adapter_files = {}
    if args.adapter:
        cfg = json.loads((args.adapter / "adapter_config.json").read_text())
        if cfg["base_model_name_or_path"] != training["model"]:
            raise ValueError("Adapter base model differs from the SFT training model")
        weights = [p for p in args.adapter.glob("adapter_model.*") if p.suffix in (".safetensors", ".bin")]
        if len(weights) != 1:
            raise ValueError("Expected one adapter weight file")
        adapter_files = {p.name: file_hash(p) for p in [args.adapter / "adapter_config.json", weights[0]]}
    settings = {"label": args.label, "model": training["model"], "model_revision": training["model_revision"],
                "chat_template_sha256": training["chat_template_sha256"],
                "training_config_sha256": file_hash(args.training_config),
                "prompts_sha256": file_hash(args.prompts), "prompt_count": len(prompts),
                "adapter": str(args.adapter) if args.adapter else None, "adapter_files": adapter_files,
                "samples_per_prompt": args.samples_per_prompt, "max_new_tokens": args.max_new_tokens,
                "assertion_cap": args.assertion_cap,
                "prompt_version": ONE_SHOT_PROMPT_VERSION if args.assertion_cap is not None else "original_sft_single_assertion",
                "one_shot_system_prompt_sha256": (hashlib.sha256(one_shot_system_prompt(args.assertion_cap).encode()).hexdigest()
                                                  if args.assertion_cap is not None else None),
                "generation_mode": "one_shot_per_signal" if args.assertion_cap is not None else "independent_single_assertion_samples",
                "temperature": args.temperature, "top_p": args.top_p, "top_k": 0,
                "repetition_penalty": 1.0, "seed": args.seed, "quantization": "4-bit NF4, double quantization, BF16 compute",
                "script_sha256": file_hash(Path(__file__)), "truncation": False,
                "packages": {name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")}}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    config_path = args.out / "generation_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Existing run has different model, prompts, adapter, or generation settings")
    else:
        write_json(config_path, dict(settings, fingerprint=fingerprint))
    attempts = args.out / "attempts"
    attempts.mkdir(exist_ok=True)
    expected = [(p, n, attempts / (p["task_id"] + "__sample_" + str(n) + ".json"))
                for p in prompts for n in range(args.samples_per_prompt)]
    saved = saved_attempts(expected, fingerprint, args.label)
    if len(saved) == len(expected):
        save_progress(args.out, expected, fingerprint, args.label, "complete")
        print("All generation attempts already saved; no GPU model loaded.", flush=True)
        return
    save_progress(args.out, expected, fingerprint, args.label, "loading_model")
    local = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(training["model"], revision=training["model_revision"], local_files_only=local)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != training["chat_template_sha256"]:
        raise ValueError("Tokenizer chat template differs from SFT")
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(training["model"], revision=training["model_revision"],
        local_files_only=local, quantization_config=quantization, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False, local_files_only=local)
    model.eval()
    model.config.use_cache = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    status = "running"
    saved_count = sum(path.exists() for _, _, path in expected)
    def progress():
        save_progress(args.out, expected, fingerprint, args.label, status,
            elapsed_seconds_this_invocation=time.monotonic() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)

    try:
        progress()
        for prompt, sample, path in expected:
            if path.exists():
                continue
            seed = sample_seed(prompt["task_id"], sample, args.seed)
            set_seed(seed)
            messages = inference_messages(prompt["messages"], args.assertion_cap)
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, truncation=False, return_dict=False)
            if len(token_ids) + args.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("Full prompt and generation budget exceed context length; no truncation performed")
            inputs = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
            record = {"task_id": prompt["task_id"], "case_id": prompt["case_id"],
                      "target_signal": prompt["target_signal"], "label": args.label,
                      "sample": sample, "seed": seed, "prompt_tokens": len(token_ids),
                      "assertion_cap": args.assertion_cap,
                      "effective_messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                      "generation_config_fingerprint": fingerprint}
            tick = time.monotonic()
            try:
                with torch.inference_mode():
                    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0,
                        repetition_penalty=1.0, num_beams=1, num_return_sequences=1,
                        max_new_tokens=args.max_new_tokens, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                generated = output[0, len(token_ids):].tolist()
                eos = model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                record.update(status="generated", response=tokenizer.decode(generated, skip_special_tokens=True),
                    new_tokens=len(generated), length_limited=len(generated) >= args.max_new_tokens and generated[-1] not in eos,
                    elapsed_seconds=time.monotonic() - tick)
                del output
            except Exception as error:
                record.update(status="generation_error", error=type(error).__name__ + ": " + str(error),
                              elapsed_seconds=time.monotonic() - tick)
                write_json(path, record)
                raise
            write_json(path, record)
            saved_count += 1
            progress()
            del inputs
            print(args.label, str(saved_count) + "/" + str(len(expected)), prompt["case_id"],
                  prompt["target_signal"], "sample", sample, "tokens", record["new_tokens"],
                  "saved; signals left", len(expected) - saved_count, flush=True)
        status = "complete"
    except BaseException:
        status = "failed_or_interrupted"
        raise
    finally:
        progress()


if __name__ == "__main__":
    main()
