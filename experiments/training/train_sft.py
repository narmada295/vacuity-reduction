"""Remote Qwen QLoRA SFT with full inputs, explicit output masks and checkpoints.

Default mode is a two-update memory/checkpoint smoke test on training examples.
Run as a standalone script beside profile_sft.py in ~/vacuity/scripts/.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import time

try:
    from .profile_sft import file_hash, validate_designs
except ImportError:
    from profile_sft import file_hash, validate_designs


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def encode_record(tokenizer, record, expected):
    messages = record["messages"]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("Unexpected message roles: " + record["id"])
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=False, truncation=False, return_dict=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
        add_generation_prompt=True, truncation=False, return_dict=False)
    if ids[:len(prompt)] != prompt or len(ids) <= len(prompt):
        raise ValueError("Cannot safely identify assistant boundary: " + record["id"])
    if (len(ids), len(prompt), len(ids) - len(prompt)) != (
        expected["total_tokens"], expected["prompt_tokens"], expected["assistant_tokens"]
    ):
        raise ValueError("Tokenizer lengths changed since profiling: " + record["id"])
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt) + ids[len(prompt):]}


def load_inputs(data_dir, profile_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    report = json.loads((profile_dir / "summary.json").read_text())
    if report.get("truncation") is not False:
        raise ValueError("Expected a full-input token profile")
    for split in ("train", "validation"):
        name = split + ".jsonl"
        digest = file_hash(data_dir / name)
        if digest != manifest["files"][name]["sha256"] or digest != report["source_files"][name]:
            raise ValueError("Dataset differs from manifest/token profile: " + name)
    lengths = {}
    for line in (profile_dir / "lengths.jsonl").open():
        item = json.loads(line)
        if item["id"] in lengths:
            raise ValueError("Duplicate example in length profile")
        lengths[item["id"]] = item
    for split in ("train", "validation"):
        items = [item for item in lengths.values() if item["split"] == split]
        if len(items) != manifest["splits"][split]["examples"]:
            raise ValueError("Token profile count mismatch: " + split)
        if max(item["total_tokens"] for item in items) != report["splits"][split]["max"]:
            raise ValueError("Token profile maximum mismatch: " + split)
    return manifest, report, lengths


def smoke_ids(lengths):
    train = [item for item in lengths.values() if item["split"] == "train"]
    validation = [item for item in lengths.values() if item["split"] == "validation"]
    longest = max(train, key=lambda item: item["total_tokens"])
    largest_output = max(train, key=lambda item: item["assistant_tokens"])
    selected = {longest["id"], largest_output["id"]}
    if len(selected) < 2 and len(train) > 1:
        selected.add(next(item["id"] for item in sorted(train, key=lambda item: -item["total_tokens"])
                          if item["id"] not in selected))
    return {"train": selected,
            "validation": {max(validation, key=lambda item: item["total_tokens"])["id"]}}


def checkpoint_complete(path, fingerprint):
    try:
        marker = json.loads((path / "checkpoint_complete.json").read_text())
        state = json.loads((path / "trainer_state.json").read_text())
        if marker["run_fingerprint"] != fingerprint or marker["global_step"] != state["global_step"]:
            return False
        required = ["adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
        if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
            return False
        return any((path / name).is_file() and (path / name).stat().st_size
                   for name in ("adapter_model.safetensors", "adapter_model.bin"))
    except (OSError, ValueError, KeyError):
        return False


def resolve_resume(output, requested, fingerprint):
    if requested is None:
        return None
    if requested == "auto":
        candidates = [p for p in output.glob("checkpoint-*")
                      if p.name.removeprefix("checkpoint-").isdigit() and checkpoint_complete(p, fingerprint)]
        if not candidates:
            raise ValueError("No complete resumable checkpoint found in " + str(output))
        return str(max(candidates, key=lambda p: int(p.name.split("-")[-1])))
    path = Path(requested).expanduser()
    if path.parent.resolve() != output.resolve() or not checkpoint_complete(path, fingerprint):
        raise ValueError("Checkpoint is incomplete, belongs to another run, or is outside --out")
    return str(path)


def make_checkpoint_callback(base_class, fingerprint, interval_seconds, step_interval, stop_state):
    class Checkpoints(base_class):
        def __init__(self):
            self.last_save = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            due = time.monotonic() - self.last_save >= interval_seconds
            if due or state.global_step % step_interval == 0 or stop_state["stop_requested"]:
                control.should_save = True
            if stop_state["stop_requested"]:
                control.should_training_stop = True
                # Save promptly instead of starting an expensive epoch evaluation.
                control.should_evaluate = False
            return control

        def on_save(self, args, state, control, **kwargs):
            folder = Path(args.output_dir) / ("checkpoint-" + str(state.global_step))
            state_path = folder / "trainer_state.json"
            if not state_path.exists():
                raise RuntimeError("Trainer did not write checkpoint state")
            write_json(folder / "checkpoint_complete.json", {
                "run_fingerprint": fingerprint, "global_step": state.global_step,
                "saved_utc": datetime.now(timezone.utc).isoformat(),
            })
            if not checkpoint_complete(folder, fingerprint):
                (folder / "checkpoint_complete.json").unlink()
                raise RuntimeError("Checkpoint is missing adapter/optimizer/scheduler/RNG state")
            self.last_save = time.monotonic()
            print("Complete resumable checkpoint:", folder, flush=True)
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if stop_state["stop_requested"]:
                control.should_evaluate = False
            return control

    return Checkpoints()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--profile-dir", type=Path, default=Path("logs/token_lengths"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-minutes", type=float, default=10)
    parser.add_argument("--save-steps", type=int, default=25)
    args = parser.parse_args()
    if min(args.epochs, args.gradient_accumulation, args.save_minutes, args.save_steps) <= 0:
        parser.error("Epochs, accumulation and checkpoint intervals must be positive")
    output = args.out or Path("outputs/qwen3_4b_sft_" + args.mode)
    if output.exists() and not args.resume:
        raise ValueError("Output already exists. Use --resume or a fresh --out")
    if args.resume and not output.exists():
        raise ValueError("Cannot resume a missing output directory")
    manifest, profile, lengths = load_inputs(args.data_dir, args.profile_dir)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expected one active allocated GPU slice; check gpustat and CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This configuration requires BF16 support")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(profile["model"], revision=profile["model_revision"])
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if template_hash != profile["chat_template_sha256"]:
        raise ValueError("Chat template differs from the profiled template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selection = smoke_ids(lengths) if args.mode == "smoke" else None
    datasets = {}
    selected_details = {}
    design_counts = {}
    for split in ("train", "validation"):
        dataset = load_dataset("json", data_files=str(args.data_dir / (split + ".jsonl")), split="train")
        if len(dataset) != manifest["splits"][split]["examples"]:
            raise ValueError("Source count changed")
        design_counts[split] = validate_designs(manifest, split, dataset["case_id"])
        if selection:
            dataset = dataset.filter(lambda record: record["id"] in selection[split])
        expected_ids = {record["id"] for record in dataset}
        for record in dataset:
            expected = lengths[record["id"]]
            if record["case_id"] not in manifest["split_plan"][split] or expected["split"] != split:
                raise ValueError("Split leakage")
            if record["case_id"] in manifest["split_plan"].get("excluded", {}):
                raise ValueError("Excluded design in training inputs")
        selected_details[split] = {
            "examples": len(dataset),
            "max_tokens": max(lengths[ident]["total_tokens"] for ident in expected_ids),
            "example_ids": sorted(expected_ids) if selection else "all examples in the assigned split",
        }
        def encode(record):
            return encode_record(tokenizer, record, lengths[record["id"]])
        datasets[split] = dataset.map(encode, remove_columns=dataset.column_names,
                                      desc="Masking prompt tokens: " + split)
    accumulation = 1 if selection else args.gradient_accumulation
    updates = 2 if selection else math.ceil(len(datasets["train"]) / accumulation) * args.epochs
    settings = {
        "script_sha256": file_hash(Path(__file__)),
        "mode": args.mode, "model": profile["model"], "model_revision": profile["model_revision"],
        "chat_template_sha256": template_hash,
        "data_sha256": profile["source_files"],
        "lora": {"r": 16, "alpha": 32, "dropout": .05, "target_modules": "all-linear"},
        "batch_size": 1, "gradient_accumulation": accumulation,
        "learning_rate": args.learning_rate, "epochs": args.epochs, "max_steps": 2 if selection else -1,
        "warmup_steps": 0 if selection else math.ceil(updates * .03),
        "seed": 42, "loss": "assistant-only token cross-entropy; prompt labels=-100",
        "truncation": False, "packing": False,
        "selection": selected_details,
        "design_counts": design_counts,
        "design_quotas": manifest["split_plan"].get("design_quotas"),
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes")},
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    resume = resolve_resume(output, args.resume, fingerprint)
    if args.resume:
        old = json.loads((output / "run_config.json").read_text())
        if old["run_fingerprint"] != fingerprint:
            raise ValueError("Run configuration, dataset, model or package versions changed since checkpoint")
    else:
        output.mkdir(parents=True)
        write_json(output / "run_config.json", dict(settings, run_fingerprint=fingerprint))
    print("Run configuration:", json.dumps(settings, indent=2), flush=True)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        profile["model"], revision=profile["model_revision"],
        quantization_config=quantization, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    if profile["minimum_sequence_length_to_keep_all_examples"] > model.config.max_position_embeddings:
        raise ValueError("Dataset exceeds the model context; inputs were not truncated")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def collate(features):
        if len(features) != 1:
            raise ValueError("This memory-tested configuration requires per-device batch size 1")
        row = features[0]
        if not any(label != -100 for label in row["labels"]):
            raise ValueError("No assistant tokens remain")
        return {key: torch.tensor([row[key]], dtype=torch.long)
                for key in ("input_ids", "attention_mask", "labels")}

    training = SFTConfig(
        output_dir=str(output), num_train_epochs=args.epochs,
        max_steps=2 if selection else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine",
        warmup_steps=settings["warmup_steps"], weight_decay=.01, max_grad_norm=1.0,
        bf16=True, fp16=False, optim="adamw_torch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=None, packing=False, padding_free=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        # Masks are explicitly constructed and tested above, not inferred by a template patch.
        assistant_only_loss=False, completion_only_loss=False,
        loss_type="nll", remove_unused_columns=False,
        prediction_loss_only=True,
        eval_strategy="steps" if selection else "epoch", eval_steps=1 if selection else None,
        save_strategy="steps" if selection else "epoch", save_steps=1 if selection else args.save_steps,
        save_total_limit=3, save_only_model=False,
        load_best_model_at_end=not bool(selection), metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1 if selection else 5, report_to=[], push_to_hub=False,
        seed=42, data_seed=42, dataloader_num_workers=0,
    )
    stop = {"stop_requested": False}
    previous_handlers = {}
    def request_stop(signum, frame):
        stop["stop_requested"] = True
        print("Stop requested; saving a checkpoint after the current optimizer update.", flush=True)
    for number in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[number] = signal.signal(number, request_stop)
    callback = make_checkpoint_callback(TrainerCallback, fingerprint,
        args.save_minutes * 60, 1 if selection else args.save_steps, stop)
    trainer = SFTTrainer(model=model, args=training, train_dataset=datasets["train"],
        eval_dataset=datasets["validation"], processing_class=tokenizer,
        data_collator=collate, callbacks=[callback])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        checkpoints = [p for p in output.glob("checkpoint-*") if checkpoint_complete(p, fingerprint)]
        if not checkpoints:
            raise RuntimeError("Run ended without a complete resumable checkpoint")
        if not stop["stop_requested"]:
            trainer.save_model(str(output / "final_adapter"))
            tokenizer.save_pretrained(output / "final_adapter")
        summary = {
            "status": "stopped_with_checkpoint" if stop["stop_requested"] else "PASS",
            "mode": args.mode, "global_step": trainer.state.global_step,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "available_device_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "complete_checkpoints": sorted(str(p) for p in checkpoints),
            "metrics": result.metrics, "selection": selected_details,
        }
        write_json(output / "result.json", summary)
        print("\nRUN RESULT\n" + json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        write_json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3})
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
