"""Run SFT-initialized QLoRA KTO: preflight, memory smoke test, then training.

Uses TRL 1.17.1's KTOTrainer, including its frozen copy of an existing SFT
adapter. Reference log probabilities are cached batch by batch for recovery.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import signal
import time


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TEMPLATE_SHA = "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"
DATASET_MANIFEST_SHA = "d92318ae23eec9e8dec7ed3b101908e2b6a7504b5cefec5eeee5953b2ceef67e"
SFT_DATA = {"train.jsonl": "81da6f5f0ccf0d071b4c2b1d212b8b7dc7edf0eb6d326036673d2d76ff7852cb",
            "validation.jsonl": "0da090d2a90f3c93f92df97c48922ab2321614b4827c9c23ba86ce797becd139"}
CATEGORIES = {"proven_non_vacuous_implication", "proven_non_implication", "proven_vacuous_implication"}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_record(record, example, split, plan):
    if example["split"] != split or example["case_id"] not in plan[split]:
        raise ValueError("An example crosses the saved design split")
    if set(record) != {"prompt", "completion", "label"} or type(record["label"]) is not bool:
        raise ValueError("Expected conversational KTO prompt/completion/Boolean label")
    if ([m["role"] for m in record["prompt"]] != ["system", "user"]
            or [m["role"] for m in record["completion"]] != ["assistant"]):
        raise ValueError("Unexpected conversation roles")
    if (record["label"] is not example["label"] or example["category"] not in CATEGORIES
            or example["assertion_status"] != "proven"):
        raise ValueError("Label/category differs from the canonical proof record")
    if record["completion"][0]["content"] != "```systemverilog\n" + example["assertion"] + "\n```":
        raise ValueError("Completion differs from the saved assertion")
    positive = example["category"] != "proven_vacuous_implication"
    if record["label"] is not positive:
        raise ValueError("Desirability conflicts with category")
    expected_vacuous = None if example["category"] == "proven_non_implication" else not positive
    if example["vacuous"] is not expected_vacuous:
        raise ValueError("Non-implication/vacuity scope was changed")


def load_data(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("version") != 2 or manifest.get("test_examples_exported") != 0:
        raise ValueError("Use the v2 KTO dataset that includes non-implications")
    plan = manifest["split_plan"]
    assigned = [cid for split in ("train", "validation", "test") for cid in plan[split]]
    if (len(assigned) != len(set(assigned)) or len(plan["train"]) != 25
            or len(plan["validation"]) != 5 or len(plan["test"]) != 5):
        raise ValueError("Expected disjoint 25/5/5 design assignments")
    for name, info in manifest["files"].items():
        if file_hash(data_dir / name) != info["sha256"]:
            raise ValueError("Dataset hash mismatch: " + name)
    records, metadata = {s: [] for s in ("train", "validation")}, {}
    seen = set()
    with ExitStack() as stack:
        handles = {s: stack.enter_context((data_dir / (s + ".jsonl")).open()) for s in records}
        for line in (data_dir / "examples.jsonl").open():
            example = json.loads(line)
            split = example["split"]
            if split not in records or example["id"] in seen:
                raise ValueError("Unexpected split or duplicate example ID")
            seen.add(example["id"])
            record = json.loads(next(handles[split]))
            validate_record(record, example, split, plan)
            index = len(records[split])
            metadata[(split, index)] = {key: example[key] for key in
                ("id", "case_id", "category", "label")}
            records[split].append(record)
        if any(next(handle, None) is not None for handle in handles.values()):
            raise ValueError("Extra trainer rows")
    for split, rows in records.items():
        labels = Counter("desirable" if r["label"] else "undesirable" for r in rows)
        kinds = Counter(metadata[(split, n)]["category"] for n in range(len(rows)))
        designs = {metadata[(split, n)]["case_id"] for n in range(len(rows))}
        if (dict(labels) != manifest["splits"][split]["labels"]
                or dict(kinds) != manifest["splits"][split]["categories"]
                or designs != set(plan[split])):
            raise ValueError("Dataset counts or design membership differ from manifest")
    return manifest, records, metadata


def class_weights(labels):
    positive, negative = labels["desirable"], labels["undesirable"]
    if min(positive, negative) <= 0:
        raise ValueError("Both KTO feedback classes are required")
    return {"desirable": 1.0, "undesirable": positive / negative}


def complete_checkpoint(path, fingerprint):
    try:
        marker = json.loads((path / "checkpoint_complete.json").read_text())
        state = json.loads((path / "trainer_state.json").read_text())
        if marker["run_fingerprint"] != fingerprint or marker["global_step"] != state["global_step"]:
            return False
        required = ("adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
        if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
            return False
        if not any((path / name).is_file() and (path / name).stat().st_size for name in
                   ("adapter_model.safetensors", "adapter_model.bin")):
            return False
        return bool(marker.get("frozen_reference_sha256") and marker.get("policy_sha256"))
    except (OSError, ValueError, KeyError):
        return False


def resolve_resume(output, requested, fingerprint):
    if not requested:
        return None
    candidates = [p for p in output.glob("checkpoint-*")
                  if p.name.removeprefix("checkpoint-").isdigit() and complete_checkpoint(p, fingerprint)]
    if requested == "auto":
        if not candidates:
            return None  # A lease can end during reference precomputation, before the first update.
        return str(max(candidates, key=lambda p: int(p.name.split("-")[-1])))
    path = Path(requested).expanduser()
    if path.parent.resolve() != output.resolve() or not complete_checkpoint(path, fingerprint):
        raise ValueError("Requested checkpoint is incomplete or belongs to another run")
    return str(path)


def cached_batch(path, key, start, count):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (value.get("key") != key or value.get("start") != start or value.get("count") != count
            or any(len(value.get(name, [])) != count for name in ("ref_logps", "ref_KL_logps"))
            or any(not math.isfinite(x) for name in ("ref_logps", "ref_KL_logps") for x in value[name])):
        raise ValueError("Reference cache does not match this batch: " + str(path))
    return value


def adapter_digest(model, name):
    import torch
    h = hashlib.sha256()
    matched = 0
    for key, tensor in sorted(model.named_parameters()):
        if "." + name + "." not in key:
            continue
        matched += 1
        h.update(key.replace("." + name + ".", ".ADAPTER.").encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    if not matched:
        raise ValueError("No parameters found for adapter " + name)
    return h.hexdigest()


def measure(tokenizer, record):
    prompt = tokenizer.apply_chat_template(record["prompt"], tokenize=True,
        add_generation_prompt=True, truncation=False, return_dict=False)
    full = tokenizer.apply_chat_template(record["prompt"] + record["completion"], tokenize=True,
        add_generation_prompt=False, truncation=False, return_dict=False)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("Cannot preserve the full SFT input/assistant token boundary")
    return {"prompt_tokens": len(prompt), "completion_tokens": len(full) - len(prompt),
            "total_tokens": len(full)}


def smoke_selection(lengths):
    train = [r for r in lengths if r["split"] == "train"]
    selected = {r["index"] for r in sorted(train, key=lambda r: -r["total_tokens"])[:2]}
    selected.update(r["index"] for r in sorted(train, key=lambda r: -r["completion_tokens"])[:2])
    for kind in ("proven_non_implication", "proven_vacuous_implication"):
        pool = [r for r in train if r["category"] == kind]
        selected.add(max(pool, key=lambda r: r["total_tokens"])["index"])
    for row in sorted(train, key=lambda r: -r["total_tokens"]):
        if len(selected) >= 8:
            break
        selected.add(row["index"])
    validation = [r for r in lengths if r["split"] == "validation"]
    val_selected = {max((r for r in validation if r["label"] is label),
                        key=lambda r: r["total_tokens"])["index"] for label in (True, False)}
    for row in sorted(validation, key=lambda r: -r["total_tokens"]):
        if len(val_selected) >= 4:
            break
        val_selected.add(row["index"])
    if len(selected) != 8 or len(val_selected) != 4:
        raise ValueError("Smoke test requires eight training and four validation examples")
    return {"train": sorted(selected), "validation": sorted(val_selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train"), default="preflight")
    parser.add_argument("--data-dir", type=Path, default=Path("data/kto_before_reduction_v2"))
    parser.add_argument("--sft-dir", type=Path, default=Path("outputs/qwen3_4b_sft_v2_train"))
    parser.add_argument("--profile-dir", type=Path, default=Path("logs/kto_v1_profile"))
    parser.add_argument("--smoke-dir", type=Path, default=Path("outputs/qwen3_4b_kto_v1_smoke"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--save-minutes", type=float, default=10)
    args = parser.parse_args()
    if min(args.epochs, args.learning_rate, args.beta, args.save_minutes) <= 0:
        parser.error("Epochs, learning rate, beta and checkpoint interval must be positive")
    manifest, records, metadata = load_data(args.data_dir)
    if file_hash(args.data_dir / "manifest.json") != DATASET_MANIFEST_SHA:
        raise ValueError("Use the frozen v2 KTO dataset prepared for this experiment")
    sft = json.loads((args.sft_dir / "run_config.json").read_text())
    if (sft["model"] != MODEL or sft["model_revision"] != REVISION
            or sft["chat_template_sha256"] != TEMPLATE_SHA or sft["data_sha256"] != SFT_DATA):
        raise ValueError("SFT model or dataset identity differs from the completed run")
    if json.loads((args.sft_dir / "result.json").read_text())["status"] != "PASS":
        raise ValueError("Source SFT run did not complete successfully")
    adapter = args.sft_dir / "final_adapter"
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    if adapter_config["base_model_name_or_path"] != MODEL:
        raise ValueError("SFT adapter is for a different base model")
    weights_files = [p for p in adapter.glob("adapter_model.*") if p.suffix in (".bin", ".safetensors")]
    if len(weights_files) != 1:
        raise ValueError("Expected exactly one SFT adapter weight file")
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import KTOConfig, KTOTrainer
    if version("trl") != "1.17.1":
        raise ValueError("This runner was prepared for your existing TRL 1.17.1; do not upgrade packages mid-run")
    required = {"max_length", "train_sampling_strategy", "precompute_ref_log_probs", "sync_ref_model"}
    if not required.issubset(KTOConfig.__dataclass_fields__):
        raise ValueError("Unexpected KTO configuration API")
    if 'model.add_adapter("ref",' not in inspect.getsource(KTOTrainer.__init__):
        raise ValueError("This KTOTrainer does not implement the checked frozen SFT reference setup")
    identity = {"model": MODEL, "model_revision": REVISION, "chat_template_sha256": TEMPLATE_SHA,
        "manifest_sha256": file_hash(args.data_dir / "manifest.json"),
        "sft_run_config_sha256": file_hash(args.sft_dir / "run_config.json"),
        "sft_adapter_files": {p.name: file_hash(p) for p in [adapter / "adapter_config.json", weights_files[0]]},
        "data_sha256": {s: file_hash(args.data_dir / (s + ".jsonl")) for s in records},
        "script_sha256": file_hash(Path(__file__)),
        "kto_trainer_source_sha256": file_hash(Path(inspect.getfile(KTOTrainer))),
        "packages": {p: version(p) for p in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes")}}
    profile_key = digest(identity)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != TEMPLATE_SHA:
        raise ValueError("Tokenizer differs from the saved SFT chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.mode == "preflight":
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        lengths = []
        for split, rows in records.items():
            for n, row in enumerate(rows):
                lengths.append(dict(measure(tokenizer, row), split=split, index=n, **metadata[(split, n)]))
                if (n + 1) % 250 == 0:
                    print("PROFILE", split, n + 1, "/", len(rows), flush=True)
        context = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).max_position_embeddings
        maximum = max(r["prompt_tokens"] for r in lengths) + max(r["completion_tokens"] for r in lengths)
        if maximum > context:
            raise ValueError("A full prompt plus mismatched KL completion can exceed model context; no truncation applied")
        report = {"status": "PASS", "fingerprint": profile_key, "identity": identity,
            "splits": manifest["splits"], "truncation": False, "model_context": context,
            "maximum_full_tokens": max(r["total_tokens"] for r in lengths),
            "maximum_KL_tokens_upper_bound": maximum, "smoke_selection": smoke_selection(lengths),
            "class_weights": class_weights(manifest["splits"]["train"]["labels"])}
        write_json(args.profile_dir / "lengths.json", lengths)
        report["lengths_sha256"] = file_hash(args.profile_dir / "lengths.json")
        write_json(args.profile_dir / "summary.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return
    profile = json.loads((args.profile_dir / "summary.json").read_text())
    if (profile["status"] != "PASS" or profile["fingerprint"] != profile_key
            or file_hash(args.profile_dir / "lengths.json") != profile["lengths_sha256"]):
        raise ValueError("Run preflight again: profile, code, data or SFT adapter changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Expected one allocated CUDA slice with BF16 support; check CUDA_VISIBLE_DEVICES")
    output = args.out or Path("outputs/qwen3_4b_kto_v1_" + args.mode)
    if output.resolve().is_relative_to(args.sft_dir.resolve()):
        raise ValueError("KTO outputs must be separate from the original SFT run")
    if output.exists() and not args.resume:
        raise ValueError("Output exists; use --resume auto or a fresh --out")
    if args.resume and not output.exists():
        raise ValueError("Cannot resume a missing output directory")
    selection = profile["smoke_selection"] if args.mode == "smoke" else None
    batch_size, accumulation = 2, 1 if selection else 4
    weights = class_weights(manifest["splits"]["train"]["labels"])
    settings = {"identity": identity, "profile_fingerprint": profile_key, "mode": args.mode,
        "batch_size": batch_size, "gradient_accumulation": accumulation,
        "epochs": 1 if selection else args.epochs, "max_steps": 4 if selection else -1,
        "learning_rate": args.learning_rate, "beta": args.beta, "class_weights": weights,
        "selection": selection or "all assigned train/validation examples", "seed": 42,
        "dataset_order": "one shuffle with seed 42, then sequential KTO batches",
        "truncation": False, "precompute_reference": "durable batch cache",
        "reference": "frozen copy of selected SFT adapter", "loss": "kto",
        "quantization": "4-bit NF4; double quantization; BF16 compute",
        "save_minutes": args.save_minutes, "save_steps": 1 if selection else 25,
        "eval_steps": 1 if selection else 200}
    if args.mode == "train":
        smoke_result = json.loads((args.smoke_dir / "result.json").read_text())
        smoke_config = json.loads((args.smoke_dir / "run_config.json").read_text())
        if (smoke_result["status"] != "PASS" or smoke_result["global_step"] != 4
                or not smoke_result["policy_changed"] or not smoke_result["frozen_reference_unchanged"]):
            raise ValueError("Complete a passing four-update memory smoke test before full training")
        for key in ("identity", "profile_fingerprint", "batch_size", "learning_rate", "beta", "class_weights"):
            if smoke_config[key] != settings[key]:
                raise ValueError("Smoke test used a different configuration: " + key)
    fingerprint = digest(settings)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".kto.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        if json.loads((output / "run_config.json").read_text())["run_fingerprint"] != fingerprint:
            raise ValueError("Cannot resume with changed code, model, dataset, settings or packages")
    else:
        write_json(output / "run_config.json", dict(settings, run_fingerprint=fingerprint))
    resume = resolve_resume(output, args.resume, fingerprint)
    if args.resume and not resume:
        print("No complete optimizer checkpoint yet; resuming cached reference work from the original SFT policy.", flush=True)
    set_seed(42)
    datasets = {s: Dataset.from_list([rows[n] for n in selection[s]] if selection else rows).shuffle(seed=42)
                for s, rows in records.items()}
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
        print("Stop requested: finishing the current reference batch or optimizer update.", flush=True)
    previous = {n: signal.signal(n, request_stop) for n in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    class CachedKTOTrainer(KTOTrainer):
        def _save(self, output_dir=None, state_dict=None):
            # Saving a ref/ subdirectory alongside the root default adapter can
            # make Trainer's multi-adapter resume path skip the root policy.
            # Recreate the fixed reference from the immutable SFT source instead.
            destination = Path(output_dir or self.args.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(destination, state_dict=state_dict,
                selected_adapters=["default"], safe_serialization=True, save_embedding_layers=False)
            self.processing_class.save_pretrained(destination)
            torch.save(self.args, destination / "training_args.bin")

        def _precompute_ref_logps(self, dataset, name, batch_size):
            if "ref" not in self.model.peft_config or adapter_digest(self.model, "ref") != initial_policy:
                raise RuntimeError("Reference cache requires the frozen SFT adapter")
            key = digest([fingerprint, dataset._fingerprint, name, batch_size])
            folder = output / "reference_cache" / name / key[:20]
            folder.mkdir(parents=True, exist_ok=True)
            values = {"ref_logps": [], "ref_KL_logps": []}
            for start in range(0, len(dataset), batch_size):
                count = min(batch_size, len(dataset) - start)
                path = folder / (str(start) + ".json")
                value = cached_batch(path, key, start, count)
                if value is None:
                    padded = self.data_collator([dataset[n] for n in range(start, start + count)])
                    padded = self._prepare_inputs(padded)
                    logps, kl_logps = self.compute_ref_log_probs(self.model, padded)
                    value = {"key": key, "start": start, "count": count,
                             "ref_logps": logps.detach().float().cpu().tolist(),
                             "ref_KL_logps": kl_logps.detach().float().cpu().tolist()}
                    if any(not math.isfinite(x) for field in values for x in value[field]):
                        raise RuntimeError("Non-finite reference log probability")
                    write_json(path, value)
                    del padded, logps, kl_logps
                    print("REFERENCE", name, start + count, "/", len(dataset), "saved", flush=True)
                for field in values:
                    values[field].extend(value[field])
                write_json(output / "progress.json", {"phase": "reference_" + name,
                    "saved_examples": start + count, "total_examples": len(dataset),
                    "remaining_examples": len(dataset) - start - count})
                if stop["requested"]:
                    raise InterruptedError("Reference progress saved; resume with --resume auto")
            for field, column in values.items():
                dataset = dataset.add_column(field, column)
            return dataset

    try:
        base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True, local_files_only=True)
        initial_policy = adapter_digest(model, "default")
        updates = 4 if selection else math.ceil(math.ceil(len(datasets["train"]) / 2) / accumulation) * args.epochs
        config = KTOConfig(output_dir=str(output), num_train_epochs=settings["epochs"],
            max_steps=settings["max_steps"], per_device_train_batch_size=2, per_device_eval_batch_size=2,
            gradient_accumulation_steps=accumulation, learning_rate=args.learning_rate, beta=args.beta,
            desirable_weight=weights["desirable"], undesirable_weight=weights["undesirable"],
            loss_type="kto", max_length=None, precompute_ref_log_probs=True, precompute_ref_batch_size=2,
            sync_ref_model=False, train_sampling_strategy="sequential", bf16=True, fp16=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch", lr_scheduler_type="cosine", warmup_steps=0 if selection else math.ceil(.03 * updates),
            weight_decay=.01, max_grad_norm=1.0, eval_strategy="steps", eval_steps=settings["eval_steps"],
            save_strategy="steps", save_steps=settings["eval_steps"], save_total_limit=3, save_only_model=False,
            load_best_model_at_end=not bool(selection), metric_for_best_model="eval_loss", greater_is_better=False,
            logging_steps=1 if selection else 5, prediction_loss_only=True, report_to=[], push_to_hub=False,
            remove_unused_columns=True, seed=42, data_seed=42, dataloader_num_workers=0, dataset_num_proc=1)
        torch.cuda.reset_peak_memory_stats()
        trainer = CachedKTOTrainer(model=model, args=config, train_dataset=datasets["train"],
            eval_dataset=datasets["validation"], processing_class=tokenizer)
        if "ref" not in model.peft_config or adapter_digest(model, "ref") != initial_policy:
            raise RuntimeError("KTO reference is not an exact copy of the SFT adapter")
        model.set_adapter("default")
        for name, parameter in model.named_parameters():
            if ".ref." in name:
                parameter.requires_grad_(False)
        if any(p.requires_grad and ".default." not in n for n, p in model.named_parameters()):
            raise RuntimeError("Only the policy LoRA adapter may be trainable")

        class Checkpoints(TrainerCallback):
            def __init__(self):
                self.last_save = time.monotonic()

            def check_reference(self):
                if adapter_digest(model, "ref") != initial_policy:
                    raise RuntimeError("Frozen SFT reference changed")

            def on_train_begin(self, args, state, control, **kwargs):
                self.check_reference()
                if resume:
                    marker = json.loads((Path(resume) / "checkpoint_complete.json").read_text())
                    if adapter_digest(model, "default") != marker["policy_sha256"]:
                        raise RuntimeError("Resumed policy differs from the saved checkpoint")

            def on_step_end(self, args, state, control, **kwargs):
                final = state.global_step == state.max_steps
                if (time.monotonic() - self.last_save >= settings["save_minutes"] * 60
                        or state.global_step % settings["save_steps"] == 0 or final or stop["requested"]):
                    control.should_save = True
                if final and not stop["requested"]:
                    control.should_evaluate = True
                if stop["requested"]:
                    control.should_training_stop = True
                    control.should_evaluate = False
                write_json(output / "progress.json", {"phase": "training", "global_step": state.global_step,
                    "total_steps": state.max_steps, "remaining_steps": state.max_steps - state.global_step,
                    "epoch": state.epoch})
                return control

            def on_log(self, args, state, control, logs=None, **kwargs):
                for key, value in (logs or {}).items():
                    if isinstance(value, (int, float)) and not math.isfinite(value):
                        raise RuntimeError("Non-finite training metric: " + key)

            def on_save(self, args, state, control, **kwargs):
                self.check_reference()
                folder = output / ("checkpoint-" + str(state.global_step))
                marker = folder / "checkpoint_complete.json"
                write_json(marker, {"run_fingerprint": fingerprint, "global_step": state.global_step,
                    "frozen_reference_sha256": initial_policy, "policy_sha256": adapter_digest(model, "default"),
                    "saved_utc": datetime.now(timezone.utc).isoformat()})
                if not complete_checkpoint(folder, fingerprint):
                    marker.unlink()
                    raise RuntimeError("Incomplete adapter/optimizer/scheduler/RNG checkpoint")
                self.last_save = time.monotonic()
                print("Complete resumable checkpoint:", folder, flush=True)
                return control

        trainer.add_callback(Checkpoints())
        result = trainer.train(resume_from_checkpoint=resume)
        if adapter_digest(model, "ref") != initial_policy:
            raise RuntimeError("Frozen reference changed during KTO")
        policy_changed = adapter_digest(model, "default") != initial_policy
        checkpoints = [str(p) for p in output.glob("checkpoint-*") if complete_checkpoint(p, fingerprint)]
        if not checkpoints or not policy_changed:
            raise RuntimeError("No complete checkpoint or no policy update")
        if not stop["requested"]:
            model.save_pretrained(output / "final_adapter", selected_adapters=["default"], save_embedding_layers=False)
            tokenizer.save_pretrained(output / "final_adapter")
        summary = {"status": "stopped_with_checkpoint" if stop["requested"] else "PASS", "mode": args.mode,
            "global_step": trainer.state.global_step, "elapsed_seconds_this_invocation": time.monotonic() - started,
            "policy_changed": policy_changed, "frozen_reference_unchanged": True,
            "best_checkpoint": trainer.state.best_model_checkpoint, "best_validation_loss": trainer.state.best_metric,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "available_device_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "complete_checkpoints": checkpoints, "metrics": result.metrics,
            "formal_evaluation": "not run; KTO loss is not an assertion correctness/vacuity result"}
        write_json(output / "result.json", summary)
        write_json(output / "progress.json", {"phase": summary["status"], "global_step": trainer.state.global_step})
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        write_json(output / "failure.json", {"error": type(error).__name__, "message": str(error),
            "elapsed_seconds_this_invocation": time.monotonic() - started})
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
