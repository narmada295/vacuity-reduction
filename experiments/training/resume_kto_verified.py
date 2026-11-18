"""Resume the frozen KTO runner with verified in-place restoration of LoRA weights.

This does not edit the runner, its fingerprints, checkpoints, or reference cache.
Only the single-device, root-level default LoRA checkpoint layout is supported.
The original runner continues to restore optimizer/scheduler/RNG and data position.
"""

import hashlib
import json
from pathlib import Path
import runpy
import sys
from datetime import datetime, timezone


FROZEN_RUNNER_SHA = "4538a70c9b826ed98f68394bd0420ccfff017c494895c6175ccee03036a8aa0d"


def canonical_saved_key(key):
    """Reconstruct the original runner's digest names for ordinary LoRA A/B."""
    parts = key.split(".")
    indexes = [n for n, part in enumerate(parts) if part in ("lora_A", "lora_B")]
    if len(indexes) != 1 or parts[indexes[0] + 1:] != ["weight"]:
        raise ValueError("Unsupported saved adapter parameter: " + key)
    parts.insert(indexes[0] + 1, "ADAPTER")
    return ".".join(parts)


def parameter_map(model):
    result = {}
    for key, parameter in model.named_parameters():
        if ".default." not in key:
            continue
        saved_key = key.replace(".default.", ".")
        if saved_key in result:
            raise ValueError("Ambiguous adapter parameter: " + saved_key)
        if canonical_saved_key(saved_key) != key.replace(".default.", ".ADAPTER."):
            raise ValueError("Unsupported adapter parameter: " + key)
        result[saved_key] = parameter
    if not result:
        raise ValueError("No default LoRA parameters found")
    return result


def saved_digest(weights, torch):
    h = hashlib.sha256()
    for key, tensor in sorted(weights.items()):
        h.update(canonical_saved_key(key).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def restore_policy(model, weights, marker, adapter_digest, torch):
    """Validate everything before copying, then verify policy and reference bytes."""
    if set(model.peft_config) != {"default", "ref"}:
        raise ValueError("Expected the existing default policy and frozen SFT ref adapter")
    if saved_digest(weights, torch) != marker["policy_sha256"]:
        raise RuntimeError("Saved adapter bytes do not match the checkpoint marker; nothing restored")
    reference = adapter_digest(model, "ref")
    if reference != marker["frozen_reference_sha256"]:
        raise RuntimeError("Reference differs from the checkpoint's original SFT reference")
    parameters = parameter_map(model)
    if set(parameters) != set(weights):
        raise RuntimeError("Checkpoint and policy parameter names differ; nothing restored")
    for key, parameter in parameters.items():
        tensor = weights[key]
        if parameter.shape != tensor.shape or parameter.dtype != tensor.dtype:
            raise RuntimeError("Checkpoint shape/dtype differs for " + key + "; nothing restored")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("Non-finite checkpoint parameter: " + key)
    before = adapter_digest(model, "default")
    # Preserve Parameter objects: optimizer state must attach to these same objects.
    with torch.no_grad():
        for key, parameter in parameters.items():
            parameter.copy_(weights[key].to(parameter.device))
    model.set_adapter("default")
    for key, parameter in model.named_parameters():
        parameter.requires_grad_(".default." in key)
    after = adapter_digest(model, "default")
    if after != marker["policy_sha256"]:
        raise RuntimeError("Explicit restoration did not reproduce the checkpoint policy")
    if adapter_digest(model, "ref") != reference:
        raise RuntimeError("Reference changed while restoring the policy")
    return {"policy_before_sha256": before, "policy_restored_sha256": after,
            "frozen_reference_sha256": reference, "parameter_count": len(parameters),
            "parameter_objects_preserved": True}


def main():
    # Keep the original file byte-for-byte: the existing profile/run/cache include its hash.
    runner = Path(__file__).with_name("train_kto.py")
    if hashlib.sha256(runner.read_bytes()).hexdigest() != FROZEN_RUNNER_SHA:
        raise RuntimeError("The original frozen train_kto.py has changed; restore its original copy first")
    if not any(arg == "--resume" or arg.startswith("--resume=") for arg in sys.argv[1:]):
        raise ValueError("This helper is only for --resume auto (or an explicit complete checkpoint)")
    frozen = runpy.run_path(str(runner), run_name="kto_frozen_runner")
    import torch
    from safetensors.torch import load_file
    from trl import KTOTrainer

    def verified_load(trainer, resume_from_checkpoint, model=None):
        if (trainer.is_deepspeed_enabled or trainer.is_fsdp_enabled
                or trainer.args.world_size != 1):
            raise ValueError("Recovery helper supports the original single-device run only")
        model = trainer.model if model is None else model
        checkpoint = Path(resume_from_checkpoint)
        output = Path(trainer.args.output_dir)
        config = json.loads((output / "run_config.json").read_text())
        if (checkpoint.parent.resolve() != output.resolve()
                or not frozen["complete_checkpoint"](checkpoint, config["run_fingerprint"])):
            raise ValueError("Checkpoint is incomplete or belongs to another run")
        marker = json.loads((checkpoint / "checkpoint_complete.json").read_text())
        path = checkpoint / "adapter_model.safetensors"
        weights = load_file(str(path), device="cpu")
        result = restore_policy(model, weights, marker, frozen["adapter_digest"], torch)
        result.update({"event": "verified_explicit_policy_restore", "checkpoint": str(checkpoint),
                       "global_step": marker["global_step"],
                       "run_fingerprint": config["run_fingerprint"],
                       "recovery_script_sha256": frozen["file_hash"](Path(__file__)),
                       "original_runner_sha256": FROZEN_RUNNER_SHA,
                       "checkpoint_weights_sha256": frozen["file_hash"](path),
                       "installed_transformers_trainer_sha256": frozen["file_hash"](
                           Path(__import__("inspect").getfile(__import__("transformers").Trainer))),
                       "saved_utc": datetime.now(timezone.utc).isoformat()})
        name = "resume_recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
        frozen["write_json"](output / name, result)
        print("Verified checkpoint policy restored at update", marker["global_step"],
              "; frozen SFT reference unchanged", flush=True)

    # All other Trainer methods, including optimizer/scheduler/RNG loading, remain original.
    KTOTrainer._load_from_checkpoint = verified_load
    frozen["main"]()


if __name__ == "__main__":
    main()
