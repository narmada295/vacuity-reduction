"""Regression checks for policy restoration without replacing optimizer parameters."""
import copy
import unittest

try:
    import torch
except ImportError:
    torch = None

from experiments.training.resume_kto_verified import canonical_saved_key, restore_policy
from experiments.training.train_kto import adapter_digest


@unittest.skipIf(torch is None, "CPU PyTorch is required for tensor recovery checks")
class KtoResumeRecoveryTests(unittest.TestCase):
    def setUp(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = torch.nn.ModuleDict({n: torch.nn.Linear(3, 2, bias=False)
                                                 for n in ("default", "ref")})
                self.lora_B = torch.nn.ModuleDict({n: torch.nn.Linear(2, 3, bias=False)
                                                 for n in ("default", "ref")})
                self.base_layer = torch.nn.Linear(3, 3, bias=False)
                self.peft_config = {"default": {}, "ref": {}}
                self.active_adapter = "default"

            def set_adapter(self, name):
                self.active_adapter = name

        torch.manual_seed(42)
        self.model = Model()
        self.weights = {key.replace(".default.", "."): p.detach().clone()
                        for key, p in self.model.named_parameters() if ".default." in key}
        self.marker = {"policy_sha256": adapter_digest(self.model, "default"),
                       "frozen_reference_sha256": adapter_digest(self.model, "ref")}
        with torch.no_grad():
            for key, p in self.model.named_parameters():
                if ".default." in key:
                    p.add_(1)

    def restore(self, weights=None, marker=None):
        return restore_policy(self.model, self.weights if weights is None else weights,
                              self.marker if marker is None else marker, adapter_digest, torch)

    def test_restore_matches_saved_policy_and_keeps_optimizer_parameter_objects(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
        base = self.model.base_layer.weight.detach().clone()
        report = self.restore()
        self.assertNotEqual(report["policy_before_sha256"], report["policy_restored_sha256"])
        self.assertEqual(adapter_digest(self.model, "default"), self.marker["policy_sha256"])
        self.assertEqual(adapter_digest(self.model, "ref"), self.marker["frozen_reference_sha256"])
        self.assertEqual(ids, [id(p) for p in self.model.parameters()])
        self.assertTrue(torch.equal(base, self.model.base_layer.weight))
        for key, p in self.model.named_parameters():
            self.assertEqual(p.requires_grad, ".default." in key)

    def test_corrupted_saved_weight_is_rejected_before_any_copy(self):
        before = adapter_digest(self.model, "default")
        bad = copy.deepcopy(self.weights)
        bad["lora_B.weight"][0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "Saved adapter bytes"):
            self.restore(weights=bad)
        self.assertEqual(adapter_digest(self.model, "default"), before)

    def test_wrong_reference_is_rejected_before_any_copy(self):
        before = adapter_digest(self.model, "default")
        with self.assertRaisesRegex(RuntimeError, "Reference differs"):
            self.restore(marker=dict(self.marker, frozen_reference_sha256="wrong"))
        self.assertEqual(adapter_digest(self.model, "default"), before)

    def test_shape_or_dtype_change_is_rejected_before_partial_copy(self):
        for data in (torch.zeros(3, 3), torch.zeros(3, 2, dtype=torch.float64)):
            with self.subTest(dtype=data.dtype, shape=data.shape):
                original = self.model.lora_B["default"].weight
                self.model.lora_B["default"].weight = torch.nn.Parameter(data)
                before = adapter_digest(self.model, "default")
                with self.assertRaisesRegex(RuntimeError, "shape/dtype differs"):
                    self.restore()
                self.assertEqual(adapter_digest(self.model, "default"), before)
                self.model.lora_B["default"].weight = original

    def test_unknown_adapter_is_rejected(self):
        self.model.peft_config["other"] = {}
        with self.assertRaisesRegex(ValueError, "existing default policy"):
            self.restore()

    def test_saved_name_reconstruction_rejects_other_parameter_families(self):
        self.assertEqual(canonical_saved_key("base.model.q_proj.lora_A.weight"),
                         "base.model.q_proj.lora_A.ADAPTER.weight")
        for key in ("base.weight", "lora_A.default.weight", "lora_embedding_A.weight"):
            with self.assertRaises(ValueError):
                canonical_saved_key(key)


if __name__ == "__main__":
    unittest.main()
