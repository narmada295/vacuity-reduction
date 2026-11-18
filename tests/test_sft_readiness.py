import copy
import unittest

from experiments.training.profile_sft import validate_designs
from experiments.training.reassess_l3_3 import assess


class SftReadinessTests(unittest.TestCase):
    def test_all_related_covers_must_be_covered(self):
        states = {"SFT_CHECK_0001": "proven", "SFT_CHECK_0001:precondition1": "covered",
                  "SFT_CHECK_0001:contrapositive1": "covered"}
        self.assertIs(assess(states, "SFT_CHECK_0001", True)["vacuous"], False)
        states["SFT_CHECK_0001:precondition2"] = "unreachable"
        self.assertIs(assess(states, "SFT_CHECK_0001", True)["vacuous"], True)
        states["SFT_CHECK_0001:precondition2"] = "undetermined"
        self.assertIsNone(assess(states, "SFT_CHECK_0001", True)["vacuous"])
        del states["SFT_CHECK_0001:contrapositive1"]
        self.assertIsNone(assess(states, "SFT_CHECK_0001", True)["vacuous"])

    def test_falsified_assertion_cannot_become_a_positive_example(self):
        result = assess({"SFT_CHECK_0001": "cex"}, "SFT_CHECK_0001", True)
        self.assertEqual(result["assertion_status"], "cex")
        self.assertEqual(result["reason"], "assertion_falsified")
        result = assess({"SFT_CHECK_0001": "proven"}, "SFT_CHECK_0001", False)
        self.assertIsNone(result["vacuous"])
        self.assertEqual(result["reason"], "outside_scope_non_implication")

    def test_remote_preflight_checks_design_membership_and_level_quotas(self):
        manifest = {
            "split_plan": {
                "train": ["L1/a", "L2/b", "L3/3"], "validation": ["L3/v"], "test": ["L3/t"],
                "pending_allowed": ["L3/t"], "excluded": {},
                "design_quotas": {
                    "train": {"L1": 1, "L2": 1, "L3": 1},
                    "validation": {"L1": 0, "L2": 0, "L3": 1},
                    "test": {"L1": 0, "L2": 0, "L3": 1},
                },
            },
            "pending": [{"case_id": "L3/t", "split": "test"}],
            "splits": {"train": {"designs": 3}, "validation": {"designs": 1}, "test": {"designs": 0}},
        }
        self.assertEqual(validate_designs(manifest, "train", ["L1/a", "L2/b", "L3/3"]),
                         {"L1": 1, "L2": 1, "L3": 1})
        with self.assertRaisesRegex(ValueError, "design membership"):
            validate_designs(manifest, "train", ["L1/a", "L2/b"])
        wrong = copy.deepcopy(manifest)
        wrong["split_plan"]["design_quotas"]["train"]["L3"] = 2
        with self.assertRaisesRegex(ValueError, "quota mismatch"):
            validate_designs(wrong, "train", ["L1/a", "L2/b", "L3/3"])
        wrong = copy.deepcopy(manifest)
        wrong["split_plan"]["excluded"] = {"L3/3": "still excluded"}
        with self.assertRaisesRegex(ValueError, "excluded design"):
            validate_designs(wrong, "train", ["L1/a", "L2/b", "L3/3"])


if __name__ == "__main__":
    unittest.main()
