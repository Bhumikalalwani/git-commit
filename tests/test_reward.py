import unittest

from env.models import Clip
from env.reward_engine import RewardEngine


class TestRewardEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RewardEngine()
        self.clips = {
            "clip_001": Clip(id="clip_001", duration=10, importance=1.0, emotion="happiness", motion="high", tags=["action"]),
            "clip_002": Clip(id="clip_002", duration=5, importance=0.8, emotion="happiness", motion="medium", tags=["b-roll"]),
        }

    def test_empty_timeline_returns_zero(self):
        val, breakdown = self.engine.compute_reward([], self.clips, 15, "balanced", step_count=1)
        self.assertEqual(val, 0.0)
        self.assertEqual(breakdown.importance, 0.0)

    def test_reward_in_unit_range(self):
        val, _ = self.engine.compute_reward(["clip_001"], self.clips, 15, "balanced", step_count=1)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_more_clips_maintain_range(self):
        val, _ = self.engine.compute_reward(
            ["clip_001", "clip_002"], self.clips, 15, "balanced", step_count=2,
        )
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_duration_penalty_applied(self):
        _, breakdown = self.engine.compute_reward(
            ["clip_001", "clip_002"], self.clips, 5, "balanced", step_count=1,
        )
        self.assertGreater(breakdown.duration_penalty, 0.0)

    def test_weight_overrides(self):
        overrides = {"importance": 0.9, "coherence": 0.05, "style": 0.05}
        val_default, _ = self.engine.compute_reward(
            ["clip_001"], self.clips, 15, "balanced", step_count=1,
        )
        val_override, _ = self.engine.compute_reward(
            ["clip_001"], self.clips, 15, "balanced", step_count=1,
            weight_overrides=overrides,
        )
        self.assertNotEqual(val_default, val_override)

    def test_breakdown_fields_present(self):
        _, breakdown = self.engine.compute_reward(
            ["clip_001"], self.clips, 15, "balanced", step_count=1,
        )
        self.assertIsNotNone(breakdown.importance)
        self.assertIsNotNone(breakdown.coherence)
        self.assertIsNotNone(breakdown.style)
        self.assertIsNotNone(breakdown.duration_penalty)
        self.assertIsNotNone(breakdown.redundancy_penalty)
        self.assertIsNotNone(breakdown.step_penalty)


if __name__ == "__main__":
    unittest.main()
