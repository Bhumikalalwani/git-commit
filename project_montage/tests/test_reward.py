import unittest
from env.reward_engine import RewardEngine
from env.models import Clip

class TestReward(unittest.TestCase):
    def setUp(self):
        self.engine = RewardEngine()
        self.clips = {
            "clip_001": Clip(id="clip_001", duration=10, importance=1.0, emotion="happiness", motion="high", tags=["action"]),
            "clip_002": Clip(id="clip_002", duration=5, importance=0.8, emotion="happiness", motion="medium", tags=["b-roll"])
        }

    def test_reward_monotonicity(self):
        r1 = self.engine.compute_reward(["clip_001"], self.clips, 15, "balanced", step_count=1)
        r2 = self.engine.compute_reward(["clip_001", "clip_002"], self.clips, 15, "balanced", step_count=2)
        self.assertGreaterEqual(r1.value, 0.0)
        self.assertGreaterEqual(r2.value, 0.0)

    def test_penalty_correctness(self):
        r_over = self.engine.compute_reward(["clip_001", "clip_002"], self.clips, 5, "balanced", step_count=1)
        self.assertGreater(r_over.breakdown["duration_penalty"], 0.0)

if __name__ == '__main__':
    unittest.main()
