import unittest
from env.montage_env import MontageEnv
from env.models import Action

class TestEnv(unittest.TestCase):
    def setUp(self):
        self.env = MontageEnv(task_name="highlight", seed=42)
        
    def test_reset_returns_valid_state(self):
        obs = self.env.reset()
        self.assertIsNotNone(obs)
        self.assertTrue(len(obs.available_clips) > 0)
        self.assertEqual(len(obs.timeline), 0)
        
    def test_step_updates_correctly(self):
        obs = self.env.reset()
        first_clip_id = obs.available_clips[0].id
        
        action = Action(action_type="select", clip_id=first_clip_id)
        next_obs, reward, done, info = self.env.step(action)
        
        self.assertIn(first_clip_id, next_obs.timeline)
        self.assertTrue(first_clip_id not in [c.id for c in next_obs.available_clips])
        self.assertFalse(done)

if __name__ == '__main__':
    unittest.main()
