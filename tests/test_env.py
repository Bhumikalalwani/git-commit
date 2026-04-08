import unittest

from env.models import Action, ActionType
from env.montage_env import MontageEnv


class TestMontageEnv(unittest.TestCase):
    def setUp(self):
        self.env = MontageEnv(task_name="highlight", seed=42)

    def test_reset_returns_valid_observation(self):
        obs = self.env.reset()
        self.assertIsNotNone(obs)
        self.assertGreater(len(obs.available_clips), 0)
        self.assertEqual(len(obs.timeline), 0)
        self.assertFalse(obs.done)
        self.assertIsNone(obs.reward)

    def test_step_select_adds_clip(self):
        obs = self.env.reset()
        clip_id = obs.available_clips[0].id

        obs = self.env.step(Action(action_type=ActionType.SELECT, clip_id=clip_id))
        self.assertIn(clip_id, obs.timeline)
        self.assertNotIn(clip_id, [c.id for c in obs.available_clips])
        self.assertIsNotNone(obs.reward)
        self.assertFalse(obs.done)

    def test_step_remove_removes_clip(self):
        obs = self.env.reset()
        clip_id = obs.available_clips[0].id
        self.env.step(Action(action_type=ActionType.SELECT, clip_id=clip_id))

        obs = self.env.step(Action(action_type=ActionType.REMOVE, clip_id=clip_id))
        self.assertNotIn(clip_id, obs.timeline)

    def test_step_finish_terminates(self):
        self.env.reset()
        obs = self.env.step(Action(action_type=ActionType.FINISH))
        self.assertTrue(obs.done)

    def test_state_property(self):
        self.env.reset()
        state = self.env.state
        self.assertEqual(state.task_name, "highlight")
        self.assertEqual(state.seed, 42)
        self.assertEqual(state.step_count, 0)
        self.assertFalse(state.done)

    def test_state_before_reset_raises(self):
        with self.assertRaises(ValueError):
            _ = self.env.state

    def test_reward_in_range(self):
        obs = self.env.reset()
        for clip in obs.available_clips[:3]:
            obs = self.env.step(Action(action_type=ActionType.SELECT, clip_id=clip.id))
        self.assertGreaterEqual(obs.reward, 0.0)
        self.assertLessEqual(obs.reward, 1.0)

    def test_all_tasks_run(self):
        for task in ["highlight", "structured", "intent"]:
            env = MontageEnv(task_name=task, seed=42)
            obs = env.reset()
            self.assertGreater(len(obs.available_clips), 0)

    def test_reproducibility(self):
        env1 = MontageEnv(task_name="highlight", seed=99)
        obs1 = env1.reset()
        env2 = MontageEnv(task_name="highlight", seed=99)
        obs2 = env2.reset()
        self.assertEqual(
            [c.id for c in obs1.available_clips],
            [c.id for c in obs2.available_clips],
        )


if __name__ == "__main__":
    unittest.main()
