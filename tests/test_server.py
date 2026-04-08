import unittest

from fastapi.testclient import TestClient

import app as app_module
from app import app


class TestOpenEnvServer(unittest.TestCase):
    def setUp(self):
        app_module._env = None
        self.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["environment"], "montage_env")

    def test_landing_page(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Montage OpenEnv", r.text)

    def test_state_before_reset_returns_400(self):
        r = self.client.get("/state")
        self.assertEqual(r.status_code, 400)

    def test_step_before_reset_returns_400(self):
        r = self.client.post("/step", json={
            "action": {"action_type": "finish"},
        })
        self.assertEqual(r.status_code, 400)

    def test_reset_returns_observation(self):
        r = self.client.post("/reset", json={"task_name": "highlight", "seed": 42})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("observation", data)
        self.assertEqual(data["task_name"], "highlight")
        self.assertEqual(data["seed"], 42)
        obs = data["observation"]
        self.assertIn("available_clips", obs)
        self.assertFalse(obs["done"])

    def test_step_returns_reward_and_done(self):
        self.client.post("/reset", json={"task_name": "highlight", "seed": 42})
        r = self.client.post("/step", json={
            "action": {"action_type": "select", "clip_id": "clip_009"},
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("observation", data)
        self.assertIn("reward", data)
        self.assertIn("done", data)
        self.assertIsNotNone(data["reward"])
        self.assertFalse(data["done"])

    def test_full_episode(self):
        r = self.client.post("/reset", json={"task_name": "highlight", "seed": 42})
        self.assertEqual(r.status_code, 200)
        obs = r.json()["observation"]
        clip_id = obs["available_clips"][0]["id"]

        r = self.client.post("/step", json={
            "action": {"action_type": "select", "clip_id": clip_id},
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn(clip_id, data["observation"]["timeline"])

        r = self.client.post("/step", json={
            "action": {"action_type": "finish"},
        })
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["done"])

        r = self.client.get("/state")
        self.assertEqual(r.status_code, 200)
        state = r.json()
        self.assertEqual(state["task_name"], "highlight")
        self.assertTrue(state["done"])

    def test_all_task_types(self):
        for task in ["highlight", "structured", "intent"]:
            r = self.client.post("/reset", json={"task_name": task, "seed": 42})
            self.assertEqual(r.status_code, 200, f"Failed for task: {task}")
            self.assertIn("observation", r.json())


if __name__ == "__main__":
    unittest.main()
