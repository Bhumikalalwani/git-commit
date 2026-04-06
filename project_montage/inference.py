import os
import logging
from typing import List
import openai
from pydantic import BaseModel

from env.montage_env import MontageEnv
from env.models import Action

logging.basicConfig(level=logging.INFO, format="%(message)s")

class AgentInference:
    def __init__(self, use_openai=False):
        self.use_openai = use_openai
        if self.use_openai:
            try:
                self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy_key"))
            except Exception:
                pass

    def decide_action(self, obs, env) -> Action:
        available = obs.available_clips
        timeline = obs.timeline
        remaining = obs.remaining_time

        if not available or remaining <= 0:
            return Action(action_type="finish")

        for clip in sorted(available, key=lambda c: c.importance, reverse=True):
            if clip.duration <= remaining + 5:
                return Action(action_type="select", clip_id=clip.id)
                
        return Action(action_type="finish")

def run_task(task_name: str):
    logging.info("[START]")
    logging.info(f"Task: {task_name}")
    
    env = MontageEnv(task_name=task_name)
    obs = env.reset()
    agent = AgentInference()
    
    done = False
    step_count = 0
    total_reward = 0.0
    
    while not done and step_count < 50:
        logging.info("[STEP]")
        action = agent.decide_action(obs, env)
        obs, reward, done, info = env.step(action)
        total_reward = reward.value
        step_count += 1
        
    logging.info("[END]")
    logging.info(f"Steps: {step_count}")
    logging.info(f"Final Reward: {total_reward:.4f}")
    
if __name__ == "__main__":
    for task in ["highlight", "structured", "intent"]:
        run_task(task)
