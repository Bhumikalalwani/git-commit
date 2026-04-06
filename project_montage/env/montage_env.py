import random
from typing import Tuple, Dict, Any, Optional
from env.models import Observation, Action, Reward
from env.state_manager import StateManager
from env.reward_engine import RewardEngine
from dao.clip_dao import ClipDAO
from dao.task_dao import TaskDAO

class MontageEnv:
    def __init__(self, task_name: str = "highlight", seed: int = 42):
        self.task_name = task_name
        self.seed = seed
        self.rng = random.Random(seed)
        
        self.task_dao = TaskDAO()
        self.clip_dao = ClipDAO(seed=seed)
        self.reward_engine = RewardEngine()
        
        self.config = self.task_dao.get_task_config(self.task_name)
        self.state_manager: Optional[StateManager] = None

    def reset(self) -> Observation:
        clip_count = self.config["clip_count"]
        clips = self.clip_dao.generate_clips(clip_count)
        
        duration_range = self.config["target_duration_range"]
        target_dur = self.rng.uniform(duration_range[0], duration_range[1])
        
        styles = ["highlight", "emotional", "balanced"]
        style = self.rng.choice(styles)
        
        self.state_manager = StateManager(target_duration=target_dur, style=style)
        self.state_manager.set_clips(clips)
        
        if self.config["difficulty"] == "hard":
            initial_clip = self.rng.choice(clips)
            self.state_manager.select_clip(initial_clip.id)

        return self.state_manager.get_observation()

    def state(self) -> Observation:
        if not self.state_manager:
            raise ValueError("Environment not initialized. Call reset() first.")
        return self.state_manager.get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.state_manager.increment_step()
        done = False
        info = {}

        action_type = action.action_type
        if action_type == "select":
            self.state_manager.select_clip(action.clip_id)
        elif action_type == "remove":
            self.state_manager.remove_clip(action.clip_id)
        elif action_type == "reorder":
            if action.params and "i" in action.params and "j" in action.params:
                self.state_manager.reorder_clips(action.params["i"], action.params["j"])
        elif action_type == "trim":
            if action.params and "duration" in action.params:
                self.state_manager.trim_clip(action.clip_id, action.params["duration"])
        elif action_type == "finish":
            done = True

        obs = self.state_manager.get_observation()
        
        reward = self.reward_engine.compute_reward(
            timeline=self.state_manager.timeline, 
            clips_map=self.state_manager.available_clips, 
            target_time=self.state_manager.target_duration, 
            style=self.state_manager.style,
            step_count=self.state_manager.step_count
        )

        # Basic task completion check
        if obs.remaining_time <= 0:
            done = True

        return (obs, reward, done, info)
