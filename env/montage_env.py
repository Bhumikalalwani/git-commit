from __future__ import annotations

import random
from typing import Any, Dict, Optional

from dao.clip_dao import ClipDAO
from dao.task_dao import TaskDAO
from env.models import Action, ActionType, Observation, RewardBreakdown, State
from env.reward_engine import RewardEngine
from env.state_manager import StateManager


class MontageEnv:
    """OpenEnv-compliant video montage editing environment.

    Provides reset(), step(action), and state property following
    the OpenEnv specification.
    """

    def __init__(self, task_name: str = "highlight", seed: int = 42):
        self.task_name = task_name
        self.seed = seed
        self.rng = random.Random(seed)

        self.task_dao = TaskDAO()
        self.clip_dao = ClipDAO(seed=seed)
        self.reward_engine = RewardEngine()

        self.config = self.task_dao.get_task_config(self.task_name)
        self.state_manager: Optional[StateManager] = None
        self._done = False
        self._last_reward: Optional[float] = None
        self._last_breakdown: Optional[RewardBreakdown] = None

    def reset(self, seed: int | None = None, **kwargs: Any) -> Observation:
        """Initialize a new episode and return the initial observation."""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)
            self.clip_dao = ClipDAO(seed=seed)

        clip_count = self.config["clip_count"]
        clips = self.clip_dao.generate_clips(clip_count)

        duration_range = self.config["target_duration_range"]
        target_dur = round(self.rng.uniform(duration_range[0], duration_range[1]), 1)

        styles = ["highlight", "emotional", "balanced"]
        style = self.rng.choice(styles)

        self.state_manager = StateManager(target_duration=target_dur, style=style)
        self.state_manager.set_clips(clips)

        if self.config["difficulty"] == "hard":
            initial_clip = self.rng.choice(clips)
            self.state_manager.select_clip(initial_clip.id)

        self._done = False
        self._last_reward = None
        self._last_breakdown = None

        return self.state_manager.get_observation(done=False, reward=None, breakdown=None)

    @property
    def state(self) -> State:
        """Return the full internal state snapshot."""
        if not self.state_manager:
            raise ValueError("Environment not initialized. Call reset() first.")
        return State(
            task_name=self.task_name,
            seed=self.seed,
            step_count=self.state_manager.step_count,
            timeline=list(self.state_manager.timeline),
            available_clip_ids=[
                cid for cid in self.state_manager.available_clips
                if cid not in self.state_manager.timeline
            ],
            target_duration=self.state_manager.target_duration,
            current_duration=self.state_manager.get_timeline_duration(),
            style=self.state_manager.style,
            done=self._done,
            last_reward=self._last_reward,
            last_reward_breakdown=self._last_breakdown,
        )

    def step(self, action: Action) -> Observation:
        """Execute an action and return the resulting observation.

        The returned Observation includes done, reward, and reward_breakdown
        fields per the OpenEnv specification.
        """
        if not self.state_manager:
            raise ValueError("Environment not initialized. Call reset() first.")
        if self._done:
            return self.state_manager.get_observation(
                done=True, reward=self._last_reward, breakdown=self._last_breakdown,
            )

        self.state_manager.increment_step()
        info: Dict[str, Any] = {}

        if action.action_type == ActionType.SELECT:
            success = self.state_manager.select_clip(action.clip_id)
            info["action_success"] = success
        elif action.action_type == ActionType.REMOVE:
            success = self.state_manager.remove_clip(action.clip_id)
            info["action_success"] = success
        elif action.action_type == ActionType.REORDER:
            if action.params and "i" in action.params and "j" in action.params:
                success = self.state_manager.reorder_clips(
                    int(action.params["i"]), int(action.params["j"]),
                )
                info["action_success"] = success
        elif action.action_type == ActionType.TRIM:
            if action.params and "duration" in action.params:
                success = self.state_manager.trim_clip(
                    action.clip_id, float(action.params["duration"]),
                )
                info["action_success"] = success
        elif action.action_type == ActionType.FINISH:
            self._done = True

        weight_overrides = self.config.get("weight_overrides")
        reward_val, breakdown = self.reward_engine.compute_reward(
            timeline=self.state_manager.timeline,
            clips_map=self.state_manager.available_clips,
            target_time=self.state_manager.target_duration,
            style=self.state_manager.style,
            step_count=self.state_manager.step_count,
            weight_overrides=weight_overrides,
        )

        obs = self.state_manager.get_observation(done=False, reward=reward_val, breakdown=breakdown)

        if obs.remaining_time <= 0:
            self._done = True

        self._last_reward = reward_val
        self._last_breakdown = breakdown

        obs.done = self._done
        obs.metadata.update(info)
        return obs
