from __future__ import annotations

from typing import Dict, List

from env.models import Clip, Observation, RewardBreakdown


class StateManager:
    def __init__(self, target_duration: float, style: str):
        self.timeline: List[str] = []
        self.available_clips: Dict[str, Clip] = {}
        self.target_duration = target_duration
        self.style = style
        self.step_count = 0

    def set_clips(self, clips: List[Clip]) -> None:
        self.available_clips = {c.id: c for c in clips}

    def get_timeline_duration(self) -> float:
        return sum(self.available_clips[c].duration for c in self.timeline if c in self.available_clips)

    def select_clip(self, clip_id: str) -> bool:
        if clip_id and clip_id in self.available_clips and clip_id not in self.timeline:
            self.timeline.append(clip_id)
            return True
        return False

    def remove_clip(self, clip_id: str) -> bool:
        if clip_id and clip_id in self.timeline:
            self.timeline.remove(clip_id)
            return True
        return False

    def reorder_clips(self, index_i: int, index_j: int) -> bool:
        if 0 <= index_i < len(self.timeline) and 0 <= index_j < len(self.timeline):
            self.timeline[index_i], self.timeline[index_j] = (
                self.timeline[index_j],
                self.timeline[index_i],
            )
            return True
        return False

    def trim_clip(self, clip_id: str, new_duration: float) -> bool:
        if clip_id and clip_id in self.available_clips and clip_id in self.timeline:
            clip = self.available_clips[clip_id]
            if 0 < new_duration <= clip.duration:
                self.available_clips[clip_id] = clip.model_copy(update={"duration": new_duration})
                return True
        return False

    def increment_step(self) -> None:
        self.step_count += 1

    def get_observation(
        self,
        done: bool = False,
        reward: float | None = None,
        breakdown: RewardBreakdown | None = None,
    ) -> Observation:
        remaining = self.target_duration - self.get_timeline_duration()
        available = [c for cid, c in self.available_clips.items() if cid not in self.timeline]
        return Observation(
            available_clips=available,
            timeline=list(self.timeline),
            remaining_time=remaining,
            style=self.style,
            done=done,
            reward=reward,
            reward_breakdown=breakdown,
            metadata={
                "step_count": self.step_count,
                "current_duration": self.get_timeline_duration(),
                "target_duration": self.target_duration,
            },
        )
