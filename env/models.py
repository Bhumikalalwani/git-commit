from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    SELECT = "select"
    REMOVE = "remove"
    REORDER = "reorder"
    TRIM = "trim"
    FINISH = "finish"


class Clip(BaseModel):
    id: str
    duration: float
    importance: float = Field(ge=0.0, le=1.0)
    emotion: str
    motion: str
    tags: List[str]


class Action(BaseModel):
    action_type: ActionType
    clip_id: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class RewardBreakdown(BaseModel):
    importance: float = 0.0
    coherence: float = 0.0
    style: float = 0.0
    duration_penalty: float = 0.0
    redundancy_penalty: float = 0.0
    step_penalty: float = 0.0


class Observation(BaseModel):
    """OpenEnv-compliant observation returned by reset() and step()."""
    available_clips: List[Clip]
    timeline: List[str]
    remaining_time: float
    style: str
    done: bool = False
    reward: Optional[float] = None
    reward_breakdown: Optional[RewardBreakdown] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    """Full internal state snapshot for the OpenEnv state() endpoint."""
    task_name: str
    seed: int
    step_count: int
    timeline: List[str]
    available_clip_ids: List[str]
    target_duration: float
    current_duration: float
    style: str
    done: bool
    last_reward: Optional[float] = None
    last_reward_breakdown: Optional[RewardBreakdown] = None


class ResetRequest(BaseModel):
    task_name: str = "highlight"
    seed: int = 42


class StepRequest(BaseModel):
    action: Action
