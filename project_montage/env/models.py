from pydantic import BaseModel
from typing import List, Dict, Optional

class Clip(BaseModel):
    id: str
    duration: float
    importance: float
    emotion: str
    motion: str
    tags: List[str]

class Observation(BaseModel):
    available_clips: List[Clip]
    timeline: List[str]
    remaining_time: float
    style: str

class Action(BaseModel):
    action_type: str   # select, remove, reorder, trim, finish
    clip_id: Optional[str] = None
    params: Optional[Dict] = None

class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float]
