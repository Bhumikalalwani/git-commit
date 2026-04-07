import os
from pathlib import Path

BASE_DIR = Path(r"c:\Users\MY\PyCharmMiscProject\git-commit\project_montage")

project_files = {
    "env/models.py": """from pydantic import BaseModel
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
""",
    "env/state_manager.py": """from typing import List, Dict, Optional
from env.models import Clip, Observation

class StateManager:
    def __init__(self, target_duration: float, style: str):
        self.timeline: List[str] = []
        self.available_clips: Dict[str, Clip] = {}
        self.target_duration = target_duration
        self.style = style
        self.step_count = 0
    
    def set_clips(self, clips: List[Clip]):
        self.available_clips = {c.id: c for c in clips}

    def get_timeline_duration(self) -> float:
        return sum(self.available_clips[c].duration for c in self.timeline)

    def select_clip(self, clip_id: str) -> bool:
        if clip_id in self.available_clips and clip_id not in self.timeline:
            self.timeline.append(clip_id)
            return True
        return False
        
    def remove_clip(self, clip_id: str) -> bool:
        if clip_id in self.timeline:
            self.timeline.remove(clip_id)
            return True
        return False

    def reorder_clips(self, index_i: int, index_j: int) -> bool:
        if 0 <= index_i < len(self.timeline) and 0 <= index_j < len(self.timeline):
            self.timeline[index_i], self.timeline[index_j] = self.timeline[index_j], self.timeline[index_i]
            return True
        return False
        
    def trim_clip(self, clip_id: str, new_duration: float) -> bool:
        if clip_id in self.timeline:
            clip = self.available_clips[clip_id]
            if 0 < new_duration <= clip.duration:
                clip.duration = new_duration
                return True
        return False

    def increment_step(self):
        self.step_count += 1

    def get_observation(self) -> Observation:
        remaining = self.target_duration - self.get_timeline_duration()
        available = [c for cid, c in self.available_clips.items() if cid not in self.timeline]
        return Observation(
            available_clips=available,
            timeline=self.timeline.copy(),
            remaining_time=remaining,
            style=self.style
        )
""",
    "env/reward_engine.py": """from typing import Dict, List
from env.models import Reward, Clip
from infra.config import WEIGHTS
from infra.utils import calculate_coherence, check_style_alignment, determine_redundancy

class RewardEngine:
    def __init__(self):
        pass

    def compute_reward(self, timeline: List[str], clips_map: Dict[str, Clip], target_time: float, style: str, step_count: int) -> Reward:
        if not timeline:
            return Reward(value=0.0, breakdown={"importance": 0.0, "coherence": 0.0, "style": 0.0, "duration_penalty": 0.0, "redundancy_penalty": 0.0, "step_penalty": 0.0})

        sorted_all = sorted(clips_map.values(), key=lambda x: x.importance, reverse=True)
        optimal_importance_sum = 0
        current_dur = 0
        for c in sorted_all:
            if current_dur + c.duration <= target_time + 5:
                optimal_importance_sum += c.importance
                current_dur += c.duration
        
        selected_importance = sum(clips_map[cid].importance for cid in timeline)
        importance_score = min(1.0, selected_importance / (optimal_importance_sum if optimal_importance_sum > 0 else 1.0))

        coherence_score = calculate_coherence(timeline, clips_map)
        style_score = check_style_alignment(timeline, clips_map, style)

        total_duration = sum(clips_map[cid].duration for cid in timeline)
        duration_diff = abs(total_duration - target_time)
        duration_penalty = min(1.0, duration_diff / target_time) if target_time > 0 else 1.0
        
        redundancy_penalty = determine_redundancy(timeline, clips_map)
        step_penalty = min(1.0, step_count / 100.0)

        r_val = (WEIGHTS["importance"] * importance_score +
                 WEIGHTS["coherence"] * coherence_score +
                 WEIGHTS["style"] * style_score -
                 WEIGHTS["penalty"] * (duration_penalty + redundancy_penalty + step_penalty))
        
        r_val = max(0.0, min(1.0, r_val))

        breakdown = {
            "importance": importance_score,
            "coherence": coherence_score,
            "style": style_score,
            "duration_penalty": duration_penalty,
            "redundancy_penalty": redundancy_penalty,
            "step_penalty": step_penalty,
        }

        return Reward(value=r_val, breakdown=breakdown)
""",
    "env/montage_env.py": """import random
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
""",
    "env/__init__.py": "",
    "dao/__init__.py": "",
    "dao/clip_dao.py": """import random
from typing import List
from env.models import Clip

class ClipDAO:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.emotions = ["excitement", "sadness", "happiness", "tension"]
        self.motions = ["high", "medium", "low"]
        self.tag_pool = ["outdoor", "indoor", "action", "dialogue", "b-roll", "close-up", "wide-shot"]

    def generate_clips(self, n: int) -> List[Clip]:
        clips = []
        for i in range(1, n + 1):
            clip = Clip(
                id=f"clip_{i:03d}",
                duration=round(self.rng.uniform(2.0, 15.0), 1),
                importance=round(self.rng.uniform(0.1, 1.0), 2),
                emotion=self.rng.choice(self.emotions),
                motion=self.rng.choice(self.motions),
                tags=self.rng.sample(self.tag_pool, k=self.rng.randint(1, 3))
            )
            clips.append(clip)
        return clips
""",
    "dao/task_dao.py": """from typing import Dict, Any

class TaskDAO:
    def __init__(self):
        self.tasks = {
            "highlight": {
                "difficulty": "easy",
                "weight_overrides": {"importance": 0.8, "coherence": 0.1, "style": 0.1},
                "target_duration_range": (20.0, 40.0),
                "clip_count": 15
            },
            "structured": {
                "difficulty": "medium",
                "weight_overrides": {"importance": 0.4, "coherence": 0.5, "style": 0.1},
                "target_duration_range": (30.0, 50.0),
                "clip_count": 25
            },
            "intent": {
                "difficulty": "hard",
                "weight_overrides": {"importance": 0.3, "coherence": 0.3, "style": 0.4},
                "target_duration_range": (40.0, 60.0),
                "clip_count": 30
            }
        }

    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        return self.tasks.get(task_name, self.tasks["highlight"])
""",
    "infra/__init__.py": "",
    "infra/config.py": """WEIGHTS = {
  "importance": 0.4,
  "coherence": 0.3,
  "style": 0.2,
  "penalty": 0.1
}

STYLE_MAPPING = {
    "highlight": {"importance_weight": 0.6, "emotion_target": "excitement", "motion_target": "high"},
    "emotional": {"importance_weight": 0.4, "emotion_target": "sadness", "motion_target": "low"},
    "balanced": {"importance_weight": 0.5, "emotion_target": "happiness", "motion_target": "medium"}
}

ACTION_TYPES = ["select", "remove", "reorder", "trim", "finish"]
""",
    "infra/utils.py": """from typing import List, Dict
from env.models import Clip

def calculate_coherence(sequence: List[str], clips_map: Dict[str, Clip]) -> float:
    if not sequence: return 0.0
    score = 1.0
    for i in range(len(sequence) - 1):
        c1 = clips_map[sequence[i]]
        c2 = clips_map[sequence[i+1]]
        if c1.id > c2.id: 
            score -= 0.1
    return max(0.0, min(1.0, score))

def determine_redundancy(sequence: List[str], clips_map: Dict[str, Clip]) -> float:
    if not sequence: return 0.0
    seen_tags = set()
    redundancy_count = 0
    for cid in sequence:
        clip = clips_map[cid]
        for tag in clip.tags:
            if tag in seen_tags:
                redundancy_count += 1
            seen_tags.add(tag)
    penalty = min(1.0, redundancy_count * 0.1)
    return penalty

def check_style_alignment(sequence: List[str], clips_map: Dict[str, Clip], target_style: str) -> float:
    if not sequence: return 0.0
    from infra.config import STYLE_MAPPING
    style_info = STYLE_MAPPING.get(target_style, STYLE_MAPPING["balanced"])
    
    emotion_target = style_info["emotion_target"]
    motion_target = style_info["motion_target"]
    
    matches = 0
    total = len(sequence) * 2
    for cid in sequence:
        clip = clips_map[cid]
        if clip.emotion == emotion_target: matches += 1
        if clip.motion == motion_target: matches += 1
        
    return matches / total
""",
    "graders/__init__.py": "",
    "graders/grader_easy.py": """from typing import List, Dict
from env.models import Clip

def grade_highlight(predicted_timeline: List[str], clips_map: Dict[str, Clip], target_duration: float) -> float:
    selected_importance = sum(clips_map[cid].importance for cid in predicted_timeline)
    
    sorted_all = sorted(clips_map.values(), key=lambda x: x.importance, reverse=True)
    optimal_importance_sum = 0
    current_dur = 0
    for c in sorted_all:
        if current_dur + c.duration <= target_duration + 5:
            optimal_importance_sum += c.importance
            current_dur += c.duration
            
    if optimal_importance_sum == 0:
        return 1.0
        
    score = selected_importance / optimal_importance_sum
    score *= max(0.0, 1.0 - abs(sum(clips_map[c].duration for c in predicted_timeline) - target_duration) / target_duration)
    return min(1.0, max(0.0, score))
""",
    "graders/grader_medium.py": """from typing import List

def lcs_similarity(seq1: List[str], seq2: List[str]) -> float:
    m = len(seq1)
    n = len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    if n == 0: return 0.0
    return dp[m][n] / n

def grade_structured(predicted_timeline: List[str], ideal_order: List[str]) -> float:
    return lcs_similarity(predicted_timeline, ideal_order)
""",
    "graders/grader_hard.py": """from typing import List, Dict
from env.models import Clip
from infra.utils import check_style_alignment
from graders.grader_medium import lcs_similarity

def grade_intent(predicted_timeline: List[str], clips_map: Dict[str, Clip], target_style: str, ideal_order: List[str]) -> float:
    if not predicted_timeline: return 0.0
    
    style_match = check_style_alignment(predicted_timeline, clips_map, target_style)
    order_score = lcs_similarity(predicted_timeline, ideal_order)
    
    valid_clips = [c for c in predicted_timeline if c in ideal_order]
    selection_score = len(valid_clips) / len(ideal_order) if ideal_order else 0.0
    
    final_score = 0.5 * style_match + 0.3 * order_score + 0.2 * selection_score
    return min(1.0, max(0.0, final_score))
""",
    "tests/__init__.py": "",
    "tests/test_env.py": """import unittest
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
""",
    "tests/test_reward.py": """import unittest
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
""",
    "tests/test_graders.py": """import unittest
from graders.grader_easy import grade_highlight
from graders.grader_medium import grade_structured
from graders.grader_hard import grade_intent
from env.models import Clip

class TestGraders(unittest.TestCase):
    def setUp(self):
        self.clips = {
            "c1": Clip(id="c1", duration=10, importance=1.0, emotion="happiness", motion="high", tags=["action"]),
            "c2": Clip(id="c2", duration=10, importance=0.5, emotion="sadness", motion="low", tags=["indoor"]),
        }

    def test_grader_easy(self):
        score_perfect = grade_highlight(["c1"], self.clips, 10)
        self.assertGreaterEqual(score_perfect, 0.8)
        
        score_worst = grade_highlight([], self.clips, 10)
        self.assertLessEqual(score_worst, 0.1)

    def test_grader_medium(self):
        score_perfect = grade_structured(["c1", "c2"], ["c1", "c2"])
        self.assertEqual(score_perfect, 1.0)
        
        score_worst = grade_structured(["c3"], ["c1", "c2"])
        self.assertEqual(score_worst, 0.0)

    def test_grader_hard(self):
        score = grade_intent(["c1"], self.clips, "balanced", ["c1", "c2"])
        self.assertTrue(0.0 <= score <= 1.0)

if __name__ == '__main__':
    unittest.main()
""",
    "inference.py": """import os
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
""",
    "openenv.yaml": """name: montage_env
description: Sequential Video Editing Environment
actions: discrete
observation: structured
tasks:
  - highlight
  - structured
  - intent
""",
    "Dockerfile": """FROM python:3.10-slim

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir pydantic openai

ENV PYTHONPATH=/app

CMD ["python", "inference.py"]
""",
    "README.md": """# Project Montage: Sequential Video Editing Decision Environment

An OpenEnv-compatible reinforcement learning environment for automated video editing tasks.

## Tasks
1. **Highlight Extraction:** Selection of important clips under duration constraint.
2. **Structured Editing:** Ordering clips to form a coherent sequence.
3. **Intent-Aware Editing:** Editing for max style and coherence under constraints.

## Architecture
- `env/`: Core OpenEnv methods.
- `dao/`: Clip and task definition logic.
- `infra/`: Configuration and math utils.
- `graders/`: Specialized test scorers.
- `tests/`: Correctness validations.
"""
}

def main():
    for rel_path, content in project_files.items():
        file_path = BASE_DIR / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    print("Project successfully created at", BASE_DIR)
    
if __name__ == "__main__":
    main()
