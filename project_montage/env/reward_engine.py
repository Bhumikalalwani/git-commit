from typing import Dict, List
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
