from typing import List, Dict
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
