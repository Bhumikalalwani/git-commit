from typing import List, Dict
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
