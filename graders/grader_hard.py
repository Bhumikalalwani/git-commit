from __future__ import annotations

from typing import Dict, List

from env.models import Clip
from graders.grader_medium import lcs_similarity
from infra.utils import check_style_alignment


def grade_intent(
    predicted_timeline: List[str],
    clips_map: Dict[str, Clip],
    target_style: str,
    ideal_order: List[str],
) -> float:
    """Grade the intent-aware editing task (hard).

    Combines three sub-scores:
        50 %  style alignment with the target editing style
        30 %  ordering similarity (LCS) against the ideal sequence
        20 %  selection coverage of ideal clips

    Returns a score in [0.0, 1.0].
    """
    if not predicted_timeline:
        return 0.0

    style_match = check_style_alignment(predicted_timeline, clips_map, target_style)
    order_score = lcs_similarity(predicted_timeline, ideal_order)

    valid_clips = [c for c in predicted_timeline if c in ideal_order]
    selection_score = len(valid_clips) / max(len(ideal_order), 1)

    final_score = 0.5 * style_match + 0.3 * order_score + 0.2 * selection_score
    return round(min(1.0, max(0.0, final_score)), 4)
