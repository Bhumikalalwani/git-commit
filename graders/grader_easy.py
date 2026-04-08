from __future__ import annotations

from typing import Dict, List

from env.models import Clip


def grade_highlight(predicted_timeline: List[str], clips_map: Dict[str, Clip], target_duration: float) -> float:
    """Grade the highlight extraction task (easy).

    Measures what fraction of the optimal importance budget was captured,
    penalized by how far total duration deviates from the target.
    Returns a score in [0.0, 1.0].
    """
    if not predicted_timeline:
        return 0.0

    selected_importance = sum(
        clips_map[cid].importance for cid in predicted_timeline if cid in clips_map
    )

    sorted_all = sorted(clips_map.values(), key=lambda x: x.importance, reverse=True)
    optimal_importance_sum = 0.0
    current_dur = 0.0
    for c in sorted_all:
        if current_dur + c.duration <= target_duration + 5:
            optimal_importance_sum += c.importance
            current_dur += c.duration

    if optimal_importance_sum == 0:
        return 1.0

    score = selected_importance / optimal_importance_sum

    total_dur = sum(clips_map[c].duration for c in predicted_timeline if c in clips_map)
    duration_factor = max(0.0, 1.0 - abs(total_dur - target_duration) / max(target_duration, 1e-9))
    score *= duration_factor

    return min(1.0, max(0.0, round(score, 4)))
