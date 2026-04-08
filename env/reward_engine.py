from __future__ import annotations

from typing import Dict, List, Optional

from env.models import Clip, RewardBreakdown
from infra.config import WEIGHTS
from infra.utils import calculate_coherence, check_style_alignment, determine_redundancy


class RewardEngine:
    def compute_reward(
        self,
        timeline: List[str],
        clips_map: Dict[str, Clip],
        target_time: float,
        style: str,
        step_count: int,
        weight_overrides: Optional[Dict[str, float]] = None,
    ) -> tuple[float, RewardBreakdown]:
        if not timeline:
            return 0.001, RewardBreakdown()

        w = dict(WEIGHTS)
        if weight_overrides:
            w["importance"] = weight_overrides.get("importance", w["importance"])
            w["coherence"] = weight_overrides.get("coherence", w["coherence"])
            w["style"] = weight_overrides.get("style", w["style"])

        sorted_all = sorted(clips_map.values(), key=lambda x: x.importance, reverse=True)
        optimal_importance_sum = 0.0
        current_dur = 0.0
        for c in sorted_all:
            if current_dur + c.duration <= target_time + 5:
                optimal_importance_sum += c.importance
                current_dur += c.duration

        selected_importance = sum(clips_map[cid].importance for cid in timeline if cid in clips_map)
        importance_score = min(1.0, selected_importance / max(optimal_importance_sum, 1e-9))

        coherence_score = calculate_coherence(timeline, clips_map)
        style_score = check_style_alignment(timeline, clips_map, style)

        total_duration = sum(clips_map[cid].duration for cid in timeline if cid in clips_map)
        duration_diff = abs(total_duration - target_time)
        duration_penalty = min(1.0, duration_diff / max(target_time, 1e-9))

        redundancy_penalty = determine_redundancy(timeline, clips_map)
        step_penalty = min(1.0, step_count / 100.0)

        r_val = (
            w["importance"] * importance_score
            + w["coherence"] * coherence_score
            + w["style"] * style_score
            - w["penalty"] * (duration_penalty + redundancy_penalty + step_penalty)
        )
        r_val = max(0.001, min(0.999, r_val))

        breakdown = RewardBreakdown(
            importance=round(importance_score, 4),
            coherence=round(coherence_score, 4),
            style=round(style_score, 4),
            duration_penalty=round(duration_penalty, 4),
            redundancy_penalty=round(redundancy_penalty, 4),
            step_penalty=round(step_penalty, 4),
        )
        return round(r_val, 4), breakdown
