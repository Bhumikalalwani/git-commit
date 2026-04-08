WEIGHTS = {
    "importance": 0.4,
    "coherence": 0.3,
    "style": 0.2,
    "penalty": 0.1,
}

STYLE_MAPPING = {
    "highlight": {"importance_weight": 0.6, "emotion_target": "excitement", "motion_target": "high"},
    "emotional": {"importance_weight": 0.4, "emotion_target": "sadness", "motion_target": "low"},
    "balanced": {"importance_weight": 0.5, "emotion_target": "happiness", "motion_target": "medium"},
}

ACTION_TYPES = ["select", "remove", "reorder", "trim", "finish"]
