from __future__ import annotations

from typing import Any, Dict


class TaskDAO:
    TASKS: Dict[str, Dict[str, Any]] = {
        "highlight": {
            "difficulty": "easy",
            "description": "Select the highest-importance clips that fit within a duration budget.",
            "weight_overrides": {"importance": 0.8, "coherence": 0.1, "style": 0.1},
            "target_duration_range": (20.0, 40.0),
            "clip_count": 15,
        },
        "structured": {
            "difficulty": "medium",
            "description": "Arrange clips into a coherent narrative sequence under duration constraints.",
            "weight_overrides": {"importance": 0.4, "coherence": 0.5, "style": 0.1},
            "target_duration_range": (30.0, 50.0),
            "clip_count": 25,
        },
        "intent": {
            "difficulty": "hard",
            "description": "Style-aware editing: select, order, and trim clips to match a target style.",
            "weight_overrides": {"importance": 0.3, "coherence": 0.3, "style": 0.4},
            "target_duration_range": (40.0, 60.0),
            "clip_count": 30,
        },
    }

    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        return self.TASKS.get(task_name, self.TASKS["highlight"])

    def list_tasks(self) -> list[str]:
        return list(self.TASKS.keys())
