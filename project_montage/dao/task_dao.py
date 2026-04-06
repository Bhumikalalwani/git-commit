from typing import Dict, Any

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
