import random
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
