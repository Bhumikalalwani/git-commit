from __future__ import annotations

import random
from typing import List

from env.models import Clip


class ClipDAO:
    EMOTIONS = ["excitement", "sadness", "happiness", "tension"]
    MOTIONS = ["high", "medium", "low"]
    TAG_POOL = ["outdoor", "indoor", "action", "dialogue", "b-roll", "close-up", "wide-shot"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_clips(self, n: int) -> List[Clip]:
        clips: List[Clip] = []
        for i in range(1, n + 1):
            clip = Clip(
                id=f"clip_{i:03d}",
                duration=round(self.rng.uniform(2.0, 15.0), 1),
                importance=round(self.rng.uniform(0.1, 1.0), 2),
                emotion=self.rng.choice(self.EMOTIONS),
                motion=self.rng.choice(self.MOTIONS),
                tags=self.rng.sample(self.TAG_POOL, k=self.rng.randint(1, 3)),
            )
            clips.append(clip)
        return clips
