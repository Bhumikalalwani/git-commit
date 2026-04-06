from typing import List, Dict
from env.models import Clip

def calculate_coherence(sequence: List[str], clips_map: Dict[str, Clip]) -> float:
    if not sequence: return 0.0
    score = 1.0
    for i in range(len(sequence) - 1):
        c1 = clips_map[sequence[i]]
        c2 = clips_map[sequence[i+1]]
        if c1.id > c2.id: 
            score -= 0.1
    return max(0.0, min(1.0, score))

def determine_redundancy(sequence: List[str], clips_map: Dict[str, Clip]) -> float:
    if not sequence: return 0.0
    seen_tags = set()
    redundancy_count = 0
    for cid in sequence:
        clip = clips_map[cid]
        for tag in clip.tags:
            if tag in seen_tags:
                redundancy_count += 1
            seen_tags.add(tag)
    penalty = min(1.0, redundancy_count * 0.1)
    return penalty

def check_style_alignment(sequence: List[str], clips_map: Dict[str, Clip], target_style: str) -> float:
    if not sequence: return 0.0
    from infra.config import STYLE_MAPPING
    style_info = STYLE_MAPPING.get(target_style, STYLE_MAPPING["balanced"])
    
    emotion_target = style_info["emotion_target"]
    motion_target = style_info["motion_target"]
    
    matches = 0
    total = len(sequence) * 2
    for cid in sequence:
        clip = clips_map[cid]
        if clip.emotion == emotion_target: matches += 1
        if clip.motion == motion_target: matches += 1
        
    return matches / total
