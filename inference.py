#!/usr/bin/env python3
"""OpenAI-based inference runner for the Montage OpenEnv environment.

Environment variables (mandatory for evaluation):
    API_BASE_URL  — OpenAI-compatible API endpoint
    MODEL_NAME    — model identifier to use for inference
    HF_TOKEN      — API key / bearer token for the LLM provider

The script emits structured stdout logs strictly in three sections:
    [START] {json}
    [STEP]  {json}
    [END]   {json}
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[assignment,misc]

from dao.clip_dao import ClipDAO
from env.models import Action, ActionType, Clip
from env.montage_env import MontageEnv
from graders.grader_easy import grade_highlight
from graders.grader_hard import grade_intent
from graders.grader_medium import grade_structured
from infra.utils import check_style_alignment

TASK_IDS = (1, 2, 3)
TASK_MAP = {1: "highlight", 2: "structured", 3: "intent"}
TASK_DIFFICULTY = {1: "easy", 2: "medium", 3: "hard"}
BENCHMARK = "montage_env"
SEED = 42
MAX_STEPS = 50
MIN_SCORE = 0.001
MAX_SCORE = 0.999

SYSTEM_PROMPT = """You are an AI video editor agent. You will receive an observation describing:
- available_clips: clips you can add to the timeline (each has id, duration, importance, emotion, motion, tags)
- timeline: clip IDs currently selected (ordered)
- remaining_time: seconds of budget remaining
- style: target editing style (highlight, emotional, or balanced)

Your goal: build the best montage by selecting, removing, reordering, or trimming clips.

Respond ONLY with a JSON object with these exact keys:
{
  "action_type": "select|remove|reorder|trim|finish",
  "clip_id": "clip_XXX or null",
  "params": {"i": 0, "j": 1} or {"duration": 5.0} or null,
  "reasoning": "brief explanation"
}

Strategy hints:
- For "highlight" style: prioritize high-importance clips
- For "emotional" style: prefer sadness emotion + low motion
- For "balanced" style: prefer happiness emotion + medium motion
- Stay within remaining_time budget
- Say "finish" when the timeline is good enough
- Output raw JSON only, no markdown."""


def _log(prefix: str, payload: Dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(payload, ensure_ascii=True, separators=(',', ':'))}", flush=True)


def _load_config() -> Tuple[Dict[str, str], List[str]]:
    api_base = os.getenv("API_BASE_URL", "").strip() or "https://api.openai.com/v1"
    model = os.getenv("MODEL_NAME", "").strip() or "gpt-4o-mini"
    hf_token = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("API_KEY", "").strip()
    )

    warnings: List[str] = []
    if not os.getenv("API_BASE_URL", "").strip():
        warnings.append("API_BASE_URL not set; defaulted to openai")
    if not os.getenv("MODEL_NAME", "").strip():
        warnings.append("MODEL_NAME not set; defaulted to gpt-4o-mini")
    if not hf_token:
        warnings.append("HF_TOKEN not set; running in fallback greedy mode")

    return {"API_BASE_URL": api_base, "MODEL_NAME": model, "HF_TOKEN": hf_token}, warnings


def _build_client(config: Dict[str, str]) -> "OpenAI | None":
    if OpenAI is None:
        return None
    token = config["HF_TOKEN"] if config["HF_TOKEN"] else "dummy-token"
    try:
        return OpenAI(base_url=config["API_BASE_URL"], api_key=token)
    except Exception:
        return None


def _normalize_score(raw: float) -> float:
    return round(min(max(float(raw), MIN_SCORE), MAX_SCORE), 4)


def _compute_ideal_order(clips_map: Dict[str, Clip], style: str, target_duration: float) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for cid, clip in clips_map.items():
        s = check_style_alignment([cid], clips_map, style)
        combined = 0.5 * clip.importance + 0.5 * s
        scored.append((combined, cid))
    scored.sort(key=lambda x: x[0], reverse=True)

    ideal: List[str] = []
    dur = 0.0
    for _score, cid in scored:
        if dur + clips_map[cid].duration <= target_duration + 5:
            ideal.append(cid)
            dur += clips_map[cid].duration
    return ideal


def _greedy_action(obs) -> Action:
    """Deterministic fallback: pick the most important clip that fits."""
    if not obs.available_clips or obs.remaining_time <= 0:
        return Action(action_type=ActionType.FINISH)

    for clip in sorted(obs.available_clips, key=lambda c: c.importance, reverse=True):
        if clip.duration <= obs.remaining_time + 2:
            return Action(action_type=ActionType.SELECT, clip_id=clip.id)

    return Action(action_type=ActionType.FINISH)


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _llm_action(client: "OpenAI", model: str, obs) -> Action:
    """Ask the LLM to pick an action given the current observation."""
    obs_summary = {
        "available_clips": [
            {"id": c.id, "duration": c.duration, "importance": c.importance,
             "emotion": c.emotion, "motion": c.motion, "tags": c.tags}
            for c in obs.available_clips[:15]
        ],
        "timeline": obs.timeline,
        "remaining_time": round(obs.remaining_time, 1),
        "style": obs.style,
    }

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(obs_summary, separators=(",", ":"))},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    parsed = _extract_json(content)

    action_type_str = parsed.get("action_type", "finish")
    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        action_type = ActionType.FINISH

    return Action(
        action_type=action_type,
        clip_id=parsed.get("clip_id"),
        params=parsed.get("params"),
    )


def run_task(
    client: "OpenAI | None",
    model_name: str,
    task_id: int,
) -> Dict[str, Any]:
    task_name = TASK_MAP[task_id]
    difficulty = TASK_DIFFICULTY[task_id]

    env = MontageEnv(task_name=task_name, seed=SEED)
    obs = env.reset()

    rewards: List[float] = []
    step_count = 0
    done = False

    while not done and step_count < MAX_STEPS:
        llm_status = "ok"
        if client and client.api_key != "dummy-token":
            try:
                action = _llm_action(client, model_name, obs)
            except Exception:
                action = _greedy_action(obs)
                llm_status = "fallback"
        else:
            action = _greedy_action(obs)
            llm_status = "greedy"

        obs = env.step(action)
        reward = obs.reward if obs.reward is not None else 0.0
        done = obs.done
        step_count += 1
        rewards.append(reward)

        _log("[STEP]", OrderedDict([
            ("task_id", task_id),
            ("task_name", task_name),
            ("step", step_count),
            ("action", action.action_type.value),
            ("reward", round(reward, 4)),
            ("done", done),
            ("llm_status", llm_status),
        ]))

    if not done:
        obs = env.step(Action(action_type=ActionType.FINISH))
        step_count += 1
        reward = obs.reward if obs.reward is not None else 0.0
        rewards.append(reward)
        _log("[STEP]", OrderedDict([
            ("task_id", task_id),
            ("task_name", task_name),
            ("step", step_count),
            ("action", "finish"),
            ("reward", round(reward, 4)),
            ("done", True),
            ("llm_status", "greedy"),
        ]))

    state = env.state
    clips_map = env.state_manager.available_clips
    timeline = list(state.timeline)
    target_dur = state.target_duration
    style = state.style
    ideal_order = _compute_ideal_order(clips_map, style, target_dur)

    if task_name == "highlight":
        grader_score = grade_highlight(timeline, clips_map, target_dur)
    elif task_name == "structured":
        grader_score = grade_structured(timeline, ideal_order)
    elif task_name == "intent":
        grader_score = grade_intent(timeline, clips_map, style, ideal_order)
    else:
        grader_score = reward

    grader_score = _normalize_score(grader_score)
    final_reward = _normalize_score(rewards[-1] if rewards else 0.0)

    return {
        "task_id": task_id,
        "task_name": task_name,
        "difficulty": difficulty,
        "steps": step_count,
        "total_reward": round(sum(rewards), 4),
        "average_reward": round(sum(rewards) / max(step_count, 1), 4),
        "grader_score": grader_score,
        "final_reward": final_reward,
    }


def run_inference() -> Dict[str, float]:
    config, warnings = _load_config()
    client = _build_client(config)

    _log("[START]", OrderedDict([
        ("script", "inference.py"),
        ("env", BENCHMARK),
        ("api_base_url", config["API_BASE_URL"]),
        ("model_name", config["MODEL_NAME"]),
        ("tasks", list(TASK_IDS)),
        ("seed", SEED),
        ("warnings", warnings),
    ]))

    results: Dict[str, float] = {}
    total_score = 0.0

    for task_id in TASK_IDS:
        task_result = run_task(client, config["MODEL_NAME"], task_id)
        score = task_result["grader_score"]
        results[f"task_{task_id}"] = score
        total_score += score

    average_score = round(total_score / len(TASK_IDS), 4)

    _log("[END]", OrderedDict([
        ("task_results", results),
        ("average_score", average_score),
        ("status", "success"),
    ]))

    return results


if __name__ == "__main__":
    try:
        run_inference()
    except Exception as exc:
        fallback = {"task_1": 0.001, "task_2": 0.001, "task_3": 0.001}
        _log("[END]", OrderedDict([
            ("task_results", fallback),
            ("average_score", 0.001),
            ("status", "error"),
            ("error", str(exc)),
        ]))
        sys.exit(0)
