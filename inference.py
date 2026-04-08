#!/usr/bin/env python3
"""OpenAI-based inference runner for the Montage OpenEnv environment.

Environment variables (mandatory for evaluation):
    API_BASE_URL  — OpenAI-compatible API endpoint
    MODEL_NAME    — model identifier to use for inference
    HF_TOKEN      — API key / bearer token for the LLM provider

The script emits structured stdout logs in per-task blocks:
    [START] task=NAME env=NAME model=NAME
    [STEP]  step=N action=JSON reward=X.XX done=true/false error=null
    [END]   success=true/false steps=N score=X.XXX rewards=R1,R2,...
"""

from __future__ import annotations

import json
import os
import sys
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
SCORE_FLOOR = 0.001
SCORE_CEILING = 0.999

SYSTEM_PROMPT = """You are an AI video editor building a montage. You receive:
- available_clips: clips you can add (id, duration, importance, emotion, motion, tags)
- timeline: clip IDs currently on the timeline
- remaining_time: seconds of budget left
- style: target style (highlight/emotional/balanced)
- step: current step number

Rules:
1. ONLY use "select" to add clips and "finish" to end. Do NOT use "remove".
2. Pick clips that match the style AND have high importance.
3. Stop selecting when remaining_time is near 0 or negative.
4. Call "finish" when you have a good timeline (3-8 clips typically).
5. For "highlight" style: maximize importance scores.
6. For "emotional" style: prefer sadness emotion + low motion.
7. For "balanced" style: prefer happiness emotion + medium motion.

Respond with ONLY a JSON object (no markdown):
{"action_type":"select|finish","clip_id":"clip_XXX or null","params":null}"""


def _strict_score(value: float) -> float:
    score = round(float(value), 3)
    if score <= SCORE_FLOOR:
        return SCORE_FLOOR
    if score >= SCORE_CEILING:
        return SCORE_CEILING
    return score


def _log_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _load_config() -> Tuple[Dict[str, str], List[str]]:
    api_base = os.getenv("API_BASE_URL", "").strip() or "https://api.openai.com/v1"
    model = os.getenv("MODEL_NAME", "").strip() or "gpt-4o-mini"
    hf_token = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
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


def _llm_action(client: "OpenAI", model: str, obs, step_num: int) -> Action:
    """Ask the LLM to pick an action given the current observation."""
    sorted_clips = sorted(obs.available_clips, key=lambda c: c.importance, reverse=True)[:10]
    obs_summary = {
        "available_clips": [
            {"id": c.id, "dur": c.duration, "imp": round(c.importance, 2),
             "emo": c.emotion, "mot": c.motion}
            for c in sorted_clips
        ],
        "timeline": obs.timeline,
        "remaining_time": round(obs.remaining_time, 1),
        "style": obs.style,
        "step": step_num,
    }

    if obs.remaining_time <= 0 or not obs.available_clips:
        return Action(action_type=ActionType.FINISH)

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=128,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(obs_summary, separators=(",", ":"))},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    parsed = _extract_json(content)

    action_type_str = parsed.get("action_type", "finish")
    if action_type_str not in ("select", "finish"):
        action_type_str = "finish"

    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        action_type = ActionType.FINISH

    return Action(
        action_type=action_type,
        clip_id=parsed.get("clip_id"),
        params=parsed.get("params"),
    )


def _format_action(action: Action) -> str:
    d: Dict[str, Any] = {"action_type": action.action_type.value}
    if action.clip_id:
        d["clip_id"] = action.clip_id
    return json.dumps(d, separators=(",", ":"))


def run_task(
    client: "OpenAI | None",
    model_name: str,
    task_id: int,
) -> Dict[str, Any]:
    task_name = TASK_MAP[task_id]
    difficulty = TASK_DIFFICULTY[task_id]

    _log_start(task_name, model_name)

    env = MontageEnv(task_name=task_name, seed=SEED)
    obs = env.reset()

    rewards: List[float] = []
    step_count = 0
    done = False
    success = False
    score = SCORE_FLOOR

    try:
        while not done and step_count < MAX_STEPS:
            error_msg: str | None = None
            if client and client.api_key != "dummy-token":
                try:
                    action = _llm_action(client, model_name, obs, step_count + 1)
                except Exception as exc:
                    error_msg = str(exc)
                    action = _greedy_action(obs)
            else:
                action = _greedy_action(obs)

            obs = env.step(action)
            reward = _strict_score(obs.reward if obs.reward is not None else SCORE_FLOOR)
            done = obs.done
            step_count += 1
            rewards.append(reward)

            _log_step(step_count, _format_action(action), reward, done, error_msg)

        if not done:
            action = Action(action_type=ActionType.FINISH)
            obs = env.step(action)
            step_count += 1
            reward = _strict_score(obs.reward if obs.reward is not None else SCORE_FLOOR)
            rewards.append(reward)
            _log_step(step_count, _format_action(action), reward, True, None)

        state = env.state
        clips_map = env.state_manager.available_clips
        timeline = list(state.timeline)
        target_dur = state.target_duration
        style = state.style
        ideal_order = _compute_ideal_order(clips_map, style, target_dur)

        if task_name == "highlight":
            score = _strict_score(grade_highlight(timeline, clips_map, target_dur))
        elif task_name == "structured":
            score = _strict_score(grade_structured(timeline, ideal_order))
        elif task_name == "intent":
            score = _strict_score(grade_intent(timeline, clips_map, style, ideal_order))
        else:
            score = _strict_score(rewards[-1] if rewards else SCORE_FLOOR)

        success = done and score > 0.01

    except Exception as exc:
        score = SCORE_FLOOR
        _log_step(step_count + 1, '{"action_type":"finish"}', 0.01, True, str(exc))
        rewards.append(0.01)
        step_count += 1

    _log_end(success=success, steps=step_count, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "task_name": task_name,
        "difficulty": difficulty,
        "steps": step_count,
        "score": score,
        "success": success,
    }


def main() -> int:
    config, _ = _load_config()
    client = _build_client(config)
    model_name = config["MODEL_NAME"]

    for task_id in TASK_IDS:
        run_task(client, model_name, task_id)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        for tid in TASK_IDS:
            tname = TASK_MAP[tid]
            _log_start(tname, os.getenv("MODEL_NAME", "gpt-4o-mini"))
            _log_step(1, '{"action_type":"finish"}', 0.01, True, str(exc))
            _log_end(success=False, steps=1, score=SCORE_FLOOR, rewards=[0.01])
        sys.exit(0)
