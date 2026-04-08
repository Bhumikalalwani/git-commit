"""OpenEnv-compliant FastAPI server for the Montage video editing environment.

Endpoints:
    POST /reset   — Start a new episode
    POST /step    — Execute an action
    GET  /state   — Retrieve current episode state
    GET  /tasks   — List available tasks
    GET  /grader  — Grade the current episode
    GET  /health  — Liveness probe
    GET  /        — Human-readable landing page
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.models import (
    Action,
    ActionType,
    Clip,
    Observation,
    ResetRequest,
    State,
    StepRequest,
)
from env.montage_env import MontageEnv
from graders.grader_easy import grade_highlight
from graders.grader_hard import grade_intent
from graders.grader_medium import grade_structured
from infra.utils import check_style_alignment

app = FastAPI(
    title="Montage OpenEnv",
    description="Sequential Video Editing Decision Environment — OpenEnv API",
    version="1.0.0",
)

_env: Optional[MontageEnv] = None

TASKS_META = [
    {"id": 1, "name": "highlight", "difficulty": "easy",
     "description": "Select clips that maximize importance within a duration budget."},
    {"id": 2, "name": "structured", "difficulty": "medium",
     "description": "Arrange clips into a coherent narrative under duration constraints."},
    {"id": 3, "name": "intent", "difficulty": "hard",
     "description": "Style-aware editing with ordering, selection, and trimming constraints."},
]


def _strict_score(value: float) -> float:
    score = round(float(value), 4)
    return max(0.001, min(0.999, score))


def _compute_ideal_order(clips_map: Dict[str, Clip], style: str, target_duration: float) -> List[str]:
    from typing import Tuple as T
    scored: list[tuple[float, str]] = []
    for cid, clip in clips_map.items():
        s = check_style_alignment([cid], clips_map, style)
        combined = 0.5 * clip.importance + 0.5 * s
        scored.append((combined, cid))
    scored.sort(key=lambda x: x[0], reverse=True)
    ideal: list[str] = []
    dur = 0.0
    for _sc, cid in scored:
        if dur + clips_map[cid].duration <= target_duration + 5:
            ideal.append(cid)
            dur += clips_map[cid].duration
    return ideal


class ResetResponse(BaseModel):
    observation: Observation
    task_name: str
    seed: int


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


@app.get("/", response_class=HTMLResponse)
def landing():
    return """
    <html><head><title>Montage OpenEnv</title>
    <style>
        body{font-family:system-ui,sans-serif;max-width:700px;margin:60px auto;padding:0 20px;color:#1a1a2e;background:#f5f5f5}
        h1{color:#16213e}
        code{background:#e8e8e8;padding:2px 6px;border-radius:4px}
        .endpoint{background:#fff;padding:16px;margin:10px 0;border-radius:8px;border-left:4px solid #0f3460}
        .method{font-weight:bold;color:#0f3460}
    </style></head><body>
    <h1>Montage OpenEnv</h1>
    <p>Sequential Video Editing Decision Environment — OpenEnv Compliant</p>
    <h2>API Endpoints</h2>
    <div class="endpoint"><span class="method">POST</span> <code>/reset</code> — Start a new episode</div>
    <div class="endpoint"><span class="method">POST</span> <code>/step</code> — Execute an action</div>
    <div class="endpoint"><span class="method">GET</span>  <code>/state</code> — Get current episode state</div>
    <div class="endpoint"><span class="method">GET</span>  <code>/health</code> — Liveness check</div>
    <div class="endpoint"><span class="method">GET</span>  <code>/docs</code> — Interactive API docs (Swagger)</div>
    <h2>Tasks</h2>
    <ul>
        <li><strong>highlight</strong> (easy) — Select high-importance clips under a duration budget</li>
        <li><strong>structured</strong> (medium) — Order clips into a coherent narrative sequence</li>
        <li><strong>intent</strong> (hard) — Style-aware editing with ordering and selection constraints</li>
    </ul>
    <h2>Status</h2>
    <p>Environment is <strong>running</strong>. Use <code>/docs</code> for interactive testing.</p>
    </body></html>
    """


@app.get("/health")
def health():
    return {"status": "ok", "environment": "montage_env", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: Request):
    global _env
    try:
        body = await request.json()
    except Exception:
        body = {}
    req = ResetRequest(**(body or {}))
    _env = MontageEnv(task_name=req.task_name, seed=req.seed)
    obs = _env.reset()
    return ResetResponse(
        observation=obs,
        task_name=req.task_name,
        seed=req.seed,
    )


@app.post("/step", response_model=StepResponse)
async def step(request: Request):
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body with action is required.")
    req = StepRequest(**body)
    obs = _env.step(req.action)
    return StepResponse(
        observation=obs,
        reward=obs.reward if obs.reward is not None else 0.0,
        done=obs.done,
        info=obs.metadata,
    )


@app.get("/state", response_model=State)
def get_state():
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    return _env.state


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS_META}


@app.get("/grader")
def grader():
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    state = _env.state
    clips_map = _env.state_manager.available_clips
    timeline = list(state.timeline)
    target_dur = state.target_duration
    style = state.style
    task_name = state.task_name
    ideal_order = _compute_ideal_order(clips_map, style, target_dur)

    if task_name == "highlight":
        raw = grade_highlight(timeline, clips_map, target_dur)
    elif task_name == "structured":
        raw = grade_structured(timeline, ideal_order)
    elif task_name == "intent":
        raw = grade_intent(timeline, clips_map, style, ideal_order)
    else:
        raw = 0.5

    score = _strict_score(raw)
    return {"score": score, "task_name": task_name, "done": state.done}
