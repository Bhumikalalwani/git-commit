"""OpenEnv-compliant FastAPI server for the Montage video editing environment.

Endpoints:
    POST /reset   — Start a new episode
    POST /step    — Execute an action
    GET  /state   — Retrieve current episode state
    GET  /health  — Liveness probe
    GET  /        — Human-readable landing page
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.models import (
    Action,
    ActionType,
    Observation,
    ResetRequest,
    State,
    StepRequest,
)
from env.montage_env import MontageEnv

app = FastAPI(
    title="Montage OpenEnv",
    description="Sequential Video Editing Decision Environment — OpenEnv API",
    version="1.0.0",
)

_env: Optional[MontageEnv] = None


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
