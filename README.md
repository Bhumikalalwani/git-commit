---
title: Montage OpenEnv
emoji: "\U0001F3AC"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Montage OpenEnv — Sequential Video Editing Decision Environment

An **OpenEnv-compliant** reinforcement learning environment that simulates
real-world video montage editing. An agent selects, removes, reorders, and
trims video clips to assemble an optimal montage under duration and style
constraints.

## Environment Description

The Montage environment models the core decision-making in professional video
editing: given a pool of raw video clips with metadata (duration, importance
score, emotion, motion level, tags), the agent builds a timeline that
maximizes quality while respecting a duration budget and a target editing
style.

This is a **real-world task** — every video editor, from YouTubers to film
studios, faces this exact optimisation problem.

## Action Space (Discrete)

| Action    | Parameters               | Description                                    |
|-----------|--------------------------|------------------------------------------------|
| `select`  | `clip_id`                | Add a clip to the end of the timeline          |
| `remove`  | `clip_id`                | Remove a clip from the timeline                |
| `reorder` | `params.i`, `params.j`   | Swap two clips at positions i and j            |
| `trim`    | `clip_id`, `params.duration` | Shorten a clip to a new duration           |
| `finish`  | —                        | End the episode                                |

**Action model** (JSON):
```json
{
  "action_type": "select",
  "clip_id": "clip_003",
  "params": null
}
```

## Observation Space (Structured)

Each observation returned by `reset()` and `step()` contains:

| Field              | Type            | Description                               |
|--------------------|-----------------|-------------------------------------------|
| `available_clips`  | `list[Clip]`    | Clips not yet on the timeline             |
| `timeline`         | `list[str]`     | Ordered clip IDs currently on the timeline|
| `remaining_time`   | `float`         | Seconds of budget remaining               |
| `style`            | `str`           | Target editing style                      |
| `done`             | `bool`          | Whether the episode has ended             |
| `reward`           | `float \| null` | Reward signal in [0.0, 1.0]              |
| `reward_breakdown` | `object \| null`| Component scores and penalties            |
| `metadata`         | `dict`          | Step count, durations, etc.               |

Each **Clip** has: `id`, `duration`, `importance` (0–1), `emotion`, `motion`, `tags`.

## Tasks (Easy → Medium → Hard)

### Task 1: Highlight Extraction (Easy)
- **Goal:** Select the highest-importance clips that fit within a duration budget.
- **Pool:** 15 clips, target duration 20–40s.
- **Grader:** Measures importance coverage × duration accuracy.
- **Strategy:** Greedy importance selection works reasonably well.

### Task 2: Structured Editing (Medium)
- **Goal:** Arrange clips into a coherent narrative sequence.
- **Pool:** 25 clips, target duration 30–50s.
- **Grader:** LCS similarity against an ideal ordering.
- **Strategy:** Requires considering both importance and clip ordering.

### Task 3: Intent-Aware Editing (Hard)
- **Goal:** Style-aware editing — match a target style while maintaining coherent order.
- **Pool:** 30 clips, target duration 40–60s.
- **Grader:** Weighted combination of style alignment (50%), ordering (30%), and selection coverage (20%).
- **Strategy:** Must balance style matching, ordering, and selection — simple greedy fails.

## Reward Function

The reward is a **weighted sum** of quality signals minus penalties, clamped to [0.0, 1.0]:

```
reward = w_imp × importance + w_coh × coherence + w_sty × style
       - w_pen × (duration_penalty + redundancy_penalty + step_penalty)
```

**Quality signals (partial progress):**
- **Importance:** Fraction of optimal importance budget captured.
- **Coherence:** Penalizes out-of-order clip sequences.
- **Style alignment:** Fraction of clips matching the target emotion/motion.

**Penalties:**
- **Duration penalty:** `|actual - target| / target` — how far from the budget.
- **Redundancy penalty:** Penalizes repeated tags across clips (0.1 per repeat).
- **Step penalty:** `step_count / 100` — rewards efficiency.

Weights are **task-specific** (e.g., highlight emphasises importance at 0.8, intent emphasises style at 0.4).

## OpenEnv API

The environment exposes an HTTP API per the OpenEnv specification:

| Method | Endpoint  | Request Body                              | Response       |
|--------|-----------|-------------------------------------------|----------------|
| POST   | `/reset`  | `{"task_name": "highlight", "seed": 42}`  | `Observation`  |
| POST   | `/step`   | `{"action": {"action_type": "select", "clip_id": "clip_003"}}` | `Observation` |
| GET    | `/state`  | —                                         | `State`        |
| GET    | `/health` | —                                         | `{"status": "ok"}` |
| GET    | `/docs`   | —                                         | Swagger UI     |

## Setup Instructions

### Prerequisites
- Python 3.10+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Then open http://localhost:7860 for the landing page or http://localhost:7860/docs for interactive API docs.

### Run the Baseline Inference

```bash
python inference.py
```

This runs the deterministic greedy agent across all three tasks and prints
reproducible scores. No API keys required.

**Expected output (seed=42):**

```
================================================================
  Montage OpenEnv — Baseline Inference (Greedy Agent)
================================================================
  Seed: 42

[START] Task: highlight   ... [END]
[START] Task: structured  ... [END]
[START] Task: intent      ... [END]

  Summary
  Task           Difficulty   Reward   Grader
  highlight      easy         0.8120   0.8333
  structured     medium       0.8093   0.2500
  intent         hard         0.6115   0.4083
  AVERAGE                     0.7443   0.4972
```

The greedy agent scores well on the easy task (0.83 grader) but struggles
with structured ordering (0.25) and intent-aware editing (0.41), showing
the difficulty progression is meaningful.

### Run Tests

```bash
python -m pytest tests/ -v
```

### Docker

```bash
docker build -t montage-openenv .
docker run -p 7860:7860 montage-openenv
```

## Project Structure

```
.
├── app.py                  # FastAPI server (OpenEnv HTTP API)
├── inference.py            # Baseline greedy agent + reproducible scores
├── openenv.yaml            # OpenEnv environment specification
├── Dockerfile              # HF Spaces / Docker deployment
├── requirements.txt        # Python dependencies
├── env/
│   ├── models.py           # Typed Pydantic models (Action, Observation, State, Clip, ...)
│   ├── montage_env.py      # Core environment: reset(), step(), state
│   ├── state_manager.py    # Timeline state management
│   └── reward_engine.py    # Weighted reward computation with partial progress
├── dao/
│   ├── clip_dao.py         # Synthetic clip generation (deterministic)
│   └── task_dao.py         # Task configurations (easy/medium/hard)
├── infra/
│   ├── config.py           # Reward weights and style mappings
│   └── utils.py            # Coherence, redundancy, style alignment helpers
├── graders/
│   ├── grader_easy.py      # Highlight extraction grader
│   ├── grader_medium.py    # Structured editing grader (LCS)
│   └── grader_hard.py      # Intent-aware editing grader (multi-signal)
└── tests/
    ├── test_env.py          # Environment reset/step/state tests
    ├── test_reward.py       # Reward engine unit tests
    ├── test_graders.py      # Grader correctness tests
    └── test_server.py       # FastAPI endpoint integration tests
```

## Architecture

- **`env/`** — Core OpenEnv environment implementing `reset()`, `step()`, and `state`.
  Observations include `done`, `reward`, and `reward_breakdown` per the spec.
- **`dao/`** — Data access layer for deterministic clip generation and task configs.
- **`infra/`** — Shared configuration (reward weights, style mappings) and math utilities.
- **`graders/`** — Independent agent graders for each task producing scores in [0.0, 1.0].
- **`tests/`** — Unit and integration tests.

## License

MIT
