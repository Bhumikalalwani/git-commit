# Project Montage Setup & Architecture Guide

**Project Montage** is a fully functional, production-quality, OpenEnv-compatible reinforcement learning environment designed for automated video editing simulation. The system acts as a modular, stateful environment where agents interact via structured actions to fulfill complex video editing requirements. 

## 1. Project Overview & Architecture

Project Montage is structured using a Single Responsibility Principle (SRP) compliant, modular approach, mimicking a true RL framework:

- **Environment (`env/`)**: The core element. It tracks state management, applies transitions, and handles timeline synchronization across a sequence of actions.
- **Data Access Object (`dao/`)**: The repository for structured data elements—specifically video clips. It defines attributes like timestamps, visual scores, themes, and audio features.
- **Grading & Reward Engine (`graders/`)**: Contains deterministic mathematical scorers. It evaluates how well a sequence follows constraints like transitions and duration without human-in-the-loop bias.
- **Infrastructure (`infra/`)**: Shared utility definitions, geometry logic for timelines, and configuration parsers.
- **Inference & Execution Scripts**: Tools like `inference.py` integrate the full loop: querying an agent, interpreting the parsed JSON outputs, applying actions to the `env`, and utilizing `graders` to capture the final reward bounds.

### Supported Tasks
1. **Task 1: Highlight Extraction**: Identify an optimal subset of clips that maximizes visual and audio scores strictly within a specified total duration.
2. **Task 2: Structured Editing**: Reorder and snap heterogeneous clips onto a master timeline to achieve a logically coherent narrative. 
3. **Task 3: Intent-Aware Editing**: Satisfy localized user constraints (e.g., "Add B-roll over the 10-20s segment") while optimizing semantic continuity scores, requiring multi-step look-ahead planning.

---

## 2. The Execution Flow

1. **Initialization**: The baseline environment parses `openenv.yaml`. A list of clips and objectives is generated via the DAO layer.
2. **Observation**: The agent receives a structured schema mapping that represents current timeline constraints, target parameters, and the library of clips.
3. **Action Generation**: The Agent (often an LLM via the `openai` API) processes the observation and predicts sequential discrete actions: e.g., mapping clip selections or defining start/end bounds for timeline placement.
4. **Environment Step**: Project Montage safely validates the generated parameters through Pydantic validators, checks spatial overlaps, and transitions the master timeline. Step verification endpoints denote state with `[START]`, `[STEP]`, and `[END]`.
5. **Reward & Grading**: Once the simulation loop concludes, the `grader` is invoked to apply constraints natively. It calculates and returns a deterministic integer or float reward.
6. **Result Output**: Final sequence layout, actions taken, and total aggregated score metrics are dumped and presented to the caller.

---

## 3. Setup and Prerequisites

You need **Python 3.10+** installed on your system.

### Create a Virtual Environment

It is highly recommended to isolate dependencies. Navigate to the project root directory and run:

```bash
# In PowerShell or Command Prompt
cd project_montage
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

*(If you are on Linux/macOS, activate using `source venv/bin/activate` instead).*

### Install Dependencies

Project Montage relies on minimal, highly-stabilized dependencies. Install them directly:

```bash
pip install pydantic openai pytest
```

**Dependency Breakdown:**
- **`pydantic`**: Enforces strict run-time type safety. Since LLM outputs and RL data streams are notoriously unconstrained, we use Pydantic models to cast, parse, and validate constraints dynamically during `env` transitions to guarantee determinism.
- **`openai`**: Required if you use `inference.py` against active foundation models. It safely handles request multiplexing, polling, and networking to inference APIs.
- **`pytest`**: Required to run the extensive suite of deterministic correctness testing located in the `tests/` directory.

*(There is no database required as the DAO uses structured native data frames injected at runtime).*

---

## 4. Environment Variables

To run the loop using an external inference engine endpoint, declare your API key:

```bash
# In PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# On Linux/MacOS
export OPENAI_API_KEY="your-api-key-here"
```

*(You can omit this if testing entirely locally or bypassing external API requests).*

---

## 5. Running the Tests

We've provided a massive suite of unit tests checking boundaries, rewards, states, and grader math.

```bash
# Using standard Python Unittest
python -m unittest discover -s tests -p "test_*.py"

# Or natively via pytest (Recommended)
pytest tests/
```

Successful output certifies all boundary checks and OpenEnv state transitions are fully correct.

---

## 6. Running Inference (Agentic Loop)

After verifying tests, deploy the reference pipeline test script:

```bash
python inference.py
```

This simulates the full event loop. It walks sequentially through Task 1 (Highlight Extraction), Task 2 (Structured Editing), and Task 3 (Intent-Aware Editing), printing step validations exactly as requested via `[START]`, `[STEP]`, `[END]` lifecycle hooks.
