# Project Montage: Sequential Video Editing Decision Environment

An OpenEnv-compatible reinforcement learning environment for automated video editing tasks.

## Tasks
1. **Highlight Extraction:** Selection of important clips under duration constraint.
2. **Structured Editing:** Ordering clips to form a coherent sequence.
3. **Intent-Aware Editing:** Editing for max style and coherence under constraints.

## Architecture
- `env/`: Core OpenEnv methods.
- `dao/`: Clip and task definition logic.
- `infra/`: Configuration and math utils.
- `graders/`: Specialized test scorers.
- `tests/`: Correctness validations.
