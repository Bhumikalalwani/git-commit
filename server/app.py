"""OpenEnv server entry point.

This module re-exports the FastAPI app from the root app module so that
the OpenEnv toolchain (which expects server/app.py) can discover it.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app  # noqa: E402, F401


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
