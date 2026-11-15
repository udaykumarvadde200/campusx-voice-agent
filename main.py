"""Single entry point for the CampusX local voice-agent codebase."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from campusx_voice_agent.bookings import show_bookings  # noqa: E402
from campusx_voice_agent.moss_index import main as create_moss_index  # noqa: E402


async def bot(runner_args: Any):
    from campusx_voice_agent.agent import bot as voice_agent_bot

    await voice_agent_bot(runner_args)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""

    if command == "bookings":
        show_bookings()
        return

    if command in {"moss-index", "create-moss-index"}:
        create_moss_index()
        return

    if command in {"download-models", "check"}:
        from campusx_voice_agent.models import check_ollama, download_models

        if command == "download-models":
            download_models()
        else:
            check_ollama()
        return

    from pipecat.runner.run import main as pipecat_main

    pipecat_main()


if __name__ == "__main__":
    main()
