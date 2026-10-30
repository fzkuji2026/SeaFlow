#!/usr/bin/env python3
"""Run the GLEE agent against the competition server with a fixed workflow."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))

from glee_agent.keys import load_all  # noqa: E402
from glee_agent.lock import AlreadyRunning, exclusive  # noqa: E402
from glee_agent.player import strategy  # noqa: E402
from glee_agent.trace import TracingClient  # noqa: E402

FAMILIES = ("bargaining", "negotiation", "persuasion")


def main() -> int:
    load_all()
    api_key = os.environ.get("GLEE_API_KEY", "")
    if not api_key:
        print("Set GLEE_API_KEY in .env or the environment", file=sys.stderr)
        return 1
    families = [
        f.strip()
        for f in os.environ.get("GLEE_GAME_FAMILIES", ",".join(FAMILIES)).split(",")
        if f.strip()
    ]
    if not families or any(f not in FAMILIES for f in families):
        print("Invalid GLEE_GAME_FAMILIES", file=sys.stderr)
        return 1
    try:
        exclusive()
    except AlreadyRunning as exc:
        print(exc, file=sys.stderr)
        return 1
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / "agent.log")],
    )
    base_url = os.environ.get("GLEE_API_URL")
    client = TracingClient(api_key=api_key, base_url=base_url) if base_url else TracingClient(api_key=api_key)
    max_games = os.environ.get("GLEE_MAX_GAMES", "20")
    max_time = os.environ.get("GLEE_MAX_TIME")
    client.run(
        strategy,
        game_families=families,
        concurrency=int(os.environ.get("GLEE_CONCURRENCY", "3")),
        poll_interval=float(os.environ.get("GLEE_POLL_INTERVAL", "2")),
        max_games=int(max_games) if max_games else None,
        max_time=float(max_time) if max_time else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
