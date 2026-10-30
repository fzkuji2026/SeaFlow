#!/usr/bin/env python3
"""Play a batch, mutate one numeric policy knob, keep it if the family rating rose."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))

from glee_agent.evolve import fail_rate  # noqa: E402
from glee_agent.keys import load_all  # noqa: E402
from glee_agent.lock import AlreadyRunning, exclusive  # noqa: E402
from glee_agent.net import patch_dns  # noqa: E402
from glee_agent.player import strategy  # noqa: E402
from glee_agent.policy import current, keep_mutation, load_policy, mutate, save_policy, set_policy  # noqa: E402
from glee_agent.trace import TracingClient  # noqa: E402
from glee_sdk.client import GleeAPIError  # noqa: E402


HISTORY = ROOT / "logs" / "evolve.jsonl"


def _pause_seconds(exc: GleeAPIError) -> float:
    """Sleep until the server's unlock time, not a blind 60s loop."""
    import re

    msg = str(exc)
    match = re.search(r"after (\d{4}-\d{2}-\d{2}T[\d:.]+Z?)", msg)
    if not match:
        return 60.0
    stamp = match.group(1)
    if stamp.endswith("Z"):
        stamp = stamp[:-1] + "+00:00"
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return 60.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max((when - datetime.now(timezone.utc)).total_seconds() + 2.0, 5.0)


def _ratings(stats: dict) -> dict[str, float]:
    scores = stats.get("scores") or stats.get("ratings") or {}
    out: dict[str, float] = {}
    for fam in ("bargaining", "negotiation", "persuasion"):
        row = scores.get(fam) or {}
        if isinstance(row, dict):
            out[fam] = float(row.get("display") or row.get("rating") or row.get("score") or 0)
        else:
            out[fam] = float(row or 0)
    return out


def _weakest(ratings: dict[str, float], batch: list[dict]) -> str:
    rates: dict[str, float] = {}
    for fam in ("bargaining", "negotiation", "persuasion"):
        n = sum(1 for row in batch if row.get("family") == fam)
        if n:
            rates[fam] = fail_rate(fam, batch)
    if rates:
        return max(rates, key=rates.get)
    return min(ratings, key=ratings.get) if ratings else "negotiation"


def _append(row: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _stats(client: TracingClient) -> dict:
    import requests

    for attempt in range(8):
        try:
            return client.stats()
        except (requests.RequestException, GleeAPIError) as exc:
            wait = min(2 ** attempt, 30)
            logging.warning("stats failed (%s); retrying in %ss", exc, wait)
            time.sleep(wait)
    raise RuntimeError("Could not reach GLEE API")


def main() -> int:
    patch_dns()
    load_all()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log_path = ROOT / "logs" / "agent.log"
    log_path.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    api_key = os.environ.get("GLEE_API_KEY", "")
    if not api_key:
        print("Set GLEE_API_KEY in .env or the environment", file=sys.stderr)
        return 1

    games = int(os.environ.get("GLEE_EVOLVE_GAMES", "24"))
    generations = int(os.environ.get("GLEE_EVOLVE_GENERATIONS", "0"))
    concurrency = int(os.environ.get("GLEE_CONCURRENCY", "8"))
    poll_interval = float(os.environ.get("GLEE_POLL_INTERVAL", "1"))

    load_policy()
    save_policy()
    try:
        exclusive()
    except AlreadyRunning as exc:
        print(exc, file=sys.stderr)
        return 1
    base_url = os.environ.get("GLEE_API_URL")
    client = TracingClient(api_key=api_key, base_url=base_url) if base_url else TracingClient(api_key=api_key)

    try:
        stats = _stats(client)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    logging.info("policy evolve start ratings=%s policy=%s", _ratings(stats), current())
    gen = 0
    while generations == 0 or gen < generations:
        gen += 1
        parent = json.loads(json.dumps(current()))
        before = _ratings(_stats(client))
        family = _weakest(before, getattr(client, "batch", []) or [])
        child, family, key = mutate(parent, family=family)
        set_policy(child)
        save_policy(child)
        before_fail = fail_rate(family, client.batch) if client.batch else 0.0
        client.batch = []
        logging.info("gen %s mutate %s.%s play %s games", gen, family, key, games)
        while True:
            try:
                client.run(
                    strategy,
                    concurrency=concurrency,
                    poll_interval=poll_interval,
                    max_games=games,
                )
                break
            except GleeAPIError as exc:
                msg = str(exc)
                if "timed out" not in msg.lower() and "paused" not in msg.lower():
                    raise
                wait = _pause_seconds(exc)
                logging.warning("queue paused (%s); sleeping %.0fs", exc, wait)
                time.sleep(wait)
        after = _ratings(_stats(client))
        after_fail = fail_rate(family, client.batch)
        kept = keep_mutation(before.get(family, 0.0), after.get(family, 0.0), before_fail, after_fail)
        if kept:
            logging.info("KEEP %s.%s %s -> %s", family, key, before.get(family), after.get(family))
        else:
            set_policy(parent)
            save_policy(parent)
            logging.info("ROLLBACK %s.%s %s -> %s", family, key, before.get(family), after.get(family))
        _append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "gen": gen,
                "family": family,
                "key": key,
                "kept": kept,
                "before": before,
                "after": after,
                "fail_rate": after_fail,
                "n": len(client.batch),
                "child": child[family],
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
