#!/usr/bin/env python3
"""Print path + calc JSON for a History game. Inspectors do not hand-calculate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "vendor"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("game_id", nargs="?")
    parser.add_argument("--file", help="replay JSON (has state) or decision-time game JSON (has game_state)")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument(
        "--action",
        default=None,
        help="offer|decision|buyer_decision|seller_recommendation|seller_message",
    )
    args = parser.parse_args()

    from glee_agent.diagnose import snapshot, snapshot_replay
    from glee_agent.policy import load_policy

    load_policy()

    if args.file:
        payload = json.loads(Path(args.file).read_text())
        if not isinstance(payload, dict):
            raise SystemExit("file is not a JSON object")
        if args.round is not None or args.action is not None or "state" in payload:
            snap = snapshot_replay(
                payload, round_n=args.round, action_type=args.action
            )
        elif "game_state" in payload:
            snap = snapshot(payload)
        else:
            snap = snapshot_replay(
                payload, round_n=args.round, action_type=args.action
            )
    else:
        parser.error("--file is required")
    print(json.dumps(snap, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
