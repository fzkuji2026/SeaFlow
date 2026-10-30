#!/usr/bin/env python3
"""Summarize logs/agent.log for the play → patch loop."""

from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "agent.log"


def summarize(text: str, after_pos: int = 0) -> dict:
    chunk = text[after_pos:]
    finished = re.findall(r"Game ([0-9a-f-]+) finished! Result: (\{.*\})", chunk)
    last_fam: dict[str, str] = {}
    for m in re.finditer(r"Game ([0-9a-f-]+) \((\w+)\) -", chunk):
        last_fam[m.group(1)] = m.group(2)

    by_fam: dict[str, list[dict]] = defaultdict(list)
    for gid, raw in finished:
        try:
            result = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            continue
        by_fam[last_fam.get(gid, "?")].append(result)

    out: dict = {
        "finished": len(finished),
        "invalid": chunk.count("move rejected as invalid"),
        "families": {},
    }
    for fam in ("bargaining", "negotiation", "persuasion"):
        rows = by_fam.get(fam, [])
        fail = 0
        for r in rows:
            outcome = r.get("outcome")
            both0 = r.get("player_1_payoff") == 0 and r.get("player_2_payoff") == 0
            if fam == "persuasion":
                if r.get("rounds_bought") == 0 or both0:
                    fail += 1
            elif outcome in ("walked_away", "no_deal") or both0:
                fail += 1
        n = len(rows)
        out["families"][fam] = {
            "n": n,
            "fail": fail,
            "fail_rate": (fail / n) if n else 1.0,
            "outcomes": dict(Counter(r.get("outcome") for r in rows)),
        }
    return out


def main() -> None:
    if not LOG.exists():
        print(f"no log at {LOG}")
        return
    text = LOG.read_text(errors="replace")
    stats = summarize(text)
    print(f"log={LOG} bytes={LOG.stat().st_size} finished={stats['finished']} invalid={stats['invalid']}")
    for fam, row in stats["families"].items():
        print(f"{fam}: n={row['n']} fail_rate={row['fail_rate']:.2f} outcomes={row['outcomes']}")


if __name__ == "__main__":
    main()
