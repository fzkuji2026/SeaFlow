"""Provisional family-level display-rating ticks from official /stats."""

from __future__ import annotations

from typing import Any


def snapshot(stats: dict[str, Any]) -> dict[str, dict[str, float]]:
    scores = stats.get("scores") or stats.get("ratings") or {}
    out: dict[str, dict[str, float]] = {}
    for fam in ("bargaining", "negotiation", "persuasion"):
        row = scores.get(fam) or {}
        if not isinstance(row, dict):
            continue
        out[fam] = {
            "rating": float(row.get("display") or row.get("rating") or row.get("score") or 0),
            "n": float(row.get("games_played") or row.get("games") or 0),
        }
    return out


def family_ticks(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    ticks = []
    for fam in ("bargaining", "negotiation", "persuasion"):
        if fam not in after or fam not in before:
            continue
        dn = int(after[fam]["n"] - before[fam]["n"])
        if dn <= 0:
            continue
        delta = float(after[fam]["rating"] - before[fam]["rating"])
        ticks.append(
            {
                "family": fam,
                "n": dn,
                "delta": round(delta, 4),
                "per_game": round(delta / dn, 4),
                # This is still an aggregate family tick. With concurrent
                # games, completion/log order can map it to the wrong id.
                "exact": False,
                "provisional": True,
            }
        )
    return ticks


def assign_ticks(
    pending: dict[str, list[str]],
    ticks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach ticks to oldest pending game ids. Mutates pending."""
    assigned: list[dict[str, Any]] = []
    for tick in ticks:
        fam = tick["family"]
        queue = list(pending.get(fam) or [])
        take = min(int(tick["n"]), len(queue))
        if take == 0:
            continue
        ids = queue[:take]
        pending[fam] = queue[take:]
        per = float(tick["per_game"])
        for gid in ids:
            assigned.append(
                {
                    "game_id": gid,
                    "family": fam,
                    "rating_delta": per,
                    "exact": False,
                    "provisional": True,
                }
            )
    return assigned
