"""Named tactics composed into a pipeline.

Numbers stay in the family modules. This file sequences them and, if
GLEE_COMPOSE_PICK=1, lets a model choose among already-legal candidates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from glee_agent.agentic import agentic_function

Tactic = Callable[[float], float]


@dataclass(frozen=True)
class Trace:
    steps: tuple[str, ...]
    value: float

    def __str__(self) -> str:
        return " → ".join(self.steps) + f" = {self.value}"


def pipeline(seed: float, steps: list[tuple[str, Tactic]]) -> Trace:
    value = float(seed)
    names = ["seed"]
    for name, fn in steps:
        value = float(fn(value))
        names.append(name)
    return Trace(steps=tuple(names), value=value)


@agentic_function(expose="io", render_range={"callers": 0})
def pick_named(brief: str, names: tuple[str, ...] = (), runtime=None) -> str:
    """Pick one tactic name. Reply with that name only."""
    if not names:
        return ""
    return runtime.exec(
        content=[{"type": "text", "text": brief}],
        choices=list(names),
        timeout_s=8.0,
    )


def choose(default: str, options: dict[str, Any], brief: str = "") -> tuple[str, Any]:
    """Return (name, value). Falls back to `default` if picking is off or fails."""
    if default not in options:
        default = next(iter(options))
    if os.environ.get("GLEE_COMPOSE_PICK") != "1" or len(options) < 2:
        return default, options[default]
    try:
        name = str(pick_named(brief, names=tuple(options)))
    except Exception:
        name = default
    if name not in options:
        name = default
    return name, options[name]
