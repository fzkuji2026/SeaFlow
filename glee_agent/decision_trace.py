"""Task-local provenance for the action currently being computed."""

from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
from typing import Any

_NODE: ContextVar[str | None] = ContextVar("glee_decision_node", default=None)
_TRACE: ContextVar[dict[str, Any] | None] = ContextVar(
    "glee_workflow_trace", default=None
)


def begin(workflow: str, *, context: dict[str, Any] | None = None) -> None:
    _NODE.set(None)
    _TRACE.set(
        {
            "schema_version": 1,
            "workflow": workflow,
            "context": dict(context or {}),
            "candidates": [],
            "eliminations": [],
            "selection": None,
        }
    )


def set_node(node: str | None) -> None:
    _NODE.set(node)
    if node is None:
        _TRACE.set(None)
        return
    trace = deepcopy(_TRACE.get()) if _TRACE.get() is not None else {
        "schema_version": 1,
        "workflow": "unknown",
        "context": {},
        "candidates": [],
        "eliminations": [],
        "selection": None,
    }
    trace["selection"] = {"node": node}
    _TRACE.set(trace)


def current_node() -> str | None:
    return _NODE.get()


def add_candidate(
    candidate_id: str,
    action: dict[str, Any],
    *,
    source: str,
    economic: dict[str, Any] | None = None,
    rating: dict[str, Any] | None = None,
) -> None:
    trace = deepcopy(_TRACE.get())
    if trace is None:
        return
    trace["candidates"].append(
        {
            "id": candidate_id,
            "action": dict(action),
            "source": source,
            "economic": economic,
            "rating": rating,
        }
    )
    _TRACE.set(trace)


def eliminate(
    candidate_id: str,
    *,
    rule_id: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    trace = deepcopy(_TRACE.get())
    if trace is None:
        return
    trace["eliminations"].append(
        {
            "candidate_id": candidate_id,
            "rule_id": rule_id,
            "reason": reason,
            "evidence": dict(evidence or {}),
        }
    )
    _TRACE.set(trace)


def choose(
    candidate_id: str,
    *,
    node: str,
    stage: str,
    reason: str,
    margin: float | None = None,
) -> None:
    _NODE.set(node)
    trace = deepcopy(_TRACE.get())
    if trace is None:
        return
    trace["selection"] = {
        "node": node,
        "candidate_id": candidate_id,
        "stage": stage,
        "reason": reason,
        "margin": margin,
    }
    _TRACE.set(trace)


def current_trace() -> dict[str, Any] | None:
    trace = _TRACE.get()
    return deepcopy(trace) if trace is not None else None
