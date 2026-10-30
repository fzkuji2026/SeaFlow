"""@agentic_function for GLEE LLM calls.

OpenProgram's decorator is used when it imports. Otherwise a local shim with
the same contract: docstring = system prompt, expose/render_range isolate
context, runtime.exec gates on choices. Live play must not depend on
OpenProgram (its extra deps are optional).
"""

from __future__ import annotations

import inspect
import logging
import os
from functools import update_wrapper
from typing import Any, Callable

from glee_agent.llm import LLMError, complete, cooling_down, parse_json_object

logger = logging.getLogger("glee_agent.agentic")

_op_agentic = None
_OPENPROGRAM = False
if os.environ.get("GLEE_USE_OPENPROGRAM") == "1":
    try:
        from openprogram.agentic_programming import agentic_function as _op_agentic

        _OPENPROGRAM = True
    except Exception as exc:
        logger.debug("OpenProgram agentic_function unavailable: %s", exc)


def _content_to_text(content: Any, context: str | None) -> str:
    parts: list[str] = []
    if context:
        parts.append(str(context))
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
    elif content:
        parts.append(str(content))
    return "\n".join(p for p in parts if p).strip()


def _pick_choice(text: str, choices: Any) -> Any:
    options = list(choices.keys()) if isinstance(choices, dict) else list(choices)
    compact = "".join(ch for ch in text.strip().lower() if ch.isalnum())
    for opt in options:
        token = "".join(ch for ch in str(opt).lower() if ch.isalnum())
        if compact == token or compact.endswith(token) or token == compact:
            return opt if not isinstance(choices, dict) else choices[opt]
    return None


class LocalRuntime:
    def __init__(self, system: str = "") -> None:
        self.system = system

    def exec(
        self,
        content: Any = None,
        context: str | None = None,
        choices: Any = None,
        response_format: dict | None = None,
        timeout_s: float | None = 20.0,
        **_kwargs: Any,
    ) -> Any:
        user = _content_to_text(content, context)
        last_err: Exception | None = None
        attempts = max(1, int(_kwargs.get("max_attempts", 3)))
        for _ in range(attempts):
            try:
                text = complete(
                    user,
                    system=self.system,
                    timeout=float(timeout_s or 20.0),
                    max_tokens=120,
                )
            except LLMError as exc:
                last_err = exc
                if cooling_down():
                    break
                continue
            if choices is not None:
                picked = _pick_choice(text, choices)
                if picked is not None:
                    return picked
                last_err = LLMError(f"no parseable pick from {list(choices)}")
                user = (
                    f"{user}\n\nYour last reply was not one of {list(choices)}. "
                    "Reply with exactly one of those values."
                )
                continue
            if response_format is not None:
                try:
                    return parse_json_object(text)
                except (LLMError, ValueError) as exc:
                    last_err = exc
                    user = f"{user}\n\nReply with a JSON object only."
                    continue
            return text
        raise LLMError(str(last_err or "agentic exec failed"))


def _decorate_local(fn: Callable, expose: str, render_range: dict | None) -> Callable:
    def wrapper(*args: Any, **kwargs: Any):
        sig = inspect.signature(fn)
        if "runtime" in sig.parameters and kwargs.get("runtime") is None:
            kwargs["runtime"] = LocalRuntime(system=(fn.__doc__ or "").strip())
        return fn(*args, **kwargs)

    update_wrapper(wrapper, fn)
    wrapper.expose = expose  # type: ignore[attr-defined]
    wrapper.render_range = render_range or {"callers": 0}  # type: ignore[attr-defined]
    return wrapper


def agentic_function(fn: Callable | None = None, **kwargs: Any):
    expose = kwargs.get("expose", "io")
    render_range = kwargs.get("render_range") or {"callers": 0}

    def apply(target: Callable) -> Callable:
        if _OPENPROGRAM and _op_agentic is not None:
            deco = _op_agentic(expose=expose, render_range=render_range)
            return deco(target)
        return _decorate_local(target, expose, render_range)

    if fn is not None:
        return apply(fn)
    return apply
