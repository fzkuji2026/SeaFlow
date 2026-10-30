from glee_agent.agentic import LocalRuntime, agentic_function
from glee_agent.llm import LLMError


@agentic_function(expose="io", render_range={"callers": 0})
def pick_side(brief: str, runtime=None) -> str:
    """Pick accept or reject."""
    return runtime.exec(content=[{"type": "text", "text": brief}], choices=["accept", "reject"])


def test_decorator_stamps_context_knobs():
    assert pick_side.expose == "io"
    assert pick_side.render_range == {"callers": 0}


def test_choices_gate_retries_then_returns_legal_value(monkeypatch):
    replies = iter(["probably a feature request", "accept"])

    def fake_complete(user, system="", timeout=20.0, max_tokens=120):
        return next(replies)

    monkeypatch.setattr("glee_agent.agentic.complete", fake_complete)
    rt = LocalRuntime(system="Pick accept or reject.")
    assert rt.exec(content="round=1", choices=["accept", "reject"]) == "accept"


def test_cooldown_error_does_not_retry(monkeypatch):
    calls = 0

    def unavailable(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise LLMError("xai 429 quota")

    monkeypatch.setattr("glee_agent.agentic.complete", unavailable)
    monkeypatch.setattr("glee_agent.agentic.cooling_down", lambda: True)
    rt = LocalRuntime(system="Pick accept or reject.")
    try:
        rt.exec(content="round=1", choices=["accept", "reject"])
    except LLMError as exc:
        assert str(exc) == "xai 429 quota"
    else:
        raise AssertionError("cooldown must fall back through LLMError")
    assert calls == 1


def test_openprogram_is_opt_in():
    import glee_agent.agentic as ag

    assert ag._OPENPROGRAM is False
    assert ag._op_agentic is None
