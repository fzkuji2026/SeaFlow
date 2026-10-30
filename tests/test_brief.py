import pytest

from glee_agent.agentic import LocalRuntime
from glee_agent.brief import brief_game
from glee_agent.brief import decision_packet
from glee_agent.player import user_prompt
from glee_agent.talk import fill_message, write_message
from glee_agent.llm import LLMError


def _bargain_game(history=None):
    return {
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "prompt": "Divide 100. This prompt is long on purpose.",
        "game_state": {
            "money_to_divide": 100,
            "current_player": "alice",
            "history": history or [{"noise": i} for i in range(20)],
        },
        "valid_actions": {"type": "offer"},
    }


def test_brief_is_compact_and_keeps_decided_numbers():
    text = brief_game(_bargain_game(), {"alice_gain": 55, "bob_gain": 45})
    assert "alice_gain" in text
    assert "pot=100" in text
    assert "history_tail=" in text
    assert text.count("noise") <= 3
    assert "Divide 100. This prompt is long on purpose." not in text


def test_user_prompt_is_the_brief():
    game = _bargain_game(history=[])
    action = {"alice_gain": 55, "bob_gain": 45}
    assert user_prompt(game, action) == brief_game(game, action)
    assert "do not change" not in user_prompt(game, action)


def test_write_message_uses_runtime_and_fallback_on_json():
    class Fake(LocalRuntime):
        def exec(self, content=None, **kwargs):
            return '{"message": "oops"}'

    assert write_message("brief", "fallback text", runtime=Fake()) == "fallback text"

    class Ok(LocalRuntime):
        def exec(self, content=None, **kwargs):
            return "A split worth taking now."

    assert write_message("brief", "fallback text", runtime=Ok()) == "A split worth taking now."


def test_fill_message_skips_llm_by_default(monkeypatch):
    monkeypatch.delenv("GLEE_LLM_MESSAGES", raising=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("LLM should not run")

    monkeypatch.setattr("glee_agent.talk.write_message", boom)
    action = {"product_price": 70, "message": "This is my number."}
    assert fill_message(_bargain_game(), action) == action


def test_fill_message_prioritizes_judge_call(monkeypatch):
    monkeypatch.setenv("GLEE_LLM_MESSAGES", "1")
    monkeypatch.setenv("GLEE_LLM_JUDGE", "1")

    def boom(*_args, **_kwargs):
        raise AssertionError("message LLM must not consume the judge quota")

    monkeypatch.setattr("glee_agent.talk.write_message", boom)
    action = {"alice_gain": 55, "bob_gain": 45, "message": "Code fallback."}
    assert fill_message(_bargain_game(), action) == action


def test_live_runtime_can_disable_internal_retries(monkeypatch):
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise LLMError("timeout")

    monkeypatch.setattr("glee_agent.agentic.complete", fail)
    runtime = LocalRuntime()
    with pytest.raises(LLMError):
        runtime.exec("brief", response_format={"type": "json_object"}, max_attempts=1)
    assert calls == 1


def test_decision_packet_gives_scenario_reasoning_not_fixed_share_rule():
    game = _bargain_game(history=[])
    snap = {
        "family": "bargaining",
        "path": "bargaining/impatient/hot/decision",
        "action": {"decision": "reject"},
        "calc": {
            "procedure": "hot",
            "last_share": 0.43,
            "discounted_last_share": 0.344,
            "next_even_if_accepted": 0.32,
            "expected_next_even": 0.30,
            "same_offer_repeats": 2,
        },
    }
    packet = decision_packet(
        game,
        snap,
        [{"name": "heuristic", "action": {"decision": "reject"}}],
    )
    assert "Compare discounted payoff now with continuation value" in packet
    assert "Repeated rejection reduces the case for waiting" in packet
    assert "waiting does not discount own payoff" in packet
    assert "taking below 0.50 after we posted" not in packet


def test_public_reference_packet_states_opponent_dimension_caveat():
    game = _bargain_game(history=[])
    snap = {
        "family": "bargaining",
        "action": {"decision": "reject"},
        "calc": {
            "procedure": "mild",
            "public_reference": {
                "median_payoff_share": 0.5,
                "median_round": 2,
                "deal_rate": 1.0,
            },
        },
    }
    packet = decision_packet(
        game, snap, [{"name": "heuristic", "action": {"decision": "reject"}}]
    )
    assert "not a fixed opponent delta" in packet
    assert "Use it as context, not a target" in packet


def test_seller_packet_does_not_treat_buyer_ev_as_seller_utility():
    game = {
        "game_family": "persuasion",
        "your_player": "player_1",
        "valid_actions": {"type": "seller_recommendation"},
        "game_state": {"current_quality": "low", "product_price": 100},
    }
    snap = {
        "family": "persuasion",
        "action_type": "seller_recommendation",
        "action": {"decision": "yes"},
        "calc": {"quality": "low", "ev": 0, "price": 100},
    }
    packet = decision_packet(game, snap, [{"name": "heuristic", "action": {"decision": "yes"}}])
    assert "every sale earns price regardless of quality" in packet
    assert "Do not use buyer EV" in packet


def test_buyer_packet_keeps_public_iid_quality_prior():
    game = {
        "game_family": "persuasion",
        "your_player": "player_2",
        "valid_actions": {"type": "buyer_decision"},
        "game_state": {"p": 0.5, "v": 300, "u": 0, "product_price": 100},
    }
    snap = {
        "family": "persuasion",
        "action_type": "buyer_decision",
        "action": {"decision": "yes"},
        "calc": {"ev": 150, "price": 100},
    }
    packet = decision_packet(
        game, snap, [{"name": "heuristic", "action": {"decision": "yes"}}]
    )
    assert "quality prior remains the public p" in packet
    assert "not the next round's base quality rate" in packet
