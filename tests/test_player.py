from glee_agent.evolve import compact_results, fail_rate
from glee_agent.llm import parse_json_object
from glee_agent.player import looks_legal, normalize_action, strategy, unwrap_action, user_prompt


def test_user_prompt_asks_for_message_only():
    text = user_prompt(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "phase": "offer",
            "prompt": "Divide 100.",
            "game_state": {"money_to_divide": 100},
            "valid_actions": {"type": "offer"},
        },
        {"alice_gain": 55, "bob_gain": 45},
    )
    assert "alice_gain" in text
    assert "family=bargaining" in text
    assert "Visible state:" not in text



def test_parse_json_fenced():
    assert parse_json_object("```json\n{\"decision\": \"yes\"}\n```") == {"decision": "yes"}


def test_unwrap_type_fields():
    wrapped = {"type": "buyer_decision", "fields": {"decision": "no"}}
    assert unwrap_action(wrapped) == {"decision": "no"}
    assert looks_legal(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {},
        },
        wrapped,
    )


def test_normalize_bargaining_and_negotiation_tokens():
    bargain = {
        "game_family": "bargaining",
        "valid_actions": {"type": "decision"},
        "game_state": {},
    }
    assert normalize_action(bargain, {"decision": "AcceptOffer"})["decision"] == "accept"
    wrapped = {"type": "decision", "fields": {"decision": "RejectOffer"}}
    assert looks_legal(bargain, wrapped)
    assert normalize_action(bargain, wrapped)["decision"] == "reject"

    deal = {
        "game_family": "negotiation",
        "valid_actions": {"type": "decision"},
        "game_state": {"horizon_known": True, "round": 1, "max_rounds": 10},
    }
    assert normalize_action(deal, {"decision": "accept"})["decision"] == "AcceptOffer"
    assert normalize_action(deal, {"decision": "walkaway"})["decision"] == "WalkAway"


def test_looks_legal_bargaining_sum():
    game = {
        "game_family": "bargaining",
        "valid_actions": {"type": "offer"},
        "game_state": {"money_to_divide": 100},
    }
    assert looks_legal(game, {"alice_gain": 55, "bob_gain": 45})
    assert not looks_legal(game, {"alice_gain": 55, "bob_gain": 40})


def test_strategy_exception_submits_legal_fallback(monkeypatch):
    monkeypatch.setattr("glee_agent.player.heuristic", lambda game: 1 / 0)
    games = [
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {"money_to_divide": 100},
        },
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "player_2_role": "buyer",
                "player_2_value": 120,
                "round": 2,
                "max_rounds": 10,
            },
        },
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {},
        },
    ]
    assert all(looks_legal(game, strategy(game)) for game in games)


def test_user_prompt_does_not_ask_the_model_to_pick_numbers():
    text = user_prompt(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "phase": "offer",
            "prompt": "Divide 100.",
            "game_state": {"money_to_divide": 100},
            "valid_actions": {"type": "offer"},
        },
        {"alice_gain": 55, "bob_gain": 45},
    )
    assert "do not change" not in text
    assert "alice_gain" in text
    assert "family=bargaining" in text


def test_fail_rate_persuasion_zero_buys():
    rows = [
        {"family": "persuasion", "result": {"rounds_bought": 0, "player_1_payoff": 0, "player_2_payoff": 0}},
        {"family": "persuasion", "result": {"rounds_bought": 20, "player_1_payoff": 1, "player_2_payoff": 1}},
    ]
    assert fail_rate("persuasion", rows) == 0.5
    assert compact_results(rows)[0]["bought"] == 0
