import json

from glee_agent.adapt import apply_deltas, dump_game, is_fail, my_role, planned_nudge
from glee_agent.policy import DEFAULT


def test_walkaway_is_fail():
    assert is_fail("negotiation", {"outcome": "walked_away", "player_1_payoff": 0, "player_2_payoff": 0})
    assert not is_fail("negotiation", {"outcome": "agreement", "player_1_payoff": 12, "player_2_payoff": 8})


def test_seller_fail_nudge_is_softer_open():
    deltas = planned_nudge("negotiation", "seller", 0.6)
    assert deltas["seller_open"] < 0
    assert deltas["walk_progress"] < 0
    child = apply_deltas(DEFAULT, "negotiation", deltas)
    assert child["negotiation"]["seller_open"] < DEFAULT["negotiation"]["seller_open"]
    assert child["negotiation"]["seller_open"] >= child["negotiation"]["seller_floor"]
    assert child["negotiation"]["walk_progress"] < DEFAULT["negotiation"]["walk_progress"]


def test_healthy_window_does_not_nudge():
    assert planned_nudge("negotiation", "seller", 0.2) == {}


def test_role_from_state():
    role = my_role(
        "negotiation",
        "player_1",
        {"game_state": {"player_1_role": "seller", "current_player": "player_1"}},
    )
    assert role == "seller"


from glee_agent.adapt import pressure_from_points


def test_points_beat_display_total():
    assert pressure_from_points(2.4, 0.8) == 0.0
    assert pressure_from_points(-3.1, 0.0) == 0.7
    assert pressure_from_points(-1.0, 0.0) == 0.45


from glee_agent.adapt import stance_from_window


def test_closed_deals_that_lose_points_tighten():
    assert stance_from_window(-2.5, 0.0) == "tighten"
    assert stance_from_window(1.2, 0.1) is None
    assert stance_from_window(-0.1, 0.5) == "soften"


def test_deal_loss_tightens_bargaining():
    deltas = planned_nudge("bargaining", "alice", 0.0, stance="tighten")
    assert deltas["offer_share"] > 0
    child = apply_deltas(DEFAULT, "bargaining", deltas)
    assert child["bargaining"]["offer_share"] > DEFAULT["bargaining"]["offer_share"]


def test_seller_adaptation_does_not_emit_dead_persuasion_knobs():
    assert planned_nudge("persuasion", "seller", 0.8) == {}


def test_dump_game_keeps_full_state_when_score_arrives(monkeypatch, tmp_path):
    monkeypatch.setattr("glee_agent.adapt.GAMES_DIR", tmp_path)
    dump_game("g", {"game_id": "g", "game_state": {"round": 20}})
    dump_game("g", {"game_id": "g", "game_state": None, "rating_delta": -2.2})
    saved = json.loads((tmp_path / "g.json").read_text())
    assert saved["game_state"] == {"round": 20}
    assert saved["rating_delta"] == -2.2
