from glee_agent.diagnose import snapshot
from glee_agent.rules import parse
from glee_agent.workflows.bargaining.offer_shadow import predict


def _game(*, player="player_2", delta=1.0, round_n=2, history=None):
    return {
        "game_family": "bargaining",
        "your_player": player,
        "valid_actions": {"type": "offer"},
        "game_state": {
            "round": round_n,
            "max_rounds": 12,
            "horizon_known": True,
            "complete_information": False,
            "messages_allowed": False,
            "money_to_divide": 100,
            "delta_1": 0.9 if player == "player_2" else delta,
            "delta_2": delta if player == "player_2" else 0.9,
            "history": history or [],
        },
    }


def test_first_counter_is_context_specific_and_shadow_only():
    result = predict(parse(_game()))
    assert result["route"] == "first_counter"
    assert result["recommended_share"] == 0.60
    assert result["candidates"][0]["train"]["n"] == 110
    assert result["candidates"][0]["holdout"]["acceptance_probability"] == 0.587
    assert predict(parse(_game(delta=0.9)))["recommended_share"] is None


def test_repeat_and_deadline_routes_do_not_choose_the_live_action():
    history = [{
        "round": 1,
        "proposer": "player_1",
        "offer": {"player_1_gain": 60, "player_2_gain": 40, "proposer": "player_1"},
        "decision": "reject",
    }]
    result = predict(parse(_game(player="player_1", round_n=3, history=history)))
    assert result["route"] == "repeat_after_reject"
    assert result["recommended_share"] == 0.55

    game = _game(round_n=12)
    action = {"player_1_gain": 1, "player_2_gain": 99}
    snap = snapshot(game, action)
    assert snap["action"] == action
    assert snap["calc"]["offer_route_shadow"]["route"] == "deadline_claim"
    assert snap["calc"]["offer_route_shadow"]["recommended_share"] == 1.0
