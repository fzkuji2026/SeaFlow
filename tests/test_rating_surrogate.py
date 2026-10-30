from glee_agent.rating_surrogate import FEATURES, _predict, features
from glee_agent.diagnose import snapshot
from glee_agent.rules import parse


def _game(round_n=2):
    return {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "round": round_n,
            "max_rounds": 10,
            "horizon_known": True,
            "complete_information": True,
            "messages_allowed": False,
            "money_to_divide": 100,
            "delta_1": 0.9,
            "delta_2": 0.8,
            "last_offer": {
                "player_1_gain": 60,
                "player_2_gain": 40,
                "proposer": "player_2",
            },
        },
    }


def test_features_are_decision_time_visible_and_discounted():
    values = features(
        parse(_game()), {"player_1_gain": 60, "player_2_gain": 40}, response=True
    )
    assert values is not None and len(values) == len(FEATURES)
    assert values[0] == 0.54
    assert values[1] == 0.6
    assert values[3] == 1.0
    assert values[10] == 1.0


def test_features_do_not_leak_hidden_opponent_delta():
    game = _game()
    game["game_state"]["complete_information"] = False
    values = features(
        parse(game),
        {"player_1_gain": 60, "player_2_gain": 40},
        response=True,
    )
    assert values is not None
    assert values[5] == 0.0
    assert values[6] == 0.0


def test_serialized_prediction_supports_both_targets():
    row = {"kind": "linear", "mean": [0], "scale": [2], "coef": [4], "intercept": 1}
    assert _predict(row, [2]) == 5
    row["kind"] = "logistic"
    assert 0.99 < _predict(row, [2]) < 1


def test_diagnostic_snapshot_exposes_shadow_without_changing_action():
    action = {"decision": "reject"}
    snap = snapshot(_game(), action)
    assert snap["action"] == action
    assert set(snap["calc"]["rating_shadow"]) == {
        "positive_probability",
        "expected_rating_delta",
    }
