from glee_agent.triage import classify


def test_triage_marks_supported_bargaining_accept_as_avoidable():
    replay = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "config": {"delta_1": 0.8},
        "result": {"player_1_payoff": 5.0},
    }
    diagnose = {
        "turns": [
            {
                "round": 10,
                "action_type": "decision",
                "actual_action": {"decision": "reject"},
                "action": {"decision": "accept"},
                "path": "bargaining/impatient/mild/decision",
                "calc": {"pot": 100.0, "discounted_last_share": 0.11},
                "segment": {"scenario_id": "cell"},
            }
        ]
    }
    row = classify({"rating_delta": -8.0}, replay, diagnose)
    assert row["classification"] == "avoidable"
    assert row["evidence"][0]["accept_value"] == 0.11
    assert "signal:discount_exposure" in row["factors"]


def test_triage_records_realized_bargaining_regret_without_forcing_policy():
    replay = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "config": {"delta_1": 0.8},
        "result": {"player_1_payoff": 30.0},
    }
    diagnose = {
        "turns": [
            {
                "round": 2,
                "action_type": "decision",
                "actual_action": {"decision": "reject"},
                "action": {"decision": "reject"},
                "path": "bargaining/impatient/hold_first_even/decision",
                "calc": {"pot": 100.0, "discounted_last_share": 0.40},
                "segment": {"scenario_id": "cell"},
            }
        ]
    }
    row = classify({"rating_delta": 1.0}, replay, diagnose)
    assert row["classification"] == "realized_regret"
    assert row["actionable"] is False
    assert row["evidence"][0]["current_policy"] == "reject"
    assert {
        "rating:nonnegative",
        "process:multi_round",
        "signal:realized_regret",
    } <= set(row["factors"])


def test_triage_marks_one_shot_negotiation_buyer_as_constrained():
    replay = {
        "game_family": "negotiation",
        "your_player": "player_2",
        "config": {"max_rounds": 1},
        "result": {"player_2_payoff": 0.0},
    }
    row = classify({"rating_delta": -4.0}, replay, {"turns": []})
    assert row == {
        "classification": "opponent_constrained",
        "reason": "one_shot_responder",
        "actionable": False,
        "evidence": [],
        "factors": [
            "family:negotiation",
            "role:player_2",
            "outcome:unknown",
            "primary:opponent_constrained",
            "rating:negative",
        ],
    }


def test_triage_marks_hidden_no_zopa_only_with_posthoc_evidence():
    replay = {
        "game_family": "negotiation",
        "your_player": "player_1",
        "config": {
            "max_rounds": 10,
            "complete_information": False,
            "seller_value": 1.2,
            "buyer_value": 1.0,
        },
        "result": {"player_1_payoff": 0.0},
    }
    row = classify({"rating_delta": -2.0}, replay, {"turns": []})
    assert row["classification"] == "posthoc_no_zopa"
    assert row["actionable"] is False


def test_triage_marks_profitable_platform_cap_reject_as_avoidable():
    replay = {
        "game_family": "negotiation",
        "your_player": "player_2",
        "config": {"max_rounds": 99, "complete_information": False},
        "result": {"player_2_payoff": 0.0},
    }
    diagnose = {
        "turns": [
            {
                "round": 99,
                "action_type": "decision",
                "actual_action": {"decision": "RejectOffer"},
                "horizon_known": False,
                "complete_information": False,
                "path": "negotiation/open_horizon/strong_buyer/decision",
                "calc": {"profit": 200.0},
                "segment": {"scenario_id": "cell"},
            }
        ]
    }
    row = classify({"rating_delta": -8.0}, replay, diagnose)
    assert row["classification"] == "avoidable"
    assert row["reason"] == "supported_platform_cap_accept"
    assert row["evidence"][0]["accept_profit"] == 200.0


def test_triage_records_realized_negotiation_regret_before_cap():
    replay = {
        "game_family": "negotiation",
        "your_player": "player_1",
        "config": {"max_rounds": 10, "complete_information": True},
        "result": {"player_1_payoff": 10.0},
    }
    diagnose = {
        "turns": [
            {
                "round": 4,
                "action_type": "decision",
                "actual_action": {"decision": "RejectOffer"},
                "action": {"decision": "RejectOffer"},
                "horizon_known": True,
                "complete_information": True,
                "path": "negotiation/complete_seller/decision",
                "calc": {"profit": 17.0},
                "segment": {"scenario_id": "cell"},
            }
        ]
    }
    row = classify({"rating_delta": -7.0}, replay, diagnose)
    assert row["classification"] == "realized_regret"
    assert row["actionable"] is False
    assert row["evidence"][0]["accept_profit"] == 17.0


def test_triage_keeps_overlapping_bargaining_factors():
    replay = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "config": {
            "delta_1": 0.8,
            "complete_information": True,
            "messages_allowed": False,
        },
        "result": {
            "player_1_payoff": 12.0,
            "outcome": "agreement",
            "agreed_round": 12,
        },
    }
    diagnose = {
        "process": {"rounds_played": 12},
        "turns": [
            {
                "round": 2,
                "action_type": "decision",
                "actual_action": {"decision": "reject"},
                "action": {"decision": "reject"},
                "calc": {
                    "pot": 100.0,
                    "discounted_last_share": 0.24,
                    "opponent_state": {
                        "plateau_repeats": 4,
                        "stall_probability": 0.9375,
                    },
                },
                "segment": {
                    "indicators": {
                        "role": "alice",
                        "own_delta": 0.8,
                        "horizon": "open",
                        "complete_information": True,
                        "messages_allowed": False,
                    }
                },
            }
        ],
    }
    row = classify({"rating_delta": -6.0}, replay, diagnose)
    assert len(row["factors"]) == len(set(row["factors"]))
    assert {
        "rating:negative",
        "process:multi_round",
        "process:long_game",
        "signal:discount_exposure",
        "signal:stable_opponent_boundary",
        "signal:realized_regret",
    } <= set(row["factors"])
