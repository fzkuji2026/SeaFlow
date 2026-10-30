from glee_agent.diagnose import replay_to_game, snapshot, snapshot_replay
from glee_agent.strategy import strategy


def test_diagnose_hot_incomplete_alice_share_is_computed():
    game = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "offer"},
        "game_state": {
            "current_player": "player_1",
            "money_to_divide": 10000,
            "delta_1": 0.8,
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 12,
            "round": 1,
        },
    }
    snap = snapshot(game)
    assert snap["path"].startswith("bargaining/impatient_hot/hot/")
    assert snap["calc"]["our_offer_share"] >= 0.45
    assert snap["calc"]["our_offer_gain"] == strategy(game)["alice_gain"]
    assert snap["calc"]["pot"] == 10000
    assert snap["calc"]["plan_ask_share"] >= 0.45
    assert snap["calc"]["spe_share"] is not None


def test_diagnose_negotiation_surplus_share():
    game = {
        "game_family": "negotiation",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_1",
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_1_value": 100.0,
            "player_2_value": 150.0,
            "complete_information": True,
            "horizon_known": True,
            "round": 1,
            "max_rounds": 10,
            "last_offer": {"price": 125.0, "from_player": "player_2"},
        },
    }
    snap = snapshot(game)
    assert snap["calc"]["surplus"] == 50.0
    assert snap["calc"]["profit"] == 25.0
    assert snap["calc"]["surplus_share"] == 0.5
    assert snap["calc"]["own_band"] == "live"
    assert snap["calc"]["close_110"] == 110.0
    assert snap["calc"]["open_seed"] == 150.0
    assert snap["calc"]["last_profitable"] is True
    assert snap["calc"]["would_accept"] is False


def test_diagnose_persuasion_ev():
    game = {
        "game_family": "persuasion",
        "your_player": "player_2",
        "valid_actions": {"type": "buyer_decision"},
        "game_state": {
            "current_player": "player_2",
            "p": 0.5,
            "v": 20000.0,
            "u": 0.0,
            "product_price": 10000,
            "seller_message": "yes",
        },
    }
    snap = snapshot(game)
    assert snap["calc"]["stated_ev"] == 10000.0
    assert snap["calc"]["ev"] == 10000.0
    assert snap["calc"]["ev_minus_price"] == 0.0
    assert snap["calc"]["ev_gt_price"] is False
    assert snap["calc"]["stated_ev_gt_price"] is False
    # da5d16ed / 837d3d14: ev_ratio_092 takes rec=yes at EV==price.
    assert snap["action"]["decision"] == "yes"


def test_replay_to_game_uses_history_before_round():
    replay = {
        "game_id": "x",
        "game_family": "negotiation",
        "your_player": "player_2",
        "state": {
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_2_value": 150.0,
            "complete_information": False,
            "horizon_known": False,
            "round": 3,
            "history": [
                {
                    "round": 1,
                    "offer": {"price": 180.0, "from_player": "player_1"},
                    "decision": "RejectOffer",
                },
                {
                    "round": 3,
                    "offer": {"price": 148.66, "from_player": "player_1"},
                    "decision": "AcceptOffer",
                },
            ],
        },
    }
    game = replay_to_game(replay, round_n=3, action_type="decision")
    assert game["game_state"]["round"] == 3
    assert len(game["game_state"]["history"]) == 1
    assert game["game_state"]["last_offer"]["price"] == 148.66
    snap = snapshot_replay(replay, round_n=3, action_type="decision")
    assert snap["path"].startswith("negotiation/")
    assert "profit" in snap["calc"]
    assert snap["calc"]["close_110"] == 135.0
    assert snap["calc"]["last_price"] == 148.66


def test_replay_to_game_accepts_canonical_game_state_logs():
    replay = {
        "game_id": "x",
        "game_family": "bargaining",
        "your_player": "player_1",
        "game_state": {
            "money_to_divide": 100,
            "delta_1": 0.95,
            "round": 3,
            "history": [
                {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
                {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 40, "player_2_gain": 60}, "decision": "reject"},
                {"round": 3, "proposer": "player_1", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
            ],
        },
    }

    game = replay_to_game(replay, round_n=2, action_type="decision")

    assert game["game_state"]["round"] == 2
    assert len(game["game_state"]["history"]) == 1
    assert game["game_state"]["last_offer"]["player_1_gain"] == 40


def test_replay_offer_preserves_message_availability():
    replay = {
        "game_family": "negotiation",
        "your_player": "player_1",
        "config": {"messages_allowed": True},
        "state": {
            "player_1_role": "seller",
            "player_1_value": 80.0,
            "messages_allowed": True,
            "round": 1,
            "history": [
                {
                    "round": 1,
                    "offer": {"price": 120.0, "from_player": "player_1"},
                    "decision": "RejectOffer",
                }
            ],
        },
    }
    game = replay_to_game(replay, round_n=1, action_type="offer")
    assert "message" in game["valid_actions"]["fields"]
    snap = snapshot_replay(replay, round_n=1, action_type="offer")
    assert "messages=1" in snap["segment"]["scenario_id"]


def test_diagnose_opp_last_share_uses_player_gain_keys():
    game = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "offer"},
        "game_state": {
            "current_player": "player_1",
            "money_to_divide": 1_000_000,
            "delta_1": 0.9,
            "delta_2": 0.95,
            "complete_information": True,
            "horizon_known": False,
            "round": 13,
            "history": [
                {
                    "round": 12,
                    "proposer": "player_2",
                    "offer": {
                        "player_1_gain": 310345.0,
                        "player_2_gain": 689655.0,
                        "proposer": "player_2",
                    },
                    "decision": "reject",
                }
            ],
        },
    }
    snap = snapshot(game)
    assert snap["calc"]["opp_last_gain"] == 310345.0
    assert snap["calc"]["opp_last_share"] == 0.310345


def test_replay_keeps_parked_share_when_last_row_is_our_offer():
    replay = {
        "game_id": "cafac839",
        "game_family": "bargaining",
        "your_player": "player_1",
        "state": {
            "money_to_divide": 1_000_000,
            "delta_1": 0.9,
            "delta_2": 0.95,
            "complete_information": True,
            "horizon_known": False,
            "round": 3,
            "history": [
                {
                    "round": 1,
                    "proposer": "player_1",
                    "offer": {
                        "player_1_gain": 500000.0,
                        "player_2_gain": 500000.0,
                        "proposer": "player_1",
                    },
                    "decision": "reject",
                },
                {
                    "round": 2,
                    "proposer": "player_2",
                    "offer": {
                        "player_1_gain": 310345.0,
                        "player_2_gain": 689655.0,
                        "proposer": "player_2",
                    },
                    "decision": "reject",
                },
                {
                    "round": 3,
                    "proposer": "player_1",
                    "offer": {
                        "player_1_gain": 500000.0,
                        "player_2_gain": 500000.0,
                        "proposer": "player_1",
                    },
                    "decision": "reject",
                },
            ],
        },
    }
    snap = snapshot_replay(replay)
    assert snap["action_type"] == "offer"
    assert snap["calc"]["opp_last_share"] == 0.310345


def test_snapshot_replay_preserves_quality_when_seller_does_not_know_cv():
    replay = {
        "game_id": "blind",
        "game_family": "persuasion",
        "your_player": "player_1",
        "config": {
            "is_seller_know_cv": False,
            "seller_message_type": "binary",
        },
        "state": {
            "round": 1,
            "current_quality": "high",
            "is_seller_know_cv": False,
            "history": [
                {"round": 1, "quality": "low", "seller_message": "yes"}
            ],
        },
    }
    snap = snapshot_replay(replay, round_n=1)
    assert snap["action_type"] == "seller_recommendation"
    assert snap["calc"]["regime"] == "seller_honest"
    assert snap["calc"]["quality"] == "low"


def test_snapshot_replay_preserves_known_text_seller_information():
    replay = {
        "game_id": "known-text",
        "game_family": "persuasion",
        "your_player": "player_1",
        "config": {
            "is_seller_know_cv": True,
            "seller_message_type": "text",
        },
        "state": {
            "round": 1,
            "is_seller_know_cv": True,
            "history": [
                {"round": 1, "quality": "low", "seller_message": "skip"}
            ],
        },
    }
    snap = snapshot_replay(replay, round_n=1)
    assert snap["action_type"] == "seller_message"
    assert snap["calc"]["regime"] == "seller_honest"
    assert snap["calc"]["quality"] == "low"
