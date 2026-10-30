from glee_agent.policy import DEFAULT, set_policy
from glee_agent.strategy import strategy


def setup_function():
    set_policy(DEFAULT)


def test_bargaining_asks_for_a_bit_more():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "messages_allowed": True,
            },
        }
    )
    assert action["alice_gain"] + action["bob_gain"] == 100
    assert action["alice_gain"] == 55


def test_bargaining_accepts_fair_offer_when_horizon_is_known():
    action = strategy(
        {
            "game_family": "bargaining",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "bob",
                "money_to_divide": 100,
                "last_offer": {"bob_gain": 50, "alice_gain": 50},
                "horizon_known": True,
                "max_rounds": 10,
                "round": 1,
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_rejects_even_split_when_patient():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "last_offer": {"alice_gain": 5000, "bob_gain": 5000},
                "inflation_2": 0,
                "messages_allowed": True,
                "round": 1,
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_does_not_demand_the_whole_pie():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 1.0,
                "delta_2": 0.0,
                "complete_information": True,
            },
        }
    )
    assert action["alice_gain"] + action["bob_gain"] == 10000
    assert action["alice_gain"] <= 7000
    assert action["bob_gain"] >= 3000


def test_patient_alice_does_not_repeat_80_20_after_even_split():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 1.0,
                "delta_2": 0.95,
                "horizon_known": False,
                "round": 3,
                "last_offer": {"alice_gain": 500_000, "bob_gain": 500_000},
                "messages_allowed": True,
            },
        }
    )
    assert action["alice_gain"] + action["bob_gain"] == 1_000_000
    assert 520_000 <= action["alice_gain"] <= 580_000


def test_patient_vs_impatient_opens_65_then_keeps_existing_fallbacks():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 1.0,
                "delta_2": 0.9,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] == 6500
    assert action["bob_gain"] == 3500


def test_patient_alice_rejects_even_split_while_opponent_inflates():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 1.0,
                "delta_2": 0.95,
                "horizon_known": False,
                "round": 2,
                "last_offer": {"alice_gain": 500_000, "bob_gain": 500_000},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_patient_alice_holds_repeated_even_split_for_edge():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 1.0,
                "delta_2": 0.95,
                "horizon_known": False,
                "round": 4,
                "last_offer": {"alice_gain": 500_000, "bob_gain": 500_000},
                "history": [
                    {"alice_gain": 500_000, "bob_gain": 500_000},
                    {"alice_gain": 800_000, "bob_gain": 200_000},
                    {"alice_gain": 500_000, "bob_gain": 500_000},
                ],
            },
        }
    )
    assert action == {"decision": "reject"}


def test_impatient_bob_takes_even_split_by_round_8():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 0.90,
                "delta_2": 0.90,
                "horizon_known": False,
                "round": 11,
                "last_offer": {"alice_gain": 50, "bob_gain": 50},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_does_not_raise_after_public_curve_opening():
    def ask(round_n, last=None):
        state = {
            "current_player": "player_1",
            "money_to_divide": 10000,
            "delta_1": 0.90,
            "delta_2": 0.90,
            "horizon_known": False,
            "round": round_n,
        }
        if last:
            state["last_offer"] = last
            state["history"] = [last]
        return strategy(
            {
                "game_family": "bargaining",
                "your_player": "player_1",
                "valid_actions": {"type": "offer"},
                "game_state": state,
            }
        )["alice_gain"]

    assert ask(1) >= ask(8)
    assert ask(11, {"alice_gain": 4518, "bob_gain": 5482}) <= 5000


def test_bargaining_takes_improving_offer_when_inflating():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 0.90,
                "delta_2": 0.90,
                "horizon_known": False,
                "round": 12,
                "last_offer": {"alice_gain": 4518, "bob_gain": 5482},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_bob_rejects_penultimate_concession_below_next_half():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "delta_1": 0.95,
                "delta_2": 0.95,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 11,
                "last_offer": {"alice_gain": 5500, "bob_gain": 4500},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_hidden_impatient_rating_vetoes_first_40_percent_counter():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 0.9,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 3,
                "last_offer": {
                    "player_1_gain": 59.82,
                    "player_2_gain": 40.18,
                    "proposer": "player_1",
                },
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 62, "player_2_gain": 38},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    },
                ],
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_known_patient_opponent_takes_exact_first_38():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 1.0,
                "delta_2": 0.9,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 62,
                    "player_2_gain": 38,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hidden_known_deadline_hot_takes_exact_first_38():
    # 1d9d540d / 794e0736: rejecting 38% at delta 0.8 and taking 40.18%
    # two rounds later lost 12.28% of the undiscounted pot in both games.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_2": 0.8,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 620_000,
                    "player_2_gain": 380_000,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hidden_known_deadline_mild_rejects_exact_first_38():
    # Recent exact matches scored -7.1, -6.1, -3.5, and -3.1 when accepted;
    # nearby rejections commonly recovered 47.5% in round 2.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 62,
                    "player_2_gain": 38,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_hidden_known_deadline_takes_discounted_half_value():
    # 8f821041: delta 0.9 rejected 47.37%, then got a nominal half at r2;
    # discounting reduced the actual payoff to 45% and scored -3.5.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_2": 0.9,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 526_315.79,
                    "player_2_gain": 473_684.21,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hidden_known_deadline_mild_holds_below_half():
    # The transfer-validated continuation policy improves all six external
    # matches in this pooled-message T=12 cell, including 48% offers.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 52,
                    "player_2_gain": 48,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hot_responder_rating_vetoes_patient_opponent_opening():
    # dce6c963: 30.92% now beat 53.25% at r7 after delta-0.8 discounting,
    # 30.92% versus 13.96% of the original pot.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 6_908,
                    "player_2_gain": 3_092,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_equal_discount_open_horizon_takes_current_value():
    # e390d11a: delta 0.9 rejected 47.47% at r1, then accepted 47.67% at r3;
    # discounting reduced the actual share to 38.61% and scored -6.8.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_1": 0.9,
                "delta_2": 0.9,
                "complete_information": True,
                "horizon_known": False,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 5_253,
                    "player_2_gain": 4_747,
                    "message": "final offer, or we both get zero",
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hidden_impatient_takes_unknown_horizon_39_counter():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.9,
                "complete_information": False,
                "horizon_known": False,
                "round": 2,
                "last_offer": {
                    "player_1_gain": 39,
                    "player_2_gain": 61,
                    "proposer": "player_2",
                },
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 52, "player_2_gain": 48},
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "accept"}


def test_negotiation_seller_opens_above_value():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 40,
                "messages_allowed": True,
            },
        }
    )
    assert action["product_price"] == 60  # 1.50x at round 1, incomplete info


def test_negotiation_buyer_rejects_unprofitable_early():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 100,
                "last_offer": {"price": 120},
                "messages_allowed": True,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 10,
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert 60 <= action["product_price"] < 120


def test_negotiation_walks_unprofitable_unknown_horizon_late():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_2_role": "buyer",
                "player_2_value": 100,
                "last_offer": {"price": 120},
                "round": 12,
            },
        }
    )
    # Live type, unproven ZOPA: reservation counter, not WalkAway.
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] <= 100


def test_negotiation_counters_repeated_unprofitable_price():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 100,
                "last_offer": {"price": 120},
                "round": 1,
                "history": [
                    {"offer": {"price": 120}},
                    {"offer": {"price": 120}},
                    {"offer": {"price": 120}},
                ],
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert 60 <= action["product_price"] <= 100


def test_negotiation_open_horizon_live_continues_flat_unprofitable_plateau():
    for you, role, value, bad_price in (
        ("player_1", "seller", 100.0, 80.0),
        ("player_2", "buyer", 10_000.0, 10_900.0),
    ):
        opponent = "player_2" if you == "player_1" else "player_1"
        history = []
        for round_n in range(1, 8):
            sender = opponent if round_n % 2 == 0 else you
            history.append(
                {
                    "round": round_n,
                    "offer": {
                        "price": bad_price if sender == opponent else value * 1.2,
                        "from_player": sender,
                        "round": round_n,
                    },
                    "decision": "RejectOffer",
                }
            )
        action = strategy(
            {
                "game_family": "negotiation",
                "your_player": you,
                "valid_actions": {"type": "decision"},
                "game_state": {
                    "current_player": you,
                    f"{you}_role": role,
                    f"{you}_value": value,
                    "complete_information": False,
                    "horizon_known": False,
                    "round": 8,
                    "history": history,
                    "last_offer": {
                        "price": bad_price,
                        "from_player": opponent,
                        "round": 8,
                    },
                },
            }
        )
        assert action["decision"] == "RejectOffer"
        if role == "seller":
            assert action["product_price"] >= value
        else:
            assert action["product_price"] <= value


def test_negotiation_buyer_takes_zero_profit_early():
    # 528bd908: r1 accept at value (gain 0) scored −5.08. Same payoff as
    # no-deal, and they can still concede. Last round still takes 0.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_2_role": "buyer",
                "player_2_value": 15000,
                "last_offer": {"price": 15000},
                "round": 1,
                "max_rounds": 10,
                "horizon_known": True,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_buyer_takes_thin_profit_early():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_2_role": "buyer",
                "player_2_value": 80,
                "last_offer": {"price": 75},
                "horizon_known": True,
                "round": 1,
                "max_rounds": 10,
                "messages_allowed": True,
            },
        }
    )
    # Official starter takes any non-negative profit. 75 vs 80 is +5.
    assert action == {"decision": "AcceptOffer"}



def test_negotiation_complete_buyer_holds_zopa_split():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 12000,
                "player_2_value": 15000,
                "complete_information": True,
                "last_offer": {"price": 13500.0},
                "horizon_known": True,
                "round": 7,
                "max_rounds": 10,
                "messages_allowed": True,
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_negotiation_buyer_accepts_real_surplus_early():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_2_role": "buyer",
                "player_2_value": 80,
                "last_offer": {"price": 60},
                "horizon_known": True,
                "round": 2,
                "max_rounds": 10,
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_negotiation_buyer_accepts_near_target_midgame():
    action = strategy(
        {
            "game_family": "negotiation",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_2_role": "buyer",
                "player_2_value": 80,
                "last_offer": {"price": 62},
                "horizon_known": True,
                "round": 5,
                "max_rounds": 10,
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_negotiation_seller_concedes_over_rounds():
    def ask(round_n):
        return strategy(
            {
                "game_family": "negotiation",
                "valid_actions": {"type": "offer"},
                "game_state": {
                    "current_player": "player_1",
                    "player_1_role": "seller",
                    "player_1_value": 40,
                    "horizon_known": True,
                    "round": round_n,
                    "max_rounds": 10,
                },
            }
        )["product_price"]

    assert ask(1) > ask(5)
    assert ask(10) == 40.0  # last known round: Faratin α=1 → reservation


def test_negotiation_one_shot_live_seller_leaves_120_buyer_positive_gain():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 1000000,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
                "messages_allowed": True,
            },
        }
    )
    assert action["product_price"] == 1198750.0


def test_negotiation_one_shot_120_seller_claims_only_hidden_zopa():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 120.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
            },
        }
    )
    assert action["product_price"] == 149.85


def test_persuasion_seller_skips_low_before_last():
    # 122a77fa / e925a022 / 29a3f2d1: r1 skip, +EV buyers never came back.
    # Recommend the first low; skip later lows only after a sale.
    fresh = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {"current_quality": "low", "round": 1, "total_rounds": 20},
        }
    )
    assert fresh == {"decision": "yes"}

    burned = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {
                "current_quality": "low",
                "round": 4,
                "total_rounds": 20,
                "history": [
                    {
                        "quality": "low",
                        "bought": True,
                        "seller_message": "This unit is high quality. I recommend buying it.",
                    }
                ],
            },
        }
    )
    assert burned == {"decision": "no"}


def test_persuasion_seller_always_yes_flag_recommends_low():
    from glee_agent.policy import DEFAULT, set_policy

    set_policy(
        {
            **DEFAULT,
            "persuasion": {**DEFAULT["persuasion"], "seller_always_yes": True},
        }
    )
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {
                "current_quality": "low",
                "round": 2,
                "total_rounds": 20,
                "history": [
                    {
                        "quality": "low",
                        "bought": True,
                        "seller_message": "yes",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "yes"}
    set_policy(DEFAULT)


def test_persuasion_f6b28640_negative_prior_ev_builds_credibility():
    base = {
        "p": 1.0 / 3.0,
        "v": 2_000_000.0,
        "u": 0.0,
        "product_price": 1_000_000,
        "is_seller_know_cv": True,
        "round": 1,
        "total_rounds": 20,
    }
    low = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_1",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {**base, "current_quality": "low"},
        }
    )
    high = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_1",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {**base, "current_quality": "high"},
        }
    )
    assert low == {"decision": "no"}
    assert high == {"decision": "yes"}


def test_persuasion_equal_prior_ev_builds_credibility():
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_1",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {
                "p": 0.8,
                "v": 1_250_000.0,
                "u": 0.0,
                "product_price": 1_000_000,
                "is_seller_know_cv": True,
                "current_quality": "low",
                "round": 1,
                "total_rounds": 20,
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_learns_repeated_low_text_after_opposite_high_label():
    history = [
        {
            "quality": "low",
            "seller_message": "A solid unit this round.",
            "buyer_decision": "yes",
            "bought": True,
        }
        for _ in range(3)
    ]
    history.append(
        {
            "quality": "high",
            "seller_message": "This one is genuinely excellent.",
            "buyer_decision": "yes",
            "bought": True,
        }
    )
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "p": 0.5,
                "v": 400.0,
                "u": 0.0,
                "product_price": 100,
                "is_seller_know_cv": True,
                "seller_message_type": "text",
                "seller_message": "A solid unit this round.",
                "history": history,
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_does_not_learn_a_fixed_message_cycle_as_quality():
    low_message = "Strong buy signal from me - it's a winner."
    history = [
        {
            "round": round_n,
            "quality": "low",
            "seller_message": low_message,
            "buyer_decision": "yes",
            "bought": True,
        }
        for round_n in (4, 8, 12)
    ]
    history.append(
        {
            "round": 13,
            "quality": "high",
            "seller_message": "This is a great buy - highly recommended!",
            "buyer_decision": "yes",
            "bought": True,
        }
    )
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "p": 0.8,
                "v": 4_000_000.0,
                "u": 0.0,
                "product_price": 1_000_000,
                "is_seller_know_cv": True,
                "seller_message_type": "text",
                "seller_message": low_message,
                "round": 16,
                "history": history,
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_c1264688_seller_uses_inverse_buyer_response():
    history = [
        {"seller_message": "yes", "buyer_decision": "yes", "bought": True},
        {"seller_message": "yes", "buyer_decision": "no", "bought": False},
        {"seller_message": "yes", "buyer_decision": "no", "bought": False},
        {"seller_message": "no", "buyer_decision": "yes", "bought": True},
        {"seller_message": "no", "buyer_decision": "yes", "bought": True},
    ]
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {
                "current_quality": "high",
                "round": 6,
                "total_rounds": 20,
                "history": history,
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_seller_recommends_high_quality():
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {"current_quality": "high"},
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_seller_without_cv_still_uses_private_quality():
    base = {
        "is_seller_know_cv": False,
        "round": 2,
        "total_rounds": 20,
    }
    for quality, decision in (("low", "yes"), ("high", "yes")):
        action = strategy(
            {
                "game_family": "persuasion",
                "your_player": "player_1",
                "valid_actions": {"type": "seller_recommendation"},
                "game_state": {**base, "current_quality": quality},
            }
        )
        assert action == {"decision": decision}

    rejected = [
        {"seller_message": "yes", "buyer_decision": "no", "bought": False}
        for _ in range(3)
    ]
    low = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_1",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {**base, "current_quality": "low", "history": rejected},
        }
    )
    high = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_1",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {**base, "current_quality": "high", "history": rejected},
        }
    )
    assert low == {"decision": "no"}
    assert high == {"decision": "yes"}


def test_persuasion_seller_may_lie_on_last_round():
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {
                "current_quality": "low",
                "horizon_known": True,
                "round": 20,
                "total_rounds": 20,
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_buyer_uses_expected_value():
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {"p": 0.8, "v": 10, "u": 0, "product_price": 6},
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_buyer_ignores_untested_cheap_talk_under_ev():
    # 1f311461: p=1/3, v=200, price=100. A first "recommend" used to
    # boost the prior and buy a lemon (-4).
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "p": 0.3333333333333333,
                "v": 200.0,
                "u": 0.0,
                "product_price": 100,
                "seller_message": "I recommend this product.",
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_buyer_honest_refuses_thin_ev():
    # After one true high, p=1/3 v=120 price=100: boosted EV still < price.
    # Old path bought on v>price and ate thin/negative units.
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "p": 0.3333333333333333,
                "v": 120.0,
                "u": 0.0,
                "product_price": 100,
                "seller_message": "I recommend this product.",
                "history": [
                    {
                        "seller_message": "I recommend this product.",
                        "bought": True,
                        "quality": "high",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_buyer_ignores_caught_liar():
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "p": 0.4,
                "v": 10,
                "u": 0,
                "product_price": 6,
                "seller_message": "This unit is worth buying at the listed price.",
                "history": [
                    {
                        "seller_message": "worth buying",
                        "bought": True,
                        "quality": "low",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_buyer_liar_takes_ev_equals_price_after_one_lemon():
    # Peak ratio 1.0: after one lemon, prior 0.50→0.35, EV==price still takes.
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "p": 0.5,
                "v": 10.0,
                "u": 0.0,
                "product_price": 3.5,
                "seller_message": "yes",
                "history": [
                    {
                        "seller_message": "yes",
                        "bought": True,
                        "quality": "low",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_negotiation_complete_info_seller_takes_most_surplus_early():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 40,
                "player_2_value": 80,
                "complete_information": True,
            },
        }
    )
    # Round 1: open 90% of surplus (field prior; t=0): 40 + 0.90*40 = 76
    assert action["product_price"] == 76.0


def test_negotiation_complete_seller_closes_repeated_lowball_at_public_payoff():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 120,
                "player_2_value": 150,
                "complete_information": True,
                "horizon_known": True,
                "round": 4,
                "max_rounds": 10,
                "last_offer": {"price": 122.4, "from_player": "player_2", "round": 4},
                "history": [
                    {"round": 1, "offer": {"price": 147, "from_player": "player_1"}, "decision": "RejectOffer"},
                    {"round": 2, "offer": {"price": 122.4, "from_player": "player_2"}, "decision": "RejectOffer"},
                    {"round": 3, "offer": {"price": 146.93, "from_player": "player_1"}, "decision": "RejectOffer"},
                ],
            },
        }
    )
    # Same-scenario public seller payoff is 0.10x base: 120 + 10 = 130.
    assert action == {"decision": "RejectOffer", "product_price": 130.0}


def test_negotiation_ultimatum_seller_asks_near_buyer_value():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 10000,
                "player_2_value": 12000,
                "complete_information": True,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
                "messages_allowed": False,
            },
        }
    )
    # One-shot known ZOPA: 99% of $2k surplus → $11,980.
    assert action["product_price"] == 11980
    assert "message" not in action


def test_negotiation_walks_when_no_zopa():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 90,
                "player_2_value": 40,
                "complete_information": True,
                "last_offer": {"price": 50},
            },
        }
    )
    assert action == {"decision": "WalkAway"}


def test_negotiation_final_reject_has_no_counteroffer():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 80,
                "last_offer": {"price": 90},
                "horizon_known": True,
                "round": 5,
                "max_rounds": 5,
            },
        }
    )
    assert action == {"decision": "RejectOffer"}


def test_negotiation_follows_opponent_concession():
    first = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 40,
                "last_offer": {"price": 45},
                "history": [{"offer": {"price": 42}}, {"offer": {"price": 45}}],
                "horizon_known": True,
                "round": 3,
                "max_rounds": 10,
                "messages_allowed": True,
            },
        }
    )
    stuck = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 40,
                "last_offer": {"price": 42},
                "history": [{"offer": {"price": 42}}, {"offer": {"price": 42}}],
                "horizon_known": True,
                "round": 3,
                "max_rounds": 10,
                "messages_allowed": True,
            },
        }
    )
    assert first["product_price"] < stuck["product_price"]
    assert first["product_price"] >= 40


def test_negotiation_89579670_dc998107_waits_on_early_repeated_offer():
    history = []
    for round_n in range(1, 7):
        opponent = round_n % 2 == 1
        history.append(
            {
                "round": round_n,
                "offer": {
                    "price": 147 if opponent else 115,
                    "from_player": "player_1" if opponent else "player_2",
                    "round": round_n,
                },
                "decision": "RejectOffer",
            }
        )
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100,
                "player_2_value": 150,
                "complete_information": True,
                "horizon_known": False,
                "round": 7,
                "history": history,
                "last_offer": {
                    "price": 147,
                    "from_player": "player_1",
                    "round": 7,
                },
            },
        }
    )
    assert action["decision"] == "RejectOffer"

    thinner_history = [
        {
            **row,
            "offer": {
                **row["offer"],
                "price": 150 if row["offer"]["from_player"] == "player_1" else 115,
            },
        }
        for row in history
    ]
    thinner = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100,
                "player_2_value": 150,
                "complete_information": True,
                "horizon_known": False,
                "round": 7,
                "history": thinner_history,
                "last_offer": {
                    "price": 150,
                    "from_player": "player_1",
                    "round": 7,
                },
            },
        }
    )
    assert thinner["decision"] == "RejectOffer"


def test_negotiation_seller_meets_unprofitable_bid():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 40,
                "last_offer": {"price": 30},
                "round": 3,
                "max_rounds": 10,
                "horizon_known": True,
                "messages_allowed": True,
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert 50 < action["product_price"] <= 52


def test_negotiation_2a2fcac0_live_seller_counters_near_130():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 2,
                "max_rounds": 10,
                "history": [
                    {"round": 1, "offer": {"price": 150.0, "from_player": "player_1"}}
                ],
                "last_offer": {"price": 75.0, "from_player": "player_2", "round": 2},
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert 125.0 < action["product_price"] <= 130.0


def test_negotiation_2b9b0ccb_answers_bounded_acceptance_token():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {
                "type": "decision",
                "fields": {"decision": {}, "product_price": {}, "message": {}},
            },
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 100,
                "complete_information": False,
                "horizon_known": True,
                "round": 2,
                "max_rounds": 10,
                "messages_allowed": True,
                "last_offer": {
                    "price": 50,
                    "from_player": "player_2",
                    "message": "if you will write zzz in your text I will accept whatever offer",
                },
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["message"] == "zzz"
    assert action["product_price"] >= 100


def test_negotiation_buyer_takes_zero_profit_late():
    # Live 1.0×: incomplete_late still takes any non-negative. Strong 1.5×
    # holds 0.90× (`4a906084` / `639a6735`).
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_2_role": "buyer",
                "player_2_value": 10000,
                "last_offer": {"price": 10000},
                "round": 9,
                "max_rounds": 10,
                "horizon_known": True,
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_bargaining_last_round_takes_tiny_positive_crumb():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "last_offer": {"player_2_gain": 1, "player_1_gain": 99},
                "horizon_known": True,
                "round": 3,
                "max_rounds": 3,
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_omits_message_when_forbidden():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer", "fields": {"alice_gain": {}, "bob_gain": {}}},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "messages_allowed": False,
            },
        }
    )
    assert "message" not in action
    assert action["alice_gain"] + action["bob_gain"] == 100


def test_negotiation_walk_progress_speeds_concession():
    from glee_agent.policy import DEFAULT, set_policy

    game = {
        "game_family": "negotiation",
        "your_player": "player_1",
        "valid_actions": {"type": "offer"},
        "game_state": {
            "current_player": "player_1",
            "player_1_role": "seller",
            "player_1_value": 40,
            "horizon_known": True,
            "round": 5,
            "max_rounds": 10,
        },
    }
    set_policy({**DEFAULT, "negotiation": {**DEFAULT["negotiation"], "walk_progress": 0.90}})
    slow = strategy(game)["product_price"]
    set_policy({**DEFAULT, "negotiation": {**DEFAULT["negotiation"], "walk_progress": 0.20}})
    fast = strategy(game)["product_price"]
    set_policy(DEFAULT)
    assert fast <= slow



def test_bargaining_rejects_sub_half_midgame():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "last_offer": {"player_2_gain": 36, "player_1_gain": 64},
                "horizon_known": True,
                "round": 6,
                "max_rounds": 10,
            },
        }
    )
    assert action == {"decision": "reject"}


def test_negotiation_one_shot_accepts_fair_complete_split():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 120,
                "player_2_value": 150,
                "complete_information": True,
                "last_offer": {"price": 135},
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}



def test_bargaining_bob_takes_hidden_deadline_opening_48():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "delta_1": None,
                "delta_2": 0.90,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {"player_1_gain": 5200, "player_2_gain": 4800},
            },
        }
    )
    # 7668a2a3: rejecting 48% with delta=.9 delayed the same 50/50 split to
    # round 12 and reduced normalized payoff from .48 to .157.
    assert action == {"decision": "accept"}


def test_bargaining_bob_takes_known_deadline_opening_42_against_patient_opponent():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "delta_2": 0.90,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {"player_1_gain": 5_800, "player_2_gain": 4_200},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_4af4fac4_takes_equal_095_deadline_opening_48():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {"player_1_gain": 5_200, "player_2_gain": 4_800},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_equal_095_takes_first_counter_after_rejected_opener():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": False,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 51.28, "player_2_gain": 48.72},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "round": 2,
                    "proposer": "player_2",
                    "player_1_gain": 45,
                    "player_2_gain": 55,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_e1163eb0_hidden_mild_rejects_45_before_discount():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 12,
                "last_offer": {"player_1_gain": 5_500, "player_2_gain": 4_500, "proposer": "player_1", "round": 1},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_e09dcd1a_complete_responder_takes_spe_share():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.9,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": False,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "player_1_gain": 31,
                    "player_2_gain": 69,
                    "proposer": "player_2",
                    "round": 2,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_b5cbba40_tests_opponent_before_taking_opening_spe_share():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_1": 0.95,
                "delta_2": 0.90,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {"player_1_gain": 700_000, "player_2_gain": 300_000},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_bob_rejects_early_42_even_if_impatient():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "delta_1": 0.90,
                "delta_2": 0.90,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {"player_1_gain": 5800, "player_2_gain": 4200},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_hot_inflation_takes_45_rather_than_wait():
    # 20%/round: waiting past r9 for 50/50 is how Bob ended at $4.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_1": 1.0,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 9,
                "last_offer": {"player_1_gain": 550000.0, "player_2_gain": 450000.0},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_last_round_proposer_claims_99_percent():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer", "fields": {"message": {}}},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "messages_allowed": True,
                "horizon_known": True,
                "round": 12,
                "max_rounds": 12,
            },
        }
    )
    assert action["alice_gain"] == 99
    assert action["bob_gain"] == 1
    assert "final round" in action["message"]


def test_bargaining_known_deadline_claims_99_after_low_counter():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 1.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 12,
                "max_rounds": 12,
                "history": [
                    {
                        "round": 11,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 60, "player_2_gain": 40},
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert action["alice_gain"] == 1
    assert action["bob_gain"] == 99


def test_bargaining_open_round_99_keeps_normal_offer_policy():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 99,
                "history": [
                    {
                        "round": 97,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    },
                    {
                        "round": 98,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 45.2, "player_2_gain": 54.8},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "round": 98,
                    "proposer": "player_2",
                    "player_1_gain": 45.2,
                    "player_2_gain": 54.8,
                },
            },
        }
    )
    assert action == {"alice_gain": 50.0, "bob_gain": 50.0}


def test_bargaining_unknown_horizon_rejects_crumb_late():
    # Horizon unknown: r12 is not a deadline. 5% Alice crumbs still ~-7.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.90,
                "delta_2": 0.90,
                "horizon_known": False,
                "round": 12,
                "last_offer": {"alice_gain": 5, "bob_gain": 95},
            },
        }
    )
    assert action == {"decision": "reject"}



def test_bargaining_last_round_takes_alice_5pct():
    # A low positive payoff still ranks above the zero from walking away.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 0.95,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 12,
                "last_offer": {"alice_gain": 500, "bob_gain": 9500},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_last_round_takes_even_split():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 12,
                "last_offer": {"alice_gain": 5000, "bob_gain": 5000},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_penultimate_ask_stays_near_half_after_crumbs():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 0.95,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 11,
                "last_offer": {"alice_gain": 1500, "bob_gain": 8500},
            },
        }
    )
    assert 4800 <= action["alice_gain"] <= 5200



def test_negotiation_complete_last_round_takes_thin_surplus():
    # d54617b8 / 3fa508ec: last-round reject of 15–17% surplus scored 0.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 1000000.0,
                "player_2_value": 1500000.0,
                "complete_information": True,
                "last_offer": {"price": 1087548.61},
                "horizon_known": True,
                "round": 10,
                "max_rounds": 10,
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}



def test_bargaining_hot_inflation_rejects_39_with_known_horizon():
    # 904ce545 / a0eb38eb: taking 39% / 38.1% immediately scored poorly;
    # a next-round 52% agreement pays 41.6% after discounting.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 0.80,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {"player_1_gain": 61, "player_2_gain": 39},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_mild_alice_opens_at_most_55():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 0.90,
                "horizon_known": False,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] + action["bob_gain"] == 10000
    assert action["alice_gain"] <= 5500
    assert action["alice_gain"] >= 5000


def test_incomplete_mild_first_proposer_uses_public_curve_optimum():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] == 5000
    assert action["bob_gain"] == 5000


def test_bargaining_patient_opponent_opening_uses_current_field_boundary():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] == 4700
    assert action["bob_gain"] == 5300


def test_bargaining_hidden_t12_bob_uses_public_response_curve():
    base = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "offer"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 10_000,
            "delta_2": 0.95,
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 12,
            "round": 2,
            "messages_allowed": False,
            "history": [
                {
                    "round": 1,
                    "proposer": "player_1",
                    "offer": {"player_1_gain": 6000, "player_2_gain": 4000},
                    "decision": "reject",
                }
            ],
        },
    }
    assert strategy(base) == {"alice_gain": 4700, "bob_gain": 5300}

    with_messages = dict(base)
    with_messages["game_state"] = dict(base["game_state"], messages_allowed=True)
    message_action = strategy(with_messages)
    assert message_action["alice_gain"] == 4700
    assert message_action["bob_gain"] == 5300


def test_persuasion_zero_ev_recovers_loss_then_stops_after_lemon():
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "round": 2,
                "total_rounds": 20,
                "p": 0.5,
                "v": 200,
                "u": 0,
                "product_price": 100,
                "is_seller_know_cv": False,
                "seller_message_type": "binary",
                "seller_message": "yes",
                "history": [
                    {
                        "round": 1,
                        "seller_message": "yes",
                        "buyer_decision": "yes",
                        "bought": True,
                        "quality": "low",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "yes"}

    recovered = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "round": 3,
                "total_rounds": 20,
                "p": 0.5,
                "v": 200,
                "u": 0,
                "product_price": 100,
                "is_seller_know_cv": False,
                "seller_message_type": "binary",
                "seller_message": "yes",
                "history": [
                    {
                        "round": 1,
                        "seller_message": "yes",
                        "buyer_decision": "yes",
                        "bought": True,
                        "quality": "low",
                    },
                    {
                        "round": 2,
                        "seller_message": "yes",
                        "buyer_decision": "yes",
                        "bought": True,
                        "quality": "high",
                    },
                ],
            },
        }
    )
    assert recovered == {"decision": "no"}


def test_bargaining_mild_takes_concession_when_it_beats_next_even_split():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.9,
                "complete_information": False,
                "horizon_known": False,
                "round": 5,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 3950, "player_2_gain": 6050},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5000, "player_2_gain": 5000},
                        "decision": "reject",
                    },
                    {
                        "round": 3,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 4250, "player_2_gain": 5750},
                        "decision": "reject",
                    },
                    {
                        "round": 4,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5000, "player_2_gain": 5000},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "player_1_gain": 4520,
                    "player_2_gain": 5480,
                    "proposer": "player_2",
                    "round": 5,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_open_horizon_takes_patient_opponents_early_counter():
    state = {
        "current_player": "player_1",
        "money_to_divide": 100,
        "delta_1": 0.9,
        "delta_2": 1.0,
        "complete_information": True,
        "horizon_known": False,
        "round": 2,
        "history": [
            {
                "round": 1,
                "proposer": "player_1",
                "offer": {"player_1_gain": 50, "player_2_gain": 50},
                "decision": "reject",
            }
        ],
        "last_offer": {
            "player_1_gain": 36,
            "player_2_gain": 64,
            "proposer": "player_2",
            "round": 2,
        },
    }
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": state,
        }
    )
    assert action == {"decision": "accept"}

    state["last_offer"]["player_1_gain"] = 34
    state["last_offer"]["player_2_gain"] = 66
    assert strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": state,
        }
    ) == {"decision": "reject"}


def test_negotiation_open_horizon_live_takes_repeated_close_110_offer():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 10_000,
                "complete_information": False,
                "horizon_known": False,
                "round": 15,
                "history": [
                    {
                        "round": 10,
                        "offer": {"price": 11_760, "from_player": "player_2"},
                        "decision": "RejectOffer",
                    },
                    {
                        "round": 12,
                        "offer": {"price": 12_000, "from_player": "player_1"},
                        "decision": "RejectOffer",
                    },
                    {
                        "round": 14,
                        "offer": {"price": 11_760, "from_player": "player_2"},
                        "decision": "RejectOffer",
                    },
                ],
                "last_offer": {
                    "price": 11_760,
                    "from_player": "player_2",
                    "round": 15,
                },
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_negotiation_open_horizon_live_rejects_early_three_percent_stall():
    history = []
    for round_n in range(7, 14):
        seller = round_n % 2 == 1
        history.append(
            {
                "round": round_n,
                "offer": {
                    "price": 120.0 if seller else 103.0,
                    "from_player": "player_1" if seller else "player_2",
                    "round": round_n,
                },
                "decision": "RejectOffer",
            }
        )
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 14,
                "history": history,
                "last_offer": {
                    "price": 103.0,
                    "from_player": "player_2",
                    "round": 14,
                },
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] >= 110.0


def test_bargaining_alice_incomplete_unknown_horizon_does_not_open_40():
    # Dashboard −5.3: Alice 5% inflation, incomplete, no round cap, r1
    # 4000/6000 accepted immediately. Defaulted opp δ=0.97 made SPE 0.40.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 0.95,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
                "messages_allowed": True,
            },
        }
    )
    assert action["alice_gain"] + action["bob_gain"] == 10000
    assert action["alice_gain"] >= 5000



def test_bargaining_powerless_alice_rating_vetoes_discounted_20():
    # A positive r1 crumb still rejects. After our offer is rejected, 20% at
    # r2 beats the behavior-weighted continuation; last-round 10% also takes.
    base = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_1",
            "money_to_divide": 1_000_000,
            "delta_1": 0.80,
            "delta_2": 1.0,
            "horizon_known": True,
            "max_rounds": 12,
            "last_offer": {"alice_gain": 100000, "bob_gain": 900000},
        },
    }
    r1 = dict(base)
    r1["game_state"] = dict(base["game_state"], round=1)
    r12 = dict(base)
    r12["game_state"] = dict(base["game_state"], round=12)
    r2 = dict(base)
    r2["game_state"] = dict(
        base["game_state"],
        round=2,
        last_offer={"alice_gain": 200000, "bob_gain": 800000},
        history=[
            {
                "round": 1,
                "proposer": "player_1",
                "offer": {"alice_gain": 500000, "bob_gain": 500000},
                "decision": "reject",
            }
        ],
    )
    from glee_agent.player import strategy
    assert strategy(r1) == {"decision": "reject"}
    assert strategy(r2) == {"decision": "accept"}
    assert strategy(r12) == {"decision": "accept"}


def test_bargaining_36c90b36_two_rejections_do_not_collapse_continuation():
    # 36c90b36: accepting 20% at r4 after two rejected 55% offers scored -7.7.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "delta_2": 0.90,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 4,
                "history": [
                    {
                        "round": round_n,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 55, "player_2_gain": 45},
                        "decision": "reject",
                    }
                    for round_n in (1, 3)
                ],
                "last_offer": {
                    "player_1_gain": 20,
                    "player_2_gain": 80,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_9a566941_hot_takes_first_38_but_not_later_repeat():
    base = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_2": 0.8,
            "complete_information": False,
            "horizon_known": False,
            "last_offer": {"player_1_gain": 62, "player_2_gain": 38},
        },
    }
    first = strategy(base)
    assert first == {"decision": "accept"}

    after_rejection = dict(base)
    after_rejection["game_state"] = dict(
        base["game_state"],
        round=3,
        history=[
            {
                "round": 1,
                "proposer": "player_1",
                "offer": {"player_1_gain": 62, "player_2_gain": 38},
                "decision": "reject",
            },
            {
                "round": 2,
                "proposer": "player_2",
                "offer": {"player_1_gain": 48, "player_2_gain": 52},
                "decision": "reject",
            },
        ],
    )
    assert strategy(after_rejection) == {"decision": "reject"}


def test_bargaining_3347d545_offers_52_after_three_repeated_halves():
    history = []
    for round_n in range(1, 9):
        mine = 600_000 if round_n % 2 else 500_000
        history.append(
            {
                "round": round_n,
                "proposer": "player_1" if round_n % 2 else "player_2",
                "offer": {
                    "player_1_gain": mine,
                    "player_2_gain": 1_000_000 - mine,
                },
                "decision": "reject",
            }
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 1.0,
                "delta_2": 0.9,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 9,
                "history": history,
            },
        }
    )
    assert action["alice_gain"] == 520_000
    assert action["bob_gain"] == 480_000


def test_bargaining_hot_rating_vetoes_30_after_two_even_rejections():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "delta_2": 0.8,
                "complete_information": False,
                "horizon_known": False,
                "round": 5,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 8785, "player_2_gain": 1215},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 5000, "player_2_gain": 5000},
                        "decision": "reject",
                    },
                    {
                        "round": 3,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 8774, "player_2_gain": 1226},
                        "decision": "reject",
                    },
                    {
                        "round": 4,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 5000, "player_2_gain": 5000},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "player_1_gain": 6983,
                    "player_2_gain": 3017,
                    "proposer": "player_1",
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_ac5b04d7_mild_rejects_38_after_even_rejection():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "player_1_gain": 38,
                    "player_2_gain": 62,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_4da63cfe_mild_keeps_half_floor_after_rejection():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.9,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 4,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 32, "player_2_gain": 68}, "decision": "reject"},
                    {"round": 3, "proposer": "player_1", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
                ],
                "last_offer": {"player_1_gain": 39, "player_2_gain": 61, "proposer": "player_2", "round": 4},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_c5b5b2ac_takes_discount_stop_after_two_rejections():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.9,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 4,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 5_200, "player_2_gain": 4_800}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 3_909.09, "player_2_gain": 6_090.91}, "decision": "reject"},
                    {"round": 3, "proposer": "player_1", "offer": {"player_1_gain": 5_000, "player_2_gain": 5_000}, "decision": "reject"},
                ],
                "last_offer": {
                    "round": 4,
                    "proposer": "player_2",
                    "player_1_gain": 4_127.27,
                    "player_2_gain": 5_872.73,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_30ef9918_takes_counter_beating_next_even():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.9,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "round": 2,
                    "proposer": "player_2",
                    "player_1_gain": 46,
                    "player_2_gain": 54,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_bb439bd2_takes_patient_counter_after_rejected_half():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5_000, "player_2_gain": 5_000},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "round": 2,
                    "proposer": "player_2",
                    "player_1_gain": 4_200,
                    "player_2_gain": 5_800,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_powerless_rejects_low_counter_below_rating_gate():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 0.8,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 400_000, "player_2_gain": 600_000},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "round": 2,
                    "proposer": "player_2",
                    "player_1_gain": 115_470,
                    "player_2_gain": 884_530,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_242d6341_takes_hidden_horizon_patient_counter_gap():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "round": 2,
                    "proposer": "player_2",
                    "player_1_gain": 38,
                    "player_2_gain": 62,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_e090294d_patient_responder_holds_for_sixty():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 0.8,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 46,
                    "player_2_gain": 54,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_2891d7ef_patient_closes_even_penultimate():
    history = []
    for round_n in (1, 3, 5, 7, 9):
        history.append(
            {
                "round": round_n,
                "proposer": "player_1",
                "offer": {"player_1_gain": 6_000, "player_2_gain": 4_000},
                "decision": "reject",
            }
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 1.0,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 11,
                "history": history,
            },
        }
    )
    assert action["alice_gain"] == 5_000
    assert action["bob_gain"] == 5_000


def test_bargaining_unvalidated_open_horizon_response_falls_back():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_1": 1.0,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": False,
                "round": 3,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 6200, "player_2_gain": 3800}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 5000, "player_2_gain": 5000}, "decision": "reject"},
                ],
                "last_offer": {"player_1_gain": 5900, "player_2_gain": 4100, "proposer": "player_1", "round": 3},
            },
        }
    )
    assert action == {"decision": "reject"}



def test_bargaining_takes_52_early_above_mild_floor():
    # 22befeb2: Bob δ=0.95 vs Alice 0, r2 51.8% was rejected by RUFL (−6.5).
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 1.0,
                "delta_2": 0.95,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "last_offer": {"player_1_gain": 48.18, "player_2_gain": 51.82},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hidden_opponent_patient_probes_60_once():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] == 600000

    fallback = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 3,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 600_000, "player_2_gain": 400_000},
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert fallback["alice_gain"] == 550000


def test_bargaining_hidden_equal_095_keeps_48_floor_before_penultimate():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 4,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 52, "player_2_gain": 48},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 43.26, "player_2_gain": 56.74},
                        "decision": "reject",
                    },
                    {
                        "round": 3,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "round": 4,
                    "proposer": "player_2",
                    "player_1_gain": 44.35,
                    "player_2_gain": 55.65,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_unknown_horizon_takes_first_hidden_48():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_2": 0.9,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 520_000,
                    "player_2_gain": 480_000,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hot_unknown_horizon_takes_exact_first_38():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 0.8,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 62,
                    "player_2_gain": 38,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hot_unknown_horizon_rating_vetoes_low_counter():
    state = {
        "current_player": "player_1",
        "money_to_divide": 1_000_000,
        "delta_1": 0.8,
        "complete_information": False,
        "horizon_known": False,
        "round": 2,
        "history": [
            {
                "round": 1,
                "proposer": "player_1",
                "offer": {"player_1_gain": 520_000, "player_2_gain": 480_000},
                "decision": "reject",
            }
        ],
        "last_offer": {
            "round": 2,
            "proposer": "player_2",
            "player_1_gain": 190_000,
            "player_2_gain": 810_000,
        },
    }
    game = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": state,
    }
    assert strategy(game) == {"decision": "accept"}

    state["last_offer"] = {
        "round": 2,
        "proposer": "player_2",
        "player_1_gain": 10_000,
        "player_2_gain": 990_000,
    }
    assert strategy(game) == {"decision": "reject"}


def test_bargaining_repeated_45_stall_taken():
    hist = [
        {
            "round": 2,
            "proposer": "player_2",
            "offer": {"player_1_gain": 4520, "player_2_gain": 5480},
            "decision": "reject",
        },
        {
            "round": 4,
            "proposer": "player_2",
            "offer": {"player_1_gain": 4520, "player_2_gain": 5480},
            "decision": "reject",
        },
    ]
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 0.9,
                "complete_information": False,
                "horizon_known": False,
                "round": 6,
                "history": hist,
                "last_offer": {"player_1_gain": 4520, "player_2_gain": 5480},
            },
        }
    )
    assert action == {"decision": "accept"}



def test_negotiation_2f950e24_complete_late_rejects_forty_percent():
    # 2f950e24: 40.18% at r8 scored -4.5. In the latest 160 negotiation
    # games, every 40-45% complete_late acceptance lost rating.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 1200000.0,
                "player_2_value": 1500000.0,
                "complete_information": True,
                "last_offer": {"price": 1320545.45},
                "horizon_known": False,
                "round": 8,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_7e009627_complete_penultimate_rejects_forty_seven_percent():
    # 7e009627: buyer accepted 47% at r9 and scored -5.5; rejecting still
    # leaves our final-round proposal.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 80.0,
                "player_2_value": 120.0,
                "complete_information": True,
                "last_offer": {"price": 101.2},
                "horizon_known": True,
                "round": 9,
                "max_rounds": 10,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_hidden_seller_takes_profit_from_round_2():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 40.0,
                "last_offer": {"price": 42.0},
                "horizon_known": True,
                "round": 3,
                "max_rounds": 10,
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_persuasion_buyer_refuses_negative_ev():
    # 0.85× / 0.92× thresholds bought when EV < price (21/166 buyer games < 0).
    action = strategy(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "p": 0.8,
                "v": 10.0,
                "u": 0.0,
                "product_price": 9.0,
                "seller_message": "yes",
            },
        }
    )
    assert action == {"decision": "no"}


def test_negotiation_oneshot_takes_any_positive_complete_gain():
    # 75f9fbdc: rejecting a positive 5% slice produced zero and -7.4. With
    # no continuation, even a thin positive payoff dominates rejection.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 10000,
                "player_2_value": 15000,
                "complete_information": True,
                "last_offer": {"price": 14999},
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
            },
        }
    )
    assert action["decision"] == "AcceptOffer"


def test_negotiation_oneshot_still_rejects_outside_zopa():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 8000,
                "player_2_value": 10000,
                "complete_information": True,
                "last_offer": {"price": 12000},
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_persuasion_takes_when_prior_ev_equals_price():
    # Peak genome untested_ev_ratio 1.0: EV≥price buys (official starter).
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 12500,
                "u": 0,
                "product_price": 10000,
                "is_seller_know_cv": False,
                "seller_message": "yes",
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_da5eb622_probes_supported_signal_after_skip():
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.5,
                "v": 120.0,
                "u": 0.0,
                "product_price": 100.0,
                "is_seller_know_cv": True,
                "seller_message_type": "text",
                "seller_message": "This one is genuinely good.",
                "history": [
                    {
                        "seller_message": "I would skip this one.",
                        "buyer_decision": "no",
                        "bought": False,
                    }
                ],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_71752dd3_does_not_probe_unsupported_signal():
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 1.0 / 3.0,
                "v": 120.0,
                "u": 0.0,
                "product_price": 100.0,
                "is_seller_know_cv": True,
                "seller_message_type": "text",
                "seller_message": "yes",
                "history": [
                    {"seller_message": "no", "buyer_decision": "no", "bought": False}
                    for _ in range(7)
                ],
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_stops_after_known_seller_lemon():
    # 7de3952f / 9b354751: known p=0.5 v=3e6 price=1e6, r1 lemon then
    # lemon_block left remaining +EV unsold. Prior 0.50→0.35, EV 1.05e6>1e6.
    # Hard block overridden; two-low p=0.8 v=2 still stops (see
    # test_persuasion_known_two_lemons_p08_v2_stops).
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.5,
                "v": 3_000_000,
                "u": 0,
                "product_price": 1_000_000,
                "is_seller_know_cv": True,
                "seller_message": "yes",
                "history": [
                    {
                        "quality": "low",
                        "seller_message": "yes",
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                ],
            },
        }
    )
    assert action == {"decision": "yes"}



def test_bargaining_hot_incomplete_takes_20_early():
    # fefd36f8 / 3eed80f4: 20–21% early take_positive is a gift.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.80,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "last_offer": {"player_1_gain": 20.0, "player_2_gain": 80.0},
            },
        }
    )
    assert action == {"decision": "reject"}



def test_bargaining_patient_penultimate_holds_43():
    # 27293183: Bob δ=1, r11/12, 43% (−7.3). We can still offer 50 on r12.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 1.0,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 11,
                "last_offer": {"player_1_gain": 57.0, "player_2_gain": 43.0},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_70aa406f_patient_penultimate_holds_conceding_46():
    history = []
    for round_n, share in ((1, 0.38), (3, 0.384), (5, 0.394), (7, 0.418)):
        history.append(
            {
                "round": round_n,
                "proposer": "player_1",
                "offer": {
                    "player_1_gain": 1 - share,
                    "player_2_gain": share,
                },
                "decision": "reject",
            }
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1,
                "delta_2": 1.0,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 11,
                "history": history,
                "last_offer": {
                    "player_1_gain": 0.535,
                    "player_2_gain": 0.465,
                    "proposer": "player_1",
                },
            },
        }
    )
    assert action == {"decision": "reject"}



def test_bargaining_patient_rejects_repeated_48():
    # 694aa25f: Alice δ=1, no horizon, took 48% on the second ask (−6.5).
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 6,
                "history": [
                    {
                        "round": 4,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 4800, "player_2_gain": 5200},
                        "decision": "reject",
                    }
                ],
                "last_offer": {"player_1_gain": 4800, "player_2_gain": 5200, "proposer": "player_2"},
            },
        }
    )
    assert action == {"decision": "reject"}



def test_bargaining_c88bf3b2_patient_last_round_takes_20():
    # c88bf3b2: walking 20% last scored -7.9; taking is strictly positive.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 1.0,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 12,
                "last_offer": {"player_1_gain": 2000, "player_2_gain": 8000},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_6545915e_mild_does_not_collapse_after_rejections():
    history = []
    for round_n in (1, 3, 5, 7, 9):
        history.append(
            {
                "round": round_n,
                "proposer": "player_1",
                "offer": {"player_1_gain": 50, "player_2_gain": 50},
                "decision": "reject",
            }
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 10,
                "history": history,
                "last_offer": {
                    "player_1_gain": 3,
                    "player_2_gain": 97,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_dd8dc13a_patient_rejects_even_then_counters_52():
    history = [
        {
            "round": 1,
            "proposer": "player_1",
            "offer": {"player_1_gain": 6000, "player_2_gain": 4000},
            "decision": "reject",
        },
        {
            "round": 2,
            "proposer": "player_2",
            "offer": {"player_1_gain": 4200, "player_2_gain": 5800},
            "decision": "reject",
        },
        {
            "round": 3,
            "proposer": "player_1",
            "offer": {"player_1_gain": 6000, "player_2_gain": 4000},
            "decision": "reject",
        },
        {
            "round": 4,
            "proposer": "player_2",
            "offer": {"player_1_gain": 5000, "player_2_gain": 5000},
            "decision": "reject",
        },
        {
            "round": 5,
            "proposer": "player_1",
            "offer": {"player_1_gain": 6000, "player_2_gain": 4000},
            "decision": "reject",
        },
    ]
    state = {
        "current_player": "player_1",
        "money_to_divide": 10000,
        "delta_1": 1.0,
        "delta_2": 0.9,
        "complete_information": True,
        "horizon_known": False,
        "round": 6,
        "history": history,
        "last_offer": {
            "player_1_gain": 5000,
            "player_2_gain": 5000,
            "proposer": "player_2",
        },
    }
    decision = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": state,
        }
    )
    assert decision == {"decision": "reject"}

    offer_state = dict(state, round=7)
    offer_state["history"] = history + [
        {
            "round": 6,
            "proposer": "player_2",
            "offer": {"player_1_gain": 5000, "player_2_gain": 5000},
            "decision": "reject",
        }
    ]
    offer = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": offer_state,
        }
    )
    assert offer["alice_gain"] == 5200
    assert offer["bob_gain"] == 4800


def test_bargaining_fc9ffb59_patient_takes_repeated_48_when_opp_delta_unknown():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 4,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5500, "player_2_gain": 4500},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 4800, "player_2_gain": 5200},
                        "decision": "reject",
                    },
                    {
                        "round": 3,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5500, "player_2_gain": 4500},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "player_1_gain": 4800,
                    "player_2_gain": 5200,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_6a4168ad_patient_does_not_discount_rejections():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1000,
                "delta_1": 1.0,
                "horizon_known": False,
                "round": 4,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 550, "player_2_gain": 450},
                        "decision": "reject",
                    },
                    {
                        "round": 3,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 550, "player_2_gain": 450},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "player_1_gain": 364,
                    "player_2_gain": 636,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_last_round_still_walks_zero():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 12,
                "last_offer": {"player_1_gain": 0, "player_2_gain": 100},
            },
        }
    )
    assert action == {"decision": "walkaway"}



def test_bargaining_impatient_vs_hotter_holds_first_even():
    # 4b8bf835 / 675d476a: Bob δ=0.95 vs Alice δ=0.8 took first 50% r1.
    game = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_1": 0.80,
            "delta_2": 0.95,
            "complete_information": True,
            "horizon_known": True,
            "max_rounds": 12,
            "round": 1,
            "last_offer": {"player_1_gain": 50, "player_2_gain": 50},
        },
    }
    assert strategy(game) == {"decision": "reject"}
    game["game_state"]["round"] = 3
    game["game_state"]["history"] = [
        {"proposer": "player_1", "player_1_gain": 50, "player_2_gain": 50},
        {"proposer": "player_1", "player_1_gain": 50, "player_2_gain": 50},
    ]
    assert strategy(game) == {"decision": "accept"}


def test_bargaining_accepts_when_now_dominates_next_planned_offer():
    game = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_1": 0.8,
            "delta_2": 0.9,
            "complete_information": True,
            "horizon_known": False,
            "round": 1,
            "last_offer": {"player_1_gain": 50, "player_2_gain": 50},
        },
    }
    assert strategy(game) == {"decision": "accept"}

    game["game_state"]["last_offer"] = {
        "player_1_gain": 51,
        "player_2_gain": 49,
    }
    assert strategy(game) == {"decision": "reject"}


def test_bargaining_patient_side_of_hotter_pair_opens_sixty():
    # 468637bc and 27 matching first-round closes: 55% was accepted but every
    # scored result was negative. Preserve the higher Rubinstein claim.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )
    assert action == {"alice_gain": 6000.0, "bob_gain": 4000.0}


def test_bargaining_patient_opponent_first_counter_respects_rating_floor():
    # A 3% close can beat a still-later 3% payoff while remaining a severe
    # rating loss. The global rating floor takes precedence over that regret.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.9,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "last_offer": {
                    "round": 2,
                    "proposer": "player_2",
                    "player_1_gain": 3,
                    "player_2_gain": 97,
                },
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {
                            "player_1_gain": 50,
                            "player_2_gain": 50,
                            "proposer": "player_1",
                        },
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_hot_low_counter_respects_common_rating_gate():
    # A 15% close can beat an even lower eventual payoff while remaining a
    # persistent rating loss; the common gate takes precedence.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.8,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "last_offer": {
                    "round": 2,
                    "proposer": "player_2",
                    "player_1_gain": 15,
                    "player_2_gain": 85,
                },
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {
                            "player_1_gain": 52,
                            "player_2_gain": 48,
                            "proposer": "player_1",
                        },
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_hidden_hot_proposer_rating_vetoes_first_counter():
    # 7ce74816: rejecting 20.83% at r2 ended at 5.38% discounted payoff on r10.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.8,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5200, "player_2_gain": 4800},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "round": 2,
                    "proposer": "player_2",
                    "player_1_gain": 2083,
                    "player_2_gain": 7917,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hidden_mild_proposer_rejects_bottom_field_first_counter():
    state = {
        "current_player": "player_1",
        "money_to_divide": 1_000_000,
        "delta_1": 0.9,
        "complete_information": False,
        "horizon_known": True,
        "max_rounds": 12,
        "round": 2,
        "history": [
            {
                "round": 1,
                "proposer": "player_1",
                "offer": {"player_1_gain": 520_000, "player_2_gain": 480_000},
                "decision": "reject",
            }
        ],
        "last_offer": {
            "round": 2,
            "proposer": "player_2",
            "player_1_gain": 113_800,
            "player_2_gain": 886_200,
        },
    }
    game = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": state,
    }
    assert strategy(game) == {"decision": "reject"}

    state["last_offer"] = {
        "round": 2,
        "proposer": "player_2",
        "player_1_gain": 432_634,
        "player_2_gain": 567_366,
    }
    assert strategy(game) == {"decision": "reject"}


def test_bargaining_hidden_patient_closes_repeated_subhalf_offer():
    history = []
    for round_n in range(1, 10):
        mine = round_n % 2 == 1
        history.append(
            {
                "round": round_n,
                "proposer": "player_1" if mine else "player_2",
                "offer": {
                    "player_1_gain": 5500 if mine else 4520,
                    "player_2_gain": 4500 if mine else 5480,
                },
                "decision": "reject",
            }
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 10,
                "history": history,
                "last_offer": {
                    "round": 10,
                    "proposer": "player_2",
                    "player_1_gain": 4520,
                    "player_2_gain": 5480,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hidden_round_99_is_not_terminal():
    state = {
        "current_player": "player_1",
        "money_to_divide": 100,
        "delta_1": 1.0,
        "complete_information": False,
        "horizon_known": False,
        "round": 99,
    }
    offer = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": state,
        }
    )
    assert offer == {"alice_gain": 50.0, "bob_gain": 50.0}

    state["last_offer"] = {
        "round": 99,
        "proposer": "player_2",
        "player_1_gain": 29.43,
        "player_2_gain": 70.57,
    }
    decision = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": state,
        }
    )
    assert decision == {"decision": "reject"}


def test_negotiation_incomplete_late_asks_close_to_reservation():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 100.0,
                "horizon_known": True,
                "round": 9,
                "max_rounds": 10,
            },
        }
    )
    assert action["product_price"] == 110.0


def test_negotiation_incomplete_late_takes_any_positive():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 100.0,
                "horizon_known": True,
                "round": 9,
                "max_rounds": 10,
                "last_offer": {"price": 101.0},
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_negotiation_82bb7150_stubborn_lowball_preserves_last_counter_cycle():
    # r8 accepted 100.4 after our 120 was rejected and scored -5.9. Counter
    # 110 at r9; a rejected counter can still be answered at r10.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {
                "type": "decision",
                "fields": {"product_price": {}},
            },
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 8,
                "max_rounds": 10,
                "last_offer": {
                    "price": 100.4,
                    "from_player": "player_2",
                    "round": 8,
                },
                "history": [
                    {
                        "round": 4,
                        "offer": {"price": 100.4, "from_player": "player_2"},
                        "decision": "RejectOffer",
                    },
                    {
                        "round": 6,
                        "offer": {"price": 100.4, "from_player": "player_2"},
                        "decision": "RejectOffer",
                    },
                    {
                        "round": 7,
                        "offer": {
                            "price": 120.0,
                            "from_player": "player_1",
                            "round": 7,
                        },
                        "decision": "RejectOffer",
                    }
                ],
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] == 110.0


def test_negotiation_open_horizon_late_stall_relaxes_live_floor():
    # 0f6234ba: a below-cost plateau kept the generic live floor at 120 through
    # round 39. Probe the profitable 105 close before walking away.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 21,
                "last_offer": {
                    "price": 82.96,
                    "from_player": "player_2",
                    "round": 20,
                },
                "history": [
                    {
                        "round": 16,
                        "offer": {"price": 82.48, "from_player": "player_2"},
                    },
                    {
                        "round": 18,
                        "offer": {"price": 82.72, "from_player": "player_2"},
                    },
                    {
                        "round": 20,
                        "offer": {"price": 82.96, "from_player": "player_2"},
                    },
                ],
            },
        }
    )
    assert action["product_price"] == 105.0


def test_negotiation_known_zopa_rejects_subhalf_dynamic_ask():
    # bc9816ce: dynamic ask fell to 25% of surplus and accepted a repeated
    # 28.8% counter, scoring -5.9 despite the complete-info 50% floor.
    history = []
    for round_n in range(65, 72):
        seller = round_n % 2 == 1
        history.append(
            {
                "round": round_n,
                "offer": {
                    "price": 100.0 if seller else 91.52,
                    "from_player": "player_1" if seller else "player_2",
                    "round": round_n,
                },
                "decision": "RejectOffer",
            }
        )
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 80.0,
                "player_2_value": 120.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 72,
                "history": history,
                "last_offer": {
                    "price": 91.52,
                    "from_player": "player_2",
                    "round": 72,
                },
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_dc250806_counter_uses_outgoing_penultimate_round():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 1_200_000.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 8,
                "max_rounds": 10,
                "last_offer": {"price": 837_255.0, "from_player": "player_2"},
                "history": [
                    {
                        "round": 2,
                        "offer": {"price": 237_495.0, "from_player": "player_2"},
                    },
                    {
                        "round": 4,
                        "offer": {"price": 337_455.0, "from_player": "player_2"},
                    },
                    {
                        "round": 6,
                        "offer": {"price": 537_375.0, "from_player": "player_2"},
                    },
                ],
            },
        }
    )
    assert action == {"decision": "RejectOffer", "product_price": 1_320_000.0}


def test_persuasion_known_untested_takes_when_ev_equals_price():
    # Peak untested_ev_ratio 1.0: EV==price is a take.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 1.0 / 3.0,
                "v": 3_000_000.0,
                "u": 0.0,
                "product_price": 1_000_000,
                "is_seller_know_cv": True,
                "seller_message": "yes",
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_skips_when_seller_says_no_even_if_ev_high():
    # 9d41cc69 r1: rec=no, p=0.8 v=300 price=100, bought the lemon.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 300.0,
                "u": 0.0,
                "product_price": 100,
                "is_seller_know_cv": False,
                "seller_message": "no",
            },
        }
    )
    assert action == {"decision": "no"}


def test_negotiation_hidden_last_round_takes_nonneg_profit():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 80,
                "last_offer": {"price": 80},
                "horizon_known": True,
                "round": 5,
                "max_rounds": 5,
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_negotiation_open_horizon_keeps_countering_stuck_lowball():
    # 8d984e8b: seller dropped 3.12M → 1.200120M then parked vs buyer 1.2M.
    hist = [
        {"offer": {"price": 3_120_000.0}},
        {"offer": {"price": 1_080_000.0}},
        {"offer": {"price": 2_083_551.0}},
        {"offer": {"price": 1_140_000.0}},
        {"offer": {"price": 1_200_120.0}},
        {"offer": {"price": 1_140_000.0}},
        {"offer": {"price": 1_200_120.0}},
    ]
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 1_200_000.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 15,
                "last_offer": {"price": 1_200_120.0},
                "history": hist,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_a8856536_live_buyer_does_not_walk_at_round_five():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 120.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 5,
                "history": [
                    {"round": 1, "offer": {"price": 147.0, "from_player": "player_1"}},
                    {"round": 2, "offer": {"price": 108.0, "from_player": "player_2"}},
                    {"round": 3, "offer": {"price": 147.0, "from_player": "player_1"}},
                    {"round": 4, "offer": {"price": 108.98, "from_player": "player_2"}},
                ],
                "last_offer": {"price": 147.0, "from_player": "player_1", "round": 5},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_persuasion_913c05f1_skip_phrasing_is_not_a_buy():
    # grok 4.6 913c05f1: "not up to my usual" was treated as untested EV and
    # bought a lemon (−5.33).
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 0.8,
                "v": 200.0,
                "u": 0.0,
                "product_price": 100,
                "is_seller_know_cv": True,
                "seller_message": "Honestly, today's product is not up to my usual standard.",
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_9d8a4b46_not_this_one_is_a_skip():
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 1.0 / 3.0,
                "v": 3_000_000.0,
                "u": 0.0,
                "product_price": 1_000_000,
                "is_seller_know_cv": False,
                "seller_message": (
                    "Not this one. I want you to keep trusting my recommendations, "
                    "so I'm telling you straight: pass this round."
                ),
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_1f241553_hold_off_is_a_skip():
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 0.5,
                "v": 400.0,
                "u": 0.0,
                "product_price": 100.0,
                "is_seller_know_cv": False,
                "seller_message": "I would hold off on this one.",
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_explicit_pass_and_wait_phrases_are_skips():
    messages = (
        "I recommend passing this round.",
        "I'd recommend passing on this one.",
        "I'd pass on this one.",
        "This one is ordinary -- I'd wait.",
        "Ordinary unit this round. Still available at the same price.",
    )
    for message in messages:
        action = strategy(
            {
                "game_family": "persuasion",
                "your_player": "player_2",
                "valid_actions": {"type": "buyer_decision"},
                "game_state": {
                    "current_player": "player_2",
                    "p": 0.8,
                    "v": 300.0,
                    "u": 0.0,
                    "product_price": 100.0,
                    "is_seller_know_cv": True,
                    "seller_message": message,
                },
            }
        )
        assert action == {"decision": "no"}, message


def test_persuasion_negative_semantics_override_positive_keywords():
    messages = (
        "This one is not worth the price. Do not buy it.",
        "I do not recommend this one; sit this one out.",
        "Hard pass even though the listing says high quality.",
    )
    for message in messages:
        action = strategy(
            {
                "game_family": "persuasion",
                "your_player": "player_2",
                "valid_actions": {"type": "buyer_decision"},
                "game_state": {
                    "current_player": "player_2",
                    "p": 0.8,
                    "v": 300.0,
                    "u": 0.0,
                    "product_price": 100.0,
                    "is_seller_know_cv": True,
                    "seller_message": message,
                },
            }
        )
        assert action == {"decision": "no"}, message


def test_persuasion_current_recommendation_beats_past_skip_language():
    messages = (
        "Genuinely good unit this round. My record here is the reason to believe me: "
        "I've told you to skip the weak ones.",
        "Strong unit. I'd take it at this price myself — and I've passed on worse ones "
        "rather than sell you something poor.",
        "This one is the real thing — I checked it myself. When a unit isn't worth your "
        "money I say so, which is why this recommendation should mean something.",
    )
    for message in messages:
        action = strategy(
            {
                "game_family": "persuasion",
                "your_player": "player_2",
                "valid_actions": {"type": "buyer_decision"},
                "game_state": {
                    "current_player": "player_2",
                    "p": 0.8,
                    "v": 125.0,
                    "u": 0.0,
                    "product_price": 100.0,
                    "is_seller_know_cv": True,
                    "seller_message": message,
                    "history": [
                        {
                            "round": 1,
                            "quality": "low",
                            "seller_message": "skip this one",
                            "buyer_decision": "no",
                            "bought": False,
                        }
                    ],
                },
            }
        )
        assert action == {"decision": "yes"}, message


def test_persuasion_zero_ev_iid_prior_recovers_after_one_bought_low():
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 1.0 / 3.0,
                "v": 3_000_000.0,
                "u": 0.0,
                "product_price": 1_000_000,
                "is_seller_know_cv": False,
                "seller_message": "yes",
                "history": [
                    {
                        "round": 1,
                        "quality": "low",
                        "seller_message": "yes",
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                ],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_iid_prior_survives_long_bad_streak():
    history = [
        {
            "round": round_n,
            "quality": "high" if round_n <= 6 else "low",
            "seller_message": "yes",
            "buyer_decision": "yes",
            "bought": True,
        }
        for round_n in range(1, 15)
    ]
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "round": 15,
                "total_rounds": 20,
                "p": 0.5,
                "v": 3_000_000.0,
                "u": 0.0,
                "product_price": 1_000_000,
                "is_seller_know_cv": True,
                "seller_message": "yes",
                "history": history,
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_69510e51_skip_and_ev_equals_price():
    # 69510e51: "not the premium" + EV==price bought a lemon (−4.4).
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 1.0 / 3.0,
                "v": 30000.0,
                "u": 0.0,
                "product_price": 10000,
                "is_seller_know_cv": False,
                "seller_message": "Being straight with you: this is not the premium offering.",
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_88ecd02a_low_p_after_skip_does_not_trust_yes():
    # 88ecd02a: known seller skipped a low then said yes; p=1/3 EV << price.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 1.0 / 3.0,
                "v": 1_250_000.0,
                "u": 0.0,
                "product_price": 1_000_000,
                "is_seller_know_cv": True,
                "seller_message": "yes",
                "history": [{"seller_message": "no", "bought": False, "quality": "low"}],
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_known_seller_skips_do_not_override_negative_ev():
    history = [
        {
            "round": round_n,
            "seller_message": "no",
            "buyer_decision": "no",
            "bought": False,
        }
        for round_n in (1, 2)
    ]
    base = {
        "game_family": "persuasion",
        "your_player": "player_2",
        "valid_actions": {"type": "buyer_decision"},
        "game_state": {
            "current_player": "player_2",
            "p": 1.0 / 3.0,
            "v": 2_000_000.0,
            "u": 0.0,
            "product_price": 1_000_000,
            "seller_message": "yes",
            "history": history,
        },
    }
    known = dict(base)
    known["game_state"] = dict(base["game_state"], is_seller_know_cv=True)
    unknown = dict(base)
    unknown["game_state"] = dict(base["game_state"], is_seller_know_cv=False)
    assert strategy(known) == {"decision": "no"}
    assert strategy(unknown) == {"decision": "no"}


def test_persuasion_after_skip_trust_uses_public_positive_cell():
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 0.8,
                "v": 12000.0,
                "u": 0.0,
                "product_price": 10000,
                "is_seller_know_cv": True,
                "seller_message": "yes",
                "round": 12,
                "history": [{"seller_message": "no", "bought": False}],
            },
        }
    )
    assert action == {"decision": "yes"}

    unknown = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 0.8,
                "v": 12000.0,
                "u": 0.0,
                "product_price": 10000,
                "is_seller_know_cv": False,
                "seller_message_type": "binary",
                "seller_message": "yes",
                "history": [{"seller_message": "no", "bought": False}],
            },
        }
    )
    assert unknown == {"decision": "yes"}


def test_persuasion_9150fd3a_after_skip_trust_takes_ev_eq_price():
    # Peak ratio 1.0: EV==price after a skip still takes.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 0.8,
                "v": 125.0,
                "u": 0.0,
                "product_price": 100,
                "is_seller_know_cv": True,
                "seller_message": "yes",
                "round": 16,
                "history": [{"seller_message": "no", "bought": False}],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_bargaining_fee0234a_hot_rejects_five_percent_vs_hot():
    # fee0234a: both δ=0.80, Bob took 5% r1 (−6.36).
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "delta_1": 0.80,
                "delta_2": 0.80,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {"player_1_gain": 9500.0, "player_2_gain": 500.0},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_66c9656e_hot_rejects_five_percent_early():
    # 66c9656e: Bob δ=0.80 vs Alice 0.95 took 5% r1 (−5.02).
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "delta_2": 0.80,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {"player_1_gain": 95.0, "player_2_gain": 5.0},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_negotiation_1f132a86_oneshot_takes_twenty_eight_percent():
    # 1f132a86: buyer 1.5M vs cost 800k, ask 1.304M is 28% of surplus.
    # RejectOffer scored −5.72; ultimatum responder takes any real slice.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 800000.0,
                "player_2_value": 1500000.0,
                "complete_information": True,
                "last_offer": {"price": 1304000.0},
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
                "messages_allowed": False,
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_negotiation_complete_early_rejects_even_split():
    # 62dec155: buyer accepted half the ZOPA at r1 and scored -5.4. Same-cell
    # low opening rejections usually finished above 75% of the surplus.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {
                "type": "decision",
                "fields": {"product_price": {}},
            },
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 80.0,
                "player_2_value": 100.0,
                "complete_information": True,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 10,
                "last_offer": {
                    "price": 90.0,
                    "from_player": "player_1",
                    "round": 1,
                },
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] == 82.0


def test_bargaining_2d1ddeca_hot_unknown_horizon_rejects_36():
    # 2d1ddeca: Bob δ=0.80, hidden opp, unknown horizon, took 36.1% r1 (−4.84).
    # Same path as 0df96c33 (36% r1, −4.85).
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "delta_2": 0.80,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
                "last_offer": {
                    "player_1_gain": 6390.5,
                    "player_2_gain": 3609.5,
                    "proposer": "player_1",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_persuasion_66c805e1_serviceable_is_a_skip():
    # 66c805e1: known seller, r1 high then "serviceable / nothing fancy" lemon.
    # skip_block should fire; buying revealed the lemon and lemon_blocked later highs.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 400.0,
                "u": 0.0,
                "product_price": 100,
                "is_seller_know_cv": True,
                "seller_message": "Today's unit is serviceable — nothing fancy, does the job. Worth it if you need one now.",
                "history": [
                    {
                        "round": 1,
                        "quality": "high",
                        "seller_message": "Inspected this one personally — it's the high-grade unit today. At this price it's an easy call.",
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                ],
            },
        }
    )
    assert action == {"decision": "no"}


def test_negotiation_9b389a22_complete_t10_rejects_thirty_percent_r1():
    # 9b389a22: buyer 1M vs cost 800k, T=10, r1 ask 940k is 30% of surplus (−4.62).
    # Same complete r1 path as 5d066c62 / 92dc0f22 (35%, −2.8 / −4.2).
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 800000.0,
                "player_2_value": 1000000.0,
                "complete_information": True,
                "last_offer": {"price": 940000.0},
                "horizon_known": True,
                "round": 1,
                "max_rounds": 10,
                "messages_allowed": False,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_92dc0f22_complete_t10_rejects_thirty_five_percent_r1():
    # 92dc0f22: buyer 1.2M vs cost 1M, T=10, r1 ask 1.13M is 35% of surplus (−4.2).
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 1000000.0,
                "player_2_value": 1200000.0,
                "complete_information": True,
                "last_offer": {"price": 1130000.0},
                "horizon_known": True,
                "round": 1,
                "max_rounds": 10,
                "messages_allowed": False,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_2feee8f2_complete_t10_rejects_forty_three_percent_r1():
    # Field even-split attractor is 50%. 43% r1 was the old 0.40 floor.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "player_2_value": 120.0,
                "complete_information": True,
                "last_offer": {"price": 111.4},
                "horizon_known": True,
                "round": 1,
                "max_rounds": 10,
                "messages_allowed": False,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_92681ff0_open_horizon_r1_rejects_zero_profit():
    # 92681ff0: hidden open-horizon buyer, they asked our value r1 (−6.35).
    # Same leaf as 3255faed. Incomplete routes already require gain > 0;
    # open_horizon still used profit >= 0.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
                "last_offer": {"price": 150.0, "from_player": "player_1"},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_3255faed_open_horizon_r1_rejects_zero_profit():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 1500000.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
                "last_offer": {"price": 1500000.0, "from_player": "player_1"},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_c0ef9a90_open_horizon_counters_thin_positive():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 2,
                "last_offer": {"price": 105.0, "from_player": "player_2"},
                "history": [
                    {"offer": {"price": 150.0, "from_player": "player_1"}, "decision": "RejectOffer"},
                    {"offer": {"price": 105.0, "from_player": "player_2"}},
                ],
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_48c03ea1_live_seller_uses_planned_ask():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 1_000_000.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 2,
                "max_rounds": 10,
                "history": [
                    {"round": 1, "offer": {"price": 1_500_000.0, "from_player": "player_1"}}
                ],
                "last_offer": {"price": 1_020_000.0, "from_player": "player_2", "round": 2},
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] > 1_020_000.0


def test_negotiation_0c4182bc_live_buyer_keeps_090_ceiling():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 1_200_000.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 6,
                "history": [
                    {"round": 1, "offer": {"price": 1_600_000.0, "from_player": "player_1"}},
                    {"round": 2, "offer": {"price": 960_000.0, "from_player": "player_2"}},
                    {"round": 3, "offer": {"price": 1_568_000.0, "from_player": "player_1"}},
                    {"round": 4, "offer": {"price": 1_005_166.98, "from_player": "player_2"}},
                    {"round": 5, "offer": {"price": 1_536_640.0, "from_player": "player_1"}},
                ],
                "last_offer": {"price": 1_536_640.0, "from_player": "player_1", "round": 5},
            },
        }
    )
    assert action["product_price"] <= 1_080_000.0


def test_negotiation_528bd908_hidden_r1_rejects_zero_profit():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 1500000.0,
                "last_offer": {"price": 1500000.0},
                "horizon_known": True,
                "round": 1,
                "max_rounds": 10,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_conceder_does_not_take_a_retreat():
    # Seller came down 160 → 140 (60% of surplus for us) then posted 145 (55%).
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "player_2_value": 200.0,
                "complete_information": True,
                "last_offer": {"price": 145.0},
                "history": [
                    {"offer": {"price": 160.0, "from_player": "player_1"}},
                    {"offer": {"price": 140.0, "from_player": "player_1"}},
                    {"offer": {"price": 145.0, "from_player": "player_1"}},
                ],
                "horizon_known": True,
                "round": 4,
                "max_rounds": 10,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_decision_tree_path_is_one_leaf():
    from glee_agent.workflows.tree import path

    leaf = path(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 1000000.0,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
            },
        }
    )
    assert leaf.startswith("negotiation/one_shot/live/close_120/")


def test_negotiation_field_prior_is_frozen_under_pytest():
    from glee_agent.workflows.negotiation._field import DEFAULT_OPEN, field_open_claim

    assert field_open_claim() == DEFAULT_OPEN


def test_negotiation_field_prior_stays_fixed_during_frozen_run(monkeypatch):
    from glee_agent.workflows.negotiation import _field

    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    monkeypatch.setenv("GLEE_FREEZE_POLICY", "1")
    monkeypatch.setitem(_field._MEM, "ts", 1.0)
    monkeypatch.setitem(_field._MEM, "open", 0.93)
    monkeypatch.setattr(
        _field,
        "_read_cache",
        lambda _now: (_ for _ in ()).throw(AssertionError("cache reread")),
    )
    assert _field.field_open_claim() == 0.93


def test_negotiation_hidden_oneshot_seller_asks_below_120_not_125():
    # Seller 1.0x / buyer 1.5x at 1.10x averaged -4.862 because the accepted
    # agreement left half the feasible 1.20x profit unused. Keep below the old
    # rejected 1.25x probe while targeting the next official value boundary.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer", "fields": {"product_price": {}}},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 1000000.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
                "messages_allowed": False,
            },
        }
    )
    assert action["product_price"] == 1198750.0
    assert "message" not in action


def test_persuasion_c1fbf051_below_our_usual_is_a_skip():
    # c1fbf051: known seller, five recommended highs, then
    # "below our usual quality" lemon (−5.36). Same skip miss as cc44e610 r4
    # and 53c6e406. rec was None so buyer_p_high / ev_ratio_p_high bought.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 2000000.0,
                "u": 0.0,
                "product_price": 1000000,
                "is_seller_know_cv": True,
                "round": 6,
                "seller_message": "This unit is below our usual quality this round.",
                "history": [
                    {
                        "round": r,
                        "quality": "high",
                        "seller_message": "This unit is high quality -- recommended.",
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                    for r in range(1, 6)
                ],
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_ad8f2712_wait_for_a_better_one_is_a_skip():
    # ad8f2712: known seller, r1 recommended high, r2
    # "Not this round. Better to wait for a better one." lemon (−6.31).
    # rec was None so buyer_p_high / ev_ratio_p_high bought.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 200.0,
                "u": 0.0,
                "product_price": 100,
                "is_seller_know_cv": True,
                "round": 2,
                "seller_message": "Not this round. Better to wait for a better one.",
                "history": [
                    {
                        "round": 1,
                        "quality": "high",
                        "seller_message": "Solid pick this round, honestly. Worth it.",
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                ],
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_ba960094_wait_for_the_next_is_a_skip():
    # ba960094: unknown seller, three rec-None highs, then
    # "ordinary. Wait for the next -- I'll tell you when it's worth it."
    # "worth" used to classify as rec True before skip. Same wait-for
    # miss as ad8f2712. buyer_p_high / ev_ratio_p_high bought the lemon.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 40000.0,
                "u": 0.0,
                "product_price": 10000,
                "is_seller_know_cv": False,
                "round": 4,
                "seller_message": (
                    "I'll be straight with you: this one is ordinary. "
                    "Wait for the next -- I'll tell you when it's worth it."
                ),
                "history": [
                    {
                        "round": r,
                        "quality": "high",
                        "seller_message": (
                            "This round's product is genuinely strong -- "
                            "I'd take it at this price without hesitating."
                        ),
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                    for r in range(1, 4)
                ],
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_cc44e610_below_our_usual_is_a_skip():
    # cc44e610: unknown seller, p=0.8, EV>price, bought the skip lemon r4.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 4000000.0,
                "u": 0.0,
                "product_price": 1000000,
                "is_seller_know_cv": False,
                "round": 4,
                "seller_message": "This unit is below our usual quality this round.",
                "history": [
                    {
                        "round": r,
                        "quality": "high",
                        "seller_message": "This unit is high quality -- recommended.",
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                    for r in range(1, 4)
                ],
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_c81026c2_nothing_special_is_a_skip():
    # c81026c2: unknown seller, p=0.5 v=400 price=100, EV>price, bought two
    # "nothing special / understand a pass" lemons (−200).
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 0.5,
                "v": 400.0,
                "u": 0.0,
                "product_price": 100,
                "is_seller_know_cv": False,
                "round": 1,
                "seller_message": "This one is nothing special. At this price I'd understand a pass.",
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_0a95bd86_nothing_special_is_a_skip():
    # 0a95bd86: p=0.8, EV>price, "nothing special; your call" lemon r3.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 0.8,
                "v": 20000.0,
                "u": 0.0,
                "product_price": 10000,
                "is_seller_know_cv": False,
                "round": 3,
                "seller_message": "This particular unit is nothing special; your call.",
            },
        }
    )
    assert action == {"decision": "no"}


def test_persuasion_2340bb79_p_high_takes_ev_eq_price():
    # Peak untested_ev_ratio 1.0: p_high EV==price is a take.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 1_250_000.0,
                "u": 0.0,
                "product_price": 1_000_000,
                "is_seller_know_cv": False,
                "round": 1,
                "seller_message": (
                    "Market-validated quality: 80% historical high-grade "
                    "probability with guaranteed payoff."
                ),
                "history": [],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_d817657d_p_high_takes_ev_eq_price():
    # Peak ratio 1.0: same EV==price take as 2340bb79.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 125.0,
                "u": 0.0,
                "product_price": 100,
                "is_seller_know_cv": False,
                "round": 1,
                "seller_message": "I recommend this product.",
                "history": [],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_negotiation_complete_last_offer_asks_80_of_surplus():
    # Last multi-round used to Nash 50%. Leaders still sit at 0.82+; last
    # round is an ultimatum (same as T=1, 0.80). ZOPA [1e6, 1.5e6] → 1.4e6.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer", "fields": {"product_price": {}}},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 1000000.0,
                "player_2_value": 1500000.0,
                "complete_information": True,
                "horizon_known": True,
                "round": 10,
                "max_rounds": 10,
            },
        }
    )
    assert action["product_price"] == 1400000.0


def test_negotiation_field_prior_floor_is_90():
    from glee_agent.workflows.negotiation._field import OPEN_MIN, DEFAULT_OPEN

    assert OPEN_MIN == 0.90
    assert DEFAULT_OPEN == 0.90


def test_negotiation_dead_seller_15x_walks_unprofitable_mid():
    # Seller cost 1.5×1M: no ZOPA vs 0.8/1.0/1.2 buyers. Mid unprofitable
    # bid is a stall, not a Faratin cue.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 1500000.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 5,
                "max_rounds": 10,
                "last_offer": {"price": 1200000.0},
            },
        }
    )
    assert action == {"decision": "WalkAway"}


def test_negotiation_dead_buyer_08x_walks_unprofitable_late():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 8000.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 8,
                "last_offer": {"price": 12000.0},
            },
        }
    )
    assert action == {"decision": "WalkAway"}


def test_negotiation_live_seller_does_not_walk_first_lowball():
    # Seller 1.0×1M is live. r1 bid below cost is RejectOffer, not WalkAway.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 1000000.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 10,
                "last_offer": {"price": 800000.0},
                "messages_allowed": True,
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_persuasion_18cda319_known_lemon_still_buys_when_ev_covers():
    # 18cda319: known seller, p=0.8 v=2e6 price=1e6. r1 rec-yes lemon then
    # lemon_block stopped remaining +EV rounds (−6.46). After one low,
    # prior 0.80→0.65, EV 1.3e6>1e6. Same leaf 87f6f3ac / 2dcd4cd0.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 2000000.0,
                "u": 0.0,
                "product_price": 1000000,
                "is_seller_know_cv": True,
                "round": 2,
                "seller_message": "I recommend this product.",
                "history": [
                    {
                        "round": 1,
                        "quality": "low",
                        "seller_message": "I recommend this product.",
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                ],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_87f6f3ac_known_lemon_still_buys_when_ev_covers():
    # 87f6f3ac: known p=0.8 v=2e6 price=1e6, r1 rec-yes lemon then block
    # (−6.77, payoff −1e6). Next recommend must still buy.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 2000000.0,
                "u": 0.0,
                "product_price": 1000000,
                "is_seller_know_cv": True,
                "round": 2,
                "seller_message": "This product is high quality. I recommend buying it.",
                "history": [
                    {
                        "round": 1,
                        "quality": "low",
                        "seller_message": "This product is high quality. I recommend buying it.",
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                ],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_2dcd4cd0_known_lemon_still_buys_when_ev_covers():
    # 2dcd4cd0: known p=0.8 v=20k price=10k, r1 lemon then lemon_block
    # (−6.52, payoff −10k). Posterior EV 13000>10000.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "p": 0.8,
                "v": 20000.0,
                "u": 0.0,
                "product_price": 10000,
                "is_seller_know_cv": True,
                "round": 2,
                "seller_message": "High-yield opportunity.",
                "history": [
                    {
                        "round": 1,
                        "quality": "low",
                        "seller_message": "High-yield opportunity.",
                        "buyer_decision": "yes",
                        "bought": True,
                    }
                ],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_known_two_lemons_p08_v2_takes_ev_eq_price():
    # Two bought lows: prior 0.80→0.50, EV==price, peak ratio 1.0 still takes.
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 0.8,
                "v": 2000000.0,
                "u": 0.0,
                "product_price": 1000000,
                "is_seller_know_cv": True,
                "round": 3,
                "seller_message": "yes",
                "history": [
                    {
                        "round": 1,
                        "quality": "low",
                        "seller_message": "yes",
                        "bought": True,
                    },
                    {
                        "round": 2,
                        "quality": "low",
                        "seller_message": "yes",
                        "bought": True,
                    },
                ],
            },
        }
    )
    assert action == {"decision": "yes"}


def test_persuasion_known_lemon_skip_still_blocks():
    action = strategy(
        {
            "game_family": "persuasion",
            "your_player": "player_2",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "current_player": "player_2",
                "p": 0.8,
                "v": 2000000.0,
                "u": 0.0,
                "product_price": 1000000,
                "is_seller_know_cv": True,
                "round": 2,
                "seller_message": "This unit is below our usual quality this round.",
                "history": [
                    {
                        "round": 1,
                        "quality": "low",
                        "seller_message": "yes",
                        "bought": True,
                    }
                ],
            },
        }
    )
    assert action == {"decision": "no"}


def test_negotiation_a6e60073_strong_buyer_rejects_reservation_crumb():
    # a6e60073: hidden 1.5× buyer, open-horizon r15 accepted 148.66/150
    # (gain 1.34, opp 28.66, rating_delta −7.46). Faratin-to-reservation.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 15,
                "last_offer": {"price": 148.66, "from_player": "player_1"},
                "history": [
                    {"offer": {"price": 180.0, "from_player": "player_1"}},
                    {"offer": {"price": 135.0, "from_player": "player_2"}},
                    {"offer": {"price": 148.66, "from_player": "player_1"}},
                ],
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_e0fcf54f_strong_buyer_rejects_r1_crumb():
    # e0fcf54f: same 1.5× buyer, r1 accepted 149.9 (gain 0.1). Twin 92d8c468
    # 14800/15000.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
                "last_offer": {"price": 149.9, "from_player": "player_1"},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_92d8c468_strong_buyer_15k_rejects_r1_crumb():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 15000.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
                "last_offer": {"price": 14800.0, "from_player": "player_1"},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_strong_buyer_uses_current_bid_as_acceptance_limit():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 3,
                "last_offer": {"price": 135.0, "from_player": "player_1"},
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert 112.5 <= action["product_price"] < 135.0


def test_negotiation_strong_buyer_last_round_still_takes_positive():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 10,
                "max_rounds": 10,
                "last_offer": {"price": 148.66, "from_player": "player_1"},
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_bargaining_cb84b406_unknown_patient_counters_48_with_half():
    # cb84b406: Bob δ=1, unknown-horizon, they parked 48% r5–99 (−7.47).
    # hold_first_even 50% held to the round cap. Impatient already uses late
    # at unknown r≥12; patient must too. r6 48% still rejects (694aa25f).
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_2": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 13,
                "last_offer": {
                    "player_1_gain": 520000.0,
                    "player_2_gain": 480000.0,
                    "proposer": "player_1",
                },
                "history": [
                    {
                        "round": 5,
                        "proposer": "player_1",
                        "offer": {
                            "player_1_gain": 520000.0,
                            "player_2_gain": 480000.0,
                        },
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_322126f3_unknown_patient_counters_45_with_half():
    # 322126f3: same leaf, 45% r13–99 (−7.6). late floor 0.42.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 13,
                "last_offer": {
                    "player_1_gain": 55.0,
                    "player_2_gain": 45.0,
                    "proposer": "player_1",
                },
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {
                            "player_1_gain": 68.965517,
                            "player_2_gain": 31.034483,
                        },
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_08745132_patient_closes_observed_plateau():
    history = []
    for round_n in range(1, 8):
        ours = round_n % 2 == 1
        history.append(
            {
                "round": round_n,
                "proposer": "player_1" if ours else "player_2",
                "offer": {
                    "player_1_gain": 50.0 if ours else 45.2,
                    "player_2_gain": 50.0 if ours else 54.8,
                },
                "decision": "reject",
            }
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 8,
                "history": history,
                "last_offer": {
                    "round": 8,
                    "proposer": "player_2",
                    "player_1_gain": 45.2,
                    "player_2_gain": 54.8,
                },
            },
        }
    )
    assert action == {"decision": "accept"}

    at_half = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 8,
                "history": [
                    {
                        **row,
                        "offer": {
                            "player_1_gain": 50.0 if row["round"] % 2 else 48.0,
                            "player_2_gain": 50.0 if row["round"] % 2 else 52.0,
                        },
                    }
                    for row in history
                ],
                "last_offer": {
                    "round": 8,
                    "proposer": "player_2",
                    "player_1_gain": 48.0,
                    "player_2_gain": 52.0,
                },
            },
        }
    )
    assert at_half == {"decision": "reject"}


def test_bargaining_rating_vetoes_low_unknown_horizon_boundary():
    history = []
    for round_n in range(1, 16):
        mine = round_n % 2 == 1
        history.append(
            {
                "round": round_n,
                "proposer": "player_1" if mine else "player_2",
                "offer": {
                    "player_1_gain": 5000 if mine else 3510,
                    "player_2_gain": 5000 if mine else 6490,
                },
                "decision": "reject",
            }
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 16,
                "history": history,
                "last_offer": {
                    "round": 16,
                    "proposer": "player_2",
                    "player_1_gain": 3510,
                    "player_2_gain": 6490,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_patient_pair_rating_vetoes_submedian_boundary():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_1": 1.0,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 9,
                "history": [
                    {
                        "round": round_n,
                        "proposer": "player_1" if round_n % 2 else "player_2",
                        "offer": {
                            "player_1_gain": 5480 if round_n % 2 else 4700,
                            "player_2_gain": 4520 if round_n % 2 else 5300,
                        },
                        "decision": "reject",
                    }
                    for round_n in range(1, 9)
                ],
                "last_offer": {
                    "round": 9,
                    "proposer": "player_1",
                    "player_1_gain": 5480,
                    "player_2_gain": 4520,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_rating_vetoes_patient_counter_after_rejection():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 1.0,
                "delta_2": 0.9,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 3,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 62, "player_2_gain": 38},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "round": 3,
                    "proposer": "player_1",
                    "player_1_gain": 59.82,
                    "player_2_gain": 40.18,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_5670188c_rejects_penultimate_below_next_half():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 11,
                "last_offer": {
                    "round": 11,
                    "proposer": "player_1",
                    "player_1_gain": 5790,
                    "player_2_gain": 4210,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_2f5ef19e_patient_stall_proposes_even_split():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 13,
                "history": [
                    {"round": round_n, "proposer": "player_2", "offer": {"player_1_gain": 440_000, "player_2_gain": 560_000}, "decision": "reject"}
                    for round_n in (2, 4, 6, 8, 10, 12)
                ],
                "last_offer": {"player_1_gain": 440_000, "player_2_gain": 560_000, "proposer": "player_2", "round": 12},
            },
        }
    )
    assert action["alice_gain"] == 500_000
    assert action["bob_gain"] == 500_000


def test_bargaining_1fc0f4d6_patient_round_50_offers_even_without_last_offer():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_2": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 50,
                "history": [
                    {
                        "round": round_n,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5_500, "player_2_gain": 4_500},
                        "decision": "reject",
                    }
                    for round_n in (41, 43, 45, 47, 49)
                ],
            },
        }
    )
    assert action == {"alice_gain": 5_000.0, "bob_gain": 5_000.0}


def test_bargaining_90ac85a2_powerless_rejects_zero_crumb():
    # 90ac85a2: Bob δ=0.8 vs Alice δ=1.0, unknown-horizon complete, r1 ~0%
    # accepted (−7.35). Powerless takes ≥8%, not 1e-9 of the pot.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "delta_1": 1.0,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": False,
                "round": 1,
                "last_offer": {
                    "player_1_gain": 9999.99999,
                    "player_2_gain": 9.999999747378752e-06,
                    "proposer": "player_1",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_d914fe8d_powerless_rejects_zero_crumb():
    # d914fe8d: Alice δ=0.8 vs Bob δ=1.0, r2 ~0% accepted (−6.69). Twin.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 0.8,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 2,
                "last_offer": {
                    "player_1_gain": 9.999959729611874e-06,
                    "player_2_gain": 999999.99999,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_negotiation_4a906084_strong_incomplete_late_holds_close_110():
    # 4a906084: hidden 1.5× buyer, T=10 r9 accepted 149.43/150 (−4.38).
    # incomplete_late took any profit≥0 and skipped the strong 0.90× floor.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 9,
                "max_rounds": 10,
                "last_offer": {"price": 149.43, "from_player": "player_1"},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_639a6735_strong_incomplete_late_rejects_at_value():
    # 639a6735: same leaf, r9 accepted 150/150 (gain 0).
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 9,
                "max_rounds": 10,
                "last_offer": {"price": 150.0, "from_player": "player_1"},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_9a6d56b4_strong_seller_opens_150_not_110():
    # 9a6d56b4: hidden 0.8× seller opened 88 on cost 80. Buyer took it
    # (payoff 8 vs 62, rating_delta −7.10). close_110 is a gift.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer", "fields": {"product_price": {}}},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 80.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
            },
        }
    )
    assert action["product_price"] == 120.0


def test_negotiation_3ad43fcc_strong_seller_opens_150_not_110():
    # 3ad43fcc: same leaf, cost 800k opened 880k, they took r1
    # (80k vs 620k, −6.60).
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer", "fields": {"product_price": {}}},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 800000.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
            },
        }
    )
    assert action["product_price"] == 1200000.0


def test_negotiation_one_shot_strong_seller_claims_next_hidden_type():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer", "fields": {"product_price": {}}},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 8000.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
            },
        }
    )
    assert action["product_price"] == 11990.0


def test_negotiation_strong_seller_rejects_hidden_123_cost_signature():
    # 4b7303aa and five twins: accepting 0.984M scored about -5; the existing
    # strong-seller counter holds near 1.19M for the inferred 1.2M buyer.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {
                "type": "decision",
                "fields": {"product_price": {}},
            },
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 800_000.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 2,
                "max_rounds": 10,
                "last_offer": {
                    "price": 984_000.0,
                    "from_player": "player_2",
                    "round": 2,
                },
                "history": [
                    {
                        "round": 1,
                        "offer": {
                            "price": 1_200_000.0,
                            "from_player": "player_1",
                            "round": 1,
                        },
                        "decision": "RejectOffer",
                        "counteroffer": 984_000.0,
                    }
                ],
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] == 1_189_727.98


def test_negotiation_strong_seller_probes_high_hidden_buyer_counter():
    # 7aff49d6: cost 80, hidden buyer countered 102 at r2. The price proves a
    # buyer type above the 1.0 base, but immediate acceptance scored -5.8.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 80.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 2,
                "max_rounds": 10,
                "history": [
                    {
                        "round": 1,
                        "offer": {"price": 120.0, "from_player": "player_1"},
                        "decision": "RejectOffer",
                    }
                ],
                "last_offer": {
                    "price": 102.0,
                    "from_player": "player_2",
                    "round": 2,
                },
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] > 118.0


def test_negotiation_447ea6b4_persists_inferred_buyer_floor_open_horizon():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 80.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 4,
                "history": [
                    {"round": 1, "offer": {"price": 120.0, "from_player": "player_1"}, "decision": "RejectOffer"},
                    {"round": 2, "offer": {"price": 102.0, "from_player": "player_2"}, "decision": "RejectOffer"},
                    {"round": 3, "offer": {"price": 118.97, "from_player": "player_1"}, "decision": "RejectOffer"},
                ],
                "last_offer": {"price": 102.6, "from_player": "player_2", "round": 4},
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] == 118.97


def test_bargaining_2993bab5_vs_hotter_holds_42_mid():
    # 2993bab5: Alice δ=0.95 vs Bob 0.8, T=12 r8 accepted 42.35% (−7.1).
    # impatient_vs_hotter + conceder used late 0.42 in mid. late is
    # penultimate / unknown r≥12 only.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 0.95,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 8,
                "last_offer": {
                    "player_1_gain": 423546.0,
                    "player_2_gain": 576454.0,
                    "proposer": "player_2",
                },
                "history": [
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {
                            "player_1_gain": 381361.0,
                            "player_2_gain": 618639.0,
                        },
                        "decision": "reject",
                    },
                    {
                        "round": 4,
                        "proposer": "player_2",
                        "offer": {
                            "player_1_gain": 387698.0,
                            "player_2_gain": 612302.0,
                        },
                        "decision": "reject",
                    },
                    {
                        "round": 6,
                        "proposer": "player_2",
                        "offer": {
                            "player_1_gain": 401213.0,
                            "player_2_gain": 598787.0,
                        },
                        "decision": "reject",
                    },
                ],
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_d53c74ec_vs_hotter_holds_46_mid():
    # d53c74ec: same leaf, r4 46.3% (−6.65).
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 0.9,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 4,
                "last_offer": {
                    "player_1_gain": 463000.0,
                    "player_2_gain": 537000.0,
                    "proposer": "player_2",
                },
                "history": [
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {
                            "player_1_gain": 433090.90909091,
                            "player_2_gain": 566909.09090909,
                        },
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_57f7e1eb_hot_incomplete_opens_52():
    # 57f7e1eb: Alice δ=0.8, hidden Bob, T=12, opened 4000/6000, they
    # took it (−5.4). 0.45 still taken (c0a8ac31). Short-window 50% sales
    # rank below the observed 52% opening in the same hidden T=12 cell.
    action = strategy(
        {
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
    )
    assert action["alice_gain"] + action["bob_gain"] == 10000
    assert action["alice_gain"] == 5200


def test_bargaining_hot_incomplete_open_uses_supported_52_share():
    # Current field data favors 52% over the regressed 55% opening, while
    # still avoiding the older 40% underclaim.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10000,
                "delta_1": 0.8,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] == 5200


def test_bargaining_c0a8ac31_hot_incomplete_uses_52_percent_opening():
    # c0a8ac31: after the 0.45 floor, Alice still opened 45/55 and they
    # took it (−5.2). Same leaf as 57f7e1eb.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.8,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] == 52
    assert action["bob_gain"] == 48


def test_bargaining_57f0394f_hot_bob_asks_52_not_45():
    # 57f0394f: Bob δ=0.8, r1 30% reject, r2 asked 45% they took (−5.7).
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_1": 0.95,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {
                            "player_1_gain": 700000.0,
                            "player_2_gain": 300000.0,
                            "proposer": "player_1",
                        },
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert action["bob_gain"] == 520000
    assert action["alice_gain"] == 480000


def test_bargaining_5a27261f_hot_incomplete_uses_52_percent_opening():
    # 5a27261f: Alice 46% r1 they took (−4.3). Twin of c0a8ac31.
    action = strategy(
        {
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
    )
    assert action["alice_gain"] == 5200


def test_bargaining_0a768fbc_powerless_opens_40_not_12():
    # 0a768fbc: Alice δ=0.8 vs Bob 1.0 opened 12/88 they took. Historical
    # 50% openings were rejected 34/39 times, so offer the patient side 60%.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.8,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] == 40
    assert action["bob_gain"] == 60


def test_bargaining_9d89a4c5_powerless_opens_40_not_12():
    # 9d89a4c5: same leaf, pie 1e6, opened 120k/880k they took.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 0.8,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] == 400000


def test_bargaining_904ce545_hot_rejects_39_with_known_horizon():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_2": 0.8,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "player_1_gain": 610000.0,
                    "player_2_gain": 390000.0,
                    "proposer": "player_1",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_64564a5b_hot_takes_42_above_continuation():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "delta_1": 0.95,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "player_1_gain": 5800.0,
                    "player_2_gain": 4200.0,
                    "proposer": "player_1",
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_open_hot_posted_takes_offer_beating_next_even():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 0.8,
                "complete_information": False,
                "horizon_known": False,
                "round": 3,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 62, "player_2_gain": 38},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 48, "player_2_gain": 52},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "player_1_gain": 59,
                    "player_2_gain": 41,
                    "proposer": "player_1",
                    "round": 3,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_open_hot_stable_boundary_uses_response_value_model():
    opponent_shares = {1: 16, 3: 16, 5: 17}
    history = []
    for round_n in range(1, 7):
        ours = round_n % 2 == 0
        opponent_share = opponent_shares.get(round_n, 0)
        history.append(
            {
                "round": round_n,
                "proposer": "player_2" if ours else "player_1",
                "offer": {
                    "player_1_gain": 50 if ours else 100 - opponent_share,
                    "player_2_gain": 50 if ours else opponent_share,
                    "proposer": "player_2" if ours else "player_1",
                    "round": round_n,
                },
                "decision": "reject",
            }
        )
    game = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_1": 0.95,
            "delta_2": 0.8,
            "complete_information": True,
            "horizon_known": False,
            "round": 7,
            "history": history,
            "last_offer": {
                "player_1_gain": 83,
                "player_2_gain": 17,
                "proposer": "player_1",
                "round": 7,
            },
        },
    }
    assert strategy(game) == {"decision": "accept"}

    slower_discount = dict(game)
    slower_discount["game_state"] = dict(game["game_state"], delta_2=0.9)
    assert strategy(slower_discount) == {"decision": "reject"}


def test_bargaining_cross_validated_open_response_policy_accepts_38_not_5():
    base = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_1": 1.0,
            "delta_2": 0.9,
            "complete_information": True,
            "horizon_known": False,
            "messages_allowed": False,
            "round": 1,
            "last_offer": {
                "player_1_gain": 62,
                "player_2_gain": 38,
                "proposer": "player_1",
                "round": 1,
            },
        },
    }
    assert strategy(base) == {"decision": "accept"}

    five = dict(base)
    five["game_state"] = dict(
        base["game_state"],
        last_offer={
            "player_1_gain": 95,
            "player_2_gain": 5,
            "proposer": "player_1",
            "round": 1,
        },
    )
    assert strategy(five) == {"decision": "reject"}


def test_bargaining_hidden_open_policy_with_external_regressions_is_excluded():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": False,
                "messages_allowed": False,
                "round": 1,
                "last_offer": {
                    "player_1_gain": 55,
                    "player_2_gain": 45,
                    "proposer": "player_1",
                    "round": 1,
                },
            },
        }
    )
    # The matching fitted policy improved seven external games but regressed
    # three; deployment requires zero external regressions.
    assert action == {"decision": "reject"}


def test_bargaining_equal_discount_opens_at_symmetric_split():
    for delta in (0.9, 0.95):
        action = strategy(
            {
                "game_family": "bargaining",
                "your_player": "player_1",
                "valid_actions": {"type": "offer"},
                "game_state": {
                    "current_player": "player_1",
                    "money_to_divide": 100,
                    "delta_1": delta,
                    "delta_2": delta,
                    "complete_information": True,
                    "horizon_known": False,
                    "round": 1,
                },
            }
        )
        assert action == {"alice_gain": 50.0, "bob_gain": 50.0}


def test_bargaining_response_value_model_requires_observed_plateau():
    opponent_shares = {1: 0.12, 3: 0.14, 5: 0.16}
    history = []
    for round_n in range(1, 7):
        ours = round_n % 2 == 0
        share = 0.50 if ours else opponent_shares[round_n]
        history.append(
            {
                "round": round_n,
                "proposer": "player_2" if ours else "player_1",
                "offer": {
                    "player_1_gain": 100 * (1 - share),
                    "player_2_gain": 100 * share,
                    "proposer": "player_2" if ours else "player_1",
                    "round": round_n,
                },
                "decision": "reject",
            }
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": False,
                "round": 7,
                "history": history,
                "last_offer": {
                    "player_1_gain": 82,
                    "player_2_gain": 18,
                    "proposer": "player_1",
                    "round": 7,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_patient_searches_after_rejected_sixty_percent():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_1": 0.9,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 4,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5263.16, "player_2_gain": 4736.84},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 4000, "player_2_gain": 6000},
                        "decision": "reject",
                    },
                    {
                        "round": 3,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5263.16, "player_2_gain": 4736.84},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "player_1_gain": 5263.16,
                    "player_2_gain": 4736.84,
                    "proposer": "player_1",
                    "round": 3,
                },
            },
        }
    )
    assert action["bob_gain"] == 6000


def test_bargaining_discounted_player_rejects_patient_lowball_after_rejected_offer():
    for delta, share in ((0.9, 0.10), (0.95, 0.1589)):
        action = strategy(
            {
                "game_family": "bargaining",
                "your_player": "player_1",
                "valid_actions": {"type": "decision"},
                "game_state": {
                    "current_player": "player_1",
                    "money_to_divide": 10_000,
                    "delta_1": delta,
                    "delta_2": 1.0,
                    "complete_information": True,
                    "horizon_known": False,
                    "round": 2,
                    "history": [
                        {
                            "round": 1,
                            "proposer": "player_1",
                            "offer": {"player_1_gain": 5000, "player_2_gain": 5000},
                            "decision": "reject",
                        }
                    ],
                    "last_offer": {
                        "round": 2,
                        "proposer": "player_2",
                        "player_1_gain": 10_000 * share,
                        "player_2_gain": 10_000 * (1 - share),
                    },
                },
            }
        )
        assert action == {"decision": "reject"}


def test_bargaining_discounted_responder_rejects_opening_patient_lowball():
    for delta, share in ((0.9, 0.10), (0.95, 0.25)):
        action = strategy(
            {
                "game_family": "bargaining",
                "your_player": "player_2",
                "valid_actions": {"type": "decision"},
                "game_state": {
                    "current_player": "player_2",
                    "money_to_divide": 100,
                    "delta_1": 1.0,
                    "delta_2": delta,
                    "complete_information": True,
                    "horizon_known": False,
                    "round": 1,
                    "last_offer": {
                        "round": 1,
                        "proposer": "player_1",
                        "player_1_gain": 100 * (1 - share),
                        "player_2_gain": 100 * share,
                    },
                },
            }
        )
        assert action == {"decision": "reject"}


def test_bargaining_bob_rejects_bottom_field_offer_from_patient_opponent():
    for delta, share, horizon_known, decision in (
        (0.9, 0.30, False, "reject"),
        (0.9, 0.39, True, "accept"),
        (0.95, 0.35, False, "reject"),
    ):
        action = strategy(
            {
                "game_family": "bargaining",
                "your_player": "player_2",
                "valid_actions": {"type": "decision"},
                "game_state": {
                    "current_player": "player_2",
                    "money_to_divide": 100,
                    "delta_1": 1.0,
                    "delta_2": delta,
                    "complete_information": True,
                    "horizon_known": horizon_known,
                    "max_rounds": 12 if horizon_known else None,
                    "round": 1,
                    "last_offer": {
                        "round": 1,
                        "proposer": "player_1",
                        "player_1_gain": 100 * (1 - share),
                        "player_2_gain": 100 * share,
                    },
                },
            }
        )
        assert action == {"decision": decision}


def test_bargaining_hot_bob_respects_common_rating_gate():
    base = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_1": 0.9,
            "delta_2": 0.8,
            "complete_information": True,
            "horizon_known": False,
            "round": 1,
        },
    }
    for share in (0.28, 0.29, 0.30):
        game = {**base, "game_state": {**base["game_state"], "last_offer": {
            "round": 1,
            "proposer": "player_1",
            "player_1_gain": 100 * (1 - share),
            "player_2_gain": 100 * share,
        }}}
        assert strategy(game) == {"decision": "reject"}


def test_bargaining_hidden_open_hot_offer_uses_current_supported_share():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.8,
                "complete_information": False,
                "messages_allowed": False,
                "horizon_known": False,
                "round": 1,
                "history": [],
            },
        }
    )
    assert action == {"alice_gain": 52.0, "bob_gain": 48.0}


def test_bargaining_hidden_twelve_delta_nine_uses_rating_aware_close_floor():
    base = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_1",
            "money_to_divide": 100,
            "delta_1": 0.9,
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 12,
            "round": 2,
            "history": [
                {
                    "round": 1,
                    "proposer": "player_1",
                    "offer": {"player_1_gain": 50, "player_2_gain": 50},
                    "decision": "reject",
                }
            ],
        },
    }
    low = {**base, "game_state": {**base["game_state"], "last_offer": {
        "round": 2,
        "proposer": "player_2",
        "player_1_gain": 22,
        "player_2_gain": 78,
    }}}
    close = {**base, "game_state": {**base["game_state"], "last_offer": {
        "round": 2,
        "proposer": "player_2",
        "player_1_gain": 39,
        "player_2_gain": 61,
    }}}
    assert strategy(low) == {"decision": "reject"}
    assert strategy(close) == {"decision": "accept"}


def test_bargaining_nonterminal_bottom_field_offer_cannot_be_accepted():
    for gain in (0.00001, 10.0):
        action = strategy(
            {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.9,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "round": 2,
                    "proposer": "player_2",
                    "player_1_gain": gain,
                    "player_2_gain": 100 - gain,
                },
            },
            }
        )
        assert action == {"decision": "reject"}


def test_bargaining_bottom_field_guard_covers_all_context_dimensions():
    for you in ("player_1", "player_2"):
        for delta in (0.8, 0.9, 0.95, 1.0):
            for complete in (False, True):
                for horizon_known in (False, True):
                    for messages in (False, True):
                        opponent = "player_2" if you == "player_1" else "player_1"
                        round_n = 2 if you == "player_1" else 1
                        state = {
                            "current_player": you,
                            "money_to_divide": 100,
                            f"delta_{you[-1]}": delta,
                            "complete_information": complete,
                            "horizon_known": horizon_known,
                            "max_rounds": 12 if horizon_known else None,
                            "messages_allowed": messages,
                            "round": round_n,
                            "history": [],
                            "last_offer": {
                                "round": round_n,
                                "proposer": opponent,
                                "player_1_gain": 10 if you == "player_1" else 90,
                                "player_2_gain": 10 if you == "player_2" else 90,
                            },
                        }
                        if complete:
                            state[f"delta_{opponent[-1]}"] = 1.0
                        if round_n == 2:
                            state["history"] = [
                                {
                                    "round": 1,
                                    "proposer": you,
                                    "offer": {
                                        "player_1_gain": 50,
                                        "player_2_gain": 50,
                                    },
                                    "decision": "reject",
                                }
                            ]
                        assert strategy(
                            {
                                "game_family": "bargaining",
                                "your_player": you,
                                "valid_actions": {"type": "decision"},
                                "game_state": state,
                            }
                        ) == {"decision": "reject"}


def test_bargaining_announced_lowball_cannot_override_rating_gate():
    base = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_2": 0.9,
            "complete_information": False,
            "horizon_known": False,
            "round": 1,
            "last_offer": {
                "round": 1,
                "proposer": "player_1",
                "player_1_gain": 79,
                "player_2_gain": 21,
                "message": "Take 21 and I keep 79 -- final offer, or we both get zero.",
            },
        },
    }
    assert strategy(base) == {"decision": "reject"}

    no_signal = dict(base)
    no_signal["game_state"] = dict(
        base["game_state"],
        last_offer={**base["game_state"]["last_offer"], "message": "Take 21."},
    )
    assert strategy(no_signal) == {"decision": "reject"}

    close_now = dict(base)
    close_now["game_state"] = dict(
        base["game_state"],
        delta_2=0.8,
        last_offer={
            **base["game_state"]["last_offer"],
            "player_1_gain": 60.1,
            "player_2_gain": 39.9,
            "message": "Proposing 39.9 to you. I am happy to close on this now.",
        },
    )
    assert strategy(close_now) == {"decision": "reject"}


def test_bargaining_3eed80f4_take_positive_rejects_21_early():
    # 3eed80f4: Bob lowballer 21% r1 take_positive accept.
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10000,
                "delta_1": 0.95,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "last_offer": {
                    "player_1_gain": 7872.88,
                    "player_2_gain": 2127.12,
                    "proposer": "player_1",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_powerless_t12_rating_vetoes_27_and_22():
    base = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_1": 1.0,
            "delta_2": 0.8,
            "complete_information": True,
            "horizon_known": True,
            "max_rounds": 12,
            "round": 1,
            "messages_allowed": True,
            "last_offer": {
                "player_1_gain": 73,
                "player_2_gain": 27,
                "proposer": "player_1",
            },
        },
    }
    assert strategy(base) == {"decision": "reject"}

    below = dict(base)
    below["game_state"] = dict(
        base["game_state"],
        last_offer={
            "player_1_gain": 78,
            "player_2_gain": 22,
            "proposer": "player_1",
        },
    )
    assert strategy(below) == {"decision": "reject"}


def test_bargaining_00f0f2d9_hot_posted_takes_43_early():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.8,
                "complete_information": False,
                "horizon_known": False,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "player_1_gain": 43,
                    "player_2_gain": 57,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_04b30fce_known_hot_posted_takes_behavior_weighted_39():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.8,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 5_200, "player_2_gain": 4_800},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "player_1_gain": 3_900,
                    "player_2_gain": 6_100,
                    "proposer": "player_2",
                    "round": 2,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hot_posted_rating_vetoes_bottom_field_stop():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 0.8,
                "delta_2": 0.9,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 8,
                "history": [
                    {
                        "round": round_n,
                        "proposer": "player_1" if round_n % 2 else "player_2",
                        "offer": {
                            "player_1_gain": 500_000 if round_n % 2 else 230_431,
                            "player_2_gain": 500_000 if round_n % 2 else 769_569,
                        },
                        "decision": "reject",
                    }
                    for round_n in range(1, 8)
                ],
                "last_offer": {
                    "player_1_gain": 230_431,
                    "player_2_gain": 769_569,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_open_horizon_round_99_is_not_a_deadline():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 1_000_000,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 99,
                "last_offer": {
                    "player_1_gain": 337_000,
                    "player_2_gain": 663_000,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_patient_open_horizon_keeps_advantage_after_round_20():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 21,
                "last_offer": {
                    "player_1_gain": 5_000,
                    "player_2_gain": 5_000,
                    "proposer": "player_1",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_e0d63d63_hidden_t12_opening_requires_47_percent():
    base = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_2": 0.9,
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 12,
            "round": 1,
            "messages_allowed": False,
            "last_offer": {
                "player_1_gain": 54,
                "player_2_gain": 46,
                "proposer": "player_1",
            },
        },
    }
    assert strategy(base) == {"decision": "reject"}
    base["game_state"]["messages_allowed"] = True
    assert strategy(base) == {"decision": "reject"}
    base["game_state"]["last_offer"] = {
        "player_1_gain": 52.63,
        "player_2_gain": 47.37,
        "proposer": "player_1",
    }
    assert strategy(base) == {"decision": "accept"}
    base["game_state"]["messages_allowed"] = False
    assert strategy(base) == {"decision": "accept"}


def test_bargaining_hidden_t12_delta_095_rejects_opening_38_percent():
    game = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision", "fields": {"decision": {}}},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 10_000,
            "delta_2": 0.95,
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 12,
            "round": 1,
            "messages_allowed": True,
            "last_offer": {
                "player_1_gain": 6_200,
                "player_2_gain": 3_800,
                "message": "settle now",
                "proposer": "player_1",
            },
        },
    }
    assert strategy(game) == {"decision": "reject"}


def test_bargaining_0a0ee944_hidden_t12_floor_precedes_model_accept():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 3,
                "messages_allowed": True,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 6_360, "player_2_gain": 3_640}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 5_000, "player_2_gain": 5_000}, "decision": "reject"},
                ],
                "last_offer": {
                    "round": 3,
                    "proposer": "player_1",
                    "player_1_gain": 6_232.72727273,
                    "player_2_gain": 3_767.27272727,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_e2c0cc7d_hidden_t12_late_value_exception():
    history = []
    opponent_shares = (39.0909, 41.2727, 43.4545)
    for index, opponent_share in enumerate(opponent_shares):
        own_round = index * 2 + 1
        history.extend(
            [
                {"round": own_round, "proposer": "player_1", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
                {"round": own_round + 1, "proposer": "player_2", "offer": {"player_1_gain": opponent_share, "player_2_gain": 100 - opponent_share}, "decision": "reject"},
            ]
        )
    history.append(
        {"round": 7, "proposer": "player_1", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"}
    )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 8,
                "history": history,
                "last_offer": {
                    "round": 8,
                    "proposer": "player_2",
                    "player_1_gain": 45.6364,
                    "player_2_gain": 54.3636,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_80db7561_penultimate_floor_precedes_fitted_accept():
    history = []
    opponent_shares = (45, 40, 38, 25, 30)
    for index, opponent_share in enumerate(opponent_shares):
        opponent_round = index * 2 + 1
        history.extend(
            [
                {"round": opponent_round, "proposer": "player_1", "offer": {"player_1_gain": 100 - opponent_share, "player_2_gain": opponent_share}, "decision": "reject"},
                {"round": opponent_round + 1, "proposer": "player_2", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
            ]
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 11,
                "history": history,
                "last_offer": {
                    "round": 11,
                    "proposer": "player_1",
                    "player_1_gain": 70,
                    "player_2_gain": 30,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_56c80261_patient_bob_preserves_final_claim():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 0.9,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 11,
                "last_offer": {
                    "round": 11,
                    "proposer": "player_1",
                    "player_1_gain": 50,
                    "player_2_gain": 50,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_powerless_t12_holds_repeated_extreme_undercut():
    state = {
        "current_player": "player_1",
        "money_to_divide": 100,
        "delta_1": 0.8,
        "delta_2": 1.0,
        "complete_information": True,
        "horizon_known": True,
        "max_rounds": 12,
        "round": 2,
        "messages_allowed": False,
        "history": [
            {
                "round": 1,
                "proposer": "player_1",
                "offer": {"player_1_gain": 40, "player_2_gain": 60},
                "decision": "reject",
            }
        ],
        "last_offer": {
            "round": 2,
            "proposer": "player_2",
            "player_1_gain": 31.36,
            "player_2_gain": 68.64,
        },
    }
    game = {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": state,
    }
    assert strategy(game) == {"decision": "reject"}

    state["round"] = 3
    state["current_player"] = "player_1"
    state["history"] += [
        {
            "round": 2,
            "proposer": "player_2",
            "offer": {"player_1_gain": 31.36, "player_2_gain": 68.64},
            "decision": "reject",
        }
    ]
    state["last_offer"] = None
    game["valid_actions"] = {
        "type": "offer",
        "fields": {"alice_gain": {}, "bob_gain": {}},
    }
    assert strategy(game) == {"alice_gain": 40.0, "bob_gain": 60.0}

    state["round"] = 4
    state["history"] += [
        {
            "round": 3,
            "proposer": "player_1",
            "offer": {"player_1_gain": 40, "player_2_gain": 60},
            "decision": "reject",
        },
    ]
    state["last_offer"] = {
        "round": 4,
        "proposer": "player_2",
        "player_1_gain": 31.36,
        "player_2_gain": 68.64,
    }
    game["valid_actions"] = {"type": "decision"}
    assert strategy(game) == {"decision": "reject"}


def test_bargaining_1f2530b7_hidden_hot_opening_requires_45_percent():
    base = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_2": 0.8,
            "complete_information": False,
            "horizon_known": False,
            "round": 1,
            "messages_allowed": False,
            "last_offer": {
                "player_1_gain": 55.56,
                "player_2_gain": 44.44,
                "proposer": "player_1",
            },
        },
    }
    assert strategy(base) == {"decision": "reject"}
    base["game_state"]["messages_allowed"] = True
    assert strategy(base) == {"decision": "reject"}
    base["game_state"]["last_offer"] = {
        "player_1_gain": 55,
        "player_2_gain": 45,
        "proposer": "player_1",
    }
    assert strategy(base) == {"decision": "accept"}
    base["game_state"]["messages_allowed"] = False
    assert strategy(base) == {"decision": "accept"}


def test_bargaining_4cb1aa6a_hidden_hot_alice_opens_52_with_messages():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {
                "type": "offer",
                "fields": {"alice_gain": {}, "bob_gain": {}, "message": {}},
            },
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.8,
                "complete_information": False,
                "horizon_known": False,
                "messages_allowed": True,
                "round": 1,
            },
        }
    )
    assert action["alice_gain"] == 5_200
    assert action["bob_gain"] == 4_800


def test_bargaining_419791e6_hidden_t12_bob_counters_52():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_2": 0.9,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "messages_allowed": False,
                "round": 2,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {
                            "player_1_gain": 600_000,
                            "player_2_gain": 400_000,
                        },
                        "decision": "reject",
                    }
                ],
            },
        }
    )
    assert action == {"alice_gain": 480_000.0, "bob_gain": 520_000.0}


def test_bargaining_1789e730_holds_first_fresh_half_concession():
    history = []
    for round_n in range(1, 15):
        mine = round_n % 2 == 0
        history.append(
            {
                "round": round_n,
                "proposer": "player_2" if mine else "player_1",
                "offer": (
                    {"player_1_gain": 4_500, "player_2_gain": 5_500}
                    if mine
                    else {"player_1_gain": 7_000, "player_2_gain": 3_000}
                ),
                "decision": "reject",
            }
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_2": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 15,
                "history": history,
                "last_offer": {
                    "round": 15,
                    "proposer": "player_1",
                    "player_1_gain": 5_000,
                    "player_2_gain": 5_000,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_a8a4ba3b_holds_moving_patient_boundary():
    opponent_shares = [0.39, 0.397684, 0.410335, 0.422435]
    history = []
    for index, share in enumerate(opponent_shares):
        odd_round = index * 2 + 1
        history.extend(
            [
                {
                    "round": odd_round,
                    "proposer": "player_1",
                    "offer": {
                        "player_1_gain": 10_000 * (1 - share),
                        "player_2_gain": 10_000 * share,
                    },
                    "decision": "reject",
                },
                {
                    "round": odd_round + 1,
                    "proposer": "player_2",
                    "offer": {"player_1_gain": 4_500, "player_2_gain": 5_500},
                    "decision": "reject",
                },
            ]
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_2": 1.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 9,
                "history": history,
                "last_offer": {
                    "round": 9,
                    "proposer": "player_1",
                    "player_1_gain": 5_650,
                    "player_2_gain": 4_350,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_4acdefb0_rejects_severe_field_underperformance():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 0.8,
                "delta_2": 0.8,
                "complete_information": True,
                "messages_allowed": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 3,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 62.4, "player_2_gain": 37.6},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 48, "player_2_gain": 52},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "round": 3,
                    "proposer": "player_1",
                    "player_1_gain": 61.12727273,
                    "player_2_gain": 38.87272727,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_a7020739_rejects_less_patient_delay_offer():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_1": 0.8,
                "delta_2": 0.9,
                "complete_information": True,
                "messages_allowed": True,
                "horizon_known": False,
                "round": 5,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 604_000, "player_2_gain": 396_000}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 450_000, "player_2_gain": 550_000}, "decision": "reject"},
                    {"round": 3, "proposer": "player_1", "offer": {"player_1_gain": 588_208, "player_2_gain": 411_792}, "decision": "reject"},
                    {"round": 4, "proposer": "player_2", "offer": {"player_1_gain": 450_000, "player_2_gain": 550_000}, "decision": "reject"},
                ],
                "last_offer": {
                    "round": 5,
                    "proposer": "player_1",
                    "player_1_gain": 575_978.6752,
                    "player_2_gain": 424_021.3248,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_97081023_stops_slow_concession_discount_loss():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_1": 0.9,
                "delta_2": 0.9,
                "complete_information": True,
                "messages_allowed": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 3,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 632_000, "player_2_gain": 368_000}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 476_076.56, "player_2_gain": 523_923.44}, "decision": "reject"},
                ],
                "last_offer": {
                    "round": 3,
                    "proposer": "player_1",
                    "player_1_gain": 619_272.72727273,
                    "player_2_gain": 380_727.27272727,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_4dbbc593_stops_hidden_slow_concession_loss():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_2": 0.8,
                "complete_information": False,
                "messages_allowed": True,
                "horizon_known": False,
                "round": 3,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 619_970, "player_2_gain": 380_030}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 480_000, "player_2_gain": 520_000}, "decision": "reject"},
                ],
                "last_offer": {
                    "round": 3,
                    "proposer": "player_1",
                    "player_1_gain": 618_819,
                    "player_2_gain": 381_181,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_e3d0981e_accepts_positive_terminal_offer():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.95,
                "delta_2": 0.9,
                "complete_information": True,
                "messages_allowed": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 12,
                "last_offer": {
                    "round": 12,
                    "proposer": "player_2",
                    "player_1_gain": 1_795,
                    "player_2_gain": 8_205,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_99a70288_ignores_stale_jump_in_continuation():
    history = [
        {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 750_000, "player_2_gain": 250_000}, "decision": "reject"},
        {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 500_000, "player_2_gain": 500_000}, "decision": "reject"},
        {"round": 3, "proposer": "player_1", "offer": {"player_1_gain": 600_000, "player_2_gain": 400_000}, "decision": "reject"},
        {"round": 4, "proposer": "player_2", "offer": {"player_1_gain": 500_000, "player_2_gain": 500_000}, "decision": "reject"},
        {"round": 5, "proposer": "player_1", "offer": {"player_1_gain": 620_000, "player_2_gain": 380_000}, "decision": "reject"},
        {"round": 6, "proposer": "player_2", "offer": {"player_1_gain": 500_000, "player_2_gain": 500_000}, "decision": "reject"},
    ]
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 1_000_000,
                "delta_2": 0.9,
                "complete_information": False,
                "horizon_known": False,
                "round": 7,
                "history": history,
                "last_offer": {"round": 7, "proposer": "player_1", "player_1_gain": 600_000, "player_2_gain": 400_000},
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_ae7dcce3_rejects_extreme_relative_loss():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 10_000,
                "delta_1": 0.8,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": False,
                "round": 4,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 5_200, "player_2_gain": 4_800}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 1_705, "player_2_gain": 8_295}, "decision": "reject"},
                    {"round": 3, "proposer": "player_1", "offer": {"player_1_gain": 5_000, "player_2_gain": 5_000}, "decision": "reject"},
                ],
                "last_offer": {"round": 4, "proposer": "player_2", "player_1_gain": 2_105, "player_2_gain": 7_895},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_cb50bfe3_rejects_fully_patient_early_lowball():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000,
                "delta_1": 1.0,
                "delta_2": 0.9,
                "complete_information": True,
                "horizon_known": False,
                "round": 3,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 8_500, "player_2_gain": 1_500}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 5_000, "player_2_gain": 5_000}, "decision": "reject"},
                ],
                "last_offer": {"round": 3, "proposer": "player_1", "player_1_gain": 8_247, "player_2_gain": 1_753},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_6923e5b1_rejects_when_discount_makes_40_extreme():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.8,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": False,
                "round": 4,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 52, "player_2_gain": 48}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 35, "player_2_gain": 65}, "decision": "reject"},
                    {"round": 3, "proposer": "player_1", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
                ],
                "last_offer": {"round": 4, "proposer": "player_2", "player_1_gain": 40, "player_2_gain": 60},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_4d74fb7c_rejects_repeated_hidden_extreme_lowball():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision", "fields": {"decision": {}}},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 5,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 79, "player_2_gain": 21}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
                    {"round": 3, "proposer": "player_1", "offer": {"player_1_gain": 77, "player_2_gain": 23}, "decision": "reject"},
                    {"round": 4, "proposer": "player_2", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
                ],
                "last_offer": {"round": 5, "proposer": "player_1", "player_1_gain": 75, "player_2_gain": 25},
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_84e33788_hot_posted_takes_49_late():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "delta_2": 0.8,
                "complete_information": True,
                "horizon_known": False,
                "round": 13,
                "history": [
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "player_1_gain": 51,
                    "player_2_gain": 49,
                    "proposer": "player_1",
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_hot_posted_still_rejects_15_percent():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 0.8,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 4,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 50, "player_2_gain": 50},
                        "decision": "reject",
                    }
                ],
                "last_offer": {
                    "player_1_gain": 15,
                    "player_2_gain": 85,
                    "proposer": "player_2",
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_negotiation_72c297b9_strong_seller_does_not_fade_to_cost():
    # 72c297b9: 0.8× seller Faratin to 80 they took, profit 0.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer", "fields": {"product_price": {}}},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 80.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 11,
                "last_offer": {"price": 70.0, "from_player": "player_2"},
                "history": [
                    {"round": 1, "offer": {"price": 120.0, "from_player": "player_1"}},
                    {"round": 2, "offer": {"price": 70.0, "from_player": "player_2"}},
                ],
            },
        }
    )
    assert action["product_price"] >= 96.0


def test_negotiation_open_strong_seller_probes_hidden_buyer_type():
    base = {
        "game_family": "negotiation",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_1",
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_1_value": 80.0,
            "complete_information": False,
            "horizon_known": False,
            "round": 2,
            "history": [
                {
                    "round": 1,
                    "offer": {"price": 120.0, "from_player": "player_1"},
                    "decision": "RejectOffer",
                }
            ],
            "last_offer": {"price": 80.8, "from_player": "player_2", "round": 2},
        },
    }
    assert strategy(base) == {"decision": "RejectOffer", "product_price": 103.0}

    middle = dict(base)
    middle["game_state"] = dict(
        base["game_state"],
        last_offer={"price": 100.0, "from_player": "player_2", "round": 2},
    )
    assert strategy(middle) == {"decision": "RejectOffer", "product_price": 103.0}

    second = dict(base)
    second["game_state"] = dict(
        base["game_state"],
        round=4,
        history=[
            *base["game_state"]["history"],
            {
                "round": 2,
                "offer": {"price": 80.8, "from_player": "player_2"},
                "decision": "RejectOffer",
            },
            {
                "round": 3,
                "offer": {"price": 103.0, "from_player": "player_1"},
                "decision": "RejectOffer",
            },
        ],
        last_offer={"price": 81.04, "from_player": "player_2", "round": 4},
    )
    assert strategy(second) == {"decision": "RejectOffer", "product_price": 103.0}

    third = dict(base)
    third["game_state"] = dict(
        second["game_state"],
        round=6,
        history=[
            *second["game_state"]["history"],
            {
                "round": 4,
                "offer": {"price": 81.04, "from_player": "player_2"},
                "decision": "RejectOffer",
            },
            {
                "round": 5,
                "offer": {"price": 103.0, "from_player": "player_1"},
                "decision": "RejectOffer",
            },
        ],
        last_offer={"price": 81.28, "from_player": "player_2", "round": 6},
    )
    # The open-horizon public seller payoff is 0.20x base: 80 + 20 = 100.
    assert strategy(third) == {"decision": "RejectOffer", "product_price": 100.0}


def test_negotiation_8bf62f78_strong_seller_floor_120():
    # 8bf62f78: cost 800k faded to cost. Public payoff median supports 960k.
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer", "fields": {"product_price": {}}},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 800000.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 29,
            },
        }
    )
    assert action["product_price"] >= 960000.0


def test_negotiation_strong_seller_acceptance_uses_public_payoff_floor():
    base = {
        "game_family": "negotiation",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_1",
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_1_value": 800000.0,
            "complete_information": False,
            "horizon_known": True,
            "round": 4,
            "max_rounds": 10,
        },
    }
    low = dict(base)
    low["game_state"] = dict(
        base["game_state"],
        last_offer={"price": 921666.67, "from_player": "player_2"},
    )
    enough = dict(base)
    enough["game_state"] = dict(
        base["game_state"],
        last_offer={"price": 984000.0, "from_player": "player_2"},
    )
    assert strategy(low)["decision"] == "RejectOffer"
    assert strategy(enough) == {"decision": "AcceptOffer"}


def test_negotiation_clod_cost_plateau_rejects_terminal_crumb():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 800_000.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 10,
                "max_rounds": 10,
                "history": [
                    {
                        "round": round_no,
                        "offer": {
                            "price": price,
                            "from_player": "player_2",
                            "round": round_no,
                        },
                        "decision": "RejectOffer",
                    }
                    for round_no, price in (
                        (2, 805_234.77),
                        (4, 805_235.03),
                        (6, 805_261.39),
                        (8, 805_637.29),
                    )
                ],
                "last_offer": {
                    "price": 808_000.0,
                    "from_player": "player_2",
                    "round": 10,
                },
            },
        }
    )
    assert action == {"decision": "RejectOffer"}


def test_negotiation_strong_buyer_bids_below_acceptance_ceiling():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "complete_information": False,
                "horizon_known": True,
                "round": 2,
                "max_rounds": 10,
                "history": [
                    {"round": 1, "offer": {"price": 147.0, "from_player": "player_1"}}
                ],
                "last_offer": {"price": 147.0, "from_player": "player_1", "round": 1},
            },
        }
    )
    assert 112.5 <= action["product_price"] < 135.0


def test_negotiation_ed00091e_holds_bid_while_seller_asks_are_infeasible():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 15_000.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 3,
                "history": [
                    {"round": 1, "offer": {"price": 31_764.8, "from_player": "player_1"}},
                    {"round": 2, "offer": {"price": 12_000.0, "from_player": "player_2"}},
                ],
                "last_offer": {
                    "price": 31_059.2,
                    "from_player": "player_1",
                    "round": 3,
                },
            },
        }
    )
    assert action == {"decision": "RejectOffer", "product_price": 12_000.0}


def test_negotiation_28de935a_live_seller_keeps_close_120_floor():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 120,
                "complete_information": False,
                "horizon_known": False,
                "round": 7,
                "history": [
                    {"round": 2, "offer": {"price": 23.75, "from_player": "player_2"}},
                    {"round": 4, "offer": {"price": 33.75, "from_player": "player_2"}},
                    {"round": 6, "offer": {"price": 103, "from_player": "player_2"}},
                ],
                "last_offer": {"price": 103, "from_player": "player_2", "round": 6},
            },
        }
    )
    assert action["product_price"] >= 144


def test_negotiation_known_zopa_stall_counters_even_before_hidden_deadline():
    history = []
    for round_n in range(1, 51):
        if round_n % 2:
            price, player = 119.99, "player_1"
        else:
            price, player = 108.0, "player_2"
        history.append(
            {"round": round_n, "offer": {"price": price, "from_player": player}}
        )
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "player_2_value": 120.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 51,
                "history": history,
                "last_offer": {
                    "price": 119.99,
                    "from_player": "player_1",
                    "round": 51,
                },
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] == 110.0


def test_negotiation_known_zopa_stall_keeps_round_95_profit_fallback():
    history = []
    for round_n in range(89, 95):
        price = 119.99 if round_n % 2 else 110.0
        player = "player_1" if round_n % 2 else "player_2"
        history.append(
            {"round": round_n, "offer": {"price": price, "from_player": player}}
        )
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "player_2_value": 120.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 95,
                "history": history,
                "last_offer": {
                    "price": 119.99,
                    "from_player": "player_1",
                    "round": 95,
                },
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_negotiation_f8f129df_accepts_before_countering_worse():
    history = []
    for round_n in range(61, 72):
        seller = round_n % 2 == 1
        history.append(
            {
                "round": round_n,
                "offer": {
                    "price": 1_010_000.0 if seller else 1_001_600.0,
                    "from_player": "player_1" if seller else "player_2",
                    "round": round_n,
                },
                "decision": "RejectOffer",
            }
        )
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 800_000.0,
                "player_2_value": 1_500_000.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 72,
                "history": history,
                "last_offer": {
                    "price": 1_001_600.0,
                    "from_player": "player_2",
                    "round": 72,
                },
            },
        }
    )
    assert action == {"decision": "AcceptOffer"}


def test_negotiation_known_zopa_stall_holds_even_after_round_50():
    history = []
    for round_n in range(43, 55):
        seller = round_n % 2 == 1
        history.append(
            {
                "round": round_n,
                "offer": {
                    "price": 147.75 if seller else 135.0,
                    "from_player": "player_1" if seller else "player_2",
                    "round": round_n,
                },
                "decision": "RejectOffer",
            }
        )
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 120.0,
                "player_2_value": 150.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 55,
                "history": history,
                "last_offer": {
                    "price": 147.75,
                    "from_player": "player_1",
                    "round": 55,
                },
            },
        }
    )
    assert action == {
        "decision": "RejectOffer",
        "product_price": 135.0,
    }


def test_negotiation_c1931e32_open_horizon_live_seller_holds_120():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 5,
                "history": [
                    {"round": 1, "offer": {"price": 150.0, "from_player": "player_1"}},
                    {"round": 2, "offer": {"price": 87.5, "from_player": "player_2"}},
                    {"round": 3, "offer": {"price": 129.23, "from_player": "player_1"}},
                    {"round": 4, "offer": {"price": 97.5, "from_player": "player_2"}},
                ],
                "last_offer": {"price": 97.5, "from_player": "player_2", "round": 4},
            },
        }
    )
    assert action["product_price"] >= 120.0


def test_negotiation_59073e28_live_seller_relaxes_on_penultimate_round():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 12000,
                "complete_information": False,
                "horizon_known": True,
                "round": 9,
                "max_rounds": 10,
                "history": [
                    {"round": 2, "offer": {"price": 11812.5, "from_player": "player_2"}},
                    {"round": 4, "offer": {"price": 11812.5, "from_player": "player_2"}},
                    {"round": 6, "offer": {"price": 11812.5, "from_player": "player_2"}},
                    {"round": 8, "offer": {"price": 11812.5, "from_player": "player_2"}},
                ],
                "last_offer": {"price": 11812.5, "from_player": "player_2", "round": 8},
            },
        }
    )
    assert 12000 < action["product_price"] < 13200


def test_negotiation_open_horizon_stall_waits_while_time_is_available():
    history = []
    for round_n, price in ((2, 127.5), (4, 130.5), (6, 130.5), (8, 130.5)):
        history.append(
            {"round": round_n, "offer": {"price": price, "from_player": "player_2"}}
        )
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 120,
                "player_2_value": 150,
                "complete_information": True,
                "horizon_known": False,
                "round": 10,
                "history": history,
                "last_offer": {"price": 130.5, "from_player": "player_2", "round": 10},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_7ad8073d_counters_long_open_horizon_stall_at_even_split():
    history = [
        {"round": round_n, "offer": {"price": 130.5, "from_player": "player_2"}}
        for round_n in (42, 44, 46, 48)
    ]
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 120,
                "player_2_value": 150,
                "complete_information": True,
                "horizon_known": False,
                "round": 50,
                "history": history,
                "last_offer": {"price": 130.5, "from_player": "player_2", "round": 50},
            },
        }
    )
    assert action["decision"] == "RejectOffer"
    assert action["product_price"] == 135.0


def test_negotiation_f35418ae_counters_stall_at_nash_price():
    history = []
    for round_n in range(1, 7):
        seller = round_n % 2 == 1
        history.append(
            {
                "round": round_n,
                "offer": {
                    "price": 11_000.0 if seller else 9_200.0,
                    "from_player": "player_1" if seller else "player_2",
                    "round": round_n,
                },
                "decision": "RejectOffer",
            }
        )
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 8_000.0,
                "player_2_value": 12_000.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 7,
                "history": history,
                "last_offer": {"price": 11_000.0, "from_player": "player_1", "round": 7},
            },
        }
    )
    assert action == {"decision": "RejectOffer", "product_price": 10_000.0}
def test_negotiation_hidden_platform_cap_accepts_positive_buyer_offer():
    assert strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 99,
                "last_offer": {"price": 148.0, "from_player": "player_1"},
            },
        }
    ) == {"decision": "AcceptOffer"}


def test_negotiation_hidden_platform_round_98_seller_accepts_positive_offer():
    assert strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 100.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 98,
                "last_offer": {"price": 102.0, "from_player": "player_2"},
            },
        }
    ) == {"decision": "AcceptOffer"}


def test_negotiation_open_horizon_does_not_force_early_positive_accept():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 120.0,
                "complete_information": False,
                "horizon_known": False,
                "round": 95,
                "last_offer": {"price": 110.0, "from_player": "player_1"},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_negotiation_complete_round_98_keeps_continuation_model():
    action = strategy(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 120.0,
                "player_2_value": 150.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 98,
                "last_offer": {"price": 124.0, "from_player": "player_2"},
            },
        }
    )
    assert action["decision"] == "RejectOffer"


def test_bargaining_21329ccc_accepts_repeated_even_boundary():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 1.0,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 4,
                "messages_allowed": True,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 65, "player_2_gain": 35}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
                    {"round": 3, "proposer": "player_1", "offer": {"player_1_gain": 60, "player_2_gain": 40}, "decision": "reject"},
                ],
                "last_offer": {
                    "round": 4,
                    "proposer": "player_2",
                    "player_1_gain": 50,
                    "player_2_gain": 50,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_repeated_even_boundary_does_not_override_patient_bob():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 0.95,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 3,
                "messages_allowed": True,
                "history": [
                    {"round": 1, "proposer": "player_1", "offer": {"player_1_gain": 50, "player_2_gain": 50}, "decision": "reject"},
                    {"round": 2, "proposer": "player_2", "offer": {"player_1_gain": 40, "player_2_gain": 60}, "decision": "reject"},
                ],
                "last_offer": {
                    "round": 3,
                    "proposer": "player_1",
                    "player_1_gain": 50,
                    "player_2_gain": 50,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_bargaining_hidden_delta_09_alice_opens_52_in_all_message_horizon_cells():
    for horizon_known in (False, True):
        for messages_allowed in (False, True):
            action = strategy(
                {
                    "game_family": "bargaining",
                    "your_player": "player_1",
                    "valid_actions": {"type": "offer"},
                    "game_state": {
                        "current_player": "player_1",
                        "money_to_divide": 100,
                        "delta_1": 0.9,
                        "complete_information": False,
                        "horizon_known": horizon_known,
                        "max_rounds": 12 if horizon_known else None,
                        "messages_allowed": messages_allowed,
                        "round": 1,
                    },
                }
            )
            assert action["alice_gain"] == 52
            assert action["bob_gain"] == 48


def test_bargaining_079e8a58_patient_bob_accepts_fixed_boundary():
    history = []
    for index in range(8):
        opponent_round = index * 2 + 1
        history.extend(
            [
                {"round": opponent_round, "proposer": "player_1", "offer": {"player_1_gain": 52.632, "player_2_gain": 47.368}, "decision": "reject"},
                {"round": opponent_round + 1, "proposer": "player_2", "offer": {"player_1_gain": 40, "player_2_gain": 60}, "decision": "reject"},
            ]
        )
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "delta_1": 0.9,
                "delta_2": 1.0,
                "complete_information": True,
                "horizon_known": False,
                "round": 17,
                "history": history,
                "last_offer": {
                    "round": 17,
                    "proposer": "player_1",
                    "player_1_gain": 52.632,
                    "player_2_gain": 47.368,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_bargaining_patient_hidden_plateau_requires_full_half():
    history = []
    for index in range(4):
        opponent_round = index * 2 + 1
        history.extend(
            [
                {"round": opponent_round, "proposer": "player_1", "offer": {"player_1_gain": 51, "player_2_gain": 49}, "decision": "reject"},
                {"round": opponent_round + 1, "proposer": "player_2", "offer": {"player_1_gain": 45, "player_2_gain": 55}, "decision": "reject"},
            ]
        )
    base = {
        "game_family": "bargaining",
        "your_player": "player_2",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_2",
            "money_to_divide": 100,
            "delta_2": 1.0,
            "complete_information": False,
            "horizon_known": False,
            "round": 9,
            "history": history,
            "last_offer": {
                "round": 9,
                "proposer": "player_1",
                "player_1_gain": 51,
                "player_2_gain": 49,
            },
        },
    }
    assert strategy(base) == {"decision": "reject"}
    for row in base["game_state"]["history"]:
        if row["proposer"] == "player_1":
            row["offer"] = {"player_1_gain": 50, "player_2_gain": 50}
    base["game_state"]["last_offer"] = {
        "round": 9,
        "proposer": "player_1",
        "player_1_gain": 50,
        "player_2_gain": 50,
    }
    assert strategy(base) == {"decision": "accept"}
