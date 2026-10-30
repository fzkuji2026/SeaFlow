from glee_agent.judge import _needs_judge


def _alts(*names):
    decision = {
        "accept": "AcceptOffer",
        "reject": "RejectOffer",
        "walk": "WalkAway",
        "yes": "yes",
        "no": "no",
    }
    return [
        {"name": name, "action": {"decision": decision[name]} if name in decision else {}}
        for name in names
    ]


def test_judge_only_handles_scenario_level_ambiguity():
    assert not _needs_judge(
        {"family": "bargaining", "action_type": "offer", "calc": {}},
        _alts("even", "ask_52"),
    )
    assert not _needs_judge(
        {
            "family": "bargaining",
            "action_type": "decision",
            "is_last_round": False,
            "calc": {
                "procedure": "mild",
                "last_share": 0.42,
                "discounted_last_share": 0.30,
                "expected_next_even": 0.32,
                "same_offer_repeats": 2,
            },
        },
        _alts("accept", "reject"),
    )
    assert not _needs_judge(
        {
            "family": "bargaining",
            "action_type": "decision",
            "is_last_round": False,
            "calc": {
                "procedure": "mild",
                "last_share": 0.03,
                "discounted_last_share": 0.02,
                "expected_next_even": 0.30,
                "same_offer_repeats": 5,
            },
        },
        _alts("accept", "reject"),
    )
    assert not _needs_judge(
        {
            "family": "bargaining",
            "action_type": "decision",
            "is_last_round": False,
            "calc": {
                "procedure": "hot",
                "last_share": 0.294,
                "discounted_last_share": 0.294,
                "expected_next_even": 0.34,
            },
        },
        _alts("accept", "reject"),
    )
    assert not _needs_judge(
        {
            "family": "negotiation",
            "action_type": "decision",
            "is_last_round": False,
            "calc": {"profit": -10, "planned_ask_profit": 20},
        },
        _alts("reject", "walk"),
    )
    assert not _needs_judge(
        {
            "family": "negotiation",
            "action_type": "decision",
            "is_last_round": False,
            "calc": {"profit": 30, "planned_ask_profit": 50, "surplus_share": 0.43},
        },
        _alts("accept", "reject"),
    )
    assert _needs_judge(
        {
            "family": "negotiation",
            "path": "negotiation/open_horizon/live/take_profit/decision",
            "action_type": "decision",
            "is_last_round": False,
            "calc": {
                "own_band": "live",
                "profit": 95.0,
                "planned_ask_profit": 100.0,
            },
        },
        _alts("accept", "reject"),
    )
    assert not _needs_judge(
        {
            "family": "negotiation",
            "path": "negotiation/complete_late/live/zopa_boulware/decision",
            "action_type": "decision",
            "is_last_round": False,
            "calc": {
                "own_band": "live",
                "profit": 95.0,
                "planned_ask_profit": 100.0,
            },
        },
        _alts("accept", "reject"),
    )
    assert not _needs_judge(
        {
            "family": "negotiation",
            "action_type": "decision",
            "is_last_round": False,
            "calc": {
                "own_band": "strong_seller",
                "profit": 61.7,
                "planned_ask_profit": 80.0,
            },
        },
        _alts("accept", "reject"),
    )
    assert not _needs_judge(
        {
            "family": "persuasion",
            "action_type": "buyer_decision",
            "calc": {
                "price": 100,
                "ev_minus_price": 20,
                "bought_high": 1,
                "bought_low": 1,
            },
        },
        _alts("yes", "no"),
    )


def test_seller_judge_respects_own_payoff_and_observed_response():
    base = {
        "family": "persuasion",
        "action_type": "seller_recommendation",
        "is_last_round": False,
        "calc": {"quality": "low"},
    }
    assert not _needs_judge(
        {**base, "calc": {"quality": "low", "signal_response": {"yes_n": 8, "yes_rate": 1.0, "no_n": 2, "no_rate": 0.0}}},
        _alts("yes", "no"),
    )
    assert not _needs_judge(
        {**base, "calc": {"quality": "low", "signal_response": {"yes_n": 8, "yes_rate": 0.375, "no_n": 0, "no_rate": 0.0}}},
        _alts("yes", "no"),
    )
