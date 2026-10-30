from glee_agent.rules import ActionError, looks_legal, parse, validate, zopa


def _bargain_offer(money=100, messages=True, **state):
    return {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "offer", "fields": {"alice_gain": {}, "bob_gain": {}, **({"message": {}} if messages else {})}},
        "game_state": {"money_to_divide": money, "messages_allowed": messages, "current_player": "player_1", **state},
    }


def _neg(action="decision", last=True, **state):
    game = {
        "game_family": "negotiation",
        "your_player": "player_2",
        "valid_actions": {"type": action},
        "game_state": {
            "current_player": "player_2",
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_1_value": 40,
            "player_2_value": 80,
            "complete_information": True,
            "messages_allowed": True,
            **state,
        },
    }
    if last and "last_offer" not in game["game_state"] and action == "decision":
        game["game_state"]["last_offer"] = {"price": 90}
    return game


def test_decision_schema_preserves_game_message_setting():
    game = _bargain_offer(messages=True)
    game["your_player"] = "player_2"
    game["valid_actions"] = {"type": "decision", "fields": {"decision": {}}}
    assert parse(game).messages_allowed is True


def test_parse_last_round_needs_known_horizon():
    view = parse(_bargain_offer(round=10, max_rounds=10, horizon_known=True))
    assert view.is_last_round
    view = parse(_bargain_offer(round=10, max_rounds=10, horizon_known=False))
    assert not view.is_last_round
    view = parse(_bargain_offer(round=1))
    assert view.max_rounds is None
    assert not view.horizon_known
    assert not view.is_last_round


def test_delta_from_zero_inflation():
    from glee_agent.rules import my_delta

    view = parse(_bargain_offer(inflation_1=0))
    assert my_delta(view) == 1.0
    view = parse(_bargain_offer(delta_1=0.9))
    assert my_delta(view) == 0.9


def test_bargaining_gains_must_sum_exactly():
    game = _bargain_offer()
    assert looks_legal(game, {"alice_gain": 55, "bob_gain": 45})
    assert not looks_legal(game, {"alice_gain": 55, "bob_gain": 40})


def test_message_forbidden_when_not_allowed():
    game = _bargain_offer(messages=False)
    assert looks_legal(game, {"alice_gain": 50, "bob_gain": 50})
    assert not looks_legal(game, {"alice_gain": 50, "bob_gain": 50, "message": "hi"})


def test_final_round_reject_cannot_carry_price():
    game = _neg(horizon_known=True, round=5, max_rounds=5)
    assert looks_legal(game, {"decision": "RejectOffer"})
    assert not looks_legal(game, {"decision": "RejectOffer", "product_price": 70})


def test_nonfinal_reject_requires_counteroffer():
    game = _neg(horizon_known=True, round=2, max_rounds=5)
    assert not looks_legal(game, {"decision": "RejectOffer"})
    assert looks_legal(game, {"decision": "RejectOffer", "product_price": 70})


def test_zopa_none_when_buyer_below_seller():
    game = _neg(action="offer")
    game["game_state"]["player_2_value"] = 30
    assert zopa(parse(game)) is None
    game["game_state"]["player_2_value"] = 80
    assert zopa(parse(game)) == (40.0, 80.0)


def test_validate_raises():
    try:
        validate(_bargain_offer(), {"alice_gain": 1})
    except ActionError:
        return
    raise AssertionError("expected ActionError")
