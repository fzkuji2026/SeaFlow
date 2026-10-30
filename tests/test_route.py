from glee_agent.route import REGIMES, fail_closed, handle_for
from glee_agent.rules import parse
from glee_agent.workflows.bargaining import dispatch as bargain_dispatch
from glee_agent.workflows.bargaining import play as bargain_play
from glee_agent.workflows.negotiation import dispatch as nego_dispatch
from glee_agent.workflows.persuasion import dispatch as pers_dispatch


KNOWN = {
    "bargaining": (
        "deadline",
        "penultimate",
        "impatient",
        "patient",
        "patient_vs_impatient",
        "symmetric",
    ),
    "negotiation": (
        "no_zopa",
        "ultimatum",
        "complete",
        "one_shot",
        "incomplete",
        "incomplete_deadline",
    ),
    "persuasion": (
        "seller_honest",
        "seller_last",
        "buyer_liar",
        "buyer_honest",
        "buyer_prior",
    ),
}


def test_known_regimes_map_to_handle():
    for family, regimes in KNOWN.items():
        table = REGIMES[family]
        for regime in regimes:
            fn = handle_for(family, regime)
            assert callable(fn)
            assert table[regime] is fn
            assert fn.__name__ == "handle"


def test_mistyped_regime_is_none_not_crash():
    assert handle_for("bargaining", "late_soft") is None
    assert handle_for("negotiation", "n0_zopa") is None
    assert handle_for("persuasion", "seller_lye") is None
    assert handle_for("unknown_family", "deadline") is None


def test_mistyped_regime_fails_closed():
    bargain = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {"money_to_divide": 100, "round": 2},
        }
    )
    nego = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {"player_1_role": "seller", "player_1_value": 40},
        }
    )
    pers = parse(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {"p": 0.5, "v": 10, "u": 0, "product_price": 4},
        }
    )
    assert bargain_dispatch(bargain, "late_soft") == {"decision": "reject"}
    assert nego_dispatch(nego, "not_a_regime") == {"decision": "reject"}
    assert pers_dispatch(pers, "seller_lye") == {"decision": "reject"}
    assert fail_closed(bargain) == {"decision": "reject"}


def test_bargaining_dispatch_and_live_play_use_the_same_pipeline():
    view = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "money_to_divide": 100,
                "delta_1": 0.8,
                "delta_2": 1.0,
                "complete_information": True,
                "round": 1,
            },
        }
    )
    assert bargain_dispatch(view, "powerless") == bargain_play(view)
    assert handle_for("bargaining", "powerless")(view) == bargain_play(view)
