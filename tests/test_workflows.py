from glee_agent.families.bargaining import classify as bargain_class
from glee_agent.families.negotiation import classify as nego_class
from glee_agent.families.persuasion import classify as pers_class
from glee_agent.policy import DEFAULT, set_policy
from glee_agent.rules import parse
from glee_agent.strategy import _with_opponent_identity


def test_visible_opponent_identity_is_preserved_for_workflow_state():
    game = {
        "opponent": {"type": "agent", "name": "gamma"},
        "game_state": {"round": 1},
    }
    enriched = _with_opponent_identity(game)
    assert enriched["game_state"]["opponent_name"] == "gamma"
    assert "opponent_name" not in game["game_state"]


def setup_function():
    set_policy(DEFAULT)


def test_bargain_regimes():
    deadline = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "horizon_known": True,
                "round": 3,
                "max_rounds": 3,
                "delta_1": 1.0,
                "delta_2": 0.95,
            },
        }
    )
    assert bargain_class(deadline) == "deadline"

    vs = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {"delta_1": 1.0, "delta_2": 0.95, "round": 2},
        }
    )
    assert bargain_class(vs) == "patient_vs_impatient"

    infl = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {"delta_1": 0.90, "delta_2": 0.90, "round": 4},
        }
    )
    assert bargain_class(infl) == "impatient"


def test_negotiation_regimes():
    none = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "complete_information": True,
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 90,
                "player_2_value": 40,
            },
        }
    )
    assert nego_class(none) == "no_zopa"

    ult = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "complete_information": True,
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 10,
                "player_2_value": 12,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
            },
        }
    )
    assert nego_class(ult) == "ultimatum"

    hidden = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {"player_1_role": "seller", "player_1_value": 40},
        }
    )
    assert nego_class(hidden) == "open_horizon"

    last = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "player_1_role": "seller",
                "player_1_value": 40,
                "horizon_known": True,
                "round": 5,
                "max_rounds": 5,
            },
        }
    )
    assert nego_class(last) == "incomplete_deadline"

    shot = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "player_1_role": "seller",
                "player_1_value": 800000,
                "horizon_known": True,
                "round": 1,
                "max_rounds": 1,
            },
        }
    )
    assert nego_class(shot) == "one_shot"


def test_negotiation_opponent_style_excludes_our_counteroffers():
    from glee_agent.workflows.negotiation._tactics import (
        best_offered_share,
        opponent_prices,
        same_price_repeats,
        style_of,
    )

    view = parse(
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
                "history": [
                    {"round": 1, "offer": {"price": 9800, "from_player": "player_1"}},
                    {"round": 2, "offer": {"price": 8100, "from_player": "player_2"}},
                    {"round": 3, "offer": {"price": 9800, "from_player": "player_1"}},
                    {"round": 4, "offer": {"price": 8600, "from_player": "player_2"}},
                ],
                "last_offer": {"price": 9800, "from_player": "player_1", "round": 5},
            },
        }
    )
    assert opponent_prices(view) == [9800, 9800, 9800]
    assert style_of(view, "buyer", 10000) == "stubborn"
    assert best_offered_share(view, "buyer") == 0.1
    assert same_price_repeats(view, 9800) == 3


def test_negotiation_opponent_prices_infer_sender_from_decider():
    from glee_agent.workflows.negotiation._tactics import opponent_prices, own_prices

    view = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "history": [
                    {"offer": {"price": 120}, "decided_by": "player_2"},
                    {"offer": {"price": 90}, "decided_by": "player_1"},
                ],
            },
        }
    )
    assert own_prices(view) == [120]
    assert opponent_prices(view) == [90]


def test_negotiation_style_uses_recent_stall_not_early_concession():
    from glee_agent.workflows.negotiation._tactics import same_price_repeats, state_of, style_of

    view = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_1_value": 120,
                "history": [
                    {"round": 2, "offer": {"price": 127.5, "from_player": "player_2"}},
                    {"round": 4, "offer": {"price": 130.5, "from_player": "player_2"}},
                    {"round": 6, "offer": {"price": 130.5, "from_player": "player_2"}},
                    {"round": 8, "offer": {"price": 130.5, "from_player": "player_2"}},
                ],
                "last_offer": {"price": 130.5, "from_player": "player_2", "round": 10},
            },
        }
    )
    assert style_of(view, "seller", 120) == "stubborn"
    assert same_price_repeats(view, 130.5) == 4
    opponent = state_of(view, "seller", 120)
    assert opponent.samples == 5
    assert opponent.current_price == 130.5
    assert opponent.plateau_repeats == 4
    assert opponent.predicted_next_price == 130.5
    assert opponent.stall_probability > 0.8


def test_bargaining_style_and_repeats_use_recent_opponent_offers_only():
    from glee_agent.workflows.bargaining._helpers import same_split_repeats
    from glee_agent.workflows.bargaining._opponent import state_of, style_of, their_shares

    view = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "money_to_divide": 100,
                "history": [
                    {"round": 1, "offer": {"player_1_gain": 60, "player_2_gain": 40, "proposer": "player_1"}},
                    {"round": 2, "offer": {"player_1_gain": 40, "player_2_gain": 60, "proposer": "player_2"}},
                    {"round": 3, "offer": {"player_1_gain": 50, "player_2_gain": 50, "proposer": "player_1"}},
                    {"round": 4, "offer": {"player_1_gain": 45.5, "player_2_gain": 54.5, "proposer": "player_2"}},
                    {"round": 6, "offer": {"player_1_gain": 46.2, "player_2_gain": 53.8, "proposer": "player_2"}},
                ],
                "last_offer": {"player_1_gain": 46, "player_2_gain": 54, "proposer": "player_2", "round": 8},
            },
        }
    )
    assert their_shares(view) == [0.4, 0.455, 0.462, 0.46]
    assert style_of(view) == "stubborn"
    assert same_split_repeats(view) == 3
    state = state_of(view)
    assert state.current_share == 0.46
    assert state.plateau_repeats == 3
    assert state.estimated_patience == 1.0
    assert state.predicted_next_share == 0.467


def test_persuasion_regimes():
    last = parse(
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
    assert pers_class(last) == "seller_last"

    honest = parse(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {"current_quality": "high", "round": 2},
        }
    )
    assert pers_class(honest) == "seller_honest"

    liar = parse(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "p": 0.4,
                "v": 10,
                "u": 0,
                "product_price": 6,
                "history": [
                    {"seller_message": "worth buying", "bought": True, "quality": "low"}
                ],
            },
        }
    )
    assert pers_class(liar) == "buyer_liar"

    high = parse(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {"p": 0.8, "v": 12.5, "u": 0, "product_price": 10},
        }
    )
    assert pers_class(high) == "buyer_p_high"


def test_bargain_impatient_vs_hotter_regime():
    view = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "delta_1": 0.80,
                "delta_2": 0.95,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )
    assert bargain_class(view) == "impatient_vs_hotter"


def test_negotiation_incomplete_late_regime():
    view = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "player_1_role": "seller",
                "player_1_value": 100.0,
                "horizon_known": True,
                "round": 9,
                "max_rounds": 10,
            },
        }
    )
    assert nego_class(view) == "incomplete_late"


def test_persuasion_known_untested_regime():
    view = parse(
        {
            "game_family": "persuasion",
            "valid_actions": {"type": "buyer_decision"},
            "game_state": {
                "p": 0.8,
                "v": 120.0,
                "u": 0.0,
                "product_price": 100,
                "is_seller_know_cv": True,
                "seller_message": "yes",
            },
        }
    )
    assert pers_class(view) == "buyer_known_untested"



def _parse_b(you="player_2", **state):
    return parse(
        {
            "game_family": "bargaining",
            "your_player": you,
            "valid_actions": {"type": "decision"},
            "game_state": state,
        }
    )


def test_bargain_matrix_cells_split():
    from glee_agent.workflows.bargaining._plan import compose

    base = dict(
        money_to_divide=100,
        delta_1=0.80,
        delta_2=0.95,
        complete_information=True,
        horizon_known=True,
        max_rounds=12,
        round=3,
        last_offer={"player_1_gain": 50, "player_2_gain": 50},
    )
    half = _parse_b(**base)
    assert compose(half, "impatient_vs_hotter", "half_bot").procedure == "hold_first_even"
    stub = _parse_b(
        **{
            **base,
            "last_offer": {"player_1_gain": 60, "player_2_gain": 40},
            "history": [
                {"proposer": "player_1", "player_1_gain": 60, "player_2_gain": 40},
                {"proposer": "player_1", "player_1_gain": 60, "player_2_gain": 40},
            ],
        }
    )
    assert compose(stub, "impatient_vs_hotter", "stubborn").procedure == "stall_half"
    hid = _parse_b(**{**base, "complete_information": False})
    assert compose(hid, "impatient_vs_hotter", "half_bot").procedure == "mild"
    conc = _parse_b(**base)
    assert compose(conc, "impatient_vs_hotter", "conceder").procedure == "hold_first_even"
    conc_late = _parse_b(**{**base, "round": 10})
    assert compose(conc_late, "impatient_vs_hotter", "conceder").procedure == "late"

    patient = dict(
        money_to_divide=100,
        delta_1=1.0,
        delta_2=1.0,
        complete_information=True,
        horizon_known=True,
        max_rounds=12,
        round=2,
        last_offer={"player_1_gain": 50, "player_2_gain": 50},
    )
    even = _parse_b(you="player_1", **patient)
    assert compose(even, "patient", "half_bot").procedure == "hold_first_even"
    pstub = _parse_b(
        you="player_1",
        **{
            **patient,
            "last_offer": {"player_1_gain": 40, "player_2_gain": 60},
            "history": [
                {"proposer": "player_2", "player_1_gain": 40, "player_2_gain": 60},
                {"proposer": "player_2", "player_1_gain": 40, "player_2_gain": 60},
            ],
        },
    )
    assert compose(pstub, "patient", "stubborn").procedure == "stall_half"
    ppen = _parse_b(you="player_1", **{**patient, "round": 11})
    assert compose(ppen, "patient", "conceder").procedure == "late"
    plow = _parse_b(you="player_1", **{**patient, "round": 2})
    assert compose(plow, "patient", "lowballer").procedure == "stall_half"

    hot = dict(
        money_to_divide=100,
        delta_1=1.0,
        delta_2=0.80,
        complete_information=True,
        horizon_known=True,
        max_rounds=12,
        round=2,
        last_offer={"player_1_gain": 90, "player_2_gain": 10},
    )
    hlow = _parse_b(**hot)
    assert compose(hlow, "impatient_hot", "lowballer").procedure == "take_positive"
    hhalf = _parse_b(
        **{
            **hot,
            "last_offer": {"player_1_gain": 50, "player_2_gain": 50},
        }
    )
    assert compose(hhalf, "impatient_hot", "half_bot").procedure == "hot"

    dead_hot = _parse_b(
        you="player_1",
        money_to_divide=100,
        delta_1=0.80,
        delta_2=1.0,
        horizon_known=True,
        max_rounds=12,
        round=12,
        last_offer={"player_1_gain": 10, "player_2_gain": 90},
    )
    assert compose(dead_hot, "deadline", "mixed").procedure == "take_positive"
    dead_pat = _parse_b(
        you="player_1",
        money_to_divide=100,
        delta_1=1.0,
        delta_2=1.0,
        horizon_known=True,
        max_rounds=12,
        round=12,
        last_offer={"player_1_gain": 20, "player_2_gain": 80},
    )
    assert compose(dead_pat, "deadline", "mixed").procedure == "late"


def test_negotiation_matrix_cells_split():
    from glee_agent.workflows.negotiation._tactics import pick_procedure

    def _n(**state):
        return parse(
            {
                "game_family": "negotiation",
                "your_player": "player_1",
                "valid_actions": {"type": "offer"},
                "game_state": state,
            }
        )

    late = _n(
        player_1_role="seller",
        player_1_value=100.0,
        horizon_known=True,
        round=9,
        max_rounds=10,
    )
    assert pick_procedure(late, "incomplete_late", "seller", 100.0) == "close_110"
    late_stub = _n(
        player_1_role="seller",
        player_1_value=100.0,
        horizon_known=True,
        round=9,
        max_rounds=10,
        last_offer={"price": 80},
        history=[{"offer": {"price": 80}}, {"offer": {"price": 80}}],
    )
    assert pick_procedure(late_stub, "incomplete_late", "seller", 100.0) == "reservation"
    late_con = _n(
        player_1_role="seller",
        player_1_value=100.0,
        horizon_known=True,
        round=9,
        max_rounds=10,
        last_offer={"price": 95},
        history=[{"offer": {"price": 70}}, {"offer": {"price": 95}}],
    )
    assert pick_procedure(late_con, "incomplete_late", "seller", 100.0) == "close_110"
    shot = _n(
        player_1_role="seller",
        player_1_value=800000,
        horizon_known=True,
        round=1,
        max_rounds=1,
    )
    assert pick_procedure(shot, "one_shot", "seller", 800000) == "close_150"
    ult = _n(
        player_1_role="seller",
        player_2_role="buyer",
        player_1_value=10000,
        player_2_value=12000,
        complete_information=True,
        horizon_known=True,
        round=1,
        max_rounds=1,
    )
    assert pick_procedure(ult, "ultimatum", "seller", 10000) == "nash_half"
    open_e = _n(player_1_role="seller", player_1_value=40, round=1)
    assert pick_procedure(open_e, "open_horizon", "seller", 40) == "faratin_boulware"
    open_l = _n(player_1_role="seller", player_1_value=40, round=8)
    assert pick_procedure(open_l, "open_horizon", "seller", 40) == "faratin_conceder"
    strong_b = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 150.0,
                "round": 15,
                "horizon_known": False,
                "last_offer": {"price": 148.66},
            },
        }
    )
    assert pick_procedure(strong_b, "open_horizon", "buyer", 150.0) == "faratin_boulware"
    from glee_agent.workflows.negotiation.classify import own_band as n_own_band

    assert n_own_band(strong_b) == "strong_buyer"
    dead_s = _n(player_1_role="seller", player_1_value=150.0, round=5, max_rounds=10, horizon_known=True)
    assert n_own_band(dead_s) == "dead_seller"
    strong_s = _n(player_1_role="seller", player_1_value=80.0, round=1)
    assert n_own_band(strong_s) == "strong_seller"
    assert pick_procedure(strong_s, "open_horizon", "seller", 80.0) == "faratin_boulware"
    stuck = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "player_1_role": "seller",
                "player_2_role": "buyer",
                "player_2_value": 1_200_000.0,
                "round": 15,
                "last_offer": {"price": 1_200_120.0},
                "history": [
                    {"offer": {"price": 3_120_000.0}},
                    {"offer": {"price": 1_140_000.0}},
                    {"offer": {"price": 1_200_120.0}},
                    {"offer": {"price": 1_140_000.0}},
                    {"offer": {"price": 1_200_120.0}},
                ],
            },
        }
    )
    assert pick_procedure(stuck, "open_horizon", "buyer", 1_200_000.0) == "faratin_conceder"
    dead = _n(
        player_1_role="seller",
        player_1_value=40,
        horizon_known=True,
        round=10,
        max_rounds=10,
    )
    assert pick_procedure(dead, "incomplete_deadline", "seller", 40) == "faratin_reservation"


def test_persuasion_matrix_cells_split():
    from glee_agent.workflows.persuasion._beliefs import (
        compose_buy,
        compose_sell,
        seller_said_buy,
    )
    from glee_agent.workflows.persuasion.classify import classify
    from glee_agent.workflows.persuasion.routes.seller_blind import handle as blind_handle
    from glee_agent.workflows.persuasion.routes.seller_honest import handle as honest_handle

    def _p(atype="buyer_decision", **state):
        return parse(
            {
                "game_family": "persuasion",
                "valid_actions": {"type": atype},
                "game_state": state,
            }
        )

    known = _p(
        p=0.8,
        v=120.0,
        u=0.0,
        product_price=100,
        is_seller_know_cv=True,
        seller_message="yes",
    )
    assert compose_buy(known, "buyer_known_untested") == "raw_ev"
    high = _p(p=0.8, v=10, u=0, product_price=6)
    assert compose_buy(high, "buyer_p_high") == "ev_ratio_p_high"
    prior = _p(p=0.4, v=10, u=0, product_price=6)
    assert compose_buy(prior, "buyer_prior") == "ev_ratio_092"
    skip = _p(
        p=0.8,
        v=300.0,
        u=0.0,
        product_price=100,
        is_seller_know_cv=False,
        seller_message="no",
    )
    assert compose_buy(skip, "buyer_p_high") == "skip_block"
    after = _p(
        p=0.8,
        v=120.0,
        u=0.0,
        product_price=100,
        is_seller_know_cv=True,
        seller_message="yes",
        history=[{"seller_message": "skip", "bought": False, "quality": "low"}],
    )
    assert compose_buy(after, "buyer_known_untested") == "after_skip_trust"
    liar = _p(
        p=0.4,
        v=10,
        u=0,
        product_price=6,
        history=[{"seller_message": "worth buying", "bought": True, "quality": "low"}],
    )
    assert compose_buy(liar, "buyer_liar") == "lemon_ev"
    known_lemon = _p(
        p=0.4,
        v=10,
        u=0,
        product_price=6,
        is_seller_know_cv=True,
        history=[{"seller_message": "worth buying", "bought": True, "quality": "low"}],
    )
    assert compose_buy(known_lemon, "buyer_liar_known") == "lemon_ev"
    last = _p(
        p=0.5,
        v=10,
        u=0,
        product_price=6,
        horizon_known=True,
        round=20,
        total_rounds=20,
        seller_message="yes",
    )
    assert compose_buy(last, "buyer_prior") == "last_raw_ev"
    honest = _p(atype="seller_recommendation", current_quality="low", round=2)
    assert compose_sell(honest, "seller_honest") == "skip_lows"
    negative_prior = _p(
        atype="seller_recommendation",
        current_quality="low",
        p=1.0 / 3.0,
        v=2.0,
        u=0.0,
        product_price=1.0,
        is_seller_know_cv=True,
        round=1,
        total_rounds=20,
    )
    assert compose_sell(negative_prior, "seller_honest") == "skip_lows"
    burned = _p(
        atype="seller_recommendation",
        current_quality="low",
        round=4,
        history=[
            {
                "seller_message": "This unit is high quality. I recommend buying it.",
                "bought": True,
                "quality": "low",
            }
        ],
    )
    assert compose_sell(burned, "seller_honest") == "skip_lows"
    stalled = _p(
        atype="seller_recommendation",
        current_quality="low",
        is_seller_know_cv=True,
        history=[{"seller_message": "yes", "bought": False} for _ in range(3)],
    )
    assert compose_sell(stalled, "seller_honest") == "skip_lows"
    assert honest_handle(stalled) == {"decision": "no"}
    blind = _p(
        atype="seller_recommendation", round=2, is_seller_know_cv=False
    )
    assert compose_sell(blind, "seller_blind") == "always_yes"
    blind.state["history"] = [
        {"seller_message": "yes", "bought": False} for _ in range(3)
    ]
    assert compose_sell(blind, "seller_blind") == "always_yes"
    assert blind_handle(blind) == {"decision": "yes"}
    blind.state["current_quality"] = "high"
    assert classify(blind) == "seller_honest"
    assert compose_sell(blind, "seller_honest") == "yes_highs"
    assert honest_handle(blind) == {"decision": "yes"}
    blind.state["history"] += [
        {"seller_message": "no", "bought": False} for _ in range(2)
    ]
    assert blind_handle(blind) == {"decision": "yes"}
    slast = _p(
        atype="seller_recommendation",
        current_quality="low",
        horizon_known=True,
        round=20,
        total_rounds=20,
        is_seller_know_cv=False,
        history=[{"seller_message": "yes", "bought": False} for _ in range(3)],
    )
    assert compose_sell(slast, "seller_last") == "last_yes"
    assert seller_said_buy(
        "I would not prioritize this product in this round."
    ) is False
    assert seller_said_buy(
        "This one isn't up to the standard of my better stock. I'd hold out for the next one."
    ) is False
    assert seller_said_buy(
        "Not my strongest item this round. Your call, but I wouldn't push it."
    ) is False
    assert seller_said_buy(
        "I'd hold off this round, not a strong one."
    ) is False
    for message in (
        "This one's below standard - I wouldn't buy it myself.",
        "Save your money for the next one - this is a low.",
        "Don't touch this. You'll regret it.",
        "This product is low quality — you should pass on it.",
    ):
        assert seller_said_buy(message) is False
    for message in (
        "This one's genuinely strong this round, I'd buy it.",
        "Strong buy signal from me - it's a winner.",
        "Buy this one — top tier.",
        "I stake my reputation on it.",
    ):
        assert seller_said_buy(message) is True


def test_persuasion_ev_buy_after_skip_has_prior_probability():
    from glee_agent.workflows.persuasion._beliefs import ev_buy

    assert ev_buy(
        {
            "p": 0.8,
            "v": 120.0,
            "u": 0.0,
            "product_price": 100.0,
            "is_seller_know_cv": True,
            "seller_message": "yes",
            "history": [{"seller_message": "skip", "bought": False}],
        },
        follow_signal=True,
    ) == {"decision": "yes"}


def test_persuasion_opponent_state_records_signal_and_response():
    from glee_agent.workflows.persuasion._beliefs import state_of

    state = {
        "p": 0.5,
        "seller_message": "yes",
        "history": [
            {"seller_message": "yes", "bought": True, "quality": "high"},
            {"seller_message": "yes", "bought": True, "quality": "low"},
            {"seller_message": "skip", "bought": False, "quality": "low"},
        ],
    }
    opponent = state_of(state, role="seller")
    assert opponent.samples == 3
    assert opponent.bought_high == 1
    assert opponent.bought_low == 1
    assert opponent.yes_buy_rate == 1.0
    assert opponent.skipped_signals == 1
    assert opponent.seller_adversarial_probability == 0.5
    assert opponent.revealed_samples == 3
    assert opponent.revealed_signal_accuracy == 2.0 / 3.0
    assert opponent.revealed_yes_high_rate == 1.0
    assert opponent.revealed_yes_low_rate == 0.5


def test_persuasion_fad3a742_stops_unknown_value_text_after_three_lemons():
    from glee_agent.workflows.persuasion._beliefs import ev_buy

    state = {
        "p": 0.8,
        "v": 125.0,
        "u": 0.0,
        "product_price": 100.0,
        "is_seller_know_cv": False,
        "seller_message_type": "text",
        "seller_message": "I recommend this product.",
        "history": [
            {"quality": quality, "bought": True, "seller_message": "I recommend this product."}
            for quality in ("high", "low", "high", "high", "low", "high", "high", "low")
        ],
    }
    assert ev_buy(state, follow_signal=True) == {"decision": "no"}
    state["history"] = state["history"][:-1]
    assert ev_buy(state, follow_signal=True) == {"decision": "yes"}
