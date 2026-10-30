from glee_agent.reference import bargaining, negotiation, persuasion
from glee_agent.rules import parse


def _view(family, action_type, you, **state):
    return parse(
        {
            "game_family": family,
            "your_player": you,
            "valid_actions": {"type": action_type},
            "game_state": state,
        }
    )


def test_public_reference_matches_competition_scenarios():
    bargain = _view(
        "bargaining",
        "decision",
        "player_1",
        delta_1=0.8,
        horizon_known=True,
        max_rounds=12,
        complete_information=False,
    )
    assert bargaining(bargain)["median_payoff_share"] == 0.36

    deal = _view(
        "negotiation",
        "offer",
        "player_1",
        player_1_role="seller",
        player_1_value=80,
        horizon_known=True,
        max_rounds=1,
        complete_information=False,
    )
    assert negotiation(deal)["median_first_price_over_base"] == 1.2

    talk = _view(
        "persuasion",
        "seller_recommendation",
        "player_1",
        p=0.5,
        v=200,
        product_price=100,
        is_seller_know_cv=True,
        seller_message_type="binary",
        is_myopic=False,
    )
    assert persuasion(talk) is not None
