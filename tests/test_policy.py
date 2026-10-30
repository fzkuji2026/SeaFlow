import json

from glee_agent.policy import DEFAULT, keep_mutation, load_policy, mutate, set_policy
from glee_agent.strategy import strategy


def test_mutate_stays_in_bounds():
    import random

    policy = DEFAULT
    for _ in range(20):
        policy, family, key = mutate(policy, rng=random.Random(0))
        if key == "seller_always_yes":
            assert isinstance(policy[family][key], bool)
        else:
            assert isinstance(policy[family][key], float)


def test_keep_on_rating_gain():
    assert keep_mutation(1500, 1510, 0.4, 0.4) is True
    assert keep_mutation(1500, 1499, 0.4, 0.4) is False
    assert keep_mutation(1500, 1490, 0.50, 0.40) is False
    assert keep_mutation(1500, 1499, 0.50, 0.40) is True
    assert keep_mutation(1500, 1510, 0.0, 0.10) is True
    assert keep_mutation(1500, 1499, 0.0, 0.10) is False



def test_strategy_reads_mutated_offer_share():
    set_policy(
        {
            **DEFAULT,
            "bargaining": {**DEFAULT["bargaining"], "offer_share": 0.60},
        }
    )
    action = strategy(
        {
            "game_family": "bargaining",
            "valid_actions": {"type": "offer"},
            "game_state": {"current_player": "player_1", "money_to_divide": 100},
        }
    )
    assert action["alice_gain"] == 60
    set_policy(DEFAULT)


def test_load_policy_drops_obsolete_experiment_dimensions(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "bargaining": {"offer_share": 0.60, "accept_early": 0.40},
                "persuasion": {"untested_ev_ratio": 1.0, "history_step": 0.15},
            }
        )
    )
    loaded = load_policy(path)
    assert loaded["bargaining"] == {"offer_share": 0.60}
    assert loaded["persuasion"] == {
        "seller_always_yes": False,
        "untested_ev_ratio": 1.0,
    }
    set_policy(DEFAULT)
