from glee_agent.rules import parse
from glee_agent.segments import MODEL_SCHEMA, segment
from glee_agent.strategy import strategy
from glee_agent.fitted import data as fitted_data
from scripts.fit_models import (
    _external_accept_validation,
    _fit_accept_threshold,
    _passes_external_validation,
    _time_decay,
)


def _bargaining_lowball(messages_allowed: bool):
    history = []
    opponent = {2: 5.0, 4: 7.5, 6: 12.5, 8: 15.0}
    for round_n in range(1, 10):
        if round_n % 2:
            history.append(
                {
                    "round": round_n,
                    "proposer": "player_1",
                    "offer": {"player_1_gain": 50.0, "player_2_gain": 50.0},
                    "decision": "reject",
                }
            )
        else:
            mine = opponent[round_n]
            history.append(
                {
                    "round": round_n,
                    "proposer": "player_2",
                    "offer": {"player_1_gain": mine, "player_2_gain": 100.0 - mine},
                    "decision": "reject",
                }
            )
    return {
        "game_family": "bargaining",
        "your_player": "player_1",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "current_player": "player_1",
            "money_to_divide": 100.0,
            "delta_1": 0.95,
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 12,
            "round": 10,
            "messages_allowed": messages_allowed,
            "history": history,
            "last_offer": {
                "round": 10,
                "proposer": "player_2",
                "player_1_gain": 17.5,
                "player_2_gain": 82.5,
            },
        },
    }


def test_expired_segmented_bargaining_policy_falls_back():
    assert strategy(_bargaining_lowball(True)) == {"decision": "reject"}


def test_segmented_bargaining_policy_does_not_pool_message_cell_for_actions():
    assert strategy(_bargaining_lowball(False)) == {"decision": "reject"}


def test_stale_response_policy_without_external_validation_falls_back():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 10_000.0,
                "delta_1": 1.0,
                "delta_2": 0.95,
                "complete_information": True,
                "horizon_known": False,
                "round": 3,
                "messages_allowed": True,
                "history": [
                    {
                        "round": 1,
                        "proposer": "player_1",
                        "offer": {"player_1_gain": 6800.0, "player_2_gain": 3200.0},
                        "decision": "reject",
                    },
                    {
                        "round": 2,
                        "proposer": "player_2",
                        "offer": {"player_1_gain": 5000.0, "player_2_gain": 5000.0},
                        "decision": "reject",
                    },
                ],
                "last_offer": {
                    "round": 3,
                    "proposer": "player_1",
                    "player_1_gain": 6550.0,
                    "player_2_gain": 3450.0,
                },
            },
        }
    )
    assert action == {"decision": "reject"}


def test_externally_validated_response_policy_accepts_supported_offer():
    action = strategy(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100.0,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "messages_allowed": True,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 52.0,
                    "player_2_gain": 48.0,
                },
            },
        }
    )
    assert action == {"decision": "accept"}


def test_response_trace_contains_external_validation_evidence():
    from glee_agent.workflows.bargaining._helpers import fitted_response_curve

    view = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100.0,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
                "messages_allowed": True,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 52.0,
                    "player_2_gain": 48.0,
                },
            },
        }
    )
    trace = fitted_response_curve(view, "mild")
    assert trace is not None
    assert trace["external_triggered"] == 3
    assert trace["external_improved"] == 3
    assert trace["external_regressed"] == 0
    assert trace["external_gain"] == 0.053


def test_response_curve_excludes_policy_with_external_regressions():
    from glee_agent.workflows.bargaining._helpers import fitted_response_curve

    view = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100.0,
                "delta_2": 0.95,
                "complete_information": False,
                "horizon_known": False,
                "round": 1,
                "messages_allowed": False,
                "last_offer": {
                    "round": 1,
                    "proposer": "player_1",
                    "player_1_gain": 52.0,
                    "player_2_gain": 48.0,
                },
            },
        }
    )

    assert fitted_response_curve(view, "mild") is None


def test_segment_id_uses_hidden_marker_and_keeps_amount_outside_id():
    view = parse(_bargaining_lowball(True))
    row = segment(view, "mild")
    assert "opp_delta=hidden" in row.scenario_id
    assert "100" not in row.scenario_id
    assert row.indicators["amount"] == 100.0


def test_model_schema_maps_estimates_to_candidate_selection():
    for family, schema in MODEL_SCHEMA.items():
        assert schema["strategy_map"], family
        for mapping in schema["strategy_map"].values():
            assert mapping["uses"]
            assert mapping["scores"]
            assert mapping["selects"]


def test_segmented_pair_uses_one_shared_parent(monkeypatch):
    import glee_agent.fitted as fitted

    row = {
        "kind": "linear",
        "mean": [0.0],
        "scale": [1.0],
        "coef": [1.0],
        "intercept": 0.0,
    }
    monkeypatch.setattr(
        fitted,
        "data",
        lambda: {
            "segmented_models": {
                "accept": {"exact": row, "pooled": row},
                "continue": {"pooled": row},
            }
        },
    )
    pair = fitted.predict_pair_segmented(
        "accept", "continue", ("exact", "pooled"), [0.25]
    )
    assert pair is not None
    assert pair[0:3] == (0.25, 0.25, "level_1")


def test_bargaining_fitted_offer_stays_inside_workflow_candidate_set(monkeypatch):
    from glee_agent.workflows.bargaining._helpers import fitted_offer_curve
    import glee_agent.workflows.bargaining._helpers as helpers

    monkeypatch.setattr(
        helpers,
        "predict_pair_segmented",
        lambda *_args, **_kwargs: (
            1.0,
            0.0,
            "level_0",
            {"guard": 0.0, "n": 20},
            {"guard": 0.0, "n": 20},
        ),
    )
    view = parse(
        {
            "game_family": "bargaining",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "money_to_divide": 100,
                "delta_1": 1.0,
                "complete_information": False,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )
    curve = fitted_offer_curve(view, 0.55, "patient_probe")
    assert curve is not None
    assert curve["share"] == 0.60


def test_persuasion_segment_marks_unobserved_values_hidden():
    view = parse(
        {
            "game_family": "persuasion",
            "your_player": "player_1",
            "valid_actions": {"type": "seller_recommendation"},
            "game_state": {
                "current_player": "player_1",
                "p": 0.5,
                "product_price": 100.0,
                "is_seller_know_cv": False,
                "current_quality": "high",
            },
        }
    )
    row = segment(view, "yes_highs")
    assert "high_mult=hidden" in row.scenario_id
    assert "low_mult=hidden" in row.scenario_id
    assert row.indicators["high_mult"] is None
    assert row.indicators["low_mult"] is None


def test_negotiation_t10_repeated_buyer_bid_probes_strict_boundary():
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
                "max_rounds": 10,
                "round": 8,
                "history": [
                    {"round": 1, "offer": {"price": 174.0, "from_player": "player_1"}},
                    {"round": 2, "offer": {"price": 120.0, "from_player": "player_2"}, "decision": "RejectOffer"},
                    {"round": 3, "offer": {"price": 173.99, "from_player": "player_1"}},
                    {"round": 4, "offer": {"price": 120.0, "from_player": "player_2"}, "decision": "RejectOffer"},
                    {"round": 5, "offer": {"price": 173.59, "from_player": "player_1"}},
                    {"round": 6, "offer": {"price": 120.0, "from_player": "player_2"}, "decision": "RejectOffer"},
                ],
                "last_offer": {"round": 8, "price": 170.8, "from_player": "player_1"},
            },
        }
    )
    assert action == {"decision": "RejectOffer", "product_price": 120.3}


def test_deployable_acceptance_curves_have_causal_claim_direction():
    tables = fitted_data()["segmented_models"]
    checked = 0
    for name, models in tables.items():
        if not name.endswith("offer_accept"):
            continue
        checked += len(models)
        assert all(float(row["coef"][0]) < 0.0 for row in models.values())
    assert checked > 0


def test_deployable_response_policies_improve_chronological_holdout():
    policies = fitted_data()["segmented_policies"]["bargaining_response_accept"]
    assert policies
    assert all(float(row["holdout_gain"]) > 0.0 for row in policies.values())
    assert all(int(row["train_triggered"]) >= 3 for row in policies.values())
    assert all(int(row["external_improved"]) >= 3 for row in policies.values())
    assert all(
        int(row["external_improved"]) > int(row["external_regressed"])
        for row in policies.values()
    )


def test_response_policy_evaluation_uses_first_trigger_per_game():
    rows = []
    for index in range(8):
        timestamp = 1.0 if index < 6 else 10.0
        game_id = f"game-{index}"
        rows.extend(
            [
                ([0.6, 0.4], 0.5, 0.5, timestamp, game_id),
                ([0.9, 0.7], 0.0, 0.5, timestamp, game_id),
            ]
        )
    policy = _fit_accept_threshold(rows, holdout_ts=5.0)
    assert policy is not None
    assert policy["effective_games"] == 8
    assert policy["holdout_triggered"] == 2
    assert policy["train_gain"] == 0.3
    assert policy["holdout_gain"] == 0.2


def test_training_weights_decay_continuously_with_age():
    rows = [
        ([0.1], 0.0, 1.0, 0.0),
        ([0.2], 0.0, 1.0, 3600.0),
        ([0.3], 0.0, 1.0, 7200.0),
    ]
    weighted = _time_decay(rows, 3600.0)
    assert [row[2] for row in weighted] == [0.25, 0.5, 1.0]


def test_external_policy_gate_requires_three_clean_improvements():
    rows = [
        ([0.60, 0.40], 0.50, 1.0, 1.0, "a"),
        ([0.55, 0.40], 0.50, 1.0, 2.0, "b"),
        ([0.52, 0.40], 0.50, 1.0, 3.0, "c"),
        ([0.90, 0.40], 0.00, 1.0, 4.0, "a"),
    ]
    result = _external_accept_validation(rows, threshold=0.10)
    assert result["external_triggered"] == 3
    assert result["external_gain"] == 0.17
    assert result["external_improved"] == 3
    assert result["external_regressed"] == 0
    assert _passes_external_validation(result)


def test_external_policy_gate_rejects_negative_deployment_batch():
    rows = [
        ([0.55, 0.40], 0.70, 1.0, float(index), f"batch-{index}")
        for index in range(20)
    ]
    result = _external_accept_validation(rows, threshold=0.10)
    assert result["external_gain"] < 0
    assert not _passes_external_validation(result)


def test_external_policy_gate_rejects_any_regression():
    assert not _passes_external_validation(
        {
            "external_triggered": 5,
            "external_improved": 4,
            "external_regressed": 1,
            "external_gain": 0.5,
        }
    )


def test_negotiation_opponent_state_exposes_validated_zopa_probability():
    from glee_agent.workflows.negotiation._tactics import state_of

    view = parse(
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
                "round": 2,
                "last_offer": {"price": 90.0, "from_player": "player_2"},
            },
        }
    )
    opponent = state_of(view, "seller", 80.0)
    assert opponent.zopa_model == "negotiation_zopa_seller_open"
    assert 0.0 <= opponent.zopa_probability <= 1.0
    assert opponent.zopa_model_auc >= 0.75
