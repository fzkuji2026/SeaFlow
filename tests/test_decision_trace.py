from glee_agent.decision_trace import current_node
from glee_agent.player import strategy as player_strategy
from glee_agent.rules import parse
from glee_agent.workflows.bargaining import _plan


def _response(share=0.4):
    return parse(
        {
            "game_family": "bargaining",
            "your_player": "player_2",
            "valid_actions": {"type": "decision"},
            "game_state": {
                "current_player": "player_2",
                "money_to_divide": 100,
                "last_offer": {"alice_gain": 100 * (1 - share), "bob_gain": 100 * share},
                "inflation_2": 10,
                "horizon_known": True,
                "max_rounds": 12,
                "round": 1,
            },
        }
    )


def test_response_safety_node_is_task_local_provenance(monkeypatch):
    monkeypatch.setattr(_plan, "fitted_response_curve", lambda *_: None)
    monkeypatch.setattr(_plan, "response_value_model", lambda *_: None)
    assert _plan._run(_response(0.05), _plan.Plan(0.55, "mild")) == {
        "decision": "reject"
    }
    assert current_node() == "bargaining.response.safety.nonterminal_bottom_offer"


def test_response_model_node_identifies_winning_model(monkeypatch):
    monkeypatch.setattr(
        _plan, "fitted_response_curve", lambda *_: {"decision": "accept"}
    )
    monkeypatch.setattr(_plan, "response_value_model", lambda *_: None)
    assert _plan._run(_response(), _plan.Plan(0.60, "mild")) == {
        "decision": "accept"
    }
    assert current_node() == "bargaining.response.model.fitted_curve"


def test_response_common_gate_rejects_low_value_dominated_offer(monkeypatch):
    monkeypatch.setattr(_plan, "fitted_response_curve", lambda *_: None)
    monkeypatch.setattr(
        _plan,
        "response_value_model",
        lambda *_: {
            "decision": "fallback",
            "accept_value": 0.10,
            "continuation_value": 0.25,
            "guard": 0.005,
        },
    )
    assert _plan._run(_response(0.15), _plan.Plan(0.60, "mild")) == {
        "decision": "reject"
    }
    assert current_node() == "bargaining.response.safety.low_value_dominated"


def test_response_common_gate_uses_scenario_rating_floor(monkeypatch):
    monkeypatch.setattr(_plan, "fitted_response_curve", lambda *_: None)
    monkeypatch.setattr(
        _plan,
        "response_value_model",
        lambda *_: {
            "decision": "fallback",
            "accept_value": 0.30,
            "continuation_value": 0.40,
            "guard": 0.005,
        },
    )
    monkeypatch.setattr(
        _plan,
        "public_bargaining_reference",
        lambda *_: {"median_payoff_share": 0.45},
    )
    assert _plan._run(_response(0.30), _plan.Plan(0.60, "mild")) == {
        "decision": "reject"
    }
    assert current_node() == "bargaining.response.safety.rating_floor_dominated"


def test_response_rating_surrogate_veto_targets_tail_loss(monkeypatch):
    monkeypatch.setattr(_plan, "fitted_response_curve", lambda *_: None)
    monkeypatch.setattr(
        _plan,
        "response_value_model",
        lambda *_: {
            "decision": "fallback",
            "accept_value": 0.45,
            "continuation_value": 0.50,
            "guard": 0.005,
        },
    )
    monkeypatch.setattr(
        _plan,
        "predict_rating",
        lambda *_args, **_kwargs: {
            "positive_probability": 0.20,
            "expected_rating_delta": -3.0,
        },
    )
    assert _plan._run(_response(0.45), _plan.Plan(0.60, "mild")) == {
        "decision": "reject"
    }
    assert current_node() == "bargaining.response.risk.rating_surrogate_veto"


def test_response_scenario_node_covers_procedure_leaf(monkeypatch):
    monkeypatch.setattr(_plan, "fitted_response_curve", lambda *_: None)
    monkeypatch.setattr(_plan, "response_value_model", lambda *_: None)
    assert _plan._run(_response(), _plan.Plan(0.60, "late")) == {
        "decision": "reject"
    }
    assert current_node() == "bargaining.response.scenario.late"


def test_patient_stable_near_even_node_rejects_sub_half_plateau(monkeypatch):
    monkeypatch.setattr(_plan, "fitted_response_curve", lambda *_: None)
    monkeypatch.setattr(_plan, "response_value_model", lambda *_: None)
    history = []
    for round_n in range(1, 10):
        proposer = "player_1" if round_n % 2 else "player_2"
        offer = (
            {"player_1_gain": 55, "player_2_gain": 45}
            if proposer == "player_1"
            else {"player_1_gain": 49.9, "player_2_gain": 50.1}
        )
        history.append(
            {
                "round": round_n,
                "proposer": proposer,
                "offer": {**offer, "proposer": proposer},
                "decision": "reject",
            }
        )
    view = parse(
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
                "round": 10,
                "history": history,
                "last_offer": {
                    "round": 10,
                    "proposer": "player_2",
                    "player_1_gain": 49.9,
                    "player_2_gain": 50.1,
                },
            },
        }
    )
    assert _plan._run(view, _plan.Plan(0.55, "hold_first_even")) == {
        "decision": "reject"
    }
    assert current_node().endswith("patient_stable_near_even_hold")


def test_player_fallback_has_node_without_leaking_it_into_action(monkeypatch):
    monkeypatch.setattr(
        "glee_agent.player._primary_strategy", lambda _game: (_ for _ in ()).throw(ValueError())
    )
    game = {
        "game_family": "bargaining",
        "valid_actions": {"type": "decision"},
        "game_state": {
            "money_to_divide": 100,
            "last_offer": {"alice_gain": 50, "bob_gain": 50},
        },
    }
    assert player_strategy(game) == {"decision": "reject"}
    assert current_node() == "player.fallback.legal_action"
