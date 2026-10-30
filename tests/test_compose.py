from glee_agent.compose import choose, pipeline
from glee_agent.families.negotiation import compose_ask
from glee_agent.policy import DEFAULT, set_policy
from glee_agent.rules import parse


def setup_function():
    set_policy(DEFAULT)


def test_pipeline_records_named_steps():
    trace = pipeline(10.0, [("double", lambda x: x * 2), ("add", lambda x: x + 1)])
    assert trace.steps == ("seed", "double", "add")
    assert trace.value == 21.0


def test_choose_stays_on_default_without_picker():
    name, value = choose("faratin", {"faratin": 50, "tft": 45})
    assert name == "faratin"
    assert value == 50


def test_negotiation_incomplete_trace_names_tactics():
    view = parse(
        {
            "game_family": "negotiation",
            "your_player": "player_1",
            "valid_actions": {"type": "offer"},
            "game_state": {
                "current_player": "player_1",
                "player_1_role": "seller",
                "player_1_value": 40,
            },
        }
    )
    trace = compose_ask(view, "incomplete", "seller", 40.0)
    assert trace.steps == ("seed", "faratin", "tit_for_tat", "meet", "belief_cap")
    assert trace.value == 60.0  # DEFAULT seller_open=1.50
