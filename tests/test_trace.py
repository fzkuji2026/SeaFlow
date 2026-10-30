import json

import glee_agent.trace
from glee_agent.trace import TracingClient


def test_tracing_client_ledgers_each_assigned_game_once(tmp_path, monkeypatch):
    client = object.__new__(TracingClient)
    client.batch = []
    client.assigned = {}
    client.ledger_path = None
    batches = iter(
        (
            [
                {"game_id": "a", "game_family": "bargaining", "your_player": "player_1"},
                {"game_id": "b", "game_family": "persuasion", "your_player": "player_2"},
            ],
            [{"game_id": "a", "game_family": "bargaining", "your_player": "player_1"}],
        )
    )
    monkeypatch.setattr("glee_sdk.GleeClient.pending_games", lambda _self: next(batches))
    ledger = tmp_path / "batch.jsonl"
    client.set_batch_ledger(ledger)

    assert len(client.pending_games()) == 2
    assert len(client.pending_games()) == 1
    assert client.assigned_count == 2
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["game_id"] for row in rows] == ["a", "b"]
    assert client.batch_summary()["family"] == {"bargaining": 1, "persuasion": 1}


def test_tracing_client_records_exact_action_snapshot(tmp_path, monkeypatch):
    from glee_agent.decision_trace import set_node

    client = object.__new__(TracingClient)
    client.ledger_path = tmp_path / "deployment-1.jsonl"
    monkeypatch.setattr(glee_agent.trace, "ACTION_TRACES", tmp_path / "actions")
    monkeypatch.setattr(
        "glee_agent.diagnose.snapshot",
        lambda game, action: {"round": game["game_state"]["round"], "action": action},
    )
    game = {
        "game_id": "g",
        "game_family": "bargaining",
        "game_state": {"round": 2},
    }
    set_node("bargaining.response.model.fitted_curve")
    client._record_action(game, {"decision": "accept"})
    row = json.loads((tmp_path / "actions" / "g.jsonl").read_text())
    assert row["deployment_id"] == "deployment-1"
    assert row["action"] == {"decision": "accept"}
    assert row["decision_node"] == "bargaining.response.model.fitted_curve"
    assert row["workflow_trace"] is not None
    assert row["snapshot"]["decision_node"] == "bargaining.response.model.fitted_curve"
    assert row["snapshot"]["round"] == 2
