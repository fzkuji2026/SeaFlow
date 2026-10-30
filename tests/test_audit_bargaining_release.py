from scripts.audit_bargaining_release import audit_game, summarize


def _game(decision: str, final: float = 20.0):
    return {
        "game_id": "g",
        "game_family": "bargaining",
        "your_player": "player_1",
        "completed_at": "2026-08-26T00:00:00+00:00",
        "config": {
            "money_to_divide": 100.0,
            "delta_1": 0.9,
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 12,
        },
        "state": {
            "money_to_divide": 100.0,
            "delta_1": 0.9,
            "complete_information": False,
            "horizon_known": True,
            "max_rounds": 12,
            "history": [
                {
                    "round": 2,
                    "proposer": "player_2",
                    "offer": {"player_1_gain": 40.0, "player_2_gain": 60.0},
                    "decision": decision,
                }
            ]
        },
        "result": {"player_1_payoff": final, "player_2_payoff": 60.0},
    }


def test_rejected_offer_to_candidate_accept_has_factual_counterfactual(monkeypatch):
    monkeypatch.setattr(
        "scripts.audit_bargaining_release.strategy",
        lambda _game: {"decision": "accept"},
    )
    rows = audit_game(_game("reject", final=20.0))
    assert len(rows) == 1
    assert rows[0]["known"] is True
    assert round(rows[0]["gain"], 6) == 0.16


def test_accepted_offer_to_candidate_reject_is_unknown(monkeypatch):
    monkeypatch.setattr(
        "scripts.audit_bargaining_release.strategy",
        lambda _game: {"decision": "reject"},
    )
    rows = audit_game(_game("accept", final=36.0))
    assert len(rows) == 1
    assert rows[0]["known"] is False
    report = summarize([_game("accept", final=36.0)])
    assert report["pass"] is False


def test_audit_stops_after_first_changed_decision(monkeypatch):
    game = _game("reject", final=20.0)
    game["state"]["history"].append(
        {
            "round": 4,
            "proposer": "player_2",
            "offer": {"player_1_gain": 45.0, "player_2_gain": 55.0},
            "decision": "reject",
        }
    )
    monkeypatch.setattr(
        "scripts.audit_bargaining_release.strategy",
        lambda _game: {"decision": "accept"},
    )

    rows = audit_game(game)

    assert len(rows) == 1
    assert rows[0]["round"] == 2
