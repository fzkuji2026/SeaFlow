from glee_agent.points import assign_ticks, family_ticks, snapshot


def test_one_game_tick_is_provisional_until_official_history_arrives():
    before = snapshot({"scores": {"negotiation": {"rating": 1500.0, "games_played": 10}}})
    after = snapshot({"scores": {"negotiation": {"rating": 1493.2, "games_played": 11}}})
    ticks = family_ticks(before, after)
    assert ticks == [{
        "family": "negotiation",
        "n": 1,
        "delta": -6.8,
        "per_game": -6.8,
        "exact": False,
        "provisional": True,
    }]
    pending = {"negotiation": ["abc"], "bargaining": [], "persuasion": []}
    assigned = assign_ticks(pending, ticks)
    assert assigned[0]["rating_delta"] == -6.8
    assert assigned[0]["exact"] is False
    assert assigned[0]["provisional"] is True
    assert pending["negotiation"] == []
