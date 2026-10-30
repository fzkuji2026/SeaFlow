from glee_agent.decision_trace import (
    add_candidate,
    begin,
    choose,
    current_node,
    current_trace,
    eliminate,
    set_node,
)


def test_workflow_trace_records_candidates_elimination_and_selection():
    begin("bargaining.response", context={"round": 2})
    add_candidate(
        "accept",
        {"decision": "accept"},
        source="current_offer",
        economic={"lower": 0.4, "expected": 0.4, "upper": 0.4},
    )
    eliminate(
        "walkaway",
        rule_id="strict_dominance",
        reason="positive accept dominates zero",
        evidence={"gap": 0.4},
    )
    choose(
        "accept",
        node="bargaining.response.select.strict_dominance.v1",
        stage="strict_dominance",
        reason="economic interval separated",
        margin=0.1,
    )
    trace = current_trace()
    assert trace["workflow"] == "bargaining.response"
    assert trace["candidates"][0]["id"] == "accept"
    assert trace["eliminations"][0]["candidate_id"] == "walkaway"
    assert trace["selection"]["candidate_id"] == "accept"
    assert current_node().endswith("strict_dominance.v1")


def test_set_node_none_clears_trace_between_actions():
    begin("bargaining.response")
    add_candidate("accept", {"decision": "accept"}, source="test")
    set_node(None)
    assert current_trace() is None
    assert current_node() is None
