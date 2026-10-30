from scripts.build_reference import _aggregate_bargaining, _aggregate_negotiation


def test_reference_aggregates_role_specific_medians():
    bargaining = _aggregate_bargaining(
        [
            (("Alice", "0.8", "12", False), (0.6, True, 2, 0.4, 0.5), ("Bob", "0.9", "12", False), (0.6, True, 2, 0.45, 0.5)),
            (("Alice", "0.8", "12", False), (0.5, False, 12, 0.0, 0.0), ("Bob", "0.9", "12", False), (0.5, False, 12, 0.0, 0.0)),
        ]
    )
    assert bargaining["Alice|0.8|12|False"]["median_payoff_share"] == 0.2

    negotiation = _aggregate_negotiation(
        [
            (("seller", "0.8", "1", False), (True, 1, 0.2, 0.5, 1.0, 0.5), ("buyer", "1.2", "1", False), (True, 1, 0.2, 0.5, 1.0, 0.5)),
        ]
    )
    assert negotiation["seller|0.8|1|False"]["median_surplus_share"] == 0.5
