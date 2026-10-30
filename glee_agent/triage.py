"""Evidence-bounded outcome triage. Negative rating is not itself a bug."""

from __future__ import annotations

from typing import Any


def _decision(value: Any) -> str:
    return str((value or {}).get("decision") or "").lower().replace("_", "")


def _primary_classification(
    row: dict[str, Any],
    replay: dict[str, Any],
    diagnose: dict[str, Any],
) -> dict[str, Any]:
    family = str(replay.get("game_family") or row.get("game_family") or "")
    you = str(replay.get("your_player") or row.get("your_player") or "player_1")
    result = replay.get("result") or row.get("result") or {}
    delta = row.get("rating_delta")
    evidence: list[dict[str, Any]] = []
    if result.get("outcome") == "timeout":
        timed_out_by = None
        for history in ((replay.get("state") or {}).get("history") or []):
            if isinstance(history, dict) and history.get("timed_out_by"):
                timed_out_by = history.get("timed_out_by")
                break
        classification = "avoidable" if timed_out_by == you else "opponent_constrained"
        return {
            "classification": classification,
            "reason": "our_timeout" if timed_out_by == you else "opponent_timeout",
            "actionable": timed_out_by == you,
            "evidence": [{"timed_out_by": timed_out_by}],
        }
    nonnegative_rating = isinstance(delta, (int, float)) and delta >= 0
    if family == "bargaining":
        payoff = float(result.get(f"{you}_payoff") or 0.0)
        realized_regret: list[dict[str, Any]] = []
        for turn in diagnose.get("turns") or []:
            if turn.get("action_type") != "decision":
                continue
            actual = _decision(turn.get("actual_action"))
            current = _decision(turn.get("action"))
            calc = turn.get("calc") or {}
            pot = float(calc.get("pot") or 0.0)
            accept_value = calc.get("discounted_last_share")
            if actual != "reject" or pot <= 0 or not isinstance(
                accept_value, (int, float)
            ):
                continue
            if float(accept_value) <= payoff / pot + 1e-9:
                continue
            item = {
                "round": turn.get("round"),
                "path": turn.get("path"),
                "accept_value": accept_value,
                "realized_final_value": payoff / pot,
                "current_policy": current,
                "segment": turn.get("segment"),
            }
            realized_regret.append(item)
            if current == "accept":
                evidence.append(item)
        if evidence:
            return {
                "classification": "avoidable",
                "reason": "supported_earlier_accept",
                "actionable": True,
                "evidence": evidence,
            }
        if realized_regret:
            return {
                "classification": "realized_regret",
                "reason": "earlier_offer_exceeded_final_payoff",
                "actionable": False,
                "evidence": realized_regret,
            }
        if nonnegative_rating:
            return {
                "classification": "positive",
                "reason": "nonnegative_rating",
                "actionable": False,
                "evidence": [],
            }
    if family == "negotiation":
        config = replay.get("config") or {}
        role = "seller" if you == "player_1" else "buyer"
        final_payoff = float(result.get(f"{you}_payoff") or 0.0)
        for turn in diagnose.get("turns") or []:
            calc = turn.get("calc") or {}
            actual = _decision(turn.get("actual_action"))
            profit = calc.get("profit")
            round_n = int(turn.get("round") or 0)
            at_cap = not bool(turn.get("horizon_known")) and (
                round_n >= 99
                or (
                    not bool(turn.get("complete_information"))
                    and round_n >= 98
                )
            )
            if (
                turn.get("action_type") == "decision"
                and actual == "rejectoffer"
                and at_cap
                and isinstance(profit, (int, float))
                and float(profit) > final_payoff + 1e-9
            ):
                evidence.append(
                    {
                        "round": round_n,
                        "path": turn.get("path"),
                        "accept_profit": float(profit),
                        "realized_final_profit": final_payoff,
                        "segment": turn.get("segment"),
                    }
                )
        if evidence:
            return {
                "classification": "avoidable",
                "reason": "supported_platform_cap_accept",
                "actionable": True,
                "evidence": evidence,
            }
        realized_regret = []
        for turn in diagnose.get("turns") or []:
            if turn.get("action_type") != "decision":
                continue
            calc = turn.get("calc") or {}
            actual = _decision(turn.get("actual_action"))
            profit = calc.get("profit")
            if (
                actual == "rejectoffer"
                and isinstance(profit, (int, float))
                and float(profit) > final_payoff + 1e-9
            ):
                realized_regret.append(
                    {
                        "round": turn.get("round"),
                        "path": turn.get("path"),
                        "accept_profit": float(profit),
                        "realized_final_profit": final_payoff,
                        "current_policy": _decision(turn.get("action")),
                        "segment": turn.get("segment"),
                    }
                )
        if realized_regret:
            return {
                "classification": "realized_regret",
                "reason": "earlier_offer_exceeded_final_payoff",
                "actionable": False,
                "evidence": realized_regret,
            }
        if nonnegative_rating:
            return {
                "classification": "positive",
                "reason": "nonnegative_rating",
                "actionable": False,
                "evidence": [],
            }
        if int(config.get("max_rounds") or 0) == 1 and role == "buyer":
            return {
                "classification": "opponent_constrained",
                "reason": "one_shot_responder",
                "actionable": False,
                "evidence": [],
            }
        seller = config.get("seller_value")
        buyer = config.get("buyer_value")
        if (
            config.get("complete_information") is False
            and isinstance(seller, (int, float))
            and isinstance(buyer, (int, float))
            and float(buyer) <= float(seller)
        ):
            return {
                "classification": "posthoc_no_zopa",
                "reason": "hidden_values_had_no_overlap",
                "actionable": False,
                "evidence": [{"seller_mult": seller, "buyer_mult": buyer}],
            }
    if nonnegative_rating:
        return {
            "classification": "positive",
            "reason": "nonnegative_rating",
            "actionable": False,
            "evidence": [],
        }
    if family == "persuasion" and you == "player_1":
        last = diagnose.get("last") or {}
        opponent = ((last.get("calc") or {}).get("opponent_state") or {})
        if opponent.get("buyer_style") == "rejects_all":
            return {
                "classification": "opponent_constrained",
                "reason": "buyer_rejects_yes_and_no",
                "actionable": False,
                "evidence": [
                    {
                        "yes_buy_rate": opponent.get("yes_buy_rate"),
                        "no_buy_rate": opponent.get("no_buy_rate"),
                    }
                ],
            }
    if family == "persuasion" and you == "player_2":
        return {
            "classification": "stochastic",
            "reason": "iid_quality_realization_under_posterior_policy",
            "actionable": False,
            "evidence": [],
        }
    return {
        "classification": "uncertain",
        "reason": "counterfactual_not_observed",
        "actionable": False,
        "evidence": evidence,
    }


def _factor_tags(
    row: dict[str, Any],
    replay: dict[str, Any],
    diagnose: dict[str, Any],
    primary: dict[str, Any],
) -> list[str]:
    """Return overlapping context/outcome/signal tags without forcing an action."""
    family = str(replay.get("game_family") or row.get("game_family") or "unknown")
    you = str(replay.get("your_player") or row.get("your_player") or "player_1")
    result = replay.get("result") or row.get("result") or {}
    process = diagnose.get("process") or {}
    turns = diagnose.get("turns") or []
    valid = [turn for turn in turns if isinstance(turn, dict)]
    first = valid[0] if valid else {}
    indicators = ((first.get("segment") or {}).get("indicators") or {})
    config = replay.get("config") or {}
    delta = row.get("rating_delta")
    rounds = int(
        process.get("rounds_played")
        or result.get("agreed_round")
        or max((int(turn.get("round") or 0) for turn in valid), default=0)
    )
    outcome = str(result.get("outcome") or "unknown")

    tags = [
        f"family:{family}",
        f"role:{indicators.get('role') or ('player_1' if you == 'player_1' else 'player_2')}",
        f"outcome:{outcome}",
        f"primary:{primary.get('classification') or 'unknown'}",
    ]
    if isinstance(delta, (int, float)):
        tags.append("rating:negative" if delta < 0 else "rating:nonnegative")
    if rounds > 1:
        tags.append("process:multi_round")
    if rounds >= 12:
        tags.append("process:long_game")
    if outcome not in ("agreement", "unknown") and family in ("bargaining", "negotiation"):
        tags.append("outcome:no_agreement")

    complete = indicators.get("complete_information", config.get("complete_information"))
    if isinstance(complete, bool):
        tags.append("info:complete" if complete else "info:incomplete")
    messages = indicators.get("messages_allowed", config.get("messages_allowed"))
    if isinstance(messages, bool):
        tags.append("messages:allowed" if messages else "messages:blocked")
    horizon = indicators.get("horizon")
    if horizon is not None:
        tags.append(f"horizon:{horizon}")

    if family == "bargaining":
        own_delta = indicators.get("own_delta")
        if not isinstance(own_delta, (int, float)):
            key = "delta_1" if you == "player_1" else "delta_2"
            own_delta = config.get(key)
        if isinstance(own_delta, (int, float)):
            tags.append(f"own_delta:{float(own_delta):.6g}")
            if rounds > 1 and float(own_delta) < 1.0:
                tags.append("signal:discount_exposure")
        states = [((turn.get("calc") or {}).get("opponent_state") or {}) for turn in valid]
        if any(
            int(state.get("plateau_repeats") or 0) >= 4
            or float(state.get("stall_probability") or 0.0) >= 0.9375
            for state in states
        ):
            tags.append("signal:stable_opponent_boundary")
        calc = (diagnose.get("last") or diagnose.get("r1") or {}).get("calc") or {}
        pot = calc.get("pot")
        reference = (calc.get("public_reference") or {}).get("median_payoff_share")
        payoff = result.get(f"{you}_payoff")
        if all(isinstance(value, (int, float)) for value in (pot, reference, payoff)) and float(pot) > 0:
            if float(payoff) / float(pot) + 1e-9 < float(reference):
                tags.append("signal:below_field_median")

    if primary.get("reason") == "earlier_offer_exceeded_final_payoff":
        tags.append("signal:realized_regret")
    if primary.get("reason") == "supported_platform_cap_accept":
        tags.append("signal:platform_cap_regret")
    if primary.get("reason") == "hidden_values_had_no_overlap":
        tags.append("signal:no_zopa")
    if primary.get("reason") == "buyer_rejects_yes_and_no":
        tags.append("signal:rejects_all")
    return list(dict.fromkeys(tags))


def classify(
    row: dict[str, Any],
    replay: dict[str, Any],
    diagnose: dict[str, Any],
) -> dict[str, Any]:
    primary = _primary_classification(row, replay, diagnose)
    return {**primary, "factors": _factor_tags(row, replay, diagnose, primary)}
