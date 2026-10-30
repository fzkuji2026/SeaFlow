"""Cheap-talk Bayes (Crawford-Sobel / Akerlof). Shared by classify and buyers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from glee_agent.fitted import prior as fitted_prior
from glee_agent.policy import current

# Start likelihoods: official-ish cheap talk is weakly informative.
P_YES_HIGH0 = 0.8
P_YES_LOW0 = 0.5


@dataclass(frozen=True)
class OpponentState:
    role: str
    samples: int
    seller_type: str
    current_recommendation: bool | None
    current_high_posterior: float
    seller_reliability: float
    seller_adversarial_probability: float
    bought_high: int
    bought_low: int
    skipped_signals: int
    buyer_style: str
    yes_samples: int
    yes_buy_rate: float
    no_samples: int
    no_buy_rate: float
    predicted_yes_buy_rate: float
    predicted_no_buy_rate: float
    signal_response_gap: float
    evidence_strength: float
    revealed_samples: int
    revealed_signal_accuracy: float
    revealed_yes_high_rate: float
    revealed_yes_low_rate: float


def seller_said_buy(message: Any) -> bool | None:
    if message is None:
        return None
    if isinstance(message, dict):
        decision = str(message.get("decision") or "").lower()
        if decision in ("yes", "buy", "recommend"):
            return True
        if decision in ("no", "skip", "pass"):
            return False
        message = message.get("message") or message.get("text")
    text = str(message).strip().lower()
    if not text:
        return None
    skip_needles = (
        "i'd skip",
        "i would skip",
        "skip this",
        "skip it",
        "not my best",
        "isn't my best",
        "isnt my best",
        "this one is low quality",
        "this is low quality",
        "i wouldn't buy this",
        "i would not buy this",
        "not the premium",
        "not up to",
        "isn't up to",
        "isnt up to",
        "not my strongest",
        "wouldn't push",
        "would not push",
        "not my usual",
        "isn't my usual",
        "this one is not worth",
        "this one isn't worth",
        "this one isnt worth",
        "it is not worth",
        "it's not worth",
        "do not buy",
        "don't buy",
        "hard pass",
        "avoid this",
        "sit this one out",
        "i'd hold back",
        "i would hold back",
        "below our usual",
        "below standard",
        "product is low quality",
        "save your money",
        "don't touch this",
        "do not touch this",
        "not a strong",
        "not the high-quality",
        "not high quality",
        "nothing fancy",
        "nothing special",
        "ordinary unit",
        "this one is ordinary",
        "understand a pass",
        "serviceable",
        "hold off on this",
        "hold off this",
        "hold off this round",
        "i'd hold off",
        "i would hold off",
        "not a strong one",
        "i'd wait",
        "i would wait",
        "wait for a better",
        "wait for the next",
        "hold out for the next",
        "hold out for next",
        "pass on this",
        "passing on this",
        "recommend passing",
        "not this round",
        "not this one",
        "pass this round",
        # 36ff3131 / 183b29fc / 5babcd97: "I do not recommend" still
        # matched "recommend" and we bought the skip.
        "do not recommend",
        "don't recommend",
        "not recommend",
        "not recommended",
        "not prioritize",
    )
    if any(w in text for w in skip_needles):
        return False
    if text in ("no", "pass", "skip"):
        return False
    buy_needles = (
        "recommend",
        "high quality",
        "high-grade",
        "top-tier",
        "high-yield",
        "worth",
        "excellent",
        "easy call",
        "expected return",
        "limited edition",
        "genuinely good",
        "strong unit",
        "genuinely strong",
        "i'd buy",
        "i would buy",
        "buy it",
        "buy this",
        "buy signal",
        "winner",
        "top tier",
        "stake my reputation",
        "green light",
        "real thing",
        "take it",
    )
    if text in ("yes", "buy") or any(w in text for w in buy_needles):
        return True
    return None


def _history_rows(state: dict) -> list[dict]:
    rows = []
    for row in state.get("history") or []:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _bought(row: dict) -> bool:
    return bool(row.get("bought") or str(row.get("buyer_decision") or "").lower() in ("yes", "buy"))


def likelihoods(state: dict) -> tuple[float, float, bool, bool]:
    """Return (P(yes|high), P(yes|low), saw_bought_high, saw_sold_lemon)."""
    pyh, pyl = P_YES_HIGH0, P_YES_LOW0
    saw_high = saw_lemon = False
    for row in _history_rows(state):
        quality = str(row.get("quality") or "")
        if not _bought(row) or quality not in ("high", "low"):
            continue
        rec = seller_said_buy(row.get("seller_message"))
        if quality == "high":
            saw_high = True
            if rec is True:
                pyh = min(0.95, (pyh * 4.0 + 1.0) / 5.0)
        else:
            saw_lemon = True
            if rec is True:
                pyl = min(0.95, pyl + 0.40)
    return pyh, pyl, saw_high, saw_lemon


def seller_type(state: dict) -> str:
    _pyh, _pyl, saw_high, saw_lemon = likelihoods(state)
    if saw_lemon:
        return "liar"
    if saw_high:
        return "honest"
    said_yes_high = said_no_low = 0
    for row in _history_rows(state):
        quality = str(row.get("quality") or "")
        if not _bought(row) or quality not in ("high", "low"):
            continue
        rec = seller_said_buy(row.get("seller_message"))
        if rec is True and quality == "high":
            said_yes_high += 1
        elif rec is False and quality == "low":
            said_no_low += 1
    if said_yes_high > 0 or said_no_low > 0:
        return "honest"
    return "unknown"


def _quality_prior(state: dict) -> float:
    """Each round's quality remains iid with the public probability p."""
    prior = state.get("p")
    return float(prior if prior is not None else 0.5)


def posterior(state: dict) -> tuple[float, str]:
    prior = _quality_prior(state)
    seller = seller_type(state)
    pyh, pyl, saw_high, saw_lemon = likelihoods(state)
    rec = current_rec(state)
    if saw_lemon:
        return prior, "liar"
    # No cheap-talk update until at least one bought high (talk is then costly).
    if rec is True and saw_high:
        den = pyh * prior + pyl * (1.0 - prior)
        if den > 1e-12:
            prior = (pyh * prior) / den
        return min(0.95, max(0.05, prior)), seller
    if rec is False:
        p_no_h = 1.0 - pyh
        p_no_l = 1.0 - pyl
        den = p_no_h * prior + p_no_l * (1.0 - prior)
        if den > 1e-12:
            prior = (p_no_h * prior) / den
        return min(0.95, max(0.05, prior)), seller
    return prior, seller


def current_rec(state: dict) -> bool | None:
    rec = seller_said_buy(state.get("seller_message"))
    if rec is None:
        rec = seller_said_buy(state.get("seller_message_type"))
    return rec


def repeated_message_is_low(state: dict) -> bool:
    """Trust an exact text label only after both sides of the label map are observed."""
    if not state.get("is_seller_know_cv") or state.get("seller_message_type") != "text":
        return False
    current = " ".join(str(state.get("seller_message") or "").lower().split())
    if not current:
        return False
    observed: dict[str, list[int]] = {}
    for row in _history_rows(state):
        if not _bought(row):
            continue
        message = " ".join(str(row.get("seller_message") or "").lower().split())
        quality = str(row.get("quality") or "")
        if not message or quality not in ("high", "low"):
            continue
        counts = observed.setdefault(message, [0, 0])
        counts[quality == "high"] += 1
    low, high = observed.get(current, (0, 0))
    rounds = [
        int(row.get("round"))
        for row in _history_rows(state)
        if row.get("round") is not None
        and " ".join(str(row.get("seller_message") or "").lower().split()) == current
    ]
    current_round = int(state.get("round") or 0)
    if current_round and (not rounds or rounds[-1] != current_round):
        rounds.append(current_round)
    gaps = [right - left for left, right in zip(rounds, rounds[1:])]
    if len(gaps) >= 3 and len(set(gaps[-3:])) == 1:
        return False
    return bool(
        low >= 3
        and high == 0
        and any(
            message != current and counts[1] and not counts[0]
            for message, counts in observed.items()
        )
    )


def _seller_has_skipped(state: dict) -> bool:
    return seller_skip_count(state) > 0


def seller_skip_count(state: dict) -> int:
    return sum(
        seller_said_buy(row.get("seller_message")) is False
        for row in _history_rows(state)
    )


def seller_has_skipped(state: dict) -> bool:
    return _seller_has_skipped(state)


def signal_response(state: dict) -> dict[str, float | int]:
    """Observed buyer response rates to yes vs skip signals."""
    yes_n = yes_buy = no_n = no_buy = 0
    for row in _history_rows(state):
        rec = seller_said_buy(row.get("seller_message"))
        if rec is None:
            continue
        bought = _bought(row)
        if rec:
            yes_n += 1
            yes_buy += bought
        else:
            no_n += 1
            no_buy += bought
    return {
        "yes_n": yes_n,
        "yes_buy": yes_buy,
        "yes_rate": yes_buy / yes_n if yes_n else 0.0,
        "no_n": no_n,
        "no_buy": no_buy,
        "no_rate": no_buy / no_n if no_n else 0.0,
    }


def revealed_signal_stats(state: dict) -> dict[str, float | int]:
    """Signal accuracy from every quality revealed after a completed round."""
    samples = correct = high_n = high_yes = low_n = low_yes = 0
    for row in _history_rows(state):
        quality = str(row.get("quality") or "").lower()
        rec = seller_said_buy(row.get("seller_message"))
        if quality not in ("high", "low") or rec is None:
            continue
        samples += 1
        correct += (quality == "high" and rec) or (quality == "low" and not rec)
        if quality == "high":
            high_n += 1
            high_yes += rec
        else:
            low_n += 1
            low_yes += rec
    return {
        "samples": samples,
        "accuracy": correct / samples if samples else 0.5,
        "high_yes_rate": high_yes / high_n if high_n else 0.5,
        "low_yes_rate": low_yes / low_n if low_n else 0.5,
    }


def _response_prior_key(state: dict) -> str:
    price = float(state.get("product_price") or 0.0)
    high = float(state.get("v") or 0.0)
    low = float(state.get("u") or 0.0)
    values = (
        float(state.get("p") or 0.5),
        high / price if price else 0.0,
        low / price if price else 0.0,
        bool(state.get("is_seller_know_cv")),
        str(state.get("seller_message_type") or "text").lower(),
        bool(state.get("is_myopic", False)),
    )
    return "|".join(str(value) for value in values)


def buy_probability(state: dict, recommendation: bool) -> tuple[float, float]:
    """Posterior buyer response probability and standard deviation."""
    response = signal_response(state)
    signal = "yes" if recommendation else "no"
    prior_mean, strength = fitted_prior(
        "persuasion_buy", _response_prior_key(state), signal
    )
    if recommendation:
        samples = int(response["yes_n"])
        bought = int(response["yes_buy"])
    else:
        samples = int(response["no_n"])
        bought = int(response["no_buy"])
    alpha = prior_mean * strength + bought
    total = strength + samples
    mean = alpha / total if total > 0 else prior_mean
    variance = mean * (1.0 - mean) / max(total + 1.0, 1.0)
    return mean, math.sqrt(max(0.0, variance))


def state_of(state: dict, *, role: str) -> OpponentState:
    """Summarize the visible seller signal or buyer response process."""
    response = signal_response(state)
    revealed = revealed_signal_stats(state)
    bought_high = bought_low = correct = misleading = observed = 0
    for row in _history_rows(state):
        if not _bought(row):
            continue
        quality = str(row.get("quality") or "")
        rec = seller_said_buy(row.get("seller_message"))
        if quality not in ("high", "low") or rec is None:
            continue
        observed += 1
        bought_high += quality == "high"
        bought_low += quality == "low"
        correct += (rec and quality == "high") or (not rec and quality == "low")
        misleading += rec and quality == "low"
    yes_n = int(response["yes_n"])
    no_n = int(response["no_n"])
    yes_rate = float(response["yes_rate"])
    no_rate = float(response["no_rate"])
    if yes_n + no_n < 2:
        buyer_style = "unknown"
    elif yes_rate >= 0.8 and no_rate <= 0.2:
        buyer_style = "follows_signal"
    elif no_rate >= yes_rate + 0.25:
        buyer_style = "inverse"
    elif yes_rate <= 0.1 and no_rate <= 0.1:
        buyer_style = "rejects_all"
    elif yes_rate >= 0.9 and no_rate >= 0.9:
        buyer_style = "buys_all"
    else:
        buyer_style = "mixed"
    post, kind = posterior(state)
    predicted_yes, _ = buy_probability(state, True)
    predicted_no, _ = buy_probability(state, False)
    return OpponentState(
        role=role,
        samples=len(_history_rows(state)),
        seller_type=kind,
        current_recommendation=current_rec(state),
        current_high_posterior=post,
        seller_reliability=(correct + 1.0) / (observed + 2.0),
        seller_adversarial_probability=(misleading + 1.0) / (observed + 2.0),
        bought_high=bought_high,
        bought_low=bought_low,
        skipped_signals=seller_skip_count(state),
        buyer_style=buyer_style,
        yes_samples=yes_n,
        yes_buy_rate=yes_rate,
        no_samples=no_n,
        no_buy_rate=no_rate,
        predicted_yes_buy_rate=predicted_yes,
        predicted_no_buy_rate=predicted_no,
        signal_response_gap=yes_rate - no_rate,
        evidence_strength=min(1.0, observed / 6.0),
        revealed_samples=int(revealed["samples"]),
        revealed_signal_accuracy=float(revealed["accuracy"]),
        revealed_yes_high_rate=float(revealed["high_yes_rate"]),
        revealed_yes_low_rate=float(revealed["low_yes_rate"]),
    )


def buyer_prefers_skip_signal(state: dict) -> bool:
    response = signal_response(state)
    return bool(
        response["no_n"] >= 2
        and response["no_rate"] >= response["yes_rate"] + 0.25
    )


def seller_should_probe_skip(state: dict) -> bool:
    """Try at most two no-signals after three rejected yes-signals."""
    if signal_response(state)["no_n"] >= 2:
        return False
    rejected_yes = 0
    for row in reversed(_history_rows(state)):
        if seller_said_buy(row.get("seller_message")) is not True or _bought(row):
            break
        rejected_yes += 1
    return rejected_yes >= 3


def buyer_prior_ev_needs_signal(state: dict) -> bool:
    """Whether buying without an informative signal lacks strictly positive EV."""
    values = [state.get(key) for key in ("p", "v", "u", "product_price")]
    if any(value is None for value in values):
        return False
    prior, high, low, price = (float(value) for value in values)
    return prior * high + (1.0 - prior) * low <= price + 1e-9


def _ev_meets(expected: float, price: float) -> bool:
    """Peak genome (2026-08-22, persuasion 1875): untested_ev_ratio 1.0 → EV≥price."""
    ratio = float(current()["persuasion"].get("untested_ev_ratio") or 1.0)
    return expected + 1e-9 >= ratio * price


def _ev_covers(expected: float, price: float) -> bool:
    return _ev_meets(expected, price)


def _lemon_ev_decision(state: dict, expected: float, price: float) -> dict[str, str]:
    if expected > price + 1e-9:
        return {"decision": "yes"}
    if expected < price - 1e-9:
        return {"decision": "no"}
    bought_lows = sum(
        _bought(row) and str(row.get("quality") or "") == "low"
        for row in _history_rows(state)
    )
    # fad3a742 / d80fa98a: an uninformed-value text seller kept recommending
    # through a bad zero-EV tail. Three observed lemons is the historical
    # downside boundary; binary and known-value cells did not support it.
    if (
        state.get("is_seller_know_cv") is False
        and str(state.get("seller_message_type") or "").lower() == "text"
        and bought_lows >= 3
    ):
        return {"decision": "no"}
    realized = 0.0
    high = float(state.get("v") or 0.0)
    low = float(state.get("u") or 0.0)
    for row in _history_rows(state):
        if not _bought(row):
            continue
        payoff = row.get("buyer_payoff")
        if isinstance(payoff, (int, float)):
            realized += float(payoff)
        elif str(row.get("quality") or "") == "high":
            realized += high - price
        elif str(row.get("quality") or "") == "low":
            realized += low - price
    return {"decision": "yes" if realized < -1e-9 else "no"}


def ev_buy(state: dict, follow_signal: bool) -> dict[str, str]:
    """Skip is a no. After a lemon, buy iff posterior EV > price."""
    _pyh, _pyl, _saw_high, saw_lemon = likelihoods(state)
    known = bool(state.get("is_seller_know_cv"))
    prior, _seller = posterior(state)
    v = float(state.get("v") or 0.0)
    u = float(state.get("u") or 0.0)
    price = float(state.get("product_price") or 0.0)
    expected = prior * v + (1 - prior) * u
    rec = current_rec(state)
    prior_p = float(state.get("p") or 0.5)
    # bae50922 / 9d41cc69: a "no" is a skip. Do not buy on raw EV.
    if rec is False:
        return {"decision": "no"}
    # 88ecd02a: skip then yes with p=1/3, v>price still bought a lemon (−1e6).
    if rec is True and known and _seller_has_skipped(state) and v > price and prior_p >= 0.75:
        return {"decision": "yes"}
    if saw_lemon:
        return _lemon_ev_decision(state, expected, price)
    if rec is not False and _ev_meets(expected, price):
        return {"decision": "yes"}
    _ = follow_signal
    return {"decision": "no"}



def compose_buy(view, regime: str | None = None) -> str:
    """Buyer tree: skip → last → lemon_ev → type. One node returns."""
    from glee_agent.workflows.persuasion.classify import classify, p_band

    state = view.state
    regime = regime or classify(view)
    rec = current_rec(state)
    _pyh, _pyl, _saw_high, saw_lemon = likelihoods(state)
    skipped = seller_has_skipped(state)
    last = bool(getattr(view, "is_last_round", False))
    band = p_band(view)
    # Skip beats lemon: a "no" after a recommended low is still a skip.
    if rec is False:
        return "skip_block"
    if repeated_message_is_low(state):
        return "learned_low_text"
    # Known seller + recommended lemon: remaining quality is still iid.
    # Hard lemon_block stopped +EV rounds vs the field's ~96% always-yes
    # (18cda319 / 87f6f3ac / 2dcd4cd0). Buy iff posterior EV > price.
    if saw_lemon:
        return "lemon_ev"
    if last:
        return "last_raw_ev"
    return _buy_by_type(regime, band, skipped, rec)


def _buy_by_type(regime: str, band: str, skipped: bool, rec: bool | None) -> str:
    if regime == "buyer_known_untested":
        return "after_skip_trust" if skipped else "raw_ev"
    if regime == "buyer_p_high" or band == "high":
        if skipped and rec is True:
            return "after_skip_trust"
        return "ev_ratio_p_high"
    if regime == "buyer_liar":
        return "lemon_ev"
    if regime == "buyer_liar_known":
        return "lemon_ev"
    if regime == "buyer_honest":
        return "follow_honest"
    if skipped and rec is True:
        return "after_skip_trust"
    return "ev_ratio_092"


def execute_buy(view, procedure: str) -> dict:
    state = view.state
    prior, _seller = posterior(state)
    v = float(state.get("v") or 0.0)
    u = float(state.get("u") or 0.0)
    price = float(state.get("product_price") or 0.0)
    expected = prior * v + (1 - prior) * u
    rec = current_rec(state)
    if procedure == "lemon_block":
        return {"decision": "no"}
    if procedure == "skip_block":
        return {"decision": "no"}
    if procedure == "learned_low_text":
        return {"decision": "no"}
    if procedure == "raw_ev":
        return {"decision": "yes" if _ev_meets(expected, price) else "no"}
    if procedure == "last_raw_ev":
        return {"decision": "yes" if _ev_meets(expected, price) else "no"}
    if procedure == "after_skip_trust":
        if rec is False:
            return {"decision": "no"}
        _pyh, _pyl, saw_high, saw_lemon = likelihoods(state)
        if (
            not saw_high
            and not saw_lemon
            and v > price
            and expected < price
        ):
            from glee_agent.reference import persuasion as public_reference

            field = public_reference(view) or {}
            low_rate = float(field.get("median_low_buy_rate", 1.0))
            field_payoff = float(
                field.get("median_buyer_payoff_per_round_price") or 0.0
            )
            supported = (low_rate <= 0.35 and field_payoff > 0.0) or (
                state.get("is_seller_know_cv") is True
                and low_rate <= 0.05
                and field_payoff >= 0.0
            )
            if (
                float(field.get("median_high_buy_rate") or 0.0) >= 0.80
                and supported
            ):
                return {"decision": "yes"}
        # 71752dd3: seven skips did not make the next yes informative; buying
        # at 0.4x price EV revealed a lemon and scored -8.3. Only probe cells
        # where the public field separates high and low purchases.
        return {"decision": "yes" if _ev_meets(expected, price) else "no"}
    if procedure == "ev_ratio_p_high":
        return {"decision": "yes" if rec is not False and _ev_meets(expected, price) else "no"}
    if procedure == "ev_ratio_092":
        if rec is False:
            return {"decision": "no"}
        return {"decision": "yes" if _ev_meets(expected, price) else "no"}
    if procedure == "lemon_ev":
        return _lemon_ev_decision(state, expected, price)
    if procedure == "follow_honest":
        return ev_buy(state, follow_signal=True)
    return ev_buy(state, follow_signal=True)


def compose_sell(view, regime: str | None = None) -> str:
    from glee_agent.workflows.persuasion.classify import classify

    regime = regime or classify(view)
    # A blind seller cannot condition on quality, and historical probes only
    # reduced purchases: all-yes beat has-skip in every p/message cell with
    # repeated observations. Keep the executable route and diagnosis aligned.
    if regime == "seller_blind":
        return "always_yes"
    if buyer_prefers_skip_signal(view.state):
        return "inverse_no"
    if regime == "seller_last":
        return "last_yes"
    quality = str(view.state.get("current_quality") or view.state.get("quality") or "").lower()
    if quality == "low":
        if view.state.get("is_seller_know_cv") is False:
            if seller_has_skipped(view.state) or seller_should_probe_skip(view.state):
                return "skip_lows"
            return "always_yes"
        if view.state.get("is_seller_know_cv") is True and (
            seller_has_skipped(view.state) or seller_should_probe_skip(view.state)
        ):
            return "skip_lows"
        if buyer_prior_ev_needs_signal(view.state):
            return "skip_lows"
        if bool(current()["persuasion"].get("seller_always_yes")):
            return "always_yes"
        return "skip_lows"
    return "yes_highs"
