"""Pick one persuasion regime. Seller info and buyer prior are separate."""

from __future__ import annotations

from glee_agent.rules import GameView

from ._beliefs import seller_has_skipped, seller_type


def classify(view: GameView) -> str:
    if view.action_type in ("seller_message", "seller_recommendation"):
        if view.is_last_round:
            return "seller_last"
        known = view.state.get("is_seller_know_cv")
        # is_seller_know_cv controls whether the seller sees buyer v/u. The
        # seller always sees this round's private product quality.
        quality = str(view.state.get("current_quality") or view.state.get("quality") or "").lower()
        if quality in ("high", "low") or bool(known):
            return "seller_honest"
        return "seller_blind"
    seller = seller_type(view.state)
    known = bool(view.state.get("is_seller_know_cv"))
    prior = float(view.state.get("p") or 0.5)
    if seller == "liar" and known:
        return "buyer_liar_known"
    if seller == "liar":
        return "buyer_liar"
    if known and seller == "unknown" and not seller_has_skipped(view.state):
        return "buyer_known_untested"
    if prior >= 0.75:
        return "buyer_p_high"
    if seller == "honest":
        return "buyer_honest"
    return "buyer_prior"



def p_band(view: GameView) -> str:
    prior = float(view.state.get("p") or 0.5)
    if prior >= 0.75:
        return "high"
    if prior <= 0.35:
        return "low"
    return "mid"


def lemon_seen(view: GameView) -> bool:
    from ._beliefs import likelihoods

    return bool(likelihoods(view.state)[3])


def skip_seen(view: GameView) -> bool:
    from ._beliefs import seller_has_skipped

    return seller_has_skipped(view.state)
