#!/usr/bin/env python3
"""Compress the public GLEE game corpus into configuration-level medians."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


def _files(root: Path, *patterns: str) -> list[str]:
    return subprocess.check_output(
        ["git", "-C", str(root), "ls-files", *patterns], text=True
    ).splitlines()


def _horizon(value) -> str:
    return "inf" if value in (None, 99, "99") else str(value)


def _f(value: float) -> str:
    return f"{float(value):.6g}"


def _med(values: list[float]) -> float | None:
    return round(float(median(values)), 6) if values else None


def _read(root: Path, rel: str) -> tuple[dict, list[dict]]:
    game = root / rel
    config = json.loads(game.with_name("config.json").read_text())
    with game.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return config, rows


def _bargaining_one(root: Path, rel: str):
    try:
        config, rows = _read(root, rel)
        game = config["game_args"]
        alice = config["player_1_args"]
        bob = config["player_2_args"]
        money = float(game["money_to_divide"])
        d1, d2 = float(alice["delta"]), float(bob["delta"])
        decisions = [row for row in rows if row.get("decision")]
        last = decisions[-1] if decisions else None
        accepted = bool(last and str(last["decision"]).lower() == "accept")
        round_n = int(float(last["round"])) if last else 0
        first = next((row for row in rows if row.get("alice_gain")), None)
        first_alice = float(first["alice_gain"]) / money if first else None
        final_alice = alice_payoff = bob_payoff = 0.0
        if accepted:
            offer = rows[rows.index(last) - 1]
            final_alice = float(offer["alice_gain"]) / money
            alice_payoff = final_alice * d1 ** max(0, round_n - 1)
            bob_payoff = (1.0 - final_alice) * d2 ** max(0, round_n - 1)
        common = (_horizon(game.get("max_rounds")), bool(game.get("complete_information")))
        return (
            ("Alice", _f(d1), *common),
            (first_alice, accepted, round_n, alice_payoff, final_alice),
            ("Bob", _f(d2), *common),
            (first_alice, accepted, round_n, bob_payoff, final_alice),
        )
    except Exception:
        return None


def _negotiation_one(root: Path, rel: str):
    try:
        config, rows = _read(root, rel)
        game = config["game_args"]
        base = float(game["product_price_order"])
        seller_mult, buyer_mult = float(game["seller_value"]), float(game["buyer_value"])
        seller = seller_mult * base if seller_mult < 10 else seller_mult
        buyer = buyer_mult * base if buyer_mult < 10 else buyer_mult
        if buyer <= seller:
            return None
        decisions = [row for row in rows if row.get("decision")]
        last = decisions[-1] if decisions else None
        accepted = bool(last and last["decision"] == "AcceptOffer")
        round_n = int(float(last["round"])) if last else 0
        price = None
        if accepted:
            price = float(rows[rows.index(last) - 1]["product_price"])
        seller_payoff = (price - seller) / base if price is not None else 0.0
        buyer_payoff = (buyer - price) / base if price is not None else 0.0
        surplus = buyer - seller
        seller_share = (price - seller) / surplus if price is not None else None
        first = float(next(row["product_price"] for row in rows if row.get("product_price")))
        first_seller_claim = (first - seller) / surplus
        common = (_horizon(game.get("max_rounds")), bool(game.get("complete_information")))
        return (
            ("seller", _f(seller_mult), *common),
            (accepted, round_n, seller_payoff, seller_share, first / base, first_seller_claim),
            ("buyer", _f(buyer_mult), *common),
            (accepted, round_n, buyer_payoff, None if seller_share is None else 1 - seller_share, first / base, first_seller_claim),
        )
    except Exception:
        return None


def _persuasion_one(root: Path, rel: str):
    try:
        config, rows = _read(root, rel)
        game = config["game_args"]
        price = float(game["product_price"])
        rounds: dict[int, dict] = defaultdict(dict)
        for row in rows:
            round_n = int(float(row["round"]))
            if row.get("round_quality"):
                rounds[round_n]["high"] = "high" in row["round_quality"].lower()
                rounds[round_n]["worth"] = float(row.get("product_worth") or 0)
            if str(row.get("player") or "").lower() in ("the buyer", "bob") and row.get("decision"):
                rounds[round_n]["buy"] = str(row["decision"]).lower() == "yes"
        total = max(1, len(rounds))
        highs = [row for row in rounds.values() if row.get("high") is True]
        lows = [row for row in rounds.values() if row.get("high") is False]
        sales = [row for row in rounds.values() if row.get("buy")]
        buyer_payoff = sum((row.get("worth", 0.0) - price) for row in sales) / (price * total)
        key = (
            _f(game["p"]),
            _f(game["v"]),
            bool(game.get("is_seller_know_cv")),
            str(game.get("seller_message_type") or "text"),
            bool(game.get("is_myopic")),
        )
        return key, (
            len(sales) / total,
            buyer_payoff,
            sum(bool(row.get("buy")) for row in highs) / len(highs) if highs else 0.0,
            sum(bool(row.get("buy")) for row in lows) / len(lows) if lows else 0.0,
        )
    except Exception:
        return None


def _aggregate_bargaining(items) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for item in items:
        if item:
            groups[item[0]].append(item[1])
            groups[item[2]].append(item[3])
    out = {}
    for key, values in groups.items():
        deals = [value for value in values if value[1]]
        out["|".join(map(str, key))] = {
            "n": len(values),
            "deal_rate": round(len(deals) / len(values), 6),
            "median_round": _med([value[2] for value in deals]),
            "median_payoff_share": _med([value[3] for value in values]),
            "median_first_alice_share": _med([value[0] for value in values if value[0] is not None]),
            "median_final_alice_share": _med([value[4] for value in deals]),
        }
    return out


def _aggregate_negotiation(items) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for item in items:
        if item:
            groups[item[0]].append(item[1])
            groups[item[2]].append(item[3])
    out = {}
    for key, values in groups.items():
        deals = [value for value in values if value[0]]
        out["|".join(map(str, key))] = {
            "n": len(values),
            "deal_rate": round(len(deals) / len(values), 6),
            "median_round": _med([value[1] for value in deals]),
            "median_payoff_over_base": _med([value[2] for value in values]),
            "median_surplus_share": _med([value[3] for value in deals if value[3] is not None]),
            "median_first_price_over_base": _med([value[4] for value in values]),
            "median_first_seller_claim": _med([value[5] for value in values]),
        }
    return out


def _aggregate_persuasion(items) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for item in items:
        if item:
            groups[item[0]].append(item[1])
    out = {}
    for key, values in groups.items():
        out["|".join(map(str, key))] = {
            "n": len(values),
            "median_sale_rate": _med([value[0] for value in values]),
            "median_buyer_payoff_per_round_price": _med([value[1] for value in values]),
            "median_high_buy_rate": _med([value[2] for value in values]),
            "median_low_buy_rate": _med([value[3] for value in values]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glee-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("glee_agent/reference.json"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.glee_root.resolve()

    specs = {
        "bargaining": (
            _bargaining_one,
            _aggregate_bargaining,
            ("Data/bargaining_500_0204/**/game.csv", "Data/bragaining_2000_0304/**/game.csv"),
        ),
        "negotiation": (
            _negotiation_one,
            _aggregate_negotiation,
            ("Data/negotiation_500_0204/**/game.csv", "Data/negotiation_2000_0404/**/game.csv"),
        ),
        "persuasion": (
            _persuasion_one,
            _aggregate_persuasion,
            (
                "Data/persuasion_100_0204/**/game.csv",
                "Data/persuasion_400_0504/**/game.csv",
                "Data/persuasion_500_0904/**/game.csv",
            ),
        ),
    }
    output = {
        "source": "https://github.com/gleeframework/GLEE",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    for family, (reader, aggregate, patterns) in specs.items():
        paths = _files(root, *patterns)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            items = list(pool.map(lambda rel: reader(root, rel), paths, chunksize=40))
        output[family] = aggregate(items)
        output[f"{family}_games"] = len(paths)
        print(f"{family}: {len(paths)} games -> {len(output[family])} configurations")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
