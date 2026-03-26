"""CLI entry point for sharp wallet discovery."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from sharp_discovery.config import APIConfig, DiscoveryConfig, ScoringConfig, SignalConfig
from sharp_discovery.db import Database
from sharp_discovery.scanner import WalletScanner
from sharp_discovery.signals import SignalEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Discover sharp wallets on Polymarket and generate market signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # ── scan subcommand ──────────────────────────────────────────────
    scan = sub.add_parser("scan", help="Discover and score sharp wallets")
    scan.add_argument("--top", type=int, default=50)
    scan.add_argument("--leaderboard-depth", type=int, default=500)
    scan.add_argument("--min-markets", type=int, default=20)
    scan.add_argument("--min-volume", type=float, default=10000.0)
    scan.add_argument("--min-hold-ratio", type=float, default=0.70)
    scan.add_argument("--min-total-volume", type=float, default=1000.0)
    scan.add_argument("--ci-confidence", type=float, default=0.90)
    scan.add_argument("--min-recency-days", type=int, default=0)
    scan.add_argument("--max-both-sides", type=float, default=0.50)
    scan.add_argument("--max-trades-per-market", type=float, default=20.0)
    scan.add_argument("--db", default="sharp_discovery.db")
    scan.add_argument("--json", action="store_true")

    # ── signals subcommand ───────────────────────────────────────────
    sig = sub.add_parser("signals", help="Generate smart-money signals for active markets")
    sig.add_argument("--db", default="sharp_discovery.db", help="Database with scored wallets")
    sig.add_argument("--top-wallets", type=int, default=50, help="Use top N wallets (default: 50)")
    sig.add_argument("--min-market-volume", type=float, default=50000.0, help="Min active market volume (default: 50000)")
    sig.add_argument("--max-markets", type=int, default=100, help="Max markets to analyze (default: 100)")
    sig.add_argument("--min-wallet-score", type=float, default=0.0, help="Min wallet score to include")
    sig.add_argument("--json", action="store_true")

    # ── full subcommand (scan + signals) ─────────────────────────────
    full = sub.add_parser("full", help="Run scan then generate signals")
    full.add_argument("--top", type=int, default=50)
    full.add_argument("--leaderboard-depth", type=int, default=2000)
    full.add_argument("--min-markets", type=int, default=20)
    full.add_argument("--min-volume", type=float, default=10000.0)
    full.add_argument("--min-hold-ratio", type=float, default=0.70)
    full.add_argument("--min-total-volume", type=float, default=1000.0)
    full.add_argument("--ci-confidence", type=float, default=0.90)
    full.add_argument("--min-recency-days", type=int, default=30)
    full.add_argument("--max-both-sides", type=float, default=0.50)
    full.add_argument("--max-trades-per-market", type=float, default=20.0)
    full.add_argument("--min-market-volume", type=float, default=50000.0)
    full.add_argument("--max-markets", type=int, default=100)
    full.add_argument("--db", default="sharp_discovery.db")
    full.add_argument("--json", action="store_true")

    args = p.parse_args()
    if not args.command:
        # Default to scan for backwards compat
        args.command = "scan"
        args = scan.parse_args(sys.argv[1:])
        args.command = "scan"
    return args


def _build_discovery_config(args: argparse.Namespace) -> DiscoveryConfig:
    scoring = ScoringConfig(
        min_resolved_markets=args.min_markets,
        min_hold_ratio=args.min_hold_ratio,
        min_total_volume=args.min_total_volume,
        ci_confidence=args.ci_confidence,
        max_both_sides_ratio=args.max_both_sides,
        max_trades_per_market=args.max_trades_per_market,
    )
    return DiscoveryConfig(
        api=APIConfig(),
        scoring=scoring,
        min_volume=args.min_volume,
        min_recency_days=args.min_recency_days,
        top_wallets=args.top,
        leaderboard_depth=args.leaderboard_depth,
        db_path=args.db,
    )


async def run_scan(args: argparse.Namespace) -> None:
    config = _build_discovery_config(args)

    async with Database(config.db_path) as db:
        scanner = WalletScanner(config, db)
        scores = await scanner.run()
        top = scores[: args.top]
        _print_wallets(top, config.db_path, args.json)


async def run_signals(args: argparse.Namespace) -> None:
    async with Database(args.db) as db:
        scores = await db.get_top_wallets(args.top_wallets)
        if not scores:
            print("No scored wallets found. Run 'scan' first.")
            return

    signal_config = SignalConfig(
        min_market_volume=args.min_market_volume,
        min_wallet_score=args.min_wallet_score,
        max_markets=args.max_markets,
    )
    engine = SignalEngine(signal_config, APIConfig())
    signals = await engine.run(scores)
    _print_signals(signals, args.json)


async def run_full(args: argparse.Namespace) -> None:
    config = _build_discovery_config(args)

    async with Database(config.db_path) as db:
        scanner = WalletScanner(config, db)
        scores = await scanner.run()
        top = scores[: args.top]
        _print_wallets(top, config.db_path, False)

    # Now generate signals from scored wallets
    signal_config = SignalConfig(
        min_market_volume=args.min_market_volume,
        max_markets=args.max_markets,
    )
    engine = SignalEngine(signal_config, APIConfig())
    signals = await engine.run(top)
    _print_signals(signals, args.json)


def _print_wallets(top: list, db_path: str, as_json: bool) -> None:
    if as_json:
        output = []
        for s in top:
            output.append({
                "address": s.address,
                "composite_score": round(s.composite_score, 4),
                "sharpe_ratio": round(s.sharpe_ratio, 4),
                "win_rate": round(s.win_rate, 4),
                "avg_roi": round(s.avg_roi, 4),
                "hold_ratio": round(s.hold_ratio, 4),
                "resolved_markets": s.resolved_market_count,
            })
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*90}")
        print(f"  Sharp Wallet Discovery — Top {len(top)} Wallets")
        print(f"{'='*90}\n")
        print(
            f"  {'#':>3}  {'Address':<44} {'Score':>7} "
            f"{'Win%':>5} {'ROI%':>6} {'Mkts':>5} {'Hold%':>6}"
        )
        print(f"  {'─'*3}  {'─'*44} {'─'*7} {'─'*5} {'─'*6} {'─'*5} {'─'*6}")

        for i, s in enumerate(top, 1):
            print(
                f"  {i:>3}  {s.address:<44} {s.composite_score:>7.3f} "
                f"{s.win_rate:>5.0%} {s.avg_roi:>5.1%} "
                f"{s.resolved_market_count:>5} {s.hold_ratio:>5.0%}"
            )

        print(f"\n  Saved to {db_path}")
        print()


def _print_signals(signals: list, as_json: bool) -> None:
    if as_json:
        output = []
        for sig in signals:
            output.append({
                "question": sig.question,
                "slug": sig.slug,
                "volume": round(sig.volume),
                "signals": {k: round(v, 3) for k, v in sig.signals.items()},
                "positions": [
                    {
                        "address": p.address,
                        "score": round(p.composite_score, 4),
                        "outcome": p.outcome,
                        "net_shares": round(p.net_shares, 1),
                        "avg_entry": round(p.avg_entry, 4),
                        "dollar_size": round(p.dollar_size, 2),
                    }
                    for p in sig.positions[:10]
                ],
            })
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*90}")
        print(f"  Smart Money Signals — {len(signals)} Markets")
        print(f"{'='*90}")

        for i, sig in enumerate(signals, 1):
            # Find dominant signal
            if sig.signals:
                best_side = max(sig.signals, key=lambda k: sig.signals[k])
                best_val = sig.signals[best_side]
            else:
                best_side, best_val = "?", 0

            n_wallets = len(set(p.address for p in sig.positions))
            print(f"\n  {i}. {sig.question[:75]}")
            print(f"     Volume: ${sig.volume:,.0f} | Wallets: {n_wallets}")

            # Print signal per outcome
            for outcome, strength in sorted(sig.signals.items(), key=lambda x: -x[1]):
                bar_len = min(int(abs(strength) * 2), 30)
                direction = "+" if strength > 0 else "-"
                bar = "█" * bar_len
                print(f"     {outcome:>6}: {direction}{abs(strength):.2f}  {bar}")

            # Top wallet positions
            shown = set()
            for p in sig.positions[:5]:
                if p.address in shown:
                    continue
                shown.add(p.address)
                direction = "LONG" if p.net_shares > 0 else "SHORT"
                print(
                    f"       {p.address[:12]}… "
                    f"score={p.composite_score:.2f} "
                    f"{direction} {p.outcome} "
                    f"{abs(p.net_shares):.0f} shares @ ${p.avg_entry:.2f}"
                )

        print()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "scan":
            asyncio.run(run_scan(args))
        elif args.command == "signals":
            asyncio.run(run_signals(args))
        elif args.command == "full":
            asyncio.run(run_full(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
