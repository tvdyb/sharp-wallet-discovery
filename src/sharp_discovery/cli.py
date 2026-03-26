"""CLI entry point for sharp wallet discovery."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from sharp_discovery.config import APIConfig, DiscoveryConfig, ScoringConfig
from sharp_discovery.db import Database
from sharp_discovery.scanner import WalletScanner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Discover sharp wallets on Polymarket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Run full discovery with defaults (top 500 leaderboard wallets)
  sharp-discover

  # Deeper leaderboard scan
  sharp-discover --leaderboard-depth 1000

  # Show top 100 wallets, require 15+ markets
  sharp-discover --top 100 --min-markets 15

  # Output as JSON
  sharp-discover --json
""",
    )
    p.add_argument("--top", type=int, default=50, help="Number of top wallets to show (default: 50)")
    p.add_argument("--leaderboard-depth", type=int, default=500, help="How many leaderboard entries to pull (default: 500)")
    p.add_argument("--min-markets", type=int, default=20, help="Minimum resolved markets (default: 20)")
    p.add_argument("--min-volume", type=float, default=10000.0, help="Minimum market volume in USD (default: 10000)")
    p.add_argument("--min-hold-ratio", type=float, default=0.70, help="Minimum hold-to-expiration ratio (default: 0.70)")
    p.add_argument("--min-total-volume", type=float, default=1000.0, help="Minimum total USDC bought (default: 1000)")
    p.add_argument("--ci-confidence", type=float, default=0.90, help="Confidence level for Sharpe CI (default: 0.90)")
    p.add_argument("--min-recency-days", type=int, default=0, help="Only include wallets active within N days (0 = no filter)")
    p.add_argument("--max-both-sides", type=float, default=0.50, help="Max both-sides ratio to filter market makers (default: 0.50)")
    p.add_argument("--max-trades-per-market", type=float, default=20.0, help="Max avg trades/market to filter market makers (default: 20)")
    p.add_argument("--db", default="sharp_discovery.db", help="Database path (default: sharp_discovery.db)")
    p.add_argument("--json", action="store_true", help="Output results as JSON")
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    scoring = ScoringConfig(
        min_resolved_markets=args.min_markets,
        min_hold_ratio=args.min_hold_ratio,
        min_total_volume=args.min_total_volume,
        ci_confidence=args.ci_confidence,
        max_both_sides_ratio=args.max_both_sides,
        max_trades_per_market=args.max_trades_per_market,
    )
    config = DiscoveryConfig(
        api=APIConfig(),
        scoring=scoring,
        min_volume=args.min_volume,
        min_recency_days=args.min_recency_days,
        top_wallets=args.top,
        leaderboard_depth=args.leaderboard_depth,
        db_path=args.db,
    )

    async with Database(config.db_path) as db:
        scanner = WalletScanner(config, db)
        scores = await scanner.run()

        top = scores[: args.top]

        if args.json:
            output = []
            for s in top:
                output.append({
                    "address": s.address,
                    "composite_score": round(s.composite_score, 4),
                    "sharpe_ratio": round(s.sharpe_ratio, 4),
                    "sharpe_ci_lower": round(s.sharpe_ci_lower, 4),
                    "sharpe_ci_upper": round(s.sharpe_ci_upper, 4),
                    "win_rate": round(s.win_rate, 4),
                    "avg_roi": round(s.avg_roi, 4),
                    "hold_ratio": round(s.hold_ratio, 4),
                    "resolved_markets": s.resolved_market_count,
                    "extreme_price_ratio": round(s.extreme_price_ratio, 4),
                    "path": "sharpe" if s.sharpe_ratio != 0.0 else "consistency",
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

            print(f"\n  Saved to {config.db_path}")
            print()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
