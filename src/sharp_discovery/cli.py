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
  # Run full discovery with defaults
  sharp-discover

  # Show top 100 wallets, require 15+ markets
  sharp-discover --top 100 --min-markets 15

  # Use stricter extreme-price penalty
  sharp-discover --extreme-penalty 0.8

  # Output as JSON
  sharp-discover --json
""",
    )
    p.add_argument("--top", type=int, default=50, help="Number of top wallets to show (default: 50)")
    p.add_argument("--min-markets", type=int, default=10, help="Minimum resolved markets (default: 10)")
    p.add_argument("--min-volume", type=float, default=10000.0, help="Minimum market volume in USD (default: 10000)")
    p.add_argument("--max-markets", type=int, default=2000, help="Max markets to scan (default: 2000)")
    p.add_argument("--min-hold-ratio", type=float, default=0.70, help="Minimum hold-to-expiration ratio (default: 0.70)")
    p.add_argument("--ci-confidence", type=float, default=0.90, help="Confidence level for Sharpe CI (default: 0.90)")
    p.add_argument("--extreme-threshold", type=float, default=0.95, help="Entry price considered extreme (default: 0.95)")
    p.add_argument("--extreme-penalty", type=float, default=0.50, help="Max penalty for 100%% extreme entries (default: 0.50)")
    p.add_argument("--db", default="sharp_discovery.db", help="Database path (default: sharp_discovery.db)")
    p.add_argument("--json", action="store_true", help="Output results as JSON")
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    scoring = ScoringConfig(
        min_resolved_markets=args.min_markets,
        min_hold_ratio=args.min_hold_ratio,
        ci_confidence=args.ci_confidence,
        extreme_price_threshold=args.extreme_threshold,
        extreme_price_penalty=args.extreme_penalty,
    )
    config = DiscoveryConfig(
        api=APIConfig(),
        scoring=scoring,
        min_volume=args.min_volume,
        max_markets=args.max_markets,
        top_wallets=args.top,
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
                f"  {'#':>3}  {'Address':<44} {'Score':>7} {'Sharpe':>7} "
                f"{'Win%':>5} {'ROI%':>6} {'Mkts':>5} {'Hold%':>6} {'Ext%':>5} {'Path':<6}"
            )
            print(f"  {'─'*3}  {'─'*44} {'─'*7} {'─'*7} {'─'*5} {'─'*6} {'─'*5} {'─'*6} {'─'*5} {'─'*6}")

            for i, s in enumerate(top, 1):
                path = "sharpe" if s.sharpe_ratio != 0.0 else "cons"
                print(
                    f"  {i:>3}  {s.address:<44} {s.composite_score:>7.3f} "
                    f"{s.sharpe_ratio:>7.3f} {s.win_rate:>5.0%} {s.avg_roi:>5.1%} "
                    f"{s.resolved_market_count:>5} {s.hold_ratio:>5.0%} "
                    f"{s.extreme_price_ratio:>5.0%} {path:<6}"
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
