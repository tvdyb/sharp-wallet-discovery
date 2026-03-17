# Sharp Wallet Discovery

Discover consistently profitable wallets on Polymarket by analyzing historical trade data.

## How it works

1. **Fetch** all resolved markets from Polymarket (filtered by minimum volume)
2. **Analyze** every wallet's trades: entry prices, hold-to-expiration behavior, win/loss outcomes
3. **Score** wallets via dual-path system:
   - **Sharpe path**: wallets with meaningful ROI variance are ranked by Sharpe ratio, gated by a confidence interval lower bound > 0
   - **Consistency path**: near-zero variance wallets (e.g., always buying at $0.999) need 85%+ win rate and are scored by `win_rate × log(markets)`
4. **Penalize** extreme-price entries: wallets that predominantly buy at ≥95¢ get their score discounted (configurable up to 50%)
5. **Output** a ranked leaderboard with full statistics

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run full discovery (fetches from Polymarket API)
sharp-discover

# Top 100 wallets, require 15+ markets
sharp-discover --top 100 --min-markets 15

# Stricter extreme-price penalty (80% discount for all-extreme wallets)
sharp-discover --extreme-penalty 0.8

# JSON output for piping to other tools
sharp-discover --json

# All options
sharp-discover --help
```

## Scoring details

### Sharpe path
For wallets with ROI standard deviation ≥ 0.001:
- Sharpe ratio = mean(ROI) / stdev(ROI), clamped to [-10, 10]
- CI computed via standard error with configurable confidence (default 90%)
- **Significance gate**: CI lower bound must be > 0
- **Composite score** = Sharpe × penalty_factor

### Consistency path
For wallets with near-zero ROI variance:
- Requires win rate ≥ 85% and minimum market count
- **Composite score** = win_rate × log(n_held) × penalty_factor

### Extreme price penalty
Wallets entering positions at ≥95¢ (configurable) get penalized:
```
penalty_factor = 1.0 - extreme_price_penalty × (fraction of held entries at ≥ threshold)
```
Default: 50% max penalty. A wallet with 100% extreme entries gets score × 0.5.

## Tests

```bash
pytest
```
