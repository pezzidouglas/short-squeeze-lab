# 🎯 Short Squeeze Lab

Real-time short squeeze detection, scoring, and backtesting system for algorithmic trading pipelines.

![Dashboard Preview](static/dashboard-preview.png)

## Features

- **Squeeze Scanner**: Monitors 40+ heavily-shorted tickers with composite scoring
- **Composite Score**: Weights Short Interest (40%), Borrow Pressure (25%), Liquidity Trap (20%), Momentum Fuel (15%)
- **Interactive Dashboard**: Dark-mode web UI with sortable tables, charts, and filters
- **Backtesting Engine**: Simulates squeeze strategy with configurable entry/exit rules
- **Free Data Sources**: Yahoo Finance + SEC (no paid APIs required for basic use)
- **Pipeline Ready**: Designed for `backtest → paper → live` workflow via TradersPost + Alpaca

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run squeeze scan
python main.py scan

# 3. Start dashboard
python main.py serve
# Then open http://localhost:5000

# 4. Run backtest
python main.py backtest
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Yahoo Finance  │────▶│  Data Fetcher   │────▶│ Squeeze Engine  │
│  (price, SI)    │     │  (data_fetcher) │     │ (squeeze_engine)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                              ┌─────────────────────────┘
                              ▼
                       ┌─────────────────┐
                       │  Flask Server   │◄──── Browser
                       │    (app.py)     │      Dashboard
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Backtester    │
                       │ (backtester.py) │
                       └─────────────────┘
```

## Squeeze Score Formula

| Component | Weight | Input |
|-----------|--------|-------|
| Short Interest Density | 40% | `% of float shorted` |
| Borrow Cost Pressure | 25% | `Days to cover + float scarcity` |
| Liquidity Trap | 20% | `Short shares / avg daily volume` |
| Momentum Fuel | 15% | `Price compression + SI increase` |

**Score Interpretation:**
- `≥ 80` — 🔥 HIGH SQUEEZE RISK
- `60-79` — ⚠️ ELEVATED
- `40-59` — 📊 WATCH
- `< 40` — ✅ LOW

## Backtest Strategy

- **Entry**: Squeeze score ≥ 70 + volume > 1.5x 20-day average
- **Hold**: Max 5 trading days
- **Take Profit**: +20%
- **Stop Loss**: -10%
- **Position**: Equal weight per signal

## Deployment

### GitHub
```bash
git init
git add .
git commit -m "Initial squeeze lab"
git remote add origin https://github.com/pezzidouglas/short-squeeze-lab.git
git push -u origin main
```

### Tiiny Site
1. Zip the `static/` folder (dashboard is self-contained)
2. Upload to [tiiny.site](https://tiiny.site)
3. Or use Tiiny CLI: `tiiny deploy static/`

### Live Data API
For production, swap Yahoo Finance with:
- **Ortex API** — Real-time SI estimates ($49/mo)
- **FinancialModelingPrep** — Short interest API ($79/mo)
- **S3 Partners** — Institutional-grade ($$$)

## Watchlist

Default watchlist covers meme stocks, biotech shorts, EV shorts, and high-SI names:
`SPRY, PCT, INDI, SLS, RUM, GME, AMC, NVAX, BYND, CVNA, UPST, TSLA, PLTR, SOFI, NIO, ASTS, RKLB, SOUN, SMCI, MSTR...`

Edit `DEFAULT_WATCHLIST` in `data_fetcher.py` to customize.

## License

MIT — Built for the algo trading community.
