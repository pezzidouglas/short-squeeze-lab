# Short Squeeze Lab

Short squeeze detection that shows its work: five data sources merged field by
field, a score that tells you how much of itself is actually supported by data,
head-to-head comparison, and a backtester that doesn't cheat.

```bash
pip install -r requirements.txt
python main.py scan            # rank the watchlist
python main.py serve           # dashboard on http://localhost:5000
```

No API keys are needed for the default source set. It runs offline out of the
box against a bundled sample dataset, clearly labelled as such everywhere it
appears.

---

## Why more than one source

Every free short-interest feed is a copy of the same twice-monthly exchange
file, reprinted at a different lag, against a different definition of float.
Ask three of them for GameStop's short interest and you get three answers. Most
tools pick one vendor and present the number as fact.

This one asks all of them, picks a winner per field by explicit priority, and
keeps the losers.

| Source | Cadence | What it is for | Key needed |
|---|---|---|---|
| **FINRA RegSHO** | daily | Off-exchange short volume — the only *unlagged* short signal here | no |
| **Yahoo Finance** | realtime | Broadest coverage: float, fundamentals, exchange short interest | no |
| **Stooq** | daily | Independent end-of-day bars; cross-checks Yahoo's price, backs the backtester | no |
| **SEC EDGAR** | quarterly | Filed share count off the 10-Q/10-K cover page — the tie-breaker on float | no* |
| **Finviz** | daily | Float short and days-to-cover for a whole watchlist in one request | `FINVIZ_AUTH_TOKEN` |
| **Snapshot** | static | Bundled offline sample so nothing breaks without a network | no |

<sub>* SEC asks automated clients to identify themselves — set `SEC_CONTACT_EMAIL`.</sub>

FINRA's consolidated short interest file is used too when
`FINRA_API_CLIENT_ID` / `FINRA_API_CLIENT_SECRET` are set. Without them the
source still supplies daily off-exchange short volume, which needs no
credentials.

**The off-exchange flow signal is the reason FINRA is here.** Exchange short
interest tells you where shorts were up to two weeks ago. FINRA publishes, every
trading day, what share of off-exchange volume was sold short. It answers a
different question — *are they still pressing?* — and it is free.

### What happens when sources disagree

`python main.py sources --ticker GME`, with three live sources enabled, prints
something shaped like this:

```
SOURCE COMPARISON · GME · confidence 84.2
     Short % float             won by finra     finra=22.4%  finviz=18.5%  yahoo=17.9%
  !! Float                     won by yahoo     yahoo=305.00M  finviz=262.00M
     Days to cover             won by finviz    finviz=3.20  yahoo=3.40
```

`!!` marks a gap wider than that field's tolerance. Every disagreement is
carried through to the API and the dashboard rather than resolved silently, and
it feeds the **data confidence** score:

```
confidence = 40% coverage      (how many scoring inputs exist at all)
           + 25% corroboration (how many sources reported each one)
           + 25% agreement     (how close they were)
           + 10% freshness     (how old the short-interest date is)
```

A single uncorroborated source scores 0.6 on agreement, not 1.0 — no second
opinion is not the same thing as being confirmed.

---

## The score

Five components, each 0–100, combined by a named weight preset.

| Component | What it measures |
|---|---|
| Short interest density | Short position as a share of float — the raw fuel |
| Borrow pressure | Days to cover and float scarcity — how hard it is to get out |
| Liquidity trap | Short position measured in days of *real* traded volume |
| Momentum fuel | Compression against the 52-week range, plus a growing short position |
| Off-exchange flow | FINRA daily short volume and its 5-day-vs-20-day trend |

```
classic      40 / 25 / 20 / 15 /  0    the original 1.x model, preserved
balanced     30 / 20 / 18 / 12 / 20    default
tight_float  30 / 30 / 28 /  7 /  5    small floats, hard borrows
momentum     22 / 13 / 15 / 35 / 15    already moving
flow         20 / 15 / 15 / 10 / 40    lean on the daily FINRA signal
```

Two things worth knowing:

**A component with no inputs is dropped, and the rest are renormalised.** It is
not scored zero. A missing short-interest feed used to quietly bury a name at
the bottom of the table; now it shrinks the evidence base, shows up in
`components_missing`, and costs data confidence. `evidence_weight` reports how
much of the model actually ran.

**`squeeze_score` and `confidence_adjusted_score` are both reported.** The first
answers *how squeezy does this look*; the second answers *how much of that do
the sources support*. Neither replaces the other.

Check whether a ranking is real or an artefact of the weights:

```bash
$ python main.py presets

TICKER        classic      balanced   tight_float      momentum          flow    SPREAD
SPRY             89.2          86.7          88.1          78.4          82.9      10.8
PSNY             71.0          58.3          52.7          79.2          55.1      26.5   ← weight-driven
```

---

## Comparison

```bash
python main.py compare GME SPRY PCT
```

```
METRIC                               GME        SPRY         PCT
Squeeze score                      63.46      *86.68       85.13
Short % float                      18.5%       34.6%      *36.3%
Days to cover                       3.20        8.20      *12.50
Off-exch short vol                 50.0%      *63.8%       59.7%
Float                            305.00M     *45.00M      85.00M

COMPONENT
Short interest density             64.10       92.14       93.67
Liquidity trap                     94.29      100.00      100.00

  SPRY leads at 86.68 — driven by liquidity trap (100.0), data confidence 73.3.
  * = leads this row (for squeeze potential, not investment merit)
```

The same view is in the dashboard: tick up to four rows and hit **Compare**.

---

## Backtesting

The 1.x backtester had a bug that made every result unreachable: it read a bar's
close, evaluated a signal derived from that same bar, and filled at that same
close. It also fed a **constant** squeeze score into the strategy, so the entry
rule reduced to "volume spike" and the score was never tested at all.

What runs now:

- **Signals are rebuilt per day** from price action, the trailing 52-week range,
  10-day volume, and FINRA's daily short-volume series.
- **Entries fill at the next bar's open.** No information the bar had not
  printed yet.
- **Holds count trading days**, not calendar days — `(exit - entry).days` used
  to count weekends, so a "5-day hold" was three sessions.
- **Take-profit and stop-loss are checked intrabar** against the high and low. A
  bar that touches both is assumed to stop out first.
- **Costs**: commission and slippage in basis points, both sides.
- **A real portfolio**: cash, position sizing, a cap on concurrent positions, and
  a dated equity curve — so drawdown, Sharpe and exposure mean something.
- **A benchmark**: equal-weight buy-and-hold of the same names over the same
  window, reported alongside as `excess_return_pct`.

```bash
python main.py backtest --universe 15 --entry-score 70 --take-profit 20 --stop-loss 10
python main.py strategies          # five exit-rule sets on identical data
```

### The limitation, stated plainly

Free short-interest **history** does not exist. Short interest is held at its
most recent value across the backtest window, so the short-interest and borrow
components are flat and only the price, volume and flow components vary. Every
reconstructed signal frame carries an `si_basis` column saying so, the backtest
summary repeats it, and the dashboard prints it above the results. Treat
backtests here as a way to test *exit rules*, not as evidence that the score
predicts anything.

Backtests also ignore borrow availability and hard-to-borrow fees — which, in
the names this tool surfaces, is exactly where the real cost lives.

---

## Dashboard

`python main.py serve` → <http://localhost:5000>

A dense terminal layout: dark by default, monospace numerics, ~25 rows visible
without scrolling. Light is one click away and remembered.

- Ranked table with sortable columns, search, and filters on risk, sector, score
  and confidence
- Click a row: component breakdown, key numbers, an off-exchange short-volume
  chart, which source supplied what, and the disagreements
- Tick up to four rows: side-by-side component bars and a metric table
- Source intelligence: per-source health, a field-provenance matrix, the widest
  disagreements in the scan
- Backtest with live parameters, equity vs benchmark, drawdown, an exit-rule
  comparison grid, and the trade list
- Light/dark, responsive, CSV export, keyboard shortcuts (`/` search, `c`
  compare, `r` refresh)

Charts are hand-rolled SVG — no CDN, no build step, works offline. The page
prefers the API and falls back to a payload embedded in the file, so the same
`static/index.html` can be deployed on its own:

```bash
python scripts/build_static.py      # embed a real scan + backtest into the page
```

A page built that way says **Sample snapshot** in its header when the data
underneath is the bundled fixture.

**Published to GitHub Pages** by `.github/workflows/pages.yml` on every push to
`main` and again each weekday after the US close, so the public page refreshes
itself. It needs one setting once: repository **Settings → Pages → Source:
"GitHub Actions"**. The workflow refuses to publish a build whose embedded
`data_mode` isn't `live`, so an outage leaves the previous page up rather than
replacing it with synthetic prices.

For a manual host instead (tiiny.site and similar), don't upload the
`static/index.html` committed here — it embeds the offline sample and labels
itself `STATIC`. The `build-static` CI job rebuilds it against live sources and
uploads it as the `dashboard-live` artifact, refusing to produce anything whose
embedded `data_mode` isn't `live`. Download that and upload it instead.

The deployed copy is a snapshot frozen at build time, not a live page: the
preset selector and the backtest parameter controls need the API, so they are
inert there. Run `python main.py serve` for the fully interactive version.

---

## CLI

```
python main.py scan        [--preset P] [--tickers A,B] [--top N] [--save]
python main.py sources     [--ticker T]        source health, or one ticker field by field
python main.py compare     A B [C ...]         head-to-head
python main.py presets                         does the ranking survive a change of weights?
python main.py backtest    [--universe N] [--period 1y] [--entry-score 70] …
python main.py strategies                      compare exit rule sets
python main.py serve       [--port 5000]
python main.py export      [--format csv|json]
```

## Python

```python
from squeezelab import MarketDataService, compare_tickers

result = MarketDataService().scan(preset="flow")
print(result.data_mode)                  # live | mixed | snapshot | empty

for row in result.rows[:5]:
    print(row["ticker"], row["squeeze_score"], row["data_confidence"], row["provenance"])

print(compare_tickers(result.rows, ["GME", "AMC"])["verdict"])
```

## HTTP API

| Endpoint | Returns |
|---|---|
| `GET /api/scan?preset=&tickers=&refresh=` | Ranked rows, weights, source health |
| `GET /api/sources` | Source health, catalog, provenance matrix, disagreements |
| `GET /api/ticker/<t>` | One row plus its full source breakdown |
| `GET /api/compare?tickers=A,B,C` | Head-to-head metrics, components, verdict |
| `GET /api/presets` | Score and rank under every weight preset |
| `GET /api/export.csv` | The current scan as CSV |
| `POST /api/backtest` | Trades, equity curve, benchmark, summary |
| `POST /api/strategies` | Several parameter sets over identical data |
| `GET /api/config`, `GET /api/health` | Presets, fields, source metadata, status |

Scan responses carry `data_mode` (`live` / `mixed` / `snapshot` / `empty`) and
`notes`. Nothing downstream has to guess whether it is looking at real data.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SQUEEZE_SOURCES` | `yahoo,stooq,finviz,finra,sec` | Which sources to run |
| `SQUEEZE_PRESET` | `balanced` | Default weight preset |
| `SQUEEZE_OFFLINE` | `0` | Skip every network source |
| `SQUEEZE_SNAPSHOT_FALLBACK` | `1` | Use the bundled sample when nothing answers |
| `SQUEEZE_CACHE_TTL` | `1800` | Disk cache for source responses, seconds |
| `SQUEEZE_SCAN_TTL` | `600` | In-process cache for whole scans, seconds |
| `SQUEEZE_DATA_DIR` | `data` | Output and cache directory |
| `SQUEEZE_MAX_WORKERS` | `6` | Source fan-out concurrency |
| `FINVIZ_AUTH_TOKEN` | — | Enables the Finviz source |
| `FINRA_API_CLIENT_ID` / `_SECRET` | — | Enables FINRA consolidated short interest |
| `SEC_CONTACT_EMAIL` | — | Contact address in the User-Agent SEC asks for |

A source that fails never kills a scan. It reports `error` / `empty` / `partial`
in its health block, the remaining sources carry the run, and the reason shows
up in the dashboard and the CLI.

## Layout

```
squeezelab/
  sources/        one module per provider, all behind one contract
  consolidate.py  field-level merge, provenance, agreement, confidence
  engine.py       five-component scoring, presets, ranking
  history.py      per-day signal reconstruction for backtests
  backtest.py     portfolio simulation, metrics, strategy comparison
  compare.py      ticker / source / preset comparison views
  service.py      source fan-out and the scan pipeline
  api.py          Flask API
  cli.py          command line
  data/           bundled offline snapshot
static/index.html the dashboard, self-contained
scripts/          snapshot + static-page builders
tests/            141 tests, no network
```

`data_fetcher.py`, `squeeze_engine.py` and `backtester.py` remain as thin
compatibility shims over the package.

## Tests

```bash
pip install pytest && python -m pytest
```

Every test runs offline against fixtures. The suite pins the behaviours that
were previously wrong — the same-bar fill, calendar-day holds, missing
components scoring zero — so they cannot come back.

## Regenerating the sample data

```bash
python scripts/build_snapshot.py    # deterministic, seeded per ticker
```

The snapshot is **not market data**. Its quote fields are derived from its own
generated bars so the table, the charts and the backtest all agree with each
other, and it exists so the real code paths can run without a network.

## License

MIT. **Not financial advice.** Short interest from free providers is stale by
construction. Treat every score as a prompt to look closer, never as a signal.
