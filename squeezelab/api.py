"""
Flask API behind the dashboard.

Every endpoint returns JSON with an envelope the front end can trust: `ok`, the
payload, and — for anything derived from a scan — `data_mode` so the UI can say
out loud when it is showing the bundled sample instead of live data.

Scans are cached per (preset, ticker set) with a TTL and a lock, because the
dashboard fires several endpoints on load and none of them should trigger a
second fan-out across every provider.
"""

from __future__ import annotations

import csv
import io
import os
import threading
import time
from typing import Any, Sequence

from flask import Flask, Response, jsonify, request, send_from_directory

from .backtest import DEFAULT_VARIANTS, SqueezeBacktester, StrategyParams, compare_strategies
from .compare import compare_presets, compare_sources, compare_tickers
from .config import (
    COMPONENT_LABELS,
    DEFAULT_WATCHLIST,
    FIELD_SPECS,
    SETTINGS,
    WEIGHT_PRESETS,
    Settings,
)
from .history import build_signals
from .service import MarketDataService, ScanResult
from .sources import SOURCE_REGISTRY, describe_sources

# The dashboard lives at the repository root, not inside the package, so this
# has to be absolute — a relative "static" resolves against squeezelab/ and 404s.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.environ.get("SQUEEZE_STATIC_DIR") or os.path.join(PROJECT_ROOT, "static")

MAX_COMPARE = 6
MAX_BACKTEST_TICKERS = 25


class ScanCache:
    """TTL cache around the scan fan-out, safe for Flask's threaded server."""

    def __init__(self, service: MarketDataService, ttl: int) -> None:
        self.service = service
        self.ttl = ttl
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, tuple[str, ...]], tuple[float, ScanResult]] = {}

    def get(
        self,
        preset: str,
        tickers: Sequence[str] | None = None,
        refresh: bool = False,
    ) -> ScanResult:
        key = (preset, tuple(sorted(t.upper() for t in tickers)) if tickers else ())
        with self._lock:
            hit = self._entries.get(key)
            if hit and not refresh and (time.time() - hit[0]) < self.ttl:
                return hit[1]
        result = self.service.scan(tickers=tickers or None, preset=preset)
        with self._lock:
            self._entries[key] = (time.time(), result)
        return result

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def create_app(
    service: MarketDataService | None = None,
    settings: Settings | None = None,
) -> Flask:
    settings = settings or SETTINGS
    service = service or MarketDataService(settings)
    cache = ScanCache(service, settings.scan_cache_ttl)

    # static_folder is None: the two routes below serve the dashboard, so
    # Flask's own static rule can't shadow them.
    app = Flask(__name__, static_folder=None)
    app.config["SQUEEZE_SERVICE"] = service
    app.config["SQUEEZE_CACHE"] = cache
    app.config["SQUEEZE_STATIC_DIR"] = STATIC_DIR

    # -- helpers ----------------------------------------------------------

    def _preset() -> str:
        preset = (request.args.get("preset") or settings.preset or "balanced").lower()
        return preset if preset in WEIGHT_PRESETS else "balanced"

    def _tickers() -> list[str] | None:
        raw = request.args.get("tickers") or ""
        parsed = [t.strip().upper() for t in raw.split(",") if t.strip()]
        return parsed or None

    def _scan() -> ScanResult:
        refresh = (request.args.get("refresh") or "").lower() in {"1", "true", "yes"}
        return cache.get(_preset(), _tickers(), refresh=refresh)

    def _envelope(result: ScanResult, payload: dict[str, Any]) -> Response:
        return jsonify(
            {
                "ok": True,
                "data_mode": result.data_mode,
                "generated_at": result.generated_at,
                "preset": result.preset,
                "notes": result.notes,
                **payload,
            }
        )

    def _fail(message: str, status: int = 400) -> Response:
        response = jsonify({"ok": False, "error": message})
        response.status_code = status
        return response

    # -- static -----------------------------------------------------------

    @app.route("/")
    def index() -> Response:
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/<path:filename>")
    def static_files(filename: str) -> Response:
        """Serve the rest of static/ (dashboard.html, any assets added later)."""
        return send_from_directory(STATIC_DIR, filename)

    # -- metadata ---------------------------------------------------------

    @app.route("/api/health")
    def health() -> Response:
        return jsonify(
            {
                "ok": True,
                "status": "ok",
                "version": _version(),
                "sources_enabled": list(settings.enabled_sources),
                "offline": settings.offline,
            }
        )

    @app.route("/api/config")
    def config() -> Response:
        return jsonify(
            {
                "ok": True,
                "presets": {
                    name: {"weights": weights, "components": COMPONENT_LABELS}
                    for name, weights in WEIGHT_PRESETS.items()
                },
                "default_preset": settings.preset,
                "watchlist": list(service.watchlist or DEFAULT_WATCHLIST),
                "fields": {
                    name: {"label": spec.label, "unit": spec.unit, "kind": spec.kind, "key": spec.key}
                    for name, spec in FIELD_SPECS.items()
                },
                "sources": describe_sources(list(SOURCE_REGISTRY)),
                "settings": settings.to_dict(),
                "strategy_variants": [p.to_dict() for p in DEFAULT_VARIANTS],
            }
        )

    # -- scan -------------------------------------------------------------

    @app.route("/api/scan")
    def scan() -> Response:
        result = _scan()
        return _envelope(
            result,
            {
                "rows": [_slim(row) for row in result.rows],
                "weights": result.weights,
                "tickers_requested": result.tickers_requested,
                "health": result.health,
            },
        )

    # Kept for compatibility with the 1.x dashboard.
    @app.route("/api/squeeze-data")
    def squeeze_data() -> Response:
        return jsonify([_slim(row) for row in _scan().rows])

    @app.route("/api/sources")
    def sources() -> Response:
        result = _scan()
        return _envelope(
            result,
            {
                "health": result.health,
                "catalog": describe_sources(list(SOURCE_REGISTRY)),
                "coverage": result.coverage,
                "disagreements": result.disagreements,
            },
        )

    @app.route("/api/ticker/<ticker>")
    def ticker_detail(ticker: str) -> Response:
        result = _scan()
        wanted = ticker.upper()
        row = next((r for r in result.rows if str(r.get("ticker", "")).upper() == wanted), None)
        if row is None:
            return _fail(f"{wanted} is not in the current scan", 404)
        return _envelope(result, {"row": row, "sources": compare_sources(row)})

    @app.route("/api/compare")
    def compare() -> Response:
        tickers = _tickers_from("tickers")
        if len(tickers) < 2:
            return _fail("pass at least two tickers, e.g. /api/compare?tickers=GME,AMC")
        if len(tickers) > MAX_COMPARE:
            return _fail(f"compare at most {MAX_COMPARE} tickers at once")
        result = cache.get(_preset())
        payload = compare_tickers(result.rows, tickers)
        payload["sources"] = {
            t: compare_sources(row)
            for t in payload["tickers"]
            for row in result.rows
            if str(row.get("ticker", "")).upper() == t
        }
        return _envelope(result, {"comparison": payload})

    @app.route("/api/presets")
    def presets() -> Response:
        result = cache.get(_preset())
        # Rescoring reuses the merged records from the cached scan; hitting
        # every provider again just to change the weights would be absurd.
        scan_obj = result.consolidated_scan
        if scan_obj is None:
            return _fail("no scan available to rescore", 503)
        return _envelope(result, {"sensitivity": compare_presets(scan_obj)})

    @app.route("/api/export.csv")
    def export_csv() -> Response:
        result = _scan()
        columns = [
            "rank", "ticker", "name", "sector", "squeeze_score", "confidence_adjusted_score",
            "data_confidence", "setup_label", "si_score", "borrow_score", "liquidity_score",
            "momentum_score", "flow_score", "short_pct_float", "short_ratio",
            "off_exchange_short_pct", "off_exchange_trend", "float", "avg_volume_10d",
            "current_price", "fifty_two_week_high", "fifty_two_week_low", "market_cap",
            "short_interest_date", "sources",
        ]
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(columns)
        for row in result.rows:
            writer.writerow([_csv_cell(row.get(column)) for column in columns])
        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=squeeze_scan.csv"},
        )

    # -- backtest ---------------------------------------------------------

    @app.route("/api/backtest", methods=["POST", "GET"])
    def backtest() -> Response:
        body = request.get_json(silent=True) or {}
        params, error = _params_from(body)
        if error:
            return _fail(error)
        period = str(body.get("period") or request.args.get("period") or "1y")

        result = cache.get(_preset())
        tickers = _pick_backtest_tickers(body, result)
        if not tickers:
            return _fail("no tickers available to backtest")

        prices = service.price_history(tickers, period=period)
        if not prices:
            return _fail("no price history available from any source", 503)

        records = {str(r["ticker"]).upper(): r for r in result.rows}
        signals = build_signals(records, prices)
        outcome = SqueezeBacktester(params).run(prices, signals)
        return _envelope(
            result,
            {
                "backtest": outcome.to_dict(),
                "tickers": sorted(prices.keys()),
                "period": period,
            },
        )

    @app.route("/api/strategies", methods=["POST", "GET"])
    def strategies() -> Response:
        body = request.get_json(silent=True) or {}
        period = str(body.get("period") or request.args.get("period") or "1y")
        result = cache.get(_preset())
        tickers = _pick_backtest_tickers(body, result)
        prices = service.price_history(tickers, period=period)
        if not prices:
            return _fail("no price history available from any source", 503)

        records = {str(r["ticker"]).upper(): r for r in result.rows}
        signals = build_signals(records, prices)

        variants = None
        if isinstance(body.get("variants"), list) and body["variants"]:
            variants = []
            for raw in body["variants"][:8]:
                params, error = _params_from(raw)
                if error:
                    return _fail(error)
                variants.append(params)

        rows = compare_strategies(prices, signals, variants)
        return _envelope(result, {"strategies": rows, "tickers": sorted(prices.keys()), "period": period})

    # -- request helpers --------------------------------------------------

    def _tickers_from(name: str) -> list[str]:
        raw = request.args.get(name) or ""
        return [t.strip().upper() for t in raw.split(",") if t.strip()][: MAX_COMPARE + 2]

    def _pick_backtest_tickers(body: dict[str, Any], result: ScanResult) -> list[str]:
        requested = body.get("tickers") or request.args.get("tickers")
        if isinstance(requested, str):
            requested = [t.strip() for t in requested.split(",") if t.strip()]
        if requested:
            return [str(t).upper() for t in requested][:MAX_BACKTEST_TICKERS]
        # Default to the names the scan actually ranked highest.
        return [str(r["ticker"]).upper() for r in result.rows[:MAX_BACKTEST_TICKERS]]

    return app


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def _params_from(raw: dict[str, Any]) -> tuple[StrategyParams, str | None]:
    """Build StrategyParams from untrusted JSON, with bounds."""
    fields = StrategyParams.__dataclass_fields__
    values: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if key not in fields or value is None:
            continue
        if key == "label":
            values[key] = str(value)[:40]
            continue
        try:
            values[key] = int(value) if key in {"hold_days", "max_positions", "cooldown_days"} else float(value)
        except (TypeError, ValueError):
            return StrategyParams(), f"{key} must be a number"

    bounds = {
        "entry_score": (0, 100),
        "volume_mult": (0, 20),
        "hold_days": (1, 250),
        "take_profit": (0.01, 10),
        "stop_loss": (-0.95, -0.001),
        "trailing_stop": (0.01, 0.95),
        "position_pct": (0.01, 1.0),
        "max_positions": (1, 50),
        "starting_capital": (100, 1e9),
        "commission_bps": (0, 500),
        "slippage_bps": (0, 500),
        "cooldown_days": (0, 60),
    }
    for key, (low, high) in bounds.items():
        if key in values and not (low <= values[key] <= high):
            return StrategyParams(), f"{key} must be between {low} and {high}"
    return StrategyParams(**values), None


def _slim(row: dict[str, Any]) -> dict[str, Any]:
    """Trim the per-row payload the table does not need (kept on /api/ticker)."""
    out = dict(row)
    out.pop("candidates", None)
    return out


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, dict):
        return ""
    return value


def _version() -> str:
    from . import __version__

    return __version__


app = None  # populated by `main.py serve` / WSGI entry points


def get_app() -> Flask:
    global app
    if app is None:
        app = create_app()
    return app
