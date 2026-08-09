"""API surface: envelopes, caching, validation and graceful degradation."""

from __future__ import annotations

import pytest

from squeezelab.api import _params_from, create_app
from squeezelab.backtest import StrategyParams


@pytest.fixture
def client(settings, service):
    app = create_app(service=service, settings=settings)
    app.config["TESTING"] = True
    return app.test_client()


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def test_health(client):
    payload = client.get("/api/health").get_json()
    assert payload["ok"] is True
    assert payload["offline"] is True


def test_config_lists_presets_sources_and_fields(client):
    payload = client.get("/api/config").get_json()
    assert "balanced" in payload["presets"]
    assert {s["key"] for s in payload["sources"]} >= {"yahoo", "finra", "sec", "stooq", "finviz", "snapshot"}
    assert payload["fields"]["short_pct_float"]["label"] == "Short % float"
    assert payload["strategy_variants"]


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------


def test_scan_returns_ranked_rows_and_flags_the_data_mode(client):
    payload = client.get("/api/scan").get_json()
    assert payload["ok"] is True
    assert payload["data_mode"] == "snapshot"
    assert payload["notes"], "a snapshot-only scan must say so"
    scores = [r["squeeze_score"] for r in payload["rows"]]
    assert scores == sorted(scores, reverse=True)
    assert "candidates" not in payload["rows"][0]      # trimmed from the table payload


def test_scan_respects_the_preset(client):
    balanced = client.get("/api/scan?preset=balanced").get_json()
    classic = client.get("/api/scan?preset=classic").get_json()
    assert classic["preset"] == "classic"
    assert balanced["rows"][0]["squeeze_score"] != classic["rows"][0]["squeeze_score"]


def test_unknown_preset_falls_back_rather_than_erroring(client):
    payload = client.get("/api/scan?preset=nonsense").get_json()
    assert payload["preset"] == "balanced"


def test_scan_can_be_narrowed_to_specific_tickers(client):
    payload = client.get("/api/scan?tickers=GME,AMC").get_json()
    assert {r["ticker"] for r in payload["rows"]} == {"GME", "AMC"}


def test_legacy_squeeze_data_endpoint_returns_a_bare_list(client):
    payload = client.get("/api/squeeze-data").get_json()
    assert isinstance(payload, list) and payload[0]["ticker"]


def test_export_csv_has_a_header_and_a_download_name(client):
    response = client.get("/api/export.csv")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    body = response.get_data(as_text=True)
    assert body.splitlines()[0].startswith("rank,ticker,name")
    assert len(body.splitlines()) > 2


# --------------------------------------------------------------------------
# Sources & detail
# --------------------------------------------------------------------------


def test_sources_reports_health_catalog_and_coverage(client):
    payload = client.get("/api/sources").get_json()
    assert payload["health"][0]["key"] == "snapshot"
    assert payload["coverage"]["tickers"] > 0
    assert isinstance(payload["disagreements"], list)


def test_ticker_detail_includes_the_source_breakdown(client):
    payload = client.get("/api/ticker/gme").get_json()
    assert payload["row"]["ticker"] == "GME"
    assert payload["sources"]["ticker"] == "GME"
    assert payload["sources"]["fields"]


def test_unknown_ticker_is_a_404_with_a_message(client):
    response = client.get("/api/ticker/ZZZZ")
    assert response.status_code == 404
    assert "not in the current scan" in response.get_json()["error"]


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------


def test_compare_two_tickers(client):
    payload = client.get("/api/compare?tickers=GME,SPRY").get_json()
    comparison = payload["comparison"]
    assert comparison["tickers"] == ["GME", "SPRY"]
    assert comparison["verdict"]["leader"] in {"GME", "SPRY"}
    assert set(comparison["sources"]) == {"GME", "SPRY"}


def test_compare_needs_at_least_two_tickers(client):
    response = client.get("/api/compare?tickers=GME")
    assert response.status_code == 400


def test_compare_rejects_too_many_tickers(client):
    response = client.get("/api/compare?tickers=A,B,C,D,E,F,G")
    assert response.status_code == 400


def test_presets_endpoint_returns_sensitivity(client):
    payload = client.get("/api/presets").get_json()
    assert payload["sensitivity"]["rows"][0]["spread"] >= 0


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------


def test_backtest_returns_trades_curves_and_a_summary(client):
    response = client.post("/api/backtest", json={"tickers": ["GME", "SPRY", "PCT"], "entry_score": 60})
    payload = response.get_json()
    assert response.status_code == 200
    result = payload["backtest"]
    assert result["equity_curve"] and result["benchmark_curve"]
    assert "total_return_pct" in result["summary"]
    assert result["params"]["entry_score"] == 60
    assert result["notes"], "the short-interest caveat must travel with the result"


def test_backtest_defaults_to_the_top_ranked_names(client):
    payload = client.post("/api/backtest", json={}).get_json()
    assert len(payload["tickers"]) > 1


def test_strategies_endpoint_compares_variants(client):
    payload = client.post("/api/strategies", json={"tickers": ["GME", "SPRY", "PCT"]}).get_json()
    labels = [row["label"] for row in payload["strategies"]]
    assert "baseline" in labels and len(labels) == 5


def test_strategies_accepts_custom_variants(client):
    body = {"tickers": ["GME", "SPRY"], "variants": [{"label": "mine", "entry_score": 50, "hold_days": 3}]}
    payload = client.post("/api/strategies", json=body).get_json()
    assert [r["label"] for r in payload["strategies"]] == ["mine"]


@pytest.mark.parametrize(
    "body",
    [
        {"entry_score": 500},
        {"stop_loss": 0.5},          # a stop must be negative
        {"hold_days": 0},
        {"position_pct": 5},
        {"take_profit": "banana"},
    ],
)
def test_backtest_rejects_out_of_range_parameters(client, body):
    response = client.post("/api/backtest", json=body)
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_params_from_ignores_unknown_keys():
    params, error = _params_from({"entry_score": 65, "not_a_field": 1})
    assert error is None
    assert params.entry_score == 65
    assert params == StrategyParams(entry_score=65)


# --------------------------------------------------------------------------
# Caching & static
# --------------------------------------------------------------------------


def test_repeated_scans_reuse_the_cached_result(client, service, monkeypatch):
    client.get("/api/scan")
    calls = {"n": 0}
    original = service.scan

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "scan", counted)
    client.get("/api/scan")
    client.get("/api/scan")
    assert calls["n"] == 0, "the second and third scans should come from cache"

    client.get("/api/scan?refresh=true")
    assert calls["n"] == 1


def test_index_serves_the_dashboard(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Short Squeeze Lab" in response.data
