"""
Snapshot source — a bundled, offline sample of the record shape every live
source produces.

It exists so the dashboard, CLI and test suite work with no network at all, and
so a scan degrades to *something labelled sample data* rather than an empty
table when every provider is down. It never overrides a live value: it sits last
in every priority list, and the API flags `data_mode: "snapshot"` whenever it is
the only source that answered.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from .base import DataSource, SourceError, SourceMeta, SourceRecords

BUNDLED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "snapshot.json")


class SnapshotSource(DataSource):
    meta = SourceMeta(
        key="snapshot",
        name="Bundled snapshot",
        url="",
        cadence="static",
        provides=("*",),
        notes="Offline sample data. Never overrides a live source; flagged as sample in the UI.",
        requires_network=False,
    )

    def __init__(self, *args: Any, path: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.path = path or os.environ.get("SQUEEZE_SNAPSHOT_PATH") or BUNDLED_PATH
        self._payload: dict[str, Any] | None = None

    def available(self) -> bool:
        return os.path.exists(self.path)

    def payload(self) -> dict[str, Any]:
        if self._payload is None:
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self._payload = json.load(fh)
            except (OSError, ValueError) as exc:
                raise SourceError(f"could not read snapshot: {exc}") from exc
        return self._payload or {}

    @property
    def captured_at(self) -> str:
        return str(self.payload().get("captured_at") or "")

    def tickers(self) -> list[str]:
        return sorted((self.payload().get("records") or {}).keys())

    def _fetch(self, tickers: Sequence[str]) -> SourceRecords:
        records = self.payload().get("records") or {}
        if not records:
            raise SourceError("snapshot contains no records")
        wanted = {t.upper() for t in tickers}
        return {t: dict(v) for t, v in records.items() if t.upper() in wanted}

    def fetch_prices(self, ticker: str, period: str = "1y") -> Any | None:
        """Bars for the sample tickers, so the backtester runs offline too."""
        import pandas as pd

        bars = (self.payload().get("prices") or {}).get(ticker.upper())
        if not bars:
            return None
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(bars["dates"]),
                "Open": bars["open"],
                "High": bars["high"],
                "Low": bars["low"],
                "Close": bars["close"],
                "Volume": bars["volume"],
            }
        )
        frame["ticker"] = ticker.upper()
        days = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504}.get(period)
        if days:
            frame = frame.tail(days).reset_index(drop=True)
        return frame
