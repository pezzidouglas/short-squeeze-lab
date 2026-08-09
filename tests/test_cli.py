"""CLI smoke tests — every command runs offline and exits cleanly."""

from __future__ import annotations

import pytest

from squeezelab.cli import main


@pytest.mark.parametrize(
    "argv",
    [
        ["scan", "--top", "5"],
        ["scan", "--preset", "classic", "--top", "3"],
        ["scan", "--tickers", "GME,AMC"],
        ["sources"],
        ["sources", "--ticker", "GME"],
        ["compare", "GME", "SPRY"],
        ["presets", "--top", "5"],
        ["backtest", "--universe", "4", "--period", "6mo"],
        ["strategies", "--universe", "4", "--period", "6mo"],
    ],
)
def test_commands_exit_zero(argv, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_scan_marks_snapshot_data(capsys):
    main(["scan", "--top", "3"])
    out = capsys.readouterr().out
    assert "mode=snapshot" in out
    assert "sample snapshot" in out


def test_compare_needs_two_symbols(capsys):
    assert main(["compare", "GME"]) == 1


def test_export_writes_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SQUEEZE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("squeezelab.cli.SETTINGS.data_dir", str(tmp_path), raising=False)
    assert main(["export", "--format", "csv"]) == 0
    assert (tmp_path / "latest_scan.csv").exists()


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["nope"])
