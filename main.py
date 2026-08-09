#!/usr/bin/env python3
"""
Short Squeeze Lab — CLI entry point.

    python main.py scan
    python main.py sources --ticker GME
    python main.py compare GME AMC SPRY
    python main.py backtest
    python main.py strategies
    python main.py presets
    python main.py serve

Run `python main.py <command> --help` for the options on each.
"""

import sys

from squeezelab.cli import main

if __name__ == "__main__":
    sys.exit(main())
