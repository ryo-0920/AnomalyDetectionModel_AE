"""Lazy timechart plotting CLI."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot time charts from anomaly CSV files.")
    parser.add_argument("--csv", help="CSV file, directory, or glob target.")
    parser.parse_known_args(argv)
    module = importlib.import_module("gofumi_ae.visualization.standard")
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + list(argv or original_argv[1:])
        module.main()
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
