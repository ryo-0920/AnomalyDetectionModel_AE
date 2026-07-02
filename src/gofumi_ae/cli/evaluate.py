"""Lazy evaluation CLI."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gofumi_ae score outputs.")
    parser.add_argument("--config", help="Shared evaluation config JSON for both ON and OFF sides.")
    parser.add_argument("--config_on", help="ON-side evaluation config JSON.")
    parser.add_argument("--config_off", help="OFF-side evaluation config JSON.")
    parser.add_argument("--on_dir", help="Directory containing ON-side anomaly CSV files.")
    parser.add_argument("--off_dir", help="Directory containing OFF-side anomaly CSV files.")
    parser.add_argument("--out_dir", help="Optional output directory.")
    parser.add_argument("--fpr_targets", help="Comma-separated FPR targets.")
    parser.add_argument("--score_col", help="Score column name.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose matching logs.")
    parser.parse_known_args(argv)
    module = importlib.import_module("gofumi_ae.evaluation.standard")
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + list(argv or original_argv[1:])
        module.main()
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
