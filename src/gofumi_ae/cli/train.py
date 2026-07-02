"""Lazy training CLI."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence

_VARIANTS = {
    "standard": "gofumi_ae.training.standard",
    "nstep": "gofumi_ae.training.nstep",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train gofumi_ae models.")
    parser.add_argument(
        "--variant",
        choices=sorted(_VARIANTS),
        default="standard",
        help="Training flow variant.",
    )
    args, remaining = parser.parse_known_args(argv)
    module = importlib.import_module(_VARIANTS[args.variant])
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]] + list(remaining)
        module.main()
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
