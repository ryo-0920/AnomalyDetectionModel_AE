"""Re-export from src.gofumi_ae.datasets.tagged_dataset for backward compatibility.

All implementation has been moved to src.gofumi_ae.datasets.tagged_dataset.
This module provides a stable import interface for existing code.
"""

from gofumi_ae.datasets.tagged_dataset import (
    TAGGED_SAMPLE_PERCENT_OPTIONS,
    build_tagged_dataset_csvs_from_config,
    is_tagged_dataset_ledger_path,
    resolve_tagged_dataset_ledger_selector,
    sample_paths_interactively,
)

__all__ = [
    "TAGGED_SAMPLE_PERCENT_OPTIONS",
    "build_tagged_dataset_csvs_from_config",
    "is_tagged_dataset_ledger_path",
    "resolve_tagged_dataset_ledger_selector",
    "sample_paths_interactively",
]
