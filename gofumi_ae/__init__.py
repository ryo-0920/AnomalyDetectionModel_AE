"""Compatibility shim for the real ``src/gofumi_ae`` package."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_SRC_PACKAGE_DIR = _PACKAGE_DIR.parent / "src" / "gofumi_ae"

__path__ = [str(_SRC_PACKAGE_DIR), str(_PACKAGE_DIR)]
