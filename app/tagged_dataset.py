import glob
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from app.ui.prompts import choose_from_list

TAGGED_SAMPLE_PERCENT_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _normalize_header_name(value: Any) -> str:
    text = str(value).strip().lower()
    for token in (" ", "\u3000", "\n", "\r", "\t", "_"):
        text = text.replace(token, "")
    text = text.replace("（", "(").replace("）", ")")
    return text


def _normalize_file_stem(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ""
    text = text.replace("/", os.sep).replace("\\", os.sep)
    base = os.path.basename(text).strip()
    if not base:
        return ""
    stem, _ = os.path.splitext(base)
    stem = stem.strip().lower()
    # ★ 先頭の 0 を無視する（台帳: 27763, ファイル: 027763 → 両方 27763 に揃える）
    stem = stem.lstrip("0")
    if not stem:
        stem = "0"
    return stem


def _excel_col_to_index(col: str) -> int:
    label = str(col).strip().upper()
    if not label or not label.isalpha():
        raise ValueError(f"Invalid Excel column label: {col}")
    idx = 0
    for ch in label:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _canonical_filter_value(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return f"text:{str(value).strip().lower()}"
    if isinstance(value, (int, float)):
        num = float(value)
        if not math.isfinite(num):
            return None
        rounded = int(round(num))
        if abs(num - rounded) < 1e-9:
            return f"num:{rounded}"
        return f"num:{format(num, '.12g')}"
    text = str(value).strip()
    if text == "":
        return None
    try:
        num = float(text)
        if math.isfinite(num):
            rounded = int(round(num))
            if abs(num - rounded) < 1e-9:
                return f"num:{rounded}"
            return f"num:{format(num, '.12g')}"
    except Exception:
        pass
    return f"text:{text.lower()}"


def _matches_filter(cell_value: Any, mode: str, values: Sequence[Any]) -> bool:
    allowed = {
        token
        for token in (_canonical_filter_value(v) for v in values)
        if token is not None
    }
    # Empty values means "no-op" to avoid accidental full-drop.
    if not allowed:
        return True
    cell_token = _canonical_filter_value(cell_value)
    is_match = cell_token in allowed
    if mode == "include":
        return is_match
    if mode == "exclude":
        return not is_match
    raise ValueError(f"Invalid filter mode: {mode}")


def _find_file_name_column(columns: Sequence[Any], expected: str) -> Optional[str]:
    if expected in columns:
        return str(expected)
    expected_norm = _normalize_header_name(expected)
    normalized = {str(col): _normalize_header_name(col) for col in columns}
    for original, col_norm in normalized.items():
        if col_norm == expected_norm:
            return original
    # Fallback for this project's ledger variants.
    token_file = _normalize_header_name("\u30d5\u30a1\u30a4\u30eb\u540d")
    token_ts = _normalize_header_name("\u6642\u7cfb\u5217")
    for original, col_norm in normalized.items():
        if ("ttdc" in col_norm and token_file in col_norm) or (token_ts in col_norm and "csv" in col_norm):
            return original
    return None


def _load_tagged_filter_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Tagged filter config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Tagged filter config must be a JSON object.")
    if not bool(cfg.get("enabled", True)):
        raise ValueError(f"Tagged filter config is disabled: {config_path}")
    return cfg


def build_tagged_dataset_csvs_from_config(config_path: str, pattern: str = "*.csv") -> Tuple[List[str], Dict[str, Any]]:
    cfg = _load_tagged_filter_config(config_path)

    ledger_cfg = cfg.get("ledger", {})
    if not isinstance(ledger_cfg, dict):
        raise ValueError("Invalid config: 'ledger' must be an object.")
    ledger_path = os.path.normpath(str(ledger_cfg.get("path", "")).strip())
    if not ledger_path:
        raise ValueError("Invalid config: ledger.path is required.")
    if not os.path.isfile(ledger_path):
        raise FileNotFoundError(f"Tagged dataset ledger not found: {ledger_path}")
    ledger_sheet = str(ledger_cfg.get("sheet", "市場走行一覧"))
    ledger_header_row = int(ledger_cfg.get("header_row", 2))
    file_col_expected = str(ledger_cfg.get("file_name_column", "TTDC提供ファイル名(タグ情報あり)"))

    network_roots_raw = cfg.get("network_roots", [])
    if not isinstance(network_roots_raw, list) or not network_roots_raw:
        raise ValueError("Invalid config: 'network_roots' must be a non-empty array.")
    network_roots = [os.path.normpath(str(x)) for x in network_roots_raw if str(x).strip()]
    if not network_roots:
        raise ValueError("Invalid config: no usable network root paths.")

    filters_cfg = cfg.get("aq_to_ay_filters", {})
    if not isinstance(filters_cfg, dict):
        raise ValueError("Invalid config: 'aq_to_ay_filters' must be an object.")

    df = pd.read_excel(
        ledger_path,
        sheet_name=ledger_sheet,
        header=ledger_header_row,
        dtype=object,
    )
    columns = list(df.columns)

    file_col = _find_file_name_column(columns, file_col_expected)
    if file_col is None:
        raise ValueError(f"Ledger file-name column not found: {file_col_expected}")

    active_filters: List[Dict[str, Any]] = []
    for key, spec in filters_cfg.items():
        if str(key).startswith("_comment"):
            continue
        if not isinstance(spec, dict):
            continue
        if not bool(spec.get("enabled", False)):
            continue
        excel_col = str(spec.get("excel_col", key)).strip().upper()
        mode = str(spec.get("mode", "include")).strip().lower()
        if mode not in {"include", "exclude"}:
            raise ValueError(f"Invalid mode in aq_to_ay_filters[{key}]: {mode}")
        values = spec.get("values", [])
        if values is None:
            values = []
        if not isinstance(values, list):
            values = [values]
        col_idx = _excel_col_to_index(excel_col)
        if col_idx < 0 or col_idx >= len(columns):
            raise ValueError(f"Filter column out of range: {excel_col} (index={col_idx}, columns={len(columns)})")
        col_name = str(columns[col_idx])
        active_filters.append(
            {
                "key": str(key),
                "excel_col": excel_col,
                "column_name": col_name,
                "mode": mode,
                "values": values,
            }
        )

    combine_mode = str(cfg.get("combine", "AND")).strip().upper()
    if combine_mode != "AND":
        raise ValueError(f"Unsupported combine mode: {combine_mode}. Only AND is supported.")

    sampling_cfg = cfg.get("sampling", {})
    sampling_enabled = False
    if isinstance(sampling_cfg, dict):
        sampling_enabled = bool(sampling_cfg.get("enabled", False))

    kept_stems: Dict[str, str] = {}
    skipped_blank_file = 0
    filtered_out = 0
    for _, row in df.iterrows():
        raw_file = row[file_col]
        file_stem = _normalize_file_stem(raw_file)
        if file_stem in {"", "ttdc_filename"}:
            skipped_blank_file += 1
            continue
        row_ok = True
        for flt in active_filters:
            cell = row[flt["column_name"]]
            if not _matches_filter(cell, flt["mode"], flt["values"]):
                row_ok = False
                break
        if not row_ok:
            filtered_out += 1
            continue
        if file_stem not in kept_stems:
            kept_stems[file_stem] = os.path.basename(str(raw_file).strip())

    if not kept_stems:
        raise ValueError("No candidate files remained after applying AQ-AY filters.")

    file_index: Dict[str, str] = {}
    duplicate_count = 0
    unreachable_roots: List[str] = []
    for root in network_roots:
        if not os.path.isdir(root):
            unreachable_roots.append(root)
            continue
        for path in glob.iglob(os.path.join(root, "**", pattern), recursive=True):
            if not os.path.isfile(path):
                continue
            key = _normalize_file_stem(os.path.basename(path))
            if not key:
                continue
            if key in file_index:
                duplicate_count += 1
                continue
            file_index[key] = os.path.normpath(path)

    if not file_index:
        raise ValueError("No CSV files found in configured network roots.")

    ledger_keys = set(kept_stems.keys())
    network_keys = set(file_index.keys())
    usable_keys = sorted(ledger_keys & network_keys)
    missing_in_network = sorted(ledger_keys - network_keys)
    missing_in_ledger = sorted(network_keys - ledger_keys)

    csv_paths = [file_index[k] for k in usable_keys]
    if not csv_paths:
        raise ValueError("No usable files remained after ledger/network matching.")

    report = {
        "config_path": os.path.normpath(config_path),
        "ledger_path": ledger_path,
        "file_name_column": file_col,
        "rows_total": int(len(df)),
        "rows_skipped_blank_file": int(skipped_blank_file),
        "rows_filtered_out": int(filtered_out),
        "filters_active": active_filters,
        "combine": combine_mode,
        "sampling_enabled": bool(sampling_enabled),
        "candidates_after_filter": int(len(kept_stems)),
        "unreachable_roots": unreachable_roots,
        "duplicate_network_filenames": int(duplicate_count),
        "missing_in_network": int(len(missing_in_network)),
        "missing_in_ledger": int(len(missing_in_ledger)),
        "usable_count": int(len(csv_paths)),
    }
    return csv_paths, report


def sample_paths_interactively(
    csv_paths: Sequence[str],
    *,
    seed: int,
    title: str,
) -> Tuple[List[str], Dict[str, Any]]:
    total_usable = len(csv_paths)
    if total_usable <= 0:
        raise ValueError("No input files available for sampling.")

    sampling_options: List[Tuple[int, int]] = []
    menu: List[str] = []
    for pct in TAGGED_SAMPLE_PERCENT_OPTIONS:
        n_files = _round_half_up(total_usable * (pct / 100.0))
        sampling_options.append((pct, n_files))
        menu.append(f"{pct}% ({n_files} files)")

    default_index = len(TAGGED_SAMPLE_PERCENT_OPTIONS)
    selected_idx = choose_from_list(menu, title, default_index=default_index)
    selected_pct, sample_count = sampling_options[selected_idx]
    if sample_count <= 0:
        raise ValueError("Selected sampling ratio produced 0 files. Please choose a larger percentage.")

    base_paths = list(csv_paths)
    if sample_count >= total_usable:
        sampled_paths = sorted(base_paths)
    else:
        rng = random.Random(int(seed))
        sampled_paths = sorted(rng.sample(base_paths, k=sample_count))

    return sampled_paths, {
        "percent": int(selected_pct),
        "sample_count": int(sample_count),
        "total_count": int(total_usable),
        "seed": int(seed),
    }
