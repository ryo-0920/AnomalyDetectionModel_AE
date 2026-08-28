import glob
import json
import math
import os
import random
import re
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


def _normalize_header_candidates(expected: Any) -> List[str]:
    if isinstance(expected, (list, tuple)):
        values = expected
    else:
        values = [expected]
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def _find_column_by_header_candidates(columns: Sequence[Any], expected: Any) -> Optional[str]:
    candidates = _normalize_header_candidates(expected)
    if not candidates:
        return None
    column_names = [str(col) for col in columns]
    for candidate in candidates:
        if candidate in columns:
            return str(candidate)
        if candidate in column_names:
            return candidate
    normalized = {str(col): _normalize_header_name(col) for col in columns}
    expected_norms = {_normalize_header_name(candidate) for candidate in candidates}
    for original, col_norm in normalized.items():
        if col_norm in expected_norms:
            return original
    return None


def _find_file_name_column(columns: Sequence[Any], expected: Any) -> Optional[str]:
    found = _find_column_by_header_candidates(columns, expected)
    if found is not None:
        return found
    # Fallback for this project's ledger variants.
    token_file = _normalize_header_name("\u30d5\u30a1\u30a4\u30eb\u540d")
    token_ts = _normalize_header_name("\u6642\u7cfb\u5217")
    normalized = {str(col): _normalize_header_name(col) for col in columns}
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


def _load_tagged_filter_profile_config(config_path: str, profile: Optional[str] = None) -> Dict[str, Any]:
    cfg = _load_tagged_filter_config(config_path)
    if not profile:
        return cfg
    profile_cfg = cfg.get(profile)
    if isinstance(profile_cfg, dict):
        if not bool(profile_cfg.get("enabled", True)):
            raise ValueError(f"Tagged filter profile is disabled: {config_path}#{profile}")
        return profile_cfg
    if any(key in cfg for key in ("ledger", "ledger_fallbacks", "network_roots")):
        return cfg
    raise ValueError(f"Tagged filter profile not found: {config_path}#{profile}")


def _candidate_ledger_paths(path_text: str) -> List[str]:
    raw = str(path_text or "").strip()
    if not raw:
        return []
    candidates: List[str] = []

    def _add(p: str) -> None:
        n = os.path.normpath(p)
        if n not in candidates:
            candidates.append(n)

    _add(raw)

    # Some synced corporate paths are available without the "OneDrive - " segment.
    raw_without_onedrive = re.sub(
        r"^([A-Za-z]:\\Users\\[^\\]+\\)OneDrive - ([^\\]+)(\\.*)$",
        r"\1\2\3",
        raw,
    )
    if raw_without_onedrive != raw:
        _add(raw_without_onedrive)

    # Convert Windows path like C:\Users\... to WSL mount /mnt/c/Users/...
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if m:
        drive = m.group(1).lower()
        tail = m.group(2).replace("\\", "/")
        _add(f"/mnt/{drive}/{tail}")

    # If path uses backslashes without drive letter, normalize slash style as a fallback.
    if "\\" in raw:
        _add(raw.replace("\\", "/"))

    return candidates


def _resolve_existing_path(path_text: str) -> Optional[str]:
    for cand in _candidate_ledger_paths(path_text):
        if os.path.isfile(cand):
            return cand
    return None


def _normalize_ledger_spec(raw: Any, *, field_name: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config: '{field_name}' must be an object.")
    ledger_path_raw = str(raw.get("path", "")).strip()
    if not ledger_path_raw:
        raise ValueError(f"Invalid config: {field_name}.path is required.")
    filters_cfg = raw.get("aq_to_ay_filters")
    if filters_cfg is not None and not isinstance(filters_cfg, dict):
        raise ValueError(f"Invalid config: {field_name}.aq_to_ay_filters must be an object.")
    return {
        "field_name": field_name,
        "path_raw": ledger_path_raw,
        "sheet": str(raw.get("sheet", "市場走行一覧")),
        "header_row": int(raw.get("header_row", 2)),
        "file_name_column": raw.get("file_name_column", "TTDC提供ファイル名(タグ情報あり)"),
        "aq_to_ay_filters": filters_cfg,
    }


def _collect_ledger_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    ledger_specs: List[Dict[str, Any]] = []
    ledger_specs.append(_normalize_ledger_spec(cfg.get("ledger", {}), field_name="ledger"))

    fallbacks_raw = cfg.get("ledger_fallbacks", [])
    if fallbacks_raw is None:
        fallbacks_raw = []
    if not isinstance(fallbacks_raw, list):
        raise ValueError("Invalid config: 'ledger_fallbacks' must be an array.")

    for idx, fallback in enumerate(fallbacks_raw):
        ledger_specs.append(_normalize_ledger_spec(fallback, field_name=f"ledger_fallbacks[{idx}]"))
    return ledger_specs


def _candidate_network_roots(root_text: str, ledger_path: str) -> List[str]:
    candidates = _candidate_ledger_paths(root_text)
    ledger_root = os.path.dirname(os.path.dirname(os.path.normpath(ledger_path)))
    if ledger_root and ledger_root not in candidates:
        candidates.append(ledger_root)
    return candidates


def is_tagged_dataset_ledger_path(path_text: str, config_path: str, profile: Optional[str] = None) -> bool:
    if not path_text:
        return False
    cfg = _load_tagged_filter_profile_config(config_path, profile=profile)
    input_norm = os.path.normcase(os.path.normpath(str(path_text).strip()))
    for ledger_spec in _collect_ledger_specs(cfg):
        for candidate in _candidate_ledger_paths(ledger_spec["path_raw"]):
            candidate_norm = os.path.normcase(os.path.normpath(candidate))
            if input_norm == candidate_norm:
                return True
    return False


def resolve_tagged_dataset_ledger_selector(config_path: str, selector: str, profile: Optional[str] = None) -> str:
    if not selector:
        raise ValueError("Tagged dataset ledger selector is empty.")
    cfg = _load_tagged_filter_profile_config(config_path, profile=profile)
    selector_norm = str(selector).strip()
    input_norm = os.path.normcase(os.path.normpath(selector_norm))
    available: List[str] = []
    for ledger_spec in _collect_ledger_specs(cfg):
        field_name = str(ledger_spec["field_name"])
        path_raw = str(ledger_spec["path_raw"])
        available.append(field_name)
        available.append(path_raw)
        if selector_norm == field_name:
            return path_raw
        for candidate in _candidate_ledger_paths(path_raw):
            candidate_norm = os.path.normcase(os.path.normpath(candidate))
            if input_norm == candidate_norm:
                return path_raw
    raise ValueError(
        "Unknown tagged dataset ledger selector: "
        f"{selector}. Available selectors: {available}"
    )


def build_tagged_dataset_csvs_from_config(
    config_path: str,
    pattern: str = "*.csv",
    preferred_ledger_path: Optional[str] = None,
    profile: Optional[str] = None,
    override_search_roots: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    cfg = _load_tagged_filter_profile_config(config_path, profile=profile)

    ledger_specs = _collect_ledger_specs(cfg)
    checked_ledger_candidates: List[str] = []
    preferred_norm = None
    if preferred_ledger_path:
        preferred_norm = os.path.normcase(os.path.normpath(str(preferred_ledger_path).strip()))

    if override_search_roots is None:
        network_roots_raw = cfg.get("network_roots", [])
        if not isinstance(network_roots_raw, list) or not network_roots_raw:
            raise ValueError("Invalid config: 'network_roots' must be a non-empty array.")
        network_roots = [os.path.normpath(str(x)) for x in network_roots_raw if str(x).strip()]
    else:
        network_roots = [os.path.normpath(str(x)) for x in override_search_roots if str(x).strip()]
    if not network_roots:
        raise ValueError("Invalid config: no usable search root paths.")
    combine_mode = str(cfg.get("combine", "AND")).strip().upper()
    if combine_mode != "AND":
        raise ValueError(f"Unsupported combine mode: {combine_mode}. Only AND is supported.")

    sampling_cfg = cfg.get("sampling", {})
    sampling_enabled = False
    if isinstance(sampling_cfg, dict):
        sampling_enabled = bool(sampling_cfg.get("enabled", False))

    def _build_for_ledger(ledger_spec: Dict[str, Any], ledger_path: str) -> Tuple[List[str], Dict[str, Any]]:
        filters_cfg = ledger_spec.get("aq_to_ay_filters")
        if filters_cfg is None:
            filters_cfg = cfg.get("aq_to_ay_filters", {})
        if not isinstance(filters_cfg, dict):
            raise ValueError("Invalid config: 'aq_to_ay_filters' must be an object.")

        df = pd.read_excel(
            ledger_path,
            sheet_name=ledger_spec["sheet"],
            header=ledger_spec["header_row"],
            dtype=object,
        )
        columns = list(df.columns)

        file_col = _find_file_name_column(columns, ledger_spec["file_name_column"])
        if file_col is None:
            raise ValueError(f"Ledger file-name column not found: {ledger_spec['file_name_column']}")

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
            header_name = spec.get("header_name")
            col_name = _find_column_by_header_candidates(columns, header_name) if header_name is not None else None
            if col_name is None:
                col_idx = _excel_col_to_index(excel_col)
                if col_idx < 0 or col_idx >= len(columns):
                    raise ValueError(f"Filter column out of range: {excel_col} (index={col_idx}, columns={len(columns)})")
                col_name = str(columns[col_idx])
            active_filters.append(
                {
                    "key": str(key),
                    "excel_col": excel_col,
                    "header_name": header_name,
                    "column_name": col_name,
                    "mode": mode,
                    "values": values,
                }
            )

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

        file_index: Dict[str, str] = {}
        duplicate_count = 0
        unreachable_roots: List[str] = []
        scanned_roots: List[str] = []
        for root in network_roots:
            resolved_root = None
            for candidate_root in _candidate_network_roots(root, ledger_path):
                if os.path.isdir(candidate_root):
                    resolved_root = candidate_root
                    break
            if resolved_root is None:
                unreachable_roots.append(root)
                continue
            if resolved_root not in scanned_roots:
                scanned_roots.append(resolved_root)
            for path in glob.iglob(os.path.join(resolved_root, "**", pattern), recursive=True):
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
        report = {
            "config_path": os.path.normpath(config_path),
            "config_profile": profile,
            "ledger_path": ledger_path,
            "ledger_field_name": ledger_spec.get("field_name"),
            "preferred_ledger_path": os.path.normpath(preferred_ledger_path) if preferred_ledger_path else None,
            "search_roots_override": [os.path.normpath(x) for x in network_roots] if override_search_roots is not None else None,
            "ledger_candidates": checked_ledger_candidates,
            "file_name_column": file_col,
            "rows_total": int(len(df)),
            "rows_skipped_blank_file": int(skipped_blank_file),
            "rows_filtered_out": int(filtered_out),
            "filters_active": active_filters,
            "combine": combine_mode,
            "sampling_enabled": bool(sampling_enabled),
            "candidates_after_filter": int(len(kept_stems)),
            "unreachable_roots": unreachable_roots,
            "scanned_roots": scanned_roots,
            "duplicate_network_filenames": int(duplicate_count),
            "missing_in_network": int(len(missing_in_network)),
            "missing_in_ledger": int(len(missing_in_ledger)),
            "usable_count": int(len(csv_paths)),
        }
        return csv_paths, report

    last_report: Optional[Dict[str, Any]] = None
    any_resolved = False
    for ledger_spec in ledger_specs:
        candidates = _candidate_ledger_paths(ledger_spec["path_raw"])
        checked_ledger_candidates.extend(candidates)
        if preferred_norm is not None:
            matches_preferred = any(
                os.path.normcase(os.path.normpath(candidate)) == preferred_norm
                for candidate in candidates
            )
            if not matches_preferred:
                continue
        resolved = _resolve_existing_path(ledger_spec["path_raw"])
        if not resolved:
            continue
        any_resolved = True
        csv_paths, report = _build_for_ledger(ledger_spec, resolved)
        last_report = report
        if csv_paths:
            return csv_paths, report
        if preferred_norm is not None:
            break

    if preferred_norm is not None and not any_resolved:
        raise FileNotFoundError(
            "Preferred tagged dataset ledger not found: "
            f"preferred={preferred_ledger_path}, checked={checked_ledger_candidates}"
        )
    if not any_resolved:
        raise FileNotFoundError(
            "Tagged dataset ledger not found: "
            f"checked={checked_ledger_candidates}"
        )
    if last_report is not None:
        raise ValueError(
            "No usable files remained after ledger/network matching. "
            f"last_ledger={last_report.get('ledger_field_name')} candidates_after_filter={last_report.get('candidates_after_filter')} usable_count={last_report.get('usable_count')}"
        )
    raise ValueError("No usable files remained after ledger/network matching.")


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
