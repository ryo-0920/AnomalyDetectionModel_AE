#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd


jp_font = "MS Gothic"
rcParams["font.family"] = jp_font
rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COLUMN_MAP_PATH = (PROJECT_ROOT / "config" / "column_map.json").resolve()
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis")
DEFAULT_SCORE_COLUMN_CANDIDATES = [
    "error",
    "y_conv_score",
    "anomaly",
    "score",
    "anomaly_score",
]
DEFAULT_FLAG_COLUMN_CANDIDATES = [
    "is_anomaly",
    "y_conv",
    "anomaly_flag",
]


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_evaluation_config(config_path: Path, section_name: Optional[str] = None) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = load_json(config_path)
    if section_name and section_name in cfg:
        return cfg[section_name]
    if "evaluation" in cfg:
        return cfg["evaluation"]
    expected = section_name if section_name else "evaluation"
    raise ValueError(f"{config_path}: {expected} section is missing")


def require_evaluation_key(eval_cfg: dict, config_path: Path, key: str) -> Any:
    if key not in eval_cfg:
        raise ValueError(f"{config_path}: evaluation.{key} is missing")
    return eval_cfg[key]


def load_column_map(path: Path) -> Tuple[Dict[str, str], str]:
    if not path.exists():
        return {}, "time"
    cfg = load_json(path)
    raw_map = cfg.get("column_map", {})
    mapping = {
        str(key).strip(): str(value).strip()
        for key, value in raw_map.items()
        if str(key).strip()
    }
    time_column = str(cfg.get("time_column", "time")).strip() or "time"
    return mapping, time_column


def normalize_basename(name: str) -> str:
    stem = Path(name).stem
    for suffix in ("_anomaly", "_normal", "_score"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.strip().lower()


def normalize_ledger_basename(name: str) -> Optional[str]:
    raw = str(name).strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"時系列csvファイル名", "csvfile"}:
        return None
    normalized = raw.replace("/", "\\")
    path_obj = Path(normalized)
    stem = path_obj.stem.strip().lower()
    if stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.isdigit() and len(stem) <= 6:
        return stem.zfill(6)
    return normalize_basename(raw)


def safe_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def variance_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.var(arr))


def resolve_data_dir(arg_value: Optional[str], eval_cfg: dict, config_path: Path) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()
    run_dir = str(eval_cfg.get("run_dir", "")).strip()
    if not run_dir:
        raise ValueError(f"{config_path}: evaluation.run_dir is required when CLI directory is omitted")
    return (config_path.parent / run_dir).resolve()


def resolve_output_dir(arg_value: Optional[str], on_dir: Path, eval_cfg: dict) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()
    output_cfg = eval_cfg.get("output", {})
    subdir = str(output_cfg.get("evaluation_subdir", "evaluation")).strip() or "evaluation"
    return (on_dir / subdir).resolve()


def resolve_column_name(
    eval_cfg: dict,
    config_key: str,
    logical_name: str,
    column_map: Dict[str, str],
    default_name: str,
) -> str:
    configured = str(eval_cfg.get(config_key, "")).strip()
    if configured:
        return configured
    mapped = str(column_map.get(logical_name, "")).strip()
    if mapped:
        return mapped
    return default_name


def resolve_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    columns = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in columns:
            return columns[key]
    return None


def resolve_score_column(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(DEFAULT_SCORE_COLUMN_CANDIDATES)
    return resolve_existing_column(df, candidates)


def resolve_flag_column(df: pd.DataFrame) -> Optional[str]:
    return resolve_existing_column(df, DEFAULT_FLAG_COLUMN_CANDIDATES)


def to_detection_flags(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float) > 0.0
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "anomaly"})


def compute_top_k_mean(series: pd.Series, top_k_percent: float) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    ratio = max(0.0, min(float(top_k_percent), 100.0)) / 100.0
    k = max(1, int(math.ceil(values.size * ratio)))
    top_values = np.partition(values, values.size - k)[-k:]
    return float(np.mean(top_values))


def nearest_value_at_time(time_series: pd.Series, value_series: pd.Series, target_time: float) -> float:
    time_values = pd.to_numeric(time_series, errors="coerce").to_numpy(dtype=float)
    value_values = pd.to_numeric(value_series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(time_values) & np.isfinite(value_values)
    if not mask.any() or not np.isfinite(target_time):
        return float("nan")
    idx = int(np.argmin(np.abs(time_values[mask] - float(target_time))))
    return float(value_values[mask][idx])


def classify_shift_group(value: Any) -> str:
    if value is None or pd.isna(value):
        return "OTHER"
    try:
        num = float(value)
        if math.isfinite(num):
            if abs(num - 20.0) < 1e-6:
                return "R"
            if abs(num - 40.0) < 1e-6 or abs(num - 50.0) < 1e-6:
                return "D/B"
    except Exception:
        pass
    text = str(value).strip().upper()
    if text in {"R"}:
        return "R"
    if text in {"D", "B", "D/B", "DB"}:
        return "D/B"
    return "OTHER"


def detect_phase(first_time: float, a1: Optional[float], a2: Optional[float], a3: Optional[float], a3_end: Optional[float]) -> str:
    if not np.isfinite(first_time):
        return "NONE"
    if a1 is not None and a2 is not None and np.isfinite(a1) and np.isfinite(a2) and a1 <= first_time < a2:
        return "A1"
    if a2 is not None and a3 is not None and np.isfinite(a2) and np.isfinite(a3) and a2 <= first_time < a3:
        return "A2"
    if a3 is not None and a3_end is not None and np.isfinite(a3) and np.isfinite(a3_end) and a3 <= first_time <= a3_end:
        return "A3"
    return "NONE"


def a2_duration_bin(duration: float) -> str:
    if not np.isfinite(duration) or duration < 0:
        return "unknown"
    if duration < 3.0:
        return "0-3s"
    if duration < 5.0:
        return "3-5s"
    return ">=5s"


def expand_accel_groups(accel_max: float) -> List[str]:
    if not np.isfinite(accel_max):
        return ["unknown"]
    groups = ["0-100%"]
    if accel_max >= 70.0:
        groups.append("70-100%")
    if accel_max >= 80.0:
        groups.append("80-100%")
    if accel_max >= 90.0:
        groups.append("90-100%")
    return groups


def find_file_column_name(df: pd.DataFrame, one_based_index: int) -> str:
    zero_index = int(one_based_index) - 1
    if zero_index < 0 or zero_index >= len(df.columns):
        raise ValueError(f"file column index is out of range: {one_based_index}")
    return str(df.columns[zero_index])


def load_label_intervals(label_cfg: dict) -> pd.DataFrame:
    path = Path(label_cfg["path"]).expanduser()
    sheet_name = label_cfg.get("sheet_name", 0)
    file_col_idx = int(label_cfg["file_column_excel_index"])
    a1_col_idx = int(label_cfg["a1_override_col"]) - 1
    k_col_idx = int(label_cfg["k_col"]) - 1
    m_col_idx = int(label_cfg["m_col"]) - 1
    n_col_idx = int(label_cfg["n_col"]) - 1
    delta = float(label_cfg.get("a1_delta_seconds", 5.0))
    df = pd.read_excel(path, sheet_name=sheet_name, header=0)
    file_col_name = find_file_column_name(df, file_col_idx)
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_file = row[file_col_name]
        if pd.isna(raw_file):
            continue
        basename = normalize_basename(str(raw_file))
        a1_override = safe_float(row.iloc[a1_col_idx])
        a2_start = safe_float(row.iloc[k_col_idx])
        a3_start = safe_float(row.iloc[m_col_idx])
        a3_end = safe_float(row.iloc[n_col_idx])
        if a2_start is None and a3_start is None and a3_end is None:
            continue
        a1_start = a1_override if a1_override is not None else (a2_start - delta if a2_start is not None else None)
        rows.append(
            {
                "basename": basename,
                "A1_start": a1_start,
                "A2_start": a2_start,
                "A3_start": a3_start,
                "A3_end": a3_end,
            }
        )
    label_df = pd.DataFrame(rows)
    if label_df.empty:
        raise ValueError("No valid label intervals were loaded")
    return label_df.drop_duplicates(subset=["basename"], keep="last")


def load_normal_basenames_from_ledger(ledger_cfg: dict) -> List[str]:
    path = Path(ledger_cfg["path"]).expanduser()
    sheet_name = ledger_cfg.get("sheet_name", 0)
    file_col_idx = int(ledger_cfg["file_column_excel_index"])
    df = pd.read_excel(path, sheet_name=sheet_name, header=0)
    file_col_name = find_file_column_name(df, file_col_idx)
    basenames: List[str] = []
    for _, row in df.iterrows():
        raw_file = row[file_col_name]
        if pd.isna(raw_file):
            continue
        normalized = normalize_ledger_basename(str(raw_file))
        if not normalized:
            continue
        basenames.append(normalized)
    return sorted(set(basenames))


def collect_result_csvs(result_dir: Path) -> List[Path]:
    return sorted(path for path in result_dir.glob("*.csv") if path.is_file())


def build_off_file_map(result_dir: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for path in collect_result_csvs(result_dir):
        key = normalize_basename(path.name)
        mapping.setdefault(key, []).append(path)
    return mapping


def find_off_candidates(key: str, file_map: Dict[str, List[Path]]) -> List[Path]:
    if key in file_map:
        return file_map[key]
    candidates: List[Path] = []
    key_compact = key.replace("_", "")
    for mapped_key, paths in file_map.items():
        mapped_compact = mapped_key.replace("_", "")
        if key in mapped_key or mapped_key in key or mapped_compact == key_compact:
            candidates.extend(paths)
    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def choose_best_candidate(key: str, candidates: Sequence[Path]) -> Path:
    exact = [path for path in candidates if normalize_basename(path.name) == key]
    if exact:
        return sorted(exact, key=lambda path: path.name)[0]
    suffix_matches = [path for path in candidates if normalize_basename(path.name).endswith(key)]
    if suffix_matches:
        return sorted(suffix_matches, key=lambda path: path.name)[0]
    return sorted(candidates, key=lambda path: path.name)[0]


def build_base_record(
    csv_path: Path,
    df: pd.DataFrame,
    time_column: str,
    flag_column: str,
    score_column: Optional[str],
    shift_column: Optional[str],
    speed_column: Optional[str],
    accel_column: Optional[str],
    top_k_percent: float,
) -> Dict[str, Any]:
    time_series = pd.to_numeric(df[time_column], errors="coerce")
    flags = to_detection_flags(df[flag_column])
    score_series = pd.to_numeric(df[score_column], errors="coerce") if score_column else pd.Series(dtype=float)
    shift_series = df[shift_column] if shift_column else pd.Series(index=df.index, dtype=object)
    speed_series = pd.to_numeric(df[speed_column], errors="coerce") if speed_column else pd.Series(index=df.index, dtype=float)
    accel_series = pd.to_numeric(df[accel_column], errors="coerce") if accel_column else pd.Series(index=df.index, dtype=float)
    time_values = time_series.to_numpy(dtype=float)
    score_values = score_series.to_numpy(dtype=float)
    finite_score_values = score_values[np.isfinite(score_values)]
    detected_indices = np.flatnonzero(flags.to_numpy(dtype=bool))
    detected = detected_indices.size > 0
    first_idx = int(detected_indices[0]) if detected else None
    first_detection_time = float(time_series.iloc[first_idx]) if detected and np.isfinite(time_series.iloc[first_idx]) else float("nan")
    first_detection_score = float(score_series.iloc[first_idx]) if detected and score_column and np.isfinite(score_series.iloc[first_idx]) else float("nan")
    first_shift_raw = shift_series.iloc[first_idx] if detected and shift_column else None
    first_shift_group = classify_shift_group(first_shift_raw) if detected else "OTHER"
    num_anomaly_frames = int(flags.sum())
    total_frames = int(len(df))
    anomaly_rate = float(num_anomaly_frames / total_frames) if total_frames > 0 else float("nan")
    file_score = compute_top_k_mean(score_series, top_k_percent) if score_column else float("nan")
    accel_max_all = float(accel_series.max()) if accel_column and accel_series.notna().any() else float("nan")
    return {
        "basename": normalize_basename(csv_path.name),
        "source_csv": str(csv_path),
        "detected": bool(detected),
        "num_anomaly_frames": num_anomaly_frames,
        "total_frames": total_frames,
        "anomaly_rate": anomaly_rate,
        "file_score": file_score,
        "score_column_used": score_column,
        "flag_column_used": flag_column,
        "first_detection_time": first_detection_time,
        "first_detection_score": first_detection_score,
        "first_detect_shift_raw": first_shift_raw,
        "first_detect_shift_group": first_shift_group,
        "accel_max_all": accel_max_all,
        "accel_groups_all": "|".join(expand_accel_groups(accel_max_all)),
        "_time_series": time_series,
        "_score_series": score_series,
        "_time_values": time_values,
        "_score_values": score_values,
        "_finite_score_values": finite_score_values,
        "_shift_series": shift_series,
        "_speed_series": speed_series,
        "_accel_series": accel_series,
        "_flags": flags,
    }


def first_detection_time_for_threshold(
    time_values: np.ndarray,
    score_values: np.ndarray,
    threshold: float,
) -> float:
    mask = np.isfinite(score_values) & np.isfinite(time_values) & (score_values >= float(threshold))
    if not mask.any():
        return float("nan")
    first_idx = int(np.flatnonzero(mask)[0])
    return float(time_values[first_idx])


def has_detection_for_threshold(score_values: np.ndarray, threshold: float) -> bool:
    finite_score_values = score_values[np.isfinite(score_values)]
    return bool(finite_score_values.size > 0 and (finite_score_values >= float(threshold)).any())


def max_score_in_mask(score_series: pd.Series, mask: np.ndarray) -> float:
    score_values = pd.to_numeric(score_series, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(score_values) & mask
    if not valid.any():
        return float("nan")
    return float(score_values[valid].max())


def classify_on_file_at_threshold(row: pd.Series, threshold: float) -> str:
    first_time = first_detection_time_for_threshold(row["_time_values"], row["_score_values"], threshold)
    return detect_phase(
        first_time,
        safe_float(row.get("A1_start")),
        safe_float(row.get("A2_start")),
        safe_float(row.get("A3_start")),
        safe_float(row.get("A3_end")),
    )


def build_on_roc_profile(row: pd.Series) -> Tuple[float, float]:
    time_values = row["_time_values"]
    finite_time = np.isfinite(time_values)
    a2_start = safe_float(row.get("A2_start"))
    a3_start = safe_float(row.get("A3_start"))
    if a2_start is None or a3_start is None or not np.isfinite(a2_start) or not np.isfinite(a3_start):
        return float("nan"), float("nan")
    before_a2_mask = finite_time & (time_values < float(a2_start))
    in_a2_mask = finite_time & (time_values >= float(a2_start)) & (time_values < float(a3_start))
    before_a2_max = max_score_in_mask(row["_score_series"], before_a2_mask)
    a2_max = max_score_in_mask(row["_score_series"], in_a2_mask)
    return before_a2_max, a2_max


def build_off_roc_profile(row: pd.Series) -> float:
    score_values = row["_finite_score_values"]
    if score_values.size == 0:
        return float("nan")
    return float(score_values.max())


def build_on_file_record(
    csv_path: Path,
    label_row: pd.Series,
    time_column: str,
    shift_column: Optional[str],
    speed_column: Optional[str],
    accel_column: Optional[str],
    top_k_percent: float,
    preferred_score_column: Optional[str],
) -> Optional[Dict[str, Any]]:
    df = read_csv_with_fallback(csv_path)
    flag_column = resolve_flag_column(df)
    if flag_column is None or time_column not in df.columns:
        return None
    score_column = resolve_score_column(df, preferred_score_column)
    base = build_base_record(
        csv_path=csv_path,
        df=df,
        time_column=time_column,
        flag_column=flag_column,
        score_column=score_column,
        shift_column=shift_column if shift_column in df.columns else None,
        speed_column=speed_column if speed_column in df.columns else None,
        accel_column=accel_column if accel_column in df.columns else None,
        top_k_percent=top_k_percent,
    )
    a1_start = safe_float(label_row.get("A1_start"))
    a2_start = safe_float(label_row.get("A2_start"))
    a3_start = safe_float(label_row.get("A3_start"))
    a3_end = safe_float(label_row.get("A3_end"))
    first_time = base["first_detection_time"]
    first_phase = detect_phase(first_time, a1_start, a2_start, a3_start, a3_end) if base["detected"] else "NONE"
    a2_duration = float(a3_start - a2_start) if a2_start is not None and a3_start is not None else float("nan")
    record: Dict[str, Any] = {
        "basename": base["basename"],
        "source_csv": base["source_csv"],
        "label": 1,
        "detected": base["detected"],
        "file_score": base["file_score"],
        "score_column_used": base["score_column_used"],
        "num_anomaly_frames": base["num_anomaly_frames"],
        "total_frames": base["total_frames"],
        "anomaly_rate": base["anomaly_rate"],
        "first_detection_time": first_time,
        "first_detection_score": base["first_detection_score"],
        "first_detect_phase": first_phase,
        "first_detect_shift_raw": base["first_detect_shift_raw"],
        "first_detect_shift_group": base["first_detect_shift_group"],
        "A1_start": a1_start,
        "A2_start": a2_start,
        "A3_start": a3_start,
        "A3_end": a3_end,
        "A2_duration_sec": a2_duration,
        "A2_duration_bin": a2_duration_bin(a2_duration),
        "collision_speed": float("nan"),
        "virtual_collision_speed": float("nan"),
        "collision_speed_reduction": float("nan"),
        "collision_speed_reduction_ratio": float("nan"),
        "time_to_collision_sec": float("nan"),
        "time_to_collision_ratio": float("nan"),
        "accel_A2_max": float("nan"),
        "accel_groups": "unknown",
        "pre_collision_detected": False,
        "_time_series": base["_time_series"],
        "_score_series": base["_score_series"],
        "_time_values": base["_time_values"],
        "_score_values": base["_score_values"],
        "_finite_score_values": base["_finite_score_values"],
    }

    time_series = base["_time_series"]
    accel_series = base["_accel_series"]
    speed_series = base["_speed_series"]

    if accel_column is not None and a2_start is not None and a3_start is not None:
        mask_a2 = (time_series >= float(a2_start)) & (time_series < float(a3_start))
        if mask_a2.any():
            accel_a2_max = pd.to_numeric(accel_series[mask_a2], errors="coerce").max()
            accel_a2_max = float(accel_a2_max) if pd.notna(accel_a2_max) else float("nan")
            record["accel_A2_max"] = accel_a2_max
            record["accel_groups"] = "|".join(expand_accel_groups(accel_a2_max))

    if first_phase in {"A1", "A2"} and a3_start is not None and np.isfinite(first_time):
        record["pre_collision_detected"] = True
        record["time_to_collision_sec"] = float(a3_start - first_time)
        if a2_start is not None and np.isfinite(a2_duration) and a2_duration > 0:
            record["time_to_collision_ratio"] = float(record["time_to_collision_sec"] / a2_duration)
        if speed_column is not None:
            collision_speed = nearest_value_at_time(time_series, speed_series, float(a3_start))
            virtual_speed = nearest_value_at_time(time_series, speed_series, first_time)
            record["collision_speed"] = collision_speed
            record["virtual_collision_speed"] = virtual_speed
            if np.isfinite(collision_speed) and np.isfinite(virtual_speed):
                reduction = float(virtual_speed - collision_speed)
                record["collision_speed_reduction"] = reduction
                if collision_speed > 0:
                    record["collision_speed_reduction_ratio"] = float(reduction / collision_speed)

    return record


def build_off_file_record(
    csv_path: Path,
    time_column: str,
    shift_column: Optional[str],
    speed_column: Optional[str],
    accel_column: Optional[str],
    top_k_percent: float,
    preferred_score_column: Optional[str],
) -> Optional[Dict[str, Any]]:
    df = read_csv_with_fallback(csv_path)
    flag_column = resolve_flag_column(df)
    if flag_column is None or time_column not in df.columns:
        return None
    score_column = resolve_score_column(df, preferred_score_column)
    base = build_base_record(
        csv_path=csv_path,
        df=df,
        time_column=time_column,
        flag_column=flag_column,
        score_column=score_column,
        shift_column=shift_column if shift_column in df.columns else None,
        speed_column=speed_column if speed_column in df.columns else None,
        accel_column=accel_column if accel_column in df.columns else None,
        top_k_percent=top_k_percent,
    )
    return {
        "basename": base["basename"],
        "source_csv": base["source_csv"],
        "label": 0,
        "detected": base["detected"],
        "file_score": base["file_score"],
        "score_column_used": base["score_column_used"],
        "num_anomaly_frames": base["num_anomaly_frames"],
        "total_frames": base["total_frames"],
        "anomaly_rate": base["anomaly_rate"],
        "false_positive_count": int(base["detected"]),
        "false_positive_rate": float(int(base["detected"])),
        "first_detection_time": base["first_detection_time"],
        "first_detection_score": base["first_detection_score"],
        "first_detect_phase": "NONE",
        "first_detect_shift_raw": base["first_detect_shift_raw"],
        "first_detect_shift_group": base["first_detect_shift_group"],
        "accel_groups": base["accel_groups_all"],
        "accel_A2_max": base["accel_max_all"],
        "_time_series": base["_time_series"],
        "_score_series": base["_score_series"],
        "_time_values": base["_time_values"],
        "_score_values": base["_score_values"],
        "_finite_score_values": base["_finite_score_values"],
    }


def build_per_file_summary_on(
    result_dir: Path,
    label_df: pd.DataFrame,
    time_column: str,
    shift_column: str,
    speed_column: str,
    accel_column: str,
    top_k_percent: float,
    preferred_score_column: Optional[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    label_map = label_df.set_index("basename")
    for csv_path in collect_result_csvs(result_dir):
        basename = normalize_basename(csv_path.name)
        if basename not in label_map.index:
            continue
        record = build_on_file_record(
            csv_path=csv_path,
            label_row=label_map.loc[basename],
            time_column=time_column,
            shift_column=shift_column,
            speed_column=speed_column,
            accel_column=accel_column,
            top_k_percent=top_k_percent,
            preferred_score_column=preferred_score_column,
        )
        if record is not None:
            rows.append(record)
    if not rows:
        raise ValueError(f"No valid ON result CSV files found in {result_dir}")
    return pd.DataFrame(rows).sort_values("basename").reset_index(drop=True)


def build_per_file_summary_off(
    result_dir: Path,
    normal_basenames: List[str],
    time_column: str,
    shift_column: str,
    speed_column: str,
    accel_column: str,
    top_k_percent: float,
    preferred_score_column: Optional[str],
    verbose: bool,
) -> pd.DataFrame:
    file_map = build_off_file_map(result_dir)
    ledger_set = set(normal_basenames)
    result_basenames = sorted(file_map.keys())
    target_basenames = [basename for basename in result_basenames if basename in ledger_set]
    not_in_ledger = [basename for basename in result_basenames if basename not in ledger_set]
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    ambiguous: Dict[str, List[str]] = {}
    for basename in target_basenames:
        candidates = find_off_candidates(basename, file_map)
        if not candidates:
            missing.append(basename)
            continue
        chosen = choose_best_candidate(basename, candidates)
        if len(candidates) > 1:
            ambiguous[basename] = [path.name for path in candidates]
            if verbose:
                print(f"[WARN] OFF ambiguous match: {basename} -> {ambiguous[basename]} ; chosen={chosen.name}")
        record = build_off_file_record(
            csv_path=chosen,
            time_column=time_column,
            shift_column=shift_column,
            speed_column=speed_column,
            accel_column=accel_column,
            top_k_percent=top_k_percent,
            preferred_score_column=preferred_score_column,
        )
        if record is None:
            missing.append(basename)
            continue
        rows.append(record)
    print(f"[INFO] OFF result count={len(target_basenames)}, matched={len(rows)}, missing={len(missing)}")
    if missing:
        print(f"[WARN] OFF missing basenames (first 20): {missing[:20]}")
    if not_in_ledger:
        print(f"[WARN] OFF result files not in ledger (first 20): {not_in_ledger[:20]}")
    if ambiguous and not verbose:
        print(f"[WARN] OFF ambiguous matches={len(ambiguous)}; use --verbose for details")
    if not rows:
        raise ValueError(f"No valid OFF result CSV files found in {result_dir}")
    return pd.DataFrame(rows).sort_values("basename").reset_index(drop=True)


def compute_confusion_matrices(per_file_df: pd.DataFrame) -> pd.DataFrame:
    df = per_file_df.copy()
    on_df = df[df["label"] == 1].copy()
    off_df = df[df["label"] == 0].copy()
    if on_df.empty or off_df.empty:
        raise ValueError("Both ON and OFF per-file results are required")

    def build_metrics(name: str, on_positive_mask: pd.Series) -> Dict[str, Any]:
        tp = int(on_positive_mask.sum())
        fn = int((~on_positive_mask).sum())
        off_detected = off_df["detected"].fillna(False).astype(bool)
        fp = int(off_detected.sum())
        tn = int((~off_detected).sum())
        total = tp + fp + fn + tn
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        tpr = recall
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else float("nan")
        tnr = float(tn / (fp + tn)) if (fp + tn) > 0 else float("nan")
        accuracy = float((tp + tn) / total) if total > 0 else float("nan")
        return {
            "condition": name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "TPR": tpr,
            "FPR": fpr,
            "FNR": fnr,
            "TNR": tnr,
        }

    condition_a = on_df["first_detect_phase"].isin(["A2", "A3"])
    condition_b = on_df["first_detect_phase"].isin(["A2"])
    rows = [
        build_metrics("A2_or_A3_detected", condition_a),
        build_metrics("A2_detected_only", condition_b),
    ]
    return pd.DataFrame(rows)


def expand_by_accel_groups(df: pd.DataFrame, accel_group_column: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        groups = [part for part in str(row.get(accel_group_column, "unknown")).split("|") if part]
        if not groups:
            groups = ["unknown"]
        for group in groups:
            new_row = row.to_dict()
            new_row["accel_group"] = group
            rows.append(new_row)
    return pd.DataFrame(rows)


def aggregate_on_groups(per_file_on: pd.DataFrame) -> pd.DataFrame:
    expanded = expand_by_accel_groups(per_file_on, "accel_groups")
    rows: List[Dict[str, Any]] = []
    grouped = expanded.groupby(["A2_duration_bin", "accel_group"], dropna=False)
    for (duration_bin, accel_group), group_df in grouped:
        detected_df = group_df[group_df["detected"].fillna(False)]
        valid_time_df = group_df[group_df["first_detect_phase"].isin(["A1", "A2"])]
        reduction_df = valid_time_df[np.isfinite(pd.to_numeric(valid_time_df["collision_speed_reduction_ratio"], errors="coerce"))]
        rows.append(
            {
                "A2_duration_bin": duration_bin,
                "accel_group": accel_group,
                "file_count": int(len(group_df)),
                "first_detect_A1_count": int((group_df["first_detect_phase"] == "A1").sum()),
                "first_detect_A2_count": int((group_df["first_detect_phase"] == "A2").sum()),
                "first_detect_A3_count": int((group_df["first_detect_phase"] == "A3").sum()),
                "first_detect_none_count": int((group_df["first_detect_phase"] == "NONE").sum()),
                "shift_DB_count": int((detected_df["first_detect_shift_group"] == "D/B").sum()),
                "shift_R_count": int((detected_df["first_detect_shift_group"] == "R").sum()),
                "shift_OTHER_count": int((detected_df["first_detect_shift_group"] == "OTHER").sum()),
                "time_to_collision_mean": float(pd.to_numeric(valid_time_df["time_to_collision_sec"], errors="coerce").mean()),
                "time_to_collision_variance": variance_or_nan(pd.to_numeric(valid_time_df["time_to_collision_sec"], errors="coerce").tolist()),
                "time_to_collision_ratio_mean": float(pd.to_numeric(valid_time_df["time_to_collision_ratio"], errors="coerce").mean()),
                "time_to_collision_ratio_variance": variance_or_nan(pd.to_numeric(valid_time_df["time_to_collision_ratio"], errors="coerce").tolist()),
                "collision_speed_reduction_ratio_mean": float(pd.to_numeric(reduction_df["collision_speed_reduction_ratio"], errors="coerce").mean()),
                "collision_speed_reduction_ratio_variance": variance_or_nan(pd.to_numeric(reduction_df["collision_speed_reduction_ratio"], errors="coerce").tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["A2_duration_bin", "accel_group"]).reset_index(drop=True)


def aggregate_off_groups(per_file_off: pd.DataFrame) -> pd.DataFrame:
    expanded = expand_by_accel_groups(per_file_off, "accel_groups")
    rows: List[Dict[str, Any]] = []
    grouped = expanded.groupby(["accel_group"], dropna=False)
    for accel_group, group_df in grouped:
        detected_df = group_df[group_df["detected"].fillna(False)]
        rows.append(
            {
                "accel_group": accel_group,
                "file_count": int(len(group_df)),
                "false_positive_count": int(group_df["false_positive_count"].fillna(0).sum()),
                "false_positive_rate": float(group_df["false_positive_count"].fillna(0).mean()),
                "shift_DB_count": int((detected_df["first_detect_shift_group"] == "D/B").sum()),
                "shift_R_count": int((detected_df["first_detect_shift_group"] == "R").sum()),
                "shift_OTHER_count": int((detected_df["first_detect_shift_group"] == "OTHER").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["accel_group"]).reset_index(drop=True)


def build_log_biased_quantile_thresholds(
    threshold_parts: Sequence[np.ndarray],
    max_points: int = 512,
    tail_bias: float = 4.0,
) -> np.ndarray:
    finite_parts = [part for part in threshold_parts if part.size > 0]
    if not finite_parts:
        return np.asarray([], dtype=float)
    all_scores = np.concatenate(finite_parts)
    all_scores = all_scores[np.isfinite(all_scores)]
    if all_scores.size == 0:
        return np.asarray([], dtype=float)
    unique_scores = np.unique(all_scores)
    if unique_scores.size <= max_points:
        return unique_scores[::-1]

    point_count = max(2, int(max_points))
    bias = max(1.0, float(tail_bias))
    u = np.linspace(0.0, 1.0, point_count)
    quantiles = 1.0 - np.power(1.0 - u, bias)
    sampled = np.quantile(all_scores, quantiles, method="linear")
    thresholds = np.unique(sampled)
    return thresholds[::-1]


def compute_roc_thresholds(per_file_on: pd.DataFrame, per_file_off: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    threshold_parts: List[np.ndarray] = []
    for _, row in per_file_on.iterrows():
        finite_scores = row["_finite_score_values"]
        if finite_scores.size > 0:
            threshold_parts.append(finite_scores)
    for _, row in per_file_off.iterrows():
        finite_scores = row["_finite_score_values"]
        if finite_scores.size > 0:
            threshold_parts.append(finite_scores)
    if not threshold_parts:
        raise ValueError("ROC requires finite point-level score values for both ON and OFF")
    thresholds = build_log_biased_quantile_thresholds(threshold_parts)
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        threshold_value = float(threshold)
        tp = int(
            sum(
                1
                for _, row in per_file_on.iterrows()
                if classify_on_file_at_threshold(row, threshold_value) == "A2"
            )
        )
        fn = int(len(per_file_on) - tp)
        fp = int(
            sum(
                1
                for _, row in per_file_off.iterrows()
                if has_detection_for_threshold(row["_score_values"], threshold_value)
            )
        )
        tn = int(len(per_file_off) - fp)
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "TPR": tpr,
                "FPR": fpr,
            }
        )
    thr_df = pd.DataFrame(rows)
    roc_points = thr_df[["FPR", "TPR"]].dropna().sort_values("FPR")
    fpr_values = np.concatenate(([0.0], roc_points["FPR"].to_numpy(dtype=float), [1.0]))
    tpr_values = np.concatenate(([0.0], roc_points["TPR"].to_numpy(dtype=float), [1.0]))
    auc = float(np.trapezoid(tpr_values, fpr_values))
    return thr_df, auc



def select_best_threshold_for_target_fpr(
    threshold_df: pd.DataFrame,
    target_fpr: float,
) -> pd.Series:
    candidates = threshold_df.copy()
    candidates = candidates[
        np.isfinite(pd.to_numeric(candidates["FPR"], errors="coerce"))
        & np.isfinite(pd.to_numeric(candidates["TPR"], errors="coerce"))
        & np.isfinite(pd.to_numeric(candidates["threshold"], errors="coerce"))
    ].copy()
    if candidates.empty:
        raise ValueError("No valid ROC threshold candidates are available")

    candidates["fpr_distance"] = (
        pd.to_numeric(candidates["FPR"], errors="coerce") - float(target_fpr)
    ).abs()
    candidates = candidates.sort_values(
        ["fpr_distance", "TPR", "threshold"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    return candidates.iloc[0]


def build_threshold_per_file_summary(
    per_file_on: pd.DataFrame,
    per_file_off: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for source_df in (per_file_on, per_file_off):
        for _, row in source_df.iterrows():
            first_time = first_detection_time_for_threshold(
                row["_time_values"],
                row["_score_values"],
                float(threshold),
            )
            rows.append(
                {
                    "main_name": str(row["basename"]),
                    "first_detection_time": first_time,
                }
            )
    return pd.DataFrame(rows, columns=["main_name", "first_detection_time"]).sort_values(
        "main_name"
    ).reset_index(drop=True)


def save_target_fpr_per_file_summaries(
    per_file_on: pd.DataFrame,
    per_file_off: pd.DataFrame,
    threshold_df: pd.DataFrame,
    out_dir: Path,
) -> List[Tuple[float, float, float, Path]]:
    target_specs = [
        (1e-3, "10^-3"),
        (1e-2, "10^-2"),
        (1e-1, "10^-1"),
    ]
    saved: List[Tuple[float, float, float, Path]] = []
    for target_fpr, label in target_specs:
        selected = select_best_threshold_for_target_fpr(threshold_df, target_fpr)
        threshold = float(selected["threshold"])
        actual_fpr = float(selected["FPR"])
        summary_df = build_threshold_per_file_summary(
            per_file_on=per_file_on,
            per_file_off=per_file_off,
            threshold=threshold,
        )
        out_path = out_dir / f"per_file_summary_all_FPR_{label}.csv"
        save_dataframe(summary_df, out_path)
        saved.append((target_fpr, actual_fpr, threshold, out_path))
    return saved

def plot_roc_curve(threshold_df: pd.DataFrame, auc: float, out_path: Path) -> None:
    roc_points = threshold_df[["FPR", "TPR"]].dropna().sort_values("FPR")
    fpr_values = np.concatenate(([0.0], roc_points["FPR"].to_numpy(dtype=float), [1.0]))
    tpr_values = np.concatenate(([0.0], roc_points["TPR"].to_numpy(dtype=float), [1.0]))
    plot_x = fpr_values.copy()
    positive = plot_x > 0.0
    min_tick = 1e-4
    if positive.any():
        min_positive = float(plot_x[positive].min())
        replacement = min(min_positive / 10.0, min_tick / 10.0)
    else:
        replacement = min_tick / 10.0
    plot_x[plot_x == 0.0] = replacement
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(plot_x, tpr_values, marker="o", label=f"AUC = {auc:.4f}")
    ax.set_xscale("log")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.set_xticklabels(["1e-4", "1e-3", "1e-2", "1e-1", "1e0"])
    ax.set_xlim(min_tick / 10.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    visible_columns = [column for column in df.columns if not str(column).startswith("_")]
    df.loc[:, visible_columns].to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Specification-aligned evaluation script for accidental acceleration models")
    parser.add_argument("--config", help="Shared ON/OFF evaluation config JSON")
    parser.add_argument("--config_on", help="ON evaluation config JSON")
    parser.add_argument("--config_off", help="OFF evaluation config JSON")
    parser.add_argument("--on_dir", default=None, help="Directory containing ON result CSV files")
    parser.add_argument("--off_dir", default=None, help="Directory containing OFF result CSV files")
    parser.add_argument("--out_dir", default=None, help="Evaluation output directory")
    parser.add_argument("--column_map", default=str(DEFAULT_COLUMN_MAP_PATH), help="Column map JSON path")
    parser.add_argument("--point_score_col", default=None, help="Preferred point-level score column")
    parser.add_argument("--top_k_percent", type=float, default=1.0, help="Top-k percent used for file score aggregation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose OFF ledger matching logs")
    args = parser.parse_args()

    if args.config and (args.config_on or args.config_off):
        parser.error("--config cannot be used together with --config_on or --config_off")
    if not args.config and not (args.config_on and args.config_off):
        parser.error("either --config or both --config_on and --config_off are required")

    if args.config:
        cfg_on_path = Path(args.config).resolve()
        cfg_off_path = cfg_on_path
    else:
        cfg_on_path = Path(args.config_on).resolve()
        cfg_off_path = Path(args.config_off).resolve()

    eval_on = load_evaluation_config(cfg_on_path, section_name="evaluation_on")
    eval_off = load_evaluation_config(cfg_off_path, section_name="evaluation_off")

    column_map, default_time_column = load_column_map(Path(args.column_map).resolve())
    time_column = resolve_column_name(eval_on, "time_column_name", "time", {"time": default_time_column}, default_time_column)
    shift_column = resolve_column_name(eval_on, "shift_column_name", "atshiftposition", column_map, "atshiftposition")
    speed_column = resolve_column_name(eval_on, "speed_column_name", "speed", column_map, "speed")
    accel_column_on = resolve_column_name(eval_on, "accel_column_name", "accelpedalangle", column_map, "accelpedalangle")
    accel_column_off = resolve_column_name(eval_off, "accel_column_name", "accelpedalangle", column_map, "accelpedalangle")

    on_dir = resolve_data_dir(args.on_dir, eval_on, cfg_on_path)
    off_dir = resolve_data_dir(args.off_dir, eval_off, cfg_off_path)
    out_dir = resolve_output_dir(args.out_dir, on_dir, eval_on)

    label_cfg = require_evaluation_key(eval_on, cfg_on_path, "label_review_sheet")
    ledger_cfg = require_evaluation_key(eval_off, cfg_off_path, "normal_ledger_sheet")

    label_df = load_label_intervals(label_cfg)
    normal_basenames = load_normal_basenames_from_ledger(ledger_cfg)

    print(f"[INFO] ON directory: {on_dir}")
    print(f"[INFO] OFF directory: {off_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] File score top-k percent: {args.top_k_percent}")

    per_file_on = build_per_file_summary_on(
        result_dir=on_dir,
        label_df=label_df,
        time_column=time_column,
        shift_column=shift_column,
        speed_column=speed_column,
        accel_column=accel_column_on,
        top_k_percent=args.top_k_percent,
        preferred_score_column=args.point_score_col,
    )
    per_file_off = build_per_file_summary_off(
        result_dir=off_dir,
        normal_basenames=normal_basenames,
        time_column=time_column,
        shift_column=shift_column,
        speed_column=speed_column,
        accel_column=accel_column_off,
        top_k_percent=args.top_k_percent,
        preferred_score_column=args.point_score_col,
        verbose=args.verbose,
    )
    per_file_all = pd.DataFrame(
        per_file_on.to_dict("records") + per_file_off.to_dict("records")
    )

    confusion_df = compute_confusion_matrices(per_file_all)
    on_group_df = aggregate_on_groups(per_file_on)
    off_group_df = aggregate_off_groups(per_file_off)
    roc_thresholds_df, auc = compute_roc_thresholds(per_file_on, per_file_off)
    roc_summary_df = pd.DataFrame(
        [
            {
                "positive_files": int(len(per_file_on)),
                "negative_files": int(len(per_file_off)),
                "auc": auc,
                "top_k_percent": float(args.top_k_percent),
                "roc_positive_condition": "first_detect_phase_is_A2",
            }
        ]
    )

    save_dataframe(per_file_on, out_dir / "per_file_summary_on.csv")
    save_dataframe(per_file_off, out_dir / "per_file_summary_off.csv")
    save_dataframe(per_file_all, out_dir / "per_file_summary_all.csv")
    save_dataframe(confusion_df, out_dir / "confusion_matrices.csv")
    save_dataframe(on_group_df, out_dir / "on_group_summary.csv")
    save_dataframe(off_group_df, out_dir / "off_group_summary.csv")
    save_dataframe(roc_thresholds_df, out_dir / "roc_thresholds.csv")
    save_dataframe(roc_summary_df, out_dir / "roc_auc_summary.csv")
    target_fpr_outputs = save_target_fpr_per_file_summaries(
        per_file_on=per_file_on,
        per_file_off=per_file_off,
        threshold_df=roc_thresholds_df,
        out_dir=out_dir,
    )

    save_plots = bool(eval_on.get("output", {}).get("save_plots", True))
    if save_plots:
        plot_roc_curve(roc_thresholds_df, auc, out_dir / "roc_curve.png")

    print(f"[INFO] Saved ON per-file summary: {out_dir / 'per_file_summary_on.csv'}")
    print(f"[INFO] Saved OFF per-file summary: {out_dir / 'per_file_summary_off.csv'}")
    print(f"[INFO] Saved combined per-file summary: {out_dir / 'per_file_summary_all.csv'}")
    print(f"[INFO] Saved confusion matrices: {out_dir / 'confusion_matrices.csv'}")
    print(f"[INFO] Saved ON group summary: {out_dir / 'on_group_summary.csv'}")
    print(f"[INFO] Saved OFF group summary: {out_dir / 'off_group_summary.csv'}")
    print(f"[INFO] Saved ROC thresholds: {out_dir / 'roc_thresholds.csv'}")
    print(f"[INFO] Saved ROC/AUC summary: {out_dir / 'roc_auc_summary.csv'}")
    for target_fpr, actual_fpr, threshold, out_path in target_fpr_outputs:
        print(
            f"[INFO] Saved target-FPR per-file summary: {out_path} "
            f"(target_FPR={target_fpr:.0e}, actual_FPR={actual_fpr:.10g}, threshold={threshold:.10g})"
        )
    if save_plots:
        print(f"[INFO] Saved ROC plot: {out_dir / 'roc_curve.png'}")


if __name__ == "__main__":
    main()
