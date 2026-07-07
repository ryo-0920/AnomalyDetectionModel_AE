import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
TRAIN_MODULE_DIR = os.path.join(PROJECT_ROOT, "1_transformer")
TRAIN_MODULE_PATH = os.path.join(TRAIN_MODULE_DIR, "train_transformer_autoencoder.py")

if TRAIN_MODULE_DIR not in sys.path:
    sys.path.insert(0, TRAIN_MODULE_DIR)

if os.path.exists(TRAIN_MODULE_PATH):
    spec = importlib.util.spec_from_file_location("train_transformer_autoencoder", TRAIN_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {TRAIN_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    read_csv_lower = module.read_csv_lower
    require_columns = module.require_columns
    preprocess_df_for_training = module.preprocess_df_for_training
    list_csvs_in_dir = module.list_csvs_in_dir
    FEATURES = module.FEATURES
    FEATURE_RULES = module.FEATURE_RULES
    CATEGORY_MAPS = module.CATEGORY_MAPS
    CATEGORICAL_FEATURES = module.CATEGORICAL_FEATURES
    DEFAULT_UNKNOWN_ID = module.DEFAULT_UNKNOWN_ID
else:
    raise ImportError(f"Transform module not found at: {TRAIN_MODULE_PATH}")


def load_hparams(hparams_path: str) -> Dict[str, Any]:
    if not os.path.isabs(hparams_path):
        hparams_path = os.path.normpath(os.path.join(PROJECT_ROOT, hparams_path))
    with open(hparams_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(state_dict, dict):
        return state_dict

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}

    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}

    return state_dict


def load_model_artifacts(artifacts_dir: str) -> Tuple[Any, Any, Dict[str, Any], torch.device]:
    config_path = os.path.join(artifacts_dir, "config.json")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    model_path = os.path.join(artifacts_dir, "model.pt")
    if not os.path.exists(config_path):
        raise ValueError(f"Model config not found in artifacts dir: {config_path}")
    if not os.path.exists(scaler_path):
        raise ValueError(f"Scaler not found in artifacts dir: {scaler_path}")
    if not os.path.exists(model_path):
        raise ValueError(f"Model weights not found in artifacts dir: {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    layout = config.get("layout")
    if not isinstance(layout, dict):
        raise ValueError(f"Invalid model config layout in {config_path}")

    scaler = joblib.load(scaler_path)
    input_dim = int(layout.get("input_dim"))
    d_model = int(config.get("d_model", 64))
    nhead = int(config.get("nhead", 4))
    num_layers = int(config.get("num_layers", 2))
    dim_ff = int(config.get("dim_ff", 128))
    dropout = float(config.get("dropout", 0.1))
    max_len = int(config.get("seq_len", 1000))

    model = module.CausalTransformerAutoencoder(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        max_len=max_len,
    )
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    normalized_state = normalize_state_dict_keys(state)
    model.load_state_dict(normalized_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, scaler, layout, device


def build_onehot_specs_from_layout(layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    onehot_specs = layout.get("onehot_specs", {})
    for col, spec in onehot_specs.items():
        codes = [float(v) for v in spec.get("codes", [])]
        specs[col] = {
            "codes": codes,
            "dim": int(spec.get("dim", len(codes))),
            "value_to_index": {float(v): i for i, v in enumerate(codes)},
        }
    return specs


def prepare_reconstruction_input(
    df: pd.DataFrame,
    scaler: Any,
    layout: Dict[str, Any],
) -> np.ndarray:
    continuous_features = list(layout.get("continuous_features", []))
    categorical_features = list(layout.get("categorical_features", []))
    if not set(continuous_features + categorical_features).issubset(df.columns):
        missing = set(continuous_features + categorical_features) - set(df.columns)
        raise ValueError(f"DataFrame missing required features for model reconstruction: {sorted(missing)}")

    X_cont = scaler.transform(df[continuous_features].astype(np.float32).to_numpy()).astype(np.float32)
    specs = build_onehot_specs_from_layout(layout)
    X_cat, _ = module.onehot_encode_df(df, specs)
    if X_cat.size == 0:
        return X_cont
    return np.concatenate([X_cont, X_cat], axis=1).astype(np.float32)


def reconstruct_last_windows(model: Any, X_norm: np.ndarray, seq_len: int, device: torch.device) -> np.ndarray:
    if len(X_norm) < seq_len:
        return np.zeros((0, X_norm.shape[1]), dtype=np.float32)

    recon_list: List[np.ndarray] = []
    inp = torch.from_numpy(X_norm).float().to(device)
    with torch.no_grad():
        for start in range(len(X_norm) - seq_len + 1):
            seq = inp[start : start + seq_len].unsqueeze(0)
            recon = model.reconstruct_last(seq)[0].cpu().numpy().astype(np.float32)
            recon_list.append(recon)

    if not recon_list:
        return np.zeros((0, X_norm.shape[1]), dtype=np.float32)
    return np.vstack(recon_list).astype(np.float32)


def decode_reconstruction_values(
    recon_norm: np.ndarray,
    scaler: Any,
    layout: Dict[str, Any],
) -> pd.DataFrame:
    feature_names = FEATURES
    feature_values = np.zeros((recon_norm.shape[0], len(feature_names)), dtype=np.float64)

    continuous_features = list(layout.get("continuous_features", []))
    categorical_features = list(layout.get("categorical_features", []))
    offsets = {col: int(v) for col, v in layout.get("offsets", {}).items()}

    if continuous_features:
        cont_count = len(continuous_features)
        recon_cont = scaler.inverse_transform(recon_norm[:, :cont_count].astype(np.float64)).astype(np.float64)
        for i, feature in enumerate(continuous_features):
            idx = FEATURES.index(feature)
            feature_values[:, idx] = recon_cont[:, i]

    for col in categorical_features:
        offset = offsets[col]
        dim = int(layout["onehot_specs"][col]["dim"])
        codes = [float(v) for v in layout["onehot_specs"][col]["codes"]]
        logits = recon_norm[:, offset : offset + dim]
        if logits.shape[1] != dim:
            raise ValueError(f"Reconstruction output length does not match one-hot spec for {col}")
        selected = np.argmax(logits, axis=1)
        feature_values[:, FEATURES.index(col)] = np.asarray([codes[i] for i in selected], dtype=np.float64)

    return pd.DataFrame(feature_values, columns=feature_names)


def build_contribution_dataframe(
    file_name: str,
    original: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> pd.DataFrame:
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed DataFrames must have the same shape for contribution analysis")

    errors = (original - reconstructed).abs()
    squared_errors = (original - reconstructed) ** 2
    mae = errors.mean()
    rmse = np.sqrt(squared_errors.mean())
    rows: List[Dict[str, Any]] = []

    for input_feature in original.columns:
        for recon_feature in reconstructed.columns:
            correlation = original[input_feature].corr(reconstructed[recon_feature])
            rows.append(
                {
                    "file_name": file_name,
                    "input_feature": input_feature,
                    "recon_feature": recon_feature,
                    "same_feature": input_feature == recon_feature,
                    "correlation": float(correlation) if pd.notna(correlation) else float("nan"),
                    "mean_abs_error": float(mae[input_feature]) if input_feature == recon_feature else float("nan"),
                    "rmse": float(rmse[input_feature]) if input_feature == recon_feature else float("nan"),
                    "n_samples": int(len(original)),
                }
            )
    return pd.DataFrame(rows)


def analyze_contribution_file(
    path: str,
    seq_len: int,
    model: Any,
    scaler: Any,
    layout: Dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    df = read_csv_lower(path)
    require_columns(df, path)
    df, _, _ = preprocess_df_for_training(df, os.path.basename(path))
    if len(df) < seq_len:
        print(f"[INFO] Skipping contribution analysis for {os.path.basename(path)} because row count {len(df)} < seq_len {seq_len}")
        return pd.DataFrame(
            columns=[
                "file_name",
                "input_feature",
                "recon_feature",
                "same_feature",
                "correlation",
                "mean_abs_error",
                "rmse",
                "n_samples",
            ]
        )

    X_norm = prepare_reconstruction_input(df, scaler, layout)
    recon_norm = reconstruct_last_windows(model, X_norm, seq_len, device)
    if recon_norm.shape[0] == 0:
        return pd.DataFrame(
            columns=[
                "file_name",
                "input_feature",
                "recon_feature",
                "same_feature",
                "correlation",
                "mean_abs_error",
                "rmse",
                "n_samples",
            ]
        )

    reconstructed = decode_reconstruction_values(recon_norm, scaler, layout)
    original = df[FEATURES].iloc[seq_len - 1 :].reset_index(drop=True)
    return build_contribution_dataframe(os.path.basename(path), original, reconstructed)


def get_csv_paths(csv: str, csvdir: str, pattern: str) -> List[str]:
    if csvdir:
        source_dir = csvdir
    elif os.path.isdir(csv):
        source_dir = csv
    elif os.path.isfile(csv):
        return [os.path.normpath(csv)]
    else:
        raise ValueError(f"--csv path is not a file or directory: {csv}")

    paths = list_csvs_in_dir(source_dir, pattern, recursive=True)
    unique_paths = sorted({os.path.normpath(p) for p in paths})
    if not unique_paths:
        raise ValueError(f"No CSV files found in {source_dir} matching pattern {pattern}")
    return unique_paths


def build_continuous_bins(feature: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rule = FEATURE_RULES.get(feature, {})
    vmin = rule.get("vmin")
    vmax = rule.get("vmax")
    if vmin is None or vmax is None:
        raise ValueError(f"Feature '{feature}' has no vmin/vmax defined in FEATURE_RULES")
    edges = np.linspace(vmin, vmax, num=11, dtype=np.float64)
    bin_min = edges[:-1]
    bin_max = edges[1:]
    return bin_min, bin_max, edges


def build_time_bins(time_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = time_values[np.isfinite(time_values)]
    if valid.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    vmin = float(np.floor(np.min(valid) * 10.0) / 10.0)
    vmax = float(np.max(valid))
    width = 1.0
    n_bins = max(1, int(np.ceil(max(0.0, vmax - vmin) / width)))
    bin_min = vmin + np.arange(n_bins, dtype=np.float64) * width
    bin_max = bin_min + width
    edges = vmin + np.arange(n_bins + 1, dtype=np.float64) * width
    return bin_min, bin_max, edges


def compute_continuous_counts(
    values: np.ndarray,
    feature: str,
    bin_edges: np.ndarray,
) -> np.ndarray:
    mask = np.isfinite(values)
    if feature in FEATURE_RULES:
        rule = FEATURE_RULES[feature]
        vmin = rule.get("vmin")
        vmax = rule.get("vmax")
        if vmin is not None:
            mask &= values >= vmin
        if vmax is not None:
            mask &= values <= vmax
    values = values[mask]
    if values.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=int)
    indices = np.digitize(values, bin_edges[1:], right=True)
    valid = (indices >= 0) & (indices < len(bin_edges) - 1)
    counts = np.bincount(indices[valid], minlength=len(bin_edges) - 1)
    return counts


def compute_time_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.zeros(max(0, len(edges) - 1), dtype=int)
    if len(edges) < 2:
        return counts

    valid = np.isfinite(values)
    valid &= values >= edges[0]
    valid &= values <= edges[-1]
    if not np.any(valid):
        return counts

    indices = np.digitize(values[valid], edges[1:], right=True)
    valid_indices = (indices >= 0) & (indices < len(counts))
    if not np.any(valid_indices):
        return counts

    return np.bincount(indices[valid_indices], minlength=len(counts))


def compute_category_counts(
    values: pd.Series,
    feature: str,
) -> Tuple[np.ndarray, List[str]]:
    if feature not in CATEGORY_MAPS:
        raise ValueError(f"Category map not found for feature '{feature}'")
    cat_map = CATEGORY_MAPS[feature]
    unknown_id = cat_map.get("UNKNOWN", DEFAULT_UNKNOWN_ID)
    if values.dtype.kind in "biufc":
        codes = pd.to_numeric(values, errors="coerce")
    else:
        codes = values.astype(str).str.strip().str.upper()
        codes = codes.map(cat_map).fillna(unknown_id)
    codes = codes[codes.notna()]
    code_values = [float(v) for v in sorted({float(v) for v in cat_map.values()})]
    code_to_name = {float(v): k for k, v in cat_map.items()}
    counts = np.zeros(len(code_values), dtype=int)
    for code in codes.astype(float).to_numpy():
        if np.isnan(code):
            continue
        if float(code) in code_to_name:
            idx = code_values.index(float(code))
        else:
            idx = code_values.index(float(unknown_id)) if float(unknown_id) in code_values else 0
        counts[idx] += 1
    names = [code_to_name.get(val, "UNKNOWN") for val in code_values]
    return counts, names


def build_category_name_map(feature: str) -> Dict[float, str]:
    cat_map = CATEGORY_MAPS[feature]
    return {float(v): k for k, v in cat_map.items()}


def build_exact_match_mask(series: pd.Series, expected_value: str) -> np.ndarray:
    numeric_series = pd.to_numeric(series, errors="coerce")
    try:
        expected_numeric = float(expected_value)
    except ValueError:
        expected_numeric = None

    if expected_numeric is not None and numeric_series.notna().any():
        return (numeric_series == expected_numeric).to_numpy(dtype=bool)

    normalized = series.astype(str).str.strip().str.upper()
    return (normalized == expected_value.strip().upper()).to_numpy(dtype=bool)


def load_and_preprocess_single(
    path: str,
    feature: str,
    use_training_preproc: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, int, int]:
    df = read_csv_lower(path)
    require_columns(df, path)

    is_categorical = feature in CATEGORICAL_FEATURES
    if use_training_preproc:
        df, mask, dropped = preprocess_df_for_training(df, os.path.basename(path))
        series = pd.to_numeric(df[feature], errors="coerce")
        if is_categorical:
            series = series.astype(float)
            values = series.to_numpy()
            return df, values, np.zeros(0, dtype=int), int(len(df)), int(len(df))
        return df, series.to_numpy(), np.zeros(0, dtype=int), int(len(df)), int(len(df))

    if is_categorical:
        raw = df[feature].astype(str).str.strip().str.upper()
        mapped = raw.map(CATEGORY_MAPS.get(feature, {}))
        unknown_id = CATEGORY_MAPS.get(feature, {}).get("UNKNOWN", DEFAULT_UNKNOWN_ID)
        mapped = mapped.where(~mapped.isna(), unknown_id).astype(float)
        return df, mapped.to_numpy(), np.zeros(0, dtype=int), int(len(df)), int(len(df))

    values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
    rule = FEATURE_RULES.get(feature, {})
    vmin = rule.get("vmin")
    vmax = rule.get("vmax")
    if vmin is not None:
        values = np.where(np.isfinite(values) & (values < vmin), np.nan, values)
    if vmax is not None:
        values = np.where(np.isfinite(values) & (values > vmax), np.nan, values)
    return df, values, np.zeros(0, dtype=int), int(len(df)), int(len(df))


def get_windows(values: np.ndarray, seq_len: int) -> List[np.ndarray]:
    if len(values) < seq_len:
        return []
    windows = [values[i : i + seq_len] for i in range(len(values) - seq_len + 1)]
    return windows


def analyze_continuous_file(
    path: str,
    feature: str,
    seq_len: int,
    use_training_preproc: bool,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    _, values, _, n_rows, _ = load_and_preprocess_single(path, feature, use_training_preproc)
    windows = get_windows(values, seq_len)
    if not windows:
        print(f"[INFO] Skipping {os.path.basename(path)} because row count {n_rows} < seq_len {seq_len}")
        return np.zeros(10, dtype=int), np.zeros(10, dtype=int), 0, 0
    bin_min, bin_max, bin_edges = build_continuous_bins(feature)
    max_counts = np.zeros(len(bin_min), dtype=int)
    all_counts = np.zeros(len(bin_min), dtype=int)
    n_valid_max = 0
    n_valid_all = 0
    for window in windows:
        valid = window[np.isfinite(window)]
        if valid.size > 0:
            max_val = valid.max()
            idx = np.digitize([max_val], bin_edges[1:], right=True)[0]
            if 0 <= idx < len(bin_min):
                max_counts[idx] += 1
            n_valid_max += 1
        valid_all = window[np.isfinite(window)]
        if valid_all.size > 0:
            counts = compute_continuous_counts(valid_all, feature, bin_edges)
            all_counts += counts
            n_valid_all += valid_all.size
    return max_counts, all_counts, n_valid_max, n_valid_all


def analyze_filtered_continuous_file(
    path: str,
    feature: str,
    seq_len: int,
    use_training_preproc: bool,
    filter_column: str,
    filter_value: str,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    df, values, _, n_rows, _ = load_and_preprocess_single(path, feature, use_training_preproc)
    filter_column_norm = filter_column.strip().lower()
    if filter_column_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: filter column '{filter_column}' not found")

    condition_mask = build_exact_match_mask(df[filter_column_norm], filter_value)
    windows = get_windows(values, seq_len)
    if not windows:
        print(f"[INFO] Skipping {os.path.basename(path)} because row count {n_rows} < seq_len {seq_len}")
        return np.zeros(10, dtype=int), np.zeros(10, dtype=int), 0, 0

    condition_windows = get_windows(condition_mask.astype(bool), seq_len)
    bin_min, bin_max, bin_edges = build_continuous_bins(feature)
    max_counts = np.zeros(len(bin_min), dtype=int)
    all_counts = np.zeros(len(bin_min), dtype=int)
    n_valid_max = 0
    n_valid_all = 0

    for window, condition_window in zip(windows, condition_windows):
        matched = window[np.isfinite(window) & condition_window]
        if matched.size > 0:
            max_val = matched.max()
            idx = np.digitize([max_val], bin_edges[1:], right=True)[0]
            if 0 <= idx < len(bin_min):
                max_counts[idx] += 1
            counts = compute_continuous_counts(matched, feature, bin_edges)
            all_counts += counts
            n_valid_max += 1
            n_valid_all += matched.size

    return max_counts, all_counts, n_valid_max, n_valid_all


def analyze_categorical_file(
    path: str,
    feature: str,
    use_training_preproc: bool,
) -> Tuple[np.ndarray, List[str], int]:
    _, values, _, n_rows, _ = load_and_preprocess_single(path, feature, use_training_preproc)
    if n_rows == 0:
        return np.zeros(len(CATEGORY_MAPS.get(feature, {})), dtype=int), list(CATEGORY_MAPS.get(feature, {}).keys()), 0
    counts, names = compute_category_counts(pd.Series(values), feature)
    total = int(counts.sum())
    return counts, names, total


def build_summary_dataframe_continuous(
    feature: str,
    max_counts_total: np.ndarray,
    all_counts_total: np.ndarray,
    total_valid_max: int,
    total_valid_all: int,
) -> pd.DataFrame:
    bin_min, bin_max, _ = build_continuous_bins(feature)
    df = pd.DataFrame(
        {
            "bin_min": bin_min,
            "bin_max": bin_max,
            "max_count_total": max_counts_total,
            "max_ratio_total": np.divide(max_counts_total, total_valid_max, out=np.zeros_like(max_counts_total, dtype=float), where=total_valid_max > 0),
            "all_count_total": all_counts_total,
            "all_ratio_total": np.divide(all_counts_total, total_valid_all, out=np.zeros_like(all_counts_total, dtype=float), where=total_valid_all > 0),
        }
    )
    return df


def build_perfile_dataframe_continuous(
    feature: str,
    file_stats: List[Dict[str, Any]],
) -> pd.DataFrame:
    bin_min, bin_max, _ = build_continuous_bins(feature)
    rows = []
    for stat in file_stats:
        for idx in range(len(bin_min)):
            total_max = int(stat["max_counts"][idx])
            total_all = int(stat["all_counts"][idx])
            max_ratio = total_max / stat["n_valid_max"] if stat["n_valid_max"] > 0 else 0.0
            all_ratio = total_all / stat["n_valid_all"] if stat["n_valid_all"] > 0 else 0.0
            rows.append(
                {
                    "file_name": stat["file_name"],
                    "bin_min": bin_min[idx],
                    "bin_max": bin_max[idx],
                    "max_count": total_max,
                    "max_ratio": max_ratio,
                    "all_count": total_all,
                    "all_ratio": all_ratio,
                }
            )
    return pd.DataFrame(rows)


def build_summary_dataframe_filtered_continuous(
    feature: str,
    filter_column: str,
    filter_value: str,
    max_counts_total: np.ndarray,
    all_counts_total: np.ndarray,
    total_valid_max: int,
    total_valid_all: int,
) -> pd.DataFrame:
    bin_min, bin_max, _ = build_continuous_bins(feature)
    return pd.DataFrame(
        {
            "feature": [feature] * len(bin_min),
            "condition_column": [filter_column] * len(bin_min),
            "condition_value": [filter_value] * len(bin_min),
            "bin_min": bin_min,
            "bin_max": bin_max,
            "max_count_total": max_counts_total,
            "max_ratio_total": np.divide(max_counts_total, total_valid_max, out=np.zeros_like(max_counts_total, dtype=float), where=total_valid_max > 0),
            "all_count_total": all_counts_total,
            "all_ratio_total": np.divide(all_counts_total, total_valid_all, out=np.zeros_like(all_counts_total, dtype=float), where=total_valid_all > 0),
        }
    )


def build_perfile_dataframe_filtered_continuous(
    feature: str,
    filter_column: str,
    filter_value: str,
    file_stats: List[Dict[str, Any]],
) -> pd.DataFrame:
    bin_min, bin_max, _ = build_continuous_bins(feature)
    rows = []
    for stat in file_stats:
        for idx in range(len(bin_min)):
            total_max = int(stat["max_counts"][idx])
            total_all = int(stat["all_counts"][idx])
            max_ratio = total_max / stat["n_valid_max"] if stat["n_valid_max"] > 0 else 0.0
            all_ratio = total_all / stat["n_valid_all"] if stat["n_valid_all"] > 0 else 0.0
            rows.append(
                {
                    "file_name": stat["file_name"],
                    "feature": feature,
                    "condition_column": filter_column,
                    "condition_value": filter_value,
                    "bin_min": bin_min[idx],
                    "bin_max": bin_max[idx],
                    "max_count": total_max,
                    "max_ratio": max_ratio,
                    "all_count": total_all,
                    "all_ratio": all_ratio,
                }
            )
    return pd.DataFrame(rows)


def collect_filtered_time_values(
    path: str,
    seq_len: int,
    use_training_preproc: bool,
    filter_column: str,
    filter_value: str,
    time_column: str,
) -> np.ndarray:
    df = read_csv_lower(path)
    require_columns(df, path)

    if use_training_preproc:
        df, _, _ = preprocess_df_for_training(df, os.path.basename(path))

    time_column_norm = time_column.strip().lower()
    if time_column_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: time column '{time_column}' not found")

    filter_column_norm = filter_column.strip().lower()
    if filter_column_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: filter column '{filter_column}' not found")

    n_rows = int(len(df))
    condition_mask = build_exact_match_mask(df[filter_column_norm], filter_value)
    time_series = pd.to_numeric(df[time_column_norm], errors="coerce").to_numpy(dtype=float)
    time_windows = get_windows(time_series, seq_len)
    condition_windows = get_windows(condition_mask.astype(bool), seq_len)
    if not time_windows:
        print(f"[INFO] Skipping {os.path.basename(path)} because row count {n_rows} < seq_len {seq_len}")
        return np.array([], dtype=np.float64)

    matched_times: List[float] = []
    for time_window, condition_window in zip(time_windows, condition_windows):
        matched = time_window[condition_window & np.isfinite(time_window)]
        matched_times.extend(matched.tolist())
    return np.asarray(matched_times, dtype=np.float64)


def build_summary_dataframe_filtered_time(
    feature: str,
    filter_column: str,
    filter_value: str,
    bin_min: np.ndarray,
    bin_max: np.ndarray,
    counts_total: np.ndarray,
    total_valid_times: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [feature] * len(bin_min),
            "condition_column": [filter_column] * len(bin_min),
            "condition_value": [filter_value] * len(bin_min),
            "time_bin_min": bin_min,
            "time_bin_max": bin_max,
            "time_count_total": counts_total,
            "time_ratio_total": np.divide(counts_total, total_valid_times, out=np.zeros_like(counts_total, dtype=float), where=total_valid_times > 0),
        }
    )


def build_summary_dataframe_false_accel_continuous(
    feature: str,
    filter_column: str,
    filter_value: str,
    max_counts_total: np.ndarray,
    all_counts_total: np.ndarray,
    total_valid_max: int,
    total_valid_all: int,
) -> pd.DataFrame:
    bin_min, bin_max, _ = build_continuous_bins(feature)
    return pd.DataFrame(
        {
            "feature": [feature] * len(bin_min),
            "condition_column": [filter_column] * len(bin_min),
            "condition_value": [filter_value] * len(bin_min),
            "bin_min": bin_min,
            "bin_max": bin_max,
            "fp_max_count_total": max_counts_total,
            "fp_max_ratio_total": np.divide(max_counts_total, total_valid_max, out=np.zeros_like(max_counts_total, dtype=float), where=total_valid_max > 0),
            "fp_count_total": all_counts_total,
            "fp_ratio_total": np.divide(all_counts_total, total_valid_all, out=np.zeros_like(all_counts_total, dtype=float), where=total_valid_all > 0),
        }
    )


def build_perfile_dataframe_false_accel_continuous(
    feature: str,
    filter_column: str,
    filter_value: str,
    file_stats: List[Dict[str, Any]],
) -> pd.DataFrame:
    bin_min, bin_max, _ = build_continuous_bins(feature)
    rows = []
    for stat in file_stats:
        for idx in range(len(bin_min)):
            total_max = int(stat["max_counts"][idx])
            total_all = int(stat["all_counts"][idx])
            rows.append(
                {
                    "file_name": stat["file_name"],
                    "feature": feature,
                    "condition_column": filter_column,
                    "condition_value": filter_value,
                    "bin_min": bin_min[idx],
                    "bin_max": bin_max[idx],
                    "fp_max_count": total_max,
                    "fp_max_ratio": total_max / stat["n_valid_max"] if stat["n_valid_max"] > 0 else 0.0,
                    "fp_count": total_all,
                    "fp_ratio": total_all / stat["n_valid_all"] if stat["n_valid_all"] > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_perfile_dataframe_filtered_time(
    feature: str,
    filter_column: str,
    filter_value: str,
    file_stats: List[Dict[str, Any]],
    bin_min: np.ndarray,
    bin_max: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for stat in file_stats:
        for idx in range(len(bin_min)):
            count = int(stat["time_counts"][idx])
            rows.append(
                {
                    "file_name": stat["file_name"],
                    "feature": feature,
                    "condition_column": filter_column,
                    "condition_value": filter_value,
                    "time_bin_min": bin_min[idx],
                    "time_bin_max": bin_max[idx],
                    "time_count": count,
                    "time_ratio": count / stat["n_valid_time"] if stat["n_valid_time"] > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_summary_dataframe_categorical(
    feature: str,
    class_names: List[str],
    counts_total: np.ndarray,
) -> pd.DataFrame:
    total_valid_all = int(counts_total.sum())
    return pd.DataFrame(
        {
            "feature": [feature] * len(class_names),
            "class_value": class_names,
            "all_count_total": counts_total,
            "all_ratio_total": np.divide(counts_total, total_valid_all, out=np.zeros_like(counts_total, dtype=float), where=total_valid_all > 0),
        }
    )


def build_perfile_dataframe_categorical(
    feature: str,
    file_stats: List[Dict[str, Any]],
    class_names: List[str],
) -> pd.DataFrame:
    rows = []
    for stat in file_stats:
        for idx, class_name in enumerate(class_names):
            count = int(stat["counts"][idx])
            rows.append(
                {
                    "file_name": stat["file_name"],
                    "feature": feature,
                    "class_value": class_name,
                    "all_count": count,
                    "all_ratio": count / stat["total"] if stat["total"] > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def analyze_false_accel_continuous_file(
    path: str,
    feature: str,
    seq_len: int,
    use_training_preproc: bool,
    filter_column: str,
    filter_value: str,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    df, values, _, n_rows, _ = load_and_preprocess_single(path, feature, use_training_preproc)
    filter_column_norm = filter_column.strip().lower()
    if filter_column_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: filter column '{filter_column}' not found")

    condition_mask = build_exact_match_mask(df[filter_column_norm], filter_value)
    windows = get_windows(values, seq_len)
    if not windows:
        print(f"[INFO] Skipping {os.path.basename(path)} because row count {n_rows} < seq_len {seq_len}")
        return np.zeros(10, dtype=int), np.zeros(10, dtype=int), 0, 0

    condition_windows = get_windows(condition_mask.astype(bool), seq_len)
    bin_min, bin_max, bin_edges = build_continuous_bins(feature)
    max_counts = np.zeros(len(bin_min), dtype=int)
    all_counts = np.zeros(len(bin_min), dtype=int)
    n_valid_max = 0
    n_valid_all = 0

    for window, condition_window in zip(windows, condition_windows):
        matched = window[np.isfinite(window) & condition_window]
        if matched.size > 0:
            max_val = matched.max()
            idx = np.digitize([max_val], bin_edges[1:], right=True)[0]
            if 0 <= idx < len(bin_min):
                max_counts[idx] += 1
            counts = compute_continuous_counts(matched, feature, bin_edges)
            all_counts += counts
            n_valid_max += 1
            n_valid_all += matched.size

    return max_counts, all_counts, n_valid_max, n_valid_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze accel distribution from CSVs")
    parser.add_argument("--csv", "-i", required=True, help="Single CSV file or directory containing CSVs")
    parser.add_argument("--csvdir", "-d", default="", help="Directory containing CSVs; takes precedence over --csv")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for CSV discovery")
    parser.add_argument("--hparams", default="config/hyperparams_common.json", help="Hyperparameters JSON path")
    parser.add_argument("--feature", default="accelpedalangle", help="Feature name to analyze")
    parser.add_argument("--output-dir", default="AE/outputs", help="Output directory for Excel")
    parser.add_argument("--output-prefix", default="accel_dist", help="Output filename prefix")
    parser.add_argument("--use-training-preproc", action="store_true", help="Use existing training preprocessing")
    parser.add_argument("--filtered-mode", action="store_true", help="Add filtered aggregation sheets based on an exact-match condition")
    parser.add_argument("--filter-column", default="is_anomaly", help="Condition column for filtered aggregation")
    parser.add_argument("--filter-value", default="1", help="Exact-match condition value for filtered aggregation")
    parser.add_argument("--time-column", default="time", help="Time column used for filtered Time distribution")
    parser.add_argument("--artifacts-dir", default="", help="Directory containing trained model artifacts for contribution analysis")
    args = parser.parse_args()

    if args.feature not in FEATURES:
        raise ValueError(f"Feature '{args.feature}' is not in FEATURES")
    if args.filtered_mode and args.feature in CATEGORICAL_FEATURES:
        raise ValueError("Filtered mode currently supports continuous features only")

    hparams = load_hparams(args.hparams)
    seq_len = int(hparams.get("seq_len"))
    csv_paths = get_csv_paths(args.csv, args.csvdir, args.pattern)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_prefix}_{args.feature}.xlsx")

    is_categorical = args.feature in CATEGORICAL_FEATURES
    filtered_summary_df = None
    filtered_perfile_df = None
    filtered_time_summary_df = None
    filtered_time_perfile_df = None
    false_accel_summary_df = None
    false_accel_perfile_df = None
    contribution_df = None
    model = None
    scaler = None
    layout = None
    device = None
    if args.artifacts_dir:
        model, scaler, layout, device = load_model_artifacts(args.artifacts_dir)
    if is_categorical:
        totals: np.ndarray = np.zeros(len(CATEGORY_MAPS[args.feature]), dtype=int)
        class_names = list(CATEGORY_MAPS[args.feature].keys())
        file_stats: List[Dict[str, Any]] = []
        for path in csv_paths:
            counts, names, total = analyze_categorical_file(path, args.feature, args.use_training_preproc)
            file_stats.append({"file_name": os.path.basename(path), "counts": counts, "total": total})
            totals += counts
        summary_df = build_summary_dataframe_categorical(args.feature, class_names, totals)
        perfile_df = build_perfile_dataframe_categorical(args.feature, file_stats, class_names)
    else:
        totals_max = np.zeros(10, dtype=int)
        totals_all = np.zeros(10, dtype=int)
        total_valid_max = 0
        total_valid_all = 0
        file_stats: List[Dict[str, Any]] = []
        for path in csv_paths:
            max_counts, all_counts, n_valid_max, n_valid_all = analyze_continuous_file(
                path, args.feature, seq_len, args.use_training_preproc
            )
            file_stats.append(
                {
                    "file_name": os.path.basename(path),
                    "max_counts": max_counts,
                    "all_counts": all_counts,
                    "n_valid_max": n_valid_max,
                    "n_valid_all": n_valid_all,
                }
            )
            totals_max += max_counts
            totals_all += all_counts
            total_valid_max += n_valid_max
            total_valid_all += n_valid_all
        summary_df = build_summary_dataframe_continuous(
            args.feature,
            totals_max,
            totals_all,
            total_valid_max,
            total_valid_all,
        )
        perfile_df = build_perfile_dataframe_continuous(args.feature, file_stats)

        if args.filtered_mode:
            filtered_totals_max = np.zeros(10, dtype=int)
            filtered_totals_all = np.zeros(10, dtype=int)
            filtered_total_valid_max = 0
            filtered_total_valid_all = 0
            filtered_file_stats: List[Dict[str, Any]] = []
            for path in csv_paths:
                max_counts, all_counts, n_valid_max, n_valid_all = analyze_filtered_continuous_file(
                    path,
                    args.feature,
                    seq_len,
                    args.use_training_preproc,
                    args.filter_column,
                    args.filter_value,
                )
                filtered_file_stats.append(
                    {
                        "file_name": os.path.basename(path),
                        "max_counts": max_counts,
                        "all_counts": all_counts,
                        "n_valid_max": n_valid_max,
                        "n_valid_all": n_valid_all,
                    }
                )
                filtered_totals_max += max_counts
                filtered_totals_all += all_counts
                filtered_total_valid_max += n_valid_max
                filtered_total_valid_all += n_valid_all

            filtered_summary_df = build_summary_dataframe_filtered_continuous(
                args.feature,
                args.filter_column,
                args.filter_value,
                filtered_totals_max,
                filtered_totals_all,
                filtered_total_valid_max,
                filtered_total_valid_all,
            )
            filtered_perfile_df = build_perfile_dataframe_filtered_continuous(
                args.feature,
                args.filter_column,
                args.filter_value,
                filtered_file_stats,
            )

            filtered_time_values_total: List[float] = []
            filtered_time_file_stats: List[Dict[str, Any]] = []
            for path in csv_paths:
                time_values = collect_filtered_time_values(
                    path,
                    seq_len,
                    args.use_training_preproc,
                    args.filter_column,
                    args.filter_value,
                    args.time_column,
                )
                filtered_time_file_stats.append(
                    {
                        "file_name": os.path.basename(path),
                        "time_values": time_values,
                        "time_counts": np.zeros(0, dtype=int),
                        "n_valid_time": int(time_values.size),
                    }
                )
                filtered_time_values_total.extend(time_values.tolist())

            time_bin_min, time_bin_max, time_edges = build_time_bins(np.asarray(filtered_time_values_total, dtype=np.float64))
            filtered_time_counts_total = np.zeros(len(time_bin_min), dtype=int)
            filtered_time_total_valid = 0
            for stat in filtered_time_file_stats:
                time_values = stat["time_values"]
                counts = compute_time_counts(time_values, time_edges)
                stat["time_counts"] = counts
                filtered_time_counts_total += counts
                filtered_time_total_valid += int(time_values.size)

            filtered_time_summary_df = build_summary_dataframe_filtered_time(
                args.feature,
                args.filter_column,
                args.filter_value,
                time_bin_min,
                time_bin_max,
                filtered_time_counts_total,
                filtered_time_total_valid,
            )
            filtered_time_perfile_df = build_perfile_dataframe_filtered_time(
                args.feature,
                args.filter_column,
                args.filter_value,
                filtered_time_file_stats,
                time_bin_min,
                time_bin_max,
            )

            false_accel_totals_max = np.zeros(10, dtype=int)
            false_accel_totals_all = np.zeros(10, dtype=int)
            false_accel_total_valid_max = 0
            false_accel_total_valid_all = 0
            false_accel_file_stats: List[Dict[str, Any]] = []
            for path in csv_paths:
                max_counts, all_counts, n_valid_max, n_valid_all = analyze_false_accel_continuous_file(
                    path,
                    args.feature,
                    seq_len,
                    args.use_training_preproc,
                    args.filter_column,
                    args.filter_value,
                )
                false_accel_file_stats.append(
                    {
                        "file_name": os.path.basename(path),
                        "max_counts": max_counts,
                        "all_counts": all_counts,
                        "n_valid_max": n_valid_max,
                        "n_valid_all": n_valid_all,
                    }
                )
                false_accel_totals_max += max_counts
                false_accel_totals_all += all_counts
                false_accel_total_valid_max += n_valid_max
                false_accel_total_valid_all += n_valid_all

            false_accel_summary_df = build_summary_dataframe_false_accel_continuous(
                args.feature,
                args.filter_column,
                args.filter_value,
                false_accel_totals_max,
                false_accel_totals_all,
                false_accel_total_valid_max,
                false_accel_total_valid_all,
            )
            false_accel_perfile_df = build_perfile_dataframe_false_accel_continuous(
                args.feature,
                args.filter_column,
                args.filter_value,
                false_accel_file_stats,
            )

    if args.artifacts_dir:
        contribution_frames: List[pd.DataFrame] = []
        for path in csv_paths:
            contribution_frames.append(
                analyze_contribution_file(
                    path,
                    seq_len,
                    model,
                    scaler,
                    layout,
                    device,
                )
            )
        contribution_df = pd.concat(contribution_frames, ignore_index=True) if contribution_frames else pd.DataFrame(
            columns=[
                "file_name",
                "input_feature",
                "recon_feature",
                "same_feature",
                "correlation",
                "mean_abs_error",
                "rmse",
                "n_samples",
            ]
        )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        perfile_df.to_excel(writer, sheet_name="PerFile", index=False)
        if filtered_summary_df is not None and filtered_perfile_df is not None:
            filtered_summary_df.to_excel(writer, sheet_name="SummaryFiltered", index=False)
            filtered_perfile_df.to_excel(writer, sheet_name="PerFileFiltered", index=False)
        if filtered_time_summary_df is not None and filtered_time_perfile_df is not None:
            filtered_time_summary_df.to_excel(writer, sheet_name="SummaryFilteredTime", index=False)
            filtered_time_perfile_df.to_excel(writer, sheet_name="PerFileFilteredTime", index=False)
        if false_accel_summary_df is not None and false_accel_perfile_df is not None:
            false_accel_summary_df.to_excel(writer, sheet_name="SummaryFalseAccel", index=False)
            false_accel_perfile_df.to_excel(writer, sheet_name="PerFileFalseAccel", index=False)
        if contribution_df is not None:
            contribution_df.to_excel(writer, sheet_name="Contribution", index=False)

    print(f"[INFO] Wrote analysis to {output_path}")


if __name__ == "__main__":
    main()
