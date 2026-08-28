import argparse
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
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
LEGACY_DIR = os.path.join(PROJECT_ROOT, "1_transformer")

for _p in (SRC_DIR, PROJECT_ROOT, LEGACY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 学習コード（nstep.py）から関数・クラスをインポート（§1.1）
from gofumi_ae.training.nstep import (  # noqa: E402  実環境に合わせて修正
    initialize_from_definition,
    read_csv_lower,
    require_columns,
    preprocess_df_for_training,
    list_csvs_in_dir,
    SequenceDatasetMasked,
    SequenceDatasetMaskedSegments,
    onehot_encode_df,
)
from gofumi_ae.config import extract_model_config_from_definition  # noqa: E402
from gofumi_ae.datasets.tagged_dataset import (  # noqa: E402
    build_tagged_dataset_csvs_from_config,
    _normalize_file_stem,
)
from models.transformer_autoencoder import CausalTransformerAutoencoder  # noqa: E402
import gofumi_ae.training.nstep as _nstep  # noqa: E402

# 定義ファイルを読み込んでグローバル変数を初期化する（§1.1）
_DEFINITION_PATH = os.path.join(PROJECT_ROOT, "config", "definition.json")
_ANALYZE_CONFIG_PATH = os.path.join(SCRIPT_DIR, "analyze_config.json")
_definition = initialize_from_definition(_DEFINITION_PATH)

# 初期化後にグローバル変数のエイリアスを作成（initialize_from_definition 呼び出し後でないと空のまま）
FEATURES: List[str] = _nstep.FEATURES
FEATURE_RULES: Dict[str, Any] = _nstep.FEATURE_RULES
CATEGORICAL_FEATURES: List[str] = _nstep.CATEGORICAL_FEATURES
DEFAULT_UNKNOWN_ID: float = _nstep.DEFAULT_UNKNOWN_ID
CATEGORY_MAPS: Dict[str, Dict[str, float]] = _nstep.CATEGORY_MAPS
DEFAULT_RULE: Dict[str, Any] = _nstep.DEFAULT_RULE
FRAME_RANGE_CONFIG: Dict[str, Any] = _definition.get('frame_range_config', {})


def load_analyze_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Analyze config must be a JSON object: {config_path}")
    return data


def resolve_setting(cli_value: Any, config: Dict[str, Any], key: str, fallback: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    if key in config and config[key] is not None:
        return config[key]
    return fallback


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

    model = CausalTransformerAutoencoder(
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
    X_cat, _ = onehot_encode_df(df, specs)
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


def get_train_csv_paths(csv: str, csvdir: str, pattern: str, train_filter_config_path: str) -> List[str]:
    source_paths = get_csv_paths(csv, csvdir, pattern)
    tagged_paths, _report = build_tagged_dataset_csvs_from_config(train_filter_config_path, pattern)
    tagged_stems = {
        _normalize_file_stem(os.path.basename(path))
        for path in tagged_paths
    }
    filtered_paths = [
        path for path in source_paths
        if _normalize_file_stem(os.path.basename(path)) in tagged_stems
    ]
    if not filtered_paths:
        raise ValueError(
            "No train CSV files remained after intersecting -i source files with tagged_dataset_train.json"
        )
    return sorted({os.path.normpath(path) for path in filtered_paths})


def extract_range_indices(df: pd.DataFrame, seq_len: int, mode: str) -> Tuple[int, int]:
    filter_col = FRAME_RANGE_CONFIG.get("filter_column", "time")
    start_value = float(FRAME_RANGE_CONFIG.get("start_value", 0))
    end_value = float(FRAME_RANGE_CONFIG.get("end_value", 0))

    filter_col_norm = str(filter_col).strip().lower()
    if filter_col_norm not in df.columns:
        raise ValueError(f"frame_range filter_column '{filter_col}' not found")

    n_rows = int(len(df))
    filter_values = pd.to_numeric(df[filter_col_norm], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(filter_values)
    if not np.any(mask):
        raise ValueError(f"no valid values in frame_range filter column '{filter_col}'")

    actual_indices = np.where(mask)[0]
    frame_idx_start = actual_indices[np.argmin(np.abs(filter_values[mask] - start_value))]
    frame_idx_end = actual_indices[np.argmin(np.abs(filter_values[mask] - end_value))]
    if frame_idx_end <= frame_idx_start:
        raise ValueError(f"invalid frame range [{frame_idx_start}, {frame_idx_end}]")

    if mode == "frame_range":
        extracted_start = frame_idx_start
    else:
        start_value_past = start_value - (seq_len * 0.1)
        frame_idx_past = actual_indices[np.argmin(np.abs(filter_values[mask] - start_value_past))]
        extracted_start = frame_idx_past

    extracted_end = min(frame_idx_end + 1, n_rows)
    return extracted_start, extracted_end


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


def _iter_windows_from_values(values: np.ndarray, seq_len: int):
    """
    SequenceDatasetMasked を使って window を生成するジェネレータ（学習コードと同一仕様、§4.1）。
    """
    if len(values) < seq_len:
        return
    X = values.reshape(-1, 1).astype(np.float32)
    M = np.zeros_like(X)
    dataset = SequenceDatasetMasked(X, M, seq_len)
    for i in range(len(dataset)):
        x, _ = dataset[i]
        yield x[:, 0].numpy()


def _iter_window_pairs(values: np.ndarray, condition_mask: np.ndarray, seq_len: int):
    """
    values と condition_mask を同期して window を生成するジェネレータ（学習コードと同一仕様、§4.1）。
    """
    if len(values) < seq_len:
        return
    X_v = values.reshape(-1, 1).astype(np.float32)
    M_v = np.zeros_like(X_v)
    X_c = condition_mask.astype(np.float32).reshape(-1, 1)
    M_c = np.zeros_like(X_c)
    ds_v = SequenceDatasetMasked(X_v, M_v, seq_len)
    ds_c = SequenceDatasetMasked(X_c, M_c, seq_len)
    for i in range(len(ds_v)):
        x_v, _ = ds_v[i]
        x_c, _ = ds_c[i]
        yield x_v[:, 0].numpy(), x_c[:, 0].numpy().astype(bool)


def analyze_continuous_file(
    path: str,
    feature: str,
    seq_len: int,
    use_training_preproc: bool,
    range_mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    df, values, _, n_rows, _ = load_and_preprocess_single(path, feature, use_training_preproc)
    if range_mode is not None:
        try:
            extracted_start, extracted_end = extract_range_indices(df, seq_len, range_mode)
        except ValueError as exc:
            print(f"[INFO] Skipping {os.path.basename(path)}: {exc}")
            return np.zeros(10, dtype=int), np.zeros(10, dtype=int), 0, 0
        values = values[extracted_start:extracted_end]
        n_rows = len(values)
    bin_min, bin_max, bin_edges = build_continuous_bins(feature)
    max_counts = np.zeros(len(bin_min), dtype=int)
    all_counts = np.zeros(len(bin_min), dtype=int)
    n_valid_max = 0
    n_valid_all = 0
    n_windows = 0
    for window in _iter_windows_from_values(values, seq_len):
        n_windows += 1
        valid = window[np.isfinite(window)]
        if valid.size > 0:
            max_val = valid.max()
            idx = np.digitize([max_val], bin_edges[1:], right=True)[0]
            if 0 <= idx < len(bin_min):
                max_counts[idx] += 1
            n_valid_max += 1
            counts = compute_continuous_counts(valid, feature, bin_edges)
            all_counts += counts
            n_valid_all += valid.size
    if n_windows == 0:
        print(f"[INFO] Skipping {os.path.basename(path)} because row count {n_rows} < seq_len {seq_len}")
    return max_counts, all_counts, n_valid_max, n_valid_all


def analyze_filtered_continuous_file(
    path: str,
    feature: str,
    seq_len: int,
    use_training_preproc: bool,
    filter_column: str,
    filter_value: str,
    time_mode: str = "first_window",
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Filtered continuous feature analysis with time range limitation (frame_range_config).
    
    Args:
        time_mode: "first_window" = start_valueより seq_len分前から end_value までの範囲
                   "frame_range" = [start_value, end_value] 厳密な範囲内
    """
    df, values, _, n_rows, _ = load_and_preprocess_single(path, feature, use_training_preproc)
    filter_column_norm = filter_column.strip().lower()
    if filter_column_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: filter column '{filter_column}' not found")

    # Time range limitation (新規追加)
    filter_col = FRAME_RANGE_CONFIG.get('filter_column', 'time')
    start_value = float(FRAME_RANGE_CONFIG.get('start_value', 0))
    end_value = float(FRAME_RANGE_CONFIG.get('end_value', 0))
    
    filter_col_norm = filter_col.strip().lower()
    if filter_col_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: frame_range filter_column '{filter_col}' not found")
    
    filter_values = pd.to_numeric(df[filter_col_norm], errors="coerce").to_numpy(dtype=float)
    
    # 有効な値のマスク
    mask = np.isfinite(filter_values)
    if not np.any(mask):
        print(f"[INFO] Skipping {os.path.basename(path)}: no valid values in filter column '{filter_col}'")
        return np.zeros(10, dtype=int), np.zeros(10, dtype=int), 0, 0
    
    # start_value に最も近い行のインデックス
    idx_start = np.argmin(np.abs(filter_values[mask] - start_value))
    actual_indices = np.where(mask)[0]
    frame_idx_start = actual_indices[idx_start]
    
    # end_value に最も近い行のインデックス
    idx_end = np.argmin(np.abs(filter_values[mask] - end_value))
    frame_idx_end = actual_indices[idx_end]
    
    # frame_idx_end が frame_idx_start より後ろにあることを確認
    if frame_idx_end <= frame_idx_start:
        print(f"[INFO] Skipping {os.path.basename(path)}: invalid frame range [{frame_idx_start}, {frame_idx_end}]")
        return np.zeros(10, dtype=int), np.zeros(10, dtype=int), 0, 0
    
    # 時間範囲の決定（mode1/mode2で異なる）
    if time_mode == "frame_range":
        # モード2: [start_value, end_value] 範囲内に収める
        extracted_start = frame_idx_start
        extracted_end = min(frame_idx_end + 1, n_rows)
    else:
        # モード1（デフォルト）: start_value より seq_len分前から開始
        start_value_past = start_value - (seq_len * 0.1)
        idx_past = np.argmin(np.abs(filter_values[mask] - start_value_past))
        frame_idx_past = actual_indices[idx_past]
        extracted_start = frame_idx_past
        extracted_end = min(frame_idx_end + 1, n_rows)
    
    # 時間範囲内でのwindow生成
    range_values = values[extracted_start:extracted_end]
    range_condition_mask = build_exact_match_mask(df[filter_column_norm], filter_value)[extracted_start:extracted_end]
    
    if len(range_values) < seq_len:
        print(f"[INFO] Skipping {os.path.basename(path)}: range length {len(range_values)} < seq_len {seq_len}")
        return np.zeros(10, dtype=int), np.zeros(10, dtype=int), 0, 0
    
    bin_min, bin_max, bin_edges = build_continuous_bins(feature)
    max_counts = np.zeros(len(bin_min), dtype=int)
    all_counts = np.zeros(len(bin_min), dtype=int)
    n_valid_max = 0
    n_valid_all = 0
    n_windows = 0

    for window, condition_window in _iter_window_pairs(range_values, range_condition_mask, seq_len):
        n_windows += 1
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

    if n_windows == 0:
        print(f"[INFO] Skipping {os.path.basename(path)} because row count {len(range_values)} < seq_len {seq_len}")
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


def collect_time_values(
    path: str,
    seq_len: int,
    use_training_preproc: bool,
    time_column: str,
    time_mode: str = "first_window",
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> np.ndarray:
    """
    Time分布を集計する（filter条件は optional）。
    
    Args:
        time_mode: "first_window" = start_valueより seq_len分前から end_value までの範囲
                   "frame_range" = [start_value, end_value] 厳密な範囲内
        filter_column: optional - 指定時はこの列で filter_value と一致するフレームのみ集計
        filter_value: optional - filter_column での一致値
    """
    df = read_csv_lower(path)
    require_columns(df, path)

    if use_training_preproc:
        df, _, _ = preprocess_df_for_training(df, os.path.basename(path))

    time_column_norm = time_column.strip().lower()
    if time_column_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: time column '{time_column}' not found")

    n_rows = int(len(df))
    time_series = pd.to_numeric(df[time_column_norm], errors="coerce").to_numpy(dtype=float)
    matched_times: List[float] = []

    # filter_column が指定されている場合の条件マスク
    if filter_column is not None and filter_value is not None:
        filter_column_norm = filter_column.strip().lower()
        if filter_column_norm not in df.columns:
            raise ValueError(f"{os.path.basename(path)}: filter column '{filter_column}' not found")
        condition_mask = build_exact_match_mask(df[filter_column_norm], filter_value)
    else:
        condition_mask = np.ones(n_rows, dtype=bool)

    # frame_range_config から start_value, end_value を取得
    filter_col = FRAME_RANGE_CONFIG.get('filter_column', 'time')
    start_value = float(FRAME_RANGE_CONFIG.get('start_value', 0))
    end_value = float(FRAME_RANGE_CONFIG.get('end_value', 0))
    
    # filter_col のデータを取得（通常は時刻カラム）
    filter_col_norm = filter_col.strip().lower()
    if filter_col_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: frame_range filter_column '{filter_col}' not found")
    
    filter_values = pd.to_numeric(df[filter_col_norm], errors="coerce").to_numpy(dtype=float)
    
    # 有効な値のマスク
    mask = np.isfinite(filter_values)
    if not np.any(mask):
        print(f"[INFO] Skipping {os.path.basename(path)}: no valid values in filter column '{filter_col}'")
        return np.asarray(matched_times, dtype=np.float64)
    
    # start_value に最も近い行のインデックス
    idx_start = np.argmin(np.abs(filter_values[mask] - start_value))
    actual_indices = np.where(mask)[0]
    frame_idx_start = actual_indices[idx_start]
    
    # end_value に最も近い行のインデックス
    idx_end = np.argmin(np.abs(filter_values[mask] - end_value))
    frame_idx_end = actual_indices[idx_end]
    
    # frame_idx_end が frame_idx_start より後ろにあることを確認
    if frame_idx_end <= frame_idx_start:
        print(f"[INFO] Skipping {os.path.basename(path)}: invalid frame range [{frame_idx_start}, {frame_idx_end}]")
        return np.asarray(matched_times, dtype=np.float64)
    
    # 範囲を決定
    if time_mode == "frame_range":
        # モード2: [start_value, end_value] 範囲内に収める
        extracted_start = frame_idx_start
        extracted_end = min(frame_idx_end + 1, n_rows)
    else:
        # モード1（デフォルト）: start_value より seq_len分前から開始
        start_value_past = start_value - (seq_len * 0.1)
        idx_past = np.argmin(np.abs(filter_values[mask] - start_value_past))
        frame_idx_past = actual_indices[idx_past]
        extracted_start = frame_idx_past
        extracted_end = min(frame_idx_end + 1, n_rows)
    
    # windowを生成
    window_time_series = time_series[extracted_start:extracted_end]
    window_condition_mask = condition_mask[extracted_start:extracted_end]
    
    if len(window_time_series) < seq_len:
        print(f"[INFO] Skipping {os.path.basename(path)}: range length {len(window_time_series)} < seq_len {seq_len}")
        return np.asarray(matched_times, dtype=np.float64)
    
    # windowを生成し、条件マスクでフィルタ
    for time_window, condition_window in _iter_window_pairs(window_time_series, window_condition_mask, seq_len):
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
    fp_counts_total: np.ndarray,
    total_valid_fp: int,
) -> pd.DataFrame:
    bin_min, bin_max, _ = build_continuous_bins(feature)
    return pd.DataFrame(
        {
            "feature": [feature] * len(bin_min),
            "condition_column": [filter_column] * len(bin_min),
            "condition_value": [filter_value] * len(bin_min),
            "bin_min": bin_min,
            "bin_max": bin_max,
            "fp_count_total": fp_counts_total,
            "fp_ratio_total": np.divide(fp_counts_total, total_valid_fp, out=np.zeros_like(fp_counts_total, dtype=float), where=total_valid_fp > 0),
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
            count = int(stat["fp_counts"][idx])
            rows.append(
                {
                    "file_name": stat["file_name"],
                    "feature": feature,
                    "condition_column": filter_column,
                    "condition_value": filter_value,
                    "bin_min": bin_min[idx],
                    "bin_max": bin_max[idx],
                    "fp_count": count,
                    "fp_ratio": count / stat["n_valid_fp"] if stat["n_valid_fp"] > 0 else 0.0,
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
    time_mode: str = "first_window",
) -> Tuple[np.ndarray, int]:
    """
    誤検知分布: window 構造は用いず、フレーム行単位で集計する（仕様§5.1.5）。
    時間範囲制限（frame_range_config）を適用。
    
    Args:
        time_mode: "first_window" = start_valueより seq_len分前から end_value までの範囲
                   "frame_range" = [start_value, end_value] 厳密な範囲内
    """
    df, values, _, n_rows, _ = load_and_preprocess_single(path, feature, use_training_preproc)
    filter_column_norm = filter_column.strip().lower()
    if filter_column_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: filter column '{filter_column}' not found")

    # Time range limitation
    filter_col = FRAME_RANGE_CONFIG.get('filter_column', 'time')
    start_value = float(FRAME_RANGE_CONFIG.get('start_value', 0))
    end_value = float(FRAME_RANGE_CONFIG.get('end_value', 0))
    
    filter_col_norm = filter_col.strip().lower()
    if filter_col_norm not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: frame_range filter_column '{filter_col}' not found")
    
    filter_values = pd.to_numeric(df[filter_col_norm], errors="coerce").to_numpy(dtype=float)
    
    # 有効な値のマスク
    mask = np.isfinite(filter_values)
    if not np.any(mask):
        print(f"[INFO] Skipping {os.path.basename(path)}: no valid values in filter column '{filter_col}'")
        return np.zeros(10, dtype=int), 0
    
    # start_value に最も近い行のインデックス
    idx_start = np.argmin(np.abs(filter_values[mask] - start_value))
    actual_indices = np.where(mask)[0]
    frame_idx_start = actual_indices[idx_start]
    
    # end_value に最も近い行のインデックス
    idx_end = np.argmin(np.abs(filter_values[mask] - end_value))
    frame_idx_end = actual_indices[idx_end]
    
    # frame_idx_end が frame_idx_start より後ろにあることを確認
    if frame_idx_end <= frame_idx_start:
        print(f"[INFO] Skipping {os.path.basename(path)}: invalid frame range [{frame_idx_start}, {frame_idx_end}]")
        return np.zeros(10, dtype=int), 0
    
    # 時間範囲の決定（mode1/mode2で異なる）
    if time_mode == "frame_range":
        # モード2: [start_value, end_value] 範囲内に収める
        extracted_start = frame_idx_start
        extracted_end = min(frame_idx_end + 1, n_rows)
    else:
        # モード1（デフォルト）: start_value より seq_len分前から開始
        start_value_past = start_value - (seq_len * 0.1)
        idx_past = np.argmin(np.abs(filter_values[mask] - start_value_past))
        frame_idx_past = actual_indices[idx_past]
        extracted_start = frame_idx_past
        extracted_end = min(frame_idx_end + 1, n_rows)
    
    # 時間範囲内でのフレーム抽出
    range_df = df.iloc[extracted_start:extracted_end]
    range_values = values[extracted_start:extracted_end]
    
    condition_mask = build_exact_match_mask(range_df[filter_column_norm], filter_value)
    fp_values = range_values[condition_mask]

    bin_min, bin_max, bin_edges = build_continuous_bins(feature)
    fp_counts = compute_continuous_counts(fp_values, feature, bin_edges)
    n_valid_fp = int(fp_counts.sum())

    return fp_counts, n_valid_fp


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze accel distribution from CSVs")
    parser.add_argument("--csv", "-i", required=True, help="Single CSV file or directory containing CSVs")
    parser.add_argument("--csvdir", "-d", default="", help="Directory containing CSVs; takes precedence over --csv")
    parser.add_argument("--pattern", default=None, help="Glob pattern for CSV discovery")
    parser.add_argument("--hparams", default=None, help="Hyperparameters JSON path")
    parser.add_argument("--feature", default=None, help="Feature name to analyze")
    parser.add_argument("--output-dir", default=None, help="Output directory for Excel")
    parser.add_argument("--output-prefix", default=None, help="Output filename prefix override")
    parser.add_argument("--use-training-preproc", action="store_true", help="Use existing training preprocessing")
    parser.add_argument("--filtered-mode", action="store_true", help="Switch to inference analysis mode")
    parser.add_argument("--filter-column", default=None, help="Condition column for inference analysis")
    parser.add_argument("--filter-value", default=None, help="Exact-match condition value for inference analysis")
    parser.add_argument("--time-column", default=None, help="Time column used for inference Time distribution")
    parser.add_argument("--time-mode", default=None, choices=["first_window", "frame_range"], help="Inference time extraction mode")
    parser.add_argument("--train-window-mode", default=None, choices=["mode1", "mode2"], help="Train window mode")
    parser.add_argument("--analyze-config", default=_ANALYZE_CONFIG_PATH, help="Analyze config JSON path")
    parser.add_argument("--train-filter-config", default=None, help="Train filter config JSON path")
    parser.add_argument("--artifacts-dir", default="", help="Directory containing trained model artifacts for contribution analysis")
    args = parser.parse_args()

    analyze_config = load_analyze_config(args.analyze_config)

    pattern = resolve_setting(args.pattern, analyze_config, "pattern", "*.csv")
    hparams_path = resolve_setting(args.hparams, analyze_config, "hparams_path", "config/hyperparams_common.json")
    feature = resolve_setting(args.feature, analyze_config, "feature", "accelpedalangle")
    output_dir = resolve_setting(args.output_dir, analyze_config, "output_dir", "AE/outputs")
    time_mode = resolve_setting(args.time_mode, analyze_config, "time_mode", "first_window")
    time_column = resolve_setting(args.time_column, analyze_config, "time_column", "time")
    train_window_mode = resolve_setting(args.train_window_mode, analyze_config, "train_window_mode", "mode1")
    use_training_preproc = bool(args.use_training_preproc or analyze_config.get("use_training_preproc", False))
    train_filter_config = resolve_setting(
        args.train_filter_config,
        analyze_config,
        "train_filter_config_path",
        os.path.join(PROJECT_ROOT, "config", "tagged_dataset_train.json"),
    )
    filter_column = resolve_setting(args.filter_column, analyze_config, "inference_filter_column", None)
    filter_value = resolve_setting(args.filter_value, analyze_config, "inference_filter_value", None)

    if feature not in FEATURES:
        raise ValueError(f"Feature '{feature}' is not in FEATURES")
    if args.filtered_mode and feature in CATEGORICAL_FEATURES:
        raise ValueError("Inference mode currently supports continuous features only")
    if args.filtered_mode and (not filter_column or filter_value is None):
        raise ValueError("Inference mode requires filter-column and filter-value via CLI or analyze_config.json")

    # seq_len は定義ファイルから取得し、フォールバックとして hparams を参照する（§1.2）
    try:
        _model_cfg = extract_model_config_from_definition(_definition)
        seq_len = int(_model_cfg.get("seq_len", 128))
    except Exception:
        hparams = load_hparams(hparams_path)
        seq_len = int(hparams.get("seq_len", 128))

    if args.filtered_mode:
        csv_paths = get_csv_paths(args.csv, args.csvdir, pattern)
        output_prefix = resolve_setting(args.output_prefix, analyze_config, "output_prefix_inference", "accel_dist_inference")
    else:
        csv_paths = get_train_csv_paths(args.csv, args.csvdir, pattern, train_filter_config)
        output_prefix = resolve_setting(args.output_prefix, analyze_config, "output_prefix_train", "accel_dist_train")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_prefix}_{feature}.xlsx")

    is_categorical = feature in CATEGORICAL_FEATURES
    inference_summary_df = None
    inference_perfile_df = None
    inference_time_summary_df = None
    inference_time_perfile_df = None
    contribution_df = None
    model = None
    scaler = None
    layout = None
    device = None
    if args.artifacts_dir:
        model, scaler, layout, device = load_model_artifacts(args.artifacts_dir)
    if is_categorical:
        totals: np.ndarray = np.zeros(len(CATEGORY_MAPS[feature]), dtype=int)
        class_names = list(CATEGORY_MAPS[feature].keys())
        file_stats: List[Dict[str, Any]] = []
        for path in csv_paths:
            counts, names, total = analyze_categorical_file(path, feature, use_training_preproc)
            file_stats.append({"file_name": os.path.basename(path), "counts": counts, "total": total})
            totals += counts
        summary_df = build_summary_dataframe_categorical(feature, class_names, totals)
        perfile_df = build_perfile_dataframe_categorical(feature, file_stats, class_names)
    else:
        totals_max = np.zeros(10, dtype=int)
        totals_all = np.zeros(10, dtype=int)
        total_valid_max = 0
        total_valid_all = 0
        file_stats: List[Dict[str, Any]] = []
        if args.filtered_mode:
            inference_totals_max = np.zeros(10, dtype=int)
            inference_totals_all = np.zeros(10, dtype=int)
            inference_total_valid_max = 0
            inference_total_valid_all = 0
            inference_file_stats: List[Dict[str, Any]] = []
            time_values_total: List[float] = []
            time_file_stats: List[Dict[str, Any]] = []
            for path in csv_paths:
                max_counts, all_counts, n_valid_max, n_valid_all = analyze_filtered_continuous_file(
                    path,
                    feature,
                    seq_len,
                    use_training_preproc,
                    filter_column,
                    str(filter_value),
                    time_mode=time_mode,
                )
                inference_file_stats.append(
                    {
                        "file_name": os.path.basename(path),
                        "max_counts": max_counts,
                        "all_counts": all_counts,
                        "n_valid_max": n_valid_max,
                        "n_valid_all": n_valid_all,
                    }
                )
                inference_totals_max += max_counts
                inference_totals_all += all_counts
                inference_total_valid_max += n_valid_max
                inference_total_valid_all += n_valid_all

                time_values = collect_time_values(
                    path,
                    seq_len,
                    use_training_preproc,
                    time_column,
                    time_mode=time_mode,
                    filter_column=filter_column,
                    filter_value=str(filter_value),
                )
                time_file_stats.append(
                    {
                        "file_name": os.path.basename(path),
                        "time_values": time_values,
                        "time_counts": np.zeros(0, dtype=int),
                        "n_valid_time": int(time_values.size),
                    }
                )
                time_values_total.extend(time_values.tolist())

            summary_df = build_summary_dataframe_filtered_continuous(
                feature,
                filter_column,
                str(filter_value),
                inference_totals_max,
                inference_totals_all,
                inference_total_valid_max,
                inference_total_valid_all,
            )
            perfile_df = build_perfile_dataframe_filtered_continuous(
                feature,
                filter_column,
                str(filter_value),
                inference_file_stats,
            )

            time_bin_min, time_bin_max, time_edges = build_time_bins(np.asarray(time_values_total, dtype=np.float64))
            time_counts_total = np.zeros(len(time_bin_min), dtype=int)
            time_total_valid = 0
            for stat in time_file_stats:
                time_values = stat["time_values"]
                counts = compute_time_counts(time_values, time_edges)
                stat["time_counts"] = counts
                time_counts_total += counts
                time_total_valid += int(time_values.size)

            inference_summary_df = summary_df
            inference_perfile_df = perfile_df
            inference_time_summary_df = build_summary_dataframe_filtered_time(
                feature,
                filter_column,
                str(filter_value),
                time_bin_min,
                time_bin_max,
                time_counts_total,
                time_total_valid,
            )
            inference_time_perfile_df = build_perfile_dataframe_filtered_time(
                feature,
                filter_column,
                str(filter_value),
                time_file_stats,
                time_bin_min,
                time_bin_max,
            )
        else:
            range_mode = "first_window" if train_window_mode == "mode1" else "frame_range"
            for path in csv_paths:
                max_counts, all_counts, n_valid_max, n_valid_all = analyze_continuous_file(
                    path,
                    feature,
                    seq_len,
                    use_training_preproc,
                    range_mode=range_mode,
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
                feature,
                totals_max,
                totals_all,
                total_valid_max,
                total_valid_all,
            )
            perfile_df = build_perfile_dataframe_continuous(feature, file_stats)

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
        if args.filtered_mode:
            summary_df.to_excel(writer, sheet_name="SummaryInference", index=False)
            perfile_df.to_excel(writer, sheet_name="PerFileInference", index=False)
            if inference_time_summary_df is not None and inference_time_perfile_df is not None:
                inference_time_summary_df.to_excel(writer, sheet_name="SummaryInferenceTime", index=False)
                inference_time_perfile_df.to_excel(writer, sheet_name="PerFileInferenceTime", index=False)
        else:
            summary_df.to_excel(writer, sheet_name="SummaryTrain", index=False)
            perfile_df.to_excel(writer, sheet_name="PerFileTrain", index=False)
        if contribution_df is not None:
            contribution_df.to_excel(writer, sheet_name="Contribution", index=False)

    print(f"[INFO] Wrote analysis to {output_path}")


if __name__ == "__main__":
    main()
