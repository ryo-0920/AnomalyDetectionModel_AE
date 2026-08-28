import argparse
import datetime
import json
import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import sys
import shutil
from contextlib import nullcontext
from models.transformer_autoencoder import CausalTransformerAutoencoder
# =========================================================
# 1) 定数・パス
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (BASE_DIR / "..").resolve()
DEFAULT_INPUT_DIR = (BASE_DIR / ".." / "datarecode_test").resolve()
DEFAULT_HPARAMS_PATH = (PROJECT_ROOT / "config" / "hyperparams_common.json").resolve()
DEFAULT_TAGGED_DATASET_INFERENCE_CONFIG_PATH = (PROJECT_ROOT / "config" / "tagged_dataset_inference.json").resolve()
DEFAULT_INFER_RUNTIME_CONFIG_PATH = (PROJECT_ROOT / "config" / "inference_runtime.json").resolve()
LEGACY_DEFAULT_ARTIFACTS_DIR = (PROJECT_ROOT / "artifacts" / "transformer_ae").resolve()
DEFAULT_VALID_RESULTS_DIR = (PROJECT_ROOT / "output" / "Valid_results").resolve()
MODEL_NAME = "transformer_ae"
DEFAULT_UNKNOWN_ID = -1.0
MAX_LOG_WIDTH = 70
Y_PRE_EWMA_WINDOW = 5
Y_PRE_THRESHOLD_SCALE = 0.6
Y_PRE_CONSECUTIVE = 3
REQUIRED_SCORE_POLICY = {
    "mae_target": "all_features",
    "feature_weights": {
        "continuous": 1.0,
        "categorical": 1.0,
    },
    "threshold_source": "full_training_dataset",
    "legacy_compatibility": False,
}
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.ui.interactive import TAGGED_DATASET_TOKEN
from app.ui.interactive import ensure_tty, prompt_artifacts_dir, prompt_csv_or_dir_or_glob
from app.tagged_dataset import build_tagged_dataset_csvs_from_config, sample_paths_interactively
# =========================================================
# 2) ユーティリティ（欠損マスク拡張・カテゴリ正規化・欠損行削除）
# =========================================================
def expand_mask_for_onehot_precomputed(
    mask: np.ndarray,
    idx_cont: List[int],
    categorical_features: List[str],
    specs: Dict[str, Dict[str, Any]],
    offsets: Dict[str, int],
    features: List[str],
) -> np.ndarray:
    """
    欠損マスクを one-hot 展開後の次元に合わせて拡張する。
    - 連続値: 指定インデックスを抽出
    - カテゴリ: その列のマスクを one-hot 次元分に複製
    """
    N = mask.shape[0]
    mask_cont = mask[:, idx_cont].astype(np.float32)
    total_cat_dim = sum(int(specs[col]["dim"]) for col in categorical_features)
    mask_cat = np.zeros((N, total_cat_dim), dtype=np.float32)
    for col in categorical_features:
        src_j = features.index(col)
        src_mask_col = mask[:, src_j:src_j + 1]  # (N,1)
        dim = int(specs[col]["dim"])
        off = int(offsets[col])
        mask_cat[:, off:off + dim] = np.repeat(src_mask_col, dim, axis=1)
    return np.concatenate([mask_cont, mask_cat], axis=1).astype(np.float32)
def apply_categorical_mapping(
    df: pd.DataFrame,
    categorical_features: List[str],
    category_maps: Dict[str, Dict[str, float]],
    default_unknown_id: float = DEFAULT_UNKNOWN_ID,
) -> pd.DataFrame:
    """
    学習時の category_maps を用いてカテゴリ列を正規化（文字列→コード化）。
    数値列は数値化し、未知は default_unknown_id で埋める。
    """
    for col in categorical_features:
        if col not in df.columns:
            continue
        col_series = df[col]
        if np.issubdtype(col_series.dtype, np.number):
            df[col] = pd.to_numeric(col_series, errors="coerce").fillna(default_unknown_id).astype(np.float32)
        else:
            s = col_series.astype(str).str.strip().str.upper()
            if col in category_maps:
                mapped = s.map(category_maps[col])
                unknown_id = float(category_maps[col].get("UNKNOWN", default_unknown_id))
                df[col] = mapped.where(~mapped.isna(), unknown_id).astype(np.float32)
            else:
                df[col] = pd.to_numeric(s, errors="coerce").fillna(default_unknown_id).astype(np.float32)
    return df
def drop_rows_with_missing_features(df: pd.DataFrame, feature_cols: List[str], context: str = "") -> pd.DataFrame:
    """
    全列 NaN の行を削除し、連番にリセットして返す。
    """
    before = len(df)
    df2 = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)
    dropped = before - len(df2)
    if dropped > 0:
        msg_ctx = f" [{context}]" if context else ""
        # print(f"[INFO]{msg_ctx} 欠損を含む行を {dropped} 件削除しました（残り {len(df2)} 行）")
    if len(df2) == 0:
        raise ValueError(f"{context}: 欠損行の削除によりデータが空になりました")
    return df2
class InlineLogger:
    def __init__(self):
        self.prev_len = 0
    def print(self, text: str):
        # 端末幅に合わせて必要なら省略
        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        max_len = max(10, cols - 2)
        if len(text) > max_len:
            text = text[:max_len - 3] + "..."
        pad = max(0, self.prev_len - len(text))
        sys.stdout.write("\r" + text + " " * pad)
        sys.stdout.flush()
        self.prev_len = len(text)
    def newline(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.prev_len = 0
# =========================================================
# 3) アーティファクト読み込み・仕様復元
# =========================================================
def load_artifacts(artifacts_dir: str) -> Tuple[Dict, Dict, "StandardScaler", Dict]:
    """
    アーティファクト（config.json, threshold.json, scaler.pkl, model.pt の存在確認）を読み込み、
    設定とスケーラ、しきい値情報を返す。モデルは別関数で構築・ロード。
    """
    cfg_path = os.path.join(artifacts_dir, "config.json")
    thr_path = os.path.join(artifacts_dir, "threshold.json")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    model_path = os.path.join(artifacts_dir, "model.pt")
    if not (os.path.exists(cfg_path) and os.path.exists(thr_path) and os.path.exists(scaler_path) and os.path.exists(model_path)):
        raise FileNotFoundError(
            f"Artifacts not found. Expected files: {cfg_path}, {thr_path}, {scaler_path}, {model_path}"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(thr_path, "r", encoding="utf-8") as f:
        thr_info = json.load(f)
    scaler = joblib.load(scaler_path)
    paths = {"cfg_path": cfg_path, "thr_path": thr_path, "scaler_path": scaler_path, "model_path": model_path}
    return cfg, thr_info, scaler, paths
def resolve_default_artifacts_dir() -> str:
    """
    Prefer config/hyperparams_common.json:out_dir when available.
    Fallback to legacy artifacts/transformer_ae.
    """
    if DEFAULT_HPARAMS_PATH.exists():
        try:
            with open(DEFAULT_HPARAMS_PATH, "r", encoding="utf-8-sig") as f:
                hparams = json.load(f)
            out_dir = hparams.get("out_dir", None)
            if isinstance(out_dir, str) and out_dir.strip():
                out_path = Path(out_dir.strip()).expanduser()
                if not out_path.is_absolute():
                    out_path = (PROJECT_ROOT / out_path).resolve()
                return str(out_path)
        except Exception:
            pass
    return str(LEGACY_DEFAULT_ARTIFACTS_DIR)
# コンパイル/DP 由来のプレフィックスを除去
def _strip_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(sd.keys())
    # torch.compile: _orig_mod.
    if any(k.startswith("_orig_mod.") for k in keys):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        keys = list(sd.keys())
    # DataParallel: module.
    if any(k.startswith("module.") for k in keys):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd
def _normalize_score_policy(policy: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(policy, dict):
        return None
    weights = policy.get("feature_weights", {})
    if not isinstance(weights, dict):
        return None
    try:
        return {
            "mae_target": str(policy.get("mae_target", "")).strip().lower(),
            "feature_weights": {
                "continuous": float(weights.get("continuous")),
                "categorical": float(weights.get("categorical")),
            },
            "threshold_source": str(policy.get("threshold_source", "")).strip().lower(),
            "legacy_compatibility": bool(policy.get("legacy_compatibility")),
        }
    except Exception:
        return None
def _validate_score_policy(cfg: Dict[str, Any], thr_info: Dict[str, Any]) -> Dict[str, Any]:
    required = _normalize_score_policy(REQUIRED_SCORE_POLICY)
    cfg_policy = _normalize_score_policy(cfg.get("score_policy"))
    thr_policy = _normalize_score_policy(thr_info.get("score_policy"))
    if cfg_policy is None or thr_policy is None:
        raise ValueError(
            "Artifacts are missing score_policy. Legacy artifacts are not supported. Please retrain and regenerate artifacts."
        )
    if cfg_policy != thr_policy:
        raise ValueError(f"score_policy mismatch between config.json and threshold.json: cfg={cfg_policy}, thr={thr_policy}")
    if cfg_policy != required:
        raise ValueError(f"Unsupported score_policy: {cfg_policy}. Required: {required}")
    return cfg_policy
def build_inference_context(
    cfg: Dict,
    thr_info: Dict,
    scaler,
    model_path: str,
    device: str,
) -> Dict[str, Any]:
    """
    config と threshold 情報から、推論に必要な全前計算（モデル・one-hot仕様・ルール配列など）を構築して返す。
    戻り値はコンテキスト辞書。
    """
    score_policy = _validate_score_policy(cfg, thr_info)
    # 基本レイアウト・設定
    features: List[str] = [c.lower() for c in cfg.get("features", [])]
    if not features:
        raise ValueError("features not found in config.json")
    layout = cfg["layout"]
    CONTINUOUS_FEATURES = layout["continuous_features"]
    CATEGORICAL_FEATURES = layout["categorical_features"] if "categorical_features" in layout else layout.get("categororical_features", [])
    onehot_specs = layout["onehot_specs"]
    offsets = layout["offsets"]
    input_dim = int(layout["input_dim"])
    seq_len: int = int(cfg.get("seq_len", 50))
    # モデル構築・ロード
    model = CausalTransformerAutoencoder(
        input_dim=input_dim,
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 2)),
        dim_feedforward=int(cfg.get("dim_ff", 512)),
        dropout=float(cfg.get("dropout", 0.2)),
        max_len=seq_len,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        # もし torch.save で {"state_dict": ...} 形式にしていた場合の互換
        state = state["state_dict"]
    state = _strip_prefixes(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    use_amp = (device == "cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # one-hot 用 specs（value_to_index）復元
    specs = {
        col: {
            "codes": [float(v) for v in onehot_specs[col]["codes"]],
            "value_to_index": {float(v): i for i, v in enumerate(onehot_specs[col]["codes"])},
            "dim": int(onehot_specs[col]["dim"]),
        } for col in CATEGORICAL_FEATURES
    }
    # ルールの前計算
    feature_rules = cfg.get("feature_rules", {})
    default_rule = {"sentinels": [], "vmin": None, "vmax": None, "interp": "linear", "clip": False}
    F = len(features)
    vmin_arr = np.full(F, -np.inf, dtype=np.float32)
    vmax_arr = np.full(F,  np.inf, dtype=np.float32)
    sentinels_map = [set() for _ in range(F)]
    for j, col in enumerate(features):
        rule = {**default_rule, **feature_rules.get(col, {})}
        if rule.get("vmin") is not None:
            vmin_arr[j] = float(rule["vmin"])
        if rule.get("vmax") is not None:
            vmax_arr[j] = float(rule["vmax"])
        sents = rule.get("sentinels", [])
        sentinels_map[j] = set(float(x) for x in sents)
    # 連続列/カテゴリ列のインデックス
    idx_cont = [features.index(f) for f in CONTINUOUS_FEATURES]
    cont_idx_by_name = {f: i for i, f in enumerate(CONTINUOUS_FEATURES)}
    category_maps = cfg.get("category_maps", {})
    unknown_id_by_col = {col: float(category_maps.get(col, {}).get("UNKNOWN", DEFAULT_UNKNOWN_ID)) for col in CATEGORICAL_FEATURES}
    cat_feature_idx = {col: features.index(col) for col in CATEGORICAL_FEATURES}
    unknown_idx_by_col = {
        col: specs[col]["value_to_index"].get(float(unknown_id_by_col[col]), 0)
        for col in CATEGORICAL_FEATURES
    }
    total_cat_dim = sum(specs[col]["dim"] for col in CATEGORICAL_FEATURES)
    # 閾値・統計（compute_threshold_on_dataset 保存値）
    threshold = float(thr_info["threshold"])
    mean = float(thr_info.get("mean", 0.0))
    std  = float(thr_info.get("std", 1.0))
    p10 = float(thr_info.get("p10", mean))
    p50 = float(thr_info.get("p50", mean))
    p90 = float(thr_info.get("p90", mean + 2.0*std))
    p99 = float(thr_info.get("p99", mean + 3.0*std))
    # tail_steps を threshold.json から取得（無ければ 1）
    tail_steps = int(thr_info.get("tail_steps", 1))
    if tail_steps < 1:
        tail_steps = 1
    temperature = float(thr_info.get("temperature", max((p90 - p50) / 6.0, 1e-6)))
    y_conv_threshold = float(np.clip((threshold - p10) / max(p99 - p10, 1e-6), 0.0, 1.0))
    y_pre_threshold = float(np.clip(y_conv_threshold * Y_PRE_THRESHOLD_SCALE, 0.0, 1.0))
    # コンテキスト辞書として返す
    return {
        "features": features,
        "CONTINUOUS_FEATURES": CONTINUOUS_FEATURES,
        "CATEGORICAL_FEATURES": CATEGORICAL_FEATURES,
        "specs": specs,
        "offsets": offsets,
        "input_dim": input_dim,
        "seq_len": seq_len,
        "model": model,
        "scaler": scaler,
        "device": device,
        # AMP
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
        "feature_rules": feature_rules,
        "vmin_arr": vmin_arr,
        "vmax_arr": vmax_arr,
        "sentinels_map": sentinels_map,
        "idx_cont": idx_cont,
        "cont_idx_by_name": cont_idx_by_name,
        "cat_feature_idx": cat_feature_idx,
        "unknown_id_by_col": unknown_id_by_col,
        "unknown_idx_by_col": unknown_idx_by_col,
        "total_cat_dim": total_cat_dim,
        "threshold": threshold,
        "mean": mean,
        "std": std,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "temperature": temperature,
        "y_conv_threshold": y_conv_threshold,
        "y_pre_threshold": y_pre_threshold,
        "y_pre_ewma_window": Y_PRE_EWMA_WINDOW,
        "y_pre_consecutive": Y_PRE_CONSECUTIVE,
        "category_maps": category_maps,
        "score_policy": score_policy,
        "tail_steps": tail_steps,
    }
# =========================================================
# X) 推論時 time 範囲設定の読み込み・適用
# =========================================================
def load_inference_time_range_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    tagged dataset 設定から inference_time_range 設定を読む。
    設定が無い / enabled=false / 読込エラー の場合は None を返す。
    """
    path = Path(config_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None
    infer_cfg = cfg.get("infer")
    if isinstance(infer_cfg, dict):
        cfg = infer_cfg
    tr = cfg.get("inference_time_range")
    if not isinstance(tr, dict):
        return None
    if not tr.get("enabled", False):
        return None
    column = tr.get("column", "time")
    start = tr.get("start", None)
    end = tr.get("end", None)
    def _to_opt_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None
    return {
        "column": str(column),
        "start": _to_opt_float(start),
        "end": _to_opt_float(end),
        "expected_step": float(tr.get("expected_step", 0.1)),
        "tolerance": float(tr.get("tolerance", 0.02)),
        "irregular_time_policy": str(tr.get("irregular_time_policy", "warn")).strip().lower(),
    }


def load_inference_runtime_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _check_time_axis_irregular(
    time_values: np.ndarray,
    expected_step: float,
    tolerance: float,
) -> Tuple[bool, Optional[float]]:
    finite = np.isfinite(time_values)
    if int(finite.sum()) < 3:
        return False, None
    t = time_values[finite]
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0.0]
    if dt.size == 0:
        return False, None
    max_dev = float(np.max(np.abs(dt - float(expected_step))))
    return max_dev > float(tolerance), max_dev
# =========================================================
# 4) 入力列挙（単一ファイル／ディレクトリ・再帰）
# =========================================================
def list_input_files(csv_arg: str) -> List[str]:
    """
    --csv がファイルならそのパス、ディレクトリなら配下の CSV を列挙して返す。
    recursive=True ならサブディレクトリまで探索。
    """
    input_path = Path(csv_arg).expanduser()
    if input_path.is_dir():
        # Directory input is always scanned recursively.
        iterator = input_path.rglob("*")
        files = sorted(
            {
                str(p)
                for p in iterator
                if p.is_file() and p.suffix.lower() == ".csv"
            }
        )
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path} (recursive scan enabled)")
        return files
    elif input_path.is_file():
        return [str(input_path)]
    else:
        raise FileNotFoundError(f"--csv not found: {input_path.resolve(strict=False)}")


def _resolve_input_directory(csv_arg: str) -> Optional[str]:
    normalized_arg = str(csv_arg or "").strip()
    if len(normalized_arg) >= 2 and normalized_arg[0] == normalized_arg[-1] and normalized_arg[0] in {'"', "'"}:
        normalized_arg = normalized_arg[1:-1].strip()
    if any(ch in normalized_arg for ch in "*?[]"):
        return None
    input_path = Path(normalized_arg).expanduser()
    if input_path.is_dir():
        return str(input_path)
    return None
def _normalize_path_key(path_str: str) -> str:
    return os.path.normpath(str(path_str)).replace("\\", "/").lower()
def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        num = float(text)
    except Exception:
        return None
    if np.isnan(num):
        return None
    return num
def load_optional_meta_map(meta_csv_path: str, project_root: str) -> Dict[str, Dict[str, Any]]:
    """
    Optional metadata CSV loader.
    Supported path columns: relative_path, path, csv_path, file
    Optional columns: label, abnormal_start_time, collision_time
    """
    if not meta_csv_path:
        return {}
    path_obj = Path(meta_csv_path).expanduser()
    if not path_obj.is_absolute():
        path_obj = (Path(project_root) / path_obj).resolve()
    if not path_obj.exists():
        print(f"[INFO] meta csv not found (optional): {path_obj}")
        return {}
    df_meta = pd.read_csv(path_obj, encoding="utf-8-sig")
    col_map = {c.strip().lower(): c for c in df_meta.columns}
    path_col = None
    for candidate in ("relative_path", "path", "csv_path", "file"):
        if candidate in col_map:
            path_col = col_map[candidate]
            break
    if path_col is None:
        print(f"[WARN] meta csv has no path column: {path_obj}")
        return {}
    label_col = col_map.get("label")
    ast_col = col_map.get("abnormal_start_time")
    ct_col = col_map.get("collision_time")
    meta_map: Dict[str, Dict[str, Any]] = {}
    loaded_rows = 0
    for _, row in df_meta.iterrows():
        raw_path = str(row[path_col]).strip()
        if not raw_path:
            continue
        key = _normalize_path_key(raw_path)
        record = {
            "label": None,
            "abnormal_start_time": _coerce_optional_float(row[ast_col]) if ast_col else None,
            "collision_time": _coerce_optional_float(row[ct_col]) if ct_col else None,
            "source": "meta_csv",
            "meta_csv": str(path_obj),
        }
        if label_col:
            try:
                record["label"] = int(float(row[label_col]))
            except Exception:
                record["label"] = None
        meta_map[key] = record
        # basename fallback key
        meta_map[_normalize_path_key(os.path.basename(raw_path))] = record
        loaded_rows += 1
    print(f"[INFO] meta csv loaded: {path_obj} (rows={loaded_rows})")
    return meta_map
def infer_label_from_path(path_str: str) -> Optional[int]:
    p = _normalize_path_key(path_str)
    has_abnormal = any(token in p for token in ("accident", "accidents", "abnormal", "anomaly"))
    has_normal = "normal" in p
    if has_abnormal and not has_normal:
        return 1
    if has_normal and not has_abnormal:
        return 0
    return None
def infer_output_suffix_from_path(path_str: str) -> str:
    """
    Output filename suffix policy requested by user:
    - path contains accidents/accident -> accidents
    - path contains normal -> normal
    - otherwise -> anomaly (legacy fallback)
    """
    p = _normalize_path_key(path_str)
    if "accidents" in p or "accident" in p:
        return "accidents"
    if "normal" in p:
        return "normal"
    return "anomaly"
def resolve_label_and_meta(
    csv_path: str,
    project_root: str,
    meta_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    abs_key = _normalize_path_key(csv_path)
    try:
        rel_key = _normalize_path_key(os.path.relpath(csv_path, project_root))
    except ValueError:
        # 別ドライブ／別マウントの場合は relpath を諦める
        rel_key = None
    base_key = _normalize_path_key(os.path.basename(csv_path))
    candidates = [abs_key, rel_key, base_key]
    for key in candidates:
        rec = meta_map.get(key)
        if rec is not None:
            label = rec.get("label", None)
            if label in (0, 1):
                return {
                    "label": int(label),
                    "label_source": "meta_csv",
                    "abnormal_start_time": rec.get("abnormal_start_time"),
                    "collision_time": rec.get("collision_time"),
                }
    inferred = infer_label_from_path(csv_path)
    return {
        "label": inferred,
        "label_source": "path_rule" if inferred in (0, 1) else "unknown",
        "abnormal_start_time": None,
        "collision_time": None,
    }
def compute_ewma_scores(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    w = max(1, int(window))
    alpha = 2.0 / (w + 1.0)
    prev = np.nan
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        if np.isfinite(prev):
            prev = alpha * float(v) + (1.0 - alpha) * prev
        else:
            prev = float(v)
        out[i] = float(prev)
    return out
def compute_consecutive_flags(values: np.ndarray, threshold: float, required: int) -> np.ndarray:
    req = max(1, int(required))
    out = np.full(values.shape, np.nan, dtype=np.float32)
    run = 0
    for i, v in enumerate(values):
        if not np.isfinite(v):
            run = 0
            continue
        if float(v) >= float(threshold):
            run += 1
        else:
            run = 0
        out[i] = 1.0 if run >= req else 0.0
    return out
def pick_file_probability(scores: np.ndarray, mode: str) -> Optional[float]:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return None
    m = (mode or "tail").strip().lower()
    if m == "max":
        return float(np.max(finite))
    return float(finite[-1])
def _extract_time_axis(df_out: pd.DataFrame) -> np.ndarray:
    if "time" in df_out.columns:
        t = pd.to_numeric(df_out["time"], errors="coerce").to_numpy(dtype=np.float32)
        if np.isfinite(t).any():
            return t
    return np.arange(len(df_out), dtype=np.float32)
def compute_decision_times_row(
    index_value: int,
    df_out: pd.DataFrame,
    label: Optional[int],
    abnormal_start_time: Optional[float],
    collision_time: Optional[float],
) -> List[Any]:
    times = _extract_time_axis(df_out)
    y_pre = pd.to_numeric(df_out.get("y_pre", pd.Series(np.nan, index=df_out.index)), errors="coerce").to_numpy(dtype=np.float32)
    y_conv = pd.to_numeric(df_out.get("y_conv", pd.Series(np.nan, index=df_out.index)), errors="coerce").to_numpy(dtype=np.float32)
    detect = ((np.nan_to_num(y_pre, nan=0.0) >= 0.5) | (np.nan_to_num(y_conv, nan=0.0) >= 0.5))
    first_detect = None
    det_idx = np.where(detect)[0]
    if det_idx.size > 0:
        first_detect = float(times[int(det_idx[0])])
    first_incorrect = None
    if label == 1 and abnormal_start_time is not None:
        wrong_idx = np.where(detect & (times < float(abnormal_start_time)))[0]
        if wrong_idx.size > 0:
            first_incorrect = float(times[int(wrong_idx[0])])
    elif label == 0:
        wrong_idx = np.where(detect)[0]
        if wrong_idx.size > 0:
            first_incorrect = float(times[int(wrong_idx[0])])
    detection_delay = None
    if label == 1 and first_detect is not None and abnormal_start_time is not None:
        detection_delay = float(first_detect - float(abnormal_start_time))
    ct_is_max = False
    if collision_time is not None and np.isfinite(collision_time):
        finite_time = times[np.isfinite(times)]
        if finite_time.size > 0:
            ct_is_max = bool(abs(float(collision_time) - float(np.max(finite_time))) < 1e-6)
    # Keep TF-compatible columns/order.
    return [
        int(index_value),
        label if label in (0, 1) else None,
        abnormal_start_time,
        first_detect,
        detection_delay,
        first_incorrect,
        collision_time,
        ct_is_max,
    ]
def save_basic_info_csv(run_dir: str, info: Dict[str, Any]) -> None:
    pd.DataFrame([info]).to_csv(os.path.join(run_dir, "basic_info.csv"), index=False, encoding="utf-8")
def save_decision_times_csv(run_dir: str, rows: List[List[Any]]) -> None:
    columns = [
        "Index",
        "TrueLabel",
        "AbnormalStartTime",
        "FirstCorrectTime",
        "DetectionDelay",
        "FirstIncorrectTime",
        "CollisionTime",
        "CollisionTimeIsMax",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        os.path.join(run_dir, "decision_times.csv"), index=False, encoding="utf-8"
    )
def save_confusion_and_roc(
    run_dir: str,
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> None:
    if y_true.size == 0 or y_score.size == 0:
        print("[WARN] skipped confusion/roc: no labeled file-level samples.")
        return
    y_pred = (y_score >= float(threshold)).astype(np.int32)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int32), y_pred.astype(np.int32)):
        if t in (0, 1) and p in (0, 1):
            cm[t, p] += 1
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_csv = os.path.join(run_dir, f"{model_name}_Inference_confusion_matrix.csv")
    cm_df.to_csv(cm_csv, encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred_0", "pred_1"]); ax.set_yticklabels(["true_0", "true_1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        ax.set_title(f"{model_name} confusion matrix")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"{model_name}_Inference_confusion_matrix.png"), dpi=140)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] failed to save confusion matrix plot: {e}")
    roc_csv = os.path.join(run_dir, f"{model_name}_Inference_ROC.csv")
    try:
        from sklearn.metrics import auc, roc_curve
        if len(np.unique(y_true)) < 2:
            raise ValueError("ROC requires both labels 0 and 1.")
        fpr, tpr, roc_thr = roc_curve(y_true, y_score)
        roc_auc = float(auc(fpr, tpr))
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr, "auc": roc_auc}).to_csv(
            roc_csv, index=False, encoding="utf-8"
        )
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"{model_name} ROC")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"{model_name}_Inference_ROC.png"), dpi=140)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] failed to save ROC plot: {e}")
    except Exception as e:
        pd.DataFrame(
            [{"fpr": np.nan, "tpr": np.nan, "threshold": np.nan, "auc": np.nan, "note": str(e)}]
        ).to_csv(roc_csv, index=False, encoding="utf-8")
        print(f"[WARN] skipped ROC curve: {e}")
# =========================================================
# 5) 前処理（DataFrame単位）
# =========================================================
def preprocess_dataframe_for_inference(
    df: pd.DataFrame,
    csv_path: str,
    features: List[str],
    categorical_features: List[str],
    category_maps: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    学習時と同じ順序で推論前の前処理を行う。
    - 欠損行削除
    - カテゴリ正規化（category_maps）
    - 数値化（安全のため）
    """
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = drop_rows_with_missing_features(df, features, context=os.path.basename(csv_path))
    df = apply_categorical_mapping(df, categorical_features, category_maps, default_unknown_id=DEFAULT_UNKNOWN_ID)
    df[features] = df[features].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return df
def mae_tail_masked_all_features(
    seq_t: torch.Tensor,           # (1, T, D) 入力の正規化後ベクトル
    mask_seq_exp: np.ndarray,      # (T, D) one-hot 拡張後の欠損マスク（全ステップ分）
    model: CausalTransformerAutoencoder,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    tail_steps: int,
) -> float:
    """
    学習時の compute_batch_mae_tail と一致する MAE 計算。
    対象は末尾 tail_steps ステップの全特徴（連続 + one-hot）。
    """
    if device == "cuda" and use_amp:
        autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()
    with torch.no_grad():
        with autocast_ctx:
            # 全ステップの再構成系列を取得
            recon_seq = model(seq_t)  # (1, T, D)
    recon_seq = recon_seq.float()
    _, T, _ = recon_seq.shape
    k = max(1, int(tail_steps))
    k = min(k, T)  # 文脈長より長くならないように
    start = T - k
    recon_tail = recon_seq[:, start:, :]          # (1, k, D)
    true_tail = seq_t[:, start:, :].float()       # (1, k, D)
    # np.ndarray -> torch.Tensor に変換し、同じ tail 区間を切り出す
    miss_tail = torch.from_numpy(mask_seq_exp[start:]).unsqueeze(0).to(device).float()  # (1, k, D)
    obs_mask = 1.0 - miss_tail
    abs_err = (recon_tail - true_tail).abs()
    num_obs = obs_mask.sum().clamp(min=1.0)
    mae = (abs_err * obs_mask).sum() / num_obs
    return float(mae.detach().cpu().item())
# =========================================================
# 6) スコア計算（ファイル単位のストリーミング推論）
# =========================================================
def compute_stream_scores_for_file(
    csv_path: str,
    ctx: Dict[str, Any],
    time_range: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    単一 CSV をストリーミング処理し、ウィンドウが満たされた箇所で逐次異常スコア（MAE）を算出。
    - 欠損は直前値、なければスケーラ平均（連続）／列ごとの UNKNOWN（カテゴリ）で埋める
    - one-hot は学習時の codes/offsets 仕様に準拠
    戻り値は anomaly/error 列を付与した DataFrame（元データ行数に一致）。
    """
    # コンテキスト取り出し
    features = ctx["features"]
    CONTINUOUS_FEATURES = ctx["CONTINUOUS_FEATURES"]
    CATEGORICAL_FEATURES = ctx["CATEGORICAL_FEATURES"]
    specs = ctx["specs"]
    offsets = ctx["offsets"]
    seq_len = ctx["seq_len"]
    model: CausalTransformerAutoencoder = ctx["model"]
    scaler = ctx["scaler"]
    device = ctx["device"]
    vmin_arr = ctx["vmin_arr"]
    vmax_arr = ctx["vmax_arr"]
    sentinels_map = ctx["sentinels_map"]
    idx_cont = ctx["idx_cont"]
    cont_idx_by_name = ctx["cont_idx_by_name"]
    cat_feature_idx = ctx["cat_feature_idx"]
    unknown_idx_by_col = ctx["unknown_idx_by_col"]
    total_cat_dim = ctx["total_cat_dim"]
    category_maps = ctx["category_maps"]
    use_amp = ctx["use_amp"]
    amp_dtype = ctx["amp_dtype"]
    threshold = float(ctx["threshold"])
    y_pre_threshold = float(ctx["y_pre_threshold"])
    y_pre_ewma_window = int(ctx["y_pre_ewma_window"])
    y_pre_consecutive = int(ctx["y_pre_consecutive"])
    tail_steps = int(ctx.get("tail_steps", 1))
    if tail_steps < 1:
        tail_steps = 1
    # CSV 読み込み
    df = pd.read_csv(
        csv_path,
        na_values=["(nan)", "nan", "NaN", "NULL", "None", "", " "],
        keep_default_na=True,
        low_memory=False,
    )
    # 前処理（学習時準拠）
    df = preprocess_dataframe_for_inference(df, csv_path, features, CATEGORICAL_FEATURES, category_maps)
    if time_range is None:
        raise ValueError("inference_time_range is required but not provided")
    # 列名は前処理で小文字化済みなので、小文字で扱う
    time_col = str(time_range.get("column", "time")).strip().lower()
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not found in CSV: {csv_path}")
    # time 列を配列にしておく
    t_arr = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float32)
    start_val = time_range.get("start", None)
    end_val = time_range.get("end", None)

    irregular_policy = str(time_range.get("irregular_time_policy", "warn")).strip().lower()
    if irregular_policy not in {"ignore", "warn", "error"}:
        irregular_policy = "warn"
    irregular, max_dev = _check_time_axis_irregular(
        t_arr,
        expected_step=float(time_range.get("expected_step", 0.1)),
        tolerance=float(time_range.get("tolerance", 0.02)),
    )
    if irregular:
        msg = (
            f"{Path(csv_path).name}: irregular time intervals detected "
            f"(max deviation={max_dev:.6f})"
        )
        if irregular_policy == "error":
            raise ValueError(msg)
        if irregular_policy == "warn":
            print(f"[WARN] {msg}")
    # ストリーム用バッファ
    buf_norm: Deque[np.ndarray] = deque(maxlen=seq_len)  # 正規化＋one-hot 結合後
    buf_raw: Deque[np.ndarray] = deque(maxlen=seq_len)   # 欠損補完後の元スケール（F）
    buf_missing: Deque[np.ndarray] = deque(maxlen=seq_len)  # 欠損マスク（F）
    anomaly_scores: List[float] = []
    is_flags: List[float] = []   # NaN も入るため float にしておく
    errors: List[float] = []
    levels: List[float] = []
    # NaN 出力行をまとめて追加する小ヘルパー
    def _append_nan_row():
        anomaly_scores.append(float("nan"))
        errors.append(float("nan"))
        is_flags.append(float("nan"))
        levels.append(float("nan"))
    for i in range(len(df)):
        # 1) 行取り出し (F,)
        row = df.loc[i, features].to_numpy(dtype=np.float32)
        # 2) 欠損判定（NaN / 範囲外 / sentinel）
        is_missing_bool = np.isnan(row) | (row < vmin_arr) | (row > vmax_arr)
        for j, sset in enumerate(sentinels_map):
            if sset:
                v = row[j]
                if not np.isnan(v) and v in sset:
                    is_missing_bool[j] = True
        is_missing = is_missing_bool.astype(np.float32)
        # 3) 欠損補完（前方補完。先頭は 連続=学習平均・カテゴリ=列ごとの UNKNOWN）
        x_raw = row.copy()
        prev = buf_raw[-1] if len(buf_raw) > 0 else None
        if is_missing.any():
            for j, f in enumerate(features):
                if is_missing[j] == 1.0:
                    if prev is not None:
                        x_raw[j] = prev[j]
                    else:
                        if f in CONTINUOUS_FEATURES and hasattr(scaler, "mean_"):
                            k = cont_idx_by_name[f]
                            x_raw[j] = float(scaler.mean_[k])
                        else:
                            unknown_val = float(category_maps.get(f, {}).get("UNKNOWN", DEFAULT_UNKNOWN_ID))
                            x_raw[j] = unknown_val
        # 4) クリップ（学習時のルールに合わせる）
        x_raw = np.clip(x_raw, vmin_arr, vmax_arr)
        # 5) 連続の標準化（学習済みスケーラ）
        x_cont = x_raw[idx_cont].reshape(1, -1).astype(np.float32)
        x_cont_norm = scaler.transform(x_cont).reshape(-1).astype(np.float32)
        # 6) カテゴリ one-hot 展開（学習時の specs/codes に準拠）
        x_cat = np.zeros((total_cat_dim,), dtype=np.float32)
        for cat_col in CATEGORICAL_FEATURES:
            sp = specs[cat_col]
            v2i = sp["value_to_index"]
            val = float(x_raw[cat_feature_idx[cat_col]])
            idx = v2i.get(val, unknown_idx_by_col[cat_col])
            x_cat[int(offsets[cat_col]) + int(idx)] = 1.0
        # 7) 入力ベクトル結合（連続＋カテゴリ）
        x_exp_norm = np.concatenate([x_cont_norm, x_cat], axis=0).astype(np.float32)
        # バッファ更新
        buf_raw.append(x_raw)
        buf_norm.append(x_exp_norm)
        buf_missing.append(is_missing)
        # この行が time 範囲内かどうか判定 --------------
        t_i = t_arr[i]
        in_range = np.isfinite(t_i)
        if in_range and start_val is not None:
            in_range = in_range and (t_i >= float(start_val))
        if in_range and end_val is not None:
            in_range = in_range and (t_i <= float(end_val))
        if not in_range:
            # 範囲外の行 → 推論はせず、出力は NaN
            _append_nan_row()
            continue
        # 8) 文脈長と系列の取り出し（最大 seq_len）
        ctx_len = min(len(buf_norm), int(seq_len))     # 直近の文脈長（最大 seq_len）
        if ctx_len <= 0:
            _append_nan_row()
            continue
        # 入力系列 (ctx_len, D) と 欠損マスク (ctx_len, F) を取り出す
        seq = np.stack(list(buf_norm)[-ctx_len:])             # (ctx_len, D)
        miss_seq = np.stack(list(buf_missing)[-ctx_len:])     # (ctx_len, F)
        # 欠損マスクを one-hot 展開後次元に拡張 -> (ctx_len, D)
        mask_seq_exp = expand_mask_for_onehot_precomputed(
            miss_seq,
            idx_cont,
            CATEGORICAL_FEATURES,
            specs,
            offsets,
            features,
        )
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device) # (1, ctx_len, D)
        # 9) 予測とスコア（末尾 tail_steps ステップの MAE）
        mae_all = mae_tail_masked_all_features(
            seq_t=seq_t,
            mask_seq_exp=mask_seq_exp,
            model=model,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            tail_steps=tail_steps,
        )
        # 正規化スコア（p10〜p99 で 0〜1）
        low, high = float(ctx["p10"]), float(ctx["p99"])
        score = (mae_all - low) / max(high - low, 1e-6)
        score = float(np.clip(score, 0.0, 1.0))
        # 閾値判定
        is_anomaly = (mae_all > threshold)
        anomaly_scores.append(score)
        errors.append(mae_all)
        is_flags.append(1.0 if is_anomaly else 0.0)
        # 分布ベースの4段階レベル（0〜3）
        bounds_mae = [float(ctx["p50"]), float(ctx["p90"]), float(ctx["threshold"])]
        level = int(np.digitize(mae_all, bounds_mae))  # 0,1,2,3
        levels.append(float(level))
    score_arr = np.asarray(anomaly_scores, dtype=np.float32)
    y_pre_score = compute_ewma_scores(score_arr, window=y_pre_ewma_window)
    y_pre_flag = compute_consecutive_flags(y_pre_score, threshold=y_pre_threshold, required=y_pre_consecutive)
    df_out = df.copy()
    # Legacy columns (kept for backward compatibility)
    df_out["anomaly"] = anomaly_scores      # normalized score in [0, 1]
    df_out["error"] = errors                # raw MAE error
    df_out["is_anomaly"] = is_flags
    df_out["anomaly_level"] = levels
    # TF-aligned naming
    df_out["y_conv_score"] = anomaly_scores
    df_out["y_conv"] = is_flags
    df_out["y_pre_score"] = y_pre_score
    df_out["y_pre"] = y_pre_flag
    return df_out
# =========================================================
# 7) メイン（CLI）
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Stream anomaly scoring for test CSV using trained Transformer AE")
    parser.add_argument("--csv", default=str(DEFAULT_INPUT_DIR), help="Path to test CSV or a directory that contains CSVs")
    parser.add_argument(
        "--artifacts_dir",
        default=None,
        help="Path to trained artifacts (default: out_dir from config/hyperparams_common.json)",
    )
    parser.add_argument(
        "--meta_csv",
        default=None,
        help="Optional metadata CSV path (label/abnormal_start_time/collision_time).",
    )
    parser.add_argument(
        "--file_score_mode",
        choices=["tail", "max", "both"],
        default=None,
        help="File-level score output mode. both=save tail and max scores.",
    )
    parser.add_argument(
        "--file_eval_mode",
        choices=["tail", "max"],
        default=None,
        help="Mode used for confusion matrix/ROC evaluation.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Deprecated: directory scanning is always recursive (kept for backward compatibility)",
    )
    parser.add_argument(
        "--legacy_out", 
        default=None, 
        help="Legacy CSV output directory (optional)"
    )
    parser.add_argument(
        "--runtime_config",
        default=str(DEFAULT_INFER_RUNTIME_CONFIG_PATH),
        help="Optional runtime settings JSON path.",
    )
    args = parser.parse_args()

    runtime_cfg = load_inference_runtime_config(args.runtime_config)

    file_score_mode = str(args.file_score_mode or runtime_cfg.get("file_score_mode", "both")).strip().lower()
    if file_score_mode not in {"tail", "max", "both"}:
        file_score_mode = "both"
    file_eval_mode = str(args.file_eval_mode or runtime_cfg.get("file_eval_mode", "tail")).strip().lower()
    if file_eval_mode not in {"tail", "max"}:
        file_eval_mode = "tail"
    if file_score_mode in {"tail", "max"} and args.file_eval_mode is None and "file_eval_mode" not in runtime_cfg:
        file_eval_mode = file_score_mode

    meta_csv_path = args.meta_csv if args.meta_csv else runtime_cfg.get("meta_csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_tty()
    artifacts_default = args.artifacts_dir if args.artifacts_dir else resolve_default_artifacts_dir()
    artifacts_dir = prompt_artifacts_dir(
        default_path=str(artifacts_default),
        project_root=str(PROJECT_ROOT),
        extra_candidates=[str(artifacts_default), str(LEGACY_DEFAULT_ARTIFACTS_DIR)],
    )
    csv_target = prompt_csv_or_dir_or_glob(
        default_target=str(args.csv),
        project_root=str(PROJECT_ROOT),
        title="Select scoring input target",
        include_tagged_option=False,
    )
    print(f"[INFO] artifacts_dir={artifacts_dir}")
    print(f"[INFO] csv_target={csv_target}")
    # アーティファクト読み込み
    cfg, thr_info, scaler, paths = load_artifacts(artifacts_dir)
    # 推論用コンテキスト構築
    ctx = build_inference_context(cfg, thr_info, scaler, paths["model_path"], device)
    # 推論時 time 範囲設定の読み込み
    inference_time_range = load_inference_time_range_config(
        str(DEFAULT_TAGGED_DATASET_INFERENCE_CONFIG_PATH)
    )
    if inference_time_range is None:
        raise ValueError(
            f"inference_time_range is required but not found or disabled in: {DEFAULT_TAGGED_DATASET_INFERENCE_CONFIG_PATH}"
        )
    # 入力列挙
    input_dir_override = None
    if csv_target != TAGGED_DATASET_TOKEN:
        input_dir_override = _resolve_input_directory(csv_target)
    if csv_target == TAGGED_DATASET_TOKEN or input_dir_override is not None:
        try:
            files, report = build_tagged_dataset_csvs_from_config(
                config_path=str(DEFAULT_TAGGED_DATASET_INFERENCE_CONFIG_PATH),
                pattern="*.csv",
                override_search_roots=[input_dir_override] if input_dir_override is not None else None,
            )
        except Exception as e:
            print(f"[WARN] tagged dataset preparation failed: {repr(e)} -> inference is skipped.")
            return
        if input_dir_override is not None:
            print(f"[INFO] input directory override for tagged matching: {input_dir_override}")
        print(f"[INFO] tagged filter config: {report['config_path']}")
        if report.get("config_profile"):
            print(f"[INFO] tagged filter profile: {report['config_profile']}")
        print(f"[INFO] ledger file-name column: {report['file_name_column']}")
        print(
            f"[INFO] ledger rows: total={report['rows_total']}, "
            f"blank_file={report['rows_skipped_blank_file']}, filtered_out={report['rows_filtered_out']}"
        )
        print(f"[INFO] tagged candidates after AQ-AY filters: {report['candidates_after_filter']}")
        sampling_enabled = bool(report.get("sampling_enabled", False))
        if sampling_enabled:
            sampling_seed = 42
            if DEFAULT_HPARAMS_PATH.exists():
                try:
                    with open(DEFAULT_HPARAMS_PATH, "r", encoding="utf-8-sig") as f:
                        hp = json.load(f)
                    sampling_seed = int(hp.get("random_seed", 42))
                except Exception:
                    sampling_seed = 42
            files, sampling = sample_paths_interactively(
                files,
                seed=sampling_seed,
                title="Select scoring usage ratio (file count based)",
            )
            print(
                f"[INFO] tagged dataset candidates={sampling['total_count']}, "
                f"selected={sampling['percent']}% -> {sampling['sample_count']} files"
            )
        else:
            print(f"[INFO] sampling disabled by config -> use 100% ({len(files)} files)")
        for flt in report.get("filters_active", []):
            print(
                f"[INFO] filter active: {flt['excel_col']} ({flt['column_name']}) "
                f"mode={flt['mode']} values={flt['values']}"
            )
        for root in report.get("unreachable_roots", []):
            print(f"[WARN] network root unreachable or missing: {root}")
        if report["duplicate_network_filenames"] > 0:
            print(
                "[INFO] duplicate filenames resolved by priority (network_roots order): "
                f"{report['duplicate_network_filenames']}"
            )
        if report["missing_in_network"] > 0:
            print(f"[WARN] excluded (listed in ledger but file missing in network): {report['missing_in_network']}")
        if report["missing_in_ledger"] > 0:
            print(f"[WARN] excluded (file exists in network but missing in ledger): {report['missing_in_ledger']}")
        print(f"[INFO] tagged usable files: {report['usable_count']}")
    else:
        files = list_input_files(csv_target)
    meta_map = load_optional_meta_map(meta_csv_path, str(PROJECT_ROOT))
    # 出力ディレクトリ設定
    # 1) TF-style valid results (run_dir)
    valid_results_dir_setting = runtime_cfg.get("valid_results_dir", str(DEFAULT_VALID_RESULTS_DIR))
    valid_results_dir = os.path.abspath(str(valid_results_dir_setting))
    os.makedirs(valid_results_dir, exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(valid_results_dir, f"{ts_str}_{MODEL_NAME}_inference_csv")
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    # 2) legacy 出力（ユーザ指定 --legacy_out があればそれを優先）
    if args.legacy_out:
        # user may pass relative path -> expanduser -> if not absolute treat as project-root relative
        leg = str(Path(args.legacy_out).expanduser())
        if not os.path.isabs(leg):
            out_dir = os.path.abspath(os.path.join(str(PROJECT_ROOT), leg))
        else:
            out_dir = os.path.abspath(leg)
    else:
        legacy_default = runtime_cfg.get("legacy_output_dir", os.path.join(str(PROJECT_ROOT), "result", "OFF_pa99"))
        out_dir = os.path.abspath(str(legacy_default))
    os.makedirs(out_dir, exist_ok=True)
    # デバッグ用ログ（ここで run_dir/out_dir は確実に定義されている）
    print(f"[INFO] legacy csv output dir: {out_dir}")
    print(f"[INFO] tf-style run output dir: {run_dir}")
    basic_info = {
        "timestamp": ts_str,
        "model": MODEL_NAME,
        "csv_target": csv_target,
        "artifacts_dir": artifacts_dir,
        "meta_csv": str(meta_csv_path) if meta_csv_path else "",
        "file_score_mode": file_score_mode,
        "file_eval_mode": file_eval_mode,
        "num_files": int(len(files)),
        "y_conv_threshold": float(ctx["y_conv_threshold"]),
        "y_pre_threshold": float(ctx["y_pre_threshold"]),
        "y_pre_ewma_window": int(ctx["y_pre_ewma_window"]),
        "y_pre_consecutive": int(ctx["y_pre_consecutive"]),
    }
    save_basic_info_csv(run_dir, basic_info)
    file_records: List[Dict[str, Any]] = []
    decision_rows: List[List[Any]] = []
    logger = InlineLogger()
    # 先頭を優先して末尾を省略（例: AZSH20-1067305_001608_2022年...）
    def short_head(s: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(s) <= max_len:
            return s
        if max_len == 1:
            return "…"
        keep = max_len - 1
        return s[:keep] + "…"
    # ファイルごとに推論・保存
    for fi, csv_path in enumerate(files, 1):
        p = Path(csv_path)
        try:
            # 進行中を表示（1行上書き対象）
            head = f"[{fi}/{len(files)}] Processing: "
            file_disp = p.name  # ファイル名のみ。親ディレクトリも出す場合は f"{p.parent.name}/{p.name}"
            remain = MAX_LOG_WIDTH - len(head)
            progress_line = head + (short_head(file_disp, remain) if remain > 0 else "")
            logger.print(progress_line)
            # 推論と保存
            df_out = compute_stream_scores_for_file(
                csv_path,
                ctx,
                time_range=inference_time_range,
            )
            base = p.stem
            suffix = infer_output_suffix_from_path(str(csv_path))
            out_path = os.path.join(out_dir, f"{base}_{suffix}.csv")
            df_out.to_csv(out_path, index=False, float_format="%.6f")
            label_meta = resolve_label_and_meta(
                csv_path=str(csv_path),
                project_root=str(PROJECT_ROOT),
                meta_map=meta_map,
            )
            label = label_meta["label"]
            label_source = label_meta["label_source"]
            ast = label_meta["abnormal_start_time"]
            ct = label_meta["collision_time"]
            score_series = pd.to_numeric(df_out["y_conv_score"], errors="coerce").to_numpy(dtype=np.float32)
            y_pre_series = pd.to_numeric(df_out["y_pre"], errors="coerce").to_numpy(dtype=np.float32)
            y_conv_series = pd.to_numeric(df_out["y_conv"], errors="coerce").to_numpy(dtype=np.float32)
            file_prob_tail = pick_file_probability(score_series, mode="tail")
            file_prob_max = pick_file_probability(score_series, mode="max")

            if file_score_mode == "tail":
                file_prob_eval = file_prob_tail
            elif file_score_mode == "max":
                file_prob_eval = file_prob_max
            else:
                file_prob_eval = file_prob_tail if file_eval_mode == "tail" else file_prob_max

            file_pred = None
            if file_prob_eval is not None:
                file_pred = int(file_prob_eval >= float(ctx["y_conv_threshold"]))
            file_records.append(
                {
                    "csv_path": str(csv_path),
                    "label": label,
                    "label_source": label_source,
                    "file_prob_tail": file_prob_tail,
                    "file_prob_max": file_prob_max,
                    "file_prob_eval": file_prob_eval,
                    "file_pred_eval": file_pred,
                    "file_score_mode": file_score_mode,
                    "file_eval_mode": file_eval_mode,
                    "y_conv_threshold": float(ctx["y_conv_threshold"]),
                    "abnormal_start_time": ast,
                    "collision_time": ct,
                    "rows_total": int(len(df_out)),
                    "rows_scored": int(np.isfinite(score_series).sum()),
                    "rows_y_pre_alert": int(np.nansum(y_pre_series >= 0.5)),
                    "rows_y_conv_alert": int(np.nansum(y_conv_series >= 0.5)),
                    "legacy_output_csv": out_path,
                    "output_suffix": suffix,
                }
            )
            decision_rows.append(
                compute_decision_times_row(
                    index_value=fi - 1,
                    df_out=df_out,
                    label=label,
                    abnormal_start_time=ast,
                    collision_time=ct,
                )
            )
            # 成功時は何も表示しない（次のファイルの Processing 行で上書き）
        except Exception as e:
            # 警告は独立した行で出すため、まず現在の行を確定
            logger.newline()
            print(f"[WARN] Failed to process {p.name}: {e}")
            # 以降は次のループで再び1行進行表示に戻る
    # ループ終了後、最後の進行表示行を確定
    logger.newline()
    if file_records:
        df_file = pd.DataFrame(file_records)
        df_file.to_csv(os.path.join(run_dir, "file_scores.csv"), index=False, encoding="utf-8")
    save_decision_times_csv(run_dir, decision_rows)
    if file_records:
        valid_rows = [
            r for r in file_records
            if r.get("label") in (0, 1) and r.get("file_prob_eval") is not None and np.isfinite(float(r["file_prob_eval"]))
        ]
        if valid_rows:
            y_true = np.asarray([int(r["label"]) for r in valid_rows], dtype=np.int32)
            y_score = np.asarray([float(r["file_prob_eval"]) for r in valid_rows], dtype=np.float32)
            save_confusion_and_roc(
                run_dir=run_dir,
                model_name=MODEL_NAME,
                y_true=y_true,
                y_score=y_score,
                threshold=float(ctx["y_conv_threshold"]),
            )
        else:
            print("[WARN] no labeled file scores for confusion/roc output.")
    print(f"[DONE] legacy csv output: {out_dir}")
    print(f"[DONE] tf-style run output: {run_dir}")
if __name__ == "__main__":
    main()