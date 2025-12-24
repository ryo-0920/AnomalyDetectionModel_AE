import argparse
import json
import os
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
from collections import deque
from contextlib import nullcontext
import glob
import random
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
# 進捗バー
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
from torch import amp
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import DataLoader, Dataset, Subset
from models.transformer_autoencoder import CausalTransformerAutoencoder
import bisect
import platform
# =========================================================
# 定数・特徴メタ情報
# =========================================================
NA_VALUES = ["(nan)", "nan", "NaN", "NULL", "None", "", " "]
amp_dtype = torch.bfloat16
FEATURES = [
    "abscontrol","accelerationfb","accelerationlr","accelpedalangle","angularvelocity",
    "atshiftposition","brake","brakepedal","brakepressure","parkingbrake","speed",
    "steeringangle","trccontrol","turnlampswitchstatus","vsccontrol",
    "wheelspeedfl","wheelspeedfr","wheelspeedrl","wheelspeedrr"
]
FEATURE_RULES = {
    "accelerationfb":  {"sentinels": [99.0], "vmin": -20.0, "vmax": 20.0, "interp": "linear", "clip": True},
    "accelerationlr":  {"sentinels": [99.0], "vmin": -20.0, "vmax": 20.0, "interp": "linear", "clip": True},
    "accelpedalangle": {"sentinels": [999.0], "vmin": 0.0, "vmax": 100.0, "interp": "linear", "clip": True},
    "angularvelocity": {"sentinels": [999.0], "vmin": -125.0, "vmax": 125.0, "interp": "linear", "clip": True},
    "brake":           {"sentinels": [99.0], "vmin": 0.0, "vmax": 1.0, "interp": "linear", "clip": True},
    "brakepedal":      {"sentinels": [99.0], "vmin": 0.0, "vmax": 1.0, "interp": "linear", "clip": True},
    "brakepressure":   {"sentinels": [99.0], "vmin": 0.0, "vmax": 20.46, "interp": "linear", "clip": True},
    "speed":           {"sentinels": [999.0], "vmin": -327.68, "vmax": 655.351, "interp": "linear", "clip": True},
    "steeringangle":   {"sentinels": [9999.0], "vmin": -1044, "vmax": 1044, "interp": "linear", "clip": True},
    "wheelspeedfl":    {"sentinels": [999.0], "vmin": -327.68, "vmax": 327.67, "interp": "linear", "clip": True},
    "wheelspeedfr":    {"sentinels": [999.0], "vmin": -327.68, "vmax": 327.67, "interp": "linear", "clip": True},
    "wheelspeedrl":    {"sentinels": [999.0], "vmin": -327.68, "vmax": 327.67, "interp": "linear", "clip": True},
    "wheelspeedrr":    {"sentinels": [999.0], "vmin": -327.68, "vmax": 327.67, "interp": "linear", "clip": True},
}
DEFAULT_RULE = {"sentinels": [], "vmin": None, "vmax": None, "interp": "linear", "clip": False}
CATEGORICAL_FEATURES = [
    "abscontrol", "atshiftposition", "brakepedal", "parkingbrake", "trccontrol",
    "turnlampswitchstatus","vsccontrol",
]
DEFAULT_UNKNOWN_ID = -1.0
CATEGORY_MAPS = {
    "abscontrol": {"OFF": 0.0, "ON": 1.0, "UNKNOWN": DEFAULT_UNKNOWN_ID},
    "atshiftposition": {"DEFAULT": 0.0, "P": 10.0, "R": 20.0, "N": 30.0, "D": 40.0, "B": 50.0, "NG":99.0, "UNKNOWN": DEFAULT_UNKNOWN_ID},
    "brakepedal": {"OFF": 0.0, "ON": 1.0, "NG":99.0, "UNKNOWN": DEFAULT_UNKNOWN_ID},
    "parkingbrake": {"OFF": 0.0, "ON": 1.0, "UNKNOWN": DEFAULT_UNKNOWN_ID},
    "trccontrol": {"OFF": 0.0, "ON": 1.0, "UNKNOWN": DEFAULT_UNKNOWN_ID},
    "turnlampswitchstatus": {"NG":0, "LEFT": 1.0, "RIGHT": 2.0, "OFF": 3.0, "UNKNOWN": DEFAULT_UNKNOWN_ID},
    "vsccontrol": {"OFF": 0.0, "ON": 1.0, "UNKNOWN": DEFAULT_UNKNOWN_ID},
}
FEATURE_RULES.update({
    "abscontrol": {"interp": None, "clip": False},
    "atshiftposition": {"interp": None, "clip": False},
    "brakepedal": {"interp": None, "clip": False},
    "parkingbrake": {"interp": None, "clip": False},
    "trccontrol": {"interp": None, "clip": False},
    "turnlampswitchstatus": {"interp": None, "clip": False},
    "vsccontrol": {"interp": None, "clip": False},
})
# =========================================================
# システムユーティリティ（デバイス・AMP・シード）
# =========================================================
def resolve_device(req: str) -> str:
    req = (req or "auto").lower()
    if req == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
def resolve_amp_dtype(name: str) -> torch.dtype:
    return torch.float16 if name.lower() == "fp16" else torch.bfloat16
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
        torch.backends.cudnn.benchmark = True
def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)
def _compile_model_if_possible(model: torch.nn.Module, use_compile: bool, device: str) -> torch.nn.Module:
    # コンパイルを明示的に制御
    if not use_compile:
        return model
    # CUDA環境でのみ試みる（CPUはcompileしてもメリットが薄い）
    if device != "cuda":
        return model
    # Tritonがある場合のみ Inductor を試す。なければ aot_eager にフォールバック
    try:
        import triton  # Windows では通常ここで ImportError
        return torch.compile(model, mode="max-autotune")
    except Exception:
        # Inductor が使えない環境では aot_eager に切り替え（Triton不要）
        try:
            return torch.compile(model, backend="aot_eager")
        except Exception:
            return model
# =========================================================
# 前処理ユーティリティ（カテゴリ化・one-hot・マスク・欠損行削除）
# =========================================================
def read_csv_lower(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def require_columns(df: pd.DataFrame, path: str) -> None:
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: CSV missing required columns: {missing}")
def apply_categorical_mapping(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        col_series = df[col]
        if np.issubdtype(col_series.dtype, np.number):
            df[col] = pd.to_numeric(col_series, errors="coerce").fillna(DEFAULT_UNKNOWN_ID).astype(np.float32)
        else:
            s = col_series.astype(str).str.strip().str.upper()
            if col in CATEGORY_MAPS:
                mapped = s.map(CATEGORY_MAPS[col])
                unknown_id = CATEGORY_MAPS[col].get("UNKNOWN", DEFAULT_UNKNOWN_ID)
                df[col] = mapped.where(~mapped.isna(), unknown_id).astype(np.float32)
            else:
                df[col] = pd.to_numeric(s, errors="coerce").fillna(DEFAULT_UNKNOWN_ID).astype(np.float32)
    return df
def drop_rows_with_missing_features(df: pd.DataFrame, context: str = "", how: str = "all") -> pd.DataFrame:
    before = len(df)
    df2 = df.dropna(subset=FEATURES, how=how).reset_index(drop=True)
    dropped = before - len(df2)
    if dropped > 0:
        msg_ctx = f" [{context}]" if context else ""
        print(f"[INFO]{msg_ctx} 欠損行を {dropped} 件削除 (残り {len(df2)} 行, how='{how}')")
    if len(df2) == 0:
        raise ValueError(f"{context}: 欠損行の削除によりデータが空になりました (how='{how}')")
    return df2
def clean_and_mask_by_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    N, F = len(df), len(FEATURES)
    mask = np.zeros((N, F), dtype=np.float32)
    for j, col in enumerate(FEATURES):
        rule = {**DEFAULT_RULE, **FEATURE_RULES.get(col, {})}
        s = pd.to_numeric(df[col], errors="coerce")
        invalid = pd.Series(False, index=s.index)
        sents = rule.get("sentinels", [])
        if isinstance(sents, (list, tuple)) and len(sents) > 0:
            invalid |= s.isin(sents)
        elif sents not in (None, [], ()):
            invalid |= s.eq(sents)
        if rule.get("vmin") is not None:
            invalid |= s.lt(rule["vmin"])
        if rule.get("vmax") is not None:
            invalid |= s.gt(rule["vmax"])
        missing_mask = invalid | s.isna()
        mask[:, j] = missing_mask.astype(np.float32).to_numpy()
        s = s.mask(missing_mask, np.nan)
        interp = rule.get("interp", "linear")
        if interp == "linear":
            s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
        elif interp == "ffill":
            s = s.ffill().bfill()
        if rule.get("clip", False) and (rule.get("vmin") is not None or rule.get("vmax") is not None):
            s = s.clip(lower=rule.get("vmin", -np.inf), upper=rule.get("vmax", np.inf))
        df[col] = s.astype(np.float32)
    return df, mask
def build_onehot_specs(category_maps: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    specs = {}
    for col in CATEGORICAL_FEATURES:
        codes = sorted(list(set(category_maps.get(col, {"UNKNOWN": DEFAULT_UNKNOWN_ID}).values())))
        specs[col] = {"codes": codes, "value_to_index": {float(v): i for i, v in enumerate(codes)}, "dim": len(codes)}
    return specs
def onehot_encode_df(df: pd.DataFrame, specs: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, int]]:
    N = len(df)
    dims = [specs[col]["dim"] for col in CATEGORICAL_FEATURES]
    total_dim = int(np.sum(dims)) if dims else 0
    if total_dim == 0:
        return np.zeros((N, 0), dtype=np.float32), {}
    X_cat = np.zeros((N, total_dim), dtype=np.float32)
    offsets, offset = {}, 0
    rows = np.arange(N)
    for col in CATEGORICAL_FEATURES:
        sp = specs[col]; v2i = sp["value_to_index"]; dim = sp["dim"]
        offsets[col] = offset
        vals = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULT_UNKNOWN_ID).astype(np.float32).to_numpy()
        unknown_idx = v2i.get(float(DEFAULT_UNKNOWN_ID), 0)
        idxs = np.fromiter((v2i.get(float(v), unknown_idx) for v in vals), dtype=np.int32, count=N)
        X_cat[rows, idxs + offset] = 1.0
        offset += dim
    return X_cat.astype(np.float32), offsets
def expand_mask_for_onehot(mask: np.ndarray, idx_cont: List[int], specs: Dict[str, Dict[str, Any]],
                           offsets: Dict[str, int]) -> np.ndarray:
    N = mask.shape[0]
    mask_cont = mask[:, idx_cont].astype(np.float32)
    total_cat_dim = sum(specs[col]["dim"] for col in CATEGORICAL_FEATURES)
    mask_cat = np.zeros((N, total_cat_dim), dtype=np.float32)
    feat_idx = {name: i for i, name in enumerate(FEATURES)}
    for col in CATEGORICAL_FEATURES:
        src_mask_col = mask[:, feat_idx[col]:feat_idx[col] + 1]
        dim = specs[col]["dim"]; off = offsets[col]
        mask_cat[:, off:off + dim] = np.repeat(src_mask_col, dim, axis=1)
    return np.concatenate([mask_cont, mask_cat], axis=1).astype(np.float32)
# =========================================================
# データセットクラス・ファイル列挙
# =========================================================
class SequenceDatasetMasked(Dataset):
    def __init__(self, data: np.ndarray, mask: np.ndarray, seq_len: int):
        assert data.shape == mask.shape
        self.data, self.mask, self.seq_len = data, mask, seq_len
        self.N = data.shape[0]
        if self.N < seq_len:
            raise ValueError(f"Not enough rows ({self.N}) for seq_len={seq_len}")
    def __len__(self) -> int:
        return self.N - self.seq_len + 1
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.data[idx:idx + self.seq_len]).float(), torch.from_numpy(self.mask[idx:idx + self.seq_len]).float()
class SequenceDatasetMaskedSegments(Dataset):
    """
    改良版: 全ウィンドウの開始位置を列挙せず、各セグメントのウィンドウ数の累積和で
    グローバル idx -> (seg, t) を計算する。
    """
    def __init__(self, xs: Sequence[np.ndarray], ms: Sequence[np.ndarray], seq_len: int,
                 names: Optional[Sequence[str]] = None, return_index: bool = False):
        assert len(xs) == len(ms), "X と mask のリスト長は一致している必要があります"
        self.xs, self.ms, self.seq_len = xs, ms, seq_len
        self.return_index = return_index
        self.seg_names = list(names) if names is not None else [f"seg_{i}" for i in range(len(xs))]
        self.seg_win_counts: List[int] = []
        total = 0
        self.cum: List[int] = [0]  # cum[k] = 先頭からセグメントk-1までのウィンドウ合計
        for s, (X, M) in enumerate(zip(xs, ms)):
            assert X.shape == M.shape, f"seg {s}: X と mask の形状が一致していません: {X.shape} != {M.shape}"
            n_w = max(0, X.shape[0] - seq_len + 1)
            self.seg_win_counts.append(n_w)
            total += n_w
            self.cum.append(total)
        if total == 0:
            raise ValueError(f"全セグメントで seq_len={seq_len} のウィンドウが構築できません")
        self.N_windows_total = total
    def __len__(self) -> int:
        return self.N_windows_total
    def _locate(self, idx: int) -> Tuple[int, int]:
        r = bisect.bisect_right(self.cum, idx)
        s = r - 1
        t = idx - self.cum[s]
        return s, t
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.N_windows_total:
            raise IndexError(idx)
        s, t = self._locate(idx)
        x = self.xs[s][t: t + self.seq_len]
        m = self.ms[s][t: t + self.seq_len]
        if self.return_index:
            return torch.from_numpy(x).float(), torch.from_numpy(m).float(), torch.tensor(s, dtype=torch.long)
        return torch.from_numpy(x).float(), torch.from_numpy(m).float()
def list_csvs_in_dir(csv_dir: str, pattern: str = "*.csv") -> List[str]:
    return sorted(glob.glob(os.path.join(csv_dir, pattern)))
# =========================================================
# データ準備（単一/複数 CSV）
# =========================================================
def _build_streaming_scaler_from_summaries(sum_vec: np.ndarray, sum_sq_vec: np.ndarray, cnt_vec: np.ndarray) -> StandardScaler:
    cnt_safe = np.where(cnt_vec > 0, cnt_vec, 1.0)
    mean = sum_vec / cnt_safe
    var = (sum_sq_vec / cnt_safe) - mean**2
    mean = np.where(cnt_vec > 0, mean, 0.0); var = np.where(cnt_vec > 0, var, 1.0)
    var = np.clip(var, 1e-12, None)
    scaler = StandardScaler(); scaler.mean_ = mean.astype(np.float64); scaler.var_ = var.astype(np.float64)
    scaler.scale_ = np.sqrt(scaler.var_); scaler.n_features_in_ = mean.shape[0]; scaler.n_samples_seen_ = int(np.sum(cnt_vec))
    return scaler
def preprocess_df_for_training(df: pd.DataFrame, context: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = drop_rows_with_missing_features(df, context=context, how="all")
    df = apply_categorical_mapping(df)
    df, mask = clean_and_mask_by_rules(df)
    return df, mask
def encode_with_scaler_and_onehot(df: pd.DataFrame, mask: np.ndarray,
                                  scaler: StandardScaler,
                                  specs: Dict[str, Dict[str, Any]],
                                  offsets: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    CONTINUOUS_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
    idx_cont = [FEATURES.index(f) for f in CONTINUOUS_FEATURES]
    X_cont_raw = df[CONTINUOUS_FEATURES].values.astype(np.float32)
    nan_mask = np.isnan(X_cont_raw)
    X_cont_filled = np.where(nan_mask, scaler.mean_.astype(np.float32), X_cont_raw) if nan_mask.any() else X_cont_raw
    X_cont = scaler.transform(X_cont_filled).astype(np.float32)
    X_cat, _ = onehot_encode_df(df, specs)
    X = np.concatenate([X_cont, X_cat], axis=1).astype(np.float32)
    mask_expanded = expand_mask_for_onehot(mask, idx_cont, specs, offsets).astype(np.float32)
    return X, mask_expanded
def load_and_prepare(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, np.ndarray, Dict[str, Any]]:
    df = read_csv_lower(csv_path); require_columns(df, csv_path)
    df, mask = preprocess_df_for_training(df, os.path.basename(csv_path))
    CONTINUOUS_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
    scaler = StandardScaler().fit(df[CONTINUOUS_FEATURES].values.astype(np.float32))
    if hasattr(scaler, "scale_"):
        scaler.scale_ = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    specs = build_onehot_specs(CATEGORY_MAPS)
    X_cat, offsets = onehot_encode_df(df, specs)
    X_cont = scaler.transform(df[CONTINUOUS_FEATURES].values.astype(np.float32)).astype(np.float32)
    X = np.concatenate([X_cont, X_cat], axis=1).astype(np.float32)
    idx_cont = [FEATURES.index(f) for f in CONTINUOUS_FEATURES]
    mask_expanded = expand_mask_for_onehot(mask, idx_cont, specs, offsets)
    if np.isnan(X).any():
        raise ValueError(f"X に NaN が残っています（{int(np.isnan(X).sum())} 個）")
    layout = {
        "continuous_features": CONTINUOUS_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "onehot_specs": {col: {"codes": specs[col]["codes"], "dim": specs[col]["dim"]} for col in CATEGORICAL_FEATURES},
        "offsets": offsets,
        "input_dim": int(X.shape[1]),
    }
    return df, X, scaler, mask_expanded, layout
def load_and_prepare_many(csv_paths: List[str]) -> Tuple[List[np.ndarray], StandardScaler, List[np.ndarray], Dict[str, Any], List[str]]:
    if not csv_paths:
        raise ValueError("csv_paths が空です")
    CONTINUOUS_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]
    specs = build_onehot_specs(CATEGORY_MAPS)
    dfs, masks_raw, used_paths = [], [], []
    n_cont = len(CONTINUOUS_FEATURES)
    sum_vec = np.zeros(n_cont, dtype=np.float64); sum_sq_vec = np.zeros(n_cont, dtype=np.float64); cnt_vec = np.zeros(n_cont, dtype=np.float64)
    for p in csv_paths:
        try:
            df = read_csv_lower(p); require_columns(df, p)
            df, mask = preprocess_df_for_training(df, os.path.basename(p))
            cont_vals = df[CONTINUOUS_FEATURES].values.astype(np.float32)
            sum_vec += np.nansum(cont_vals, axis=0, dtype=np.float64)
            sum_sq_vec += np.nansum(cont_vals.astype(np.float64) ** 2, axis=0)
            cnt_vec += np.sum(~np.isnan(cont_vals), axis=0, dtype=np.float64)
            dfs.append(df); masks_raw.append(mask.astype(np.float32)); used_paths.append(p)
        except Exception as e:
            print(f"[WARN] {os.path.basename(p)} の前処理でエラー: {repr(e)} -> スキップします。")
    if not dfs:
        raise ValueError("有効なCSVが1つもありません。")
    scaler = _build_streaming_scaler_from_summaries(sum_vec, sum_sq_vec, cnt_vec)
    tmp_df = pd.DataFrame({col: dfs[0][col] for col in CATEGORICAL_FEATURES})
    _, offsets = onehot_encode_df(tmp_df, specs)
    X_list, mask_list = [], []
    for df, mask in zip(dfs, masks_raw):
        X, mask_expanded = encode_with_scaler_and_onehot(df, mask, scaler, specs, offsets)
        if np.isnan(X).any():
            print(f"[WARN] 前処理後の X に NaN が残っています（{int(np.isnan(X).sum())} 個）。スキップします。")
            continue
        X_list.append(X); mask_list.append(mask_expanded)
    if not X_list:
        raise ValueError("全セグメントが不正（空またはNaN残り）だったため、学習データがありません。")
    in_dim = X_list[0].shape[1]
    for Xi, Mi in zip(X_list, mask_list):
        assert Xi.shape[1] == in_dim
        assert Xi.shape == Mi.shape
    layout = {
        "continuous_features": CONTINUOUS_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "onehot_specs": {col: {"codes": specs[col]["codes"], "dim": specs[col]["dim"]} for col in CATEGORICAL_FEATURES},
        "offsets": offsets,
        "input_dim": int(in_dim),
    }
    return X_list, scaler, mask_list, layout, used_paths
def prepare_training_data(csv: str, csvdir: Optional[str], pattern: str, seq_len: int
) -> Tuple[Optional[Dataset], int, Optional[StandardScaler], Optional[Dict[str, Any]], List[str]]:
    use_multi = bool(csvdir) or os.path.isdir(csv)
    if use_multi:
        csv_paths = list_csvs_in_dir(csvdir or csv, pattern)
        if not csv_paths:
            print(f"[WARN] {(csvdir or csv)} に {pattern} が見つかりません -> 学習をスキップします。")
            return None, 0, None, None, []
        try:
            X_list, scaler, mask_list, layout, used_paths = load_and_prepare_many(csv_paths)
            input_dim = int(X_list[0].shape[1])
            dataset = SequenceDatasetMaskedSegments(X_list, mask_list, seq_len, names=used_paths, return_index=True)
            return dataset, input_dim, scaler, layout, used_paths
        except Exception as e:
            print(f"[WARN] データセット構築に失敗: {repr(e)} -> 学習をスキップします。")
            return None, 0, None, None, []
    else:
        try:
            _, X, scaler, mask, layout = load_and_prepare(csv)
            input_dim = int(X.shape[1])
            dataset = SequenceDatasetMasked(X, mask, seq_len)
            return dataset, input_dim, scaler, layout, [csv]
        except Exception as e:
            print(f"[WARN] {repr(e)} -> 学習をスキップします。")
            return None, 0, None, None, []
# =========================================================
# データ分割（時間/セグメント）
# =========================================================
def split_dataset_timewise(dataset: Dataset, train_ratio: float = 0.8) -> Tuple[Subset, Optional[Subset]]:
    n = len(dataset); n_train = max(1, min(n - 1, int(n * train_ratio)))
    train_idx = list(range(n_train)); val_idx = list(range(n_train, n))
    return Subset(dataset, train_idx), (Subset(dataset, val_idx) if val_idx else None)
def split_segments(dataset: SequenceDatasetMaskedSegments, train_ratio: float = 0.8,
                   gap: int = 0, seed: Optional[int] = None) -> Tuple[Subset, Optional[Subset]]:
    S = len(dataset.seg_win_counts)
    seg_indices = list(range(S))
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(seg_indices)
    n_train_seg = max(1, int(round(train_ratio * S)))
    train_segs = set(seg_indices[:n_train_seg])
    val_segs = set(seg_indices[n_train_seg:])
    train_idx: List[int] = []; val_idx: List[int] = []
    for s, n_w in enumerate(dataset.seg_win_counts):
        start = dataset.cum[s]
        a = start + (gap if gap > 0 else 0)
        b = start + n_w - (gap if gap > 0 else 0)
        if b <= a:
            continue
        idxs = range(a, b)
        if s in train_segs:
            train_idx.extend(idxs)
        else:
            val_idx.extend(idxs)
    if train_idx and val_idx:
        assert set(train_idx).isdisjoint(set(val_idx)), "train/val インデックスが重複しています"
    train_subset = Subset(dataset, train_idx)
    val_subset = (Subset(dataset, val_idx) if len(val_idx) > 0 else None)
    print(f"[SPLIT] segments: train={len(train_segs)} / val={len(val_segs)} | windows: train={len(train_idx)} / val={len(val_idx)} | gap={gap}")
    return train_subset, val_subset
def split_timewise_with_gap(dataset: SequenceDatasetMasked, train_ratio: float = 0.8,
                            gap: int = 0) -> Tuple[Subset, Optional[Subset]]:
    n = len(dataset)
    n_train = max(1, min(n - 1, int(round(n * train_ratio))))
    train_end = max(0, n_train - gap)
    val_start = min(n, n_train + gap)
    train_idx = list(range(0, train_end))
    val_idx = list(range(val_start, n)) if val_start < n else []
    if train_idx and val_idx:
        assert set(train_idx).isdisjoint(set(val_idx)), "train/val インデックスが重複しています"
    print(f"[SPLIT] windows: train={len(train_idx)} / val={len(val_idx)} | gap={gap}")
    return Subset(dataset, train_idx), (Subset(dataset, val_idx) if len(val_idx) > 0 else None)
# =========================================================
# ローダ・最適化器・AMP・モデル構築
# =========================================================
def build_loader(dataset: Dataset, device: str, batch_size: int, shuffle: bool, drop_last: bool, num_workers_limit: int = 8) -> DataLoader:
    pin_memory = (device == "cuda")
    is_windows = (platform.system() == "Windows")
    if is_windows:
        num_workers = 0
    else:
        n_cpus = os.cpu_count() or 0
        num_workers = min(max((3 * n_cpus) // 4, 0), num_workers_limit)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs)
def _build_optimizer(model: torch.nn.Module, lr: float, device: str, weight_decay: float = 1e-2):
    try:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=(device == "cuda"))
    except TypeError:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
def amp_ctx(device: str, use_amp: bool):
    return amp.autocast('cuda', enabled=(use_amp and device == "cuda"), dtype=amp_dtype) if device == "cuda" else nullcontext()
def compute_masked_mse_loss(recon: torch.Tensor, target: torch.Tensor, miss_mask: torch.Tensor) -> torch.Tensor:
    obs_mask = 1.0 - miss_mask
    mse = F.mse_loss(recon, target, reduction="none")
    return (mse * obs_mask).sum() / obs_mask.sum().clamp(min=1.0)
def build_model_and_optim(input_dim: int, seq_len: int, d_model: int, nhead: int, num_layers: int, dim_ff: int,
                          dropout: float, lr: float, weight_decay: float, device: str, epochs: int,
                          use_compile: bool = False) -> Tuple[CausalTransformerAutoencoder, torch.optim.Optimizer]:
    model = CausalTransformerAutoencoder(
        input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_ff, dropout=dropout, max_len=seq_len,
    ).to(device)
    model = _compile_model_if_possible(model, use_compile=use_compile, device=device)
    opt = _build_optimizer(model, lr=lr, device=device, weight_decay=weight_decay)
    return model, opt
def train_one_batch(model: torch.nn.Module, batch: Tuple[torch.Tensor, torch.Tensor],
                    opt: torch.optim.Optimizer, scaler: amp.GradScaler, device: str, use_amp: bool
) -> Tuple[float, int]:
    batch_x, batch_m = (batch[0], batch[1]) if not (isinstance(batch, (list, tuple)) and len(batch) == 3) else (batch[0], batch[1])
    batch_x = batch_x.to(device, non_blocking=True); batch_m = batch_m.to(device, non_blocking=True)
    opt.zero_grad(set_to_none=True)
    with amp_ctx(device, use_amp):
        recon = model(batch_x); loss = compute_masked_mse_loss(recon, batch_x, batch_m)
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(opt); scaler.update()
    return float(loss.item()), int(batch_x.size(0))
def train_one_epoch(model: torch.nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, scaler: amp.GradScaler,
                    device: str, use_amp: bool, show_progress: bool = True, epoch: int = 1, epochs: int = 1,
                    sched: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> float:
    model.train(); total = 0.0; count = 0
    use_bar = bool(show_progress and (tqdm is not None))
    iterator = loader; pbar = None
    if use_bar:
        try:
            pbar = tqdm(loader, total=len(loader), leave=False, desc=f"Epoch {epoch}/{epochs}")
        except Exception:
            use_bar = False; iterator = loader
    for batch in (pbar if use_bar else iterator):
        l, b = train_one_batch(model, batch, opt, scaler, device, use_amp)
        total += l * b; count += b
        if sched is not None:
            sched.step()  # バッチごとに LR 更新
        if use_bar:
            # 現在LRを覗きたい場合（任意）
            try:
                cur_lr = sched.get_last_lr()[0] if sched is not None else opt.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{l:.4f}", avg=f"{(total/max(count,1)):.4f}", lr=f"{cur_lr:.2e}")
            except Exception:
                pbar.set_postfix(loss=f"{l:.4f}", avg=f"{(total/max(count,1)):.4f}")
    if use_bar and pbar is not None:
        pbar.close()
    return total / max(count, 1)
@torch.no_grad()
def eval_one_epoch(model: torch.nn.Module, loader: DataLoader, device: str, use_amp: bool,
                   show_progress: bool = True, epoch: int = 1, epochs: int = 1) -> float:
    model.eval(); total = 0.0; count = 0
    use_bar = bool(show_progress and (tqdm is not None))
    iterator = loader; pbar = None
    if use_bar:
        try:
            pbar = tqdm(loader, total=len(loader), leave=False, desc=f"Eval {epoch}/{epochs}")
        except Exception:
            use_bar = False; iterator = loader
    for batch in (pbar if use_bar else iterator):
        batch_x, batch_m = (batch[0], batch[1]) if not (isinstance(batch, (list, tuple)) and len(batch) == 3) else (batch[0], batch[1])
        batch_x = batch_x.to(device, non_blocking=True); batch_m = batch_m.to(device, non_blocking=True)
        with amp_ctx(device, use_amp):
            recon = model(batch_x); loss = compute_masked_mse_loss(recon, batch_x, batch_m)
        total += float(loss.item()) * int(batch_x.size(0)); count += int(batch_x.size(0))
        if use_bar:
            pbar.set_postfix(val_loss=f"{float(loss.item()):.4f}", avg=f"{(total/max(count,1)):.4f}")
    if use_bar and pbar is not None:
        pbar.close()
    return total / max(count, 1)
def train_on_dataset(dataset: Dataset, input_dim: int, seq_len: int, batch_size: int, epochs: int, lr: float,
                     d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float, device: str, use_amp: bool,
                     weight_decay: float , train_ratio: float, loss_plot_path: Optional[str] = None,
                     compile_model: bool = False
) -> Tuple[CausalTransformerAutoencoder, Dict[str, float]]:
    drop_last_train = False
    drop_last_val = False
    if isinstance(dataset, SequenceDatasetMaskedSegments):
        train_ds, val_ds = split_segments(dataset, train_ratio=train_ratio, gap=seq_len, seed=42)
    elif isinstance(dataset, SequenceDatasetMasked):
        train_ds, val_ds = split_timewise_with_gap(dataset, train_ratio=train_ratio, gap=seq_len)
    else:
        train_ds, val_ds = split_dataset_timewise(dataset, train_ratio=train_ratio)
    train_loader = build_loader(train_ds, device=device, batch_size=batch_size, shuffle=False, drop_last=drop_last_train)
    val_loader   = build_loader(val_ds, device=device, batch_size=batch_size, shuffle=False, drop_last=drop_last_val) if val_ds is not None else None
    model, opt = build_model_and_optim(input_dim, seq_len, d_model, nhead, num_layers, dim_ff, dropout, lr, weight_decay, device, epochs, use_compile=compile_model)
    # OneCycleLR はバッチごとに step する前提
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    print(f"[DBG] train_windows={len(train_ds)}, batch_size={batch_size}, drop_last={drop_last_train} => train_batches={len(train_loader)}")
    if val_loader is not None:
        print(f"[DBG] val_windows={len(val_ds)}, batch_size={batch_size}, drop_last={drop_last_val} => val_batches={len(val_loader)}")
    use_fp16 = (use_amp and device == "cuda" and amp_dtype == torch.float16)
    scaler_amp = amp.GradScaler('cuda', enabled=use_fp16)
    train_epoch_losses: List[float] = []
    val_epoch_losses: List[float] = []
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, epochs + 1):
        avg_train = train_one_epoch(model, train_loader, opt, scaler_amp, device, use_amp, show_progress=True, epoch=epoch, epochs=epochs, sched=sched)
        train_epoch_losses.append(float(avg_train))
        if val_loader is not None:
            avg_val = eval_one_epoch(model, val_loader, device, use_amp, show_progress=True, epoch=epoch, epochs=epochs)
            val_epoch_losses.append(float(avg_val))
            print(f"epoch {epoch:3d}/{epochs} - train {avg_train:.6f} | val {avg_val:.6f}")
            if avg_val < best_loss:
                best_loss, best_epoch = float(avg_val), epoch
        else:
            print(f"epoch {epoch:3d}/{epochs} - masked loss {avg_train:.6f}")
            if avg_train < best_loss:
                best_loss, best_epoch = float(avg_train), epoch
    if loss_plot_path is not None:
        save_loss_curve(train_epoch_losses, val_epoch_losses if val_loader is not None else None, loss_plot_path)
        print(f"[PLOT] Loss curve saved: {loss_plot_path}")
    metrics = {
        "train_masked_mse": float(train_epoch_losses[-1]),
        "epoch_losses": train_epoch_losses,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
    }
    if val_loader is not None and len(val_epoch_losses) > 0:
        metrics["val_epoch_losses"] = val_epoch_losses
        metrics["best_val_loss"] = min(val_epoch_losses)
    return model, metrics
# =========================================================
# 評価（MAE 分布・閾値算出）
# =========================================================
@torch.no_grad()
def compute_batch_mae_last(model, batch_x, batch_m, device, use_amp, cont_dim: int):
    with amp_ctx(device, use_amp):
        recon_last = model.reconstruct_last(batch_x)
    last_true = batch_x[:, -1, :cont_dim]
    last_obs_mask = 1.0 - batch_m[:, -1, :cont_dim]
    mae = ((recon_last[:, :cont_dim] - last_true).abs() * last_obs_mask).sum(dim=1) / last_obs_mask.sum(dim=1).clamp(min=1.0)
    return mae.detach().cpu().numpy()
@torch.no_grad()
def collect_mae_distribution(model, loader, device, use_amp, cont_dim: int):
    model.eval(); errs = []
    for batch in loader:
        batch_x, batch_m = (batch[0], batch[1]) if not (isinstance(batch,(list,tuple)) and len(batch)==3) else (batch[0], batch[1])
        batch_x = batch_x.to(device, non_blocking=True); batch_m = batch_m.to(device, non_blocking=True)
        errs.extend(compute_batch_mae_last(model, batch_x, batch_m, device, use_amp, cont_dim).tolist())
    return np.asarray(errs)
@torch.no_grad()
def compute_threshold_on_dataset(model, dataset, device, percentile=99.5, use_amp=False, cont_dim: int = None):
    if cont_dim is None:
        raise ValueError("cont_dim を指定してください（連続量の次元数）")
    dl = build_loader(dataset, device=device, batch_size=128, shuffle=False, drop_last=False)
    errs_arr = collect_mae_distribution(model, dl, device, use_amp, cont_dim)
    if errs_arr.size == 0:
        raise ValueError("MAE 分布が空です（学習データが不正の可能性）")
    mean = float(np.mean(errs_arr))
    std  = float(np.std(errs_arr, ddof=1))
    p10 = float(np.percentile(errs_arr, 10))
    p50 = float(np.percentile(errs_arr, 50))
    p90 = float(np.percentile(errs_arr, 90))
    p99 = float(np.percentile(errs_arr, 99))
    thr = float(np.percentile(errs_arr, percentile))
    temperature = float(max((p90 - p50) / 6.0, 1e-6))
    return {
        "threshold": thr,
        "mean": mean,
        "std": std,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "percentile": float(percentile),
        "temperature": temperature,
        "n_samples": int(errs_arr.size),
    }
# =========================================================
# 成果物保存・ストリーミング推論・設定構築
# =========================================================
def save_artifacts(out_dir: str, model: CausalTransformerAutoencoder, scaler: StandardScaler, config: Dict, threshold_stats: Dict):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump(threshold_stats, f, ensure_ascii=False, indent=2)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
@torch.no_grad()
def stream_score_csv(csv_path: str, artifacts_dir: str, seq_len: int = 50) -> List[Dict]:
    """
    推論時の MAE は「連続特徴のみ」で計算する（学習時の閾値定義と一致させる）。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(artifacts_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(os.path.join(artifacts_dir, "threshold.json"), "r", encoding="utf-8") as f:
        thr_info = json.load(f)
    scaler: StandardScaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
    layout = cfg["layout"]
    CONTINUOUS_FEATURES = layout["continuous_features"]; CATEGORICAL_FEATURES = layout["categorical_features"]
    onehot_specs = layout["onehot_specs"]; offsets = layout["offsets"]; input_dim = layout["input_dim"]
    seq_len = int(cfg["seq_len"])
    cont_dim = len(CONTINUOUS_FEATURES)
    model = CausalTransformerAutoencoder(
        input_dim=input_dim, d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_ff"], dropout=cfg["dropout"], max_len=seq_len,
    ).to(device)
    buf_norm: Deque[np.ndarray] = deque(maxlen=seq_len); buf_raw: Deque[np.ndarray] = deque(maxlen=seq_len)
    state = torch.load(os.path.join(artifacts_dir, "model.pt"), map_location=device); model.load_state_dict(state); model.eval()
    df_raw = read_csv_lower(csv_path); require_columns(df_raw, csv_path)
    df_raw = apply_categorical_mapping(df_raw); df_raw[FEATURES] = df_raw[FEATURES].apply(pd.to_numeric, errors="coerce")
    specs = {col: {"codes": [float(v) for v in onehot_specs[col]["codes"]],
                   "value_to_index": {float(v): i for i, v in enumerate(onehot_specs[col]["codes"])},
                   "dim": onehot_specs[col]["dim"]} for col in CATEGORICAL_FEATURES}
    N_feat = len(FEATURES)
    vmin = np.full(N_feat, -np.inf, dtype=np.float32); vmax = np.full(N_feat,  np.inf, dtype=np.float32)
    sentinels_map: Dict[int, set] = {}
    for j, col in enumerate(FEATURES):
        rule = {**DEFAULT_RULE, **FEATURE_RULES.get(col, {})}
        if rule.get("vmin") is not None: vmin[j] = float(rule["vmin"])
        if rule.get("vmax") is not None: vmax[j] = float(rule["vmax"])
        sentinels_map[j] = set(float(x) for x in rule.get("sentinels", []))
    total_cat_dim = sum(specs[col]["dim"] for col in CATEGORICAL_FEATURES)
    idx_cont = [FEATURES.index(f) for f in CONTINUOUS_FEATURES]
    cont_idx_by_name = {f: i for i, f in enumerate(CONTINUOUS_FEATURES)}
    cat_feature_idx = {col: FEATURES.index(col) for col in CATEGORICAL_FEATURES}
    unknown_idx_by_col = {col: specs[col]["value_to_index"].get(float(DEFAULT_UNKNOWN_ID), 0) for col in CATEGORICAL_FEATURES}
    results: List[Dict] = []
    for i in range(len(df_raw)):
        row = df_raw.iloc[i][FEATURES].to_numpy(dtype=np.float32)
        # 欠損判定
        is_missing = np.zeros(len(FEATURES), dtype=np.float32)
        for j, v in enumerate(row):
            miss = (np.isnan(v) or float(v) in sentinels_map[j] or v < vmin[j] or v > vmax[j])
            is_missing[j] = 1.0 if miss else 0.0
        # 欠損補完（前値、なければ scaler.mean / UNKNOWN）
        x_raw = row.copy(); prev = buf_raw[-1] if len(buf_raw) > 0 else None
        for j, f in enumerate(FEATURES):
            if is_missing[j] == 1.0:
                if prev is not None:
                    x_raw[j] = prev[j]
                else:
                    if f in CONTINUOUS_FEATURES and hasattr(scaler, "mean_"):
                        x_raw[j] = float(scaler.mean_[cont_idx_by_name[f]])
                    else:
                        x_raw[j] = float(CATEGORY_MAPS.get(f, {}).get("UNKNOWN", DEFAULT_UNKNOWN_ID))
        # 正規化＋one-hot
        x_cont = x_raw[idx_cont].reshape(1, -1).astype(np.float32)
        x_cont_norm = scaler.transform(x_cont).reshape(-1).astype(np.float32)
        x_cat = np.zeros((total_cat_dim,), dtype=np.float32)
        for col in CATEGORICAL_FEATURES:
            v2i = specs[col]["value_to_index"]; val = float(x_raw[cat_feature_idx[col]])
            idx = v2i.get(val, unknown_idx_by_col[col]); x_cat[offsets[col] + idx] = 1.0
        x_exp_norm = np.concatenate([x_cont_norm, x_cat], axis=0).astype(np.float32)
        buf_raw.append(x_raw); buf_norm.append(x_exp_norm)
        # ウォームアップ：学習と同じ seq_len が溜まるまでスコアは None
        if len(buf_norm) < seq_len:
            results.append({"idx": i, "score": None, "is_anomaly": False})
            continue
        # マスクの展開（one-hot 分へ）
        mask_last_exp = expand_mask_for_onehot(is_missing.reshape(1, -1), idx_cont, specs, offsets)[0]
        # 推論
        seq = np.stack(list(buf_norm))
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
        with amp_ctx(device, use_amp=(device == "cuda")):
            out = model.reconstruct_last(seq_t)[0]
            recon_last = out.detach().float().cpu().numpy()
        # 連続特徴のみで MAE（学習時の閾値定義と一致）
        last_true_cont = seq[-1][:cont_dim]
        recon_last_cont = recon_last[:cont_dim]
        obs_mask_cont = 1.0 - mask_last_exp[:cont_dim]
        denom = float(obs_mask_cont.sum()) if obs_mask_cont.sum() > 0 else 1.0
        mae = float((np.abs(recon_last_cont - last_true_cont) * obs_mask_cont).sum() / denom)
        results.append({
            "idx": i,
            "score": mae,
            "is_anomaly": mae > float(thr_info["threshold"]),
            "missing_ratio_last": float(mask_last_exp.mean())
        })
    return results
def build_config(args: SimpleNamespace, layout: Dict[str, Any], metrics: Dict[str, Any], csv_paths: List[str]) -> Dict[str, Any]:
    return {"features": FEATURES, "seq_len": args.seq_len, "d_model": args.d_model, "nhead": args.nhead,
            "num_layers": args.num_layers, "dim_ff": args.dim_ff, "dropout": args.dropout, "metrics": metrics,
            "feature_rules": convert_rules_for_json(FEATURE_RULES), "categorical_features": CATEGORICAL_FEATURES,
            "category_maps": CATEGORY_MAPS, "layout": layout, "n_csvs": len(csv_paths), "csv_paths": csv_paths}
def convert_rules_for_json(rules: dict) -> dict:
    def convert_value(v):
        if isinstance(v, (np.float32, np.float64)): return float(v)
        if isinstance(v, (np.int32, np.int64)):     return int(v)
        if isinstance(v, (list, tuple)):            return [convert_value(x) for x in v]
        if isinstance(v, dict):                     return {k: convert_value(val) for k, val in v.items()}
        return v
    return convert_value(rules)
def save_loss_curve(train_losses: List[float], val_losses: Optional[List[float]], out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, color="blue", marker="o", label="Training loss")
    if val_losses is not None and len(val_losses) == len(train_losses):
        plt.plot(epochs, val_losses, color="orange", marker="o", label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Masked MSE loss"); plt.title("Training / Validation Loss")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()
# =========================================================
# メイン
# =========================================================
def main():
    default_csv = r"..\datarecode_train"
    parser = argparse.ArgumentParser(description="Train Transformer Autoencoder for pedal misapplication detection")
    parser.add_argument("--csv", "-i", dest="csv", type=str, default=default_csv, help="Path to single training CSV or directory")
    parser.add_argument("--csvdir", "-d", dest="csvdir", type=str, default=None, help="Directory containing multiple CSVs")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for CSVs in --csvdir or --csv when it is a directory")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    a = parser.parse_args()
    # 実行環境の自動判定（引数なしでOK）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    global amp_dtype
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[INFO] device={device}, amp={'ON' if use_amp else 'OFF'}, dtype={('bf16' if amp_dtype==torch.bfloat16 else 'fp16') if use_amp else 'N/A'}")
    args = SimpleNamespace(
        csv=a.csv, csvdir=a.csvdir, pattern=a.pattern,
        seq_len=128, batch_size=128, epochs=10, lr=5e-4,
        d_model=128, nhead=8, num_layers=2, dim_ff=512, dropout=0.2,
        percentile=99.5, out_dir=os.path.join("artifacts", "transformer_ae"),
        seed=a.seed, amp=use_amp, val_ratio=0.2, max_norm=1.0, weight_decay=1e-4,
        grad_clip=1.0, early_stopping=True, compile=True,
    )
    set_global_seed(args.seed)
    if torch.cuda.is_available():
        try:
            from torch.backends.cuda import sdp_kernel
            # Flash と MemEfficient を有効（Mathを無効）に設定
            sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
            print("[INFO] SDPA kernels: flash/mem_efficient enabled")
        except Exception as e:
            print(f"[WARN] SDPA enable failed: {e}")
    dataset, input_dim, scaler, layout, csv_paths = prepare_training_data(args.csv, args.csvdir, args.pattern, args.seq_len)
    if not csv_paths or dataset is None or scaler is None or layout is None or input_dim == 0:
        print("[WARN] 有効な学習データがないため、学習をスキップします。"); return
    if len(csv_paths) > 1:
        print(f"Loading {len(csv_paths)} CSVs from: {os.path.dirname(csv_paths[0])} (pattern={args.pattern})")
    else:
        print(f"Loading: {csv_paths[0]}")
    os.makedirs(args.out_dir, exist_ok=True)
    loss_png = os.path.join(args.out_dir, "loss_curve.png")
    model, metrics = train_on_dataset(
        dataset=dataset, input_dim=input_dim, seq_len=args.seq_len, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_ff=args.dim_ff, dropout=args.dropout,
        device=device, use_amp=use_amp, weight_decay=args.weight_decay,
        train_ratio=(1.0 - args.val_ratio), loss_plot_path=loss_png, compile_model=args.compile,
    )
    cont_dim = len(layout["continuous_features"])
    threshold_stats = compute_threshold_on_dataset(model=model, dataset=dataset, device=device,
                                                   percentile=args.percentile, use_amp=use_amp,
                                                   cont_dim=cont_dim)
    print(f"Threshold (p{args.percentile}): {threshold_stats['threshold']:.6f} "
          f"(mean={threshold_stats['mean']:.6f}, std={threshold_stats['std']:.6f}, "
          f"p10={threshold_stats['p10']:.6f}, p99={threshold_stats['p99']:.6f})")
    config = build_config(args, layout, metrics, csv_paths)
    save_artifacts(args.out_dir, model, scaler, config, threshold_stats)
    print(f"[RESULT] 最終エポックロス: {metrics['train_masked_mse']:.6f} / 最良エポック: {metrics['best_epoch']} / 最良ロス: {metrics['best_loss']:.6f}")
    ts = threshold_stats
    print(f"Threshold (percentile {args.percentile}): {ts['threshold']:.6f} (mean={ts['mean']:.6f}, std={ts['std']:.6f})")
if __name__ == "__main__":
    main()