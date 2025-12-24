import argparse
import json
import os
from collections import deque
from typing import Any, Deque, Dict, List, Tuple
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import sys
import shutil
from models.transformer_autoencoder import CausalTransformerAutoencoder
# =========================================================
# 1) 定数・パス
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = (BASE_DIR / ".." / "datarecode_test").resolve()
DEFAULT_UNKNOWN_ID = -1.0
MAX_LOG_WIDTH = 70
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
    temperature = float(thr_info.get("temperature", max((p90 - p50) / 6.0, 1e-6)))
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
       # 追加: AMP
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
        "threshold": float(thr_info["threshold"]),
        "mean": float(thr_info.get("mean", 0.0)),
        "std": float(thr_info.get("std", 1.0)),
        "p10": float(thr_info.get("p10", 0.0)),
        "p50": float(thr_info.get("p50", 0.0)),
        "p90": float(thr_info.get("p90", 0.0)),
        "p99": float(thr_info.get("p99", 0.0)),
        "temperature": float(thr_info.get("temperature", 1e-6)),
        "category_maps": category_maps,
    }
# =========================================================
# 4) 入力列挙（単一ファイル／ディレクトリ・再帰）
# =========================================================
def list_input_files(csv_arg: str, recursive: bool) -> List[str]:
    """
    --csv がファイルならそのパス、ディレクトリなら配下の CSV を列挙して返す。
    recursive=True ならサブディレクトリまで探索。
    """
    input_path = Path(csv_arg).expanduser()
    if input_path.is_dir():
        patterns = ["*.csv", "*.CSV"]
        files: List[str] = []
        if recursive:
            for pat in patterns:
                files += [str(p) for p in input_path.rglob(pat)]
        else:
            for pat in patterns:
                files += [str(p) for p in input_path.glob(pat)]
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path} (recursive={recursive})")
        return files
    elif input_path.is_file():
        return [str(input_path)]
    else:
        raise FileNotFoundError(f"--csv not found: {input_path.resolve(strict=False)}")
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
from contextlib import nullcontext
def mae_last_continuous(
    seq_t: torch.Tensor,           # (1, T, D) 入力の正規化後ベクトル
    mask_last_exp: np.ndarray,     # (D,) one-hot 拡張後の欠損マスク（最後ステップ分）
    model: CausalTransformerAutoencoder,
    cont_dim: int,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    """
    学習時の compute_batch_mae_last と一致する MAE 計算（連続特徴のみ、最後ステップ、観測マスクで重み付け）。
    """
    autocast_ctx = torch.amp.autocast('cuda', enabled=(use_amp and device == "cuda"), dtype=amp_dtype) if device == "cuda" else nullcontext()
    with autocast_ctx:
        with torch.no_grad():
            recon_last = model.reconstruct_last(seq_t)[0]  # (D,)
    # 連続成分を抽出
    recon_last_cont = recon_last[:cont_dim].float()
    last_true_cont = seq_t[0, -1, :cont_dim].float()
    # 観測マスク（連続部分のみ）
    obs_mask_cont = torch.from_numpy(1.0 - mask_last_exp[:cont_dim]).to(device).float()
    denom = obs_mask_cont.sum().clamp(min=1.0)
    mae = (recon_last_cont.sub(last_true_cont).abs() * obs_mask_cont).sum() / denom
    return float(mae.detach().cpu().item())
# =========================================================
# 6) スコア計算（ファイル単位のストリーミング推論）
# =========================================================
def compute_stream_scores_for_file(
    csv_path: str,
    ctx: Dict[str, Any],
    min_ctx: int,
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
    cont_dim = len(CONTINUOUS_FEATURES)

    # CSV 読み込み
    df = pd.read_csv(
        csv_path,
        na_values=["(nan)", "nan", "NaN", "NULL", "None", "", " "],
        keep_default_na=True,
        low_memory=False,
    )
    # 前処理（学習時準拠）
    df = preprocess_dataframe_for_inference(df, csv_path, features, CATEGORICAL_FEATURES, category_maps)
    # ストリーム用バッファ
    buf_norm: Deque[np.ndarray] = deque(maxlen=seq_len)  # 正規化＋one-hot 結合後
    buf_raw: Deque[np.ndarray] = deque(maxlen=seq_len)   # 欠損補完後の元スケール（F）
    anomaly_scores: List[float] = []
    is_flags: List[int] = []
    errors: List[float] = []
    levels: List[float] = []
    # ウォームアップは最低限 seq_len を満たす
    warmup_needed = max(int(min_ctx), int(seq_len))
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
        for col in CATEGORICAL_FEATURES:
            sp = specs[col]
            v2i = sp["value_to_index"]
            val = float(x_raw[cat_feature_idx[col]])
            idx = v2i.get(val, unknown_idx_by_col[col])
            x_cat[int(offsets[col]) + int(idx)] = 1.0
        # 7) 入力ベクトル結合（連続＋カテゴリ）
        x_exp_norm = np.concatenate([x_cont_norm, x_cat], axis=0).astype(np.float32)
        # バッファ更新
        buf_raw.append(x_raw)
        buf_norm.append(x_exp_norm)
        # ウォームアップ（最低限 seq_len を満たす）
        if len(buf_norm) < (warmup_needed/3):
            anomaly_scores.append(float("nan"))
            errors.append(float("nan"))
            is_flags.append(float("nan"))
            levels.append(float("nan"))
            continue
        # 8) 欠損マスク（最後ステップ分）を入力次元に拡張
        mask_last_exp = expand_mask_for_onehot_precomputed(
            is_missing.reshape(1, -1),
            idx_cont,
            CATEGORICAL_FEATURES,
            specs,
            offsets,
            features
        )[0]
        # 9) 予測とスコア
        ctx_len = min(len(buf_norm), int(seq_len))     # 直近の文脈長（最大 seq_len）
        seq = np.stack(list(buf_norm)[-ctx_len:])
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
        # 学習時と同じ MAE（最後ステップ・連続特徴のみ・欠損除外）で算出
        mae_cont = mae_last_continuous(seq_t, mask_last_exp, model, cont_dim, device, use_amp, amp_dtype)

        # 正規化スコア（p10〜p99 で 0〜1）
        low, high = float(ctx["p10"]), float(ctx["p99"])
        score = (mae_cont - low) / max(high - low, 1e-6)
        score = float(np.clip(score, 0.0, 1.0))

        # 閾値判定（学習時と一致）
        is_anomaly = (mae_cont > threshold)

        anomaly_scores.append(score)     # 0〜1 の連続スコア
        errors.append(mae_cont)          # 生の MAE（連続のみ）
        is_flags.append(1.0 if is_anomaly else 0.0)
        # 分布ベースの4段階レベル（0〜3）
        bounds_mae = [float(ctx["p50"]), float(ctx["p90"]), float(ctx["threshold"])]
        level = int(np.digitize(mae_cont, bounds_mae))  # 0,1,2,3
        levels.append(float(level))

    df_out = df.copy()
    df_out["anomaly"] = anomaly_scores      # 正規化スコア（0〜1）
    df_out["error"] = errors              # MAE（連続のみ、最後ステップ、欠損除外）
    df_out["is_anomaly"] = is_flags
    df_out["anomaly_level"] = levels
    # ウォームアップ後から is_anomaly を埋める場合は保持した配列を使って代入
    # 例）df_out.loc[~np.isnan(df_out["score"].values), "is_anomaly"] = is_flags[start_idx:]
    return df_out
# =========================================================
# 7) メイン（CLI）
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Stream anomaly scoring for test CSV using trained Transformer AE")
    parser.add_argument("--csv", default=DEFAULT_INPUT_DIR, help="Path to test CSV or a directory that contains CSVs")
    parser.add_argument("--artifacts_dir", default=os.path.join("artifacts", "transformer_ae"), help="Path to trained artifacts")
    parser.add_argument("--min_ctx", type=int, default=2, help="Minimum steps before scoring (stream warm-up)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search CSVs in subdirectories when --csv is a directory")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # アーティファクト読み込み
    cfg, thr_info, scaler, paths = load_artifacts(args.artifacts_dir)
    # 推論用コンテキスト構築
    ctx = build_inference_context(cfg, thr_info, scaler, paths["model_path"], device)
    # 入力列挙
    files = list_input_files(args.csv, args.recursive)
    # 出力ディレクトリ
    out_dir = "result"
    os.makedirs(out_dir, exist_ok=True)
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
            df_out = compute_stream_scores_for_file(csv_path, ctx, min_ctx=args.min_ctx)
            base = p.stem
            out_path = os.path.join(out_dir, f"{base}_anomaly.csv")
            df_out.to_csv(out_path, index=False, float_format="%.6f")
            # 成功時は何も表示しない（次のファイルの Processing 行で上書き）
        except Exception as e:
            # 警告は独立した行で出すため、まず現在の行を確定
            logger.newline()
            print(f"[WARN] Failed to process {p.name}: {e}")
            # 以降は次のループで再び1行進行表示に戻る
    # ループ終了後、最後の進行表示行を確定
    logger.newline()
if __name__ == "__main__":
    main()