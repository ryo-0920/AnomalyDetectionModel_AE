#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ON (事故) と OFF (正常) の *_anomaly.csv から per-file サマリを生成し、
TPR を ON 側（ただし「初検知が A2 のファイルのみ」）で、FPR を OFF 側で計算するスクリプト。
改良点:
- OFF 側の台帳照合でログを抑制し、問題点（未検出 / あいまいマッチ）を要約して出力するようにしました。
- plot_macro_heatmap を復元して main で呼び出します。
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple
# Headless 対応: backend を Agg にする（必ず matplotlib.pyplot より前）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 日本語フォント設定（必要なら環境に合わせて変更）
jp_font = "MS Gothic"  # Windows の例。Linux なら "Noto Sans CJK JP" 等に変更してください
rcParams["font.family"] = jp_font
rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止
import numpy as np
import pandas as pd
import seaborn as sns
# ============================
# 共通ユーティリティ
# ============================
def normalize_basename(name: str) -> str:
    p = Path(name)
    stem = p.stem
    if stem.endswith("_anomaly"):
        stem = stem[:-len("_anomaly")]
    if stem.isdigit():
        try:
            stem = str(int(stem))
        except ValueError:
            pass
    return stem
def safe_float(v) -> Optional[float]:
    try:
        return float(v) if not pd.isna(v) else None
    except Exception:
        return None
def to_bool(
    series: Optional[pd.Series],
    index: Optional[pd.Index] = None,
    default: bool = False,
) -> pd.Series:
    if series is None:
        if index is None:
            raise ValueError("index is required when series is None")
        return pd.Series(default, index=index)
    return series.fillna(default).astype(bool)
def safe_ratio(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    denom = np.where(denom > 0, denom, np.nan)
    return num / denom
def basic_stats(x: np.ndarray) -> Dict[str, Any]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": np.nan, "median": np.nan, "count": 0}
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "count": int(x.size)}
# ============================
# ON / OFF 固定設定（ラベル・台帳）
# ============================
def load_label_intervals(label_cfg: dict) -> pd.DataFrame:
    path = label_cfg["path"]
    sheet_name = label_cfg.get("sheet_name", 0)
    file_col_idx = int(label_cfg["file_column_excel_index"]) - 1
    a1_col_idx = int(label_cfg["a1_override_col"]) - 1
    k_col_idx = int(label_cfg["k_col"]) - 1
    m_col_idx = int(label_cfg["m_col"]) - 1
    n_col_idx = int(label_cfg["n_col"]) - 1
    delta = float(label_cfg.get("a1_delta_seconds", 5.0))
    df = pd.read_excel(path, sheet_name=sheet_name, header=0)
    max_idx = max(file_col_idx, a1_col_idx, k_col_idx, m_col_idx, n_col_idx)
    if max_idx >= df.shape[1]:
        raise ValueError(f"label sheet の列数が足りません: 必要列インデックス={max_idx}, df.shape={df.shape}")
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        raw_file = r.iloc[file_col_idx]
        if pd.isna(raw_file):
            continue
        basename = normalize_basename(str(raw_file))
        a1_over_raw = r.iloc[a1_col_idx]
        k_raw = r.iloc[k_col_idx]
        m_raw = r.iloc[m_col_idx]
        n_raw = r.iloc[n_col_idx]
        k = safe_float(k_raw)
        m = safe_float(m_raw)
        n = safe_float(n_raw)
        a1_over = safe_float(a1_over_raw)
        A1_start = None
        A2_start = None
        A3_start = None
        A3_end = None
        if k is not None:
            A2_start = k
            if a1_over is not None:
                A1_start = a1_over
            else:
                A1_start = k - delta
        if m is not None:
            A3_start = m
        if n is not None:
            A3_end = n
        if A1_start is None and A2_start is None and A3_start is None:
            continue
        rows.append({"basename": basename, "A1_start": A1_start, "A2_start": A2_start, "A3_start": A3_start, "A3_end": A3_end})
    label_df = pd.DataFrame(rows)
    label_df = label_df.drop_duplicates(subset=["basename"], keep="last")
    return label_df
def load_normal_basenames_from_ledger(ledger_cfg: dict) -> List[str]:
    path = ledger_cfg["path"]
    sheet_name = ledger_cfg.get("sheet_name", 0)
    file_col_idx = int(ledger_cfg["file_column_excel_index"]) - 1
    df = pd.read_excel(path, sheet_name=sheet_name, header=0)
    basenames: List[str] = []
    for _, r in df.iterrows():
        raw_file = r.iloc[file_col_idx]
        if pd.isna(raw_file):
            continue
        b = normalize_basename(str(raw_file))
        basenames.append(b)
    basenames = sorted(set(basenames))
    return basenames
def build_per_file_summary_from_dir(
    result_dir: Path,
    label_df: pd.DataFrame,
    accel_col_name: str = "accelpedalangle",
) -> pd.DataFrame:
    result_dir = Path(result_dir)
    rows: List[Dict[str, Any]] = []
    for csv_path in result_dir.glob("*_anomaly.csv"):
        df = pd.read_csv(csv_path)
        basename = normalize_basename(csv_path.name)
        lbl = label_df[label_df["basename"] == basename]
        if lbl.empty:
            print(f"[WARN] ラベルシートに basename={basename} の行がありません。スキップします。")
            continue
        lbl_row = lbl.iloc[0]
        A1_start = safe_float(lbl_row["A1_start"])
        A2_start = safe_float(lbl_row["A2_start"])
        A3_start = safe_float(lbl_row["A3_start"])
        A3_end = safe_float(lbl_row["A3_end"])
        if "time" not in df.columns or "is_anomaly" not in df.columns:
            print(f"[WARN] {csv_path.name}: 'time' または 'is_anomaly' 列がありません。スキップします。")
            continue
        time = pd.to_numeric(df["time"], errors="coerce")
        is_anom = df["is_anomaly"].fillna(0).astype(int)
        if accel_col_name in df.columns:
            accel = pd.to_numeric(df[accel_col_name], errors="coerce")
        else:
            accel = None
        has_anom = (is_anom == 1)
        total_frames = len(df)
        num_anom_frames = int(has_anom.sum())
        anomaly_rate = num_anom_frames / total_frames if total_frames > 0 else 0.0
        detected = bool(has_anom.any())
        if detected:
            first_idx = has_anom.to_numpy().nonzero()[0][0]
            first_det_time = float(time.iloc[first_idx])
        else:
            first_det_time = np.nan
        collision_time = A3_start if (A3_start is not None and np.isfinite(A3_start)) else np.nan
        if detected and all(v is not None and np.isfinite(v) for v in (A1_start, A2_start, A3_start)):
            t = first_det_time
            if A1_start <= t < A2_start:
                first_phase = "A1"
            elif A2_start <= t < A3_start:
                first_phase = "A2"
            elif A3_end is not None and np.isfinite(A3_end) and A3_start <= t < A3_end:
                first_phase = "A3"
            else:
                first_phase = "OTHER"
        else:
            first_phase = "OTHER"
        pre_collision_detected = bool(detected and collision_time is not None and np.isfinite(collision_time) and (first_det_time < collision_time))
        a1_detected = False
        a2_coverage = 0.0
        a3_detected = False
        if detected and A1_start is not None and A2_start is not None and np.isfinite(A1_start) and np.isfinite(A2_start):
            mask_A1 = (time >= A1_start) & (time < A2_start)
            a1_detected = bool((has_anom & mask_A1).any())
        if detected and A2_start is not None and A3_start is not None and np.isfinite(A2_start) and np.isfinite(A3_start):
            mask_A2 = (time >= A2_start) & (time < A3_start)
            a2_coverage = float((has_anom & mask_A2).sum())
        if detected and A3_start is not None and A3_end is not None and np.isfinite(A3_start) and np.isfinite(A3_end):
            mask_A3 = (time >= A3_start) & (time < A3_end)
            a3_detected = bool((has_anom & mask_A3).any())
        if accel is not None and A2_start is not None and A3_start is not None and np.isfinite(A2_start) and np.isfinite(A3_start):
            mask_A2_for_accel = (time >= A2_start) & (time < A3_start)
            if mask_A2_for_accel.any():
                accel_A2_max = float(accel[mask_A2_for_accel].max())
            else:
                accel_A2_max = np.nan
        else:
            accel_A2_max = np.nan
        def bin_accel(x: float) -> str:
            if pd.isna(x) or x < 0:
                return "unknown"
            if x >= 90:
                return "90-100%"
            if x >= 80:
                return "80-100%"
            if x >= 70:
                return "70-100%"
            return "0-100%"
        accel_A2_bin = bin_accel(accel_A2_max)
        label = 1
        rows.append({
            "basename": basename, "label": label, "detected": detected,
            "a1_detected": a1_detected, "a2_coverage": a2_coverage, "a3_detected": a3_detected,
            "first_detect_phase": first_phase, "first_detection_time": first_det_time,
            "pre_collision_detected": pre_collision_detected, "collision_time": collision_time,
            "A1_start": A1_start, "A2_start": A2_start, "A3_start": A3_start, "A3_end": A3_end,
            "accel_A2_max": accel_A2_max, "accel_A2_bin": accel_A2_bin,
            "num_anomaly_frames": num_anom_frames, "total_frames": total_frames, "anomaly_rate": anomaly_rate
        })
    if not rows:
        raise ValueError(f"[ERROR] {result_dir} 内に有効な *_anomaly.csv がありません。")
    return pd.DataFrame(rows)
def build_per_file_summary_normal_from_dir(
    result_dir: Path,
    normal_basenames: List[str],
    accel_col_name: str = "accelpedalangle",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    OFF ディレクトリの *_anomaly.csv を台帳 normal_basenames に照合して per-file summary を作成。
    ログは必要最小限に抑える（verbose=True で詳細ログ出力）。
    """
    result_dir = Path(result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)
    # OFF ファイル一覧を辞書化 (key: normalized lower basename -> list[Path])
    file_map: Dict[str, List[Path]] = {}
    for p in result_dir.glob("*_anomaly.csv"):
        stem = p.stem
        if stem.endswith("_anomaly"):
            key = stem[:-len("_anomaly")]
        else:
            key = stem
        key_norm = key.lower().strip()
        file_map.setdefault(key_norm, []).append(p)
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    ambiguous: Dict[str, List[str]] = {}  # ledger basename -> list of candidate filenames
    matched_count = 0
    # helper: find candidate list by flexible rules
    def find_candidates(key: str) -> List[Path]:
        # 1) exact match (already covered by file_map)
        if key in file_map:
            return file_map[key]
        # 2) key contained in file_map keys or vice versa
        cand = []
        for k, paths in file_map.items():
            if key in k or k in key:
                cand.extend(paths)
        if cand:
            return list(dict.fromkeys(cand))  # unique preserve order
        # 3) try removing underscores/zeros etc. (simple normalization)
        k2 = key.replace("_", "")
        cand2 = []
        for k, paths in file_map.items():
            if k.replace("_", "") == k2:
                cand2.extend(paths)
        if cand2:
            return list(dict.fromkeys(cand2))
        return []
    for nb in sorted(normal_basenames):
        key = str(nb).lower().strip()
        candidates = find_candidates(key)
        if not candidates:
            missing.append(nb)
            continue
        if len(candidates) > 1:
            ambiguous[nb] = [p.name for p in candidates]
            # choose the best candidate heuristically: prefer exact ending with nb or shortest name
            chosen = None
            for p in candidates:
                k = p.stem
                if k.lower().endswith(key):
                    chosen = p
                    break
            if chosen is None:
                # fallback: choose lexicographically smallest (stable)
                chosen = sorted(candidates, key=lambda p: p.name)[0]
            chosen_path = chosen
            if verbose:
                print(f"[WARN] 部分一致で複数候補 (OFF): ledger='{nb}' -> 使用='{chosen_path.name}' 他候補数={len(candidates)-1}")
        else:
            chosen_path = candidates[0]
            if verbose:
                print(f"[INFO] 部分一致でマッチ (OFF): ledger='{nb}' -> file='{chosen_path.name}'")
        # 読み込み・サマリ作成
        try:
            df = pd.read_csv(chosen_path)
        except Exception as e:
            if verbose:
                print(f"[WARN] {chosen_path.name} を読み込めませんでした: {e}. スキップします。")
            missing.append(nb)
            continue
        if "time" not in df.columns or "is_anomaly" not in df.columns:
            if verbose:
                print(f"[WARN] {chosen_path.name}: 'time' または 'is_anomaly' 列がありません。スキップします。")
            missing.append(nb)
            continue
        is_anom = df["is_anomaly"].fillna(0).astype(int)
        has_anom = (is_anom == 1)
        detected = bool(has_anom.any())
        total_frames = len(df)
        num_anom_frames = int(has_anom.sum())
        anomaly_rate = num_anom_frames / total_frames if total_frames > 0 else 0.0
        rows.append({
            "basename": normalize_basename(chosen_path.name), "label": 0, "detected": detected,
            "a1_detected": False, "a2_coverage": 0.0, "a3_detected": False,
            "first_detect_phase": "OTHER", "first_detection_time": np.nan, "pre_collision_detected": False,
            "collision_time": np.nan, "A1_start": np.nan, "A2_start": np.nan, "A3_start": np.nan, "A3_end": np.nan,
            "accel_A2_max": np.nan, "accel_A2_bin": "unknown",
            "num_anomaly_frames": num_anom_frames, "total_frames": total_frames, "anomaly_rate": anomaly_rate
        })
        matched_count += 1
    # サマリログ（簡潔に1回だけ）
    print(f"[INFO] OFF: 台帳件数={len(normal_basenames)}, 正常にマッチした件数={matched_count}, 見つからなかった件数={len(missing)}")
    if missing:
        MAX_SHOW = 20
        print(f"[WARN] OFF 台帳の basename のうち見つからなかったもの（先頭 {min(MAX_SHOW, len(missing))} 件）: {missing[:MAX_SHOW]}")
    if ambiguous and not verbose:
        print(f"[WARN] OFF: 部分一致で複数候補が見つかった台帳が {len(ambiguous)} 件あります。代表例を表示します（必要なら verbose=True で個別ログを確認してください）。")
        for ledger_name, cand_list in list(ambiguous.items())[:10]:
            print(f"  ledger='{ledger_name}' -> candidates={cand_list}")
        if len(ambiguous) > 10:
            print(f"  ... (他 {len(ambiguous)-10} 件)")
    if not rows:
        raise ValueError(f"[ERROR] {result_dir} 内に台帳に載っている *_anomaly.csv がありません。")
    return pd.DataFrame(rows)
def plot_first_phase_distribution(
    pf: pd.DataFrame,
    out_dir: Path,
    group_col: str = "A2_duration_bin",
    use_priority_only: bool = False,
) -> None:
    df = pf.copy()
    df = df[df["label"] == 1].copy()
    if df.empty:
        print("[INFO] first_phase plot: label==1 のファイルがありません。")
        return
    if use_priority_only and "is_priority_case" in df.columns:
        df = df[df["is_priority_case"].fillna(False)].copy()
        if df.empty:
            print("[INFO] first_phase plot: 重点事故がありません。")
            return
    df["first_detect_phase"] = df.get("first_detect_phase", pd.Series(index=df.index)).fillna("OTHER").astype(str)
    if group_col not in df.columns:
        print(f"[WARN] group_col={group_col} が存在しないため、全体で 1 グループとして描画します。")
        df[group_col] = "all"
    groups = [g for g in df[group_col].unique() if pd.notna(g)]
    if not groups:
        print("[INFO] first_phase plot: グループがありません。")
        return
    phases: Sequence[str] = ["A1", "A2", "A3", "OTHER"]
    n_groups = len(groups)
    n_cols = 2
    n_rows = (n_groups + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for idx, g in enumerate(sorted(groups)):
        ax = axes[idx // n_cols][idx % n_cols]
        gdf = df[df[group_col] == g]
        vc = gdf["first_detect_phase"].value_counts(normalize=True)
        ratios = [vc.get(p, 0.0) for p in phases]
        x = list(range(len(phases)))
        ax.bar(x, ratios, color="skyblue", alpha=0.7)
        ax.plot(x, ratios, color="blue", marker="o")
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("割合")
        ax.set_title(f"{group_col} = {g} での初検知フェーズ分布")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    for i in range(n_groups, n_rows * n_cols):
        fig.delaxes(axes[i // n_cols][i % n_cols])
    fig.tight_layout()
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "first_phase_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] first_detect_phase 分布グラフを保存しました: {out_path}")
def compute_confusion_matrices(per_file_df: pd.DataFrame) -> dict:
    df = per_file_df.copy()
    df = df[df["label"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("label 0/1 のファイルがありません。")
    df["is_abnormal"] = df["label"] == 1
    has_det_A1 = to_bool(df.get("a1_detected"), df.index)
    has_det_A3 = to_bool(df.get("a3_detected"), df.index)
    a2_cov = df.get("a2_coverage")
    has_det_A2 = (pd.Series(False, index=df.index) if a2_cov is None else a2_cov.fillna(0.0) > 0.0)
    has_det_label_window = has_det_A1 | has_det_A2 | has_det_A3
    detected_any = to_bool(df.get("detected"), df.index)
    first_phase = df.get("first_detect_phase", pd.Series(index=df.index)).fillna("OTHER").astype(str)
    fd_in_A123 = first_phase.isin(["A1", "A2", "A3"])
    fd_A2_window = detected_any & (first_phase == "A2")
    fd_A23_window = detected_any & first_phase.isin(["A2", "A3"])
    res: Dict[str, Any] = {}
    def cm_stats(y_pred: np.ndarray, name: str) -> None:
        is_pos = df["is_abnormal"]
        is_neg = ~df["is_abnormal"]
        tp = int(((y_pred == 1) & is_pos).sum())
        fn = int(((y_pred == 0) & is_pos).sum())
        fp = int(((y_pred == 1) & is_neg).sum())
        tn = int(((y_pred == 0) & is_neg).sum())
        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        res[name] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "total": total, "Accuracy": acc, "Precision": prec, "Recall": rec, "FPR": fpr}
    y_pred_lenient = np.where(df["is_abnormal"], has_det_label_window, detected_any).astype(int)
    cm_stats(y_pred_lenient, "lenient")
    y_pred_strict = np.where(df["is_abnormal"], fd_in_A123, detected_any).astype(int)
    cm_stats(y_pred_strict, "strict")
    y_pred_A2_window = np.where(df["is_abnormal"], fd_A2_window, detected_any).astype(int)
    cm_stats(y_pred_A2_window, "A2_to_A3start")
    y_pred_A23_window = np.where(df["is_abnormal"], fd_A23_window, detected_any).astype(int)
    cm_stats(y_pred_A23_window, "A2_to_A3end")
    res["y_pred_lenient"] = y_pred_lenient
    res["y_pred_strict"] = y_pred_strict
    res["y_pred_A2_to_A3start"] = y_pred_A2_window
    res["y_pred_A2_to_A3end"] = y_pred_A23_window
    res["first_phase"] = first_phase
    res["has_det_label_window"] = has_det_label_window
    res["detected_any"] = detected_any
    return res
def ensure_anomaly_rate_column(pf: pd.DataFrame) -> pd.DataFrame:
    df = pf.copy()
    if "anomaly_rate" not in df.columns:
        if {"num_anomaly_frames", "total_frames"}.issubset(df.columns):
            num = pd.to_numeric(df["num_anomaly_frames"], errors="coerce").fillna(0.0)
            tot = pd.to_numeric(df["total_frames"], errors="coerce").replace(0, np.nan)
            df["anomaly_rate"] = (num / tot).fillna(0.0)
        elif "detected" in df.columns:
            df["anomaly_rate"] = df["detected"].fillna(False).astype(int).astype(float)
        else:
            df["anomaly_rate"] = 0.0
    return df
def compute_margin_stats(per_file_df: pd.DataFrame) -> dict:
    df = per_file_df.copy()
    df = df[df["label"] == 1].copy()
    df = df[df["pre_collision_detected"].fillna(False)].copy()
    if df.empty:
        return {}
    for col in ["first_detection_time", "collision_time", "A1_start", "A2_start"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["first_detection_time", "collision_time", "A1_start", "A2_start"]).copy()
    if df.empty:
        return {}
    t_det = df["first_detection_time"].to_numpy(dtype=float)
    tA3 = df["collision_time"].to_numpy(dtype=float)
    tA1 = df["A1_start"].to_numpy(dtype=float)
    tA2 = df["A2_start"].to_numpy(dtype=float)
    dt = tA3 - t_det
    LA1A2 = tA3 - tA1
    LA1 = tA2 - tA1
    LA2 = tA3 - tA2
    r_A1A2 = safe_ratio(dt, LA1A2)
    phase = df["first_detect_phase"].fillna("OTHER").astype(str).to_numpy()
    mask_A1 = phase == "A1"
    mask_A2 = phase == "A2"
    r_A1_phase = safe_ratio(dt[mask_A1], LA1A2[mask_A1])
    r_A2_phase = safe_ratio(dt[mask_A2], LA2[mask_A2])
    stats: Dict[str, Any] = {}
    stats["margin_seconds_all"] = basic_stats(dt)
    stats["ratio_A1A2_all"] = basic_stats(r_A1A2)
    stats["ratio_A1_phase"] = basic_stats(r_A1_phase)
    stats["ratio_A2_phase"] = basic_stats(r_A2_phase)
    return stats
def add_segment_columns(pf: pd.DataFrame) -> pd.DataFrame:
    df = pf.copy()
    if "A2_duration_bin" not in df.columns:
        if "A2_start" in df.columns and "A3_start" in df.columns:
            dur = pd.to_numeric(df["A3_start"], errors="coerce") - pd.to_numeric(df["A2_start"], errors="coerce")
            def bin_duration(x: float) -> str:
                if pd.isna(x) or x < 0:
                    return "unknown"
                if x < 3:
                    return "0-3s"
                if x < 5:
                    return "3-5s"
                return ">=5s"
            df["A2_duration_bin"] = dur.map(bin_duration)
        else:
            df["A2_duration_bin"] = "unknown"
    if "accel_A2_bin" not in df.columns:
        df["accel_A2_bin"] = "unknown"
    if "shift_at_collision" not in df.columns:
        df["shift_at_collision"] = np.nan
    return df
def overlapping_accel_bins(x: float) -> List[str]:
    if pd.isna(x) or x < 0:
        return ["unknown"]
    bins = ["0-100%"]
    if x >= 70:
        bins.append("70-100%")
    if x >= 80:
        bins.append("80-100%")
    if x >= 90:
        bins.append("90-100%")
    return bins
def aggregate_by_segments(
    pf: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    use_priority_only: bool = False,
) -> pd.DataFrame:
    df = pf.copy()
    if group_cols is None:
        group_cols = ["A2_duration_bin"]
    df = df[df["label"] == 1].copy()
    if df.empty:
        print("[INFO] label==1 のファイルがありません。")
        return pd.DataFrame()
    if use_priority_only and "is_priority_case" in df.columns:
        df = df[df["is_priority_case"].fillna(False)].copy()
        if df.empty:
            print("[INFO] 重点事故 (is_priority_case=True) がありません。")
            return pd.DataFrame()
    if "accel_A2_bin" in group_cols and "accel_A2_max" in df.columns:
        expanded_rows = []
        for _, row in df.iterrows():
            bins = overlapping_accel_bins(row["accel_A2_max"])
            for b in bins:
                new_row = row.copy()
                new_row["accel_A2_bin"] = b
                expanded_rows.append(new_row)
        df = pd.DataFrame(expanded_rows)
        if df.empty:
            print("[INFO] aggregate_by_segments: accel_A2_bin 展開後にデータが空になりました。")
            return pd.DataFrame()
    df["first_detect_phase"] = df.get("first_detect_phase", pd.Series(index=df.index)).fillna("OTHER").astype(str)
    has_det_A1 = to_bool(df.get("a1_detected"), df.index)
    has_det_A3 = to_bool(df.get("a3_detected"), df.index)
    a2_cov = df.get("a2_coverage", pd.Series(0.0, index=df.index))
    has_det_A2 = a2_cov.fillna(0.0) > 0.0
    df["has_det_label_window"] = has_det_A1 | has_det_A2 | has_det_A3
    df["pre_collision_detected"] = df.get("pre_collision_detected", pd.Series(False, index=df.index)).fillna(False)
    for col in ["first_detection_time", "collision_time", "A1_start", "A2_start"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["margin_sec"] = df["collision_time"] - df["first_detection_time"]
    df["A1A2_len"] = df["collision_time"] - df["A1_start"]
    df["margin_ratio_A1A2"] = df["margin_sec"] / df["A1A2_len"]
    df.loc[df["A1A2_len"] <= 0, "margin_ratio_A1A2"] = np.nan
    df["A1_len"] = df["A2_start"] - df["A1_start"]
    df["A2_len"] = df["collision_time"] - df["A2_start"]
    results: List[Dict[str, Any]] = []
    grouped = df.groupby(list(group_cols), dropna=False)
    for g_keys, g in grouped:
        if not isinstance(g_keys, tuple):
            g_keys = (g_keys,)
        N_abn = int(g.shape[0])
        N_det = int(g["has_det_label_window"].sum())
        vc_phase = g["first_detect_phase"].value_counts()
        N_A1 = int(vc_phase.get("A1", 0))
        N_A2 = int(vc_phase.get("A2", 0))
        N_A3 = int(vc_phase.get("A3", 0))
        N_OTHER = int(vc_phase.get("OTHER", 0))
        def ratio(n: int) -> float:
            return n / N_abn if N_abn > 0 else np.nan
        R_A1 = ratio(N_A1)
        R_A2 = ratio(N_A2)
        R_A3 = ratio(N_A3)
        R_OTHER = ratio(N_OTHER)
        g_margin = g[g["pre_collision_detected"]].copy()
        def stat(x: np.ndarray) -> Tuple[float, float, int]:
            if x.size == 0:
                return (np.nan, np.nan, 0)
            return (float(np.mean(x)), float(np.median(x)), int(x.size))
        if not g_margin.empty:
            margin_sec = g_margin["margin_sec"].to_numpy(dtype=float)
            margin_sec = margin_sec[~np.isnan(margin_sec)]
            ratio_A1A2 = g_margin["margin_ratio_A1A2"].to_numpy(dtype=float)
            ratio_A1A2 = ratio_A1A2[~np.isnan(ratio_A1A2)]
            m_sec_mean, m_sec_med, m_sec_cnt = stat(margin_sec)
            rA1A2_mean, rA1A2_med, rA1A2_cnt = stat(ratio_A1A2)
        else:
            m_sec_mean = m_sec_med = np.nan
            m_sec_cnt = 0
            rA1A2_mean = rA1A2_med = np.nan
            rA1A2_cnt = 0
        g_A1 = g_margin[g_margin["first_detect_phase"] == "A1"].copy()
        if not g_A1.empty:
            dt_A1 = g_A1["margin_sec"].to_numpy(dtype=float)
            LA1A3 = g_A1["A1A2_len"].to_numpy(dtype=float)
            mask = LA1A3 > 0
            r_A1 = dt_A1[mask] / LA1A3[mask] if mask.any() else np.array([], dtype=float)
            m_sec_A1_mean, m_sec_A1_med, m_sec_A1_cnt = stat(dt_A1)
            rA1_mean, rA1_med, rA1_cnt = stat(r_A1)
        else:
            m_sec_A1_mean = m_sec_A1_med = np.nan
            m_sec_A1_cnt = 0
            rA1_mean = rA1_med = np.nan
            rA1_cnt = 0
        g_A2 = g_margin[g_margin["first_detect_phase"] == "A2"].copy()
        if not g_A2.empty:
            dt_A2 = g_A2["margin_sec"].to_numpy(dtype=float)
            LA2 = g_A2["A2_len"].to_numpy(dtype=float)
            mask2 = LA2 > 0
            r_A2 = dt_A2[mask2] / LA2[mask2] if mask2.any() else np.array([], dtype=float)
            m_sec_A2_mean, m_sec_A2_med, m_sec_A2_cnt = stat(dt_A2)
            rA2_mean, rA2_med, rA2_cnt = stat(r_A2)
        else:
            m_sec_A2_mean = m_sec_A2_med = np.nan
            m_sec_A2_cnt = 0
            rA2_mean = rA2_med = np.nan
            rA2_cnt = 0
        row: Dict[str, Any] = {}
        for col_name, key_val in zip(group_cols, g_keys):
            row[col_name] = key_val
        row["N_abnormal"] = N_abn
        row["N_detected_label_window"] = N_det
        row["N_phase_A1"] = N_A1
        row["N_phase_A2"] = N_A2
        row["N_phase_A3"] = N_A3
        row["N_phase_OTHER"] = N_OTHER
        row["R_phase_A1"] = R_A1
        row["R_phase_A2"] = R_A2
        row["R_phase_A3"] = R_A3
        row["R_phase_OTHER"] = R_OTHER
        row["margin_sec_mean_A1"] = m_sec_A1_mean
        row["margin_sec_median_A1"] = m_sec_A1_med
        row["margin_sec_count_A1"] = m_sec_A1_cnt
        row["margin_sec_mean_A2"] = m_sec_A2_mean
        row["margin_sec_median_A2"] = m_sec_A2_med
        row["margin_sec_count_A2"] = m_sec_A2_cnt
        row["margin_ratio_A1_mean"] = rA1_mean
        row["margin_ratio_A1_median"] = rA1_med
        row["margin_ratio_A1_count"] = rA1_cnt
        row["margin_ratio_A2_mean"] = rA2_mean
        row["margin_ratio_A2_median"] = rA2_med
        row["margin_ratio_A2_count"] = rA2_cnt
        results.append(row)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)
def plot_margin_boxplots(
    pf: pd.DataFrame,
    out_dir: Path,
    group_col: str = "A2_duration_bin",
    use_priority_only: bool = False,
) -> None:
    df = pf.copy()
    df = df[df["label"] == 1].copy()
    if df.empty:
        print("[INFO] margin boxplot: label==1 のファイルがありません。")
        return
    if use_priority_only and "is_priority_case" in df.columns:
        df = df[df["is_priority_case"].fillna(False)].copy()
    if df.empty:
        print("[INFO] margin boxplot: 対象データがありません。")
        return
    df["pre_collision_detected"] = df.get("pre_collision_detected", pd.Series(False, index=df.index)).fillna(False)
    df = df[df["pre_collision_detected"]].copy()
    if df.empty:
        print("[INFO] margin boxplot: pre_collision_detected=True のデータがありません。")
        return
    for col in ["first_detection_time", "collision_time", "A1_start"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["margin_sec"] = df["collision_time"] - df["first_detection_time"]
    df["A1A2_len"] = df["collision_time"] - df["A1_start"]
    df["margin_ratio_A1A2"] = df["margin_sec"] / df["A1A2_len"]
    df.loc[df["A1A2_len"] <= 0, "margin_ratio_A1A2"] = np.nan
    if group_col not in df.columns:
        print(f"[WARN] group_col={group_col} が存在しないため、全体1グループとして描画します。")
        df[group_col] = "all"
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=group_col, y="margin_sec", ax=ax1)
    ax1.set_xlabel(group_col)
    ax1.set_ylabel("余裕時間 [s]")
    ax1.set_title("余裕時間（秒）の分布")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    fig1.tight_layout()
    out_path1 = out_dir / "margin_seconds_boxplot.png"
    fig1.savefig(out_path1, dpi=150)
    plt.close(fig1)
    print(f"[INFO] 余裕時間（秒）の箱ひげ図を保存しました: {out_path1}")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=group_col, y="margin_ratio_A1A2", ax=ax2)
    ax2.set_xlabel(group_col)
    ax2.set_ylabel("余裕時間割合 (A1+A2基準)")
    ax2.set_title("余裕時間割合の分布")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    fig2.tight_layout()
    out_path2 = out_dir / "margin_ratio_A1A2_boxplot.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"[INFO] 余裕時間割合の箱ひげ図を保存しました: {out_path2}")
def plot_macro_heatmap(
    seg_macro_df: pd.DataFrame,
    out_dir: Path,
    value_col: str = "R_phase_A2",
) -> None:
    """
    A2_duration_bin x accel_A2_bin のマクロヒートマップを描画して保存する。
    seg_macro_df は aggregate_by_segments() の出力を想定。
    """
    if seg_macro_df is None or seg_macro_df.empty:
        print("[INFO] macro heatmap: 入力 DataFrame が空です。")
        return
    df = seg_macro_df.copy()
    if "A2_duration_bin" not in df.columns or "accel_A2_bin" not in df.columns:
        print("[WARN] macro heatmap: 必要な列 (A2_duration_bin, accel_A2_bin) がありません。")
        return
    try:
        pivot = df.pivot_table(index="A2_duration_bin", columns="accel_A2_bin", values=value_col, aggfunc="mean")
    except Exception as e:
        print(f"[WARN] macro heatmap: pivot_table に失敗しました: {e}")
        return
    if pivot.empty:
        print("[INFO] macro heatmap: pivot が空です。")
        return
    fig, ax = plt.subplots(figsize=(6, max(3, pivot.shape[0] * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title(f"Macro Heatmap: {value_col} (A2_duration x Accel)")
    ax.set_xlabel("accel_A2_bin")
    ax.set_ylabel("A2_duration_bin")
    fig.tight_layout()
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"macro_heatmap_{value_col}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] マクロヒートマップ ({value_col}) を保存しました: {out_path}")
# ---------------------------
# ROC: 評価対象を first_detect_phase == 'A2' のみ、x 軸を対数目盛に固定
# ---------------------------
def plot_roc_curve(
    pf: pd.DataFrame,
    out_dir: Path,
    score_col: str = "anomaly_rate",
) -> None:
    """
    ROC を描画（評価対象: first_detect_phase == 'A2' のファイルのみ）。
    FPR 軸は対数、目盛りは 1e-4,1e-3,1e-2,1e-1,1e0 に固定。
    """
    df = pf.copy()
    if "first_detect_phase" not in df.columns:
        print("[WARN] first_detect_phase 列が無いため ROC を描けません。")
        return
    df_eval = df[df["first_detect_phase"].fillna("OTHER").astype(str) == "A2"].copy()
    df_eval = df_eval[df_eval["label"].isin([0, 1])].copy()
    if df_eval.empty:
        print("[INFO] ROC(A2-only): 評価対象データ（first_detect_phase==A2）がありません。")
        return
    if score_col not in df_eval.columns:
        print(f"[WARN] ROC: score_col={score_col} がありません。スキップします。")
        return
    df_eval[score_col] = pd.to_numeric(df_eval[score_col], errors="coerce").fillna(0.0)
    y_true = (df_eval["label"] == 1).astype(int).to_numpy()
    scores = df_eval[score_col].to_numpy(dtype=float)
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        print("[INFO] ROC(A2-only): 正例または負例が存在しないため ROC を描けません。")
        return
    thresholds = np.unique(scores)[::-1]
    tprs = []
    fprs = []
    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tprs.append(tp / P if P > 0 else np.nan)
        fprs.append(fp / N if N > 0 else np.nan)
    fprs = np.array([0.0] + fprs.tolist() + [1.0], dtype=float)
    tprs = np.array([0.0] + tprs.tolist() + [1.0], dtype=float)
    df_diff = fprs[1:] - fprs[:-1]
    auc = float(np.nansum(df_diff * (tprs[1:] + tprs[:-1]) / 2.0))
    ticks = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0], dtype=float)
    min_tick = float(ticks[0])
    max_tick = float(ticks[-1])
    fprs_plot = fprs.copy()
    positive_mask = fprs_plot > 0
    if positive_mask.any():
        smallest_pos = float(fprs_plot[positive_mask].min())
        replace_val = min(smallest_pos / 10.0, min_tick / 10.0)
    else:
        replace_val = min_tick / 10.0
    fprs_plot[fprs_plot == 0.0] = replace_val
    x_min = min_tick / 10.0
    x_max = max_tick
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fprs_plot, tprs, marker="o", label=f"AUC = {auc:.3f}")
    ax.plot([x_min, x_max], [0, 1], linestyle="--", color="gray", label="reference")
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1e-4", "1e-3", "1e-2", "1e-1", "1e0"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR (Recall)")
    ax.set_title(f"ROC curve ({score_col}) - eval=A2-first only")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"roc_{score_col}_A2only.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] ROC(A2-only) 曲線を保存しました: {out_path}")
# ---------------------------
# マクロヒートマップ描画（追加）
# ---------------------------
def plot_macro_heatmap(
    seg_macro_df: pd.DataFrame,
    out_dir: Path,
    value_col: str = "R_phase_A2",
) -> None:
    if seg_macro_df is None or seg_macro_df.empty:
        print("[INFO] macro heatmap: 入力 DataFrame が空です。")
        return
    df = seg_macro_df.copy()
    if "A2_duration_bin" not in df.columns or "accel_A2_bin" not in df.columns:
        print("[WARN] macro heatmap: 必要な列 (A2_duration_bin, accel_A2_bin) がありません。")
        return
    try:
        pivot = df.pivot_table(index="A2_duration_bin", columns="accel_A2_bin", values=value_col, aggfunc="mean")
    except Exception as e:
        print(f"[WARN] macro heatmap: pivot_table に失敗しました: {e}")
        return
    if pivot.empty:
        print("[INFO] macro heatmap: pivot が空です。")
        return
    fig, ax = plt.subplots(figsize=(6, max(3, pivot.shape[0] * 0.6)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title(f"Macro Heatmap: {value_col} (A2_duration x Accel)")
    ax.set_xlabel("accel_A2_bin")
    ax.set_ylabel("A2_duration_bin")
    fig.tight_layout()
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"macro_heatmap_{value_col}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] マクロヒートマップ ({value_col}) を保存しました: {out_path}")
# ---------------------------
# 追加: FPR->TPR（ON/A2 と OFF を分離して処理）
# ---------------------------
def compute_threshold_curve_onoff_a2(
    per_file_on: pd.DataFrame,
    per_file_off: pd.DataFrame,
    score_col: str = "anomaly_rate",
    n_thresholds_max: int = 1000,
) -> pd.DataFrame:
    if score_col not in per_file_on.columns:
        raise ValueError(f"score_col '{score_col}' not in per_file_on")
    if score_col not in per_file_off.columns:
        raise ValueError(f"score_col '{score_col}' not in per_file_off")
    if "first_detect_phase" in per_file_on.columns:
        on_a2_mask = (per_file_on["first_detect_phase"].fillna("OTHER").astype(str) == "A2").to_numpy(dtype=bool)
    else:
        on_a2_mask = np.zeros(len(per_file_on), dtype=bool)
    off_mask = np.ones(len(per_file_off), dtype=bool)
    scores_on = pd.to_numeric(per_file_on[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scores_off = pd.to_numeric(per_file_off[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    uniq = np.unique(np.concatenate((scores_on, scores_off)))
    if uniq.size > n_thresholds_max:
        idx = np.linspace(0, uniq.size - 1, n_thresholds_max, dtype=int)
        thresholds = uniq[::-1][idx]
    else:
        thresholds = uniq[::-1]
    rows: List[Dict[str, Any]] = []
    P_on_A2 = int(on_a2_mask.sum())
    N_off = int(off_mask.sum())
    for th in thresholds:
        y_on = (scores_on >= float(th)).astype(int)
        y_off = (scores_off >= float(th)).astype(int)
        tp = int(((y_on == 1) & on_a2_mask).sum())
        fn = int(((y_on == 0) & on_a2_mask).sum())
        fp = int(((y_off == 1) & off_mask).sum())
        tn = int(((y_off == 0) & off_mask).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        rows.append({
            "threshold": float(th),
            "TP": tp, "FN": fn, "P_on_A2": P_on_A2,
            "FP": fp, "TN": tn, "N_off": N_off,
            "TPR": tpr, "FPR": fpr
        })
    res_df = pd.DataFrame(rows).sort_values(by="threshold", ascending=False).reset_index(drop=True)
    return res_df
def plot_roc_onoff_a2(thr_df: pd.DataFrame, out_path: Path, fixed_ticks: bool = True) -> None:
    mask = ~(np.isnan(thr_df["FPR"].to_numpy(dtype=float)) | np.isnan(thr_df["TPR"].to_numpy(dtype=float)))
    if mask.sum() == 0:
        print("[WARN] ROC(on/off A2): 有効な点がありません。スキップします。")
        return
    fpr = thr_df["FPR"].to_numpy(dtype=float)[mask]
    tpr = thr_df["TPR"].to_numpy(dtype=float)[mask]
    fpr_all = np.concatenate(([0.0], fpr, [1.0]))
    tpr_all = np.concatenate(([0.0], tpr, [1.0]))
    fig, ax = plt.subplots(figsize=(6,5))
    fpr_plot = fpr_all.copy()
    if fixed_ticks:
        ticks = np.array([1e-4,1e-3,1e-2,1e-1,1e0], dtype=float)
        min_tick = float(ticks[0])
        pos = fpr_plot[fpr_plot > 0]
        if pos.size > 0:
            smallest_pos = pos.min()
            replace_val = min(smallest_pos / 10.0, min_tick / 10.0)
        else:
            replace_val = min_tick / 10.0
        fpr_plot[fpr_plot == 0.0] = replace_val
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(["1e-4","1e-3","1e-2","1e-1","1e0"])
        ax.set_xlim(min_tick/10.0, ticks[-1])
    ax.plot(fpr_plot, tpr_all, marker="o")
    ax.set_xlabel("FPR (OFF files)")
    ax.set_ylabel("TPR (ON files whose first_detect_phase=='A2')")
    ax.set_title("ROC (ON A2 TPR vs OFF FPR, file-level)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] ROC (ON A2 vs OFF) を保存しました: {out_path}")
def tpr_at_fpr_targets_onoff(
    per_file_on: pd.DataFrame,
    per_file_off: pd.DataFrame,
    score_col: str,
    fpr_targets: Sequence[float],
) -> Dict[float, float]:
    thr_df = compute_threshold_curve_onoff_a2(per_file_on, per_file_off, score_col=score_col)
    valid = thr_df[~(thr_df["FPR"].isna() | thr_df["TPR"].isna())].copy()
    if valid.empty:
        return {float(ft): np.nan for ft in fpr_targets}
    fpr = valid["FPR"].to_numpy(dtype=float)
    tpr = valid["TPR"].to_numpy(dtype=float)
    fpr_all = np.concatenate(([0.0], fpr, [1.0]))
    tpr_all = np.concatenate(([0.0], tpr, [1.0]))
    order = np.argsort(fpr_all)
    fpr_s = fpr_all[order]
    tpr_s = tpr_all[order]
    results: Dict[float, float] = {}
    for ft in fpr_targets:
        ft = float(ft)
        if ft <= fpr_s[0]:
            tpr_val = float(tpr_s[0])
        elif ft >= fpr_s[-1]:
            tpr_val = float(tpr_s[-1])
        else:
            tpr_val = float(np.interp(ft, fpr_s, tpr_s))
        results[ft] = tpr_val
    return results
def plot_tpr_vs_fpr_targets(results: Dict[float, float], out_path: Path, title: str = "TPR at target FPRs") -> None:
    fprs = np.array(sorted(results.keys()), dtype=float)
    tprs = np.array([results[f] for f in fprs], dtype=float)
    positive_mask = fprs > 0.0
    fig, ax = plt.subplots(figsize=(6, 4))
    if not positive_mask.any():
        ax.plot(fprs, tprs, marker="o", linestyle="-")
        ax.set_xlabel("FPR")
    else:
        fprs_plot = fprs.copy()
        min_pos = fprs[positive_mask].min()
        fprs_plot[fprs_plot == 0.0] = max(min_pos / 10.0, 1e-12)
        ax.plot(fprs_plot, tprs, marker="o", linestyle="-")
        ax.set_xscale("log")
        ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["1e-4", "1e-3", "1e-2", "1e-1", "1e0"])
        ax.set_xlabel("FPR (log scale)")
    ax.set_ylabel("TPR (ON rate)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    for x, y in zip(fprs, tprs):
        xp = x if x > 0 else ax.get_xlim()[0]
        ax.annotate(f"{y:.3f}", xy=(xp, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
# ---------------------------
# メイン
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "ON (事故) と OFF (正常) の *_anomaly.csv から per-file サマリを生成し、"
            "結合して混同行列・ROC・区間別集計・グラフを出力するスクリプト"
        )
    )
    ap.add_argument("--config_on", required=True, help="ON (事故) 用 設定ファイル (JSON) のパス")
    ap.add_argument("--config_off", required=True, help="OFF (正常) 用 設定ファイル (JSON) のパス")
    ap.add_argument("--on_dir", required=True, help="ON (事故) 用 *_anomaly.csv が入っているフォルダ")
    ap.add_argument("--off_dir", required=True, help="OFF (正常) 用 *_anomaly.csv が入っているフォルダ")
    ap.add_argument("--out_dir", default=None, help="結果をまとめて出力するフォルダ（省略時は ON フォルダ配下に 'evaluation_onoff' を作成）")
    ap.add_argument("--fpr_targets", default="1e-4,1e-3,1e-2,1e-1,1e0", help="カンマ区切りの FPR 目標（例: 1e-4,1e-3,1e-2）")
    ap.add_argument("--score_col", default="anomaly_rate", help="スコア列名（デフォルト: anomaly_rate）")
    ap.add_argument("--verbose", action="store_true", help="OFF 台帳照合の詳細ログを有効にする")
    args = ap.parse_args()
    cfg_on_path = Path(args.config_on)
    cfg_off_path = Path(args.config_off)
    if not cfg_on_path.exists():
        raise FileNotFoundError(cfg_on_path)
    if not cfg_off_path.exists():
        raise FileNotFoundError(cfg_off_path)
    cfg_on = json.loads(cfg_on_path.read_text(encoding="utf-8"))
    cfg_off = json.loads(cfg_off_path.read_text(encoding="utf-8"))
    if "evaluation" not in cfg_on:
        raise ValueError(f"{cfg_on_path}: 'evaluation' セクションがありません。")
    if "evaluation" not in cfg_off:
        raise ValueError(f"{cfg_off_path}: 'evaluation' セクションがありません。")
    eval_on = cfg_on["evaluation"]
    eval_off = cfg_off["evaluation"]
    if "label_review_sheet" not in eval_on:
        raise ValueError(f"{cfg_on_path}: evaluation.label_review_sheet がありません (ON 用)。")
    label_cfg = eval_on["label_review_sheet"]
    accel_col_on = eval_on.get("accel_column_name", "accelpedalangle")
    if "normal_ledger_sheet" not in eval_off:
        raise ValueError(f"{cfg_off_path}: evaluation.normal_ledger_sheet がありません (OFF 用)。")
    ledger_cfg = eval_off["normal_ledger_sheet"]
    accel_col_off = eval_off.get("accel_column_name", "accelpedalangle")
    on_dir = Path(args.on_dir).resolve()
    off_dir = Path(args.off_dir).resolve()
    if not on_dir.exists():
        raise FileNotFoundError(f"ON ディレクトリが存在しません: {on_dir}")
    if not off_dir.exists():
        raise FileNotFoundError(f"OFF ディレクトリが存在しません: {off_dir}")
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = on_dir / "evaluation_onoff"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] ON (事故) データを処理します: {on_dir}")
    label_df = load_label_intervals(label_cfg)
    pf_on = build_per_file_summary_from_dir(on_dir, label_df, accel_col_name=accel_col_on)
    per_file_on_path = out_dir / "per_file_summary_on.csv"
    pf_on.to_csv(per_file_on_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] ON per_file_summary を保存しました: {per_file_on_path}")
    print(f"[INFO] OFF (正常) データを処理します: {off_dir}")
    normal_basenames = load_normal_basenames_from_ledger(ledger_cfg)
    pf_off = build_per_file_summary_normal_from_dir(off_dir, normal_basenames, accel_col_name=accel_col_off, verbose=args.verbose)
    per_file_off_path = out_dir / "per_file_summary_off.csv"
    pf_off.to_csv(per_file_off_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] OFF per_file_summary を保存しました: {per_file_off_path}")
    pf_all = pd.concat([pf_on, pf_off], ignore_index=True)
    per_file_all_path = out_dir / "per_file_summary_all.csv"
    pf_all.to_csv(per_file_all_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] ON+OFF per_file_summary を保存しました: {per_file_all_path}")
    pf_all = add_segment_columns(pf_all)
    pf_all = ensure_anomaly_rate_column(pf_all)
    cm = compute_confusion_matrices(pf_all)
    cm_path = out_dir / "confusion_matrices.txt"
    with cm_path.open("w", encoding="utf-8") as f:
        for key in ["A2_to_A3start", "A2_to_A3end", "lenient", "strict"]:
            if key not in cm:
                continue
            f.write(f"=== Confusion Matrix ({key}) ===\n")
            for k, v in cm[key].items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
    print(f"[INFO] 混同行列を保存しました: {cm_path}")
    margin_stats = compute_margin_stats(pf_all)
    if margin_stats:
        ms_path = out_dir / "margin_stats.txt"
        with ms_path.open("w", encoding="utf-8") as f:
            f.write("=== Margin Stats (pre_collision_detected=True, label=1) ===\n")
            f.write(f"margin_seconds_all: {margin_stats['margin_seconds_all']}\n")
            f.write(f"ratio_A1A2_all: {margin_stats['ratio_A1A2_all']}\n")
            f.write(f"ratio_A1_phase: {margin_stats['ratio_A1_phase']}\n")
            f.write(f"ratio_A2_phase: {margin_stats['ratio_A2_phase']}\n")
        print(f"[INFO] 余裕時間統計を保存しました: {ms_path}")
    else:
        print("[INFO] 余裕時間統計を計算できませんでした（対象データなし）")
    seg_tbl_A2 = aggregate_by_segments(pf_all, group_cols=["A2_duration_bin"], use_priority_only=False)
    out_seg_A2 = out_dir / "per_file_seg_A2_duration.csv"
    seg_tbl_A2.to_csv(out_seg_A2, index=False, encoding="utf-8-sig")
    print(f"[INFO] 区間別集計(A2_duration_bin) を保存しました: {out_seg_A2}")
    seg_tbl_macro = aggregate_by_segments(pf_all, group_cols=["A2_duration_bin", "accel_A2_bin", "shift_at_collision"], use_priority_only=False)
    out_seg_macro = out_dir / "per_file_seg_macro.csv"
    seg_tbl_macro.to_csv(out_seg_macro, index=False, encoding="utf-8-sig")
    print(f"[INFO] 区間別集計(A2_duration_bin×accel_A2_bin×shift) を保存しました: {out_seg_macro}")
    seg_tbl_macro_key = aggregate_by_segments(pf_all, group_cols=["A2_duration_bin", "accel_A2_bin", "shift_at_collision"], use_priority_only=True)
    out_seg_macro_key = out_dir / "per_file_seg_macro_keycases.csv"
    seg_tbl_macro_key.to_csv(out_seg_macro_key, index=False, encoding="utf-8-sig")
    print(f"[INFO] 区間別集計(重点事故) を保存しました: {out_seg_macro_key}")
    fig_dir = out_dir / "figs"
    os.makedirs(fig_dir, exist_ok=True)
    thr_df = compute_threshold_curve_onoff_a2(pf_on, pf_off, score_col=args.score_col)
    thr_csv = fig_dir / "threshold_metrics_onoff_A2only.csv"
    thr_df.to_csv(thr_csv, index=False, encoding="utf-8-sig", float_format="%.6g")
    print(f"[INFO] 閾値テーブルを保存しました: {thr_csv}")
    roc_png = fig_dir / f"roc_{args.score_col}_onoff_A2only.png"
    plot_roc_onoff_a2(thr_df, roc_png, fixed_ticks=True)
    plot_first_phase_distribution(pf_on, out_dir=fig_dir, group_col="A2_duration_bin", use_priority_only=False)
    plot_margin_boxplots(pf_all, out_dir=fig_dir, group_col="A2_duration_bin", use_priority_only=False)
    plot_macro_heatmap(seg_macro_df=seg_tbl_macro, out_dir=fig_dir, value_col="R_phase_A2")
    # ---------------------------
    # 指定 FPR に対する TPR 算出＆保存（評価対象: 初検知が A2 の ON ファイル、負例は OFF 全体）
    # ---------------------------
    fpr_targets = [float(x) for x in args.fpr_targets.split(",")]
    if args.score_col not in pf_on.columns or args.score_col not in pf_off.columns:
        print(f"[WARN] score_col={args.score_col} が ON/OFF の per-file に存在しないため FPR->TPR 計算をスキップします。")
    else:
        res = tpr_at_fpr_targets_onoff(pf_on, pf_off, score_col=args.score_col, fpr_targets=fpr_targets)
        res_df = pd.DataFrame({"fpr_target": list(res.keys()), "tpr_est": [res[k] for k in res.keys()]})
        res_csv = fig_dir / "tpr_at_fpr_table_A2only.csv"
        res_df.to_csv(res_csv, index=False, encoding="utf-8-sig", float_format="%.6g")
        print(f"[INFO] FPR->TPR テーブルを保存しました: {res_csv}")
        plot_png = fig_dir / "tpr_at_fpr_A2only.png"
        plot_tpr_vs_fpr_targets(res, plot_png, title=f"TPR at target FPRs (score={args.score_col}, eval=A2-first only)")
        print(f"[INFO] FPR->TPR プロットを保存しました: {plot_png}")
        print("FPR_target\tTPR_est")
        for f, t in sorted(res.items()):
            if np.isnan(t):
                print(f"{f:.0e}\tNaN")
            else:
                print(f"{f:.0e}\t{t:.6f}")
        P_on_A2 = int(((pf_on["first_detect_phase"].fillna("OTHER").astype(str) == "A2")).sum()) if "first_detect_phase" in pf_on.columns else 0
        N_off = len(pf_off)
        print(f"[INFO] Positive (ON with first_detect_phase=='A2'): {P_on_A2}, Negative (OFF files used for FPR): {N_off}")
        print("[NOTE] Minimum non-zero achievable FPR = 1/N_off. Very small FPR targets below that are not strictly realizable (results are interpolated).")
    print(f"[INFO] すべての処理が完了しました。出力先: {out_dir}")
if __name__ == "__main__":
    main()