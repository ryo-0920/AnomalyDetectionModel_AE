#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 日本語フォント設定（Windows の例）
jp_font = "MS Gothic"  # "MS Mincho", "Yu Gothic" などでも可
rcParams["font.family"] = jp_font
rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止
import numpy as np
import pandas as pd
import seaborn as sns
# ============================
# 共通ユーティリティ
# ============================
def normalize_basename(name: str) -> str:
    """
    ファイル名から共通の basename を作る。
    例:
        "A000021.csv"           -> "A000021"
        "A000021_anomaly.csv"   -> "A000021"
        "C:\\path\\A000021.csv" -> "A000021"
    """
    p = Path(name)
    stem = p.stem  # 例: "A000021" or "A000021_anomaly"
    # 推論結果ファイル側の "_anomaly" サフィックスを落とす
    if stem.endswith("_anomaly"):
        stem = stem[:-len("_anomaly")]
    return stem
def safe_float(v) -> Optional[float]:
    """Excel から読んだ値などを float に変換（失敗時は None）。"""
    try:
        return float(v) if not pd.isna(v) else None
    except Exception:
        return None
def to_bool(series: Optional[pd.Series],
            index: Optional[pd.Index] = None,
            default: bool = False) -> pd.Series:
    """None や NaN を含む Series を安全に bool に変換するヘルパ。"""
    if series is None:
        if index is None:
            raise ValueError("index is required when series is None")
        return pd.Series(default, index=index)
    return series.fillna(default).astype(bool)
def safe_ratio(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """ゼロ・負の分母を NaN にした上で安全に割り算するヘルパ."""
    denom = np.where(denom > 0, denom, np.nan)
    return num / denom
def basic_stats(x: np.ndarray) -> Dict[str, Any]:
    """平均・中央値・件数を返すヘルパ."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"mean": np.nan, "median": np.nan, "count": 0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "count": int(x.size),
    }
# ---------------------------
# ラベルシートの読み込み（ON用）
# ---------------------------
def load_label_intervals(label_cfg: dict) -> pd.DataFrame:
    """
    ラベルシート (Excel) から A1_start, A2_start, A3_start, A3_end を読み込んで
    basename ごとの DataFrame を返す。
    必須キー:
        path, sheet_name, file_column_excel_index,
        a1_override_col, k_col, m_col, n_col, a1_delta_seconds
    """
    path = label_cfg["path"]
    sheet_name = label_cfg.get("sheet_name", 0)
    file_col_idx = int(label_cfg["file_column_excel_index"]) - 1  # 1始まり -> 0始まり
    a1_col_idx = int(label_cfg["a1_override_col"]) - 1
    k_col_idx = int(label_cfg["k_col"]) - 1    # A2_start
    m_col_idx = int(label_cfg["m_col"]) - 1    # A3_start
    n_col_idx = int(label_cfg["n_col"]) - 1    # A3_end
    delta = float(label_cfg.get("a1_delta_seconds", 5.0))
    df = pd.read_excel(path, sheet_name=sheet_name, header=0)
    # 安全のため、列数チェック
    max_idx = max(file_col_idx, a1_col_idx, k_col_idx, m_col_idx, n_col_idx)
    if max_idx >= df.shape[1]:
        raise ValueError(
            f"label sheet の列数が足りません: 必要列インデックス={max_idx}, df.shape={df.shape}"
        )
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
        k = safe_float(k_raw)  # A2_start
        m = safe_float(m_raw)  # A3_start
        n = safe_float(n_raw)  # A3_end
        a1_over = safe_float(a1_over_raw)
        A1_start = None
        A2_start = None
        A3_start = None
        A3_end = None
        if k is not None:
            A2_start = k
            if a1_over is not None:
                # override があればそのまま使う（負でもOK）
                A1_start = a1_over
            else:
                # override が無ければ k - delta
                A1_start = k - delta
        if m is not None:
            A3_start = m
        if n is not None:
            A3_end = n
        if A1_start is None and A2_start is None and A3_start is None:
            # 情報が何もない行はスキップ
            continue
        rows.append(
            {
                "basename": basename,
                "A1_start": A1_start,
                "A2_start": A2_start,
                "A3_start": A3_start,
                "A3_end": A3_end,
            }
        )
    label_df = pd.DataFrame(rows)
    # basename ごとに最後の行を残す（重複があれば後勝ち）
    label_df = label_df.drop_duplicates(subset=["basename"], keep="last")
    return label_df
# ---------------------------
# 正常データ用：台帳 (K列) から basename リスト取得（OFF用）
# ---------------------------
def load_normal_basenames_from_ledger(ledger_cfg: dict) -> List[str]:
    """
    正常データ用の台帳Excelから、指定列(例: K列=時系列csvファイル名)にある
    ファイル名を読み取り、basename のリストを返す。
    ledger_cfg 必須キー:
        path, sheet_name, file_column_excel_index
    """
    path = ledger_cfg["path"]
    sheet_name = ledger_cfg.get("sheet_name", 0)
    file_col_idx = int(ledger_cfg["file_column_excel_index"]) - 1  # 1始まり -> 0始まり
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
# ---------------------------
# *_anomaly.csv → per_file_summary（ON：事故データ）
# ---------------------------
def build_per_file_summary_from_dir(
    result_dir: Path,
    label_df: pd.DataFrame,
    accel_col_name: str = "accelpedalangle",
) -> pd.DataFrame:
    """
    result_dir 内の *_anomaly.csv を走査し、1ファイル1行の per_file_summary を作る。
    前提:
      - 各ファイルに 'time' 列と 'is_anomaly' 列がある
      - ラベルシートからの label_df は
          basename, A1_start, A2_start, A3_start, A3_end
        を持つ
      - 今扱っているのは事故データ（label=1）とする
      - time は秒（ラベルシートの A1/A2/A3 も秒）で同一スケール
    """
    result_dir = Path(result_dir)
    rows: List[Dict[str, Any]] = []
    for csv_path in result_dir.glob("*_anomaly.csv"):
        df = pd.read_csv(csv_path)
        # 共通 basename (例: A000021_anomaly.csv -> A000021)
        basename = normalize_basename(csv_path.name)
        # 対応するラベル行を取得
        lbl = label_df[label_df["basename"] == basename]
        if lbl.empty:
            print(f"[WARN] ラベルシートに basename={basename} の行がありません。スキップします。")
            continue
        lbl_row = lbl.iloc[0]
        # A1/A2/A3 の時刻（秒）
        A1_start = safe_float(lbl_row["A1_start"])
        A2_start = safe_float(lbl_row["A2_start"])
        A3_start = safe_float(lbl_row["A3_start"])
        A3_end = safe_float(lbl_row["A3_end"])
        # time / is_anomaly 列
        if "time" not in df.columns or "is_anomaly" not in df.columns:
            print(f"[WARN] {csv_path.name}: 'time' または 'is_anomaly' 列がありません。スキップします。")
            continue
        time = pd.to_numeric(df["time"], errors="coerce")
        is_anom = df["is_anomaly"].fillna(0).astype(int)
        # アクセル列
        if accel_col_name in df.columns:
            accel = pd.to_numeric(df[accel_col_name], errors="coerce")
        else:
            accel = None
        # 検知フラグ
        has_anom = (is_anom == 1)
        detected = bool(has_anom.any())
        if detected:
            # 初検知時刻
            first_idx = has_anom.to_numpy().nonzero()[0][0]
            first_det_time = float(time.iloc[first_idx])
        else:
            first_det_time = np.nan
        # 衝突時刻（A3_start）を collision_time として使う
        collision_time = A3_start if (A3_start is not None and np.isfinite(A3_start)) else np.nan
        # 初検知フェーズ
        if detected and all(
            v is not None and np.isfinite(v) for v in (A1_start, A2_start, A3_start)
        ):
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
        # 衝突前検知フラグ
        pre_collision_detected = bool(
            detected
            and collision_time is not None
            and np.isfinite(collision_time)
            and (first_det_time < collision_time)
        )
        # A1/A2/A3 区間での検知
        a1_detected = False
        a2_coverage = 0.0
        a3_detected = False
        if (
            detected
            and A1_start is not None
            and A2_start is not None
            and np.isfinite(A1_start)
            and np.isfinite(A2_start)
        ):
            mask_A1 = (time >= A1_start) & (time < A2_start)
            a1_detected = bool((has_anom & mask_A1).any())
        if (
            detected
            and A2_start is not None
            and A3_start is not None
            and np.isfinite(A2_start)
            and np.isfinite(A3_start)
        ):
            mask_A2 = (time >= A2_start) & (time < A3_start)
            # カバレッジは「A2区間で is_anomaly==1 のフレーム数」とする
            a2_coverage = float((has_anom & mask_A2).sum())
        if (
            detected
            and A3_start is not None
            and A3_end is not None
            and np.isfinite(A3_start)
            and np.isfinite(A3_end)
        ):
            mask_A3 = (time >= A3_start) & (time < A3_end)
            a3_detected = bool((has_anom & mask_A3).any())
        # A2区間内のアクセル最大値
        if (
            accel is not None
            and A2_start is not None
            and A3_start is not None
            and np.isfinite(A2_start)
            and np.isfinite(A3_start)
        ):
            mask_A2_for_accel = (time >= A2_start) & (time < A3_start)
            if mask_A2_for_accel.any():
                accel_A2_max = float(accel[mask_A2_for_accel].max())
            else:
                accel_A2_max = np.nan
        else:
            accel_A2_max = np.nan
        def bin_accel(x: float) -> str:
            # NaN や負値は unknown
            if pd.isna(x) or x < 0:
                return "unknown"
            # 0〜100%, 70〜100%, 80〜100%, 90〜100% の4区分
            if x >= 90:
                return "90-100%"
            if x >= 80:
                return "80-100%"
            if x >= 70:
                return "70-100%"
            return "0-100%"
        accel_A2_bin = bin_accel(accel_A2_max)
        # 事故データなので label=1
        label = 1
        rows.append(
            {
                "basename": basename,
                "label": label,
                "detected": detected,
                "a1_detected": a1_detected,
                "a2_coverage": a2_coverage,
                "a3_detected": a3_detected,
                "first_detect_phase": first_phase,
                "first_detection_time": first_det_time,
                "pre_collision_detected": pre_collision_detected,
                "collision_time": collision_time,
                "A1_start": A1_start,
                "A2_start": A2_start,
                "A3_start": A3_start,
                "A3_end": A3_end,
                "accel_A2_max": accel_A2_max,
                "accel_A2_bin": accel_A2_bin,
            }
        )
    if not rows:
        raise ValueError(f"[ERROR] {result_dir} 内に有効な *_anomaly.csv がありません。")
    return pd.DataFrame(rows)
# ---------------------------
# *_anomaly.csv → per_file_summary（OFF：正常データ）
# ---------------------------
def build_per_file_summary_normal_from_dir(
    result_dir: Path,
    normal_basenames: List[str],
    accel_col_name: str = "accelpedalangle",
) -> pd.DataFrame:
    """
    正常データ用:
    - result_dir 内の *_anomaly.csv を走査
    - normal_basenames に含まれる basename だけを対象
    - label=0 として per_file_summary を1ファイル1行で作る
    A1/A2/A3 の情報は無い前提なので、関連する列は NaN / False / "OTHER" で埋める。
    """
    result_dir = Path(result_dir)
    name_set = set(normal_basenames)
    rows: List[Dict[str, Any]] = []
    for csv_path in result_dir.glob("*_anomaly.csv"):
        df = pd.read_csv(csv_path)
        basename = normalize_basename(csv_path.name)
        # 台帳に無いファイルはスキップ
        if basename not in name_set:
            continue
        if "time" not in df.columns or "is_anomaly" not in df.columns:
            print(f"[WARN] {csv_path.name}: 'time' または 'is_anomaly' 列がありません。スキップします。")
            continue
        time = pd.to_numeric(df["time"], errors="coerce")
        is_anom = df["is_anomaly"].fillna(0).astype(int)
        # アクセル列
        if accel_col_name in df.columns:
            accel = pd.to_numeric(df[accel_col_name], errors="coerce")
        else:
            accel = None
        has_anom = (is_anom == 1)
        detected = bool(has_anom.any())
        # 正常データなので label=0
        label = 0
        # A1/A2/A3 が無いので関連情報は NaN/False/OTHER
        rows.append(
            {
                "basename": basename,
                "label": label,
                "detected": detected,
                "a1_detected": False,
                "a2_coverage": 0.0,
                "a3_detected": False,
                "first_detect_phase": "OTHER",
                "first_detection_time": np.nan,
                "pre_collision_detected": False,
                "collision_time": np.nan,
                "A1_start": np.nan,
                "A2_start": np.nan,
                "A3_start": np.nan,
                "A3_end": np.nan,
                "accel_A2_max": np.nan,
                "accel_A2_bin": "unknown",
            }
        )
    if not rows:
        raise ValueError(f"[ERROR] {result_dir} 内に台帳に載っている *_anomaly.csv がありません。")
    return pd.DataFrame(rows)
# ---------------------------
# first_detect_phase 分布
# ---------------------------
def plot_first_phase_distribution(
    pf: pd.DataFrame,
    out_dir: Path,
    group_col: str = "A2_duration_bin",
    use_priority_only: bool = False,
) -> None:
    """
    2-1: first_detect_phase の分布を「正規分布っぽい」棒グラフ＋折れ線で描画する。
    - 異常ファイル (label==1) を対象
    - group_col (デフォルト: A2_duration_bin) ごとにサブプロット
    - 各グループ内で first_detect_phase (A1/A2/A3/OTHER) の
      割合を棒グラフで描き、その上を折れ線でつなぐ
    """
    df = pf.copy()
    df = df[df["label"] == 1].copy()  # 異常のみ
    if df.empty:
        print("[INFO] first_phase plot: label==1 のファイルがありません。")
        return
    if use_priority_only and "is_priority_case" in df.columns:
        df = df[df["is_priority_case"].fillna(False)].copy()
        if df.empty:
            print("[INFO] first_phase plot: 重点事故がありません。")
            return
    df["first_detect_phase"] = df.get(
        "first_detect_phase", pd.Series(index=df.index)
    ).fillna("OTHER").astype(str)
    # グループ列
    if group_col not in df.columns:
        print(f"[WARN] group_col={group_col} が存在しないため、全体で 1 グループとして描画します。")
        df[group_col] = "all"
    groups = [g for g in df[group_col].unique() if pd.notna(g)]
    if not groups:
        print("[INFO] first_phase plot: グループがありません。")
        return
    phases: Sequence[str] = ["A1", "A2", "A3", "OTHER"]
    # サブプロット行数を決定
    n_groups = len(groups)
    n_cols = 2
    n_rows = (n_groups + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
    )
    for idx, g in enumerate(sorted(groups)):
        ax = axes[idx // n_cols][idx % n_cols]
        gdf = df[df[group_col] == g]
        # first_detect_phase ごとの割合
        vc = gdf["first_detect_phase"].value_counts(normalize=True)
        ratios = [vc.get(p, 0.0) for p in phases]
        # 棒グラフ
        x = list(range(len(phases)))
        ax.bar(x, ratios, color="skyblue", alpha=0.7)
        # 折れ線（棒の頂点をつなぐ）
        ax.plot(x, ratios, color="blue", marker="o")
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("割合")
        ax.set_title(f"{group_col} = {g} での初検知フェーズ分布")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    # 余ったサブプロットを消す
    for i in range(n_groups, n_rows * n_cols):
        fig.delaxes(axes[i // n_cols][i % n_cols])
    fig.tight_layout()
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "first_phase_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] first_detect_phase 分布グラフを保存しました: {out_path}")
# ---------------------------
# 混同行列・指標計算
# ---------------------------
def compute_confusion_matrices(per_file_df: pd.DataFrame) -> dict:
    """
    per_file_summary DataFrame から、ファイル単位の混同行列
    (lenient / strict の2種類) を計算する。
    期待する列:
        label, a1_detected, a3_detected, a2_coverage, detected,
        first_detect_phase
    """
    df = per_file_df.copy()
    # label 0/1 に絞る
    df = df[df["label"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("label 0/1 のファイルがありません。")
    df["is_abnormal"] = df["label"] == 1
    has_det_A1 = to_bool(df.get("a1_detected"), df.index)
    has_det_A3 = to_bool(df.get("a3_detected"), df.index)
    # A2 は coverage > 0 で検知ありとみなす
    a2_cov = df.get("a2_coverage")
    has_det_A2 = (
        pd.Series(False, index=df.index)
        if a2_cov is None
        else a2_cov.fillna(0.0) > 0.0
    )
    has_det_label_window = has_det_A1 | has_det_A2 | has_det_A3
    detected_any = to_bool(df.get("detected"), df.index)
    # first_detect_phase
    first_phase = df.get("first_detect_phase", pd.Series(index=df.index))
    first_phase = first_phase.fillna("OTHER").astype(str)
    fd_in_A123 = first_phase.isin(["A1", "A2", "A3"])

   #初検知区間の切り分けを定義
    fd_A2_window   = detected_any & (first_phase == "A2")
    fd_A23_window  = detected_any & first_phase.isin(["A2", "A3"])

    res: Dict[str, Any] = {}

    # 共通の集計ロジック
    def cm_stats(y_pred: np.ndarray, name: str) -> None:
        is_pos = df["is_abnormal"]
        is_neg = ~df["is_abnormal"]

        tp = int(((y_pred == 1) & is_pos).sum())
        fn = int(((y_pred == 0) & is_pos).sum())
        fp = int(((y_pred == 1) & is_neg).sum())
        tn = int(((y_pred == 0) & is_neg).sum())

        total = tp + fp + fn + tn
        acc  = (tp + tn) / total if total > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        res[name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "total": total,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "FPR": fpr,
        }

    # 既存: lenient (A1/A2/A3のどこかで検知ならOK)
    y_pred_lenient = np.where(
        df["is_abnormal"], has_det_label_window, detected_any
    ).astype(int)
    cm_stats(y_pred_lenient, "lenient")

    # 既存: strict (初検知フェーズがA1/A2/A3のいずれか)
    y_pred_strict = np.where(
        df["is_abnormal"], fd_in_A123, detected_any
    ).astype(int)
    cm_stats(y_pred_strict, "strict")

    # ★追加1: A2start〜A3start で「初検知」できたか
    #   異常ファイル: first_phase == "A2" を陽性
    #   正常ファイル: どこかで検知されたら陽性 (detected_any)
    y_pred_A2_window = np.where(
        df["is_abnormal"], fd_A2_window, detected_any
    ).astype(int)
    cm_stats(y_pred_A2_window, "A2_to_A3start")

    # ★追加2: A2start〜A3end で「初検知」できたか
    #   異常ファイル: first_phase in {"A2","A3"} を陽性
    y_pred_A23_window = np.where(
        df["is_abnormal"], fd_A23_window, detected_any
    ).astype(int)
    cm_stats(y_pred_A23_window, "A2_to_A3end")

    # 既存と同様、予測ベクトルも返しておく（必要なら後続で利用）
    res["y_pred_lenient"]      = y_pred_lenient
    res["y_pred_strict"]       = y_pred_strict
    res["y_pred_A2_to_A3start"] = y_pred_A2_window
    res["y_pred_A2_to_A3end"]   = y_pred_A23_window
    res["first_phase"]         = first_phase
    res["has_det_label_window"] = has_det_label_window
    res["detected_any"]        = detected_any

    return res
# ---------------------------
# 余裕時間統計
# ---------------------------
def compute_margin_stats(per_file_df: pd.DataFrame) -> dict:
    """
    pre_collision_detected=True の異常ファイルについて、
    余裕時間 (秒) と、A1+A2 / A1 / A2 基準の割合を集計する。
    期待する列:
        label, pre_collision_detected, first_detection_time,
        collision_time (A3_start), A1_start, A2_start
    """
    df = per_file_df.copy()
    df = df[df["label"] == 1].copy()  # 異常のみ
    # pre_collision_detected==True のみ対象
    df = df[df["pre_collision_detected"].fillna(False)].copy()
    if df.empty:
        return {}
    # 必要列を float に
    for col in ["first_detection_time", "collision_time", "A1_start", "A2_start"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=["first_detection_time", "collision_time", "A1_start", "A2_start"]
    ).copy()
    if df.empty:
        return {}
    t_det = df["first_detection_time"].to_numpy(dtype=float)
    tA3 = df["collision_time"].to_numpy(dtype=float)
    tA1 = df["A1_start"].to_numpy(dtype=float)
    tA2 = df["A2_start"].to_numpy(dtype=float)
    dt = tA3 - t_det  # 余裕時間（秒）
    LA1A2 = tA3 - tA1
    LA1 = tA2 - tA1
    LA2 = tA3 - tA2
    r_A1A2 = safe_ratio(dt, LA1A2)
    # フェーズ別
    phase = df["first_detect_phase"].fillna("OTHER").astype(str).to_numpy()
    mask_A1 = phase == "A1"
    mask_A2 = phase == "A2"
    r_A1_phase = safe_ratio(dt[mask_A1], LA1[mask_A1])
    r_A2_phase = safe_ratio(dt[mask_A2], LA2[mask_A2])
    stats: Dict[str, Any] = {}
    stats["margin_seconds_all"] = basic_stats(dt)
    stats["ratio_A1A2_all"] = basic_stats(r_A1A2)
    stats["ratio_A1_phase"] = basic_stats(r_A1_phase)
    stats["ratio_A2_phase"] = basic_stats(r_A2_phase)
    return stats
# ---------------------------
# 区間ごとの集計
# ---------------------------
def add_segment_columns(pf: pd.DataFrame) -> pd.DataFrame:
    """
    per_file_summary に A2_duration_bin, accel_A2_bin, shift_at_collision 列を付与する。
    - A2_duration_bin: A3_start - A2_start の長さをビン分け
    - accel_A2_bin   : 既に per_file にあればそのまま、無ければ 'unknown'
    - shift_at_collision: （現状は情報が無いので NaN 固定）
    """
    df = pf.copy()
    # --- A2_duration_bin ---
    if "A2_duration_bin" not in df.columns:
        if "A2_start" in df.columns and "A3_start" in df.columns:
            dur = pd.to_numeric(df["A3_start"], errors="coerce") - pd.to_numeric(
                df["A2_start"], errors="coerce"
            )
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
    # --- accel_A2_bin ---
    if "accel_A2_bin" not in df.columns:
        df["accel_A2_bin"] = "unknown"
    # --- shift_at_collision ---
    if "shift_at_collision" not in df.columns:
        df["shift_at_collision"] = np.nan
    return df
def aggregate_by_segments(
    pf: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    use_priority_only: bool = False,
) -> pd.DataFrame:
    """
    区間ごと（例: A2_duration_bin, accel_A2_bin, shift_at_collision）に、
    1-3 / 2 で決めた項目（初検知フェーズ・全体検知数・余裕時間）を集計して表にする。
    """
    df = pf.copy()
    # グルーピング列
    if group_cols is None:
        group_cols = ["A2_duration_bin"]
    # 異常ファイルだけ対象
    df = df[df["label"] == 1].copy()
    if df.empty:
        print("[INFO] label==1 のファイルがありません。")
        return pd.DataFrame()
    # 重点事故だけに絞るオプション
    if use_priority_only and "is_priority_case" in df.columns:
        df = df[df["is_priority_case"].fillna(False)].copy()
        if df.empty:
            print("[INFO] 重点事故 (is_priority_case=True) がありません。")
            return pd.DataFrame()
    # first_detect_phase
    df["first_detect_phase"] = df.get(
        "first_detect_phase", pd.Series(index=df.index)
    ).fillna("OTHER").astype(str)
    # A1/A2/A3区間内で検知したか（lenient の TP に対応）
    has_det_A1 = to_bool(df.get("a1_detected"), df.index)
    has_det_A3 = to_bool(df.get("a3_detected"), df.index)
    a2_cov = df.get("a2_coverage", pd.Series(0.0, index=df.index))
    has_det_A2 = a2_cov.fillna(0.0) > 0.0
    df["has_det_label_window"] = has_det_A1 | has_det_A2 | has_det_A3
    # 余裕時間用 (pre_collision_detected / t_det / tA3 / tA1)
    df["pre_collision_detected"] = df.get(
        "pre_collision_detected", pd.Series(False, index=df.index)
    ).fillna(False)
    for col in ["first_detection_time", "collision_time", "A1_start"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    # 衝突前に検知できたものだけの余裕時間
    df["margin_sec"] = df["collision_time"] - df["first_detection_time"]
    df["A1A2_len"] = df["collision_time"] - df["A1_start"]
    df["margin_ratio_A1A2"] = df["margin_sec"] / df["A1A2_len"]
    # ゼロ・負の分母は NaN
    df.loc[df["A1A2_len"] <= 0, "margin_ratio_A1A2"] = np.nan
    results: List[Dict[str, Any]] = []
    grouped = df.groupby(list(group_cols), dropna=False)
    for g_keys, g in grouped:
        # g_keys はタプル or 単一値
        if not isinstance(g_keys, tuple):
            g_keys = (g_keys,)
        # ベース情報
        N_abn = int(g.shape[0])
        N_det = int(g["has_det_label_window"].sum())
        # first_detect_phase 件数
        vc_phase = g["first_detect_phase"].value_counts()
        N_A1 = int(vc_phase.get("A1", 0))
        N_A2 = int(vc_phase.get("A2", 0))
        N_A3 = int(vc_phase.get("A3", 0))
        N_OTHER = int(vc_phase.get("OTHER", 0))
        # 割合
        def ratio(n: int) -> float:
            return n / N_abn if N_abn > 0 else np.nan
        R_A1 = ratio(N_A1)
        R_A2 = ratio(N_A2)
        R_A3 = ratio(N_A3)
        R_OTHER = ratio(N_OTHER)
        # 余裕時間（pre_collision_detected=True のみ）
        g_margin = g[g["pre_collision_detected"]].copy()
        if not g_margin.empty:
            margin_sec = g_margin["margin_sec"].to_numpy(dtype=float)
            margin_sec = margin_sec[~np.isnan(margin_sec)]
            ratio_A1A2 = g_margin["margin_ratio_A1A2"].to_numpy(dtype=float)
            ratio_A1A2 = ratio_A1A2[~np.isnan(ratio_A1A2)]
            def stat(x: np.ndarray) -> tuple[float, float, int]:
                if x.size == 0:
                    return (np.nan, np.nan, 0)
                return (float(np.mean(x)), float(np.median(x)), int(x.size))
            m_sec_mean, m_sec_med, m_sec_cnt = stat(margin_sec)
            rA1A2_mean, rA1A2_med, rA1A2_cnt = stat(ratio_A1A2)
        else:
            m_sec_mean = m_sec_med = np.nan
            m_sec_cnt = 0
            rA1A2_mean = rA1A2_med = np.nan
            rA1A2_cnt = 0
        row: Dict[str, Any] = {}
        # グループキー列
        for col_name, key_val in zip(group_cols, g_keys):
            row[col_name] = key_val
        # 件数系
        row["N_abnormal"] = N_abn
        row["N_detected_label_window"] = N_det
        row["N_phase_A1"] = N_A1
        row["N_phase_A2"] = N_A2
        row["N_phase_A3"] = N_A3
        row["N_phase_OTHER"] = N_OTHER
        # 割合系（first_detect_phase）
        row["R_phase_A1"] = R_A1
        row["R_phase_A2"] = R_A2
        row["R_phase_A3"] = R_A3
        row["R_phase_OTHER"] = R_OTHER
        # 余裕時間（秒）
        row["margin_sec_mean"] = m_sec_mean
        row["margin_sec_median"] = m_sec_med
        row["margin_sec_count"] = m_sec_cnt
        # 余裕時間割合（A1+A2基準）
        row["margin_ratio_A1A2_mean"] = rA1A2_mean
        row["margin_ratio_A1A2_median"] = rA1A2_med
        row["margin_ratio_A1A2_count"] = rA1A2_cnt
        results.append(row)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)
# ---------------------------
# 余裕時間の箱ひげ図
# ---------------------------
def plot_margin_boxplots(
    pf: pd.DataFrame,
    out_dir: Path,
    group_col: str = "A2_duration_bin",
    use_priority_only: bool = False,
) -> None:
    """
    2-2: 余裕時間（秒）および A1+A2 区間長基準の割合の箱ひげ図を描画する。
    - 対象: label==1 & pre_collision_detected==True
    - group_col (デフォルト: A2_duration_bin) ごとに比較
    """
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
    df["pre_collision_detected"] = df.get(
        "pre_collision_detected", pd.Series(False, index=df.index)
    ).fillna(False)
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
    # 余裕時間（秒）の箱ひげ図
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
    # 余裕時間割合（A1+A2基準）の箱ひげ図
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
# ---------------------------
# マクロヒートマップ
# ---------------------------
def plot_macro_heatmap(
    seg_macro_df: pd.DataFrame,
    out_dir: Path,
    value_col: str = "R_phase_A2",
) -> None:
    """
    2-3: マクロヒートマップ
    - 行: A2_duration_bin
    - 列: accel_A2_bin
    - 色: 指定した value_col (例: R_phase_A2)
    """
    if seg_macro_df.empty:
        print("[INFO] macro heatmap: 入力 DataFrame が空です。")
        return
    df = seg_macro_df.copy()
    if "A2_duration_bin" not in df.columns or "accel_A2_bin" not in df.columns:
        print("[WARN] macro heatmap: 必要な列 (A2_duration_bin, accel_A2_bin) がありません。")
        return
    pivot = df.pivot_table(
        index="A2_duration_bin",
        columns="accel_A2_bin",
        values=value_col,
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
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
# メイン
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "config(JSON) と *_anomaly.csv から per-file サマリを生成し、"
            "混同行列・余裕時間指標・区間別集計・グラフを出力するスクリプト"
        )
    )
    ap.add_argument(
        "--config",
        required=True,
        help="tagged_dataset_filter_eval.json のパス",
    )
    ap.add_argument(
        "--per_file",
        default=None,
        help=(
            "既存の per_file_summary.csv のパス。"
            "指定しない場合は run_dir の *_anomaly.csv から自動生成する"
        ),
    )
    ap.add_argument(
        "--run_dir",
        default=None,
        help="config.evaluation.run_dir を上書きするパス（相対 or 絶対）",
    ),
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "evaluation" not in cfg:
        raise ValueError("config に 'evaluation' セクションがありません。")
    eval_cfg = cfg["evaluation"]
    # run_dir の決定
    run_dir = (cfg_path.parent / eval_cfg["run_dir"]).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir が存在しません: {run_dir}")
    # アクセル列名（ON/OFF 共通）
    accel_col_name = eval_cfg.get("accel_column_name", "accelpedalangle")
    # ON / OFF モード判定
    # - ON: label_review_sheet がある
    # - OFF: normal_ledger_sheet がある
    mode = None
    if "label_review_sheet" in eval_cfg:
        mode = "ON"
    if "normal_ledger_sheet" in eval_cfg:
        # 両方ある場合は明示的に分けたほうが良いが、
        # ここでは normal_ledger_sheet を優先して OFF とみなす
        mode = "OFF"
    if mode is None:
        raise ValueError(
            "evaluation に 'label_review_sheet'(ON) も 'normal_ledger_sheet'(OFF) もありません。"
        )
    # per_file_summary の取得（既存ファイル優先／なければ *_anomaly.csv から生成）
    if args.per_file is not None:
        per_file_path = Path(args.per_file)
        if not per_file_path.exists():
            raise FileNotFoundError(per_file_path)
        pf = pd.read_csv(per_file_path)
        # ON モードの場合のみ、ラベルシートの A1/A2/A3 を付与
        if mode == "ON":
            label_cfg = eval_cfg["label_review_sheet"]
            label_df = load_label_intervals(label_cfg)
            pf["basename"] = pf["basename"].astype(str).map(normalize_basename)
            pf = pf.merge(label_df, on="basename", how="left", suffixes=("", "_label"))
    else:
        if mode == "ON":
            # ON: 事故データ（label=1, ラベルシート使用）
            if "label_review_sheet" not in eval_cfg:
                raise ValueError("config.evaluation に 'label_review_sheet' セクションがありません。")
            label_cfg = eval_cfg["label_review_sheet"]
            label_df = load_label_intervals(label_cfg)
            pf = build_per_file_summary_from_dir(run_dir, label_df, accel_col_name=accel_col_name)
        else:
            # OFF: 正常データ（label=0, 市場走行管理台帳の K列を使用）
            if "normal_ledger_sheet" not in eval_cfg:
                raise ValueError("config.evaluation に 'normal_ledger_sheet' セクションがありません。")
            ledger_cfg = eval_cfg["normal_ledger_sheet"]
            normal_basenames = load_normal_basenames_from_ledger(ledger_cfg)
            pf = build_per_file_summary_normal_from_dir(
                run_dir,
                normal_basenames,
                accel_col_name=accel_col_name,
            )
        # 参考用に per_file_summary.csv を保存しておく
        per_file_path = run_dir / "per_file_summary_auto.csv"
        pf.to_csv(per_file_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] per_file_summary を自動生成して保存しました: {per_file_path}")
    # セグメント用の列 (A2_duration_bin など) を付与
    pf = add_segment_columns(pf)
    # 混同行列
    cm = compute_confusion_matrices(pf)
    print("=== Confusion Matrix (A2_to_A3start) ===")
    for k, v in cm["A2_to_A3start"].items():
        print(f"{k}: {v}")
    print()
    print("=== Confusion Matrix (A2_to_A3end) ===")
    for k, v in cm["A2_to_A3end"].items():
        print(f"{k}: {v}")
    print()
    # 余裕時間統計（ON のみ label==1 なので、OFF では何も出ない）
    margin_stats = compute_margin_stats(pf)
    if margin_stats:
        print("=== Margin Stats (pre_collision_detected=True, label=1) ===")
        print("margin_seconds_all:", margin_stats["margin_seconds_all"])
        print("ratio_A1A2_all:", margin_stats["ratio_A1A2_all"])
        print("ratio_A1_phase:", margin_stats["ratio_A1_phase"])
        print("ratio_A2_phase:", margin_stats["ratio_A2_phase"])
    else:
        print("[INFO] 余裕時間統計を計算できませんでした（対象データなし）")
    # 区間ごとの集計表（ON のみ有効）
    seg_tbl_A2 = aggregate_by_segments(
        pf, group_cols=["A2_duration_bin"], use_priority_only=False
    )
    out_seg_A2 = run_dir / "per_file_seg_A2_duration.csv"
    seg_tbl_A2.to_csv(out_seg_A2, index=False, encoding="utf-8-sig")
    print(f"[INFO] 区間別集計(A2_duration_bin) を保存しました: {out_seg_A2}")
    seg_tbl_macro = aggregate_by_segments(
        pf,
        group_cols=["A2_duration_bin", "accel_A2_bin", "shift_at_collision"],
        use_priority_only=False,
    )
    out_seg_macro = run_dir / "per_file_seg_macro.csv"
    seg_tbl_macro.to_csv(out_seg_macro, index=False, encoding="utf-8-sig")
    print(
        f"[INFO] 区間別集計(A2_duration_bin×accel_A2_bin×shift) を保存しました: {out_seg_macro}"
    )
    seg_tbl_macro_key = aggregate_by_segments(
        pf,
        group_cols=["A2_duration_bin", "accel_A2_bin", "shift_at_collision"],
        use_priority_only=True,
    )
    out_seg_macro_key = run_dir / "per_file_seg_macro_keycases.csv"
    seg_tbl_macro_key.to_csv(out_seg_macro_key, index=False, encoding="utf-8-sig")
    print(f"[INFO] 区間別集計(重点事故) を保存しました: {out_seg_macro_key}")
    # グラフ出力ディレクトリ
    fig_dir = run_dir / "figs"
    # 2-1: first_detect_phase 分布（ONのみ label==1 がある）
    plot_first_phase_distribution(
        pf,
        out_dir=fig_dir,
        group_col="A2_duration_bin",
        use_priority_only=False,
    )
    # 2-2: 余裕時間（秒・割合）の箱ひげ図（ONのみ）
    plot_margin_boxplots(
        pf,
        out_dir=fig_dir,
        group_col="A2_duration_bin",
        use_priority_only=False,
    )
    # 2-3: マクロヒートマップ（ONのみ有効）
    plot_macro_heatmap(
        seg_macro_df=seg_tbl_macro,
        out_dir=fig_dir,
        value_col="R_phase_A2",  # 例: A2で初検知した割合
    )
if __name__ == "__main__":
    main()