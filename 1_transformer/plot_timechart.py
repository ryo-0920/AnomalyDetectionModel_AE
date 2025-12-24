import argparse
import os
import glob
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import shutil
import json
import re
# カテゴリー定義（表示順はこの定義順に従う）
CATEGORIES: List[Tuple[str, List[str]]] = [
    ("動き", ["speed", "accelerationfb","accelerationlr","steeringangle","angularvelocity"]),
    ("運転者操作", ["accelpedalangle"]),
    ("車両システム", ["brakepressure"]),
    ("診断", ["error", "anomaly","is_anomaly","anomaly_level"]),
]
# 離散値としてステップ描画するシグナル
DISCRETE_SIGNALS = {"anomaly","is_anomaly", "error", "trccontrol", "vsccontrol", "atshiftposition"}
# ==============================
# 読み込み・前処理
# ==============================
def read_csv_to_df(csv_path: str) -> pd.DataFrame:
    """
    CSVを読み込み、列名を小文字・前後空白除去に正規化したDataFrameを返す。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def get_time_axis(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """
    時間軸を返す。frame列があればそれを使用、なければインデックス。
    """
    if "frame" in df.columns:
        return df["frame"].values, "frame"
    else:
        return df.index.values, "index"
def collect_signals_by_category(df: pd.DataFrame) -> List[str]:
    """
    CATEGORIESの順に、存在する列だけを抽出してフラットなシグナル一覧を作る。
    """
    signals: List[str] = []
    for _, cols in CATEGORIES:
        for c in cols:
            if c in df.columns:
                signals.append(c)
    if not signals:
        raise ValueError("描画対象の既知シグナルが見つかりませんでした")
    return signals

class InlineLogger:
    def __init__(self):
        self.prev_len = 0
    def print(self, text: str):
        # 端末幅に合わせて必要なら省略
        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        # 端末幅より少し短くしておく（…とスペース用に余裕）
        max_len = max(10, cols - 2)
        if len(text) > max_len:
            text = text[:max_len - 3] + "..."
        # 前回より短い場合の残り文字を空白で上書き
        pad = max(0, self.prev_len - len(text))
        sys.stdout.write("\r" + text + " " * pad)
        sys.stdout.flush()
        self.prev_len = len(text)
    def newline(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.prev_len = 0

def short_name(path: str, keep=40):
    name = os.path.basename(path).replace("⇒", "->")
    return name if len(name) <= keep else name[:keep-3] + "..."

# ==============================
# 可視化関連
# ==============================
def create_figure(nrows: int) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    サブプロットを作成して返す。縦長をシグナル数に応じて調整。
    """
    height = max(2.0 * nrows, 6.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, height), sharex=True)
    if nrows == 1:
        axes = [axes]
    return fig, axes

def plot_single_signal(ax: plt.Axes, t: np.ndarray, df: pd.DataFrame, sig: str):
    """
    単一シグナルの描画。anomalyは特別扱い、離散シグナルはステップ描画。
    """
    if sig == "anomaly":
        # 連続スコア（0〜1）をそのまま描画
        anom = pd.to_numeric(df[sig], errors="coerce").fillna(0).clip(0, 1).astype(np.float32).values
        # 線描画（連続値を見たいならこちら）
        ax.plot(t, anom, color="crimson", linewidth=1.2, label="anomaly")
        # ステップ表示にしたい場合は以下に切り替え：
        # ax.step(t, anom, where="post", color="crimson", linewidth=1.0, label="anomaly")
        ax.set_ylim(-0.05, 1.05)
    elif sig == "is_anomaly":
        # 0/1 の離散フラグをステップ描画（赤）
        y = pd.to_numeric(df[sig], errors="coerce").fillna(0).clip(0, 1).astype(np.float32).values
        ax.step(t, y, where="post", color="crimson", linewidth=1.2, label="is_anomaly")
        ax.set_ylim(-0.05, 1.05)
    elif sig == "anomaly_level":
        # 数値化（NaNはそのまま扱う）
        y = pd.to_numeric(df[sig], errors="coerce").values
        ax.step(t, y, where="post", color="crimson", linewidth=1.2, label="anomaly_level")
        # 目盛りと表示範囲をユニークな値に合わせて調整（離散レベル向け）
        finite = y[np.isfinite(y)]
        if finite.size > 0:
            uniq = np.unique(finite)
            ax.set_yticks(uniq)
            ax.set_ylim(uniq.min() - 0.5, uniq.max() + 0.5)
    elif sig in DISCRETE_SIGNALS:
        y = pd.to_numeric(df[sig], errors="coerce").fillna(np.nan).values
        ax.step(t, y, where="post", linewidth=1.0, label=sig)
    else:
        y = pd.to_numeric(df[sig], errors="coerce").values
        ax.plot(t, y, linewidth=1.1, label=sig)
    ax.set_ylabel(sig)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
def shade_spans(ax: plt.Axes, t: np.ndarray, mask: np.ndarray, color: str, label: str | None):
    """
    0/1マスクの連続区間を検出して塗る。ラベルは最初の軸のみ付与可能。
    """
    mask = np.nan_to_num(mask, nan=0).astype(int)
    diff = np.diff(np.r_[0, mask, 0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        # e-1 までを含める（where="post"と整合が良い）
        right = min(e - 1, len(t) - 1)
        ax.axvspan(t[s], t[right], color=color, alpha=0.12, label=label)
def shade_accel_spans_on_axes(axes: List[plt.Axes], t: np.ndarray, df: pd.DataFrame):
    """
    is_accelが存在する場合に、全軸へ加速区間の帯を重ねる。
    凡例は最初の軸にのみ表示して重複を避ける。
    """
    if "is_accel" not in df.columns:
        return
    accel = pd.to_numeric(df["is_accel"], errors="coerce").fillna(0).astype(int).values
    for i, ax in enumerate(axes):
        label = "accel" if i == 0 else None
        shade_spans(ax, t, accel, color="gold", label=label)
        # 凡例の重複を整理
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="upper right")
def add_on_off_metrics_text(ax: plt.Axes, df: pd.DataFrame):
    """
    is_accelのON/OFFに対するanomaly発生率を算出して、axに注記を追加。
    """
    if "anomaly" not in df.columns or "is_accel" not in df.columns:
        return
    anom = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).clip(0, 1).astype(int).values
    accel = pd.to_numeric(df["is_accel"], errors="coerce").fillna(0).astype(int).values
    on_mask = accel == 1
    off_mask = ~on_mask
    on_total = int(on_mask.sum())
    off_total = int(off_mask.sum())
    on_hits = int(np.nansum(anom[on_mask])) if on_total > 0 else 0
    off_hits = int(np.nansum(anom[off_mask])) if off_total > 0 else 0
    on_rate = (on_hits / on_total * 100.0) if on_total > 0 else float("nan")
    off_rate = (off_hits / off_total * 100.0) if off_total > 0 else float("nan")
    txt = (
        f"ON(accel): {on_hits}/{on_total} = {on_rate:.1f}%\n"
        f"OFF(non-accel): {off_hits}/{off_total} = {off_rate:.1f}%"
    )
    ax.text(
        0.01, 0.98, txt,
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", fc=(1.0, 1.0, 0.6, 0.5), ec="none"),
    )
def build_output_path(csv_path: str) -> str:
    """
    出力画像のパスをCSVと同ディレクトリに生成。
    """
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join(out_dir, f"{base}_timechart_transformer.png")
def save_figure(fig: plt.Figure, out_path: str):
    """
    図を保存してクローズ。
    """
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
# ==============================
# 1ファイル処理
# ==============================
def process_one(csv_path: str) -> str:
    """
    単一CSVの可視化処理を実行し、出力画像パスを返す。
    """

    df = read_csv_to_df(csv_path)
    t, x_label = get_time_axis(df)
    signals = collect_signals_by_category(df)
    fig, axes = create_figure(nrows=len(signals))
    if len(axes) == 1:
        axes = [axes[0]]
    # 描画
    for idx, sig in enumerate(signals):
        ax = axes[idx]
        plot_single_signal(ax, t, df, sig)
        ax.set_xlabel(x_label if idx == len(signals) - 1 else "")
    # 帯の重ね合わせとON/OFFメトリクス
    shade_accel_spans_on_axes(axes, t, df)
    add_on_off_metrics_text(axes[0], df)
    out_path = build_output_path(csv_path)
    save_figure(fig, out_path)
    return out_path
# ==============================
# CLI関連
# ==============================
def gather_csv_list(target: str) -> List[str]:
    """
    文字列がグロブ・ディレクトリ・単一ファイルのいずれかに応じてCSVリストを作る。
    """
    if any(ch in target for ch in "*?[]"):
        csv_list = sorted(glob.glob(target))
    elif os.path.isdir(target):
        csv_list = sorted(glob.glob(os.path.join(target, "*.csv")))
    else:
        csv_list = [target]
    return csv_list
def parse_args() -> argparse.Namespace:
    """
    引数を解析して返す。
    """
    parser = argparse.ArgumentParser(
        description="Plot time chart from anomaly CSV (file, directory, or glob)"
    )
    parser.add_argument(
        "--csv",
        default="result",
        help=(
            "CSVファイル, ディレクトリ, もしくはグロブパターンを指定できます "
            "(例: --csv result, --csv result\\*.csv, --csv result\\foo.csv)"
        ),
    )
    return parser.parse_args()

def main():
    args = parse_args()
    csv_list = gather_csv_list(args.csv)
    if not csv_list:
        raise FileNotFoundError(f"対象CSVが見つかりませんでした: {args.csv}")
    print(f"対象CSVファイル数: {len(csv_list)}")
    success, failed = 0, 0
    logger = InlineLogger()

    for i, csv_path in enumerate(csv_list, 1):
        name = short_name(csv_path, keep=40)
        logger.print(f"[{i}/{len(csv_list)}] : {name}")
        try:
            out_path = process_one(csv_path)
            #print(f"  -> Saved chart: {out_path}")
            success += 1
        except Exception as e:
            logger.newline()
            print(f"  -> スキップ (エラー): {e}")
            failed += 1
    logger.newline()
    print(f"完了: 成功 {success} / 失敗 {failed}")


if __name__ == "__main__":
    main()