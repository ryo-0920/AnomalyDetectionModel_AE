import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


METHOD_PATHS: Dict[str, str] = {
    "Transformer": os.path.join("1_transformer", "result", "gofumi_accel_anomaly.csv"),
    "Transformer_with_pic": os.path.join(
        "1_transformer_with_pic", "result", "gofumi_accel_anomaly.csv"
    ),
    "BERT": os.path.join("2_BERT", "result", "gofumi_accel_anomaly.csv"),
    "BERT_with_pic": os.path.join("2_BERT_wit_pic", "result", "gofumi_accel_anomaly.csv"),
    "LSTM": os.path.join("3_LSTM", "result", "gofumi_accel_anomaly.csv"),
    "LSTM_with_pic": os.path.join("3_LSTM_wit_pic", "result", "gofumi_accel_anomaly.csv"),
    "GRU": os.path.join("4_GRU", "result", "gofumi_accel_anomaly.csv"),
    "GRU_with_pic": os.path.join("4_GRU_wit_pic", "result", "gofumi_accel_anomaly.csv"),
}


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics = {}
    df2 = df.copy()
    df2.columns = [column.lower() for column in df2.columns]

    metrics["rows"] = int(len(df2))
    if "anomaly" in df2.columns:
        anomaly = df2["anomaly"].astype(float).values
        metrics["anomaly_count"] = int(np.nansum(anomaly))
        metrics["anomaly_rate_%"] = float(np.nansum(anomaly) / max(len(anomaly), 1) * 100.0)
    else:
        metrics["anomaly_count"] = np.nan
        metrics["anomaly_rate_%"] = np.nan

    if "error" in df2.columns:
        metrics["error_mean"] = float(np.nanmean(df2["error"].values))
        metrics["error_std"] = float(np.nanstd(df2["error"].values))
    else:
        metrics["error_mean"] = np.nan
        metrics["error_std"] = np.nan

    if "anomaly" in df2.columns and "is_accel" in df2.columns:
        accel = df2["is_accel"].astype(int).values
        anomaly = df2["anomaly"].astype(int).values
        on_mask = accel == 1
        off_mask = ~on_mask
        on_total = int(on_mask.sum())
        off_total = int(off_mask.sum())
        on_hits = int(np.nansum(anomaly[on_mask])) if on_total > 0 else 0
        off_hits = int(np.nansum(anomaly[off_mask])) if off_total > 0 else 0
        metrics["on_total"] = on_total
        metrics["on_hits"] = on_hits
        metrics["on_rate_%"] = (on_hits / on_total * 100.0) if on_total > 0 else np.nan
        metrics["off_total"] = off_total
        metrics["off_hits"] = off_hits
        metrics["off_rate_%"] = (off_hits / off_total * 100.0) if off_total > 0 else np.nan
        if "error" in df2.columns:
            metrics["error_on_mean"] = (
                float(np.nanmean(df2.loc[on_mask, "error"])) if on_total > 0 else np.nan
            )
            metrics["error_off_mean"] = (
                float(np.nanmean(df2.loc[off_mask, "error"])) if off_total > 0 else np.nan
            )
    else:
        metrics.update(
            {
                "on_total": np.nan,
                "on_hits": np.nan,
                "on_rate_%": np.nan,
                "off_total": np.nan,
                "off_hits": np.nan,
                "off_rate_%": np.nan,
                "error_on_mean": np.nan,
                "error_off_mean": np.nan,
            }
        )
    return metrics


def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in ["frame", "steer", "throttle", "brake", "speed", "error", "anomaly", "is_accel"]
        if column in df.columns
    ]
    if columns:
        return df[columns]
    return df


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_xlsx = os.path.join(root, "experiments", "reporting", "gofumi_accel_anomaly_summary.xlsx")

    loaded: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, float]] = []

    for name, rel_path in METHOD_PATHS.items():
        path = os.path.join(root, rel_path)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [column.strip().lower() for column in df.columns]
            loaded[name] = df
            metrics = compute_metrics(df)
            metrics["method"] = name
            metrics["source_csv"] = os.path.relpath(path, root)
            summary_rows.append(metrics)
        except Exception as exc:
            summary_rows.append(
                {
                    "method": name,
                    "error": str(exc),
                    "source_csv": os.path.relpath(path, root),
                }
            )

    if not loaded:
        print("No result CSVs found. Expected files under */result/gofumi_accel_anomaly.csv")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows)
    cols_order = [
        "method",
        "rows",
        "anomaly_count",
        "anomaly_rate_%",
        "on_total",
        "on_hits",
        "on_rate_%",
        "off_total",
        "off_hits",
        "off_rate_%",
        "error_mean",
        "error_std",
        "error_on_mean",
        "error_off_mean",
        "source_csv",
    ]
    columns = [column for column in cols_order if column in summary_df.columns] + [
        column for column in summary_df.columns if column not in cols_order
    ]
    summary_df = summary_df[columns]

    engine = None
    for candidate in ("openpyxl", "xlsxwriter"):
        try:
            __import__(candidate)
            engine = candidate
            break
        except Exception:
            continue
    if engine is None:
        raise RuntimeError("Neither openpyxl nor xlsxwriter is installed. Please install one.")

    with pd.ExcelWriter(out_xlsx, engine=engine) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for name, df in loaded.items():
            pick_columns(df).to_excel(writer, sheet_name=name[:31], index=False)

    print(f"Saved Excel: {out_xlsx}")


if __name__ == "__main__":
    main()
