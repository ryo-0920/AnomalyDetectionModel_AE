import os
import sys
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


METHOD_PATHS: Dict[str, str] = {
    "Transformer": os.path.join("1_transformer", "result", "gofumi_accel_anomaly.csv"),
    "Transformer_with_pic": os.path.join("1_transformer_with_pic", "result", "gofumi_accel_anomaly.csv"),
    "BERT": os.path.join("2_BERT", "result", "gofumi_accel_anomaly.csv"),
    "BERT_with_pic": os.path.join("2_BERT_wit_pic", "result", "gofumi_accel_anomaly.csv"),
    "LSTM": os.path.join("3_LSTM", "result", "gofumi_accel_anomaly.csv"),
    "LSTM_with_pic": os.path.join("3_LSTM_wit_pic", "result", "gofumi_accel_anomaly.csv"),
    "GRU": os.path.join("4_GRU", "result", "gofumi_accel_anomaly.csv"),
    "GRU_with_pic": os.path.join("4_GRU_wit_pic", "result", "gofumi_accel_anomaly.csv"),
}


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    d = {}
    cols = {c.lower(): c for c in df.columns}
    df2 = df.copy()
    df2.columns = [c.lower() for c in df2.columns]

    d["rows"] = int(len(df2))
    if "anomaly" in df2.columns:
        anom = df2["anomaly"].astype(float).values
        d["anomaly_count"] = int(np.nansum(anom))
        d["anomaly_rate_%"] = float(np.nansum(anom) / max(len(anom), 1) * 100.0)
    else:
        d["anomaly_count"] = np.nan
        d["anomaly_rate_%"] = np.nan

    if "error" in df2.columns:
        d["error_mean"] = float(np.nanmean(df2["error"].values))
        d["error_std"] = float(np.nanstd(df2["error"].values))
    else:
        d["error_mean"] = np.nan
        d["error_std"] = np.nan

    if "anomaly" in df2.columns and "is_accel" in df2.columns:
        accel = df2["is_accel"].astype(int).values
        anom = df2["anomaly"].astype(int).values
        on_mask = accel == 1
        off_mask = ~on_mask
        on_total = int(on_mask.sum())
        off_total = int(off_mask.sum())
        on_hits = int(np.nansum(anom[on_mask])) if on_total > 0 else 0
        off_hits = int(np.nansum(anom[off_mask])) if off_total > 0 else 0
        d["on_total"] = on_total
        d["on_hits"] = on_hits
        d["on_rate_%"] = (on_hits / on_total * 100.0) if on_total > 0 else np.nan
        d["off_total"] = off_total
        d["off_hits"] = off_hits
        d["off_rate_%"] = (off_hits / off_total * 100.0) if off_total > 0 else np.nan
        if "error" in df2.columns:
            d["error_on_mean"] = float(np.nanmean(df2.loc[on_mask, "error"])) if on_total > 0 else np.nan
            d["error_off_mean"] = float(np.nanmean(df2.loc[off_mask, "error"])) if off_total > 0 else np.nan
    else:
        d.update({
            "on_total": np.nan,
            "on_hits": np.nan,
            "on_rate_%": np.nan,
            "off_total": np.nan,
            "off_hits": np.nan,
            "off_rate_%": np.nan,
            "error_on_mean": np.nan,
            "error_off_mean": np.nan,
        })
    return d


def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep a concise set of columns if present
    cols = [c for c in ["frame", "steer", "throttle", "brake", "speed", "error", "anomaly", "is_accel"] if c in df.columns]
    if cols:
        return df[cols]
    return df


def main():
    root = os.path.abspath(os.path.dirname(__file__))
    out_xlsx = os.path.join(root, "gofumi_accel_anomaly_summary.xlsx")

    loaded: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, float]] = []

    for name, rel in METHOD_PATHS.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            loaded[name] = df
            m = compute_metrics(df)
            m["method"] = name
            m["source_csv"] = os.path.relpath(path, root)
            summary_rows.append(m)
        except Exception as e:
            err_row = {"method": name, "error": str(e), "source_csv": os.path.relpath(path, root)}
            summary_rows.append(err_row)

    if not loaded:
        print("No result CSVs found. Expected files under */result/gofumi_accel_anomaly.csv")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows)
    # Order columns
    cols_order = [
        "method", "rows", "anomaly_count", "anomaly_rate_%",
        "on_total", "on_hits", "on_rate_%", "off_total", "off_hits", "off_rate_%",
        "error_mean", "error_std", "error_on_mean", "error_off_mean", "source_csv",
    ]
    cols = [c for c in cols_order if c in summary_df.columns] + [c for c in summary_df.columns if c not in cols_order]
    summary_df = summary_df[cols]

    # Write Excel
    engine = None
    for eng in ("openpyxl", "xlsxwriter"):
        try:
            __import__(eng)
            engine = eng
            break
        except Exception:
            continue
    if engine is None:
        raise RuntimeError("Neither openpyxl nor xlsxwriter is installed. Please install one to write .xlsx.")

    with pd.ExcelWriter(out_xlsx, engine=engine) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for name, df in loaded.items():
            df_out = pick_columns(df)
            # Sheet names are limited to 31 chars
            sheet = name[:31]
            df_out.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Saved Excel: {out_xlsx}")


if __name__ == "__main__":
    main()

