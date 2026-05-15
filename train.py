"""
train.py
--------
End-to-end pipeline for the ChurnAI Time-Series sub-system.

Steps:
    1. Generate (or load) the synthetic dataset.
    2. Train the BiLSTM with attention on 12-month sequences.
    3. Run inference on every customer and write predictions.csv.
    4. Save per-epoch curves + a metrics.json bundle for the UI.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import numpy as np
import pandas as pd
import torch

import config as C
from generate_data import generate
from models import lstm_encoder as M


def _ensure_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(C.CUSTOMERS_CSV) and os.path.exists(C.TIMESERIES_CSV):
        customers = pd.read_csv(C.CUSTOMERS_CSV)
        ts        = pd.read_csv(C.TIMESERIES_CSV)
        return customers, ts
    print("[data] generating synthetic dataset...")
    return generate()


def main() -> None:
    print("=" * 72)
    print("  CHURNAAI · TIME-SERIES TRAINING PIPELINE")
    print("=" * 72)

    # ---------------- 1. dataset ----------------
    customers, ts = _ensure_dataset()
    print(f"[data] customers={len(customers):,}  ts_rows={len(ts):,}  "
          f"churn_rate={customers['churn'].mean():.1%}")

    # ---------------- 2. train ----------------
    print("\n[train] training BiLSTM + attention encoder ...")
    result = M.train_lstm(ts, verbose=True)
    model   = result["model"]
    scaler  = result["scaler"]
    metrics = result["metrics"]

    # ---------------- 3. inference for all customers ----------------
    print("\n[predict] scoring every customer ...")
    ids, X_raw, y = M.build_sequences(ts)
    X = M.apply_scaler(X_raw, scaler)
    probs, _, attns = M.predict_with_attention(model, X)

    def risk_tier(p: float) -> str:
        if p >= C.RISK_HIGH:    return "High"
        if p >= C.RISK_MEDIUM:  return "Medium"
        return "Low"

    last_month = ts.sort_values(["customerID", "month"]).groupby("customerID").tail(1)
    last_month_map = last_month.set_index("customerID")[
        ["data_gb", "monthly_charge", "support_tickets", "login_count"]
    ].to_dict("index")

    rows = []
    for cid, p, a, label in zip(ids, probs, attns, y):
        last = last_month_map.get(cid, {})
        rows.append({
            "customerID":         cid,
            "churn_prob":         float(p),
            "risk_tier":          risk_tier(float(p)),
            "true_label":         int(label),
            "month12_data_gb":    last.get("data_gb"),
            "month12_charge":     last.get("monthly_charge"),
            "month12_tickets":    last.get("support_tickets"),
            "month12_logins":     last.get("login_count"),
            # attention weight on the most recent month — a useful UI cue:
            "attn_last_month":    float(a[-1]),
            "attn_last3_months":  float(a[-3:].sum()),
        })
    preds = pd.DataFrame(rows)
    preds = preds.merge(customers, on="customerID", how="left")
    preds = preds.sort_values("churn_prob", ascending=False).reset_index(drop=True)
    preds.to_csv(C.PREDICTIONS_CSV, index=False)

    # ---------------- 4. metrics bundle ----------------
    bundle = {
        "metrics": metrics,
        "thresholds": {"high": C.RISK_HIGH, "medium": C.RISK_MEDIUM},
        "feature_columns": C.TS_FEATURES,
        "sequence_length": C.SEQ_LEN,
        "n_customers": int(len(customers)),
        "predicted_high_risk": int((preds["risk_tier"] == "High").sum()),
        "predicted_medium_risk": int((preds["risk_tier"] == "Medium").sum()),
        "predicted_low_risk": int((preds["risk_tier"] == "Low").sum()),
    }
    with open(C.METRICS_JSON, "w") as f:
        json.dump(bundle, f, indent=2)

    # ---------------- summary ----------------
    auc = metrics["test_auc"]
    acc = metrics["test_accuracy"]
    rec = metrics["test_recall"]
    pre = metrics["test_precision"]

    print("\n" + "=" * 72)
    print("  TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Test AUC-ROC : {auc:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test Recall  : {rec:.4f}")
    print(f"  Test Precision: {pre:.4f}")
    print("-" * 72)
    print(f"  Predicted High risk   : {bundle['predicted_high_risk']:>5,}")
    print(f"  Predicted Medium risk : {bundle['predicted_medium_risk']:>5,}")
    print(f"  Predicted Low risk    : {bundle['predicted_low_risk']:>5,}")
    print("-" * 72)
    print(f"  Model     -> {C.LSTM_MODEL_PATH}")
    print(f"  Scaler    -> {C.TS_SCALER_PATH}")
    print(f"  Preds     -> {C.PREDICTIONS_CSV}")
    print(f"  Metrics   -> {C.METRICS_JSON}")
    print("=" * 72)
    print("  Next: streamlit run app/dashboard.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
