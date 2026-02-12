from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, f1_score


FEATURES_PARQUET = Path("data/db/features/p300_erp_windows_v1/features.parquet")


def make_model():
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def main():
    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)

    feat_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feat_cols]
    y = df["label"].astype(int).to_numpy()

    for train_dsid, test_dsid in [("ds003061", "ds002578"), ("ds002578", "ds003061")]:
        tr = df["dsid"].astype(str).eq(train_dsid).to_numpy()
        te = df["dsid"].astype(str).eq(test_dsid).to_numpy()

        if tr.sum() == 0 or te.sum() == 0:
            print(f"[skip] missing dsid rows for {train_dsid} or {test_dsid}")
            continue

        model = make_model()
        model.fit(X[tr], y[tr])

        y_pred = model.predict(X[te])
        y_proba = model.predict_proba(X[te])[:, 1]

        metrics = {
            "roc_auc": float(roc_auc_score(y[te], y_proba)),
            "balanced_accuracy": float(balanced_accuracy_score(y[te], y_pred)),
            "accuracy": float(accuracy_score(y[te], y_pred)),
            "f1": float(f1_score(y[te], y_pred)),
            "n_test": int(te.sum()),
            "pos_test": int(y[te].sum()),
        }

        print(f"\n[train {train_dsid} -> test {test_dsid}] {metrics}")


if __name__ == "__main__":
    main()
