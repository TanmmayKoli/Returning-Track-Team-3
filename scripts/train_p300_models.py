from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import GroupShuffleSplit


# ---------- config ----------
FEATURES_PARQUET = Path("data/db/features/p300_erp_windows_v1/features.parquet")
OUT_DIR = Path("data/db/models/p300_erp_windows_v1")


@dataclass
class SplitConfig:
    test_size: float = 0.20
    random_state: int = 42
    # groups control leakage. default: subject within dsid
    group_cols: tuple[str, ...] = ("dsid", "subject")


def _load_features(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    # sanity
    required = {"dsid", "subject", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _make_X_y_groups(df: pd.DataFrame):
    # Use only numeric feature columns starting with "f_"
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    if not feat_cols:
        raise ValueError("No feature columns found (expected columns starting with 'f_').")

    X = df[feat_cols].copy()
    y = df["label"].astype(int).to_numpy()

    return X, y, feat_cols


def _make_groups(df: pd.DataFrame, group_cols: tuple[str, ...]) -> np.ndarray:
    # Example group: dsid|subject
    parts = []
    for c in group_cols:
        if c not in df.columns:
            raise ValueError(f"Group col '{c}' missing from df.")
        parts.append(df[c].astype(str).fillna(""))
    g = parts[0]
    for p in parts[1:]:
        g = g + "|" + p
    return g.to_numpy()


def _train_test_split_grouped(df: pd.DataFrame, cfg: SplitConfig):
    groups = _make_groups(df, cfg.group_cols)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state
    )
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx


def _evaluate_binary(y_true, y_proba, y_pred, prefix=""):
    out = {}
    # Some models might not output calibrated probs; handle gracefully
    if y_proba is not None:
        out[prefix + "roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out[prefix + "avg_precision"] = float(average_precision_score(y_true, y_proba))
    out[prefix + "accuracy"] = float(accuracy_score(y_true, y_pred))
    out[prefix + "balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out[prefix + "f1"] = float(f1_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    out[prefix + "confusion_matrix"] = cm.tolist()
    return out


def _evaluate_by_group(df_test: pd.DataFrame, y_true, y_pred, y_proba, group_col: str):
    rows = []
    for key, sub in df_test.groupby(group_col):
        idx = sub.index.to_numpy()
        yt = y_true[idx]
        yp = y_pred[idx]
        ypb = None if y_proba is None else y_proba[idx]
        m = _evaluate_binary(yt, ypb, yp)
        m[group_col] = str(key)
        m["n"] = int(len(sub))
        m["pos"] = int(yt.sum())
        rows.append(m)
    return pd.DataFrame(rows).sort_values(["n"], ascending=False)


def _build_models():
    """
    A few solid baselines:
    - Logistic Regression (strong for ERP features)
    - Linear SVM (calibrated)
    - RandomForest (nonlinear baseline)
    """
    models = {}

    models["logreg"] = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    # LinearSVC doesn't output probabilities; calibrate
    base_svc = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("svc", LinearSVC(class_weight="balanced")),
        ]
    )
    models["linear_svm_calibrated"] = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)

    models["rf"] = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1
            )),
        ]
    )

    return models


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load]", FEATURES_PARQUET)
    df = _load_features(FEATURES_PARQUET)

    # Keep only rows with valid label 0/1
    df = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)

    # Helpful quick stats
    print("[rows]", len(df))
    print("[datasets]\n", df["dsid"].value_counts())
    print("[label balance] pos:", int(df["label"].sum()), "neg:", int((df["label"] == 0).sum()))

    # Split grouped by subject within dataset to avoid leakage
    split_cfg = SplitConfig(test_size=0.2, random_state=42, group_cols=("dsid", "subject"))
    train_idx, test_idx = _train_test_split_grouped(df, split_cfg)

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    print("[split] train:", len(df_train), "test:", len(df_test))
    print("[test groups]", len(set(_make_groups(df_test, split_cfg.group_cols))))

    X_train, y_train, feat_cols = _make_X_y_groups(df_train)
    X_test, y_test, _ = _make_X_y_groups(df_test)

    models = _build_models()
    all_results = {}

    for name, model in models.items():
        print(f"\n[train] {name}")
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

        metrics = _evaluate_binary(y_test, y_proba, y_pred)
        all_results[name] = metrics

        print("[metrics]", {k: v for k, v in metrics.items() if "matrix" not in k})

        # Save per-dataset + per-subject breakdowns
        df_test_local = df_test.copy()
        # Use original indices for group evaluation alignment
        df_test_local.index = np.arange(len(df_test_local))

        by_dsid = _evaluate_by_group(df_test_local, y_test, y_pred, y_proba, "dsid")
        by_sub = _evaluate_by_group(df_test_local, y_test, y_pred, y_proba, "subject")

        by_dsid.to_csv(OUT_DIR / f"{name}_by_dsid.csv", index=False)
        by_sub.to_csv(OUT_DIR / f"{name}_by_subject.csv", index=False)

        # Save sklearn model via joblib
        import joblib
        joblib.dump(
            {"model": model, "feature_columns": feat_cols, "split_cfg": split_cfg.__dict__},
            OUT_DIR / f"{name}.joblib",
        )

        # Also dump a readable report
        rep = classification_report(y_test, y_pred, digits=4)
        (OUT_DIR / f"{name}_report.txt").write_text(rep)

    (OUT_DIR / "metrics.json").write_text(json.dumps(all_results, indent=2))
    print("\n[saved]", OUT_DIR)


if __name__ == "__main__":
    main()
