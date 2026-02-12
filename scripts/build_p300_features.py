# scripts/build_p300_features.py
# calls the feature builder for P300 datasets, as specified in the config file,
# and appends the results to a parquet file in the db directory

from pathlib import Path
import yaml
from src.eegdb.store import append_parquet
from src.eegdb.ingest.bids_p300 import PreprocessConfig, build_p300_features_for_dataset


def main():
    cfg = yaml.safe_load(Path("configs/p300_phase1.yaml").read_text())
    raw_dir = Path(cfg["storage"]["raw_openneuro_dir"])
    db_dir = Path(cfg["storage"]["db_dir"])

    feat_name = cfg["features"]["name"]
    windows = cfg["features"]["windows_sec"]
    out_path = db_dir / "features" / feat_name / "features.parquet"

    pp_cfg = cfg["preprocess"]
    pp = PreprocessConfig(
        l_freq=float(pp_cfg["l_freq"]),
        h_freq=float(pp_cfg["h_freq"]),
        tmin=float(pp_cfg["tmin"]),
        tmax=float(pp_cfg["tmax"]),
        baseline=tuple(pp_cfg["baseline"]) if pp_cfg.get("baseline") else None,
        resample_sfreq=float(pp_cfg["resample_sfreq"]) if pp_cfg.get("resample_sfreq") else None,
        min_per_class=int(pp_cfg.get("min_per_class", 2)),  # <= NEW (default 2)
    )

    total = 0
    for ds in cfg["phase1_datasets"]:
        dsid = ds["id"]
        root = raw_dir / dsid
        if not root.exists():
            print(f"[warn] missing dataset folder: {root}")
            continue

        print(f"\n[build] {dsid} ({root})")
        df, stats = build_p300_features_for_dataset(dsid, root, pp, windows, return_stats=True)

        print(
            f"[stats:{dsid}] "
            + " ".join([f"{k}={v}" for k, v in stats.items()])
        )

        if df.empty:
            print(f"[warn] produced 0 rows for {dsid}")
            continue

        append_parquet(out_path, df)
        total += len(df)
        print(f"[ok] appended {len(df)} rows from {dsid}")

    print(f"\n[done] added {total} rows")
    print(f"[db] {out_path}")


if __name__ == "__main__":
    main()
