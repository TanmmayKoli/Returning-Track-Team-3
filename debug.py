from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, get_entity_vals

from src.eegdb.ingest.bids_p300 import (
    _load_events_tsv,
    _filter_to_stimulus,
    _make_labels,
    _read_raw_no_participants,
    _prep_channels,
)

DSID = "ds003061"
ROOT = Path(f"data/raw/openneuro/{DSID}")

def iter_combos(root: Path):
    subs = get_entity_vals(root, "subject") or []
    sess = get_entity_vals(root, "session") or [None]
    tasks = get_entity_vals(root, "task") or [None]
    runs = get_entity_vals(root, "run") or [None]
    for sub in subs:
        for ses in sess:
            for task in tasks:
                for run in runs:
                    yield dict(subject=sub, session=ses, task=task, run=run)

def main():
    print("DS:", DSID)
    print("ROOT:", ROOT, "exists:", ROOT.exists())
    bad = 0
    ok = 0

    for ent in iter_combos(ROOT):
        bp = BIDSPath(root=ROOT, datatype="eeg", **ent)

        # events
        try:
            ev = _load_events_tsv(bp)
        except Exception:
            continue

        ev = _filter_to_stimulus(ev)
        if len(ev) == 0:
            continue

        try:
            labels = _make_labels(ev)
        except Exception:
            bad += 1
            print("\n[NO LABELS]", ent)
            print("columns:", ev.columns.tolist())
            if "value" in ev.columns:
                print("value unique:", ev["value"].astype(str).str.lower().str.strip().unique()[:20])
            continue

        keep = labels.notna()
        ev = ev.loc[keep].reset_index(drop=True)
        labels = labels.loc[keep].astype(int).reset_index(drop=True)

        n_t = int(labels.sum())
        n_nt = int((labels == 0).sum())
        if n_t == 0 or n_nt == 0:
            bad += 1
            print("\n[MISSING CLASS]", ent, "targets:", n_t, "nontargets:", n_nt)
            if "value" in ev.columns:
                print("value unique:", ev["value"].astype(str).str.lower().str.strip().unique()[:20])
            continue

        # epoch sanity check
        try:
            raw = _read_raw_no_participants(bp)
            raw.load_data()
            raw = _prep_channels(raw)
            sfreq = float(raw.info["sfreq"])

            if "sample" in ev.columns and pd.api.types.is_numeric_dtype(ev["sample"]):
                sample = ev["sample"].round().astype(int).to_numpy()
            else:
                sample = np.round(ev["onset"].astype(float).to_numpy() * sfreq).astype(int)

            mne_events = np.c_[sample, np.zeros(len(sample), int), labels.to_numpy() + 1]
            codes = set(mne_events[:, 2].tolist())
            if 1 not in codes or 2 not in codes:
                raise RuntimeError("codes missing 1 or 2 unexpectedly")

            epochs = mne.Epochs(
                raw, mne_events, event_id={"nontarget": 1, "target": 2},
                tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
                preload=True,  # IMPORTANT
                reject_by_annotation=False,
                on_missing="ignore",
                verbose=False
            )

            print("[OK]", ent, "epochs:", len(epochs), "targets:", n_t, "nontargets:", n_nt)
            ok += 1

        except Exception as e:
            bad += 1
            print("\n[EPOCH FAIL]", ent, "targets:", n_t, "nontargets:", n_nt)
            print("error:", repr(e))

    print("\nDONE. ok:", ok, "bad:", bad)

if __name__ == "__main__":
    main()
