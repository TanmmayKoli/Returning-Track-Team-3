# src/eegdb/ingest/bids_p300.py
# Workhorse: load BIDS EEG, make epochs, extract ERP-window features (P300).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

# --- Label inference (text-based, for datasets like ds003061) ---
TARGET_PATTERNS = [r"\btarget\b", r"\bdeviant\b", r"\boddball\b", r"\brare\b"]
NONTARGET_PATTERNS = [r"\bnon[-_ ]?target\b", r"\bstandard\b", r"\bfrequent\b"]

# Channels that frequently appear in BioSemi / OpenNeuro as aux and cause montage warnings
AUX_CHANNELS = [
    "EXG1","EXG2","EXG3","EXG4","EXG5","EXG6","EXG7","EXG8",
    "GSR1","GSR2","Erg1","Erg2","Resp","Plet",
    "Temp",
]


def _norm(x: str) -> str:
    return str(x).strip().lower()


def _infer_label_from_text(x: str) -> int | None:
    s = _norm(x)
    for pat in TARGET_PATTERNS:
        if re.search(pat, s):
            return 1
    for pat in NONTARGET_PATTERNS:
        if re.search(pat, s):
            return 0
    return None


def _load_events_tsv(bp: BIDSPath) -> pd.DataFrame:
    events_bp = bp.copy().update(suffix="events", extension=".tsv")
    if events_bp.fpath is None or not events_bp.fpath.exists():
        raise FileNotFoundError(f"Missing events.tsv for {bp}. Expected: {events_bp.fpath}")
    return pd.read_csv(events_bp.fpath, sep="\t")


def _filter_to_stimulus(events_df: pd.DataFrame, dsid: str | None = None) -> pd.DataFrame:
    """
    Keep stimulus rows.

    - If trial_type exists and contains real 'stimulus' rows, use that.
    - Else fall back to filtering 'value' to remove response/boundary.
    - ds002578 specific: keep ONLY 2-digit numeric stimulus codes (e.g., "11".."55").
    """
    if dsid == "ds002578":
        if "value" not in events_df.columns:
            return events_df.iloc[0:0].copy()
        v = events_df["value"].astype(str).str.strip()
        # keep only codes like "11", "23", "55"
        mask = v.str.fullmatch(r"\d{2}")
        return events_df.loc[mask].copy()

    # Generic path: if trial_type has actual stimulus values, use it
    if "trial_type" in events_df.columns:
        tt = events_df["trial_type"]
        tt_non_na = tt.dropna()
        if len(tt_non_na) > 0:
            tt2 = tt.astype(str).str.lower().str.strip()
            if (tt2 == "stimulus").any():
                return events_df.loc[tt2.eq("stimulus")].copy()

    # Fallback: filter value column if possible
    if "value" in events_df.columns:
        v = events_df["value"].astype(str).str.lower().str.strip()
        # drop obvious non-stimulus markers
        drop = v.isin({"response", "boundary"})
        return events_df.loc[~drop].copy()

    return events_df


def _make_labels(events_df: pd.DataFrame, dsid: str | None = None) -> pd.Series:
    """
    Return a Series of {0,1,None} aligned to events_df rows.
    """
    # ds002578 rule: value is a 2-digit code XY, target if X==Y.
    if dsid == "ds002578":
        if "value" not in events_df.columns:
            raise ValueError("ds002578 requires 'value' column for labeling.")
        v = events_df["value"].astype(str).str.strip()
        # only codes like '11'
        ok = v.str.fullmatch(r"\d{2}")
        out = pd.Series([None] * len(events_df), index=events_df.index, dtype="object")
        vv = v.loc[ok]
        # target if first digit == second digit
        lab = (vv.str[0] == vv.str[1]).astype(int)
        out.loc[ok] = lab
        if out.notna().any():
            return out
        raise ValueError("ds002578: could not label any rows (no 2-digit codes found).")

    # Text-based labels (ds003061 etc)
    for col in ["value", "trial_type", "stim_type", "event_type", "condition"]:
        if col in events_df.columns:
            labels = events_df[col].astype(str).map(_infer_label_from_text)
            if labels.notna().any():
                return labels

    # Numeric fallback: if exactly 2 unique numeric values, treat max as target
    for col in events_df.columns:
        if pd.api.types.is_numeric_dtype(events_df[col]):
            vals = events_df[col].dropna().unique()
            if len(vals) == 2:
                vmax = np.max(vals)
                return events_df[col].map(lambda v: 1 if v == vmax else 0)

    raise ValueError("Could not infer target/non-target labels from events.tsv.")


def _find_eeg_file_from_bp(bp: BIDSPath) -> Path:
    """
    Fallback to load EEG without mne-bids participants.tsv parsing.
    We search for *_eeg.set/.edf/.bdf. Handles run-3 vs run-03.
    """
    if bp.fpath is not None and Path(bp.fpath).exists():
        return Path(bp.fpath)

    root = Path(bp.root)
    sub = bp.subject
    task = bp.task
    run = bp.run
    ses = bp.session

    if ses:
        base = root / f"sub-{sub}" / f"ses-{ses}" / "eeg"
        prefix = f"sub-{sub}_ses-{ses}_task-{task}"
    else:
        base = root / f"sub-{sub}" / "eeg"
        prefix = f"sub-{sub}_task-{task}"

    run_candidates = []
    if run is None:
        run_candidates = [None]
    else:
        try:
            rint = int(str(run))
            run_candidates = [str(run), f"{rint:02d}"]
        except Exception:
            run_candidates = [str(run)]

    exts = [".set", ".edf", ".bdf"]
    candidates: list[Path] = []
    for r in run_candidates:
        if r is None:
            for ext in exts:
                candidates.append(base / f"{prefix}_eeg{ext}")
        else:
            for ext in exts:
                candidates.append(base / f"{prefix}_run-{r}_eeg{ext}")

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError("Could not find EEG file. Tried:\n" + "\n".join(map(str, candidates)))


def _read_raw_no_participants(bp: BIDSPath) -> mne.io.BaseRaw:
    """
    Robust raw loader:
    1) Try read_raw_bids (fast when it works)
    2) If it fails (e.g., participants.tsv gender issue), load EEG file directly.
    """
    try:
        return read_raw_bids(bp, verbose=False)
    except Exception:
        eeg_path = _find_eeg_file_from_bp(bp)
        suf = eeg_path.suffix.lower()
        if suf == ".set":
            return mne.io.read_raw_eeglab(eeg_path, preload=False, verbose=False)
        if suf == ".edf":
            return mne.io.read_raw_edf(eeg_path, preload=False, verbose=False)
        if suf == ".bdf":
            return mne.io.read_raw_bdf(eeg_path, preload=False, verbose=False)
        raise RuntimeError(f"Unsupported EEG file type for fallback: {eeg_path}")


def _prep_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    - Mark aux channels as 'misc' so montage doesn't complain
    - Pick EEG only for feature extraction
    """
    present = [ch for ch in AUX_CHANNELS if ch in raw.ch_names]
    if present:
        raw.set_channel_types({ch: "misc" for ch in present}, verbose=False)

    raw.pick_types(eeg=True, misc=False, stim=False, eog=False, ecg=False, emg=False, exclude=[])
    return raw


def erp_window_features(epochs: mne.Epochs, windows_sec: list[list[float]]) -> pd.DataFrame:
    data = epochs.get_data()  # (n_epochs, n_ch, n_times)
    times = epochs.times
    ch_names = epochs.ch_names

    feats = {}
    for wi, (a, b) in enumerate(windows_sec):
        mask = (times >= a) & (times <= b)
        if not np.any(mask):
            raise ValueError(f"Window {a}-{b}s does not intersect epoch times.")
        win_mean = data[:, :, mask].mean(axis=2)
        for ci, ch in enumerate(ch_names):
            feats[f"f_{ch}_w{wi}"] = win_mean[:, ci]
    return pd.DataFrame(feats)


def _iter_bids_combos(dataset_root: Path):
    subjects = get_entity_vals(dataset_root, "subject") or []
    sessions = get_entity_vals(dataset_root, "session") or [None]
    tasks = get_entity_vals(dataset_root, "task") or [None]
    runs = get_entity_vals(dataset_root, "run") or [None]
    for sub in subjects:
        for ses in sessions:
            for task in tasks:
                for run in runs:
                    yield {"subject": sub, "session": ses, "task": task, "run": run}


@dataclass
class PreprocessConfig:
    l_freq: float
    h_freq: float
    tmin: float
    tmax: float
    baseline: tuple[float, float] | None
    resample_sfreq: float | None
    min_per_class: int = 2  # NEW


def build_p300_features_for_dataset(
    dsid: str,
    dataset_root: Path,
    pp: PreprocessConfig,
    windows_sec: list[list[float]],
    return_stats: bool = False,
) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
    rows = []

    stats = {
        "raw_fail": 0,
        "events_missing": 0,
        "events_empty_after_filter": 0,
        "label_fail": 0,
        "no_onset": 0,
        "one_class": 0,
        "too_few_per_class": 0,
        "bad_samples": 0,
        "epoch_fail": 0,
        "epoch_zero": 0,
        "ok": 0,
    }

    for ent in _iter_bids_combos(dataset_root):
        bp = BIDSPath(
            root=dataset_root,
            subject=ent["subject"],
            session=ent["session"],
            task=ent["task"],
            run=ent["run"],
            datatype="eeg",
        )

        # --- raw ---
        try:
            raw = _read_raw_no_participants(bp)
        except Exception:
            stats["raw_fail"] += 1
            continue

        # --- events ---
        try:
            events_df = _load_events_tsv(bp)
        except Exception:
            stats["events_missing"] += 1
            continue

        if "onset" not in events_df.columns and "sample" not in events_df.columns:
            stats["no_onset"] += 1
            continue

        events_df = _filter_to_stimulus(events_df, dsid=dsid)
        if len(events_df) == 0:
            stats["events_empty_after_filter"] += 1
            continue

        # --- labels ---
        try:
            labels = _make_labels(events_df, dsid=dsid)
        except Exception:
            stats["label_fail"] += 1
            continue

        keep = labels.notna()
        events_df = events_df.loc[keep].reset_index(drop=True)
        labels = labels.loc[keep].astype(int).reset_index(drop=True)
        if len(events_df) == 0:
            stats["label_fail"] += 1
            continue

        n_t = int(labels.sum())
        n_nt = int((labels == 0).sum())
        if n_t == 0 or n_nt == 0:
            stats["one_class"] += 1
            continue
        if n_t < pp.min_per_class or n_nt < pp.min_per_class:
            stats["too_few_per_class"] += 1
            continue

        # --- preprocess ---
        raw.load_data()
        raw = _prep_channels(raw)
        raw.filter(pp.l_freq, pp.h_freq, verbose=False)

        sfreq = float(raw.info["sfreq"])

        # Prefer explicit sample column when present and numeric
        if "sample" in events_df.columns and pd.api.types.is_numeric_dtype(events_df["sample"]):
            sample = events_df["sample"].round().astype(int).to_numpy()
        else:
            onsets_s = events_df["onset"].astype(float).to_numpy()
            sample = np.round(onsets_s * sfreq).astype(int)

        if np.any(sample < 0) or np.any(sample >= raw.n_times):
            stats["bad_samples"] += 1
            continue

        # Build MNE events correctly (FIXED ordering bug)
        mne_events = np.c_[sample, np.zeros(len(sample), dtype=int), labels.to_numpy() + 1]
        codes = set(mne_events[:, 2].tolist())
        if 1 not in codes or 2 not in codes:
            stats["one_class"] += 1
            continue

        event_id = {"nontarget": 1, "target": 2}

        try:
            epochs = mne.Epochs(
                raw,
                mne_events,
                event_id=event_id,
                tmin=pp.tmin,
                tmax=pp.tmax,
                baseline=pp.baseline,
                preload=True,
                reject_by_annotation=False,  # keep data despite boundary annotations
                on_missing="ignore",
                verbose=False,
            )
        except Exception:
            stats["epoch_fail"] += 1
            continue

        if len(epochs) == 0:
            stats["epoch_zero"] += 1
            continue

        if pp.resample_sfreq:
            epochs.resample(pp.resample_sfreq, npad="auto", verbose=False)

        X = erp_window_features(epochs, windows_sec)

        meta = pd.DataFrame(
            {
                "dsid": dsid,
                "subject": ent["subject"],
                "session": ent["session"] if ent["session"] is not None else "",
                "task": ent["task"] if ent["task"] is not None else "",
                "run": ent["run"] if ent["run"] is not None else "",
                "trial_idx": np.arange(len(epochs)),
                "label": (epochs.events[:, 2] - 1).astype(int),  # back to 0/1
            }
        )

        rows.append(pd.concat([meta, X], axis=1))
        stats["ok"] += 1

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return (df, stats) if return_stats else df
