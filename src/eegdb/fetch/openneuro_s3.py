from __future__ import annotations
from pathlib import Path
import subprocess

BIDS_EEG_INCLUDES = [
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "*_events.json",
    "sub-*/eeg/*_eeg.*",
    "sub-*/eeg/*_eeg.json",
    "sub-*/eeg/*_channels.tsv",
    "sub-*/eeg/*_events.tsv",
]

def s3_sync_openneuro_dataset(dsid: str, out_dir: Path, minimal_eeg_only: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "sync",
        "--no-sign-request",
        f"s3://openneuro.org/{dsid}",
        str(out_dir),
    ]
    if minimal_eeg_only:
        cmd += ["--exclude", "*"]
        for pat in BIDS_EEG_INCLUDES:
            cmd += ["--include", pat]
    subprocess.run(cmd, check=True)
