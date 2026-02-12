# calls openneuro_s3 to download datasets from openneuro, as specified in the config file

from pathlib import Path
import yaml
from src.eegdb.fetch.openneuro_s3 import s3_sync_openneuro_dataset

def main():
    cfg = yaml.safe_load(Path("configs/p300_phase1.yaml").read_text())
    base = Path(cfg["storage"]["raw_openneuro_dir"])
    minimal = bool(cfg["download"]["minimal_eeg_only"])

    for ds in cfg["phase1_datasets"]:
        dsid = ds["id"]
        out = base / dsid
        print(f"[fetch] {dsid} -> {out}")
        s3_sync_openneuro_dataset(dsid, out, minimal_eeg_only=minimal)

if __name__ == "__main__":
    main()


