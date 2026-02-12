from __future__ import annotations
from pathlib import Path
import pandas as pd

def append_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    out.to_parquet(path, index=False)
