"""Dataset tiers: the same files, growing in size."""
from __future__ import annotations

import glob
import os

DATA_DIR = os.getenv("DATA_DIR", "data")


def files_for(tier: str, data_dir: str = DATA_DIR) -> list[str]:
    allf = sorted(glob.glob(os.path.join(data_dir, "yellow_tripdata_*.parquet")))
    pick = {
        "1m": lambda f: "2024-01" in f,
        "6m": lambda f: any(f"2024-0{m}" in f for m in range(1, 7)),
        "12m": lambda f: "2024-" in f,
        "24m": lambda f: True,
    }[tier]
    return [f for f in allf if pick(f)]


TIERS = ["1m", "6m", "12m", "24m"]
