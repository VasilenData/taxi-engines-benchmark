"""Dataset tiers: the same files, growing in size."""
from __future__ import annotations

import os

DATA_DIR = os.getenv("DATA_DIR", "data")


def files_for(tier: str, data_dir: str = DATA_DIR) -> list[str]:
    months = {
        "1m": ["2024-01"],
        "6m": [f"2024-{month:02d}" for month in range(1, 7)],
        "12m": [f"2024-{month:02d}" for month in range(1, 13)],
        "24m": [f"{year}-{month:02d}" for year in (2023, 2024) for month in range(1, 13)],
    }
    if tier not in months:
        raise ValueError(f"unknown tier {tier!r}")
    files = [os.path.join(data_dir, f"yellow_tripdata_{month}.parquet") for month in months[tier]]
    missing = [os.path.basename(path) for path in files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Incomplete {tier} tier; missing: {', '.join(missing)}")
    return files


TIERS = ["1m", "6m", "12m", "24m"]
