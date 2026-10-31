"""Run one (engine, tier, query) in this process and print JSON. Called as a fresh subprocess so
memory and caches are not shared between measurements."""
from __future__ import annotations

import argparse
import json
import os
import time

from . import engines
from .data import DATA_DIR, files_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("tier")
    ap.add_argument("query")
    ap.add_argument("--out", help="save the result as parquet so engines can be compared")
    a = ap.parse_args()
    files = files_for(a.tier)
    zones = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")
    t0 = time.perf_counter()
    df = engines.run(a.engine, a.query, files, zones)
    secs = time.perf_counter() - t0
    if a.out:
        df.to_parquet(a.out)
    print(json.dumps({"engine": a.engine, "tier": a.tier, "query": a.query, "seconds": round(secs, 3), "result_rows": len(df)}))


if __name__ == "__main__":
    main()
