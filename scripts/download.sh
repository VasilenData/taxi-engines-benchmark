#!/usr/bin/env bash
# Download 24 months of NYC yellow-taxi trips (2023-2024, about 1.3 GB) and the zone lookup.
set -euo pipefail
DIR="${1:-data}"
mkdir -p "$DIR" && cd "$DIR"
for y in 2023 2024; do for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  echo "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_$y-$m.parquet"
done; done | xargs -P 8 -n 1 curl -sfLO
curl -sfLO https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
ls | wc -l
