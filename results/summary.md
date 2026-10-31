# Results

## Total time for the four queries (seconds, median of runs; lower is better)

| Dataset | Rows | duckdb | polars | pandas | spark |
|---|---:|---:|---:|---:|---:|
| 1 months | 2,964,619 | 0.5 | 0.5 | 1.4 | 21.9 |
| 6 months | 20,332,067 | 0.7 | 0.7 | 8.6 | 26.5 |
| 12 months | 41,169,674 | 0.9 | 1.0 | 17.1 | 30.9 |
| 24 months | 79,479,802 | 1.4 | 1.6 | 32.8 | 36.6 |

## Peak memory (GB, whole process tree, worst query)

| Dataset | duckdb | polars | pandas | spark |
|---|---:|---:|---:|---:|
| 1 months | 0.4 | 1.0 | 1.1 | 2.2 |
| 6 months | 0.6 | 2.2 | 5.2 | 7.8 |
| 12 months | 0.7 | 3.0 | 10.1 | 6.7 |
| 24 months | 0.7 | 4.3 | 19.4 | 8.2 |

## Per query at the largest size

| Query | duckdb | polars | pandas | spark |
|---|---:|---:|---:|---:|
| q1_groupby | 0.28 | 0.34 | 5.04 | 8.84 |
| q2_filter_derive | 0.37 | 0.42 | 8.52 | 9.35 |
| q3_join | 0.34 | 0.40 | 9.38 | 9.03 |
| q4_window | 0.38 | 0.41 | 9.90 | 9.37 |

## Correctness

Every engine's answer was compared with DuckDB's for every query and size (numbers within 1e-6 relative). All answers agreed.
