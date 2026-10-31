"""Turn results/results.csv into results/summary.md and results/benchmark.png."""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ENGINES = ["duckdb", "polars", "pandas", "spark"]
COLORS = {"duckdb": "#0F766E", "polars": "#4F6BED", "pandas": "#B45309", "spark": "#6B7280"}


def load(path: str = "results/results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["median_s"] = pd.to_numeric(df["median_s"], errors="coerce")
    df["peak_gb"] = pd.to_numeric(df["peak_gb"], errors="coerce")
    return df


def totals(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of the four query medians per engine and tier (only where all four succeeded)."""
    ok = df[df["status"] == "ok"]
    g = ok.groupby(["tier", "engine"]).agg(total_s=("median_s", "sum"), n=("query", "nunique"), peak_gb=("peak_gb", "max")).reset_index()
    return g[g["n"] == 4]


def markdown(df: pd.DataFrame, rows_by_tier: dict[str, int]) -> str:
    out = ["# Results", ""]
    t = totals(df)
    out += ["## Total time for the four queries (seconds, median of runs; lower is better)", "",
            "| Dataset | Rows | " + " | ".join(ENGINES) + " |", "|---|---:|" + "---:|" * len(ENGINES)]
    for tier in ["1m", "6m", "12m", "24m"]:
        cells = []
        for e in ENGINES:
            r = t[(t.tier == tier) & (t.engine == e)]
            cells.append(f"{r.total_s.iloc[0]:.1f}" if len(r) else "failed")
        out.append(f"| {tier.replace('m', ' months')} | {rows_by_tier.get(tier, 0):,} | " + " | ".join(cells) + " |")
    out += ["", "## Peak memory (GB, whole process tree, worst query)", "",
            "| Dataset | " + " | ".join(ENGINES) + " |", "|---|" + "---:|" * len(ENGINES)]
    for tier in ["1m", "6m", "12m", "24m"]:
        cells = []
        for e in ENGINES:
            r = t[(t.tier == tier) & (t.engine == e)]
            cells.append(f"{r.peak_gb.iloc[0]:.1f}" if len(r) else "failed")
        out.append(f"| {tier.replace('m', ' months')} | " + " | ".join(cells) + " |")
    out += ["", "## Per query at the largest size", "", "| Query | " + " | ".join(ENGINES) + " |", "|---|" + "---:|" * len(ENGINES)]
    big = df[df.tier == "24m"]
    for q in big["query"].unique():
        cells = []
        for e in ENGINES:
            r = big[(big["query"] == q) & (big.engine == e)]
            cells.append(f"{r.median_s.iloc[0]:.2f}" if len(r) and r.status.iloc[0] == "ok" else "failed")
        out.append(f"| {q} | " + " | ".join(cells) + " |")
    bad = df[(df.engine != "duckdb") & (df.status == "ok") & (df.agrees_with_duckdb.astype(str) != "True")]
    out += ["", "## Correctness", "",
            "Every engine's answer was compared with DuckDB's for every query and size (numbers within 1e-6 relative). "
            + ("All answers agreed." if bad.empty else f"{len(bad)} result(s) did NOT agree: " + ", ".join(f"{r.engine}/{r.tier}/{r['query']}" for _, r in bad.iterrows()))]
    return "\n".join(out) + "\n"


def chart(df: pd.DataFrame, path: str = "results/benchmark.png") -> None:
    t = totals(df)
    tiers = ["1m", "6m", "12m", "24m"]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    for e in ENGINES:
        s = [t[(t.tier == x) & (t.engine == e)].total_s for x in tiers]
        ys = [v.iloc[0] if len(v) else float("nan") for v in s]
        ax.plot(range(len(tiers)), ys, marker="o", lw=2, color=COLORS[e], label=e)
        if not pd.isna(ys[-1]):
            ax.annotate(f"{ys[-1]:.0f}s" if ys[-1] >= 10 else f"{ys[-1]:.1f}s", (len(tiers) - 1, ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=COLORS[e], fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(["1 month", "6 months", "12 months", "24 months"])
    ax.set_ylabel("Seconds for all four queries (log scale)")
    ax.set_title("Same queries, same data, four engines (NYC taxi trips)", loc="left", fontsize=11)
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_xlim(-0.2, len(tiers) - 0.4)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    import duckdb

    from .data import files_for
    d = load(sys.argv[1] if len(sys.argv) > 1 else "results/results.csv")
    rows = {}
    for tier in ["1m", "6m", "12m", "24m"]:
        fl = ", ".join(f"'{f}'" for f in files_for(tier))
        rows[tier] = duckdb.connect().execute(f"select count(*) from read_parquet([{fl}], union_by_name=true) where tpep_pickup_datetime >= '2023-01-01' and tpep_pickup_datetime < '2025-01-01'").fetchone()[0]
    open("results/summary.md", "w").write(markdown(d, rows))
    chart(d)
    print(open("results/summary.md").read())
