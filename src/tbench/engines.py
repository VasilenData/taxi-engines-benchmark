"""One implementation per engine of the same four queries. Each returns a pandas DataFrame.

Normalised columns: pickup, pu (pickup zone id), pax (passengers), fare, tip, total.
Every engine casts per file first because the source Parquet files differ in column types
between months (for example passenger_count is int64 in some and double in others).
All engines apply the same base filter: pickups from 2023-01-01 up to 2025-01-01. The raw
data contains rows with impossible dates (some from 2001, some in the future).
"""
from __future__ import annotations

import pandas as pd

LO, HI = "2023-01-01", "2025-01-01"
COLS = ["tpep_pickup_datetime", "PULocationID", "passenger_count", "fare_amount", "tip_amount", "total_amount"]
QUERIES = ["q1_groupby", "q2_filter_derive", "q3_join", "q4_window"]
KEYS = {"q1_groupby": ["pu"], "q2_filter_derive": ["h", "pax"], "q3_join": ["borough"], "q4_window": ["month", "pu"]}


def _tidy(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """Same dtypes and order for every engine so results can be compared."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    for c in df.columns:
        if c in ("pu", "pax", "h", "n", "rk"):
            df[c] = df[c].astype("int64")
        elif c == "month":
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m")
        elif c not in ("borough",):
            df[c] = df[c].astype("float64")
    return df.sort_values(KEYS[q]).reset_index(drop=True)


# ---------------------------------------------------------------- DuckDB
def duckdb_run(q: str, files: list[str], zones: str) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect()
    flist = ", ".join(f"'{f}'" for f in files)
    con.execute(f"""CREATE VIEW t AS SELECT tpep_pickup_datetime AS pickup, CAST(PULocationID AS INTEGER) AS pu,
        CAST(passenger_count AS INTEGER) AS pax, fare_amount AS fare, tip_amount AS tip, total_amount AS total
        FROM read_parquet([{flist}], union_by_name=true)
        WHERE tpep_pickup_datetime >= '{LO}' AND tpep_pickup_datetime < '{HI}'""")
    con.execute(f"CREATE VIEW z AS SELECT LocationID, Borough FROM read_csv_auto('{zones}')")
    sql = {
        "q1_groupby": "SELECT pu, count(*) AS n, avg(fare) AS avg_fare, sum(total) AS revenue FROM t GROUP BY pu",
        "q2_filter_derive": """SELECT hour(pickup) AS h, pax, avg(tip / fare) AS tip_rate, count(*) AS n FROM t
                               WHERE fare > 0 AND tip >= 0 AND pax > 0 GROUP BY 1, 2""",
        "q3_join": """SELECT z.Borough AS borough, count(*) AS n, sum(total) AS revenue FROM t
                      JOIN z ON z.LocationID = t.pu GROUP BY 1""",
        "q4_window": """WITH m AS (SELECT date_trunc('month', pickup) AS month, pu, sum(total) AS revenue FROM t GROUP BY 1, 2),
                        r AS (SELECT *, rank() OVER (PARTITION BY month ORDER BY revenue DESC) AS rk FROM m)
                        SELECT month, pu, revenue, rk FROM r WHERE rk <= 3""",
    }[q]
    return con.execute(sql).df()


# ---------------------------------------------------------------- Polars
def polars_run(q: str, files: list[str], zones: str) -> pd.DataFrame:
    import polars as pl
    frames = [pl.scan_parquet(f).select(
        pl.col("tpep_pickup_datetime").cast(pl.Datetime("us")).alias("pickup"), pl.col("PULocationID").cast(pl.Int32).alias("pu"),
        pl.col("passenger_count").cast(pl.Int32).alias("pax"), pl.col("fare_amount").alias("fare"),
        pl.col("tip_amount").alias("tip"), pl.col("total_amount").alias("total")) for f in files]
    t = pl.concat(frames).filter((pl.col("pickup") >= pl.lit(LO).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("us"))) & (pl.col("pickup") < pl.lit(HI).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("us"))))
    if q == "q1_groupby":
        r = t.group_by("pu").agg(pl.len().alias("n"), pl.col("fare").mean().alias("avg_fare"), pl.col("total").sum().alias("revenue"))
    elif q == "q2_filter_derive":
        r = (t.filter((pl.col("fare") > 0) & (pl.col("tip") >= 0) & (pl.col("pax") > 0))
             .group_by(pl.col("pickup").dt.hour().alias("h"), "pax")
             .agg((pl.col("tip") / pl.col("fare")).mean().alias("tip_rate"), pl.len().alias("n")))
    elif q == "q3_join":
        z = pl.scan_csv(zones).select(pl.col("LocationID").cast(pl.Int32), pl.col("Borough").alias("borough"))
        r = t.join(z, left_on="pu", right_on="LocationID").group_by("borough").agg(pl.len().alias("n"), pl.col("total").sum().alias("revenue"))
    else:
        m = t.group_by(pl.col("pickup").dt.truncate("1mo").alias("month"), "pu").agg(pl.col("total").sum().alias("revenue"))
        r = (m.with_columns(pl.col("revenue").rank("min", descending=True).over("month").alias("rk"))
             .filter(pl.col("rk") <= 3))
    return r.collect().to_pandas()


# ---------------------------------------------------------------- pandas
def _pandas_load(files: list[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=COLS)
        d.columns = ["pickup", "pu", "pax", "fare", "tip", "total"]
        d["pu"] = d["pu"].astype("int32")
        d["pax"] = d["pax"].astype("float64")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    return df[(df["pickup"] >= LO) & (df["pickup"] < HI)]


def pandas_run(q: str, files: list[str], zones: str) -> pd.DataFrame:
    df = _pandas_load(files)
    if q == "q1_groupby":
        return df.groupby("pu").agg(n=("fare", "size"), avg_fare=("fare", "mean"), revenue=("total", "sum")).reset_index()
    if q == "q2_filter_derive":
        d = df[(df["fare"] > 0) & (df["tip"] >= 0) & (df["pax"] > 0)].copy()
        d["r"] = d["tip"] / d["fare"]
        d["h"] = d["pickup"].dt.hour
        return d.groupby(["h", "pax"]).agg(tip_rate=("r", "mean"), n=("r", "size")).reset_index()
    if q == "q3_join":
        z = pd.read_csv(zones, keep_default_na=False)[["LocationID", "Borough"]].rename(columns={"LocationID": "pu", "Borough": "borough"})
        z["pu"] = z["pu"].astype("int32")
        return df.merge(z, on="pu").groupby("borough").agg(n=("total", "size"), revenue=("total", "sum")).reset_index()
    m = df.assign(month=df["pickup"].dt.to_period("M").dt.to_timestamp()).groupby(["month", "pu"])["total"].sum().rename("revenue").reset_index()
    m["rk"] = m.groupby("month")["revenue"].rank(method="min", ascending=False)
    return m[m["rk"] <= 3]


# ---------------------------------------------------------------- Spark
def spark_run(q: str, files: list[str], zones: str) -> pd.DataFrame:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    spark = (SparkSession.builder.master("local[*]").config("spark.driver.memory", "16g")
             .config("spark.sql.session.timeZone", "UTC").config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    frames = [spark.read.parquet(f).select(
        F.col("tpep_pickup_datetime").alias("pickup"), F.col("PULocationID").cast("int").alias("pu"),
        F.col("passenger_count").cast("int").alias("pax"), F.col("fare_amount").alias("fare"),
        F.col("tip_amount").alias("tip"), F.col("total_amount").alias("total")) for f in files]
    t = frames[0]
    for f in frames[1:]:
        t = t.unionAll(f)
    t = t.filter((F.col("pickup") >= LO) & (F.col("pickup") < HI))
    t.createOrReplaceTempView("t")
    spark.read.option("header", True).csv(zones).select(F.col("LocationID").cast("int").alias("LocationID"), "Borough").createOrReplaceTempView("z")
    sql = {
        "q1_groupby": "SELECT pu, count(*) AS n, avg(fare) AS avg_fare, sum(total) AS revenue FROM t GROUP BY pu",
        "q2_filter_derive": """SELECT hour(pickup) AS h, pax, avg(tip / fare) AS tip_rate, count(*) AS n FROM t
                               WHERE fare > 0 AND tip >= 0 AND pax > 0 GROUP BY 1, 2""",
        "q3_join": "SELECT z.Borough AS borough, count(*) AS n, sum(total) AS revenue FROM t JOIN z ON z.LocationID = t.pu GROUP BY 1",
        "q4_window": """WITH m AS (SELECT date_trunc('month', pickup) AS month, pu, sum(total) AS revenue FROM t GROUP BY 1, 2),
                        r AS (SELECT *, rank() OVER (PARTITION BY month ORDER BY revenue DESC) AS rk FROM m)
                        SELECT month, pu, revenue, rk FROM r WHERE rk <= 3""",
    }[q]
    return spark.sql(sql).toPandas()


ENGINES = {"duckdb": duckdb_run, "polars": polars_run, "pandas": pandas_run, "spark": spark_run}


def run(engine: str, q: str, files: list[str], zones: str) -> pd.DataFrame:
    return _tidy(ENGINES[engine](q, files, zones), q)
