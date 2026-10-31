"""All engines must return the same answers on a small, deliberately awkward dataset:
mixed column types between files, rows with impossible dates, null passengers and a borough called 'N/A'."""
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tbench import engines

HAVE_JAVA = shutil.which("java") is not None


def make_file(path, n, pax_float, start="2024-01-01"):
    ts = pd.date_range(start, periods=n, freq="37min")
    t = pa.table({
        "tpep_pickup_datetime": pa.array(ts.to_pydatetime(), pa.timestamp("us")),
        "PULocationID": pa.array([(i % 5) + 1 for i in range(n)], pa.int64() if pax_float else pa.int32()),
        "passenger_count": pa.array([None if i % 11 == 0 else float(i % 4) if pax_float else i % 4 for i in range(n)],
                                    pa.float64() if pax_float else pa.int64()),
        "fare_amount": [float((i % 9) - 1) + 5.5 for i in range(n)],
        "tip_amount": [float(i % 4) for i in range(n)],
        "total_amount": [float(i % 7) + 10.25 for i in range(n)],
    })
    pq.write_table(t, path)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("d")
    make_file(d / "a.parquet", 3000, pax_float=False, start="2023-12-01")
    make_file(d / "b.parquet", 3000, pax_float=True, start="2024-01-01")
    # impossible dates must be filtered by every engine
    pq.write_table(pa.table({
        "tpep_pickup_datetime": pa.array([pd.Timestamp("2001-01-01").to_pydatetime(), pd.Timestamp("2026-06-01").to_pydatetime()], pa.timestamp("us")),
        "PULocationID": pa.array([1, 2], pa.int64()), "passenger_count": pa.array([1.0, 1.0]),
        "fare_amount": [5.0, 5.0], "tip_amount": [1.0, 1.0], "total_amount": [9.0, 9.0]}), d / "c.parquet")
    (d / "zones.csv").write_text('"LocationID","Borough","Zone","service_zone"\n1,"Bronx","a","x"\n2,"N/A","b","x"\n3,"Queens","c","x"\n4,"Unknown","d","x"\n5,"Manhattan","e","x"\n')
    return [str(d / f) for f in ("a.parquet", "b.parquet", "c.parquet")], str(d / "zones.csv")


def same(a, b):
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for c in a.columns:
        if a[c].dtype.kind == "f":
            pd.testing.assert_series_equal(a[c], b[c], rtol=1e-9, check_names=False)
        else:
            assert (a[c].values == b[c].values).all(), c


@pytest.mark.parametrize("q", engines.QUERIES)
@pytest.mark.parametrize("eng", ["polars", "pandas"])
def test_light_engines_match_duckdb(data, eng, q):
    files, zones = data
    same(engines.run(eng, q, files, zones), engines.run("duckdb", q, files, zones))


def test_bad_dates_are_excluded(data):
    files, zones = data
    r = engines.run("duckdb", "q1_groupby", files, zones)
    assert r["n"].sum() == 6000


def test_na_borough_is_kept(data):
    files, zones = data
    for eng in ("duckdb", "polars", "pandas"):
        assert "N/A" in set(engines.run(eng, "q3_join", files, zones)["borough"]), eng


@pytest.mark.skipif(not HAVE_JAVA, reason="Spark needs Java")
@pytest.mark.parametrize("q", engines.QUERIES)
def test_spark_matches_duckdb(data, q):
    files, zones = data
    same(engines.run("spark", q, files, zones), engines.run("duckdb", q, files, zones))
