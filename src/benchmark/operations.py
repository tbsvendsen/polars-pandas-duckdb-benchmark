"""Benchmark operations for each engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import polars as pl

# --- Groupby low cardinality ---


def groupby_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("group_key").agg(
        pl.col("amount").sum().alias("total"),
        pl.col("amount").mean().alias("avg"),
        pl.col("quantity").sum().alias("qty_sum"),
    )


def groupby_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("group_key", as_index=False).agg(
        total=("amount", "sum"),
        avg=("amount", "mean"),
        qty_sum=("quantity", "sum"),
    )


def groupby_duckdb(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"""
        SELECT group_key,
               SUM(amount) AS total,
               AVG(amount) AS avg,
               SUM(quantity) AS qty_sum
        FROM read_parquet('{path}')
        GROUP BY group_key
    """).fetchdf()


# --- Groupby high cardinality ---


def groupby_high_card_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("group_key_2").agg(
        pl.col("amount").sum().alias("total"),
        pl.len().alias("cnt"),
    )


def groupby_high_card_pandas(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("group_key_2", as_index=False).agg(
        total=("amount", "sum"), cnt=("amount", "count")
    )


def groupby_high_card_duckdb(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"""
        SELECT group_key_2,
               SUM(amount) AS total,
               COUNT(*) AS cnt
        FROM read_parquet('{path}')
        GROUP BY group_key_2
    """).fetchdf()


# --- Join ---


def _make_lookup_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("group_key").unique().with_row_index("lookup_id")


def _make_lookup_pandas(df: pd.DataFrame) -> pd.DataFrame:
    lookup = df[["group_key"]].drop_duplicates().reset_index(drop=True)
    lookup["lookup_id"] = range(len(lookup))
    return lookup


def join_inner_polars(df: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    return df.join(lookup, on="group_key", how="inner")


def join_inner_pandas(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    return df.merge(lookup, on="group_key", how="inner")


def join_inner_duckdb(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"""
        WITH lookup AS (
            SELECT DISTINCT group_key, ROW_NUMBER() OVER () AS lookup_id
            FROM read_parquet('{path}')
        )
        SELECT t.*, l.lookup_id
        FROM read_parquet('{path}') t
        INNER JOIN lookup l ON t.group_key = l.group_key
    """).fetchdf()


def join_left_polars(df: pl.DataFrame, lookup: pl.DataFrame) -> pl.DataFrame:
    return df.join(lookup, on="group_key", how="left")


def join_left_pandas(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    return df.merge(lookup, on="group_key", how="left")


def join_left_duckdb(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"""
        WITH lookup AS (
            SELECT DISTINCT group_key, ROW_NUMBER() OVER () AS lookup_id
            FROM read_parquet('{path}')
        )
        SELECT t.*, l.lookup_id
        FROM read_parquet('{path}') t
        LEFT JOIN lookup l ON t.group_key = l.group_key
    """).fetchdf()


def join_anti_polars(df: pl.DataFrame, exclude: pl.DataFrame) -> pl.DataFrame:
    return df.join(exclude, on="group_key", how="anti")


def join_anti_pandas(df: pd.DataFrame, exclude: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(exclude, on="group_key", how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge")


def join_anti_duckdb(path: Path, exclude_keys: list[str]) -> pd.DataFrame:
    keys_str = ", ".join(f"'{k}'" for k in exclude_keys)
    return duckdb.sql(f"""
        SELECT *
        FROM read_parquet('{path}')
        WHERE group_key NOT IN ({keys_str})
    """).fetchdf()


# --- Window functions ---


def window_rank_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("amount")
        .rank(method="ordinal", descending=True)
        .over("group_key")
        .alias("rank")
    )


def window_rank_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = df.groupby("group_key")["amount"].rank(method="first", ascending=False)
    return df


def window_rank_duckdb(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"""
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY group_key ORDER BY amount DESC) AS rank
        FROM read_parquet('{path}')
    """).fetchdf()


def window_running_sum_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort("date", "id").with_columns(
        pl.col("amount").cum_sum().over("group_key").alias("running_total")
    )


def window_running_sum_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "id"]).copy()
    df["running_total"] = df.groupby("group_key")["amount"].cumsum()
    return df


def window_running_sum_duckdb(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"""
        SELECT *,
               SUM(amount) OVER (
                   PARTITION BY group_key ORDER BY date, id
                   ROWS UNBOUNDED PRECEDING
               ) AS running_total
        FROM read_parquet('{path}')
    """).fetchdf()


# --- Filter + sort ---


def filter_sort_polars(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("amount") > 500) & (pl.col("quantity") > 10)).sort(
        "amount", descending=True
    )


def filter_sort_pandas(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[(df["amount"] > 500) & (df["quantity"] > 10)]
    return filtered.sort_values("amount", ascending=False)


def filter_sort_duckdb(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"""
        SELECT *
        FROM read_parquet('{path}')
        WHERE amount > 500 AND quantity > 10
        ORDER BY amount DESC
    """).fetchdf()


# --- Parquet read ---


def read_parquet_polars(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def read_parquet_pandas(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_parquet_duckdb(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{path}')").fetchdf()


# --- Helpers ---


def make_lookups(
    pl_df: pl.DataFrame, pd_df: pd.DataFrame
) -> tuple[pl.DataFrame, pd.DataFrame]:
    return _make_lookup_polars(pl_df), _make_lookup_pandas(pd_df)


def get_anti_exclude(
    pl_df: pl.DataFrame, pd_df: pd.DataFrame, n: int = 5
) -> tuple[pl.DataFrame, pd.DataFrame, list[str]]:
    keys = pl_df.select("group_key").unique().head(n)
    exclude_keys: list[str] = keys["group_key"].to_list()
    pd_exclude = pd_df[pd_df["group_key"].isin(exclude_keys)][["group_key"]]
    return keys, pd_exclude, exclude_keys


OP_REGISTRY: dict[str, dict[str, Any]] = {
    "groupby": {
        "polars": groupby_polars,
        "pandas": groupby_pandas,
        "duckdb": groupby_duckdb,
    },
    "groupby_high_card": {
        "polars": groupby_high_card_polars,
        "pandas": groupby_high_card_pandas,
        "duckdb": groupby_high_card_duckdb,
    },
    "join_inner": {
        "polars": join_inner_polars,
        "pandas": join_inner_pandas,
        "duckdb": join_inner_duckdb,
    },
    "join_left": {
        "polars": join_left_polars,
        "pandas": join_left_pandas,
        "duckdb": join_left_duckdb,
    },
    "join_anti": {
        "polars": join_anti_polars,
        "pandas": join_anti_pandas,
        "duckdb": join_anti_duckdb,
    },
    "window_rank": {
        "polars": window_rank_polars,
        "pandas": window_rank_pandas,
        "duckdb": window_rank_duckdb,
    },
    "window_running_sum": {
        "polars": window_running_sum_polars,
        "pandas": window_running_sum_pandas,
        "duckdb": window_running_sum_duckdb,
    },
    "filter_sort": {
        "polars": filter_sort_polars,
        "pandas": filter_sort_pandas,
        "duckdb": filter_sort_duckdb,
    },
    "read_parquet": {
        "polars": read_parquet_polars,
        "pandas": read_parquet_pandas,
        "duckdb": read_parquet_duckdb,
    },
}
