"""Tests that all operations produce valid results and agree across engines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from benchmark.operations import (
    filter_sort_duckdb,
    filter_sort_pandas,
    filter_sort_polars,
    get_anti_exclude,
    groupby_duckdb,
    groupby_pandas,
    groupby_polars,
    join_anti_duckdb,
    join_anti_pandas,
    join_anti_polars,
    join_inner_duckdb,
    join_inner_pandas,
    join_inner_polars,
    join_left_pandas,
    join_left_polars,
    make_lookups,
    read_parquet_duckdb,
    read_parquet_pandas,
    read_parquet_polars,
    window_rank_duckdb,
    window_rank_pandas,
    window_rank_polars,
    window_running_sum_duckdb,
    window_running_sum_pandas,
    window_running_sum_polars,
)


class TestGroupby:
    def test_polars_returns_results(self, pl_df: pl.DataFrame) -> None:
        result = groupby_polars(pl_df)
        assert result.height > 0
        assert "total" in result.columns

    def test_pandas_returns_results(self, pd_df: pd.DataFrame) -> None:
        result = groupby_pandas(pd_df)
        assert len(result) > 0
        assert "total" in result.columns

    def test_duckdb_returns_results(self, small_parquet: Path) -> None:
        result = groupby_duckdb(small_parquet).fetchdf()
        assert len(result) > 0

    def test_engines_agree_on_group_count(
        self, pl_df: pl.DataFrame, pd_df: pd.DataFrame, small_parquet: Path
    ) -> None:
        pl_count = groupby_polars(pl_df).height
        pd_count = len(groupby_pandas(pd_df))
        duck_count = len(groupby_duckdb(small_parquet).fetchdf())
        assert pl_count == pd_count == duck_count


class TestJoin:
    def test_inner_join_all_engines(
        self, pl_df: pl.DataFrame, pd_df: pd.DataFrame, small_parquet: Path
    ) -> None:
        pl_lookup, pd_lookup = make_lookups(pl_df, pd_df)
        pl_result = join_inner_polars(pl_df, pl_lookup)
        pd_result = join_inner_pandas(pd_df, pd_lookup)
        duck_result = join_inner_duckdb(small_parquet).fetchdf()
        assert pl_result.height > 0
        assert len(pd_result) > 0
        assert len(duck_result) > 0

    def test_left_join_preserves_row_count(
        self, pl_df: pl.DataFrame, pd_df: pd.DataFrame
    ) -> None:
        pl_lookup, pd_lookup = make_lookups(pl_df, pd_df)
        assert join_left_polars(pl_df, pl_lookup).height == pl_df.height
        assert len(join_left_pandas(pd_df, pd_lookup)) == len(pd_df)

    def test_anti_join_removes_rows(
        self, pl_df: pl.DataFrame, pd_df: pd.DataFrame, small_parquet: Path
    ) -> None:
        pl_exclude, pd_exclude, exclude_keys = get_anti_exclude(pl_df, pd_df)
        pl_result = join_anti_polars(pl_df, pl_exclude)
        pd_result = join_anti_pandas(pd_df, pd_exclude)
        duck_result = join_anti_duckdb(small_parquet, exclude_keys).fetchdf()
        assert pl_result.height < pl_df.height
        assert len(pd_result) < len(pd_df)
        assert len(duck_result) < len(pd_df)


class TestWindowFunctions:
    def test_rank_adds_column(self, pl_df: pl.DataFrame, pd_df: pd.DataFrame) -> None:
        pl_result = window_rank_polars(pl_df)
        pd_result = window_rank_pandas(pd_df)
        assert "rank" in pl_result.columns
        assert "rank" in pd_result.columns

    def test_running_sum_adds_column(
        self, pl_df: pl.DataFrame, pd_df: pd.DataFrame
    ) -> None:
        pl_result = window_running_sum_polars(pl_df)
        pd_result = window_running_sum_pandas(pd_df)
        assert "running_total" in pl_result.columns
        assert "running_total" in pd_result.columns

    def test_duckdb_window_operations(self, small_parquet: Path) -> None:
        rank_result = window_rank_duckdb(small_parquet).fetchdf()
        sum_result = window_running_sum_duckdb(small_parquet).fetchdf()
        assert "rank" in rank_result.columns
        assert "running_total" in sum_result.columns


class TestFilterSort:
    def test_filters_reduce_rows(
        self, pl_df: pl.DataFrame, pd_df: pd.DataFrame
    ) -> None:
        pl_result = filter_sort_polars(pl_df)
        pd_result = filter_sort_pandas(pd_df)
        assert pl_result.height <= pl_df.height
        assert len(pd_result) <= len(pd_df)

    def test_duckdb_filter_sort(self, small_parquet: Path) -> None:
        result = filter_sort_duckdb(small_parquet).fetchdf()
        assert len(result) >= 0


class TestParquetRead:
    def test_all_engines_read_same_count(self, small_parquet: Path) -> None:
        pl_result = read_parquet_polars(small_parquet)
        pd_result = read_parquet_pandas(small_parquet)
        duck_result = read_parquet_duckdb(small_parquet).fetchdf()
        assert pl_result.height == len(pd_result) == len(duck_result)
