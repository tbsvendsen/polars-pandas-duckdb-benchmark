"""Shared fixtures for benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from benchmark.datasets import DatasetSpec, load_pandas, load_polars, save_parquet

SMALL_SPEC = DatasetSpec("test_small", 1_000, "low", False)
SMALL_NULLS_SPEC = DatasetSpec("test_small_nulls", 1_000, "low", True)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("benchmark_data")


@pytest.fixture(scope="session")
def small_parquet(data_dir: Path) -> Path:
    return save_parquet(SMALL_SPEC, data_dir)


@pytest.fixture(scope="session")
def small_nulls_parquet(data_dir: Path) -> Path:
    return save_parquet(SMALL_NULLS_SPEC, data_dir)


@pytest.fixture(scope="session")
def pl_df(small_parquet: Path) -> pl.DataFrame:
    return load_polars(small_parquet)


@pytest.fixture(scope="session")
def pd_df(small_parquet: Path) -> pd.DataFrame:
    return load_pandas(small_parquet)
