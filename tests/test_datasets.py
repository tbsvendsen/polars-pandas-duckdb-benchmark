"""Tests for dataset generation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from benchmark.datasets import DatasetSpec, generate_dataset, save_parquet


class TestDatasetGeneration:
    def test_deterministic_output(self) -> None:
        spec = DatasetSpec("det_test", 100, "low", False)
        data1 = generate_dataset(spec)
        data2 = generate_dataset(spec)
        assert list(data1["id"]) == list(data2["id"])
        assert list(data1["amount"]) == list(data2["amount"])

    def test_correct_row_count(self) -> None:
        spec = DatasetSpec("count_test", 500, "low", False)
        data = generate_dataset(spec)
        assert len(data["id"]) == 500

    def test_null_variant_has_nulls(self) -> None:
        spec = DatasetSpec("null_test", 10_000, "low", True)
        data = generate_dataset(spec)
        has_none = any(v is None for v in data["amount"])
        assert has_none

    def test_no_null_variant_clean(self) -> None:
        spec = DatasetSpec("clean_test", 10_000, "low", False)
        data = generate_dataset(spec)
        has_none = any(v is None for v in data["amount"])
        assert not has_none

    def test_save_and_load_parquet(self, tmp_path: Path) -> None:
        spec = DatasetSpec("io_test", 200, "low", False)
        path = save_parquet(spec, tmp_path)
        assert path.exists()
        df = pl.read_parquet(path)
        assert df.height == 200
        assert "id" in df.columns
        assert "group_key" in df.columns
        assert "amount" in df.columns

    def test_high_cardinality_has_many_groups(self) -> None:
        spec = DatasetSpec("high_card", 10_000, "high", False)
        data = generate_dataset(spec)
        unique_keys = len(set(data["group_key_2"]))
        assert unique_keys > 100
