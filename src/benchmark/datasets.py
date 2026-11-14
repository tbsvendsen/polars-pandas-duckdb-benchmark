"""Generate deterministic benchmark datasets."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from faker import Faker

SEED = 42


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_rows: int
    cardinality: str
    has_nulls: bool


SPECS: list[DatasetSpec] = [
    DatasetSpec("small", 100_000, "low", False),
    DatasetSpec("small_nulls", 100_000, "low", True),
    DatasetSpec("small_high_card", 100_000, "high", False),
    DatasetSpec("medium", 1_000_000, "low", False),
    DatasetSpec("medium_nulls", 1_000_000, "low", True),
    DatasetSpec("medium_high_card", 1_000_000, "high", False),
    DatasetSpec("large", 10_000_000, "low", False),
    DatasetSpec("large_nulls", 10_000_000, "low", True),
    DatasetSpec("large_high_card", 10_000_000, "high", False),
]


def generate_dataset(spec: DatasetSpec) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    fake = Faker(["en_US"])
    Faker.seed(SEED)

    n = spec.n_rows
    low_card = 20
    high_card = n // 10

    card = high_card if spec.cardinality == "high" else low_card

    group_values = [fake.company()[:20] for _ in range(min(card, 1000))]
    groups = rng.choice(group_values, size=n)

    ids = np.arange(1, n + 1)
    amounts = rng.uniform(1.0, 10_000.0, size=n).round(2)
    quantities = rng.integers(1, 100, size=n)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    date_col = rng.choice(dates, size=n)

    if spec.cardinality == "high":
        group_key_2 = rng.integers(0, high_card, size=n).astype(str)
    else:
        group_key_2 = rng.choice([f"cat_{i}" for i in range(5)], size=n)

    data: dict[str, Any] = {
        "id": ids,
        "group_key": groups,
        "group_key_2": group_key_2,
        "amount": amounts,
        "quantity": quantities,
        "date": date_col,
    }

    if spec.has_nulls:
        null_mask = rng.random(n) < 0.05
        amounts_with_nulls = amounts.astype(object)
        amounts_with_nulls[null_mask] = None
        data["amount"] = amounts_with_nulls

    return data


def save_parquet(spec: DatasetSpec, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = generate_dataset(spec)
    df = pl.DataFrame(data)
    path = output_dir / f"{spec.name}.parquet"
    df.write_parquet(path)
    gc.collect()
    return path


def load_polars(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_pandas(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
