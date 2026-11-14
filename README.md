# polars-pandas-duckdb-benchmark

Systematic performance comparison of Polars, pandas and DuckDB on representative data operations.

## What it measures

| Operation | Variants |
|-----------|----------|
| Groupby | Low cardinality, high cardinality |
| Join | Inner, left, anti |
| Window functions | Rank, running sum |
| Filter + sort | Compound conditions |
| Parquet read | Cold start |

Each operation runs on three dataset sizes (100K, 1M, 10M rows) with variants for null handling and key cardinality.

## Methodology

- **Warmup** rounds before measurement
- **7 timed rounds** per operation
- **Median + IQR** (interquartile range) — not mean
- **GC disabled** during measurement
- **Answer validation** — all engines produce the same result
- **Machine specs** logged and shown in reports
- **Deterministic seed** — reproducible datasets

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m benchmark.run --size small
```

Results land in `results/` as interactive Plotly HTML reports.

## Run with larger datasets

```bash
python -m benchmark.run --size medium   # 1M rows
python -m benchmark.run --size large    # 10M rows
python -m benchmark.run --size all      # all sizes
```

## Run tests

```bash
pytest
```

## Tech stack

- **Polars** 1.x
- **pandas** 2.x
- **DuckDB** 1.x
- **Plotly** for interactive HTML reports
- **NumPy** for dataset generation
- **Faker** for realistic group keys

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions.

## License

MIT
