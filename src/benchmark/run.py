"""CLI entry point: generate data, run benchmarks, produce report."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any

from benchmark.datasets import (
    SPECS,
    load_pandas,
    load_polars,
    save_parquet,
)
from benchmark.operations import (
    OP_REGISTRY,
    get_anti_exclude,
    make_lookups,
)
from benchmark.runner import MachineInfo, TimingResult, run_benchmark
from benchmark.visualize import (
    plot_grouped_bar,
    plot_heatmap,
    results_to_json,
)

DATASET_FILTER: dict[str, list[str]] = {
    "small": ["small", "small_nulls", "small_high_card"],
    "medium": ["medium", "medium_nulls", "medium_high_card"],
    "large": ["large", "large_nulls", "large_high_card"],
    "all": [s.name for s in SPECS],
}

SIMPLE_OPS = [
    "groupby",
    "groupby_high_card",
    "window_rank",
    "window_running_sum",
    "filter_sort",
]

PATH_OPS = ["read_parquet"]

JOIN_OPS = ["join_inner", "join_left"]

ENGINES = ["polars", "pandas", "duckdb"]


def _dispatch(
    op: str,
    engine: str,
    pl_df: Any,
    pd_df: Any,
    path: Path,
    pl_lookup: Any,
    pd_lookup: Any,
    pl_exclude: Any,
    pd_exclude: Any,
    exclude_keys: list[str],
) -> Any:
    func = OP_REGISTRY[op][engine]

    if op in PATH_OPS:
        return func(path)

    if op in SIMPLE_OPS:
        if engine == "duckdb":
            return func(path)
        return func(pl_df) if engine == "polars" else func(pd_df)

    if op in JOIN_OPS:
        if engine == "duckdb":
            return func(path)
        if engine == "polars":
            return func(pl_df, pl_lookup)
        return func(pd_df, pd_lookup)

    if op == "join_anti":
        if engine == "duckdb":
            return func(path, exclude_keys)
        if engine == "polars":
            return func(pl_df, pl_exclude)
        return func(pd_df, pd_exclude)

    msg = f"Unknown operation: {op}"
    raise ValueError(msg)


def run_all(
    data_dir: Path,
    results_dir: Path,
    size: str = "small",
    rounds: int = 7,
) -> list[TimingResult]:
    dataset_names = DATASET_FILTER.get(size, [size])
    specs = [s for s in SPECS if s.name in dataset_names]
    all_ops = SIMPLE_OPS + JOIN_OPS + ["join_anti"] + PATH_OPS

    all_results: list[TimingResult] = []

    for spec in specs:
        print(f"\n=== Dataset: {spec.name} ({spec.n_rows:,} rows) ===")
        path = save_parquet(spec, data_dir)
        pl_df = load_polars(path)
        pd_df = load_pandas(path)
        n = spec.n_rows

        pl_lookup, pd_lookup = make_lookups(pl_df, pd_df)
        pl_exclude, pd_exclude, exclude_keys = get_anti_exclude(pl_df, pd_df)

        for op in all_ops:
            for engine in ENGINES:
                print(
                    f"  {op:25s} {engine:8s}",
                    end=" ",
                    flush=True,
                )
                bench_fn = functools.partial(
                    _dispatch,
                    op,
                    engine,
                    pl_df,
                    pd_df,
                    path,
                    pl_lookup,
                    pd_lookup,
                    pl_exclude,
                    pd_exclude,
                    exclude_keys,
                )
                result = run_benchmark(
                    op, engine, spec.name, bench_fn, n, rounds=rounds
                )
                all_results.append(result)
                print(
                    f"{result.median_seconds * 1000:8.1f} ms"
                    f" (IQR {result.iqr_seconds * 1000:.1f} ms)"
                )

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "all"],
        default="small",
    )
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    machine = MachineInfo()
    print(f"Machine: {machine.system} {machine.machine}")
    print(f"CPU: {machine.cpu_count} cores ({machine.cpu_count_logical} logical)")
    print(f"RAM: {machine.ram_gb} GB")
    print(f"Python: {machine.python_version}")

    results = run_all(args.data_dir, args.results_dir, args.size, args.rounds)

    results_to_json(results, machine, args.results_dir / "results.json")
    plot_grouped_bar(results, machine, args.results_dir)
    plot_heatmap(results, machine, args.results_dir)

    print(f"\nResults saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
