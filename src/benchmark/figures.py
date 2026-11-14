"""Generate static matplotlib figures from benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ENGINE_COLORS: dict[str, str] = {
    "polars": "#4C78A8",
    "pandas": "#F58518",
    "duckdb": "#54A24B",
}

ENGINE_ORDER = ["polars", "pandas", "duckdb"]

OP_LABELS: dict[str, str] = {
    "groupby": "Group By\n(low card)",
    "groupby_high_card": "Group By\n(high card)",
    "join_inner": "Inner Join",
    "join_left": "Left Join",
    "join_anti": "Anti Join",
    "window_rank": "Window\nRank",
    "window_running_sum": "Window\nRunning Sum",
    "filter_sort": "Filter +\nSort",
    "read_parquet": "Read\nParquet",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )


def load_results(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(path.read_text())
    return raw["results"], raw["machine"]


def plot_overview(
    results: list[dict[str, Any]],
    machine: dict[str, Any],
    output_dir: Path,
    dataset: str = "small",
) -> Path:
    _setup_style()
    subset = [r for r in results if r["dataset"] == dataset]
    if not subset:
        msg = f"No results for dataset '{dataset}'"
        raise ValueError(msg)

    ops = [r["operation"] for r in subset if r["engine"] == "polars"]
    ops_ordered = [op for op in OP_LABELS if op in ops]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ops_ordered))
    width = 0.25

    for i, engine in enumerate(ENGINE_ORDER):
        medians: list[float] = []
        err_lo: list[float] = []
        err_hi: list[float] = []
        for op in ops_ordered:
            match = next(
                (r for r in subset if r["engine"] == engine and r["operation"] == op),
                None,
            )
            if match:
                med = match["median_seconds"] * 1000
                times = sorted(match.get("times_seconds", []))
                if len(times) >= 4:
                    q1 = times[len(times) // 4] * 1000
                    q3 = times[3 * len(times) // 4] * 1000
                else:
                    q1 = q3 = med
                medians.append(med)
                err_lo.append(max(med - q1, 0))
                err_hi.append(max(q3 - med, 0))
            else:
                medians.append(0)
                err_lo.append(0)
                err_hi.append(0)

        bars = ax.bar(
            x + i * width,
            medians,
            width,
            label=engine.capitalize(),
            color=ENGINE_COLORS[engine],
            yerr=[err_lo, err_hi],
            capsize=3,
            error_kw={"linewidth": 0.8},
        )

        for bar, val, hi in zip(bars, medians, err_hi, strict=True):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + hi + max(err_hi) * 0.15,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#555",
                )

    ax.set_xlabel("Operation")
    ax.set_ylabel("Median time (ms, log scale)")
    row_count = subset[0]["row_count"]
    ax.set_title(
        f"Performance Overview — {dataset} ({row_count:,} rows)",
        fontweight="bold",
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([OP_LABELS.get(op, op) for op in ops_ordered])
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)

    fig.text(
        0.99,
        0.01,
        f"{machine['system']} {machine['machine']} · "
        f"{machine['cpu_count']} cores · "
        f"{machine['ram_gb']} GB RAM · "
        f"Python {machine['python_version']}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#999",
    )

    path = output_dir / f"overview_{dataset}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_relative(
    results: list[dict[str, Any]], output_dir: Path, dataset: str = "small"
) -> Path:
    _setup_style()
    subset = [r for r in results if r["dataset"] == dataset]
    ops = [r["operation"] for r in subset if r["engine"] == "polars"]
    ops_ordered = [op for op in OP_LABELS if op in ops]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ops_ordered))
    width = 0.25

    for i, engine in enumerate(ENGINE_ORDER):
        ratios: list[float] = []
        for op in ops_ordered:
            engine_result = next(
                (r for r in subset if r["engine"] == engine and r["operation"] == op),
                None,
            )
            fastest = min(
                (
                    r["median_seconds"]
                    for r in subset
                    if r["operation"] == op and r["median_seconds"] > 0
                ),
                default=1,
            )
            if engine_result and engine_result["median_seconds"] > 0:
                ratios.append(engine_result["median_seconds"] / fastest)
            else:
                ratios.append(0)

        bars = ax.bar(
            x + i * width,
            ratios,
            width,
            label=engine.capitalize(),
            color=ENGINE_COLORS[engine],
        )

        for bar, val in zip(bars, ratios, strict=True):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{val:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#555",
                )

    ax.axhline(y=1.0, color="#ccc", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Operation")
    ax.set_ylabel("Relative time (1.0x = fastest, log scale)")
    ax.set_title(
        f"Relative Performance — {dataset} (normalized to fastest engine)",
        fontweight="bold",
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([OP_LABELS.get(op, op) for op in ops_ordered])
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)

    path = output_dir / f"relative_{dataset}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_variability(
    results: list[dict[str, Any]], output_dir: Path, dataset: str = "small"
) -> Path:
    _setup_style()
    subset = [r for r in results if r["dataset"] == dataset and "times_seconds" in r]
    ops = [r["operation"] for r in subset if r["engine"] == "polars"]
    ops_ordered = [op for op in OP_LABELS if op in ops]

    fig, ax = plt.subplots(figsize=(12, 5))

    positions: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    data: list[list[float]] = []
    pos: float = 0

    for op in ops_ordered:
        for engine in ENGINE_ORDER:
            match = next(
                (r for r in subset if r["engine"] == engine and r["operation"] == op),
                None,
            )
            if match and match.get("times_seconds"):
                data.append([t * 1000 for t in match["times_seconds"]])
                positions.append(pos)
                colors.append(ENGINE_COLORS[engine])
                labels.append("")
            pos += 1
        pos += 0.5

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )

    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    group_centers = []
    for i, _op in enumerate(ops_ordered):
        center = i * 3.5 + 1
        group_centers.append(center)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([OP_LABELS.get(op, op) for op in ops_ordered])

    for engine in ENGINE_ORDER:
        ax.bar(
            0,
            0,
            color=ENGINE_COLORS[engine],
            label=engine.capitalize(),
            alpha=0.7,
        )
    ax.legend(loc="upper left")

    ax.set_ylabel("Time (ms)")
    ax.set_title(
        f"Measurement Variability — {dataset} (5 rounds per operation)",
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)

    path = output_dir / f"variability_{dataset}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_heatmap(
    results: list[dict[str, Any]],
    output_dir: Path,
    dataset: str = "small",
) -> Path:
    _setup_style()
    subset = [r for r in results if r["dataset"] == dataset]
    ops = [r["operation"] for r in subset if r["engine"] == "polars"]
    ops_ordered = [op for op in OP_LABELS if op in ops]

    matrix = []
    for engine in ENGINE_ORDER:
        row = []
        for op in ops_ordered:
            match = next(
                (r for r in subset if r["engine"] == engine and r["operation"] == op),
                None,
            )
            row.append(match["median_seconds"] * 1000 if match else float("nan"))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 3))
    data = np.array(matrix)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(ops_ordered)))
    ax.set_xticklabels(
        [OP_LABELS.get(op, op).replace("\n", " ") for op in ops_ordered],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(ENGINE_ORDER)))
    ax.set_yticklabels([e.capitalize() for e in ENGINE_ORDER])

    for i in range(len(ENGINE_ORDER)):
        for j in range(len(ops_ordered)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > np.nanmax(data) * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Median time (ms)")

    ax.set_title(
        f"Performance Heatmap — {dataset} (ms, lower = faster)",
        fontweight="bold",
    )

    path = output_dir / f"heatmap_{dataset}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all(results_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results, machine = load_results(results_path)
    datasets = sorted(set(r["dataset"] for r in results))
    paths = []

    for dataset in datasets:
        paths.append(plot_overview(results, machine, output_dir, dataset))
        paths.append(plot_relative(results, output_dir, dataset))
        paths.append(plot_variability(results, output_dir, dataset))
        paths.append(plot_heatmap(results, output_dir, dataset))

    return paths
