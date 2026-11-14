"""Generate Plotly HTML reports from benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from benchmark.runner import MachineInfo, TimingResult

ENGINE_COLORS: dict[str, str] = {
    "polars": "#4C78A8",
    "pandas": "#F58518",
    "duckdb": "#54A24B",
}


def results_to_json(
    results: list[TimingResult], machine: MachineInfo, output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "machine": {
            "system": machine.system,
            "machine": machine.machine,
            "processor": machine.processor,
            "python_version": machine.python_version,
            "cpu_count": machine.cpu_count,
            "cpu_count_logical": machine.cpu_count_logical,
            "ram_gb": machine.ram_gb,
        },
        "results": [
            {
                "operation": r.operation,
                "engine": r.engine,
                "dataset": r.dataset,
                "median_seconds": r.median_seconds,
                "iqr_seconds": r.iqr_seconds,
                "row_count": r.row_count,
                "times_seconds": r.times_seconds,
            }
            for r in results
        ],
    }
    output.write_text(json.dumps(data, indent=2))


def load_results(path: Path) -> tuple[list[TimingResult], MachineInfo]:
    raw = json.loads(path.read_text())
    machine = MachineInfo(**raw["machine"])
    results = [
        TimingResult(
            operation=r["operation"],
            engine=r["engine"],
            dataset=r["dataset"],
            times_seconds=r["times_seconds"],
            median_seconds=r["median_seconds"],
            iqr_seconds=r["iqr_seconds"],
            row_count=r["row_count"],
        )
        for r in raw["results"]
    ]
    return results, machine


def plot_grouped_bar(
    results: list[TimingResult],
    machine: MachineInfo,
    output: Path,
) -> None:
    operations = sorted(set(r.operation for r in results))
    datasets = sorted(set(r.dataset for r in results))

    for dataset in datasets:
        fig = go.Figure()
        subset = [r for r in results if r.dataset == dataset]
        engines = sorted(set(r.engine for r in subset))

        for engine in engines:
            engine_results = [r for r in subset if r.engine == engine]
            engine_results.sort(key=lambda r: operations.index(r.operation))
            fig.add_trace(
                go.Bar(
                    name=engine,
                    x=[r.operation for r in engine_results],
                    y=[r.median_seconds * 1000 for r in engine_results],
                    error_y=dict(
                        type="data",
                        array=[r.iqr_seconds * 1000 / 2 for r in engine_results],
                        visible=True,
                    ),
                    marker_color=ENGINE_COLORS.get(engine, "#999"),
                )
            )

        row_count = next((r.row_count for r in subset), 0)
        fig.update_layout(
            title=f"Benchmark: {dataset} ({row_count:,} rows)",
            xaxis_title="Operation",
            yaxis_title="Median time (ms)",
            barmode="group",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            annotations=[
                dict(
                    text=(
                        f"{machine.system} {machine.machine} | "
                        f"{machine.cpu_count} cores | "
                        f"{machine.ram_gb} GB RAM | "
                        f"Python {machine.python_version}"
                    ),
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=-0.15,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                )
            ],
        )

        fig_path = output / f"benchmark_{dataset}.html"
        fig.write_html(str(fig_path), include_plotlyjs="cdn")


def plot_heatmap(
    results: list[TimingResult],
    machine: MachineInfo,
    output: Path,
) -> None:
    operations = sorted(set(r.operation for r in results))
    engines = sorted(set(r.engine for r in results))
    datasets = sorted(set(r.dataset for r in results))

    fig = make_subplots(
        rows=len(datasets),
        cols=1,
        subplot_titles=datasets,
        vertical_spacing=0.05,
    )

    for i, dataset in enumerate(datasets, 1):
        subset = [r for r in results if r.dataset == dataset]
        z = []
        for engine in engines:
            row = []
            for op in operations:
                match = next(
                    (r for r in subset if r.engine == engine and r.operation == op),
                    None,
                )
                row.append(match.median_seconds * 1000 if match else 0)
            z.append(row)

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=operations,
                y=engines,
                colorscale="YlOrRd",
                text=[[f"{v:.1f} ms" for v in row] for row in z],
                texttemplate="%{text}",
                showscale=i == 1,
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        title="Performance heatmap (median ms, lower = faster)",
        height=300 * len(datasets),
        template="plotly_white",
    )
    fig.write_html(str(output / "heatmap.html"), include_plotlyjs="cdn")
