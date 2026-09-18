"""Plot event-level applicability and cancellation from the saved sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"
FIGURES_DIRECTORY = REPOSITORY_ROOT / "figures"


def draw_reference_layers(
    axis: plt.Axes,
    axis_limit: float,
    tolerance_percent: float,
) -> None:
    axis.fill_between(
        [0.0, tolerance_percent],
        tolerance_percent,
        axis_limit,
        color="#f3b562",
        alpha=0.10,
        label="Cancellation-masked region",
        zorder=0,
    )
    axis.plot(
        [0.0, axis_limit],
        [0.0, axis_limit],
        color="#666666",
        linestyle="--",
        linewidth=0.8,
        alpha=0.72,
        label="No cancellation",
        zorder=1,
    )
    axis.axvline(
        tolerance_percent,
        color="#b24c4c",
        linestyle=":",
        linewidth=0.8,
        alpha=0.72,
        zorder=1,
    )
    axis.axhline(
        tolerance_percent,
        color="#b24c4c",
        linestyle=":",
        linewidth=0.8,
        alpha=0.72,
        zorder=1,
    )


def draw_model_points(
    axis: plt.Axes,
    rows: list[dict[str, object]],
    model: str,
    cases: dict[tuple[str, str], tuple[str, str]],
    color_map: object,
    normalizer: Normalize,
    show_labels: bool,
) -> None:
    plotted_cases: set[tuple[str, str]] = set()
    model_rows = sorted(
        (row for row in rows if row["model"] == model),
        key=lambda row: float(row["switch_time_us"]),
    )
    for row in model_rows:
        case_key = (str(row["scenario"]), str(row["fas_orientation"]))
        label, marker = cases[case_key]
        use_label = show_labels and case_key not in plotted_cases
        axis.scatter(
            [100.0 * float(row["net_relative_error"])],
            [100.0 * float(row["internal_relative_mismatch"])],
            c=[float(row["switch_time_us"])],
            cmap=color_map,
            norm=normalizer,
            marker=marker,
            s=28,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.85,
            label=label if use_label else "_nolegend_",
            zorder=2,
        )
        plotted_cases.add(case_key)


def main() -> None:
    source_path = RESULTS_DIRECTORY / "literature_anchored_decomposition_switch_sweep.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["switch_time_us"] > 0]
    models = (
        ("TRD", "Tx/Rx-Side Decoupled (TRD)"),
        ("STS", "Space-Time Separable (STS)"),
    )
    cases = {
        ("yoo_fig8_same_direction", "longitudinal"): (
            "Same direction, longitudinal",
            "o",
        ),
        ("yoo_fig8_same_direction", "transverse"): (
            "Same direction, transverse",
            "s",
        ),
        ("yoo_fig9_opposite_direction", "longitudinal"): (
            "Opposite direction, longitudinal",
            "^",
        ),
        ("yoo_fig9_opposite_direction", "transverse"): (
            "Opposite direction, transverse",
            "D",
        ),
    }
    max_percent = max(
        100.0 * row["internal_relative_mismatch"] for row in rows
    )
    axis_max = 5.0 * np.ceil(1.08 * max_percent / 5.0)
    tolerance_percent = 5.0
    normalizer = Normalize(vmin=10.0, vmax=100.0)
    color_map = plt.get_cmap("viridis")

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), sharex=True, sharey=True)
    for axis, (model, title) in zip(axes, models, strict=True):
        draw_reference_layers(axis, axis_max, tolerance_percent)
        draw_model_points(
            axis,
            rows,
            model,
            cases,
            color_map,
            normalizer,
            show_labels=True,
        )
        axis.set_title(title)
        axis.set_xlabel("Net relative outage error (%)")
        axis.set_xlim(0.0, axis_max)
        axis.set_ylim(0.0, axis_max)
        axis.grid(which="major", alpha=0.12, linewidth=0.5)

    inset_limit = 6.0
    inset_axis = axes[1].inset_axes([0.53, 0.08, 0.43, 0.43])
    draw_reference_layers(inset_axis, inset_limit, tolerance_percent)
    draw_model_points(
        inset_axis,
        rows,
        "STS",
        cases,
        color_map,
        normalizer,
        show_labels=False,
    )
    inset_axis.set_xlim(0.0, inset_limit)
    inset_axis.set_ylim(0.0, inset_limit)
    inset_axis.set_xticks([0.0, 2.0, 4.0, 6.0])
    inset_axis.set_yticks([0.0, 2.0, 4.0, 6.0])
    inset_axis.tick_params(labelsize=7, width=0.7, length=2.5)
    inset_axis.grid(which="major", alpha=0.10, linewidth=0.45)
    for spine in inset_axis.spines.values():
        spine.set_linewidth(0.8)
    axes[1].indicate_inset_zoom(
        inset_axis,
        edgecolor="#555555",
        alpha=0.65,
        linewidth=0.75,
    )

    axes[0].set_ylabel("Uncanceled mechanism contribution (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.47, 0.015),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.86,
        top=0.90,
        bottom=0.23,
        wspace=0.12,
    )
    color_bar_axis = figure.add_axes([0.895, 0.23, 0.014, 0.66])
    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalizer, cmap=color_map),
        cax=color_bar_axis,
    )
    color_bar.set_label("Per-port switching delay (μs)", fontsize=9, labelpad=9)
    color_bar.ax.tick_params(labelsize=8, width=0.7, length=3)

    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIRECTORY / "literature_anchored_applicability_map.png"
    pdf_path = png_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    print(
        json.dumps(
            {
                "source_json": str(source_path),
                "png": str(png_path),
                "pdf": str(pdf_path),
                "axis_max_percent": axis_max,
                "classification_tolerance_percent": tolerance_percent,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
