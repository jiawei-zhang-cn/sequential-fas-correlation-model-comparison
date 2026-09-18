"""Plot the raw outage trajectories used in the switching-delay study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"
FIGURES_DIRECTORY = REPOSITORY_ROOT / "figures"


def selected_rows(
    rows: list[dict[str, object]],
    scenario: str,
    orientation: str,
    model: str,
) -> list[dict[str, object]]:
    return sorted(
        (
            row
            for row in rows
            if row["scenario"] == scenario
            and row["fas_orientation"] == orientation
            and row["model"] == model
        ),
        key=lambda row: float(row["switch_time_us"]),
    )


def main() -> None:
    input_path = RESULTS_DIRECTORY / "literature_anchored_decomposition_switch_sweep.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    cases = (
        ("yoo_fig8_same_direction", "longitudinal", "(a) Same-direction, longitudinal"),
        ("yoo_fig8_same_direction", "transverse", "(b) Same-direction, transverse"),
        ("yoo_fig9_opposite_direction", "longitudinal", "(c) Opposite-direction, longitudinal"),
        ("yoo_fig9_opposite_direction", "transverse", "(d) Opposite-direction, transverse"),
    )

    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), sharex=True)
    model_styles = {
        "TRD": ("Tx/Rx-Side Decoupled (TRD)", "#1f77b4", "o"),
        "STS": ("Space-Time Separable (STS)", "#d95f02", "s"),
    }

    for axis, (scenario, orientation, title) in zip(axes.ravel(), cases):
        trd_rows = selected_rows(rows, scenario, orientation, "TRD")
        delays = [float(row["switch_time_us"]) for row in trd_rows]
        common_outages = [float(row["common_outage"]) for row in trd_rows]
        axis.plot(
            delays,
            common_outages,
            color="#222222",
            marker="^",
            markersize=3.5,
            linewidth=1.6,
            label="Common-Scatterer (CS)",
        )
        for model, (label, color, marker) in model_styles.items():
            model_rows = selected_rows(rows, scenario, orientation, model)
            model_outages = [
                float(row["common_outage"]) - float(row["common_minus_model"])
                for row in model_rows
            ]
            axis.plot(
                delays,
                model_outages,
                color=color,
                marker=marker,
                markersize=3.5,
                linewidth=1.4,
                label=label,
            )
        axis.set_title(title, fontsize=9)
        axis.grid(which="major", alpha=0.16, linewidth=0.5)

    figure.supxlabel(
        r"Per-port switching delay $T_{\mathrm{switch}}$ ($\mu\mathrm{s}$)",
        y=0.105,
    )
    figure.supylabel("Post-selection outage probability", x=0.02)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.99,
        top=0.94,
        bottom=0.22,
        hspace=0.28,
        wspace=0.22,
    )

    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_stem = FIGURES_DIRECTORY / "literature_anchored_outage_curves"
    figure.savefig(
        output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03
    )
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )


if __name__ == "__main__":
    main()
