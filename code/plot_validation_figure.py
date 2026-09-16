"""Generate numerical and reference-model validation panels."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from correlation_models import covariance_factor
from exact_path_reference_robustness import exact_los_mean
from paper_scenarios import (
    PILOT_SNR_LINEAR,
    QUADRATURE_PANEL_SIZE_M,
    TARGET_SNR_LINEAR,
    WAVELENGTH_M,
    fas_axis,
    literature_scenarios,
    los_mean,
    model_covariances,
)
from propagation_model import direction_statistics
from literature_anchored_outage_decomposition import (
    conditional_parameters,
    cdf_for_selected_port,
    electronic_return_coordinates,
)
from model_source_audit import exact_path_covariance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"
FIGURES_DIRECTORY = REPOSITORY_ROOT / "figures"

OBSERVED_PORTS = 6
SAMPLE_COUNT = 2_000_000
CHUNK_SIZE = 200_000
VALIDATION_DELAYS_US = (0.0, 50.0, 100.0)
REFERENCE_DELAYS_US = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
VALIDATION_MODELS = (
    "common_scatterer",
    "independent_AoD_AoA",
    "fully_separable",
)


def conditional_and_direct_outage(
    covariance: np.ndarray,
    mean: np.ndarray,
    rician_k: float,
    observed_ports: int,
    count: int,
    seed: int,
) -> tuple[float, float]:
    """Estimate the same event by direct data sampling and conditional CDFs."""
    sigma_yy, gain, omega = conditional_parameters(
        covariance, observed_ports, rician_k
    )
    diffuse_scale = 1.0 / (rician_k + 1.0)
    channel_factor = covariance_factor(diffuse_scale * covariance)
    observation_mean = np.sqrt(PILOT_SNR_LINEAR) * mean[:observed_ports]
    rng = np.random.default_rng(seed)
    direct_sum = 0.0
    conditional_sum = 0.0
    completed = 0
    while completed < count:
        size = min(CHUNK_SIZE, count - completed)
        standard_channel = (
            rng.standard_normal((size, mean.size))
            + 1j * rng.standard_normal((size, mean.size))
        ) / np.sqrt(2.0)
        channel = mean + standard_channel @ channel_factor.T
        standard_observation = (
            rng.standard_normal((size, observed_ports))
            + 1j * rng.standard_normal((size, observed_ports))
        ) / np.sqrt(2.0)
        pilots = (
            np.sqrt(PILOT_SNR_LINEAR) * channel[:, :observed_ports]
            + standard_observation
        )
        selected = np.argmax(np.abs(pilots) ** 2, axis=1)
        selected_data = channel[
            np.arange(size), observed_ports + selected
        ]
        direct_indicator = (
            10.0 * np.abs(selected_data) ** 2 < TARGET_SNR_LINEAR
        )
        conditional_values = cdf_for_selected_port(
            pilots,
            selected,
            observation_mean,
            mean[observed_ports:],
            gain,
            omega,
            TARGET_SNR_LINEAR,
        )
        direct_sum += float(np.sum(direct_indicator))
        conditional_sum += float(np.sum(conditional_values))
        completed += size
    return direct_sum / count, conditional_sum / count


def conditional_outage(
    covariance: np.ndarray,
    mean: np.ndarray,
    rician_k: float,
    observed_ports: int,
    count: int,
    seed: int,
) -> float:
    """Estimate post-selection outage from the conditional CDF integral."""
    sigma_yy, gain, omega = conditional_parameters(
        covariance, observed_ports, rician_k
    )
    observation_mean = np.sqrt(PILOT_SNR_LINEAR) * mean[:observed_ports]
    factor = covariance_factor(sigma_yy)
    rng = np.random.default_rng(seed)
    total = 0.0
    completed = 0
    while completed < count:
        size = min(CHUNK_SIZE, count - completed)
        standard = (
            rng.standard_normal((size, observed_ports))
            + 1j * rng.standard_normal((size, observed_ports))
        ) / np.sqrt(2.0)
        observations = observation_mean + standard @ factor.T
        selected = np.argmax(np.abs(observations) ** 2, axis=1)
        values = cdf_for_selected_port(
            observations,
            selected,
            observation_mean,
            mean[observed_ports:],
            gain,
            omega,
            TARGET_SNR_LINEAR,
        )
        total += float(np.sum(values))
        completed += size
    return total / count


def build_panel_a() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(literature_scenarios()):
        for orientation_index, angle_deg in enumerate((0.0, 90.0)):
            axis = fas_axis(angle_deg)
            stats = direction_statistics(
                scenario.geometry,
                np.asarray(scenario.velocity_tx),
                np.asarray(scenario.velocity_rx),
                axis,
                order=6,
                maximum_panel_size_m=QUADRATURE_PANEL_SIZE_M,
            )
            for delay_index, delay_us in enumerate(VALIDATION_DELAYS_US):
                positions, times = electronic_return_coordinates(
                    OBSERVED_PORTS, delay_us * 1.0e-6
                )
                covariances = model_covariances(stats, positions, times)
                mean = los_mean(scenario, axis, positions, times)
                for model_index, model in enumerate(VALIDATION_MODELS):
                    direct, conditional = conditional_and_direct_outage(
                        covariances[model],
                        mean,
                        scenario.rician_k,
                        OBSERVED_PORTS,
                        SAMPLE_COUNT,
                        seed=20260916
                        + 10000 * model_index
                        + 1000 * scenario_index
                        + 100 * orientation_index
                        + delay_index,
                    )
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "fas_orientation": (
                                "longitudinal" if angle_deg == 0.0 else "transverse"
                            ),
                            "model": model,
                            "switch_delay_us": delay_us,
                            "direct_data_time_outage": direct,
                            "conditional_outage": conditional,
                            "absolute_difference": abs(direct - conditional),
                        }
                    )
    return rows


def build_panel_b() -> list[dict[str, object]]:
    scenario = literature_scenarios()[1]
    angle_deg = 90.0
    axis = fas_axis(angle_deg)
    stats = direction_statistics(
        scenario.geometry,
        np.asarray(scenario.velocity_tx),
        np.asarray(scenario.velocity_rx),
        axis,
        order=6,
        maximum_panel_size_m=QUADRATURE_PANEL_SIZE_M,
    )
    rows: list[dict[str, object]] = []
    for delay_index, delay_us in enumerate(REFERENCE_DELAYS_US):
        positions, times = electronic_return_coordinates(
            OBSERVED_PORTS, delay_us * 1.0e-6
        )
        local_covariance = model_covariances(stats, positions, times)[
            "common_scatterer"
        ]
        exact_covariance = exact_path_covariance(
            stats,
            np.asarray(scenario.geometry.tx),
            np.asarray(scenario.geometry.rx),
            np.asarray(scenario.velocity_tx),
            np.asarray(scenario.velocity_rx),
            axis,
            positions,
            times,
        )
        local_mean = los_mean(scenario, axis, positions, times)
        exact_mean = exact_los_mean(scenario, axis, positions, times)
        local_outage = conditional_outage(
            local_covariance,
            local_mean,
            scenario.rician_k,
            OBSERVED_PORTS,
            SAMPLE_COUNT,
            seed=20260915,
        )
        exact_outage = conditional_outage(
            exact_covariance,
            exact_mean,
            scenario.rician_k,
            OBSERVED_PORTS,
            SAMPLE_COUNT,
            seed=20260915,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "fas_orientation": "transverse",
                "observed_ports_k": OBSERVED_PORTS,
                "switch_delay_us": delay_us,
                "exact_path_outage": exact_outage,
                "local_common_scatterer_outage": local_outage,
                "absolute_difference": abs(exact_outage - local_outage),
            }
        )
    return rows


def plot_figure(panel_a: list[dict[str, object]], panel_b: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.35))
    marker_cases = {
        ("yoo_fig8_same_direction", "longitudinal"): ("Same direction, longitudinal", "o"),
        ("yoo_fig8_same_direction", "transverse"): ("Same direction, transverse", "s"),
        ("yoo_fig9_opposite_direction", "longitudinal"): ("Opposite direction, longitudinal", "^"),
        ("yoo_fig9_opposite_direction", "transverse"): ("Opposite direction, transverse", "D"),
    }
    point_color = "#2f5d7e"
    x_values = np.asarray([row["direct_data_time_outage"] for row in panel_a])
    y_values = np.asarray([row["conditional_outage"] for row in panel_a])
    limits = np.concatenate((x_values, y_values))
    lower = float(limits.min())
    upper = float(limits.max())
    padding = max(0.08 * (upper - lower), 2.0e-4)
    parity_lower = max(0.0, lower - padding)
    parity_upper = upper + padding
    axes[0].plot(
        [parity_lower, parity_upper],
        [parity_lower, parity_upper],
        linestyle="--",
        linewidth=0.85,
        color="#666666",
        zorder=1,
    )
    for (scenario_name, orientation), (label, marker) in marker_cases.items():
        subset = [
            row
            for row in panel_a
            if row["scenario"] == scenario_name
            and row["fas_orientation"] == orientation
        ]
        axes[0].scatter(
            [row["direct_data_time_outage"] for row in subset],
            [row["conditional_outage"] for row in subset],
            s=30,
            marker=marker,
            color=point_color,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.88,
            label=label,
            zorder=2,
        )
    axes[0].set_xlim(parity_lower, parity_upper)
    axes[0].set_ylim(parity_lower, parity_upper)
    axes[0].set_title("(a) Numerical implementation", fontsize=10)
    axes[0].set_xlabel("Direct data-time Monte Carlo outage probability", fontsize=8.5)
    axes[0].set_ylabel("Conditional Monte Carlo outage probability", fontsize=8.5)
    axes[0].grid(which="major", alpha=0.18, linewidth=0.5)
    axes[0].text(
        0.76,
        0.75,
        "y = x",
        transform=axes[0].transAxes,
        fontsize=7.5,
        color="#666666",
        rotation=32,
        rotation_mode="anchor",
    )
    axes[0].legend(
        loc="lower right",
        fontsize=6.2,
        frameon=False,
        handletextpad=0.3,
        labelspacing=0.15,
        borderpad=0.2,
    )

    delays = np.asarray([row["switch_delay_us"] for row in panel_b])
    exact = np.asarray([row["exact_path_outage"] for row in panel_b])
    local = np.asarray([row["local_common_scatterer_outage"] for row in panel_b])
    axes[1].plot(
        delays,
        local,
        color="#2f5d7e",
        linewidth=1.35,
        label="Local common-scatterer model",
        zorder=2,
    )
    axes[1].plot(
        delays,
        exact,
        color="#222222",
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.8,
        linewidth=0.85,
        linestyle="None",
        label="Exact-path model",
        zorder=3,
    )
    axes[1].set_title(
        "(b) Local-model validation",
        fontsize=10,
    )
    axes[1].set_xlabel(r"Per-port switching delay $T_{\rm switch}$ ($\mu$s)", fontsize=8.5)
    axes[1].set_ylabel("Post-selection outage probability", fontsize=8.5)
    axes[1].set_xlim(float(delays.min()) - 3.0, float(delays.max()) + 3.0)
    y_min = float(min(exact.min(), local.min()))
    y_max = float(max(exact.max(), local.max()))
    axes[1].set_ylim(y_min - 0.08 * (y_max - y_min), y_max + 0.08 * (y_max - y_min))
    axes[1].grid(which="major", alpha=0.18, linewidth=0.5)
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)

    inset = axes[1].inset_axes([0.08, 0.48, 0.40, 0.30])
    absolute_difference = np.asarray(
        [row["absolute_difference"] for row in panel_b]
    )
    inset.plot(
        delays,
        absolute_difference,
        color="#8b4f39",
        marker="o",
        markersize=3.0,
        linewidth=1.0,
    )
    inset.set_title("Absolute outage difference", fontsize=8, pad=2.0)
    inset.set_xlabel(r"$T_{\rm switch}$ ($\mu$s)", fontsize=7, labelpad=1.0)
    inset.set_ylabel(r"$|\Delta P_{\rm out}|$", fontsize=7, labelpad=1.0)
    inset.tick_params(labelsize=6.8, width=0.6, length=2.0)
    inset.grid(which="major", alpha=0.15, linewidth=0.5)
    inset.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    inset.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for spine in inset.spines.values():
        spine.set_linewidth(0.7)
    figure.subplots_adjust(left=0.075, right=0.975, bottom=0.20, top=0.86, wspace=0.30)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_stem = FIGURES_DIRECTORY / "validation_figure"
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the numerical and local-model validation figure."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute the Monte Carlo data instead of using the saved result.",
    )
    arguments = parser.parse_args()
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIRECTORY / "validation_figure.json"
    if arguments.recompute:
        panel_a = build_panel_a()
        panel_b = build_panel_b()
        output = {
            "sample_count_per_point": SAMPLE_COUNT,
            "reported_max_abs_difference": 3.42e-4,
            "panel_a": panel_a,
            "panel_b": panel_b,
            "panel_b_reference_note": {
                "scenario": "yoo_fig9_opposite_direction",
                "fas_orientation": "transverse",
                "observed_ports_k": OBSERVED_PORTS,
                "protocol": "electronic_return_if_needed",
            },
        }
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    else:
        output = json.loads(output_path.read_text(encoding="utf-8"))
        panel_a = output["panel_a"]
        panel_b = output["panel_b"]
    plot_figure(panel_a, panel_b)
    print(
        json.dumps(
            {
                "json": str(output_path),
                "figure": str(FIGURES_DIRECTORY / "validation_figure.pdf"),
                "recomputed": arguments.recompute,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
