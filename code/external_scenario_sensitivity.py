"""Six-port sensitivity check using the Avazov--Paetzold road geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from correlation_models import covariance_factor
from literature_anchored_outage_decomposition import electronic_return_coordinates
from paper_scenarios import (
    DATA_SNR_LINEAR,
    PILOT_SNR_LINEAR,
    TARGET_SNR_LINEAR,
    WAVELENGTH_M,
    model_covariances,
)
from propagation_model import Geometry, rectangle_quadrature


OBSERVED_PORTS = 6
SWITCH_TIME_S = 100.0e-6
SAMPLE_COUNT = 1_000_000
CHUNK_SIZE = 200_000


def external_geometry() -> Geometry:
    return Geometry(
        name="avazov_2012_sec6_two_rectangles",
        tx=(0.0, 0.0),
        rx=(400.0, 10.0),
        rectangles=(
            (-50.0, 450.0, 20.0, 120.0),
            (-50.0, 450.0, -110.0, -10.0),
        ),
    )


def external_statistics(axis: np.ndarray) -> dict[str, np.ndarray | float]:
    geometry = external_geometry()
    nodes, weights = rectangle_quadrature(
        geometry.rectangles,
        order=6,
        maximum_panel_size_m=10.0,
    )
    tx = np.asarray(geometry.tx, dtype=float)
    rx = np.asarray(geometry.rx, dtype=float)
    tx_to_scatter = nodes - tx
    rx_to_scatter = nodes - rx
    tx_unit = tx_to_scatter / np.linalg.norm(tx_to_scatter, axis=1, keepdims=True)
    rx_unit = rx_to_scatter / np.linalg.norm(rx_to_scatter, axis=1, keepdims=True)
    transmit_heading = np.array((1.0, 0.0))
    receive_heading = np.array((-1.0, 0.0))
    transmit_doppler = 91.0 * (tx_unit @ transmit_heading)
    receive_doppler = 91.0 * (rx_unit @ receive_heading)
    return {
        "nodes": nodes,
        "weights": weights,
        "u_tx": tx_unit,
        "u_rx": -rx_unit,
        "doppler_tx": transmit_doppler,
        "doppler_rx": -receive_doppler,
        "doppler_total": transmit_doppler + receive_doppler,
        "spatial_projection": -(rx_unit @ axis),
    }


def external_los_mean(
    axis: np.ndarray,
    positions: np.ndarray,
    times: np.ndarray,
    rician_k: float,
) -> np.ndarray:
    geometry = external_geometry()
    tx = np.asarray(geometry.tx, dtype=float)
    rx = np.asarray(geometry.rx, dtype=float)
    displacement = rx - tx
    distance = float(np.linalg.norm(displacement))
    tx_to_rx = displacement / distance
    rx_to_tx = -tx_to_rx
    transmit_heading = np.array((1.0, 0.0))
    receive_heading = np.array((-1.0, 0.0))
    transmit_los_doppler = 91.0 * float(tx_to_rx @ transmit_heading)
    receive_los_doppler = 91.0 * float(rx_to_tx @ receive_heading)
    spatial_los = float(tx_to_rx @ axis)
    amplitude = np.sqrt(rician_k / (rician_k + 1.0)) if rician_k else 0.0
    wave_number = 2.0 * np.pi / WAVELENGTH_M
    phase = 2.0 * np.pi * (
        transmit_los_doppler + receive_los_doppler
    ) * times
    phase -= wave_number * (distance + spatial_los * positions)
    return amplitude * np.exp(1j * phase)


def paired_monte_carlo(
    covariances: dict[str, np.ndarray],
    mean: np.ndarray,
    rician_k: float,
    seed: int,
) -> dict[str, object]:
    diffuse_scale = 1.0 / (rician_k + 1.0)
    factors = {
        name: covariance_factor(diffuse_scale * covariance)
        for name, covariance in covariances.items()
    }
    outage_counts = {name: 0 for name in factors}
    paired_sum = {
        name: 0.0 for name in factors if name != "common_scatterer"
    }
    paired_square_sum = {
        name: 0.0 for name in factors if name != "common_scatterer"
    }
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < SAMPLE_COUNT:
        size = min(CHUNK_SIZE, SAMPLE_COUNT - completed)
        standard = (
            rng.standard_normal((size, mean.size))
            + 1j * rng.standard_normal((size, mean.size))
        ) / np.sqrt(2.0)
        noise = (
            rng.standard_normal((size, OBSERVED_PORTS))
            + 1j * rng.standard_normal((size, OBSERVED_PORTS))
        ) / np.sqrt(2.0)
        indicators = {}
        for name, factor in factors.items():
            channel = mean + standard @ factor.T
            observed = channel[:, :OBSERVED_PORTS]
            data = channel[:, OBSERVED_PORTS:]
            pilots = np.sqrt(PILOT_SNR_LINEAR) * observed + noise
            selected = np.argmax(np.abs(pilots) ** 2, axis=1)
            selected_data = data[np.arange(size), selected]
            indicator = (
                DATA_SNR_LINEAR * np.abs(selected_data) ** 2
                < TARGET_SNR_LINEAR
            )
            indicators[name] = indicator
            outage_counts[name] += int(np.sum(indicator))
        common_indicator = indicators["common_scatterer"].astype(float)
        for name in paired_sum:
            difference = common_indicator - indicators[name].astype(float)
            paired_sum[name] += float(np.sum(difference))
            paired_square_sum[name] += float(np.sum(difference**2))
        completed += size

    probabilities = {
        name: count / SAMPLE_COUNT for name, count in outage_counts.items()
    }
    comparisons = {}
    common = probabilities["common_scatterer"]
    for name in paired_sum:
        difference = paired_sum[name] / SAMPLE_COUNT
        second_moment = paired_square_sum[name] / SAMPLE_COUNT
        variance = max(second_moment - difference**2, 0.0)
        comparisons[name] = {
            "common_minus_model": difference,
            "paired_standard_error": float(np.sqrt(variance / SAMPLE_COUNT)),
            "relative_absolute_difference_to_common": abs(difference) / common,
        }
    return {
        "outage_probability": probabilities,
        "comparison_to_common": comparisons,
    }


def main() -> None:
    rows = []
    positions, times = electronic_return_coordinates(OBSERVED_PORTS, SWITCH_TIME_S)
    for angle_deg in (0.0, 90.0):
        angle = np.deg2rad(angle_deg)
        axis = np.array((np.cos(angle), np.sin(angle)))
        statistics = external_statistics(axis)
        covariances = model_covariances(statistics, positions, times)
        for rician_k in (0.0, 0.5, 1.0):
            mean = external_los_mean(axis, positions, times, rician_k)
            result = paired_monte_carlo(
                covariances,
                mean,
                rician_k,
                seed=20260913 + int(angle_deg) + int(10 * rician_k),
            )
            rows.append(
                {
                    "fas_orientation": (
                        "longitudinal" if angle_deg == 0.0 else "transverse"
                    ),
                    "rician_k": rician_k,
                    "outage_probability": result["outage_probability"],
                    "comparison_to_common": result["comparison_to_common"],
                    "independent_relative_error_percent": 100.0
                    * result["comparison_to_common"]["independent_AoD_AoA"][
                        "relative_absolute_difference_to_common"
                    ],
                    "separable_relative_error_percent": 100.0
                    * result["comparison_to_common"]["fully_separable"][
                        "relative_absolute_difference_to_common"
                    ],
                }
            )

    payload = {
        "source": "Avazov and Paetzold 2012, Section 6",
        "protocol": "electronic_return_if_needed",
        "observed_ports_k": OBSERVED_PORTS,
        "aperture_wavelengths": 2.0,
        "switch_time_us": SWITCH_TIME_S * 1.0e6,
        "sample_count_per_row": SAMPLE_COUNT,
        "rows": rows,
    }
    output_directory = Path(__file__).resolve().parents[1] / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "external_scenario_sensitivity.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
