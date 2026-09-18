"""Six-port idealized sensitivity check without selected-port return delay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from correlation_models import covariance_factor
from paper_scenarios import (
    DATA_SNR_LINEAR,
    PILOT_SNR_LINEAR,
    QUADRATURE_PANEL_SIZE_M,
    SWITCH_TIME_S,
    TARGET_SNR_LINEAR,
    fas_axis,
    literature_scenarios,
    los_mean,
    model_covariances,
    sample_coordinates,
)
from propagation_model import direction_statistics


OBSERVED_PORTS = 6
SAMPLE_COUNT = 2_000_000
CHUNK_SIZE = 200_000


def paired_outage(
    common_covariance: np.ndarray,
    trd_covariance: np.ndarray,
    mean: np.ndarray,
    rician_k: float,
    seed: int,
) -> dict[str, float]:
    """Estimate both outage events with the same Gaussian samples and noise."""
    diffuse_scale = 1.0 / (rician_k + 1.0)
    factors = {
        "CS": covariance_factor(diffuse_scale * common_covariance),
        "TRD": covariance_factor(
            diffuse_scale * trd_covariance
        ),
    }
    outage_counts = {name: 0 for name in factors}
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
        for name, factor in factors.items():
            channel = mean + standard @ factor.T
            observed = channel[:, :OBSERVED_PORTS]
            data = channel[:, OBSERVED_PORTS:]
            pilots = np.sqrt(PILOT_SNR_LINEAR) * observed + noise
            selected = np.argmax(np.abs(pilots) ** 2, axis=1)
            selected_data = data[np.arange(size), selected]
            outage_counts[name] += int(
                np.sum(
                    DATA_SNR_LINEAR * np.abs(selected_data) ** 2
                    < TARGET_SNR_LINEAR
                )
            )
        completed += size

    common = outage_counts["CS"] / SAMPLE_COUNT
    trd = outage_counts["TRD"] / SAMPLE_COUNT
    return {
        "CS_outage": common,
        "TRD_outage": trd,
        "relative_error_percent": 100.0 * abs(trd - common) / common,
    }


def main() -> None:
    rows = []
    positions, times = sample_coordinates(OBSERVED_PORTS, SWITCH_TIME_S)
    for scenario in literature_scenarios():
        for angle_deg in (0.0, 90.0):
            axis = fas_axis(angle_deg)
            stats = direction_statistics(
                scenario.geometry,
                np.asarray(scenario.velocity_tx),
                np.asarray(scenario.velocity_rx),
                axis,
                order=6,
                maximum_panel_size_m=QUADRATURE_PANEL_SIZE_M,
            )
            covariances = model_covariances(stats, positions, times)
            mean = los_mean(scenario, axis, positions, times)
            rows.append(
                {
                    "scenario": scenario.name,
                    "fas_orientation": (
                        "longitudinal" if angle_deg == 0.0 else "transverse"
                    ),
                    **paired_outage(
                        covariances["CS"],
                        covariances["TRD"],
                        mean,
                        scenario.rician_k,
                        seed=(
                            20260913
                            + int(angle_deg)
                            + 10 * OBSERVED_PORTS
                            + int(round(SWITCH_TIME_S * 1.0e9))
                            + (800 if scenario.rician_k > 0.0 else 900)
                        ),
                    ),
                }
            )

    payload = {
        "protocol": "data transmission starts at the final-pilot time without an additional selected-port return delay",
        "observed_ports_k": OBSERVED_PORTS,
        "switch_time_us": SWITCH_TIME_S * 1.0e6,
        "sample_count_per_case": SAMPLE_COUNT,
        "rows": rows,
    }
    output_directory = Path(__file__).resolve().parents[1] / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "zero_return_delay_sensitivity.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
