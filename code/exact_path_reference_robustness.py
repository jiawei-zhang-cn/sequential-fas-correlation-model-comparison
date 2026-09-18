"""Check whether finite-path geometry changes a main-protocol outage result.

The local-plane-wave common-scattering covariance is the reference model used
throughout the paper.  This script first finds the main six-port case with the
largest finite-path covariance deviation, then compares its selected-port
outage against the finite-path covariance and exact LoS phase.  It is a
reference-model robustness check, not a fourth comparison channel model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from .correlation_models import covariance_factor
    from .paper_scenarios import (
        fas_axis,
        literature_scenarios,
        los_mean,
        model_covariances,
    )
    from .literature_anchored_outage_decomposition import (
        PILOT_SNR_LINEAR,
        cdf_for_selected_port,
        conditional_parameters,
        electronic_return_coordinates,
    )
    from .model_source_audit import exact_los_vector, exact_path_covariance
    from .propagation_model import direction_statistics
except ImportError:
    from correlation_models import covariance_factor
    from paper_scenarios import (
        fas_axis,
        literature_scenarios,
        los_mean,
        model_covariances,
    )
    from literature_anchored_outage_decomposition import (
        PILOT_SNR_LINEAR,
        cdf_for_selected_port,
        conditional_parameters,
        electronic_return_coordinates,
    )
    from model_source_audit import exact_los_vector, exact_path_covariance
    from propagation_model import direction_statistics


OBSERVED_PORTS = 6
SWITCH_DELAY_S = 100.0e-6
SAMPLE_COUNT = 2_000_000
CHUNK_SIZE = 200_000


def exact_los_mean(scenario, axis, positions, times):
    if scenario.rician_k == 0.0:
        return np.zeros_like(times, dtype=complex)
    amplitude = np.sqrt(scenario.rician_k / (scenario.rician_k + 1.0))
    return amplitude * exact_los_vector(
        np.asarray(scenario.geometry.tx),
        np.asarray(scenario.geometry.rx),
        np.asarray(scenario.velocity_tx),
        np.asarray(scenario.velocity_rx),
        axis,
        positions,
        times,
    )


def main_cases():
    cases = []
    positions, times = electronic_return_coordinates(
        OBSERVED_PORTS, SWITCH_DELAY_S
    )
    for scenario in literature_scenarios():
        for angle_deg in (0.0, 90.0):
            axis = fas_axis(angle_deg)
            stats = direction_statistics(
                scenario.geometry,
                np.asarray(scenario.velocity_tx),
                np.asarray(scenario.velocity_rx),
                axis,
                order=6,
                maximum_panel_size_m=10.0,
            )
            local_covariance = model_covariances(stats, positions, times)[
                "CS"
            ]
            finite_path_covariance = exact_path_covariance(
                stats,
                np.asarray(scenario.geometry.tx),
                np.asarray(scenario.geometry.rx),
                np.asarray(scenario.velocity_tx),
                np.asarray(scenario.velocity_rx),
                axis,
                positions,
                times,
            )
            cases.append(
                {
                    "scenario": scenario,
                    "fas_angle_deg": angle_deg,
                    "axis": axis,
                    "positions": positions,
                    "times": times,
                    "local_covariance": local_covariance,
                    "finite_path_covariance": finite_path_covariance,
                    "local_mean": los_mean(scenario, axis, positions, times),
                    "finite_path_mean": exact_los_mean(
                        scenario, axis, positions, times
                    ),
                    "covariance_max_absolute_difference": float(
                        np.max(
                            np.abs(
                                finite_path_covariance - local_covariance
                            )
                        )
                    ),
                }
            )
    return cases


def paired_conditional_outage(case):
    scenario = case["scenario"]
    local_yy, local_gain, local_omega = conditional_parameters(
        case["local_covariance"], OBSERVED_PORTS, scenario.rician_k
    )
    exact_yy, exact_gain, exact_omega = conditional_parameters(
        case["finite_path_covariance"], OBSERVED_PORTS, scenario.rician_k
    )
    local_factor = covariance_factor(local_yy)
    exact_factor = covariance_factor(exact_yy)
    local_mean_observation = (
        np.sqrt(PILOT_SNR_LINEAR) * case["local_mean"][:OBSERVED_PORTS]
    )
    exact_mean_observation = (
        np.sqrt(PILOT_SNR_LINEAR) * case["finite_path_mean"][:OBSERVED_PORTS]
    )
    rng = np.random.default_rng(20260915)
    sum_local = sum_exact = sum_difference = sum_square_difference = 0.0
    completed = 0
    while completed < SAMPLE_COUNT:
        size = min(CHUNK_SIZE, SAMPLE_COUNT - completed)
        standard = (
            rng.standard_normal((size, OBSERVED_PORTS))
            + 1j * rng.standard_normal((size, OBSERVED_PORTS))
        ) / np.sqrt(2.0)
        local_y = local_mean_observation + standard @ local_factor.T
        exact_y = exact_mean_observation + standard @ exact_factor.T
        local_selected = np.argmax(np.abs(local_y) ** 2, axis=1)
        exact_selected = np.argmax(np.abs(exact_y) ** 2, axis=1)
        local_value = cdf_for_selected_port(
            local_y,
            local_selected,
            local_mean_observation,
            case["local_mean"][OBSERVED_PORTS:],
            local_gain,
            local_omega,
        )
        exact_value = cdf_for_selected_port(
            exact_y,
            exact_selected,
            exact_mean_observation,
            case["finite_path_mean"][OBSERVED_PORTS:],
            exact_gain,
            exact_omega,
        )
        difference = exact_value - local_value
        sum_local += float(np.sum(local_value))
        sum_exact += float(np.sum(exact_value))
        sum_difference += float(np.sum(difference))
        sum_square_difference += float(np.sum(difference**2))
        completed += size

    local_outage = sum_local / SAMPLE_COUNT
    exact_outage = sum_exact / SAMPLE_COUNT
    difference = sum_difference / SAMPLE_COUNT
    standard_error = float(
        np.sqrt(
            max(
                sum_square_difference / SAMPLE_COUNT - difference**2,
                0.0,
            )
            / SAMPLE_COUNT
        )
    )
    return {
        "sample_count": SAMPLE_COUNT,
        "finite_path_outage": exact_outage,
        "local_plane_wave_outage": local_outage,
        "finite_path_minus_local_outage": difference,
        "relative_outage_difference": abs(difference) / exact_outage,
        "paired_conditional_estimator_standard_error": standard_error,
    }


def main() -> None:
    cases = main_cases()
    worst_case = max(cases, key=lambda item: item["covariance_max_absolute_difference"])
    result = paired_conditional_outage(worst_case)
    output = {
        "selection_rule": (
            "The four main six-port, 100-us electronic-return cases were "
            "screened, and the largest finite-path covariance deviation was "
            "used for the outage check."
        ),
        "screened_cases": [
            {
                "scenario": item["scenario"].name,
                "fas_orientation": (
                    "longitudinal" if item["fas_angle_deg"] == 0.0 else "transverse"
                ),
                "covariance_max_absolute_difference": item[
                    "covariance_max_absolute_difference"
                ],
            }
            for item in cases
        ],
        "selected_case": {
            "scenario": worst_case["scenario"].name,
            "fas_orientation": (
                "longitudinal"
                if worst_case["fas_angle_deg"] == 0.0
                else "transverse"
            ),
            "observed_ports_k": OBSERVED_PORTS,
            "switch_delay_us": SWITCH_DELAY_S * 1.0e6,
            "protocol": "electronic_return_if_needed",
            "covariance_max_absolute_difference": worst_case[
                "covariance_max_absolute_difference"
            ],
            "los_max_absolute_difference": float(
                np.max(
                    np.abs(
                        worst_case["finite_path_mean"] - worst_case["local_mean"]
                    )
                )
            ),
            **result,
        },
    }
    output_directory = Path(__file__).resolve().parents[1] / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "exact_path_reference_robustness.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
