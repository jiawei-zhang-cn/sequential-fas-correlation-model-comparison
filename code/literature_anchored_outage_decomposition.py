"""Decompose outage mismatch into observation-law and conditional-aging effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import ncx2

from correlation_models import covariance_factor
from paper_scenarios import (
    DATA_SNR_LINEAR,
    PILOT_SNR_LINEAR,
    TARGET_SNR_LINEAR,
    fas_axis,
    literature_scenarios,
    los_mean,
    model_covariances,
    sample_coordinates,
)
from propagation_model import direction_statistics


def electronic_return_coordinates(
    observed_ports: int,
    switch_delay_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates for one extra electronic switch unless the last port wins."""
    positions, times = sample_coordinates(observed_ports, switch_delay_s)
    data_times = times[observed_ports:]
    data_times[:-1] += switch_delay_s
    return positions, times


def conditional_parameters(
    channel_covariance: np.ndarray,
    observed_ports: int,
    rician_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = channel_covariance / (rician_k + 1.0)
    sigma_yy = (
        PILOT_SNR_LINEAR * covariance[:observed_ports, :observed_ports]
        + np.eye(observed_ports)
    )
    sigma_dy = (
        np.sqrt(PILOT_SNR_LINEAR)
        * covariance[observed_ports:, :observed_ports]
    )
    gain = sigma_dy @ np.linalg.inv(sigma_yy)
    conditional_covariance = (
        covariance[observed_ports:, observed_ports:]
        - gain @ sigma_dy.conj().T
    )
    omega = np.real(np.diag(conditional_covariance))
    if np.any(omega <= 0.0):
        raise ValueError(f"Non-positive conditional variance: {omega}")
    return sigma_yy, gain, omega


def cdf_for_selected_port(
    observations: np.ndarray,
    selected: np.ndarray,
    mean_observation: np.ndarray,
    mean_data: np.ndarray,
    gain: np.ndarray,
    omega: np.ndarray,
    target_snr_linear: float = TARGET_SNR_LINEAR,
) -> np.ndarray:
    conditional_mean = mean_data + (observations - mean_observation) @ gain.T
    row_index = np.arange(observations.shape[0])
    selected_mean = conditional_mean[row_index, selected]
    selected_omega = omega[selected]
    noncentrality = 2.0 * np.abs(selected_mean) ** 2 / selected_omega
    return ncx2.cdf(
        2.0 * target_snr_linear / (DATA_SNR_LINEAR * selected_omega),
        2.0,
        noncentrality,
    )


def accumulate(sum_value: dict[str, float], sum_square: dict[str, float], values: dict[str, np.ndarray]) -> None:
    for name, value in values.items():
        sum_value[name] += float(np.sum(value))
        sum_square[name] += float(np.sum(value**2))


def decomposition(
    common_covariance: np.ndarray,
    model_covariance: np.ndarray,
    mean_channel: np.ndarray,
    observed_ports: int,
    rician_k: float,
    count: int,
    seed: int,
    chunk_size: int = 200_000,
    target_snr_linear: float = TARGET_SNR_LINEAR,
) -> dict[str, object]:
    common_yy, common_gain, common_omega = conditional_parameters(
        common_covariance, observed_ports, rician_k
    )
    model_yy, model_gain, model_omega = conditional_parameters(
        model_covariance, observed_ports, rician_k
    )
    mean_observation = np.sqrt(PILOT_SNR_LINEAR) * mean_channel[:observed_ports]
    mean_data = mean_channel[observed_ports:]
    factors = {
        "common": covariance_factor(common_yy),
        "model": covariance_factor(model_yy),
    }
    names = (
        "common_y_common_conditional",
        "common_y_model_conditional",
        "model_y_common_conditional",
        "model_y_model_conditional",
        "total_common_minus_model",
        "observation_law_contribution",
        "conditional_aging_contribution",
    )
    sum_value = {name: 0.0 for name in names}
    sum_square = {name: 0.0 for name in names}
    rng = np.random.default_rng(seed)
    completed = 0
    while completed < count:
        size = min(chunk_size, count - completed)
        standard = (
            rng.standard_normal((size, observed_ports))
            + 1j * rng.standard_normal((size, observed_ports))
        ) / np.sqrt(2.0)
        y_common = mean_observation + standard @ factors["common"].T
        y_model = mean_observation + standard @ factors["model"].T
        selected_common = np.argmax(np.abs(y_common) ** 2, axis=1)
        selected_model = np.argmax(np.abs(y_model) ** 2, axis=1)
        p_cc = cdf_for_selected_port(
            y_common,
            selected_common,
            mean_observation,
            mean_data,
            common_gain,
            common_omega,
            target_snr_linear,
        )
        p_cm = cdf_for_selected_port(
            y_common,
            selected_common,
            mean_observation,
            mean_data,
            model_gain,
            model_omega,
            target_snr_linear,
        )
        p_mc = cdf_for_selected_port(
            y_model,
            selected_model,
            mean_observation,
            mean_data,
            common_gain,
            common_omega,
            target_snr_linear,
        )
        p_mm = cdf_for_selected_port(
            y_model,
            selected_model,
            mean_observation,
            mean_data,
            model_gain,
            model_omega,
            target_snr_linear,
        )
        observation = 0.5 * ((p_cc - p_mc) + (p_cm - p_mm))
        conditional = 0.5 * ((p_cc - p_cm) + (p_mc - p_mm))
        values = {
            "common_y_common_conditional": p_cc,
            "common_y_model_conditional": p_cm,
            "model_y_common_conditional": p_mc,
            "model_y_model_conditional": p_mm,
            "total_common_minus_model": p_cc - p_mm,
            "observation_law_contribution": observation,
            "conditional_aging_contribution": conditional,
        }
        accumulate(sum_value, sum_square, values)
        completed += size

    means = {name: sum_value[name] / count for name in names}
    standard_errors = {
        name: float(
            np.sqrt(
                max(sum_square[name] / count - means[name] ** 2, 0.0) / count
            )
        )
        for name in names
    }
    total = means["total_common_minus_model"]
    absolute_contribution_sum = (
        abs(means["observation_law_contribution"])
        + abs(means["conditional_aging_contribution"])
    )
    share_is_defined = (
        absolute_contribution_sum > 10.0 * np.finfo(float).eps
    )
    return {
        "sample_count": count,
        "hybrid_outage_probabilities": {
            name: means[name] for name in names[:4]
        },
        "common_minus_model": total,
        "observation_law_contribution": means["observation_law_contribution"],
        "conditional_aging_contribution": means[
            "conditional_aging_contribution"
        ],
        "decomposition_residual": total
        - means["observation_law_contribution"]
        - means["conditional_aging_contribution"],
        "absolute_contribution_shares": {
            "observation_law": (
                abs(means["observation_law_contribution"])
                / absolute_contribution_sum
                if share_is_defined
                else 0.0
            ),
            "conditional_aging": (
                abs(means["conditional_aging_contribution"])
                / absolute_contribution_sum
                if share_is_defined
                else 0.0
            ),
        },
        "standard_errors": {
            name: standard_errors[name]
            for name in (
                "total_common_minus_model",
                "observation_law_contribution",
                "conditional_aging_contribution",
            )
        },
    }


def main() -> None:
    count = 2_000_000
    rows: list[dict[str, object]] = []
    for scenario in literature_scenarios():
        for angle_deg in (0.0, 90.0):
            stats = direction_statistics(
                scenario.geometry,
                np.asarray(scenario.velocity_tx),
                np.asarray(scenario.velocity_rx),
                fas_axis(angle_deg),
                order=6,
                maximum_panel_size_m=10.0,
            )
            observed_ports = 6
            positions, times = electronic_return_coordinates(
                observed_ports, 100.0e-6
            )
            covariances = model_covariances(stats, positions, times)
            mean_channel = los_mean(
                scenario,
                fas_axis(angle_deg),
                positions,
                times,
            )
            for model_index, model in enumerate(
                ("independent_AoD_AoA", "fully_separable")
            ):
                result = decomposition(
                    covariances["common_scatterer"],
                    covariances[model],
                    mean_channel,
                    observed_ports,
                    scenario.rician_k,
                    count,
                    seed=(
                        20260913
                        + int(angle_deg)
                        + 10 * observed_ports
                        + 100 * model_index
                        + (800 if scenario.rician_k > 0.0 else 900)
                    ),
                )
                rows.append(
                    {
                        "scenario": scenario.name,
                        "fas_orientation": (
                            "longitudinal" if angle_deg == 0.0 else "transverse"
                        ),
                        "observed_ports_k": observed_ports,
                        "scan_duration_us": (observed_ports - 1) * 100.0,
                        "model": model,
                        **result,
                    }
                )

    payload = {
        "protocol": "electronic_return_if_needed",
        "protocol_definition": (
            "The last observed port transmits at the final-pilot time; every "
            "other selected port transmits one switching interval later."
        ),
        "decomposition": (
            "Shapley average of two replacement orders: observation "
            "distribution and conditional data-channel law."
        ),
        "rows": rows,
    }
    output_directory = Path(__file__).resolve().parents[1] / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "literature_anchored_outage_decomposition.json"
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "row_count": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
