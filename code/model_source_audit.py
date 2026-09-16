"""Source-to-code numerical audit for the propagation model.

The checks in this file do not establish empirical validity.  They verify that
the implemented covariance follows the stated local plane-wave geometry, that
the comparison covariances have the intended limiting cases, and that all
matrices used by the reliability simulation are valid covariance matrices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from .correlation_models import cross_covariance, separable_covariance
    from .paper_scenarios import (
        SWITCH_TIME_S,
        WAVELENGTH_M,
        fas_axis,
        literature_scenarios,
        los_mean,
        model_covariances,
        sample_coordinates,
    )
    from .propagation_model import Geometry, direction_statistics
except ImportError:
    from correlation_models import cross_covariance, separable_covariance
    from paper_scenarios import (
        SWITCH_TIME_S,
        WAVELENGTH_M,
        fas_axis,
        literature_scenarios,
        los_mean,
        model_covariances,
        sample_coordinates,
    )
    from propagation_model import Geometry, direction_statistics


def split_geometry(geometry: Geometry, maximum_panel_size: float) -> Geometry:
    panels = []
    for x_min, x_max, y_min, y_max in geometry.rectangles:
        x_count = int(np.ceil((x_max - x_min) / maximum_panel_size))
        y_count = int(np.ceil((y_max - y_min) / maximum_panel_size))
        x_edges = np.linspace(x_min, x_max, x_count + 1)
        y_edges = np.linspace(y_min, y_max, y_count + 1)
        panels.extend(
            (x_edges[ix], x_edges[ix + 1], y_edges[iy], y_edges[iy + 1])
            for ix in range(x_count)
            for iy in range(y_count)
        )
    return Geometry(
        name=f"{geometry.name}_panels_{maximum_panel_size:g}m",
        tx=geometry.tx,
        rx=geometry.rx,
        rectangles=tuple(panels),
    )


def common_covariance_memory_light(
    stats: dict[str, np.ndarray | float],
    positions: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(stats["weights"])
    doppler = np.asarray(stats["doppler_total"])
    spatial = np.asarray(stats["spatial_projection"])
    wave_number = 2.0 * np.pi / WAVELENGTH_M
    covariance = np.empty((positions.size, positions.size), dtype=complex)
    for row in range(positions.size):
        for column in range(positions.size):
            phase = wave_number * (
                doppler * (times[row] - times[column])
                - spatial * (positions[row] - positions[column])
            )
            covariance[row, column] = np.sum(weights * np.exp(1j * phase))
    return covariance


def exact_path_covariance(
    stats: dict[str, np.ndarray | float],
    tx: np.ndarray,
    rx: np.ndarray,
    velocity_tx: np.ndarray,
    velocity_rx: np.ndarray,
    axis: np.ndarray,
    positions: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Evaluate covariance from finite path lengths at every sample point."""
    nodes = np.asarray(stats["nodes"])
    weights = np.asarray(stats["weights"])
    wave_number = 2.0 * np.pi / WAVELENGTH_M
    phase_vectors = []
    for position, time in zip(positions, times, strict=True):
        tx_now = tx + velocity_tx * time
        rx_now = rx + velocity_rx * time + axis * position
        path_length = np.linalg.norm(nodes - tx_now, axis=1) + np.linalg.norm(
            rx_now - nodes, axis=1
        )
        phase_vectors.append(np.exp(-1j * wave_number * path_length))
    features = np.column_stack(phase_vectors)
    return np.einsum("si,s,sj->ij", features, weights, features.conj())


def exact_los_vector(
    tx: np.ndarray,
    rx: np.ndarray,
    velocity_tx: np.ndarray,
    velocity_rx: np.ndarray,
    axis: np.ndarray,
    positions: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    wave_number = 2.0 * np.pi / WAVELENGTH_M
    values = []
    for position, time in zip(positions, times, strict=True):
        tx_now = tx + velocity_tx * time
        rx_now = rx + velocity_rx * time + axis * position
        distance = np.linalg.norm(rx_now - tx_now)
        values.append(np.exp(-1j * wave_number * distance))
    return np.asarray(values)


def matrix_checks(covariance: np.ndarray) -> dict[str, float]:
    hermitian = (covariance + covariance.conj().T) / 2.0
    return {
        "hermitian_residual": float(
            np.max(np.abs(covariance - covariance.conj().T))
        ),
        "unit_diagonal_residual": float(
            np.max(np.abs(np.diag(covariance) - 1.0))
        ),
        "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(hermitian))),
    }


def audit_case(scenario, angle_deg: float, observed_ports: int) -> dict[str, object]:
    axis = fas_axis(angle_deg)
    velocity_tx = np.asarray(scenario.velocity_tx)
    velocity_rx = np.asarray(scenario.velocity_rx)
    tx = np.asarray(scenario.geometry.tx)
    rx = np.asarray(scenario.geometry.rx)
    stats = direction_statistics(
        scenario.geometry, velocity_tx, velocity_rx, axis, order=128
    )
    intermediate_stats = direction_statistics(
        scenario.geometry, velocity_tx, velocity_rx, axis, order=72
    )
    production_stats = direction_statistics(
        scenario.geometry, velocity_tx, velocity_rx, axis, order=48
    )
    positions, times = sample_coordinates(observed_ports, SWITCH_TIME_S)
    covariances = model_covariances(stats, positions, times)
    production_common = cross_covariance(
        production_stats,
        WAVELENGTH_M,
        positions,
        times,
        positions,
        times,
        independent=False,
    )
    intermediate_common = cross_covariance(
        intermediate_stats,
        WAVELENGTH_M,
        positions,
        times,
        positions,
        times,
        independent=False,
    )

    finite_path = exact_path_covariance(
        stats, tx, rx, velocity_tx, velocity_rx, axis, positions, times
    )
    local_plane_wave = covariances["common_scatterer"]

    spatial_only_times = np.zeros_like(times)
    common_spatial = cross_covariance(
        stats,
        WAVELENGTH_M,
        positions,
        spatial_only_times,
        positions,
        spatial_only_times,
        independent=False,
    )
    independent_spatial = cross_covariance(
        stats,
        WAVELENGTH_M,
        positions,
        spatial_only_times,
        positions,
        spatial_only_times,
        independent=True,
    )
    separable_spatial = separable_covariance(
        stats, WAVELENGTH_M, positions, spatial_only_times
    )

    fixed_positions = np.zeros_like(positions)
    common_temporal = cross_covariance(
        stats,
        WAVELENGTH_M,
        fixed_positions,
        times,
        fixed_positions,
        times,
        independent=False,
    )
    separable_temporal = separable_covariance(
        stats, WAVELENGTH_M, fixed_positions, times
    )

    stationary_tx_stats = direction_statistics(
        scenario.geometry, np.zeros(2), velocity_rx, axis, order=72
    )
    stationary_tx_common = cross_covariance(
        stationary_tx_stats,
        WAVELENGTH_M,
        positions,
        times,
        positions,
        times,
        independent=False,
    )
    stationary_tx_independent = cross_covariance(
        stationary_tx_stats,
        WAVELENGTH_M,
        positions,
        times,
        positions,
        times,
        independent=True,
    )

    los_error = None
    if scenario.rician_k > 0.0:
        linearized_los = los_mean(scenario, axis, positions, times)
        exact_los = exact_los_vector(
            tx, rx, velocity_tx, velocity_rx, axis, positions, times
        )
        amplitude = np.sqrt(scenario.rician_k / (scenario.rician_k + 1.0))
        los_error = float(np.max(np.abs(linearized_los - amplitude * exact_los)))

    return {
        "scenario": scenario.name,
        "fas_angle_deg": angle_deg,
        "observed_ports": observed_ports,
        "scan_duration_s": (observed_ports - 1) * SWITCH_TIME_S,
        "covariance_checks": {
            name: matrix_checks(covariance)
            for name, covariance in covariances.items()
        },
        "local_plane_wave_vs_finite_path_max_error": float(
            np.max(np.abs(local_plane_wave - finite_path))
        ),
        "production_order48_vs_order128_max_error": float(
            np.max(np.abs(production_common - local_plane_wave))
        ),
        "intermediate_order72_vs_order128_max_error": float(
            np.max(np.abs(intermediate_common - local_plane_wave))
        ),
        "los_local_linearization_max_error": los_error,
        "zero_time_common_vs_independent_max_error": float(
            np.max(np.abs(common_spatial - independent_spatial))
        ),
        "zero_time_common_vs_separable_max_error": float(
            np.max(np.abs(common_spatial - separable_spatial))
        ),
        "zero_displacement_common_vs_separable_max_error": float(
            np.max(np.abs(common_temporal - separable_temporal))
        ),
        "stationary_tx_common_vs_independent_max_error": float(
            np.max(np.abs(stationary_tx_common - stationary_tx_independent))
        ),
    }


def isotropic_jakes_check() -> dict[str, float]:
    angles = np.linspace(0.0, 2.0 * np.pi, 200_000, endpoint=False)
    spatial_lags = np.linspace(0.0, 2.0, 21)
    temporal_lags = np.linspace(0.0, 2.0, 21)
    from scipy.special import j0

    numerical_spatial = np.mean(
        np.exp(-1j * 2.0 * np.pi * np.outer(spatial_lags, np.cos(angles))),
        axis=1,
    )
    numerical_temporal = np.mean(
        np.exp(1j * 2.0 * np.pi * np.outer(temporal_lags, np.cos(angles))),
        axis=1,
    )
    return {
        "spatial_j0_max_error": float(
            np.max(np.abs(numerical_spatial - j0(2.0 * np.pi * spatial_lags)))
        ),
        "temporal_j0_max_error": float(
            np.max(np.abs(numerical_temporal - j0(2.0 * np.pi * temporal_lags)))
        ),
    }


def composite_quadrature_convergence_check() -> dict[str, float]:
    scenario = literature_scenarios()[0]
    axis = fas_axis(0.0)
    velocity_tx = np.asarray(scenario.velocity_tx)
    velocity_rx = np.asarray(scenario.velocity_rx)
    positions, times = sample_coordinates(6, SWITCH_TIME_S)

    covariances = {}
    for panel_size in (20.0, 10.0, 5.0):
        panel_geometry = split_geometry(scenario.geometry, panel_size)
        stats = direction_statistics(
            panel_geometry, velocity_tx, velocity_rx, axis, order=6
        )
        covariances[panel_size] = common_covariance_memory_light(
            stats, positions, times
        )

    unsplit_48 = direction_statistics(
        scenario.geometry, velocity_tx, velocity_rx, axis, order=48
    )
    unsplit_128 = direction_statistics(
        scenario.geometry, velocity_tx, velocity_rx, axis, order=128
    )
    covariance_48 = common_covariance_memory_light(unsplit_48, positions, times)
    covariance_128 = common_covariance_memory_light(unsplit_128, positions, times)
    reference = covariances[5.0]
    return {
        "case": "yoo_fig8_same_direction_longitudinal_K6",
        "panel20m_vs_panel5m_max_error": float(
            np.max(np.abs(covariances[20.0] - reference))
        ),
        "panel10m_vs_panel5m_max_error": float(
            np.max(np.abs(covariances[10.0] - reference))
        ),
        "unsplit_order48_vs_panel5m_max_error": float(
            np.max(np.abs(covariance_48 - reference))
        ),
        "unsplit_order128_vs_panel5m_max_error": float(
            np.max(np.abs(covariance_128 - reference))
        ),
    }


def main() -> None:
    cases = [
        audit_case(scenario, angle_deg, observed_ports)
        for scenario in literature_scenarios()
        for angle_deg in (0.0, 90.0)
        for observed_ports in (6,)
    ]
    payload = {
        "audit_scope": {
            "propagation_reference": "Yoo et al., IEEE TWC 2018",
            "sequential_protocol_reference": "Dinis and Wichman, IEEE Communications Letters 2026",
            "common_scatterer_status": "Yoo geometry with a local receive-port extension",
            "independence_status": "matched-marginal AoD/AoA-independence ablation",
            "separable_status": "matched-marginal space-time-separable ablation",
            "sample_scope": "the four six-port propagation and FAS-orientation cases used in the paper",
        },
        "isotropic_jakes_check": isotropic_jakes_check(),
        "composite_quadrature_convergence_check": (
            composite_quadrature_convergence_check()
        ),
        "cases": cases,
    }
    output_directory = Path(__file__).resolve().parents[1] / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "model_source_audit.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "case_count": len(cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
