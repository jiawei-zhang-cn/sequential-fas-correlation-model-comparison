"""Propagation scenarios and sequential-FAS settings used in the paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from correlation_models import cross_covariance, separable_covariance
from propagation_model import Geometry


WAVELENGTH_M = 0.0508
APERTURE_WAVELENGTHS = 2.0
SWITCH_TIME_S = 100.0e-6
PILOT_SNR_LINEAR = 10.0
DATA_SNR_LINEAR = 10.0
TARGET_SNR_LINEAR = 10.0 ** (5.0 / 10.0)
QUADRATURE_PANEL_SIZE_M = 10.0


@dataclass(frozen=True)
class Scenario:
    name: str
    source_figure: str
    geometry: Geometry
    velocity_tx: tuple[float, float]
    velocity_rx: tuple[float, float]
    rician_k: float


def literature_scenarios() -> tuple[Scenario, Scenario]:
    same_direction = Scenario(
        name="yoo_fig8_same_direction",
        source_figure="Yoo et al. TWC 2018, Table I, Fig. 8 column",
        geometry=Geometry(
            name="yoo_fig8_two_roadside_rectangles",
            tx=(-200.0, -8.75),
            rx=(200.0, -8.75),
            rectangles=(
                (-263.917, 276.045, 18.364, 106.396),
                (-263.146, 277.483, -103.747, -20.605),
            ),
        ),
        velocity_tx=(105.0 / 3.6, 0.0),
        velocity_rx=(105.0 / 3.6, 0.0),
        rician_k=1.535,
    )
    opposite_direction = Scenario(
        name="yoo_fig9_opposite_direction",
        source_figure="Yoo et al. TWC 2018, Table I, Fig. 9 column",
        geometry=Geometry(
            name="yoo_fig9_two_roadside_rectangles",
            tx=(-50.0, -1.75),
            rx=(50.0, 1.75),
            rectangles=(
                (-58.557, 58.753, 8.000, 13.351),
                (-58.658, 57.919, -19.114, -8.003),
            ),
        ),
        velocity_tx=(32.8 / 3.6, 0.0),
        velocity_rx=(-38.0 / 3.6, 0.0),
        rician_k=0.0,
    )
    return same_direction, opposite_direction


def fas_axis(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array((np.cos(angle), np.sin(angle)))


def sample_coordinates(
    observed_ports: int, switch_time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    if observed_ports < 2:
        raise ValueError("At least two observed ports are required.")
    if switch_time_s < 0.0:
        raise ValueError("The switching time cannot be negative.")
    half_aperture = APERTURE_WAVELENGTHS * WAVELENGTH_M / 2.0
    observed_positions = np.linspace(-half_aperture, half_aperture, observed_ports)
    observed_times = np.arange(observed_ports, dtype=float) * switch_time_s
    data_time = observed_times[-1]
    positions = np.concatenate((observed_positions, observed_positions))
    times = np.concatenate((observed_times, np.full(observed_ports, data_time)))
    return positions, times


def model_covariances(
    stats: dict[str, np.ndarray | float], positions: np.ndarray, times: np.ndarray
) -> dict[str, np.ndarray]:
    common = cross_covariance(
        stats,
        WAVELENGTH_M,
        positions,
        times,
        positions,
        times,
        independent=False,
    )
    independent = cross_covariance(
        stats,
        WAVELENGTH_M,
        positions,
        times,
        positions,
        times,
        independent=True,
    )
    separable = separable_covariance(stats, WAVELENGTH_M, positions, times)
    return {
        "common_scatterer": common,
        "independent_AoD_AoA": independent,
        "fully_separable": separable,
    }


def los_mean(
    scenario: Scenario,
    axis: np.ndarray,
    positions: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Return the normalized LoS mean implied by the V2V geometry."""
    if scenario.rician_k == 0.0:
        return np.zeros_like(times, dtype=complex)
    tx = np.asarray(scenario.geometry.tx, dtype=float)
    rx = np.asarray(scenario.geometry.rx, dtype=float)
    displacement = rx - tx
    distance = float(np.linalg.norm(displacement))
    direction = displacement / distance
    velocity_tx = np.asarray(scenario.velocity_tx)
    velocity_rx = np.asarray(scenario.velocity_rx)
    doppler_length_rate = float(direction @ (velocity_tx - velocity_rx))
    spatial_projection = float(direction @ axis)
    wave_number = 2.0 * np.pi / WAVELENGTH_M
    phase = wave_number * (
        doppler_length_rate * times - spatial_projection * positions - distance
    )
    amplitude = np.sqrt(scenario.rician_k / (scenario.rician_k + 1.0))
    return amplitude * np.exp(1j * phase)
