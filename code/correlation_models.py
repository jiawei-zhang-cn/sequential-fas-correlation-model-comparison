"""Correlation constructions compared in the paper."""

from __future__ import annotations

import numpy as np


def cross_covariance(
    stats: dict[str, np.ndarray | float],
    wavelength: float,
    positions_a: np.ndarray,
    times_a: np.ndarray,
    positions_b: np.ndarray,
    times_b: np.ndarray,
    independent: bool,
) -> np.ndarray:
    """Return the common-scatterer or AoD/AoA-independent covariance."""
    weights = np.asarray(stats["weights"])
    doppler_tx = np.asarray(stats["doppler_tx"])
    doppler_rx = np.asarray(stats["doppler_rx"])
    doppler_total = np.asarray(stats["doppler_total"])
    spatial_projection = np.asarray(stats["spatial_projection"])
    wave_number = 2.0 * np.pi / wavelength
    time_difference = times_a[:, None] - times_b[None, :]
    position_difference = positions_a[:, None] - positions_b[None, :]

    if not independent:
        phases = np.exp(
            1j
            * wave_number
            * (
                doppler_total[:, None, None] * time_difference[None, ...]
                - spatial_projection[:, None, None]
                * position_difference[None, ...]
            )
        )
        return np.einsum("s,sij->ij", weights, phases)

    transmit_factor = np.exp(
        1j * wave_number * doppler_tx[:, None, None] * time_difference[None, ...]
    )
    receive_factor = np.exp(
        -1j
        * wave_number
        * (
            doppler_rx[:, None, None] * time_difference[None, ...]
            + spatial_projection[:, None, None]
            * position_difference[None, ...]
        )
    )
    return np.einsum("s,sij->ij", weights, transmit_factor) * np.einsum(
        "s,sij->ij", weights, receive_factor
    )


def separable_covariance(
    stats: dict[str, np.ndarray | float],
    wavelength: float,
    positions: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Build the product of matched spatial and temporal marginals."""
    weights = np.asarray(stats["weights"])
    doppler_total = np.asarray(stats["doppler_total"])
    spatial_projection = np.asarray(stats["spatial_projection"])
    wave_number = 2.0 * np.pi / wavelength
    time_difference = times[:, None] - times[None, :]
    position_difference = positions[:, None] - positions[None, :]
    temporal = np.exp(
        1j * wave_number * time_difference[..., None] * doppler_total
    ).dot(weights)
    spatial = np.exp(
        -1j * wave_number * position_difference[..., None] * spatial_projection
    ).dot(weights)
    return temporal * spatial


def nearest_psd(covariance: np.ndarray) -> np.ndarray:
    """Remove roundoff-level negative eigenvalues before sampling."""
    covariance = (covariance + covariance.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(1.0e-12, float(eigenvalues.max()) * 1.0e-12)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.conj().T


def covariance_factor(covariance: np.ndarray) -> np.ndarray:
    """Return a square-root factor for a Hermitian covariance matrix."""
    covariance = nearest_psd(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
