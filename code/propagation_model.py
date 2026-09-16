"""Roadside-scatterer geometry and quadrature for the paper model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class Geometry:
    name: str
    tx: tuple[float, float]
    rx: tuple[float, float]
    rectangles: tuple[tuple[float, float, float, float], ...]


def rectangle_quadrature(
    rectangles: tuple[tuple[float, float, float, float], ...],
    order: int = 48,
    maximum_panel_size_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and normalized weights for uniform rectangular regions."""
    roots, base_weights = leggauss(order)
    if maximum_panel_size_m is not None and maximum_panel_size_m <= 0.0:
        raise ValueError("The maximum quadrature panel size must be positive.")

    integration_rectangles = []
    for x_min, x_max, y_min, y_max in rectangles:
        if maximum_panel_size_m is None:
            integration_rectangles.append((x_min, x_max, y_min, y_max))
            continue
        x_count = int(np.ceil((x_max - x_min) / maximum_panel_size_m))
        y_count = int(np.ceil((y_max - y_min) / maximum_panel_size_m))
        x_edges = np.linspace(x_min, x_max, x_count + 1)
        y_edges = np.linspace(y_min, y_max, y_count + 1)
        integration_rectangles.extend(
            (
                x_edges[ix],
                x_edges[ix + 1],
                y_edges[iy],
                y_edges[iy + 1],
            )
            for ix in range(x_count)
            for iy in range(y_count)
        )

    total_area = sum(
        (x_max - x_min) * (y_max - y_min)
        for x_min, x_max, y_min, y_max in integration_rectangles
    )
    if total_area <= 0.0:
        raise ValueError("The total scatterer-region area must be positive.")

    all_nodes: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for x_min, x_max, y_min, y_max in integration_rectangles:
        if not (x_min < x_max and y_min < y_max):
            raise ValueError(
                f"Invalid rectangle: {(x_min, x_max, y_min, y_max)}"
            )
        x = (x_min + x_max) / 2.0 + (x_max - x_min) * roots / 2.0
        y = (y_min + y_max) / 2.0 + (y_max - y_min) * roots / 2.0
        wx = (x_max - x_min) * base_weights / 2.0
        wy = (y_max - y_min) * base_weights / 2.0
        xx, yy = np.meshgrid(x, y, indexing="xy")
        wxx, wyy = np.meshgrid(wx, wy, indexing="xy")
        all_nodes.append(np.column_stack((xx.ravel(), yy.ravel())))
        all_weights.append((wxx * wyy).ravel() / total_area)

    nodes = np.vstack(all_nodes)
    weights = np.concatenate(all_weights)
    weights /= weights.sum()
    return nodes, weights


def direction_statistics(
    geometry: Geometry,
    velocity_tx: np.ndarray,
    velocity_rx: np.ndarray,
    fas_axis: np.ndarray,
    order: int = 48,
    maximum_panel_size_m: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Evaluate scatterer directions, Doppler terms, and FAS projections."""
    nodes, weights = rectangle_quadrature(
        geometry.rectangles,
        order=order,
        maximum_panel_size_m=maximum_panel_size_m,
    )
    tx = np.asarray(geometry.tx, dtype=float)
    rx = np.asarray(geometry.rx, dtype=float)
    delta_tx = nodes - tx
    delta_rx = rx - nodes
    u_tx = delta_tx / np.linalg.norm(delta_tx, axis=1, keepdims=True)
    u_rx = delta_rx / np.linalg.norm(delta_rx, axis=1, keepdims=True)
    doppler_tx = u_tx @ velocity_tx
    doppler_rx = u_rx @ velocity_rx
    doppler_total = doppler_tx - doppler_rx
    spatial_projection = u_rx @ fas_axis

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.sum(weights * values))

    mean_d = weighted_mean(doppler_total)
    mean_y = weighted_mean(spatial_projection)
    centered_d = doppler_total - mean_d
    centered_y = spatial_projection - mean_y
    covariance = weighted_mean(centered_d * centered_y)
    variance_d = weighted_mean(centered_d**2)
    variance_y = weighted_mean(centered_y**2)
    denominator = np.sqrt(max(variance_d * variance_y, 0.0))
    correlation = covariance / denominator if denominator > 0.0 else float("nan")

    return {
        "nodes": nodes,
        "weights": weights,
        "u_tx": u_tx,
        "u_rx": u_rx,
        "doppler_tx": doppler_tx,
        "doppler_rx": doppler_rx,
        "doppler_total": doppler_total,
        "spatial_projection": spatial_projection,
        "covariance_d_y": covariance,
        "correlation_d_y": correlation,
    }
