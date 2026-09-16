"""Track outage-mismatch mechanisms over the literature switching-delay axis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_scenarios import (
    fas_axis,
    literature_scenarios,
    los_mean,
    model_covariances,
)
from propagation_model import direction_statistics
from literature_anchored_outage_decomposition import (
    decomposition,
    electronic_return_coordinates,
)


def mechanism_relation(observation: float, conditional: float) -> str:
    if abs(observation) + abs(conditional) < 1.0e-12:
        return "identical"
    if observation * conditional < 0.0:
        return "opposing"
    return "same_direction"


def dominant_mechanism(observation: float, conditional: float) -> str:
    if abs(observation) + abs(conditional) < 1.0e-12:
        return "none"
    if abs(conditional) > abs(observation):
        return "conditional_aging"
    return "observation_selection"


def cancellation_index(
    total: float, observation: float, conditional: float
) -> float:
    component_sum = abs(observation) + abs(conditional)
    if component_sum <= 10.0 * np.finfo(float).eps:
        return 0.0
    value = 1.0 - abs(total) / component_sum
    return float(np.clip(value, 0.0, 1.0))


def applicability_class(
    net_relative_error: float,
    internal_relative_mismatch: float,
    tolerance: float,
) -> str:
    if internal_relative_mismatch < tolerance:
        return "true_low_mismatch"
    if net_relative_error < tolerance:
        return "masked_low_net_error"
    return "visible_error"


def first_crossing(
    rows: list[dict[str, object]], level: float
) -> dict[str, object] | None:
    return next(
        (
            row
            for row in rows
            if float(row["net_relative_error"]) >= level
        ),
        None,
    )


def main() -> None:
    observed_ports = 6
    sample_count = 500_000
    switching_delays_us = tuple(range(0, 101, 10))
    models = ("independent_AoD_AoA", "fully_separable")
    rows: list[dict[str, object]] = []

    for scenario_index, scenario in enumerate(literature_scenarios()):
        for orientation_index, angle_deg in enumerate((0.0, 90.0)):
            stats = direction_statistics(
                scenario.geometry,
                np.asarray(scenario.velocity_tx),
                np.asarray(scenario.velocity_rx),
                fas_axis(angle_deg),
                order=6,
                maximum_panel_size_m=10.0,
            )
            for switch_time_us in switching_delays_us:
                positions, times = electronic_return_coordinates(
                    observed_ports, switch_time_us * 1.0e-6
                )
                covariances = model_covariances(stats, positions, times)
                mean_channel = los_mean(
                    scenario,
                    fas_axis(angle_deg),
                    positions,
                    times,
                )
                seed = (
                    20260914
                    + 10_000 * scenario_index
                    + 1_000 * orientation_index
                    + switch_time_us
                )
                for model in models:
                    result = decomposition(
                        covariances["common_scatterer"],
                        covariances[model],
                        mean_channel,
                        observed_ports,
                        scenario.rician_k,
                        sample_count,
                        seed,
                    )
                    total = float(result["common_minus_model"])
                    observation = float(result["observation_law_contribution"])
                    conditional = float(result["conditional_aging_contribution"])
                    common_outage = float(
                        result["hybrid_outage_probabilities"][
                            "common_y_common_conditional"
                        ]
                    )
                    net_relative_error = (
                        abs(total) / common_outage
                        if common_outage > 0.0
                        else 0.0
                    )
                    internal_relative_mismatch = (
                        (abs(observation) + abs(conditional))
                        / common_outage
                        if common_outage > 0.0
                        else 0.0
                    )
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "fas_orientation": (
                                "longitudinal"
                                if angle_deg == 0.0
                                else "transverse"
                            ),
                            "model": model,
                            "observed_ports_k": observed_ports,
                            "switch_time_us": switch_time_us,
                            "total_scan_time_us": (
                                observed_ports - 1
                            )
                            * switch_time_us,
                            "common_outage": common_outage,
                            "common_minus_model": total,
                            "net_relative_error": net_relative_error,
                            "internal_relative_mismatch": (
                                internal_relative_mismatch
                            ),
                            "cancellation_index": cancellation_index(
                                total, observation, conditional
                            ),
                            "applicability_classes": {
                                f"{int(100 * tolerance)}_percent": (
                                    applicability_class(
                                        net_relative_error,
                                        internal_relative_mismatch,
                                        tolerance,
                                    )
                                )
                                for tolerance in (0.01, 0.05, 0.10)
                            },
                            "observation_law_contribution": observation,
                            "conditional_aging_contribution": conditional,
                            "conditional_absolute_share": result[
                                "absolute_contribution_shares"
                            ]["conditional_aging"],
                            "mechanism_relation": mechanism_relation(
                                observation, conditional
                            ),
                            "dominant_mechanism": dominant_mechanism(
                                observation, conditional
                            ),
                            "standard_errors": result["standard_errors"],
                        }
                    )

    summaries: list[dict[str, object]] = []
    for scenario in literature_scenarios():
        for orientation in ("longitudinal", "transverse"):
            for model in models:
                subset = [
                    row
                    for row in rows
                    if row["scenario"] == scenario.name
                    and row["fas_orientation"] == orientation
                    and row["model"] == model
                ]
                positive_delay = [
                    row for row in subset if int(row["switch_time_us"]) > 0
                ]
                row_100 = next(
                    row
                    for row in subset
                    if int(row["switch_time_us"]) == 100
                )
                crossing_5 = first_crossing(subset, 0.05)
                crossing_10 = first_crossing(subset, 0.10)
                summaries.append(
                    {
                        "scenario": scenario.name,
                        "fas_orientation": orientation,
                        "model": model,
                        "first_5_percent_switch_time_us": (
                            crossing_5["switch_time_us"]
                            if crossing_5 is not None
                            else None
                        ),
                        "first_10_percent_switch_time_us": (
                            crossing_10["switch_time_us"]
                            if crossing_10 is not None
                            else None
                        ),
                        "conditional_dominant_positive_delay_points": sum(
                            row["dominant_mechanism"] == "conditional_aging"
                            for row in positive_delay
                        ),
                        "opposing_contribution_positive_delay_points": sum(
                            row["mechanism_relation"] == "opposing"
                            for row in positive_delay
                        ),
                        "positive_delay_point_count": len(positive_delay),
                        "at_100_us": {
                            "net_relative_error": row_100[
                                "net_relative_error"
                            ],
                            "internal_relative_mismatch": row_100[
                                "internal_relative_mismatch"
                            ],
                            "cancellation_index": row_100[
                                "cancellation_index"
                            ],
                            "common_minus_model": row_100[
                                "common_minus_model"
                            ],
                            "observation_law_contribution": row_100[
                                "observation_law_contribution"
                            ],
                            "conditional_aging_contribution": row_100[
                                "conditional_aging_contribution"
                            ],
                            "conditional_absolute_share": row_100[
                                "conditional_absolute_share"
                            ],
                            "mechanism_relation": row_100[
                                "mechanism_relation"
                            ],
                        },
                    }
                )

    classification_summaries: list[dict[str, object]] = []
    for model in models:
        positive_delay = [
            row
            for row in rows
            if row["model"] == model and int(row["switch_time_us"]) > 0
        ]
        for tolerance in (0.01, 0.05, 0.10):
            tolerance_name = f"{int(100 * tolerance)}_percent"
            counts = {
                class_name: sum(
                    row["applicability_classes"][tolerance_name]
                    == class_name
                    for row in positive_delay
                )
                for class_name in (
                    "true_low_mismatch",
                    "masked_low_net_error",
                    "visible_error",
                )
            }
            masked = [
                row
                for row in positive_delay
                if row["applicability_classes"][tolerance_name]
                == "masked_low_net_error"
            ]
            classification_summaries.append(
                {
                    "model": model,
                    "tolerance": tolerance,
                    "positive_delay_point_count": len(positive_delay),
                    "class_counts": counts,
                    "masked_internal_relative_mismatch_max": (
                        max(
                            float(row["internal_relative_mismatch"])
                            for row in masked
                        )
                        if masked
                        else None
                    ),
                    "masked_cancellation_index_max": (
                        max(float(row["cancellation_index"]) for row in masked)
                        if masked
                        else None
                    ),
                }
            )

    payload = {
        "protocol": "electronic_return_if_needed",
        "protocol_definition": (
            "The last observed port transmits at the final-pilot time; every "
            "other selected port transmits one switching interval later."
        ),
        "source_for_switching_axis": (
            "Dinis and Wichman, IEEE Communications Letters 2026, Fig. 2"
        ),
        "observed_ports_k": observed_ports,
        "sample_count_per_model_case": sample_count,
        "switching_delays_us": switching_delays_us,
        "metric_definitions": {
            "net_relative_error": (
                "abs(common_minus_model) / common_outage"
            ),
            "internal_relative_mismatch": (
                "(abs(observation_contribution) + "
                "abs(conditional_contribution)) / common_outage"
            ),
            "cancellation_index": (
                "1 - abs(total_difference) / sum_abs_contributions"
            ),
        },
        "classification_summaries": classification_summaries,
        "summaries": summaries,
        "rows": rows,
    }
    output_directory = Path(__file__).resolve().parents[1] / "results"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "literature_anchored_decomposition_switch_sweep.json"
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "classification_summaries": classification_summaries,
                "summaries": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
