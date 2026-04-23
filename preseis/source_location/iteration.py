"""Utilities for iterative source-location residual calibration reruns."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .calibration import (
    default_calibration_dir,
    default_catalogue_csv,
)


@dataclass(frozen=True)
class SourceLocationCalibrationConvergenceCriteria:
    """Thresholds used to decide when iterative calibration has stabilized."""

    phase_sigma_tolerance_s: float = 0.001
    mode_correlation_tolerance: float = 0.010
    phase_delay_tolerance_s: float = 0.002
    station_delay_tolerance_s: float = 0.010
    station_phase_interaction_tolerance_s: float = 0.010
    mean_standard_error_relative_tolerance: float = 0.020
    solved_event_count_tolerance: int = 0


@dataclass(frozen=True)
class SourceLocationCalibrationIterationSnapshot:
    """Compact representation of one calibration iteration output."""

    label: str
    output_dir: Path
    calibration_dir: Path
    solved_event_count: int
    mean_map_correlated_whitened_rms: float
    mean_map_standard_error: float
    phase_sigma_models: dict[str, dict[str, float]]
    mode_correlation_models: dict[str, float]
    phase_delay_terms: dict[str, float]
    additive_phase_delay_terms: dict[str, float]
    station_delay_terms: dict[str, float]
    station_phase_interaction_terms: dict[tuple[str, str], float]


def _event_set_slug(event_set_name: str) -> str:
    return event_set_name.lower().replace(" ", "")


def default_event_set_csv(events_dir: Path, event_set_name: str) -> Path:
    """Return the default event-set CSV path for a configured event set."""
    return events_dir / f"events_{_event_set_slug(event_set_name)}.csv"


def load_event_ids_from_event_set_csv(events_csv: Path) -> list[str]:
    """Load event identifiers from an event-set CSV file."""
    events_df = pd.read_csv(events_csv)
    if "event_id" not in events_df.columns:
        raise ValueError(f"Event CSV does not contain 'event_id': {events_csv}")
    event_ids = (
        events_df["event_id"].dropna().astype(str).str.split("/").str[-1].tolist()
    )
    return list(dict.fromkeys(event_ids))


def default_iteration_output_dir(
    iteration_output_prefix_path: Path,
    iteration: int,
    *,
    step_number_offset: int = 0,
    output_label: str | None = None,
) -> Path:
    """Return the output directory for a numbered calibration iteration."""
    step_number = iteration + step_number_offset
    suffix = f"{step_number}"
    if output_label:
        suffix = f"{suffix}_{output_label}"
    return Path(f"{iteration_output_prefix_path}{suffix}")


def default_iteration_summary_csv(iteration_output_prefix_path: Path) -> Path:
    """Return the CSV path used for iterative convergence summaries."""
    return (
        iteration_output_prefix_path.parent
        / f"{iteration_output_prefix_path.name}_convergence_summary.csv"
    )


def default_iteration_summary_md(iteration_output_prefix_path: Path) -> Path:
    """Return the markdown path used for iterative convergence summaries."""
    return (
        iteration_output_prefix_path.parent
        / f"{iteration_output_prefix_path.name}_convergence_summary.md"
    )


def prepare_iteration_config(
    base_config: dict[str, Any],
    *,
    output_dir: Path,
    calibration_dir: Path,
    enable_calibration: bool,
    generate_plots: bool,
) -> dict[str, Any]:
    """Clone a base config and retarget it to one iteration output directory."""
    config = deepcopy(base_config)
    source_location_cfg = config.setdefault("source_location", {})
    calibration_cfg = source_location_cfg.setdefault("calibration", {})
    calibration_cfg["enabled"] = enable_calibration
    calibration_cfg["calibration_dir"] = str(calibration_dir)
    source_location_cfg["output_dir"] = str(output_dir)

    plots_cfg = source_location_cfg.get("plots")
    if isinstance(plots_cfg, dict):
        plots_cfg["generate"] = bool(generate_plots)
        plots_cfg["output_dir"] = str(output_dir / "plots")

    return config


def write_iteration_config(config: dict[str, Any], output_path: Path) -> Path:
    """Write a generated iteration config to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def iteration_output_is_complete(output_dir: Path, event_set_name: str) -> bool:
    """Return True when an iteration output has the core artefacts we need."""
    calibration_dir = default_calibration_dir(output_dir)
    return (
        (calibration_dir / "phase_calibration_summary.csv").exists()
        and (calibration_dir / "station_calibration_summary.csv").exists()
        and (calibration_dir / "station_phase_calibration_summary.csv").exists()
        and (calibration_dir / "mode_correlation_summary.csv").exists()
        and default_catalogue_csv(output_dir, event_set_name).exists()
    )


def _bool_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _safe_float(value: Any) -> float:
    numeric = pd.to_numeric([value], errors="coerce")[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _load_catalogue(output_dir: Path, event_set_name: str) -> pd.DataFrame:
    catalogue_csv = default_catalogue_csv(output_dir, event_set_name)
    if catalogue_csv.exists():
        return pd.read_csv(catalogue_csv)

    summary_csv = output_dir / "source_locations.csv"
    if summary_csv.exists():
        return pd.read_csv(summary_csv)
    return pd.DataFrame()


def _extract_solved_event_count(
    catalogue_df: pd.DataFrame,
    catalogue_scope_df: pd.DataFrame,
) -> int:
    if (
        not catalogue_scope_df.empty
        and "solved_event_count" in catalogue_scope_df.columns
    ):
        return int(
            pd.to_numeric(
                catalogue_scope_df["solved_event_count"], errors="coerce"
            ).iloc[0]
        )

    if "source_location_solved" in catalogue_df.columns:
        return int(
            catalogue_df["source_location_solved"].fillna(False).astype(bool).sum()
        )

    if "location_type" in catalogue_df.columns and "event_id" in catalogue_df.columns:
        solved_df = catalogue_df.loc[catalogue_df["location_type"] == "map"]
        return int(solved_df["event_id"].nunique())

    return 0


def _extract_mean_map_standard_error(catalogue_df: pd.DataFrame) -> float:
    if {"source_location_solved", "map_standard_error"}.issubset(catalogue_df.columns):
        solved_mask = catalogue_df["source_location_solved"].fillna(False).astype(bool)
        values = pd.to_numeric(
            catalogue_df.loc[solved_mask, "map_standard_error"],
            errors="coerce",
        )
        return float(values.mean()) if not values.empty else float("nan")

    if {"location_type", "standard_error"}.issubset(catalogue_df.columns):
        values = pd.to_numeric(
            catalogue_df.loc[catalogue_df["location_type"] == "map", "standard_error"],
            errors="coerce",
        )
        return float(values.mean()) if not values.empty else float("nan")

    return float("nan")


def _extract_mean_map_correlated_whitened_rms(catalogue_df: pd.DataFrame) -> float:
    if {"source_location_solved", "map_correlated_whitened_rms"}.issubset(
        catalogue_df.columns
    ):
        solved_mask = catalogue_df["source_location_solved"].fillna(False).astype(bool)
        values = pd.to_numeric(
            catalogue_df.loc[solved_mask, "map_correlated_whitened_rms"],
            errors="coerce",
        )
        return float(values.mean()) if not values.empty else float("nan")

    if {"location_type", "correlated_whitened_rms"}.issubset(catalogue_df.columns):
        values = pd.to_numeric(
            catalogue_df.loc[
                catalogue_df["location_type"] == "map", "correlated_whitened_rms"
            ],
            errors="coerce",
        )
        return float(values.mean()) if not values.empty else float("nan")

    return float("nan")


def load_iteration_snapshot(
    *,
    label: str,
    output_dir: Path,
    event_set_name: str,
) -> SourceLocationCalibrationIterationSnapshot:
    """Load the calibration and catalogue summaries for one iteration."""
    calibration_dir = default_calibration_dir(output_dir)
    phase_df = pd.read_csv(calibration_dir / "phase_calibration_summary.csv")
    station_df = pd.read_csv(calibration_dir / "station_calibration_summary.csv")
    station_phase_df = pd.read_csv(
        calibration_dir / "station_phase_calibration_summary.csv"
    )
    mode_correlation_csv = calibration_dir / "mode_correlation_summary.csv"
    catalogue_scope_csv = calibration_dir / "catalogue_scope_summary.csv"
    catalogue_scope_df = (
        pd.read_csv(catalogue_scope_csv)
        if catalogue_scope_csv.exists()
        else pd.DataFrame()
    )
    catalogue_df = _load_catalogue(output_dir, event_set_name)
    mode_correlation_df = (
        pd.read_csv(mode_correlation_csv)
        if mode_correlation_csv.exists()
        else pd.DataFrame(columns=["strategy", "mode_correlation_coefficient"])
    )

    phase_sigma_models = {
        str(row["phase"]): {
            "current": _safe_float(row.get("current_sigma_s")),
            "phase_empirical": _safe_float(row.get("empirical_sigma_s")),
            "phase_after_additive_correction": _safe_float(
                row.get("empirical_sigma_after_additive_s")
            ),
            "phase_after_shrinkage_correction": _safe_float(
                row.get("empirical_sigma_after_shrinkage_s")
            ),
        }
        for row in phase_df.to_dict("records")
    }
    phase_delay_column = (
        "phase_delay_term_s"
        if "phase_delay_term_s" in phase_df.columns
        else "mean_residual_s"
    )
    phase_delay_terms = {
        str(row["phase"]): _safe_float(row.get(phase_delay_column))
        for row in phase_df.to_dict("records")
        if pd.notna(row.get(phase_delay_column))
    }
    additive_phase_delay_terms = {
        str(row["phase"]): _safe_float(row.get("additive_phase_term_s"))
        for row in phase_df.to_dict("records")
        if pd.notna(row.get("additive_phase_term_s"))
    }
    station_delay_terms = {
        str(row["station"]): _safe_float(row.get("station_delay_term_s"))
        for row in station_df.to_dict("records")
        if pd.notna(row.get("station_delay_term_s"))
    }
    station_phase_interaction_terms = {
        (str(row["station"]), str(row["phase"])): _safe_float(
            row.get("shrunken_interaction_term_s")
        )
        for row in station_phase_df.to_dict("records")
        if pd.notna(row.get("shrunken_interaction_term_s"))
    }
    mode_correlation_models = {
        str(row["strategy"]): _safe_float(row.get("mode_correlation_coefficient"))
        for row in mode_correlation_df.to_dict("records")
        if pd.notna(row.get("mode_correlation_coefficient"))
    }

    return SourceLocationCalibrationIterationSnapshot(
        label=label,
        output_dir=output_dir,
        calibration_dir=calibration_dir,
        solved_event_count=_extract_solved_event_count(
            catalogue_df, catalogue_scope_df
        ),
        mean_map_correlated_whitened_rms=(
            _extract_mean_map_correlated_whitened_rms(catalogue_df)
        ),
        mean_map_standard_error=_extract_mean_map_standard_error(catalogue_df),
        phase_sigma_models=phase_sigma_models,
        mode_correlation_models=mode_correlation_models,
        phase_delay_terms=phase_delay_terms,
        additive_phase_delay_terms=additive_phase_delay_terms,
        station_delay_terms=station_delay_terms,
        station_phase_interaction_terms=station_phase_interaction_terms,
    )


def _selected_phase_sigma(
    snapshot: SourceLocationCalibrationIterationSnapshot,
    sigma_strategy: str,
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for phase, models in snapshot.phase_sigma_models.items():
        if sigma_strategy in models and pd.notna(models[sigma_strategy]):
            selected[phase] = float(models[sigma_strategy])
    return selected


def _safe_relative_delta(previous: float, current: float) -> float:
    if not np.isfinite(previous) or not np.isfinite(current) or previous == 0.0:
        return float("nan")
    return float((current - previous) / previous)


def _selected_mode_correlation(
    snapshot: SourceLocationCalibrationIterationSnapshot,
    mode_correlation_strategy: str,
) -> float:
    if mode_correlation_strategy in snapshot.mode_correlation_models:
        return float(snapshot.mode_correlation_models[mode_correlation_strategy])
    return float("nan")


def _max_abs_difference(
    previous: dict[Any, float],
    current: dict[Any, float],
) -> float:
    shared_keys = sorted(set(previous) & set(current), key=str)
    if not shared_keys:
        return float("nan")
    return float(max(abs(current[key] - previous[key]) for key in shared_keys))


def _mean_abs_difference(
    previous: dict[Any, float],
    current: dict[Any, float],
) -> float:
    shared_keys = sorted(set(previous) & set(current), key=str)
    if not shared_keys:
        return float("nan")
    return float(np.mean([abs(current[key] - previous[key]) for key in shared_keys]))


def compare_iteration_snapshots(
    previous: SourceLocationCalibrationIterationSnapshot,
    current: SourceLocationCalibrationIterationSnapshot,
    *,
    delay_strategy: str,
    sigma_strategy: str,
    mode_correlation_strategy: str,
) -> dict[str, Any]:
    """Compute convergence metrics between two iteration snapshots."""
    previous_sigma = _selected_phase_sigma(previous, sigma_strategy)
    current_sigma = _selected_phase_sigma(current, sigma_strategy)
    previous_mode_correlation = _selected_mode_correlation(
        previous,
        mode_correlation_strategy,
    )
    current_mode_correlation = _selected_mode_correlation(
        current,
        mode_correlation_strategy,
    )
    previous_interaction_keys = set(previous.station_phase_interaction_terms)
    current_interaction_keys = set(current.station_phase_interaction_terms)

    phase_only_delay_max_abs_delta_s = _max_abs_difference(
        previous.phase_delay_terms,
        current.phase_delay_terms,
    )
    additive_phase_delay_max_abs_delta_s = _max_abs_difference(
        previous.additive_phase_delay_terms,
        current.additive_phase_delay_terms,
    )
    station_delay_max_abs_delta_s = _max_abs_difference(
        previous.station_delay_terms,
        current.station_delay_terms,
    )
    station_phase_interaction_max_abs_delta_s = _max_abs_difference(
        previous.station_phase_interaction_terms,
        current.station_phase_interaction_terms,
    )
    station_phase_interaction_mean_abs_delta_s = _mean_abs_difference(
        previous.station_phase_interaction_terms,
        current.station_phase_interaction_terms,
    )

    selected_phase_delay_max_abs_delta_s = phase_only_delay_max_abs_delta_s
    if delay_strategy in {"additive", "shrinkage"}:
        selected_phase_delay_max_abs_delta_s = additive_phase_delay_max_abs_delta_s

    metrics: dict[str, Any] = {
        "previous_label": previous.label,
        "current_label": current.label,
        "previous_output_dir": str(previous.output_dir),
        "current_output_dir": str(current.output_dir),
        "previous_solved_event_count": previous.solved_event_count,
        "current_solved_event_count": current.solved_event_count,
        "solved_event_count_delta": (
            current.solved_event_count - previous.solved_event_count
        ),
        "previous_mean_map_correlated_whitened_rms": (
            previous.mean_map_correlated_whitened_rms
        ),
        "current_mean_map_correlated_whitened_rms": (
            current.mean_map_correlated_whitened_rms
        ),
        "mean_map_correlated_whitened_rms_delta": (
            current.mean_map_correlated_whitened_rms
            - previous.mean_map_correlated_whitened_rms
        ),
        "mean_map_correlated_whitened_rms_relative_delta": _safe_relative_delta(
            previous.mean_map_correlated_whitened_rms,
            current.mean_map_correlated_whitened_rms,
        ),
        "previous_mean_map_standard_error": previous.mean_map_standard_error,
        "current_mean_map_standard_error": current.mean_map_standard_error,
        "mean_map_standard_error_delta": (
            current.mean_map_standard_error - previous.mean_map_standard_error
        ),
        "mean_map_standard_error_relative_delta": _safe_relative_delta(
            previous.mean_map_standard_error,
            current.mean_map_standard_error,
        ),
        "previous_mode_correlation_coefficient": previous_mode_correlation,
        "current_mode_correlation_coefficient": current_mode_correlation,
        "mode_correlation_delta": (
            current_mode_correlation - previous_mode_correlation
        ),
        "phase_sigma_max_abs_delta_s": _max_abs_difference(
            previous_sigma,
            current_sigma,
        ),
        "phase_only_delay_max_abs_delta_s": phase_only_delay_max_abs_delta_s,
        "additive_phase_delay_max_abs_delta_s": additive_phase_delay_max_abs_delta_s,
        "selected_phase_delay_max_abs_delta_s": (selected_phase_delay_max_abs_delta_s),
        "station_delay_max_abs_delta_s": station_delay_max_abs_delta_s,
        "selected_station_delay_max_abs_delta_s": (
            station_delay_max_abs_delta_s
            if delay_strategy in {"additive", "shrinkage"}
            else float("nan")
        ),
        "station_phase_interaction_max_abs_delta_s": (
            station_phase_interaction_max_abs_delta_s
        ),
        "selected_station_phase_interaction_max_abs_delta_s": (
            station_phase_interaction_max_abs_delta_s
            if delay_strategy == "shrinkage"
            else float("nan")
        ),
        "station_phase_interaction_mean_abs_delta_s": (
            station_phase_interaction_mean_abs_delta_s
        ),
        "shared_station_phase_interaction_count": len(
            previous_interaction_keys & current_interaction_keys
        ),
    }

    for phase in sorted(set(previous_sigma) & set(current_sigma)):
        metrics[f"phase_sigma_delta_{phase}_s"] = (
            current_sigma[phase] - previous_sigma[phase]
        )
    for phase in sorted(
        set(previous.phase_delay_terms) & set(current.phase_delay_terms)
    ):
        metrics[f"phase_only_delay_delta_{phase}_s"] = (
            current.phase_delay_terms[phase] - previous.phase_delay_terms[phase]
        )
    for phase in sorted(
        set(previous.additive_phase_delay_terms)
        & set(current.additive_phase_delay_terms)
    ):
        metrics[f"additive_phase_delay_delta_{phase}_s"] = (
            current.additive_phase_delay_terms[phase]
            - previous.additive_phase_delay_terms[phase]
        )
    return metrics


def evaluate_iteration_convergence(
    metrics: dict[str, Any],
    *,
    criteria: SourceLocationCalibrationConvergenceCriteria,
    delay_strategy: str,
    sigma_strategy: str,
    mode_correlation_strategy: str,
) -> tuple[bool, list[str]]:
    """Evaluate convergence metrics against configured tolerances."""
    blockers: list[str] = []

    if sigma_strategy != "current":
        value = _safe_float(metrics.get("phase_sigma_max_abs_delta_s"))
        if np.isfinite(value) and abs(value) > criteria.phase_sigma_tolerance_s:
            blockers.append(
                "phase sigma delta "
                f"{value:.4f}s > {criteria.phase_sigma_tolerance_s:.4f}s"
            )

    if mode_correlation_strategy != "current":
        value = _safe_float(metrics.get("mode_correlation_delta"))
        if np.isfinite(value) and abs(value) > criteria.mode_correlation_tolerance:
            blockers.append(
                "mode-correlation delta "
                f"{value:.4f} > {criteria.mode_correlation_tolerance:.4f}"
            )

    phase_metric_name = "phase_only_delay_max_abs_delta_s"
    if delay_strategy in {"additive", "shrinkage"}:
        phase_metric_name = "additive_phase_delay_max_abs_delta_s"
    if delay_strategy in {"phase", "additive", "shrinkage"}:
        value = _safe_float(metrics.get(phase_metric_name))
        if np.isfinite(value) and abs(value) > criteria.phase_delay_tolerance_s:
            blockers.append(
                "phase delay delta "
                f"{value:.4f}s > {criteria.phase_delay_tolerance_s:.4f}s"
            )

    if delay_strategy in {"additive", "shrinkage"}:
        value = _safe_float(metrics.get("station_delay_max_abs_delta_s"))
        if np.isfinite(value) and abs(value) > criteria.station_delay_tolerance_s:
            blockers.append(
                "station delay delta "
                f"{value:.4f}s > "
                f"{criteria.station_delay_tolerance_s:.4f}s"
            )

    if delay_strategy == "shrinkage":
        value = _safe_float(metrics.get("station_phase_interaction_max_abs_delta_s"))
        if (
            np.isfinite(value)
            and abs(value) > criteria.station_phase_interaction_tolerance_s
        ):
            blockers.append(
                "station/phase interaction delta "
                f"{value:.4f}s > "
                f"{criteria.station_phase_interaction_tolerance_s:.4f}s"
            )

    solved_delta = int(metrics.get("solved_event_count_delta", 0))
    if abs(solved_delta) > criteria.solved_event_count_tolerance:
        blockers.append(
            "solved event delta "
            f"{solved_delta} exceeds {criteria.solved_event_count_tolerance}"
        )

    stderr_relative_delta = _safe_float(
        metrics.get("mean_map_correlated_whitened_rms_relative_delta")
    )
    if not np.isfinite(stderr_relative_delta):
        stderr_relative_delta = _safe_float(
            metrics.get("mean_map_standard_error_relative_delta")
        )
    if (
        np.isfinite(stderr_relative_delta)
        and abs(stderr_relative_delta) > criteria.mean_standard_error_relative_tolerance
    ):
        blockers.append(
            "mean fit-metric relative delta "
            f"{stderr_relative_delta:.4f} exceeds "
            f"{criteria.mean_standard_error_relative_tolerance:.4f}"
        )

    return (not blockers), blockers


def _format_metric(value: Any, *, digits: int = 4) -> str:
    numeric = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):.{digits}f}"


def build_iteration_summary_markdown(
    *,
    event_set_name: str,
    comparison_df: pd.DataFrame,
    criteria: SourceLocationCalibrationConvergenceCriteria,
) -> str:
    """Render a concise markdown summary of iterative convergence results."""
    lines = [f"# {event_set_name} Iterative Calibration Convergence", ""]
    lines.extend(
        [
            "## Criteria",
            "",
            f"- phase sigma tolerance: {criteria.phase_sigma_tolerance_s:.4f} s",
            (
                "- mode-correlation tolerance: "
                f"{criteria.mode_correlation_tolerance:.4f}"
            ),
            f"- phase delay tolerance: {criteria.phase_delay_tolerance_s:.4f} s",
            (
                "- station delay tolerance: "
                f"{criteria.station_delay_tolerance_s:.4f} s"
            ),
            (
                "- station/phase interaction tolerance: "
                f"{criteria.station_phase_interaction_tolerance_s:.4f} s"
            ),
            (
                "- mean fit-metric relative tolerance: "
                f"{criteria.mean_standard_error_relative_tolerance:.4f}"
            ),
            (
                "- solved event count tolerance: "
                f"{criteria.solved_event_count_tolerance}"
            ),
            "",
        ]
    )

    if comparison_df.empty:
        lines.append("No iteration comparisons were available.")
        return "\n".join(lines)

    lines.extend(["## Iteration Deltas", ""])
    header = (
        "| Step | Delay | Sigma | Mode corr | Previous | Current | Solved Δ | "
        "Mean fit rel Δ | Mode corr Δ | Max phase sigma Δ (s) | "
        "Max selected phase Δ (s) | Max station Δ (s) | "
        "Max interaction Δ (s) | Converged |"
    )
    divider = (
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | --- |"
    )
    lines.extend([header, divider])

    for row in comparison_df.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("step", "n/a")),
                    str(row.get("delay_strategy", "n/a")),
                    str(row.get("sigma_strategy", "n/a")),
                    str(row.get("mode_correlation_strategy", "n/a")),
                    str(row.get("previous_label", "n/a")),
                    str(row.get("current_label", "n/a")),
                    str(int(row.get("solved_event_count_delta", 0))),
                    _format_metric(
                        row.get("mean_map_correlated_whitened_rms_relative_delta"),
                        digits=4,
                    ),
                    _format_metric(row.get("mode_correlation_delta"), digits=4),
                    _format_metric(row.get("phase_sigma_max_abs_delta_s"), digits=4),
                    _format_metric(
                        row.get("selected_phase_delay_max_abs_delta_s"),
                        digits=4,
                    ),
                    _format_metric(
                        row.get("selected_station_delay_max_abs_delta_s"),
                        digits=4,
                    ),
                    _format_metric(
                        row.get("selected_station_phase_interaction_max_abs_delta_s"),
                        digits=4,
                    ),
                    "yes" if bool(row.get("converged", False)) else "no",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Notes", ""])
    for row in comparison_df.to_dict("records"):
        blockers = str(row.get("convergence_blockers", "")).strip()
        if not blockers:
            blockers = "all criteria satisfied"
        lines.append(f"- step {row.get('step', 'n/a')}: {blockers}")

    return "\n".join(lines)
