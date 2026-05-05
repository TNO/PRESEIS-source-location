"""Residual calibration utilities for source-location travel-time analysis."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class SourceLocationCalibrationModel:
    """Calibration terms derived from residual reports."""

    phase_delay_terms: dict[str, float]
    additive_phase_delay_terms: dict[str, float]
    station_delay_terms: dict[str, float]
    station_phase_interaction_terms: dict[tuple[str, str], float]
    phase_sigma_models: dict[str, dict[str, float]]
    mode_correlation_models: dict[str, float]


def _event_set_slug(event_set_name: str) -> str:
    return event_set_name.lower().replace(" ", "")


def default_catalogue_csv(output_dir: Path, event_set_name: str) -> Path:
    """Return the default source-location catalogue path for an event set."""
    return (
        output_dir / f"{_event_set_slug(event_set_name)}_source_location_catalogue.csv"
    )


def default_calibration_dir(output_dir: Path) -> Path:
    """Return the default residual calibration output directory."""
    return output_dir / "residual_calibration"


def default_report_qmd(output_dir: Path, event_set_name: str) -> Path:
    """Return the default QMD report path."""
    return (
        output_dir
        / f"{_event_set_slug(event_set_name)}_residual_calibration_report.qmd"
    )


def _ensure_bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def collect_residual_reports(
    *,
    source_location_dir: Path,
    catalogue_csv: Path | None = None,
    event_ids: list[str] | None = None,
    solved_only: bool = True,
) -> pd.DataFrame:
    """Load residual report CSVs for the requested event scope."""
    if event_ids:
        target_event_ids = list(dict.fromkeys(event_ids))
    elif catalogue_csv is not None and catalogue_csv.exists():
        catalogue_df = pd.read_csv(catalogue_csv)
        if solved_only and "source_location_solved" in catalogue_df.columns:
            solved_mask = (
                catalogue_df["source_location_solved"].fillna(False).astype(bool)
            )
            catalogue_df = catalogue_df.loc[solved_mask]
        target_event_ids = catalogue_df["event_id"].dropna().astype(str).tolist()
    else:
        target_event_ids = sorted(
            path.name
            for path in source_location_dir.iterdir()
            if path.is_dir() and not path.name.startswith("traveltimes")
        )

    frames: list[pd.DataFrame] = []
    for event_id in target_event_ids:
        residual_csv = (
            source_location_dir / event_id / f"{event_id}_residual_report.csv"
        )
        if not residual_csv.exists():
            continue
        frame = pd.read_csv(residual_csv)
        frame["event_id"] = event_id
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = _to_numeric(
        combined,
        [
            "observed_time_s",
            "synthetic_time_s",
            "residual_s",
            "sigma_s",
            "normalized_residual_sigma",
            "pick_probability",
            "probability_threshold",
        ],
    )
    combined["has_observation"] = _ensure_bool_column(combined, "has_observation")
    combined["used_in_inference"] = _ensure_bool_column(combined, "used_in_inference")
    combined["station"] = combined["station"].astype(str)
    combined["phase"] = combined["phase"].astype(str)
    return combined


def _sqrt_mean_square(values: pd.Series) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(clean))))


def _mean_absolute(values: pd.Series) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return float("nan")
    return float(np.mean(np.abs(clean)))


def _coverage_diagnostics(values: pd.Series) -> pd.Series:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return pd.Series(
            {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "p_abs_le_1": np.nan,
                "p_abs_le_2": np.nan,
                "p_abs_gt_3": np.nan,
            }
        )
    series = pd.Series(clean)
    return pd.Series(
        {
            "n": int(clean.size),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "p_abs_le_1": float((np.abs(clean) <= 1.0).mean()),
            "p_abs_le_2": float((np.abs(clean) <= 2.0).mean()),
            "p_abs_gt_3": float((np.abs(clean) > 3.0).mean()),
        }
    )


def _clip_mode_correlation_coefficient(
    value: float,
    *,
    max_abs_value: float = 0.99,
) -> tuple[float, bool]:
    if not np.isfinite(value):
        return float("nan"), False
    clipped = float(np.clip(value, -max_abs_value, max_abs_value))
    return clipped, not np.isclose(clipped, value)


def _summarize_mode_correlation(
    frame: pd.DataFrame,
    *,
    residual_column: str,
    strategy: str,
    notes: str,
    min_mode_correlation_pair_count: int,
) -> dict[str, Any]:
    pair_df = frame.pivot_table(
        index=["event_id", "station"],
        columns="phase",
        values=residual_column,
        aggfunc="first",
    )
    if {"P", "S"}.issubset(pair_df.columns):
        pair_df = pair_df[["P", "S"]].dropna()
    else:
        pair_df = pd.DataFrame(columns=["P", "S"])

    pair_count = int(len(pair_df))
    if pair_count >= 2:
        raw_correlation = float(pair_df["P"].corr(pair_df["S"]))
    else:
        raw_correlation = float("nan")
    mode_correlation_coefficient, clipped = _clip_mode_correlation_coefficient(
        raw_correlation
    )

    return {
        "strategy": strategy,
        "pair_count": pair_count,
        "pair_count_threshold": min_mode_correlation_pair_count,
        "mode_correlation_coefficient": mode_correlation_coefficient,
        "raw_mode_correlation_coefficient": raw_correlation,
        "supported_for_inference": bool(
            pair_count >= min_mode_correlation_pair_count
            and np.isfinite(mode_correlation_coefficient)
        ),
        "clipped_to_valid_range": clipped,
        "notes": notes,
    }


def _fit_additive_delay_terms(
    frame: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    working = frame.loc[:, ["station", "phase", "residual_s"]].dropna().copy()
    if working.empty:
        return {}, {}

    working["station"] = working["station"].astype(str)
    working["phase"] = working["phase"].astype(str)
    phases = sorted(working["phase"].unique().tolist())
    stations = sorted(working["station"].unique().tolist())

    design_columns = [np.ones(len(working), dtype=float)]
    phase_values = working["phase"].to_numpy(dtype=str)
    station_values = working["station"].to_numpy(dtype=str)

    for phase in phases[1:]:
        design_columns.append((phase_values == phase).astype(float))
    for station in stations[1:]:
        design_columns.append((station_values == station).astype(float))

    design_matrix = np.column_stack(design_columns)
    residual_values = working["residual_s"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design_matrix, residual_values, rcond=None)

    intercept = float(coefficients[0])
    phase_terms_raw = {phases[0]: intercept}
    for index, phase in enumerate(phases[1:], start=1):
        phase_terms_raw[phase] = intercept + float(coefficients[index])

    station_terms_raw = {stations[0]: 0.0}
    station_offset = len(phases)
    for index, station in enumerate(stations[1:], start=station_offset):
        station_terms_raw[station] = float(coefficients[index])

    station_counts = working.groupby("station").size().to_dict()
    station_shift = float(
        sum(
            station_counts[station] * station_terms_raw.get(station, 0.0)
            for station in stations
        )
        / len(working)
    )

    phase_terms = {
        phase: float(value + station_shift) for phase, value in phase_terms_raw.items()
    }
    station_terms = {
        station: float(value - station_shift)
        for station, value in station_terms_raw.items()
    }
    return phase_terms, station_terms


def _estimate_shrinkage_interaction_terms(
    frame: pd.DataFrame,
    *,
    phase_sigma_after_additive: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    interaction_summary = (
        frame.groupby(["station", "phase"])
        .agg(
            n=("additive_residual_s", "count"),
            raw_interaction_term_s=("additive_residual_s", "mean"),
            empirical_sigma_after_additive_s=("additive_residual_s", "std"),
            rms_after_additive_s=("additive_residual_s", _sqrt_mean_square),
            mean_abs_after_additive_s=("additive_residual_s", _mean_absolute),
        )
        .reset_index()
    )
    if interaction_summary.empty:
        return interaction_summary, {}

    interaction_summary["phase_sigma_after_additive_s"] = interaction_summary[
        "phase"
    ].map(phase_sigma_after_additive)
    phase_sigma_fallback = pd.to_numeric(
        interaction_summary["empirical_sigma_after_additive_s"], errors="coerce"
    )
    interaction_summary["phase_sigma_after_additive_s"] = pd.to_numeric(
        interaction_summary["phase_sigma_after_additive_s"], errors="coerce"
    ).where(
        lambda s: np.isfinite(s),
        phase_sigma_fallback,
    )
    interaction_summary["observation_variance_s2"] = np.square(
        interaction_summary["phase_sigma_after_additive_s"]
    ) / interaction_summary["n"].clip(lower=1)
    interaction_summary["observation_variance_s2"] = pd.to_numeric(
        interaction_summary["observation_variance_s2"], errors="coerce"
    ).fillna(0.0)

    prior_variances: dict[str, float] = {}
    interaction_summary["shrinkage_prior_variance_s2"] = np.nan
    for phase in sorted(interaction_summary["phase"].dropna().unique()):
        phase_mask = interaction_summary["phase"] == phase
        phase_rows = interaction_summary.loc[phase_mask]
        raw_terms = pd.to_numeric(
            phase_rows["raw_interaction_term_s"], errors="coerce"
        ).to_numpy(dtype=float)
        observation_variances = pd.to_numeric(
            phase_rows["observation_variance_s2"], errors="coerce"
        ).to_numpy(dtype=float)
        tau_squared = float(np.nanmean(np.square(raw_terms) - observation_variances))
        if not np.isfinite(tau_squared) or tau_squared < 0.0:
            tau_squared = 0.0
        prior_variances[str(phase)] = tau_squared
        interaction_summary.loc[phase_mask, "shrinkage_prior_variance_s2"] = tau_squared

    denominator = (
        interaction_summary["shrinkage_prior_variance_s2"]
        + interaction_summary["observation_variance_s2"]
    )
    interaction_summary["shrinkage_factor"] = np.where(
        np.isfinite(denominator) & (denominator > 0.0),
        interaction_summary["shrinkage_prior_variance_s2"] / denominator,
        0.0,
    )
    interaction_summary["shrunken_interaction_term_s"] = (
        interaction_summary["raw_interaction_term_s"]
        * interaction_summary["shrinkage_factor"]
    )
    return interaction_summary, prior_variances


def _shrink_station_delay_terms(
    station_terms: dict[str, float],
    used_df: pd.DataFrame,
    phase_sigma: dict[str, float],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Apply empirical-Bayes shrinkage to OLS per-station delay terms.

    Mirrors :func:`_estimate_shrinkage_interaction_terms` at the station
    main-effect level.  For station *s* with *n_s* observations across all
    phases, the observation variance is

        σ²_obs(s) = mean_obs(σ²_phase) / n_s

    where σ_phase is the empirical phase sigma passed in via *phase_sigma*.
    The between-station prior variance is estimated by the method-of-moments
    estimator

        τ̂² = max(0, mean_s(b_s² − σ²_obs(s)))

    and each raw OLS station term is shrunk toward zero as

        b̂_s = κ_s · b_s,   κ_s = τ² / (τ² + σ²_obs(s)).

    When the between-station signal (τ²) is small compared with the estimation
    noise (σ²_obs), κ approaches 0 and the correction is suppressed.  When the
    signal dominates, κ approaches 1 and the full OLS term is retained.  This
    naturally keeps large, well-supported S-wave station corrections while
    zeroing out noisy P-wave corrections that carry no real signal.
    """
    phase_sigma_sq: dict[str, float] = {
        phase: float(sigma) ** 2 for phase, sigma in phase_sigma.items()
    }
    rows = []
    for station, sub in used_df.groupby("station"):
        n = int(len(sub))
        per_obs_sigma_sq = sub["phase"].astype(str).map(phase_sigma_sq)
        mean_obs_var = float(per_obs_sigma_sq.mean()) / max(n, 1)
        raw_term = float(station_terms.get(str(station), 0.0))
        rows.append(
            {
                "station": str(station),
                "n_station": n,
                "raw_station_delay_s": raw_term,
                "station_obs_variance_s2": mean_obs_var,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return {}, summary

    raw = summary["raw_station_delay_s"].to_numpy(dtype=float)
    obs_var = summary["station_obs_variance_s2"].to_numpy(dtype=float)
    tau_sq = float(np.nanmean(np.square(raw) - obs_var))
    if not np.isfinite(tau_sq) or tau_sq < 0.0:
        tau_sq = 0.0

    summary["station_shrinkage_prior_variance_s2"] = tau_sq
    denom = tau_sq + obs_var
    summary["station_shrinkage_factor"] = np.where(
        np.isfinite(denom) & (denom > 0.0),
        tau_sq / denom,
        0.0,
    )
    summary["shrunken_station_delay_s"] = (
        summary["raw_station_delay_s"] * summary["station_shrinkage_factor"]
    )
    shrunken_terms: dict[str, float] = dict(
        zip(
            summary["station"].astype(str),
            summary["shrunken_station_delay_s"].astype(float),
        )
    )
    return shrunken_terms, summary


def _filter_degenerate_events(
    used_df: pd.DataFrame,
    *,
    max_per_event_mean_residual_s: float = 10.0,
    max_per_event_std_residual_s: float = 50.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove events with degenerate MAP locations from the calibration frame.

    Events whose per-event mean residual exceeds ``max_per_event_mean_residual_s``
    or whose per-event residual standard deviation exceeds
    ``max_per_event_std_residual_s`` have converged to a grid-boundary or otherwise
    physically impossible MAP solution.  Their residuals are hundreds of seconds
    rather than fractions of a second, and including them would corrupt the
    phase/station delay estimates.
    """
    if "event_id" not in used_df.columns:
        return used_df, []

    per_event = used_df.groupby("event_id")["residual_s"].agg(["mean", "std"])
    degenerate_mask = (per_event["mean"].abs() > max_per_event_mean_residual_s) | (
        per_event["std"] > max_per_event_std_residual_s
    )
    degenerate_events = per_event.index[degenerate_mask].tolist()
    if degenerate_events:
        clean_df = used_df.loc[~used_df["event_id"].isin(degenerate_events)].copy()
    else:
        clean_df = used_df
    return clean_df, degenerate_events


def analyze_residual_calibration(
    residual_df: pd.DataFrame,
    *,
    current_mode_correlation_coefficient: float = 0.2,
    min_mode_correlation_pair_count: int = 5,
    max_per_event_mean_residual_s: float = 10.0,
    max_per_event_std_residual_s: float = 50.0,
) -> dict[str, pd.DataFrame]:
    """Build calibration summaries from residual report exports."""
    if residual_df.empty:
        raise ValueError("Residual dataframe is empty")

    observed_df = residual_df.loc[residual_df["has_observation"]].copy()
    used_df = observed_df.loc[observed_df["used_in_inference"]].copy()
    if used_df.empty:
        raise ValueError("No residual rows were used in inference")

    used_df, degenerate_events = _filter_degenerate_events(
        used_df,
        max_per_event_mean_residual_s=max_per_event_mean_residual_s,
        max_per_event_std_residual_s=max_per_event_std_residual_s,
    )
    if degenerate_events:
        warnings.warn(
            f"Excluded {len(degenerate_events)} degenerate event(s) from residual "
            f"calibration (|per-event mean| > {max_per_event_mean_residual_s} s or "
            f"per-event std > {max_per_event_std_residual_s} s): "
            + ", ".join(degenerate_events),
            UserWarning,
            stacklevel=2,
        )
    if used_df.empty:
        raise ValueError("No residual rows remain after filtering degenerate events")

    current_z = used_df["residual_s"] / used_df["sigma_s"]
    global_option_summary = pd.DataFrame(
        [
            {
                "option": "current",
                "delay_strategy": "none",
                "sigma_strategy": "current",
                "combined_mean_normalized_residual": float(current_z.mean()),
                "combined_std_normalized_residual": float(current_z.std()),
                "combined_rms_normalized_residual": _sqrt_mean_square(current_z),
                "notes": "Current fixed phase sigmas with no delay correction.",
            },
            {
                "option": "single_scale_on_current_sigma",
                "delay_strategy": "none",
                "sigma_strategy": "global_scale",
                "combined_mean_normalized_residual": 0.0,
                "combined_std_normalized_residual": 1.0,
                "combined_rms_normalized_residual": 1.0,
                "global_sigma_scale_factor": float(current_z.std()),
                "notes": (
                    "Apply one scale factor to current phase sigmas while preserving "
                    "their P/S ratio."
                ),
            },
        ]
    )

    phase_summary = (
        used_df.groupby("phase")
        .agg(
            n=("residual_s", "count"),
            mean_residual_s=("residual_s", "mean"),
            empirical_sigma_s=("residual_s", "std"),
            rms_residual_s=("residual_s", _sqrt_mean_square),
            mean_abs_residual_s=("residual_s", _mean_absolute),
            mean_normalized_residual_sigma=("normalized_residual_sigma", "mean"),
            std_normalized_residual_sigma=("normalized_residual_sigma", "std"),
            current_sigma_s=("sigma_s", "first"),
        )
        .reset_index()
    )
    phase_summary["sigma_scale_factor"] = (
        phase_summary["empirical_sigma_s"] / phase_summary["current_sigma_s"]
    )

    phase_delay_map = phase_summary.set_index("phase")["mean_residual_s"].to_dict()
    phase_summary["phase_delay_term_s"] = phase_summary["phase"].map(phase_delay_map)

    additive_phase_terms, station_delay_terms = _fit_additive_delay_terms(used_df)

    phase_sigma_empirical = phase_summary.set_index("phase")[
        "empirical_sigma_s"
    ].to_dict()
    shrunken_station_delay_terms, station_shrinkage_info = _shrink_station_delay_terms(
        station_delay_terms, used_df, phase_sigma_empirical
    )

    phase_calibrated = used_df.copy()
    phase_calibrated["phase_delay_term_s"] = phase_calibrated["phase"].map(
        phase_delay_map
    )
    phase_calibrated["empirical_sigma_s"] = phase_calibrated["phase"].map(
        phase_summary.set_index("phase")["empirical_sigma_s"].to_dict()
    )
    phase_calibrated["phase_corrected_residual_s"] = (
        phase_calibrated["residual_s"] - phase_calibrated["phase_delay_term_s"]
    )

    used_additive = used_df.copy()
    used_additive["additive_phase_term_s"] = used_additive["phase"].map(
        additive_phase_terms
    )
    used_additive["station_delay_term_s"] = used_additive["station"].map(
        station_delay_terms
    )
    used_additive["additive_delay_term_s"] = (
        used_additive["additive_phase_term_s"] + used_additive["station_delay_term_s"]
    )
    used_additive["additive_residual_s"] = (
        used_additive["residual_s"] - used_additive["additive_delay_term_s"]
    )

    additive_phase_summary = (
        used_additive.groupby("phase")
        .agg(
            n=("additive_residual_s", "count"),
            empirical_sigma_after_additive_s=("additive_residual_s", "std"),
            rms_after_additive_s=("additive_residual_s", _sqrt_mean_square),
            current_sigma_s=("sigma_s", "first"),
        )
        .reset_index()
    )
    additive_phase_summary["sigma_scale_factor_after_additive"] = (
        additive_phase_summary["empirical_sigma_after_additive_s"]
        / additive_phase_summary["current_sigma_s"]
    )
    phase_summary = phase_summary.merge(
        additive_phase_summary[["phase", "empirical_sigma_after_additive_s"]],
        on="phase",
        how="left",
    )
    phase_summary["additive_phase_term_s"] = phase_summary["phase"].map(
        additive_phase_terms
    )

    station_summary = (
        used_df.groupby("station")
        .agg(
            n=("residual_s", "count"),
            mean_residual_s=("residual_s", "mean"),
            std_residual_s=("residual_s", "std"),
            p_count=("phase", lambda s: int((s == "P").sum())),
            s_count=("phase", lambda s: int((s == "S").sum())),
        )
        .reset_index()
        .sort_values(["n", "mean_residual_s"], ascending=[False, True])
    )
    station_summary["station_delay_term_s"] = station_summary["station"].map(
        station_delay_terms
    )
    if not station_shrinkage_info.empty:
        station_summary = station_summary.merge(
            station_shrinkage_info[
                ["station", "station_shrinkage_factor", "shrunken_station_delay_s"]
            ],
            on="station",
            how="left",
        )
    else:
        station_summary["station_shrinkage_factor"] = float("nan")
        station_summary["shrunken_station_delay_s"] = station_summary[
            "station_delay_term_s"
        ]
    additive_station_residual_mean = (
        used_additive.groupby("station")["additive_residual_s"].mean().to_dict()
    )
    station_summary["mean_additive_residual_s"] = station_summary["station"].map(
        additive_station_residual_mean
    )

    phase_calibrated["z_phase_calibrated"] = (
        phase_calibrated["phase_corrected_residual_s"]
    ) / phase_calibrated["empirical_sigma_s"]

    phase_sigma_after_additive = phase_summary.set_index("phase")[
        "empirical_sigma_after_additive_s"
    ].to_dict()
    station_phase_summary, prior_variances = _estimate_shrinkage_interaction_terms(
        used_additive,
        phase_sigma_after_additive=phase_sigma_after_additive,
    )

    used_shrinkage = used_additive.merge(
        station_phase_summary[["station", "phase", "shrunken_interaction_term_s"]],
        on=["station", "phase"],
        how="left",
    )
    used_shrinkage["shrunken_interaction_term_s"] = used_shrinkage[
        "shrunken_interaction_term_s"
    ].fillna(0.0)
    used_shrinkage["shrinkage_delay_term_s"] = (
        used_shrinkage["additive_delay_term_s"]
        + used_shrinkage["shrunken_interaction_term_s"]
    )
    used_shrinkage["shrinkage_residual_s"] = (
        used_shrinkage["residual_s"] - used_shrinkage["shrinkage_delay_term_s"]
    )

    shrinkage_phase_summary = (
        used_shrinkage.groupby("phase")
        .agg(
            empirical_sigma_after_shrinkage_s=("shrinkage_residual_s", "std"),
            rms_after_shrinkage_s=("shrinkage_residual_s", _sqrt_mean_square),
        )
        .reset_index()
    )
    phase_summary = phase_summary.merge(
        shrinkage_phase_summary[["phase", "empirical_sigma_after_shrinkage_s"]],
        on="phase",
        how="left",
    )
    phase_summary["interaction_prior_variance_s2"] = phase_summary["phase"].map(
        prior_variances
    )
    phase_summary["interaction_prior_std_s"] = np.sqrt(
        phase_summary["interaction_prior_variance_s2"]
    )

    station_phase_summary["full_shrinkage_delay_term_s"] = (
        station_phase_summary["phase"].map(additive_phase_terms)
        + station_phase_summary["station"].map(shrunken_station_delay_terms)
        + station_phase_summary["shrunken_interaction_term_s"]
    )

    phase_sigma_after_shrinkage = phase_summary.set_index("phase")[
        "empirical_sigma_after_shrinkage_s"
    ].to_dict()
    used_shrinkage["z_current"] = (
        used_shrinkage["residual_s"] / used_shrinkage["sigma_s"]
    )
    used_shrinkage["z_additive_calibrated"] = used_shrinkage[
        "additive_residual_s"
    ] / used_shrinkage["phase"].map(phase_sigma_after_additive)
    used_shrinkage["z_shrinkage_calibrated"] = used_shrinkage[
        "shrinkage_residual_s"
    ] / used_shrinkage["phase"].map(phase_sigma_after_shrinkage)

    current_mode_correlation_coefficient, current_mode_correlation_clipped = (
        _clip_mode_correlation_coefficient(current_mode_correlation_coefficient)
    )
    current_pair_count = int(
        len(
            used_df.pivot_table(
                index=["event_id", "station"],
                columns="phase",
                values="residual_s",
                aggfunc="first",
            ).dropna()
        )
    )
    mode_correlation_summary = pd.DataFrame(
        [
            {
                "strategy": "current",
                "pair_count": current_pair_count,
                "pair_count_threshold": min_mode_correlation_pair_count,
                "mode_correlation_coefficient": current_mode_correlation_coefficient,
                "raw_mode_correlation_coefficient": current_mode_correlation_coefficient,
                "supported_for_inference": True,
                "clipped_to_valid_range": current_mode_correlation_clipped,
                "notes": "Current configured global P/S mode-correlation coefficient.",
            },
            _summarize_mode_correlation(
                used_df,
                residual_column="residual_s",
                strategy="empirical",
                notes=(
                    "Estimate one global P/S mode-correlation coefficient from paired "
                    "used residuals at event/station combinations where both phases are present."
                ),
                min_mode_correlation_pair_count=min_mode_correlation_pair_count,
            ),
            _summarize_mode_correlation(
                used_additive,
                residual_column="additive_residual_s",
                strategy="after_additive_correction",
                notes=(
                    "Estimate one global P/S mode-correlation coefficient after "
                    "applying additive phase-plus-station delay correction."
                ),
                min_mode_correlation_pair_count=min_mode_correlation_pair_count,
            ),
            _summarize_mode_correlation(
                used_shrinkage,
                residual_column="shrinkage_residual_s",
                strategy="after_shrinkage_correction",
                notes=(
                    "Estimate one global P/S mode-correlation coefficient after "
                    "applying additive delay correction plus shrunk station-phase "
                    "interaction terms."
                ),
                min_mode_correlation_pair_count=min_mode_correlation_pair_count,
            ),
        ]
    )

    coverage_rows: list[dict[str, Any]] = []
    for phase in sorted(used_df["phase"].dropna().unique()):
        phase_mask = used_shrinkage["phase"] == phase
        current_diag = _coverage_diagnostics(
            used_shrinkage.loc[phase_mask, "z_current"]
        )
        phase_diag = _coverage_diagnostics(
            phase_calibrated.loc[
                phase_calibrated["phase"] == phase, "z_phase_calibrated"
            ]
        )
        additive_diag = _coverage_diagnostics(
            used_shrinkage.loc[phase_mask, "z_additive_calibrated"]
        )
        shrinkage_diag = _coverage_diagnostics(
            used_shrinkage.loc[phase_mask, "z_shrinkage_calibrated"]
        )
        for model_name, diag in [
            ("current", current_diag),
            ("phase_calibrated", phase_diag),
            ("additive_calibrated", additive_diag),
            ("shrinkage_calibrated", shrinkage_diag),
        ]:
            row = {"phase": phase, "model": model_name}
            row.update(diag.to_dict())
            coverage_rows.append(row)
    coverage_df = pd.DataFrame(coverage_rows)

    option_rows = [
        {
            "option": "phase_delay_plus_phase_sigma",
            "phase": row["phase"],
            "delay_strategy": "phase",
            "sigma_strategy": "phase_empirical",
            "recommended_sigma_s": row["empirical_sigma_s"],
            "notes": "Apply phase-mean delay terms and empirical sigma per phase.",
        }
        for row in phase_summary.to_dict("records")
    ]
    option_rows.extend(
        {
            "option": "additive_delay_plus_phase_sigma",
            "phase": row["phase"],
            "delay_strategy": "additive",
            "sigma_strategy": "phase_after_additive_correction",
            "recommended_sigma_s": row["empirical_sigma_after_additive_s"],
            "notes": (
                "Apply additive phase-plus-station delay terms, then calibrate "
                "phase sigma on additive residuals."
            ),
        }
        for row in phase_summary.to_dict("records")
    )
    option_rows.extend(
        {
            "option": "shrinkage_delay_plus_phase_sigma",
            "phase": row["phase"],
            "delay_strategy": "shrinkage",
            "sigma_strategy": "phase_after_shrinkage_correction",
            "recommended_sigma_s": row["empirical_sigma_after_shrinkage_s"],
            "notes": (
                "Apply additive delay terms plus shrunk station-phase interaction "
                "terms, then calibrate phase sigma on shrinkage residuals."
            ),
        }
        for row in phase_summary.to_dict("records")
    )
    option_summary = pd.DataFrame(option_rows)

    return {
        "observed_residuals": observed_df,
        "used_residuals": used_df,
        "phase_summary": phase_summary,
        "station_phase_summary": station_phase_summary,
        "station_summary": station_summary,
        "mode_correlation_summary": mode_correlation_summary,
        "coverage_diagnostics": coverage_df,
        "option_summary": option_summary,
        "global_option_summary": global_option_summary,
    }


def scan_available_residual_reports(source_location_dir: Path) -> pd.DataFrame:
    """Return a simple inventory of all residual reports currently available."""
    rows: list[dict[str, Any]] = []
    for residual_csv in sorted(source_location_dir.glob("*/**/*_residual_report.csv")):
        event_id = residual_csv.parent.name
        try:
            frame = pd.read_csv(residual_csv)
        except Exception:
            continue
        rows.append(
            {
                "event_id": event_id,
                "residual_csv": str(residual_csv),
                "row_count": int(len(frame)),
                "used_row_count": int(
                    _ensure_bool_column(frame, "used_in_inference").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _format_float(value: Any, digits: int = 3) -> str:
    numeric = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric):.{digits}f}"


def _table_from_frame(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    rows: list[list[str]] = []
    for row in frame[columns].to_dict("records"):
        rendered: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (int, np.integer)):
                rendered.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                rendered.append(_format_float(value, digits=digits))
            else:
                rendered.append(str(value))
        rows.append(rendered)
    return _markdown_table(columns, rows)


def build_static_report_qmd(
    *,
    event_set_name: str,
    catalogue_scope_df: pd.DataFrame,
    analysis: dict[str, pd.DataFrame],
    wider_inventory_df: pd.DataFrame,
) -> str:
    """Build a static QMD report summarizing calibration options and results."""
    phase_summary = analysis["phase_summary"].copy()
    station_phase_summary = analysis["station_phase_summary"].copy()
    station_summary = analysis["station_summary"].copy()
    coverage_diagnostics = analysis["coverage_diagnostics"].copy()
    option_summary = analysis["option_summary"].copy()
    global_option_summary = analysis["global_option_summary"].copy()
    mode_correlation_summary = analysis["mode_correlation_summary"].copy()

    solved_events = int(
        catalogue_scope_df.get("solved_event_count", pd.Series([0])).iloc[0]
    )
    observed_residual_count = int(
        catalogue_scope_df.get("observed_residual_count", pd.Series([0])).iloc[0]
    )
    used_residual_count = int(
        catalogue_scope_df.get("used_residual_count", pd.Series([0])).iloc[0]
    )
    wider_event_count = int(len(wider_inventory_df))
    wider_used_count = int(
        wider_inventory_df.get("used_row_count", pd.Series(dtype=int)).sum()
    )

    top_station_terms = (
        station_summary.copy()
        .sort_values("station_delay_term_s", key=lambda s: s.abs(), ascending=False)
        .head(12)
    )
    top_interactions = (
        station_phase_summary.copy()
        .sort_values(
            "shrunken_interaction_term_s", key=lambda s: s.abs(), ascending=False
        )
        .head(12)
    )

    coverage_pivot = coverage_diagnostics.copy()
    coverage_pivot["abs_gt_3_pct"] = coverage_pivot["p_abs_gt_3"] * 100.0

    single_scale = global_option_summary.loc[
        global_option_summary["option"] == "single_scale_on_current_sigma"
    ]
    single_scale_factor = (
        float(single_scale["global_sigma_scale_factor"].iloc[0])
        if not single_scale.empty
        else float("nan")
    )

    return f"""---
title: "{event_set_name} Source-Location Residual Calibration"
subtitle: "Additive and shrinkage delay calibration"
format:
  html:
    toc: true
    number-sections: true
  pdf:
    toc: true
    number-sections: true
execute:
  enabled: false
---

# Scope

This report summarizes travel-time residual calibration options for the
{event_set_name} source-location catalogue.
The implemented hierarchy now distinguishes:

$$
r_{{e,s,m}} = a_m + \varepsilon_{{e,s,m}}
$$

for phase-only correction,

$$
r_{{e,s,m}} = a_m + b_s + \varepsilon_{{e,s,m}}
$$

for additive phase-plus-station correction, and

$$
r_{{e,s,m}} = a_m + b_s + c_{{s,m}} + \varepsilon_{{e,s,m}}
$$

for shrinkage calibration, where $c_{{s,m}}$ is estimated from additive residuals
and shrunk toward zero with an empirical-Bayes normal prior.

- Solved events in calibration scope: {solved_events}
- Observed residual rows available: {observed_residual_count}
- Residual rows actually used in inference: {used_residual_count}
- Wider residual-report inventory currently present under
    `results/source_location`: {wider_event_count} events,
    {wider_used_count} used residual rows

# Models Considered

The following calibration options were evaluated:

{_table_from_frame(global_option_summary.fillna("n/a"), [
    "option",
    "delay_strategy",
    "sigma_strategy",
    "combined_std_normalized_residual",
    "combined_rms_normalized_residual",
], digits=3)}

Phase-specific and hierarchical operational options:

{_table_from_frame(option_summary.fillna("n/a"), [
    "option",
    "phase",
    "delay_strategy",
    "sigma_strategy",
    "recommended_sigma_s",
], digits=3)}

Interpretation of the main options:

1. `phase_delay_plus_phase_sigma`: use phase-mean delays and empirical phase sigma.
2. `additive_delay_plus_phase_sigma`: use explicit phase-plus-station offsets.
3. `shrinkage_delay_plus_phase_sigma`: add shrunk station-phase interactions on top
   of the additive model.

Global P/S mode-correlation options:

{_table_from_frame(mode_correlation_summary.fillna("n/a"), [
    "strategy",
    "pair_count",
    "mode_correlation_coefficient",
    "supported_for_inference",
], digits=3)}

The current workflow uses one global P/S correlation coefficient via

$$
R_{{\\text{{mode}}}} =
\\begin{{bmatrix}}
1 & \\rho_{{PS}} \\
\\rho_{{PS}} & 1
\\end{{bmatrix}}
$$

where $\rho_{{PS}}$ is inferred from event/station pairs that contain both phases.

# Phase-Level Results

Phase summary from currently used residuals:

{_table_from_frame(phase_summary, [
    "phase",
    "n",
    "mean_residual_s",
    "phase_delay_term_s",
    "additive_phase_term_s",
    "current_sigma_s",
    "empirical_sigma_s",
    "empirical_sigma_after_additive_s",
    "empirical_sigma_after_shrinkage_s",
    "interaction_prior_std_s",
], digits=3)}

Key observations:

- `mean_residual_s` is the raw phase mean before imposing the origin-time gauge.
- `phase_delay_term_s` is the fitted phase-only delay term exported for runtime
    use.
- `additive_phase_term_s` is the fitted phase component in the additive model;
    together with `station_delay_term_s` it reconstructs the additive mean
    correction.
- `interaction_prior_std_s` summarizes how much station-phase interaction
  variance remains after additive correction.
- A single common scale factor of about {single_scale_factor:.3f}x is still
  too coarse to describe the phase asymmetry visible in the archive.

# Station Terms

Largest additive station terms:

{_table_from_frame(top_station_terms, [
    "station",
    "n",
    "station_delay_term_s",
    "mean_residual_s",
    "mean_additive_residual_s",
], digits=3)}

# Station-Phase Interactions

Largest shrunk station-phase interaction terms:

{_table_from_frame(top_interactions, [
    "station",
    "phase",
    "n",
    "raw_interaction_term_s",
    "shrinkage_factor",
    "shrunken_interaction_term_s",
], digits=3)}

Interpretation:

- The additive station terms absorb shared P/S timing bias per station.
- The shrinkage factors show how much extra station-phase structure is supported
  beyond the additive model.
- When the prior variance is small relative to the observation variance,
  interaction terms collapse toward zero instead of relying on hard cutoffs.

# Normality Diagnostics

Coverage diagnostics for normalized residuals:

{_table_from_frame(coverage_pivot, [
    "phase",
    "model",
    "n",
    "mean",
    "std",
    "p_abs_le_1",
    "p_abs_le_2",
    "abs_gt_3_pct",
], digits=3)}

For a perfect standard normal model, one would expect approximately
68.3% inside $|z| \\le 1$, 95.4% inside $|z| \\le 2$, and 0.27% beyond
$|z| > 3$.

Findings:

- The current model has over-dispersed `P` residuals and biased `S`
    residuals.
- Phase-only calibration removes the catalogue-wide phase bias but leaves
    common station structure in the residuals.
- The additive model usually gives the biggest gain in interpretability per
    parameter.
- Shrinkage can further reduce tails while keeping weak station-phase cells
    close to zero.

# Recommended Operational Path

Recommended short-term option for source-location reruns:

1. Use the additive model as the first calibrated rerun.
2. Use the shrinkage model as the second rerun, initialized from the additive archive.
3. Use phase sigma calibrated after the selected delay correction.
4. Use the mode-correlation coefficient inferred after the same correction stage.

This keeps the model compatible with the current source-location
covariance implementation while using a continuous partial-pooling model for
station-phase interactions.

# Caveats

- The calibration scope is still modest: only {solved_events} solved
    events were available in the current {event_set_name} catalogue.
- The shrinkage prior is empirical-Bayes, not a full joint Bayesian posterior over
    calibration unknowns.
- Station/phase interactions can still be unstable if the archive remains very sparse,
    but weak cells now shrink toward zero instead of toggling on or off.
- Residual tails are heavier than Gaussian, so a pure normal model is
    likely optimistic for uncertainty quantification.
- The current implementation can update phase sigma directly, but not a
    fully station-specific covariance model.

# Prospects For A Wider Catalogue

The wider residual inventory already contains {wider_event_count} events
with residual exports. That is enough to justify a broader follow-up
calibration pass.

With a wider catalogue, the next improvements become realistic:

1. Better prior estimation for the station-phase interaction layer.
2. Separate calibrations by station class, borehole level, or catalogue epoch.
3. Robust likelihoods such as Student-$t$ residuals for heavier tails.
4. Better separation of travel-time modelling error from pick-timing error.

# Files Produced Alongside This Report

- `catalogue_scope_summary.csv`
- `phase_calibration_summary.csv`
- `station_phase_calibration_summary.csv`
- `station_calibration_summary.csv`
- `mode_correlation_summary.csv`
- `coverage_diagnostics.csv`
- `calibration_option_summary.csv`

These tables are intended for downstream review, reproducibility, and
optional reuse by the source-location workflow.
"""


def write_calibration_outputs(
    *,
    output_dir: Path,
    event_set_name: str,
    analysis: dict[str, pd.DataFrame],
    wider_inventory_df: pd.DataFrame,
    solved_event_count: int,
) -> dict[str, Path]:
    """Write calibration tables and a static QMD report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    catalogue_scope_df = pd.DataFrame(
        [
            {
                "event_set_name": event_set_name,
                "solved_event_count": solved_event_count,
                "observed_residual_count": int(len(analysis["observed_residuals"])),
                "used_residual_count": int(len(analysis["used_residuals"])),
            }
        ]
    )
    file_map = {
        "catalogue_scope_summary": output_dir / "catalogue_scope_summary.csv",
        "phase_summary": output_dir / "phase_calibration_summary.csv",
        "station_phase_summary": output_dir / "station_phase_calibration_summary.csv",
        "station_summary": output_dir / "station_calibration_summary.csv",
        "mode_correlation_summary": output_dir / "mode_correlation_summary.csv",
        "coverage": output_dir / "coverage_diagnostics.csv",
        "options": output_dir / "calibration_option_summary.csv",
        "global_options": output_dir / "global_calibration_option_summary.csv",
        "wider_inventory": output_dir / "wider_residual_inventory.csv",
        "report_qmd": default_report_qmd(output_dir, event_set_name),
    }

    catalogue_scope_df.to_csv(file_map["catalogue_scope_summary"], index=False)
    analysis["phase_summary"].to_csv(file_map["phase_summary"], index=False)
    analysis["station_phase_summary"].to_csv(
        file_map["station_phase_summary"], index=False
    )
    analysis["station_summary"].to_csv(file_map["station_summary"], index=False)
    analysis["mode_correlation_summary"].to_csv(
        file_map["mode_correlation_summary"], index=False
    )
    analysis["coverage_diagnostics"].to_csv(file_map["coverage"], index=False)
    analysis["option_summary"].to_csv(file_map["options"], index=False)
    analysis["global_option_summary"].to_csv(file_map["global_options"], index=False)
    wider_inventory_df.to_csv(file_map["wider_inventory"], index=False)

    report_text = build_static_report_qmd(
        event_set_name=event_set_name,
        catalogue_scope_df=catalogue_scope_df,
        analysis=analysis,
        wider_inventory_df=wider_inventory_df,
    )
    file_map["report_qmd"].write_text(report_text, encoding="utf-8")
    return file_map


def load_calibration_model(calibration_dir: Path) -> SourceLocationCalibrationModel:
    """Load exported calibration tables into an inference-ready model."""
    phase_csv = calibration_dir / "phase_calibration_summary.csv"
    station_csv = calibration_dir / "station_calibration_summary.csv"
    station_phase_csv = calibration_dir / "station_phase_calibration_summary.csv"
    mode_correlation_csv = calibration_dir / "mode_correlation_summary.csv"

    if not phase_csv.exists():
        raise FileNotFoundError(f"Phase calibration summary not found: {phase_csv}")
    if not station_csv.exists():
        raise FileNotFoundError(f"Station calibration summary not found: {station_csv}")
    if not station_phase_csv.exists():
        raise FileNotFoundError(
            f"Station-phase calibration summary not found: {station_phase_csv}"
        )

    phase_df = pd.read_csv(phase_csv)
    station_df = pd.read_csv(station_csv)
    station_phase_df = pd.read_csv(station_phase_csv)
    mode_correlation_df = (
        pd.read_csv(mode_correlation_csv)
        if mode_correlation_csv.exists()
        else pd.DataFrame(columns=["strategy", "mode_correlation_coefficient"])
    )

    phase_delay_column = (
        "phase_delay_term_s"
        if "phase_delay_term_s" in phase_df.columns
        else "mean_residual_s"
    )
    phase_delay_terms = (
        phase_df.set_index("phase")[phase_delay_column].dropna().astype(float).to_dict()
    )
    additive_phase_delay_terms = {
        str(row["phase"]): float(row["additive_phase_term_s"])
        for row in phase_df.to_dict("records")
        if pd.notna(row.get("additive_phase_term_s"))
    }
    station_delay_terms = {}
    for row in station_df.to_dict("records"):
        station = str(row["station"])
        # Prefer the shrunken term written by the current calibration run;
        # fall back to the raw OLS term for files written by older versions.
        delay: float | None = None
        shrunken = row.get("shrunken_station_delay_s")
        if shrunken is not None and pd.notna(shrunken):
            delay = float(shrunken)
        else:
            raw = row.get("station_delay_term_s")
            if raw is not None and pd.notna(raw):
                delay = float(raw)
        if delay is not None:
            station_delay_terms[station] = delay
    station_phase_interaction_terms = {
        (str(row["station"]), str(row["phase"])): float(
            row["shrunken_interaction_term_s"]
        )
        for row in station_phase_df.to_dict("records")
        if pd.notna(row.get("shrunken_interaction_term_s"))
    }
    phase_sigma_models: dict[str, dict[str, float]] = {}
    for row in phase_df.to_dict("records"):
        phase_models = {"current": float(row["current_sigma_s"])}
        for strategy, column in [
            ("phase_empirical", "empirical_sigma_s"),
            ("phase_after_additive_correction", "empirical_sigma_after_additive_s"),
            (
                "phase_after_shrinkage_correction",
                "empirical_sigma_after_shrinkage_s",
            ),
        ]:
            if pd.notna(row.get(column)):
                phase_models[strategy] = float(row[column])
        phase_sigma_models[str(row["phase"])] = phase_models
    mode_correlation_models = {
        str(row["strategy"]): float(row["mode_correlation_coefficient"])
        for row in mode_correlation_df.to_dict("records")
        if pd.notna(row.get("mode_correlation_coefficient"))
        and (
            str(row.get("strategy")) == "current"
            or bool(row.get("supported_for_inference", False))
        )
    }
    return SourceLocationCalibrationModel(
        phase_delay_terms=phase_delay_terms,
        additive_phase_delay_terms=additive_phase_delay_terms,
        station_delay_terms=station_delay_terms,
        station_phase_interaction_terms=station_phase_interaction_terms,
        phase_sigma_models=phase_sigma_models,
        mode_correlation_models=mode_correlation_models,
    )


def calibrate_phase_sigma(
    sigma: xr.DataArray,
    *,
    calibration_model: SourceLocationCalibrationModel,
    sigma_strategy: str,
) -> xr.DataArray:
    """Return a phase-level sigma vector calibrated according to the selected strategy."""
    if sigma_strategy == "current":
        return sigma

    calibrated = sigma.copy(deep=True)
    for phase in calibrated.mode.values:
        phase_id = str(phase)
        if phase_id not in calibration_model.phase_sigma_models:
            continue
        phase_models = calibration_model.phase_sigma_models[phase_id]
        if sigma_strategy not in phase_models:
            raise ValueError(f"Unsupported sigma strategy: {sigma_strategy}")
        calibrated.loc[dict(mode=phase_id)] = phase_models[sigma_strategy]
    return calibrated


def calibrate_mode_correlation_coefficient(
    mode_correlation_coefficient: float,
    *,
    calibration_model: SourceLocationCalibrationModel,
    mode_correlation_strategy: str,
) -> float:
    """Return a calibrated global P/S mode-correlation coefficient."""
    if mode_correlation_strategy == "current":
        return float(mode_correlation_coefficient)

    if mode_correlation_strategy not in calibration_model.mode_correlation_models:
        raise ValueError(
            f"Unsupported mode-correlation strategy: {mode_correlation_strategy}"
        )

    return float(calibration_model.mode_correlation_models[mode_correlation_strategy])


def apply_delay_terms(
    obs_at: xr.DataArray,
    *,
    calibration_model: SourceLocationCalibrationModel,
    delay_strategy: str,
) -> xr.DataArray:
    """Apply configured delay terms to observed arrival times before inference."""
    if delay_strategy == "none":
        return obs_at

    adjusted = obs_at.copy(deep=True)
    for station in adjusted.station.values:
        station_id = str(station)
        for mode in adjusted.mode.values:
            phase = str(mode)
            value = pd.to_numeric(
                [adjusted.sel(station=station_id, mode=phase).item()], errors="coerce"
            )[0]
            if pd.isna(value):
                continue

            delay = 0.0
            if delay_strategy == "phase":
                delay = calibration_model.phase_delay_terms.get(phase, 0.0)
            elif delay_strategy == "additive":
                delay = calibration_model.additive_phase_delay_terms.get(
                    phase,
                    calibration_model.phase_delay_terms.get(phase, 0.0),
                )
                delay += calibration_model.station_delay_terms.get(station_id, 0.0)
            elif delay_strategy == "shrinkage":
                delay = calibration_model.additive_phase_delay_terms.get(
                    phase,
                    calibration_model.phase_delay_terms.get(phase, 0.0),
                )
                delay += calibration_model.station_delay_terms.get(station_id, 0.0)
                delay += calibration_model.station_phase_interaction_terms.get(
                    (station_id, phase),
                    0.0,
                )
            else:
                raise ValueError(f"Unsupported delay strategy: {delay_strategy}")

            adjusted.loc[dict(station=station_id, mode=phase)] = float(value) - float(
                delay
            )

    return adjusted
