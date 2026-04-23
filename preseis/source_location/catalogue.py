from __future__ import annotations

from pathlib import Path

import pandas as pd


def _pivot_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["event_id"])

    value_columns = [
        column
        for column in summary_df.columns
        if column not in {"event_id", "location_type"}
    ]
    pivot_df = summary_df.pivot(
        index="event_id",
        columns="location_type",
        values=value_columns,
    )
    pivot_df.columns = [
        f"{location_type}_{column}" for column, location_type in pivot_df.columns
    ]
    return pivot_df.reset_index()


def _order_catalogue_columns(output_df: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "event_id",
        "catalog_event_id",
        "catalog_origin_time_reference",
        "catalog_latitude",
        "catalog_longitude",
        "catalog_depth_km",
        "catalog_magnitude",
        "catalog_magnitude_type",
        "catalog_author",
        "source_location_solved",
        "source_location_status",
        "source_location_failure_stage",
        "source_location_failure_reason",
        "source_location_valid_observations",
        "source_location_contributing_stations",
        "catalog_origin_time",
        "catalog_x",
        "catalog_y",
        "catalog_z",
        "catalog_time_shift_uncertainty",
        "catalog_x_uncertainty",
        "catalog_y_uncertainty",
        "catalog_z_uncertainty",
        "catalog_n_stations",
        "catalog_n_phases",
        "catalog_standard_error",
        "catalog_azimuthal_gap",
        "map_origin_time",
        "map_origin_time_shift",
        "map_time_shift_uncertainty",
        "map_x",
        "map_y",
        "map_z",
        "map_x_uncertainty",
        "map_y_uncertainty",
        "map_z_uncertainty",
        "map_semi_major",
        "map_semi_minor",
        "map_azimuth",
        "map_n_stations",
        "map_n_phases",
        "map_correlated_whitened_rms",
        "map_reduced_squared_misfit",
        "map_squared_misfit",
        "map_standard_error",
        "posterior_mean_origin_time",
        "posterior_mean_origin_time_shift",
        "posterior_mean_time_shift_uncertainty",
        "posterior_mean_x",
        "posterior_mean_y",
        "posterior_mean_z",
        "posterior_mean_x_uncertainty",
        "posterior_mean_y_uncertainty",
        "posterior_mean_z_uncertainty",
        "posterior_mean_semi_major",
        "posterior_mean_semi_minor",
        "posterior_mean_azimuth",
        "posterior_mean_n_stations",
        "posterior_mean_n_phases",
        "posterior_mean_correlated_whitened_rms",
        "posterior_mean_reduced_squared_misfit",
        "posterior_mean_squared_misfit",
        "posterior_mean_standard_error",
    ]
    ordered_columns = [
        column for column in preferred_columns if column in output_df.columns
    ]
    remaining_columns = [
        column for column in output_df.columns if column not in ordered_columns
    ]
    return output_df[ordered_columns + remaining_columns]


def _solved_mask(output_df: pd.DataFrame) -> pd.Series:
    if "map_x" in output_df.columns:
        return output_df["map_x"].notna()
    if "posterior_mean_x" in output_df.columns:
        return output_df["posterior_mean_x"].notna()
    if "catalog_x" in output_df.columns:
        return output_df["catalog_x"].notna()
    return pd.Series(False, index=output_df.index, dtype=bool)


def build_uncertainty_catalogue(
    summary_csv: Path,
    output_csv: Path,
    *,
    event_set_csv: Path | None = None,
    status_csv: Path | None = None,
) -> Path:
    """Build a flat event catalogue with prefixed columns for each location type."""
    summary_df = pd.read_csv(summary_csv)
    output_df = _pivot_summary(summary_df)

    if event_set_csv is not None:
        event_set_df = pd.read_csv(event_set_csv).copy()
        event_set_df["event_id"] = (
            event_set_df["event_id"].astype(str).str.split("/").str[-1]
        )
        event_set_df = event_set_df.rename(
            columns={
                "event_id": "catalog_event_id",
                "origin_time": "catalog_origin_time_reference",
                "latitude": "catalog_latitude",
                "longitude": "catalog_longitude",
                "depth_km": "catalog_depth_km",
                "magnitude": "catalog_magnitude",
                "magnitude_type": "catalog_magnitude_type",
                "author": "catalog_author",
            }
        )
        event_set_df["event_id"] = event_set_df["catalog_event_id"]
        output_df = event_set_df.merge(output_df, on="event_id", how="left")

    if status_csv is not None and status_csv.exists():
        status_df = pd.read_csv(status_csv)
        status_columns = [
            "event_id",
            *[
                column
                for column in status_df.columns
                if column.startswith("source_location_")
            ],
        ]
        status_df = status_df[status_columns].drop_duplicates(subset=["event_id"])
        output_df = output_df.merge(status_df, on="event_id", how="left")

    output_df["source_location_solved"] = _solved_mask(output_df)
    output_df = _order_catalogue_columns(output_df)
    output_df.to_csv(output_csv, index=False)
    return output_csv
