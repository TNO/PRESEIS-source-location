from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import contextily as ctx
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj as prj
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from obspy import Catalog, UTCDateTime, read_events
from obspy.core.event import (
    Comment,
    CreationInfo,
    Event,
    Magnitude,
    Origin,
    ResourceIdentifier,
)
from shapely.affinity import rotate as _shapely_rotate, scale as _shapely_scale
from shapely.geometry import Point, box

from preseis.processing.io import (
    sanitize_invalid_resource_identifiers,
    write_quakeml_catalog,
)
from .methods import (
    available_source_location_methods,
    default_method_output_dir,
)


@dataclass(frozen=True)
class SourceLocationMethodSpec:
    key: str
    label: str
    output_dir: Path
    learning_inventory_scope: str
    learning_inventory_dir: Path | None = None
    step: int = 0

    def catalogue_csv(self, event_set_name: str) -> Path:
        return (
            self.output_dir
            / f"{_event_set_slug(event_set_name)}_source_location_catalogue.csv"
        )

    @property
    def summary_csv(self) -> Path:
        return self.output_dir / "source_locations.csv"

    @property
    def status_csv(self) -> Path:
        return self.output_dir / "source_location_status.csv"

    def event_quakeml(
        self, event_id: str, *, suffix: str = "_with_source_locations"
    ) -> Path:
        return self.output_dir / event_id / f"event_{event_id}{suffix}.xml"


METHOD_COLORS = {
    "step0_baseline": "#1f1f1f",
    "step1_phase_only": "#1f77b4",
    "step2_additive": "#ff7f0e",
    "step3_shrinkage": "#2ca02c",
}

ESTIMATE_LABELS = {
    "map": "MAP",
    "posterior_mean": "Posterior mean",
}

_QUAKEML_TOKEN_RE = re.compile(r"[^A-Za-z0-9._\-~'()*]+")
_ELLIPSE_95_SCALE = float(np.sqrt(5.991464547107979))
_LABEL_OFFSETS = [
    (int(round(radius * dx)), int(round(radius * dy)))
    for radius in (16, 26, 36, 46, 56, 68, 82)
    for dx, dy in (
        (1.0, 0.0),
        (0.9239, 0.3827),
        (0.7071, 0.7071),
        (0.3827, 0.9239),
        (0.0, 1.0),
        (-0.3827, 0.9239),
        (-0.7071, 0.7071),
        (-0.9239, 0.3827),
        (-1.0, 0.0),
        (-0.9239, -0.3827),
        (-0.7071, -0.7071),
        (-0.3827, -0.9239),
        (0.0, -1.0),
        (0.3827, -0.9239),
        (0.7071, -0.7071),
        (0.9239, -0.3827),
    )
]
_LABEL_BBOX_PADDING_PX = 3.0
_EVENT_LABEL_FONT_SIZE = 6.8
_FAULTS_DEFAULT_CRS = "EPSG:28992"


def _event_set_slug(event_set_name: str) -> str:
    return event_set_name.lower().replace(" ", "")


def _quakeml_uri_token(value: Any) -> str:
    text = "unknown" if value is None else str(value)
    sanitized = _QUAKEML_TOKEN_RE.sub("-", text)
    return sanitized.strip("-") or "unknown"


_sanitize_invalid_resource_identifiers = sanitize_invalid_resource_identifiers


def default_method_specs(base_dir: Path) -> list[SourceLocationMethodSpec]:
    base_results = base_dir / "results"
    specs: list[SourceLocationMethodSpec] = []
    for method in available_source_location_methods():
        learning_inventory_dir = None
        if method.input_calibration_dirname is not None:
            learning_inventory_dir = base_results / method.input_calibration_dirname
        specs.append(
            SourceLocationMethodSpec(
                key=method.key,
                label=method.label,
                output_dir=default_method_output_dir(base_results, method),
                learning_inventory_scope=method.learning_inventory_scope,
                learning_inventory_dir=learning_inventory_dir,
                step=method.step,
            )
        )
    return specs


def available_method_specs(
    *,
    base_dir: Path,
    event_set_name: str,
    include_keys: set[str] | None = None,
) -> list[SourceLocationMethodSpec]:
    specs: list[SourceLocationMethodSpec] = []
    for spec in default_method_specs(base_dir):
        if include_keys is not None and spec.key not in include_keys:
            continue
        if spec.catalogue_csv(event_set_name).exists():
            specs.append(spec)
    return specs


def load_event_set_catalogue(event_set_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(event_set_csv).copy()
    frame["catalog_event_id"] = frame["event_id"].astype(str)
    frame["event_id"] = frame["catalog_event_id"].str.split("/").str[-1]
    frame = frame.rename(
        columns={
            "origin_time": "catalog_origin_time_reference",
            "latitude": "catalog_latitude",
            "longitude": "catalog_longitude",
            "depth_km": "catalog_depth_km",
            "magnitude": "catalog_magnitude",
            "magnitude_type": "catalog_magnitude_type",
            "author": "catalog_author",
        }
    )
    preferred = [
        "event_id",
        "catalog_event_id",
        "catalog_origin_time_reference",
        "catalog_latitude",
        "catalog_longitude",
        "catalog_depth_km",
        "catalog_magnitude",
        "catalog_magnitude_type",
        "catalog_author",
    ]
    return frame[preferred]


def _load_learning_inventory_counts(calibration_dir: Path | None) -> dict[str, Any]:
    if calibration_dir is None:
        return {
            "learning_inventory_event_count": pd.NA,
            "learning_inventory_used_residual_count": pd.NA,
            "report_catalogue_solved_event_count": pd.NA,
            "report_catalogue_used_residual_count": pd.NA,
        }

    wider_inventory_csv = calibration_dir / "wider_residual_inventory.csv"
    scope_csv = calibration_dir / "catalogue_scope_summary.csv"

    wider_event_count: int | Any = pd.NA
    wider_used_residual_count: int | Any = pd.NA
    if wider_inventory_csv.exists():
        wider_inventory = pd.read_csv(wider_inventory_csv)
        wider_event_count = int(len(wider_inventory))
        if "used_row_count" in wider_inventory.columns:
            wider_used_residual_count = int(
                pd.to_numeric(wider_inventory["used_row_count"], errors="coerce")
                .fillna(0)
                .sum()
            )

    report_solved_event_count: int | Any = pd.NA
    report_used_residual_count: int | Any = pd.NA
    if scope_csv.exists():
        scope_df = pd.read_csv(scope_csv)
        if not scope_df.empty:
            report_solved_event_count = int(scope_df["solved_event_count"].iloc[0])
            report_used_residual_count = int(scope_df["used_residual_count"].iloc[0])

    return {
        "learning_inventory_event_count": wider_event_count,
        "learning_inventory_used_residual_count": wider_used_residual_count,
        "report_catalogue_solved_event_count": report_solved_event_count,
        "report_catalogue_used_residual_count": report_used_residual_count,
    }


def build_method_metadata_table(
    method_specs: list[SourceLocationMethodSpec],
    *,
    event_set_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in method_specs:
        catalogue_csv = spec.catalogue_csv(event_set_name)
        catalogue_df = pd.read_csv(catalogue_csv)
        rows.append(
            {
                "method_step": spec.step,
                "method_key": spec.key,
                "method_label": spec.label,
                "output_dir": str(spec.output_dir),
                "catalogue_csv": str(catalogue_csv),
                "summary_csv": str(spec.summary_csv),
                "learning_inventory_scope": spec.learning_inventory_scope,
                **_load_learning_inventory_counts(spec.learning_inventory_dir),
                "official_catalogue_event_count": int(len(catalogue_df)),
                "official_catalogue_solved_event_count": int(
                    catalogue_df["source_location_solved"]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _selected_method_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "source_location_",
        "map_",
        "posterior_mean_",
    )
    return [column for column in frame.columns if column.startswith(prefixes)]


def build_method_comparison_catalogue(
    method_specs: list[SourceLocationMethodSpec],
    *,
    event_set_csv: Path,
    event_set_name: str,
) -> pd.DataFrame:
    output_df = load_event_set_catalogue(event_set_csv)
    for spec in method_specs:
        method_df = pd.read_csv(spec.catalogue_csv(event_set_name)).copy()
        rename_map = {
            column: f"{spec.key}_{column}"
            for column in _selected_method_columns(method_df)
        }
        method_df = method_df[["event_id", *rename_map]].rename(columns=rename_map)
        output_df = output_df.merge(method_df, on="event_id", how="left")
    return output_df


def _to_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _numeric_scalar(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _add_lat_lon(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame["latitude"] = pd.Series(dtype=float)
        frame["longitude"] = pd.Series(dtype=float)
        return frame

    transformer = prj.Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    frame = frame.copy()
    x = pd.to_numeric(frame["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    lon = np.full(frame.shape[0], np.nan, dtype=float)
    lat = np.full(frame.shape[0], np.nan, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        lon_values, lat_values = transformer.transform(
            x[finite].tolist(),
            y[finite].tolist(),
        )
        lon[finite] = np.asarray(lon_values, dtype=float)
        lat[finite] = np.asarray(lat_values, dtype=float)
    frame["longitude"] = lon
    frame["latitude"] = lat
    return frame


def events_without_estimates(
    event_set_df: pd.DataFrame,
    estimate_df: pd.DataFrame,
) -> pd.DataFrame:
    if estimate_df.empty:
        return event_set_df.copy()
    solved_event_ids = set(estimate_df["event_id"].astype(str))
    mask = ~event_set_df["event_id"].astype(str).isin(solved_event_ids)
    return event_set_df.loc[mask].copy()


def build_method_estimate_table(
    method_specs: list[SourceLocationMethodSpec],
    *,
    event_set_name: str,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for spec in method_specs:
        method_df = pd.read_csv(spec.catalogue_csv(event_set_name)).copy()
        for estimate_type, estimate_label in ESTIMATE_LABELS.items():
            prefix = f"{estimate_type}_"
            x_column = f"{prefix}x"
            if x_column not in method_df.columns:
                continue
            estimate_columns = [
                column for column in method_df.columns if column.startswith(prefix)
            ]
            base_columns = [
                "event_id",
                "catalog_origin_time_reference",
                "catalog_magnitude",
                "catalog_magnitude_type",
                "catalog_latitude",
                "catalog_longitude",
                "source_location_solved",
                "source_location_status",
                "source_location_failure_stage",
                "source_location_failure_reason",
                "source_location_valid_observations",
                "source_location_contributing_stations",
            ]
            available_base_columns = [
                column for column in base_columns if column in method_df.columns
            ]
            subset = method_df[available_base_columns + estimate_columns].copy()
            rename_map = {
                column: column.removeprefix(prefix) for column in estimate_columns
            }
            subset = subset.rename(columns=rename_map)
            subset = subset.loc[
                pd.to_numeric(subset["x"], errors="coerce").notna()
            ].copy()
            if subset.empty:
                continue
            subset["method_key"] = spec.key
            subset["method_step"] = spec.step
            subset["method_label"] = spec.label
            subset["learning_inventory_scope"] = spec.learning_inventory_scope
            subset["estimate_type"] = estimate_type
            subset["estimate_label"] = estimate_label
            subset["display_label"] = f"{spec.label} - {estimate_label}"
            if "correlated_whitened_rms" in subset.columns:
                corr = pd.to_numeric(subset["correlated_whitened_rms"], errors="coerce")
            else:
                corr = pd.Series(np.nan, index=subset.index, dtype=float)
            standard_error = _to_numeric_series(subset, "standard_error")
            subset["fit_metric_name"] = np.where(
                corr.notna(),
                "correlated_whitened_rms",
                "standard_error",
            )
            subset["fit_metric_value"] = np.where(corr.notna(), corr, standard_error)
            records.append(_add_lat_lon(subset))

    if not records:
        return pd.DataFrame()

    output_df = pd.concat(records, ignore_index=True)
    output_df = output_df.sort_values(
        ["event_id", "method_key", "estimate_type"],
        ascending=[True, True, True],
    )
    return output_df.reset_index(drop=True)


def _add_catalogue_rd_coordinates(event_set_df: pd.DataFrame) -> pd.DataFrame:
    if event_set_df.empty:
        return event_set_df.copy()

    frame = event_set_df.copy()
    transformer = prj.Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    lon = pd.to_numeric(frame["catalog_longitude"], errors="coerce").to_numpy(
        dtype=float
    )
    lat = pd.to_numeric(frame["catalog_latitude"], errors="coerce").to_numpy(
        dtype=float
    )
    x = np.full(frame.shape[0], np.nan, dtype=float)
    y = np.full(frame.shape[0], np.nan, dtype=float)
    finite = np.isfinite(lon) & np.isfinite(lat)
    if finite.any():
        x_values, y_values = transformer.transform(
            lon[finite].tolist(),
            lat[finite].tolist(),
        )
        x[finite] = np.asarray(x_values, dtype=float)
        y[finite] = np.asarray(y_values, dtype=float)
    frame["catalog_x"] = x
    frame["catalog_y"] = y
    frame["catalog_depth_m"] = (
        pd.to_numeric(frame["catalog_depth_km"], errors="coerce") * 1000.0
    )
    return frame


def _ellipse_area(semi_major: pd.Series, semi_minor: pd.Series) -> pd.Series:
    return np.pi * semi_major * semi_minor


def build_method_shift_table(
    method_specs: list[SourceLocationMethodSpec],
    *,
    estimate_df: pd.DataFrame,
    event_set_df: pd.DataFrame,
) -> pd.DataFrame:
    if estimate_df.empty:
        return pd.DataFrame()

    output_df = estimate_df.copy()
    output_df["x"] = pd.to_numeric(output_df["x"], errors="coerce")
    output_df["y"] = pd.to_numeric(output_df["y"], errors="coerce")
    output_df["depth_m"] = -pd.to_numeric(output_df["z"], errors="coerce")
    output_df["semi_major"] = pd.to_numeric(output_df["semi_major"], errors="coerce")
    output_df["semi_minor"] = pd.to_numeric(output_df["semi_minor"], errors="coerce")
    output_df["standard_error"] = pd.to_numeric(
        output_df.get("standard_error"),
        errors="coerce",
    )
    output_df["ellipse_area_m2"] = _ellipse_area(
        output_df["semi_major"],
        output_df["semi_minor"],
    )

    catalogue_df = _add_catalogue_rd_coordinates(event_set_df)[
        ["event_id", "catalog_x", "catalog_y", "catalog_depth_m"]
    ]
    output_df = output_df.merge(catalogue_df, on="event_id", how="left")
    output_df["catalog_east_shift_m"] = output_df["x"] - output_df["catalog_x"]
    output_df["catalog_north_shift_m"] = output_df["y"] - output_df["catalog_y"]
    output_df["catalog_horizontal_shift_m"] = np.hypot(
        output_df["catalog_east_shift_m"],
        output_df["catalog_north_shift_m"],
    )
    output_df["catalog_depth_shift_m"] = (
        output_df["depth_m"] - output_df["catalog_depth_m"]
    )

    baseline_spec = min(method_specs, key=lambda spec: spec.step, default=None)
    if baseline_spec is None:
        return output_df

    baseline_df = output_df.loc[
        output_df["method_key"] == baseline_spec.key,
        [
            "event_id",
            "estimate_type",
            "x",
            "y",
            "depth_m",
            "semi_major",
            "semi_minor",
            "standard_error",
            "ellipse_area_m2",
            "fit_metric_value",
        ],
    ].rename(
        columns={
            "x": "baseline_x",
            "y": "baseline_y",
            "depth_m": "baseline_depth_m",
            "semi_major": "baseline_semi_major",
            "semi_minor": "baseline_semi_minor",
            "standard_error": "baseline_standard_error",
            "ellipse_area_m2": "baseline_ellipse_area_m2",
            "fit_metric_value": "baseline_fit_metric_value",
        }
    )
    output_df = output_df.merge(
        baseline_df,
        on=["event_id", "estimate_type"],
        how="left",
    )
    output_df["baseline_east_shift_m"] = output_df["x"] - output_df["baseline_x"]
    output_df["baseline_north_shift_m"] = output_df["y"] - output_df["baseline_y"]
    output_df["baseline_horizontal_shift_m"] = np.hypot(
        output_df["baseline_east_shift_m"],
        output_df["baseline_north_shift_m"],
    )
    output_df["baseline_depth_shift_m"] = (
        output_df["depth_m"] - output_df["baseline_depth_m"]
    )
    output_df["baseline_semi_major_change_m"] = (
        output_df["semi_major"] - output_df["baseline_semi_major"]
    )
    output_df["baseline_semi_minor_change_m"] = (
        output_df["semi_minor"] - output_df["baseline_semi_minor"]
    )
    output_df["baseline_standard_error_change"] = (
        output_df["standard_error"] - output_df["baseline_standard_error"]
    )
    output_df["baseline_fit_metric_change"] = pd.to_numeric(
        output_df["fit_metric_value"], errors="coerce"
    ) - pd.to_numeric(output_df["baseline_fit_metric_value"], errors="coerce")
    output_df["baseline_ellipse_area_change_m2"] = (
        output_df["ellipse_area_m2"] - output_df["baseline_ellipse_area_m2"]
    )
    output_df["baseline_ellipse_area_ratio"] = np.where(
        pd.to_numeric(output_df["baseline_ellipse_area_m2"], errors="coerce") > 0.0,
        output_df["ellipse_area_m2"] / output_df["baseline_ellipse_area_m2"],
        np.nan,
    )
    return output_df


def _nanpercentile(values: pd.Series, percentile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(np.nanpercentile(numeric.to_numpy(dtype=float), percentile))


def build_method_shift_summary(shift_df: pd.DataFrame) -> pd.DataFrame:
    if shift_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_columns = [
        "method_step",
        "method_key",
        "method_label",
        "estimate_type",
        "estimate_label",
    ]
    for keys, subset in shift_df.groupby(group_columns, dropna=False):
        subset = subset.copy()
        rows.append(
            {
                "method_step": keys[0],
                "method_key": keys[1],
                "method_label": keys[2],
                "estimate_type": keys[3],
                "estimate_label": keys[4],
                "event_count": int(subset["event_id"].nunique()),
                "mean_catalog_horizontal_shift_m": pd.to_numeric(
                    subset["catalog_horizontal_shift_m"],
                    errors="coerce",
                ).mean(),
                "median_catalog_horizontal_shift_m": pd.to_numeric(
                    subset["catalog_horizontal_shift_m"],
                    errors="coerce",
                ).median(),
                "p90_catalog_horizontal_shift_m": _nanpercentile(
                    subset["catalog_horizontal_shift_m"],
                    90.0,
                ),
                "mean_abs_catalog_depth_shift_m": pd.to_numeric(
                    subset["catalog_depth_shift_m"],
                    errors="coerce",
                )
                .abs()
                .mean(),
                "median_abs_catalog_depth_shift_m": pd.to_numeric(
                    subset["catalog_depth_shift_m"],
                    errors="coerce",
                )
                .abs()
                .median(),
                "mean_baseline_horizontal_shift_m": pd.to_numeric(
                    subset.get("baseline_horizontal_shift_m"),
                    errors="coerce",
                ).mean(),
                "median_baseline_horizontal_shift_m": pd.to_numeric(
                    subset.get("baseline_horizontal_shift_m"),
                    errors="coerce",
                ).median(),
                "p90_baseline_horizontal_shift_m": _nanpercentile(
                    subset.get("baseline_horizontal_shift_m"),
                    90.0,
                ),
                "mean_abs_baseline_depth_shift_m": pd.to_numeric(
                    subset.get("baseline_depth_shift_m"),
                    errors="coerce",
                )
                .abs()
                .mean(),
                "mean_baseline_semi_major_change_m": pd.to_numeric(
                    subset.get("baseline_semi_major_change_m"),
                    errors="coerce",
                ).mean(),
                "mean_baseline_semi_minor_change_m": pd.to_numeric(
                    subset.get("baseline_semi_minor_change_m"),
                    errors="coerce",
                ).mean(),
                "mean_baseline_standard_error_change": pd.to_numeric(
                    subset.get("baseline_standard_error_change"),
                    errors="coerce",
                ).mean(),
                "mean_baseline_ellipse_area_ratio": pd.to_numeric(
                    subset.get("baseline_ellipse_area_ratio"),
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                .mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["method_step", "estimate_type"],
        ascending=[True, True],
    )


def _filter_method_shift_rows(
    shift_df: pd.DataFrame,
    *,
    method_key: str,
    estimate_type: str,
) -> pd.DataFrame:
    if shift_df.empty:
        return pd.DataFrame()
    return shift_df.loc[
        (shift_df["method_key"] == method_key)
        & (shift_df["estimate_type"] == estimate_type)
    ].copy()


def build_method_catalogue_shift_overview(
    *,
    shift_df: pd.DataFrame,
    event_set_df: pd.DataFrame,
    method_key: str,
    estimate_type: str,
) -> dict[str, Any]:
    solved_df = _filter_method_shift_rows(
        shift_df,
        method_key=method_key,
        estimate_type=estimate_type,
    )
    solved_event_ids = set(solved_df.get("event_id", pd.Series(dtype=str)).astype(str))
    unsolved_df = event_set_df.loc[
        ~event_set_df["event_id"].astype(str).isin(solved_event_ids)
    ].copy()

    if solved_df.empty:
        return {
            "method_key": method_key,
            "estimate_type": estimate_type,
            "method_label": method_key,
            "estimate_label": ESTIMATE_LABELS.get(estimate_type, estimate_type),
            "official_event_count": int(len(event_set_df)),
            "solved_event_count": 0,
            "unsolved_event_count": int(len(unsolved_df)),
            "solved_df": solved_df,
            "unsolved_df": unsolved_df,
            "top_shift_df": pd.DataFrame(),
        }

    solved_df = solved_df.copy()
    solved_df["catalog_horizontal_shift_m"] = pd.to_numeric(
        solved_df["catalog_horizontal_shift_m"],
        errors="coerce",
    )
    solved_df["catalog_depth_shift_m"] = pd.to_numeric(
        solved_df["catalog_depth_shift_m"],
        errors="coerce",
    )
    solved_df["semi_major_95_m"] = (
        pd.to_numeric(solved_df["semi_major"], errors="coerce") * _ELLIPSE_95_SCALE
    )
    solved_df["semi_minor_95_m"] = (
        pd.to_numeric(solved_df["semi_minor"], errors="coerce") * _ELLIPSE_95_SCALE
    )
    top_shift_df = solved_df.sort_values(
        "catalog_horizontal_shift_m",
        ascending=False,
    ).head(10)

    return {
        "method_key": method_key,
        "estimate_type": estimate_type,
        "method_label": str(solved_df["method_label"].iloc[0]),
        "estimate_label": str(solved_df["estimate_label"].iloc[0]),
        "official_event_count": int(len(event_set_df)),
        "solved_event_count": int(solved_df["event_id"].nunique()),
        "unsolved_event_count": int(len(unsolved_df)),
        "mean_horizontal_shift_m": float(
            solved_df["catalog_horizontal_shift_m"].mean()
        ),
        "median_horizontal_shift_m": float(
            solved_df["catalog_horizontal_shift_m"].median()
        ),
        "p90_horizontal_shift_m": _nanpercentile(
            solved_df["catalog_horizontal_shift_m"],
            90.0,
        ),
        "max_horizontal_shift_m": float(solved_df["catalog_horizontal_shift_m"].max()),
        "mean_abs_depth_shift_m": float(
            solved_df["catalog_depth_shift_m"].abs().mean()
        ),
        "median_abs_depth_shift_m": float(
            solved_df["catalog_depth_shift_m"].abs().median()
        ),
        "max_abs_depth_shift_m": float(solved_df["catalog_depth_shift_m"].abs().max()),
        "mean_semi_major_95_m": float(solved_df["semi_major_95_m"].mean()),
        "median_semi_major_95_m": float(solved_df["semi_major_95_m"].median()),
        "solved_df": solved_df,
        "unsolved_df": unsolved_df,
        "top_shift_df": top_shift_df,
    }


def _plot_confidence_ellipse(
    ax: Any,
    *,
    center_x_m: float,
    center_y_m: float,
    semi_major_m: float,
    semi_minor_m: float,
    azimuth_deg: float,
    edgecolor: str,
) -> None:
    if any(
        pd.isna(value)
        for value in [center_x_m, center_y_m, semi_major_m, semi_minor_m, azimuth_deg]
    ):
        return

    # azimuth_deg is geodetic (from North, clockwise); matplotlib Ellipse.angle
    # is CCW from the x-axis.  Convert: matplotlib_angle = 90 - azimuth.
    matplotlib_angle = 90.0 - float(azimuth_deg)
    patch = Ellipse(
        xy=(center_x_m, center_y_m),
        width=2.0 * _ELLIPSE_95_SCALE * semi_major_m,
        height=2.0 * _ELLIPSE_95_SCALE * semi_minor_m,
        angle=matplotlib_angle,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=0.45,
    )
    ax.add_patch(patch)


def _magnitude_label(row: Mapping[str, Any]) -> str | None:
    magnitude = _numeric_scalar(row.get("catalog_magnitude"))
    if pd.isna(magnitude):
        event_id = str(row.get("event_id") or "").strip()
        return event_id or None
    magnitude_type = str(row.get("catalog_magnitude_type") or "ML").strip() or "ML"
    return f"{magnitude_type} {magnitude:.2f}"


_EVENT_LABEL_MIN_MAGNITUDE = 2.0


def _select_solved_event_label_rows(solved_df: pd.DataFrame) -> list[dict[str, Any]]:
    if solved_df.empty:
        return []

    frame = solved_df.copy()
    frame["catalog_magnitude"] = pd.to_numeric(
        frame["catalog_magnitude"],
        errors="coerce",
    )
    frame = frame.loc[frame["catalog_magnitude"] > _EVENT_LABEL_MIN_MAGNITUDE]
    if frame.empty:
        return []
    frame = frame.sort_values(
        ["catalog_magnitude", "catalog_origin_time_reference"],
        ascending=[False, True],
    )
    return frame.to_dict("records")


def _expand_bbox(bbox: Bbox, padding_px: float) -> Bbox:
    return Bbox.from_extents(
        bbox.x0 - padding_px,
        bbox.y0 - padding_px,
        bbox.x1 + padding_px,
        bbox.y1 + padding_px,
    )


def _bbox_contains_point(bbox: Bbox, x_px: float, y_px: float) -> bool:
    return bbox.x0 <= x_px <= bbox.x1 and bbox.y0 <= y_px <= bbox.y1


def _estimated_annotation_bbox(
    *,
    figure: Any,
    anchor_x_px: float,
    anchor_y_px: float,
    label: str,
    offset_x_points: float,
    offset_y_points: float,
    horizontal_alignment: str,
    vertical_alignment: str,
) -> Bbox:
    points_to_pixels = figure.dpi / 72.0
    offset_x_px = float(offset_x_points) * points_to_pixels
    offset_y_px = float(offset_y_points) * points_to_pixels
    text_anchor_x = anchor_x_px + offset_x_px
    text_anchor_y = anchor_y_px + offset_y_px

    font_size_px = _EVENT_LABEL_FONT_SIZE * points_to_pixels
    text_width_px = max(1, len(label)) * font_size_px * 0.62 + 8.0
    text_height_px = font_size_px * 1.55 + 4.0

    if horizontal_alignment == "left":
        x0 = text_anchor_x
        x1 = text_anchor_x + text_width_px
    elif horizontal_alignment == "right":
        x0 = text_anchor_x - text_width_px
        x1 = text_anchor_x
    else:
        x0 = text_anchor_x - (text_width_px / 2.0)
        x1 = text_anchor_x + (text_width_px / 2.0)

    if vertical_alignment == "bottom":
        y0 = text_anchor_y
        y1 = text_anchor_y + text_height_px
    elif vertical_alignment == "top":
        y0 = text_anchor_y - text_height_px
        y1 = text_anchor_y
    else:
        y0 = text_anchor_y - (text_height_px / 2.0)
        y1 = text_anchor_y + (text_height_px / 2.0)

    return _expand_bbox(Bbox.from_extents(x0, y0, x1, y1), _LABEL_BBOX_PADDING_PX)


def _draw_solved_event_labels(ax: Any, solved_df: pd.DataFrame) -> int:
    label_rows = _select_solved_event_label_rows(solved_df)
    if not label_rows:
        return 0

    figure = ax.figure

    anchor_points_px: list[tuple[str, float, float]] = []
    for row in label_rows:
        event_id = str(row.get("event_id") or "")
        label_x = _numeric_scalar(row.get("x"))
        label_y = _numeric_scalar(row.get("y"))
        if any(pd.isna(value) for value in [label_x, label_y]):
            continue
        anchor_x_px, anchor_y_px = ax.transData.transform((label_x, label_y))
        anchor_points_px.append((event_id, float(anchor_x_px), float(anchor_y_px)))

    occupied_bboxes: list[Bbox] = []
    placed_count = 0
    for row in label_rows:
        event_id = str(row.get("event_id") or "")
        label = _magnitude_label(row)
        label_x = _numeric_scalar(row.get("x"))
        label_y = _numeric_scalar(row.get("y"))
        if label is None or any(pd.isna(value) for value in [label_x, label_y]):
            continue

        anchor_x_px, anchor_y_px = ax.transData.transform((label_x, label_y))
        for offset_x, offset_y in _LABEL_OFFSETS:
            horizontal_alignment = "center"
            if offset_x > 0:
                horizontal_alignment = "left"
            elif offset_x < 0:
                horizontal_alignment = "right"

            vertical_alignment = "center"
            if offset_y > 0:
                vertical_alignment = "bottom"
            elif offset_y < 0:
                vertical_alignment = "top"

            bbox = _estimated_annotation_bbox(
                figure=figure,
                anchor_x_px=float(anchor_x_px),
                anchor_y_px=float(anchor_y_px),
                label=label,
                offset_x_points=float(offset_x),
                offset_y_points=float(offset_y),
                horizontal_alignment=horizontal_alignment,
                vertical_alignment=vertical_alignment,
            )
            overlaps_label = any(
                bbox.overlaps(existing) for existing in occupied_bboxes
            )
            covers_other_anchor = any(
                other_event_id != event_id
                and _bbox_contains_point(bbox, other_x_px, other_y_px)
                for other_event_id, other_x_px, other_y_px in anchor_points_px
            )
            covers_own_anchor = _bbox_contains_point(bbox, anchor_x_px, anchor_y_px)
            if overlaps_label or covers_other_anchor or covers_own_anchor:
                continue

            ax.annotate(
                label,
                xy=(label_x, label_y),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=_EVENT_LABEL_FONT_SIZE,
                color="#111827",
                ha=horizontal_alignment,
                va=vertical_alignment,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.78,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#6b7280",
                    "linewidth": 0.7,
                    "alpha": 0.72,
                    "shrinkA": 2.0,
                    "shrinkB": 2.0,
                },
                zorder=4,
            )
            occupied_bboxes.append(bbox)
            placed_count += 1
            break

    return placed_count


def _event_space_circle(
    event_space: Mapping[str, Any] | None,
) -> tuple[float, float, float] | None:
    if not isinstance(event_space, Mapping):
        return None
    if str(event_space.get("type") or "").lower() != "circle":
        return None
    center_lon = _numeric_scalar(event_space.get("center_lon"))
    center_lat = _numeric_scalar(event_space.get("center_lat"))
    radius_km = _numeric_scalar(event_space.get("radius_km"))
    if any(pd.isna(value) for value in [center_lon, center_lat, radius_km]):
        return None
    transformer = prj.Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    center_x_m, center_y_m = transformer.transform(center_lon, center_lat)
    return float(center_x_m), float(center_y_m), float(radius_km) * 1000.0


def _vector_read_target(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        return f"zip://{path.resolve()}"
    return str(path)


def find_ellipse_fault_crossings(
    solved_df: pd.DataFrame,
    *,
    faults_path: Path | None,
    fault_name: str,
) -> pd.DataFrame:
    """Return rows from *solved_df* whose 95 % uncertainty ellipse intersects *fault_name*.

    Parameters
    ----------
    solved_df:
        Rows as produced by :func:`build_method_catalogue_shift_overview`
        (must have columns x, y, semi_major, semi_minor, azimuth, event_id).
    faults_path:
        Path to the fault shapefile / zip.  Returns empty DataFrame when None.
    fault_name:
        Case-insensitive fault name to match against the name column.

    Returns
    -------
    pd.DataFrame with the matching rows from *solved_df*.
    """
    faults_gdf = _load_fault_geometries(faults_path)
    if faults_gdf is None or faults_gdf.empty:
        return solved_df.iloc[0:0].copy()

    name_column = _fault_name_column(faults_gdf)
    if name_column is None:
        return solved_df.iloc[0:0].copy()

    fault_mask = faults_gdf[name_column].astype(str).str.upper() == fault_name.upper()
    fault_geom = faults_gdf.loc[fault_mask, "geometry"]
    if fault_geom.empty:
        return solved_df.iloc[0:0].copy()

    if hasattr(fault_geom, "union_all"):
        fault_line = fault_geom.union_all()
    else:
        fault_line = fault_geom.unary_union

    crossing_rows = []
    for row in solved_df.to_dict("records"):
        cx = _numeric_scalar(row.get("x"))
        cy = _numeric_scalar(row.get("y"))
        semi_major = _numeric_scalar(row.get("semi_major"))
        semi_minor = _numeric_scalar(row.get("semi_minor"))
        azimuth = _numeric_scalar(row.get("azimuth"))
        if any(pd.isna(v) for v in [cx, cy, semi_major, semi_minor, azimuth]):
            continue
        # Build shapely ellipse: unit circle → scale to axes → rotate
        rx = float(semi_major) * _ELLIPSE_95_SCALE
        ry = float(semi_minor) * _ELLIPSE_95_SCALE
        circle = Point(float(cx), float(cy)).buffer(1.0, resolution=64)
        ellipse = _shapely_scale(
            circle, xfact=rx, yfact=ry, origin=(float(cx), float(cy))
        )
        # azimuth is geodetic (from North CW); shapely rotate is CCW from x-axis
        ellipse = _shapely_rotate(
            ellipse, -(90.0 - float(azimuth)), origin=(float(cx), float(cy))
        )
        if ellipse.intersects(fault_line):
            crossing_rows.append(row)

    if not crossing_rows:
        return solved_df.iloc[0:0].copy()
    return pd.DataFrame(crossing_rows)


def _load_fault_geometries(faults_path: Path | None) -> gpd.GeoDataFrame | None:
    if faults_path is None or not faults_path.exists():
        return None

    faults_gdf = gpd.read_file(_vector_read_target(faults_path))
    if faults_gdf.empty or "geometry" not in faults_gdf.columns:
        return None

    if faults_gdf.crs is None:
        faults_gdf = faults_gdf.set_crs(_FAULTS_DEFAULT_CRS, allow_override=True)
    else:
        faults_gdf = faults_gdf.to_crs(_FAULTS_DEFAULT_CRS)
    return faults_gdf


def _fault_name_column(faults_gdf: gpd.GeoDataFrame) -> str | None:
    for candidate in ("Name", "name", "NAME", "fault_name", "FaultName"):
        if candidate in faults_gdf.columns:
            return candidate
    return None


def _plot_fault_overlay(
    ax: Any,
    *,
    faults_path: Path | None,
    highlight_fault_names: Collection[str] | None,
) -> dict[str, Any]:
    faults_gdf = _load_fault_geometries(faults_path)
    if faults_gdf is None:
        return {"has_faults": False, "highlight_names": []}

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    viewport = box(x_min, y_min, x_max, y_max)
    faults_gdf = faults_gdf.copy()
    faults_gdf["geometry"] = faults_gdf.geometry.intersection(viewport)
    faults_gdf = faults_gdf.loc[~faults_gdf.geometry.is_empty].copy()
    if faults_gdf.empty:
        return {"has_faults": False, "highlight_names": []}

    faults_gdf.plot(
        ax=ax,
        color="#44403c",
        linewidth=1.5,
        alpha=0.9,
        zorder=2.5,
    )

    highlight_names_upper = {
        str(name).strip().upper()
        for name in (highlight_fault_names or [])
        if str(name).strip()
    }
    if not highlight_names_upper:
        return {"has_faults": True, "highlight_names": []}

    name_column = _fault_name_column(faults_gdf)
    if name_column is None:
        return {"has_faults": True, "highlight_names": []}

    highlight_mask = (
        faults_gdf[name_column].astype(str).str.upper().isin(highlight_names_upper)
    )
    highlight_gdf = faults_gdf.loc[highlight_mask].copy()
    if highlight_gdf.empty:
        return {"has_faults": True, "highlight_names": []}

    highlight_gdf.plot(
        ax=ax,
        color="white",
        linewidth=5.0,
        alpha=0.95,
        zorder=4.4,
    )

    highlight_gdf.plot(
        ax=ax,
        color="#c1121f",
        linewidth=2.8,
        alpha=0.98,
        zorder=4.5,
    )

    highlighted_labels: list[str] = []
    for fault_name, subset in highlight_gdf.groupby(name_column, sort=False):
        if hasattr(subset.geometry, "union_all"):
            combined_geometry = subset.geometry.union_all()
        else:
            combined_geometry = subset.geometry.unary_union
        if combined_geometry.is_empty:
            continue
        label_point = combined_geometry.representative_point()
        ax.annotate(
            str(fault_name),
            xy=(label_point.x, label_point.y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#9d0208",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "#c1121f",
                "alpha": 0.9,
                "linewidth": 0.8,
            },
            zorder=6,
        )
        highlighted_labels.append(str(fault_name))

    return {"has_faults": True, "highlight_names": highlighted_labels}


def write_method_catalogue_shift_map_png(
    *,
    shift_df: pd.DataFrame,
    event_set_df: pd.DataFrame,
    output_path: Path,
    method_key: str,
    estimate_type: str,
    event_space: Mapping[str, Any] | None = None,
    faults_path: Path | None = None,
    highlight_fault_names: Collection[str] | None = None,
    add_basemap: bool = True,
    show_unsolved: bool = True,
    max_semi_major_95_m: float | None = None,
) -> Path:
    overview = build_method_catalogue_shift_overview(
        shift_df=shift_df,
        event_set_df=event_set_df,
        method_key=method_key,
        estimate_type=estimate_type,
    )
    solved_df = overview["solved_df"]
    if max_semi_major_95_m is not None:
        mask = (
            pd.to_numeric(solved_df["semi_major_95_m"], errors="coerce")
            < max_semi_major_95_m
        )
        solved_df = solved_df.loc[mask]
    unsolved_df = overview["unsolved_df"]
    catalogue_df = _add_catalogue_rd_coordinates(event_set_df)
    event_circle = _event_space_circle(event_space)

    fig, ax = plt.subplots(figsize=(14.5, 8.4))

    catalogue_x = pd.to_numeric(catalogue_df["catalog_x"], errors="coerce")
    catalogue_y = pd.to_numeric(catalogue_df["catalog_y"], errors="coerce")

    if event_circle is not None:
        center_x_m, center_y_m, radius_m = event_circle
        circle_patch = Circle(
            (center_x_m, center_y_m),
            radius=radius_m,
            facecolor="none",
            edgecolor="#b56576",
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(circle_patch)

    ax.scatter(
        catalogue_x,
        catalogue_y,
        s=24,
        c="#f3f4f6",
        edgecolors="#495057",
        linewidths=0.5,
        alpha=0.95,
        zorder=3,
    )

    for row in solved_df.to_dict("records"):
        start_x = _numeric_scalar(row.get("catalog_x"))
        start_y = _numeric_scalar(row.get("catalog_y"))
        end_x = _numeric_scalar(row.get("x"))
        end_y = _numeric_scalar(row.get("y"))
        if any(pd.isna(value) for value in [start_x, start_y, end_x, end_y]):
            continue
        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops={
                "arrowstyle": "->",
                "color": "#1f77b4",
                "linewidth": 1.1,
                "alpha": 0.75,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=2,
        )
        ax.scatter(
            [end_x],
            [end_y],
            s=20,
            c="#1f77b4",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        _plot_confidence_ellipse(
            ax,
            center_x_m=end_x,
            center_y_m=end_y,
            semi_major_m=_numeric_scalar(row.get("semi_major")),
            semi_minor_m=_numeric_scalar(row.get("semi_minor")),
            azimuth_deg=_numeric_scalar(row.get("azimuth")),
            edgecolor="#2a9d8f",
        )

    if show_unsolved and not unsolved_df.empty:
        unsolved_df = _add_catalogue_rd_coordinates(unsolved_df)
        unsolved_x = pd.to_numeric(unsolved_df["catalog_x"], errors="coerce")
        unsolved_y = pd.to_numeric(unsolved_df["catalog_y"], errors="coerce")
        ax.scatter(
            unsolved_x,
            unsolved_y,
            s=42,
            marker="x",
            c="#d62828",
            linewidths=1.2,
            zorder=4,
        )

    solved_x = (
        pd.to_numeric(solved_df.get("x"), errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    solved_y = (
        pd.to_numeric(solved_df.get("y"), errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    all_x_parts = [catalogue_x.dropna().to_numpy(dtype=float), solved_x]
    all_y_parts = [catalogue_y.dropna().to_numpy(dtype=float), solved_y]
    if event_circle is not None:
        center_x_m, center_y_m, radius_m = event_circle
        all_x_parts.append(np.array([center_x_m - radius_m, center_x_m + radius_m]))
        all_y_parts.append(np.array([center_y_m - radius_m, center_y_m + radius_m]))
    all_x = np.concatenate(all_x_parts)
    all_y = np.concatenate(all_y_parts)
    if all_x.size and all_y.size:
        pad_x = max(150.0, 0.10 * (float(all_x.max()) - float(all_x.min()) or 1.0))
        pad_y = max(150.0, 0.10 * (float(all_y.max()) - float(all_y.min()) or 1.0))
        ax.set_xlim(float(all_x.min()) - pad_x, float(all_x.max()) + pad_x)
        ax.set_ylim(float(all_y.min()) - pad_y, float(all_y.max()) + pad_y)

    if add_basemap:
        ctx.add_basemap(
            ax,
            crs="EPSG:28992",
            source=ctx.providers.CartoDB.Positron,
            attribution=False,
        )

    fault_overlay = _plot_fault_overlay(
        ax,
        faults_path=faults_path,
        highlight_fault_names=highlight_fault_names,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.20, color="#4b5563")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000.0:.1f}")
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000.0:.1f}")
    )
    ax.set_xlabel("RD X (km)")
    ax.set_ylabel("RD Y (km)")
    ax.set_title(
        f"{overview['method_label']} {overview['estimate_label']} shifts relative to catalogue\n"
        f"Solved: {overview['solved_event_count']}/{overview['official_event_count']} "
        "official events"
    )
    placed_event_labels = _draw_solved_event_labels(ax, solved_df)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#c0c6cf",
            markeredgecolor="#495057",
            markersize=6,
            label="Official catalogue location",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1f77b4",
            markeredgecolor="white",
            markersize=6,
            label=f"{overview['method_label']} {overview['estimate_label']}",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color="#1f77b4",
            linewidth=1.1,
            label="Shift arrow",
        ),
        Ellipse(
            xy=(0, 0),
            width=0.5,
            height=0.2,
            facecolor="none",
            edgecolor="#2a9d8f",
            linewidth=1.0,
            label="Posterior mean 95% ellipse",
        ),
        *(
            [
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    color="#d62828",
                    linestyle="none",
                    markersize=7,
                    markeredgewidth=1.2,
                    label="Official event not reprocessed",
                ),
            ]
            if show_unsolved
            else []
        ),
    ]
    if event_circle is not None:
        legend_handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color="#b56576",
                linewidth=1.4,
                linestyle="--",
                label="Configured Zeerijp region circle",
            )
        )
    if fault_overlay["has_faults"]:
        legend_handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color="#6b7280",
                linewidth=1.0,
                label="Mapped Groningen faults",
            )
        )
    if fault_overlay["highlight_names"]:
        highlighted_label = ", ".join(fault_overlay["highlight_names"])
        legend_handles.append(
            Line2D(
                [0, 1],
                [0, 0],
                color="#c1121f",
                linewidth=2.4,
                label=f"Highlighted fault: {highlighted_label}",
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="none",
            label=(
                f"Labels: {placed_event_labels}/{overview['solved_event_count']} "
                f"reprocessed posterior-mean events (M > {_EVENT_LABEL_MIN_MAGNITUDE:g})"
            ),
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(right=0.77)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _overview_stat_line(label: str, value: Any, *, precision: int = 1) -> str:
    return f"- {label}: {_format_markdown_scalar(value, precision=precision)}"


def write_method_catalogue_shift_overview_markdown(
    *,
    shift_df: pd.DataFrame,
    event_set_df: pd.DataFrame,
    output_path: Path,
    method_key: str,
    estimate_type: str,
    map_filename: str,
) -> Path:
    overview = build_method_catalogue_shift_overview(
        shift_df=shift_df,
        event_set_df=event_set_df,
        method_key=method_key,
        estimate_type=estimate_type,
    )
    lines = [
        f"# {overview['method_label']} {overview['estimate_label']} Relative To Catalogue",
        "",
        f"- Official catalogue events: {overview['official_event_count']}",
        f"- Reprocessed events: {overview['solved_event_count']}",
        f"- Not reprocessed: {overview['unsolved_event_count']}",
        _overview_stat_line(
            "Mean horizontal shift (m)",
            overview.get("mean_horizontal_shift_m"),
        ),
        _overview_stat_line(
            "Median horizontal shift (m)",
            overview.get("median_horizontal_shift_m"),
        ),
        _overview_stat_line(
            "P90 horizontal shift (m)",
            overview.get("p90_horizontal_shift_m"),
        ),
        _overview_stat_line(
            "Maximum horizontal shift (m)",
            overview.get("max_horizontal_shift_m"),
        ),
        _overview_stat_line(
            "Mean absolute depth shift (m)",
            overview.get("mean_abs_depth_shift_m"),
        ),
        _overview_stat_line(
            "Median absolute depth shift (m)",
            overview.get("median_abs_depth_shift_m"),
        ),
        _overview_stat_line(
            "Maximum absolute depth shift (m)",
            overview.get("max_abs_depth_shift_m"),
        ),
        _overview_stat_line(
            "Mean 95% semi-major axis (m)",
            overview.get("mean_semi_major_95_m"),
        ),
        _overview_stat_line(
            "Median 95% semi-major axis (m)",
            overview.get("median_semi_major_95_m"),
        ),
        "",
        f"Map: {map_filename}",
        "",
    ]

    top_shift_df = overview["top_shift_df"]
    if not top_shift_df.empty:
        lines.extend(["## Largest Horizontal Shifts", ""])
        top_rows = []
        for row in top_shift_df.to_dict("records"):
            top_rows.append(
                [
                    str(row.get("event_id", "n/a")),
                    _format_markdown_scalar(
                        row.get("catalog_horizontal_shift_m"), precision=1
                    ),
                    _format_markdown_scalar(
                        row.get("catalog_depth_shift_m"), precision=1
                    ),
                    _format_markdown_scalar(row.get("semi_major_95_m"), precision=1),
                    str(row.get("n_stations", "n/a")),
                    str(row.get("n_phases", "n/a")),
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "Event ID",
                    "Horiz. shift (m)",
                    "Depth shift (m)",
                    "95% semi-major (m)",
                    "Stations",
                    "Phases",
                ],
                top_rows,
            )
        )
        lines.append("")

    unsolved_df = overview["unsolved_df"]
    if not unsolved_df.empty:
        lines.extend(["## Official Events Not Reprocessed", ""])
        unsolved_rows = []
        for row in unsolved_df.to_dict("records"):
            unsolved_rows.append(
                [
                    str(row.get("event_id", "n/a")),
                    _format_markdown_scalar(row.get("catalog_latitude")),
                    _format_markdown_scalar(row.get("catalog_longitude")),
                    _format_markdown_scalar(row.get("catalog_depth_km"), precision=1),
                    _format_markdown_scalar(row.get("catalog_magnitude"), precision=2),
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "Event ID",
                    "Latitude",
                    "Longitude",
                    "Depth (km)",
                    "Magnitude",
                ],
                unsolved_rows,
            )
        )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _clone_origin_for_method(
    origin: Origin,
    *,
    spec: SourceLocationMethodSpec,
    estimate_type: str,
) -> Origin:
    cloned = deepcopy(origin)
    origin_token = _quakeml_uri_token(origin.time)
    cloned.resource_id = (
        f"smi:local/source_location/{spec.key}/{estimate_type}/{origin_token}"
    )
    cloned.method_id = f"smi:local/preseis/source_location/{spec.key}/{estimate_type}"
    cloned.comments = list(cloned.comments)
    cloned.comments.append(Comment(text=f"Method label: {spec.label}"))
    cloned.comments.append(Comment(text=f"Method step: {spec.step}"))
    cloned.comments.append(Comment(text=f"Method key: {spec.key}"))
    cloned.comments.append(
        Comment(text=f"Estimate label: {ESTIMATE_LABELS[estimate_type]}")
    )
    cloned.comments.append(
        Comment(text=f"Learning inventory: {spec.learning_inventory_scope}")
    )
    return cloned


def _build_catalogue_event(row: dict[str, Any]) -> Event:
    event_id = str(row.get("event_id", "unknown_event"))
    catalog_event_id = str(row.get("catalog_event_id") or f"smi:local/{event_id}")

    origin = Origin()
    origin.time = UTCDateTime(str(row["catalog_origin_time_reference"]))

    latitude = _numeric_scalar(row.get("catalog_latitude"))
    longitude = _numeric_scalar(row.get("catalog_longitude"))
    depth_km = _numeric_scalar(row.get("catalog_depth_km"))

    if not pd.isna(latitude):
        origin.latitude = latitude
    if not pd.isna(longitude):
        origin.longitude = longitude
    if not pd.isna(depth_km):
        origin.depth = depth_km * 1000.0

    origin.method_id = "smi:local/catalog/reference_origin"
    origin.creation_info = CreationInfo(
        agency_id=str(row.get("catalog_author") or "KNMI")
    )
    origin.comments.append(
        Comment(
            text=(
                "Fallback base event created from the official event-set catalogue "
                "because no pick or source-location QuakeML file was available."
            )
        )
    )

    event = Event(resource_id=ResourceIdentifier(catalog_event_id), origins=[origin])
    event.preferred_origin_id = origin.resource_id
    event.creation_info = CreationInfo(
        agency_id=str(row.get("catalog_author") or "KNMI")
    )

    magnitude_value = _numeric_scalar(row.get("catalog_magnitude"))
    if not pd.isna(magnitude_value):
        magnitude = Magnitude(
            mag=magnitude_value,
            magnitude_type=str(row.get("catalog_magnitude_type") or ""),
        )
        event.magnitudes.append(magnitude)
        event.preferred_magnitude_id = magnitude.resource_id

    return event


def _load_base_event(
    event_id: str,
    *,
    picks_dir: Path,
    method_specs: list[SourceLocationMethodSpec],
    catalogue_row: dict[str, Any] | None = None,
) -> Event | None:
    pick_xml = picks_dir / f"event_{event_id}_with_arrivals.xml"
    if pick_xml.exists():
        return read_events(str(pick_xml))[0]
    for spec in method_specs:
        quakeml = spec.event_quakeml(event_id)
        if quakeml.exists():
            return read_events(str(quakeml))[0]
    if catalogue_row is not None:
        return _build_catalogue_event(catalogue_row)
    return None


def _iter_method_origins(event: Event) -> list[tuple[str, Origin]]:
    origins: list[tuple[str, Origin]] = []
    for origin in event.origins:
        method_id = str(origin.method_id or "")
        if "Bayesian_maximum_a_posteriori" in method_id:
            origins.append(("map", origin))
        elif "Bayesian_posterior_mean" in method_id:
            origins.append(("posterior_mean", origin))
    return origins


def write_combined_quakeml_catalogue(
    method_specs: list[SourceLocationMethodSpec],
    *,
    event_ids: list[str],
    picks_dir: Path,
    output_dir: Path,
    event_set_df: pd.DataFrame | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_output_dir = output_dir / "quakeml"
    event_output_dir.mkdir(parents=True, exist_ok=True)
    catalogue_lookup: dict[str, dict[str, Any]] = {}
    if event_set_df is not None:
        catalogue_lookup = {
            str(row["event_id"]): row for row in event_set_df.to_dict("records")
        }

    events: list[Event] = []
    for event_id in event_ids:
        event = _load_base_event(
            event_id,
            picks_dir=picks_dir,
            method_specs=method_specs,
            catalogue_row=catalogue_lookup.get(event_id),
        )
        if event is None:
            continue

        base_origin_count = len(event.origins)
        for spec in method_specs:
            method_xml = spec.event_quakeml(event_id)
            if not method_xml.exists():
                continue
            method_event = read_events(str(method_xml))[0]
            for estimate_type, origin in _iter_method_origins(method_event):
                event.origins.append(
                    _clone_origin_for_method(
                        origin, spec=spec, estimate_type=estimate_type
                    )
                )

        event.comments.append(
            Comment(
                text=(
                    "Combined source-location comparison export with descriptive method "
                    "labels for all available MAP and posterior-mean estimates."
                )
            )
        )
        event.comments.append(
            Comment(
                text=(
                    "Calibration methods may use a larger solved residual inventory than "
                    "the official Zeerijp reporting catalogue."
                )
            )
        )

        _sanitize_invalid_resource_identifiers(event)

        if len(event.origins) == base_origin_count:
            event.comments.append(
                Comment(
                    text=(
                        "No MAP or posterior-mean source-location comparison origins "
                        "were available for this official event."
                    )
                )
            )

        event_output_xml = (
            event_output_dir / f"event_{event_id}_with_location_method_comparison.xml"
        )
        write_quakeml_catalog(Catalog(events=[event]), event_output_xml)
        events.append(event)

    catalogue_path = output_dir / "zeerijp_source_location_method_comparison.xml"
    write_quakeml_catalog(Catalog(events=events), catalogue_path)
    return catalogue_path


def _ellipse_points_rd(
    *,
    x: float,
    y: float,
    semi_major: float,
    semi_minor: float,
    azimuth_deg: float,
    point_count: int = 72,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=True)
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    cos_az = np.cos(azimuth_rad)
    sin_az = np.sin(azimuth_rad)
    dx = semi_major * np.cos(angles)
    dy = semi_minor * np.sin(angles)
    x_rot = x + dx * cos_az - dy * sin_az
    y_rot = y + dx * sin_az + dy * cos_az
    return np.column_stack([x_rot, y_rot])


def _ellipse_points_latlon(row: pd.Series) -> list[tuple[float, float]]:
    semi_major = _numeric_scalar(row.get("semi_major"))
    semi_minor = _numeric_scalar(row.get("semi_minor"))
    azimuth = _numeric_scalar(row.get("azimuth"))
    x = _numeric_scalar(row.get("x"))
    y = _numeric_scalar(row.get("y"))
    if any(pd.isna(value) for value in [semi_major, semi_minor, azimuth, x, y]):
        return []

    points_rd = _ellipse_points_rd(
        x=float(x),
        y=float(y),
        semi_major=float(semi_major),
        semi_minor=float(semi_minor),
        azimuth_deg=float(azimuth),
    )
    transformer = prj.Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(points_rd[:, 0].tolist(), points_rd[:, 1].tolist())
    return list(zip(lat, lon))


def _popup_html(row: pd.Series) -> str:
    fit_value = _numeric_scalar(row.get("fit_metric_value"))
    fit_text = "n/a" if pd.isna(fit_value) else f"{float(fit_value):.3f}"
    return (
        f"<b>{row['event_id']}</b><br>"
        f"Method: {row['method_label']}<br>"
        f"Estimate: {row['estimate_label']}<br>"
        f"Fit ({row['fit_metric_name']}): {fit_text}<br>"
        f"Stations: {row.get('n_stations', 'n/a')}<br>"
        f"Phases: {row.get('n_phases', 'n/a')}<br>"
        f"Origin shift (s): {row.get('origin_time_shift', 'n/a')}"
    )


def _new_interactive_map(
    *,
    center_lat: float,
    center_lon: float,
    zoom_start: int,
) -> folium.Map:
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    return fmap


def _add_method_estimate_layers(
    fmap: folium.Map,
    *,
    method_specs: list[SourceLocationMethodSpec],
    estimate_df: pd.DataFrame,
    event_id: str | None = None,
    show_ellipses: bool = False,
    show_posterior_mean: bool = False,
) -> None:
    for spec in method_specs:
        color = METHOD_COLORS.get(spec.key, "#9467bd")
        for estimate_type, estimate_label in ESTIMATE_LABELS.items():
            subset = estimate_df.loc[
                (estimate_df["method_key"] == spec.key)
                & (estimate_df["estimate_type"] == estimate_type)
            ].copy()
            if event_id is not None:
                subset = subset.loc[subset["event_id"] == event_id].copy()
            if subset.empty:
                continue

            point_group = folium.FeatureGroup(
                name=f"{spec.label} - {estimate_label} points",
                show=estimate_type == "map" or show_posterior_mean,
            )
            ellipse_group = folium.FeatureGroup(
                name=f"{spec.label} - {estimate_label} ellipses",
                show=show_ellipses and (estimate_type == "map" or show_posterior_mean),
            )
            for _, row in subset.iterrows():
                lat = _numeric_scalar(row.get("latitude"))
                lon = _numeric_scalar(row.get("longitude"))
                if pd.isna(lat) or pd.isna(lon):
                    continue
                marker_radius = 6 if estimate_type == "map" else 5
                fill_opacity = 0.85 if estimate_type == "map" else 0.55
                popup = folium.Popup(_popup_html(row), max_width=320)
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=marker_radius,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                    popup=popup,
                    tooltip=str(row["event_id"]),
                ).add_to(point_group)

                ellipse_points = _ellipse_points_latlon(row)
                if ellipse_points:
                    folium.PolyLine(
                        locations=ellipse_points,
                        color=color,
                        weight=2,
                        opacity=0.7 if estimate_type == "map" else 0.45,
                        popup=popup,
                    ).add_to(ellipse_group)
            point_group.add_to(fmap)
            ellipse_group.add_to(fmap)


def write_interactive_method_map(
    method_specs: list[SourceLocationMethodSpec],
    *,
    estimate_df: pd.DataFrame,
    event_set_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    center_lat = float(
        pd.to_numeric(event_set_df["catalog_latitude"], errors="coerce").mean()
    )
    center_lon = float(
        pd.to_numeric(event_set_df["catalog_longitude"], errors="coerce").mean()
    )
    unsolved_df = events_without_estimates(event_set_df, estimate_df)

    fmap = _new_interactive_map(
        center_lat=center_lat,
        center_lon=center_lon,
        zoom_start=12,
    )

    catalog_group = folium.FeatureGroup(
        name=f"Official catalogue ({len(event_set_df)} events)",
        show=True,
    )
    for row in event_set_df.to_dict("records"):
        lat = _numeric_scalar(row.get("catalog_latitude"))
        lon = _numeric_scalar(row.get("catalog_longitude"))
        if pd.isna(lat) or pd.isna(lon):
            continue
        popup = (
            f"<b>{row['event_id']}</b><br>"
            f"Catalog time: {row.get('catalog_origin_time_reference', 'n/a')}<br>"
            f"Magnitude: {row.get('catalog_magnitude', 'n/a')}"
        )
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=3,
            color="#6c757d",
            fill=True,
            fill_opacity=0.5,
            popup=popup,
            tooltip=str(row["event_id"]),
        ).add_to(catalog_group)
    catalog_group.add_to(fmap)

    if not unsolved_df.empty:
        unsolved_group = folium.FeatureGroup(
            name=f"Official events without solved estimates ({len(unsolved_df)})",
            show=True,
        )
        for row in unsolved_df.to_dict("records"):
            lat = _numeric_scalar(row.get("catalog_latitude"))
            lon = _numeric_scalar(row.get("catalog_longitude"))
            if pd.isna(lat) or pd.isna(lon):
                continue
            popup = (
                f"<b>{row['event_id']}</b><br>"
                f"Catalog time: {row.get('catalog_origin_time_reference', 'n/a')}<br>"
                f"Magnitude: {row.get('catalog_magnitude', 'n/a')}<br>"
                "Comparison status: no solved MAP/posterior-mean estimate available"
            )
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=7,
                color="#c92a2a",
                weight=2,
                fill=True,
                fill_color="#fff5f5",
                fill_opacity=0.35,
                popup=popup,
                tooltip=f"{row['event_id']} (unsolved)",
            ).add_to(unsolved_group)
        unsolved_group.add_to(fmap)

    _add_method_estimate_layers(
        fmap,
        method_specs=method_specs,
        estimate_df=estimate_df,
        show_ellipses=True,
    )

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(output_path))
    return output_path


def write_event_interactive_method_maps(
    method_specs: list[SourceLocationMethodSpec],
    *,
    estimate_df: pd.DataFrame,
    event_set_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for row in event_set_df.to_dict("records"):
        event_id = str(row["event_id"])
        catalog_lat = _numeric_scalar(row.get("catalog_latitude"))
        catalog_lon = _numeric_scalar(row.get("catalog_longitude"))
        if "event_id" in estimate_df.columns:
            event_estimates = estimate_df.loc[
                estimate_df["event_id"] == event_id
            ].copy()
        else:
            event_estimates = pd.DataFrame()
        estimate_latitudes = pd.to_numeric(
            event_estimates.get("latitude", pd.Series(dtype=float)),
            errors="coerce",
        )
        estimate_longitudes = pd.to_numeric(
            event_estimates.get("longitude", pd.Series(dtype=float)),
            errors="coerce",
        )

        if pd.isna(catalog_lat):
            catalog_lat = float(estimate_latitudes.mean())
        if pd.isna(catalog_lon):
            catalog_lon = float(estimate_longitudes.mean())
        if pd.isna(catalog_lat) or pd.isna(catalog_lon):
            continue

        fmap = _new_interactive_map(
            center_lat=float(catalog_lat),
            center_lon=float(catalog_lon),
            zoom_start=14,
        )

        catalog_group = folium.FeatureGroup(name="Catalog reference point", show=True)
        popup = (
            f"<b>{event_id}</b><br>"
            f"Catalog time: {row.get('catalog_origin_time_reference', 'n/a')}<br>"
            f"Magnitude: {row.get('catalog_magnitude', 'n/a')}"
        )
        folium.CircleMarker(
            location=[float(catalog_lat), float(catalog_lon)],
            radius=5,
            color="#6c757d",
            weight=2,
            fill=True,
            fill_color="#6c757d",
            fill_opacity=0.5,
            popup=popup,
            tooltip=event_id,
        ).add_to(catalog_group)
        catalog_group.add_to(fmap)

        if event_estimates.empty:
            unsolved_group = folium.FeatureGroup(
                name="No solved estimates available",
                show=True,
            )
            folium.CircleMarker(
                location=[float(catalog_lat), float(catalog_lon)],
                radius=8,
                color="#c92a2a",
                weight=2,
                fill=True,
                fill_color="#fff5f5",
                fill_opacity=0.35,
                popup=(
                    f"<b>{event_id}</b><br>"
                    "No solved MAP/posterior-mean estimates are available for this event."
                ),
                tooltip=f"{event_id} (unsolved)",
            ).add_to(unsolved_group)
            unsolved_group.add_to(fmap)

        _add_method_estimate_layers(
            fmap,
            method_specs=method_specs,
            estimate_df=estimate_df,
            event_id=event_id,
            show_ellipses=True,
        )

        folium.LayerControl(collapsed=False).add_to(fmap)
        output_path = output_dir / f"event_{event_id}_source_location_method_map.html"
        fmap.save(str(output_path))
        output_paths.append(output_path)

    return output_paths


def _format_markdown_scalar(value: Any, *, precision: int = 3) -> str:
    numeric = _numeric_scalar(value)
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.{precision}f}"


def _event_method_status_table(
    method_specs: list[SourceLocationMethodSpec],
    *,
    comparison_row: dict[str, Any] | None,
) -> str:
    rows: list[list[str]] = []
    for spec in method_specs:
        if comparison_row is None:
            rows.append([spec.label, "n/a", "n/a", "n/a", "n/a"])
            continue
        rows.append(
            [
                spec.label,
                str(comparison_row.get(f"{spec.key}_source_location_solved", "n/a")),
                str(comparison_row.get(f"{spec.key}_source_location_status", "n/a")),
                str(
                    comparison_row.get(
                        f"{spec.key}_source_location_valid_observations",
                        "n/a",
                    )
                ),
                str(
                    comparison_row.get(
                        f"{spec.key}_source_location_contributing_stations",
                        "n/a",
                    )
                ),
            ]
        )
    return _markdown_table(
        ["Method", "Solved", "Status", "Valid observations", "Stations"],
        rows,
    )


def build_event_method_report(
    method_specs: list[SourceLocationMethodSpec],
    *,
    event_row: dict[str, Any],
    event_estimates: pd.DataFrame,
    event_shifts: pd.DataFrame,
    comparison_row: dict[str, Any] | None,
) -> str:
    event_id = str(event_row["event_id"])
    lines = [f"# {event_id} Source Location Comparison", ""]
    lines.extend(
        [
            "## Catalogue Reference",
            "",
            f"- Origin time: {event_row.get('catalog_origin_time_reference', 'n/a')}",
            f"- Latitude: {_format_markdown_scalar(event_row.get('catalog_latitude'))}",
            f"- Longitude: {_format_markdown_scalar(event_row.get('catalog_longitude'))}",
            f"- Depth (km): {_format_markdown_scalar(event_row.get('catalog_depth_km'))}",
            f"- Magnitude: {_format_markdown_scalar(event_row.get('catalog_magnitude'))}",
            f"- Magnitude type: {event_row.get('catalog_magnitude_type', 'n/a')}",
            f"- Author: {event_row.get('catalog_author', 'n/a')}",
            "",
            "## Method Status",
            "",
            _event_method_status_table(method_specs, comparison_row=comparison_row),
            "",
            "## Available Estimates",
            "",
        ]
    )

    if event_estimates.empty:
        lines.append(
            "No solved MAP or posterior-mean estimates were available for this event."
        )
        lines.append("")
        return "\n".join(lines)

    estimate_rows: list[list[str]] = []
    for row in event_estimates.to_dict("records"):
        estimate_rows.append(
            [
                str(row.get("method_label", "n/a")),
                str(row.get("estimate_label", "n/a")),
                _format_markdown_scalar(row.get("latitude")),
                _format_markdown_scalar(row.get("longitude")),
                _format_markdown_scalar(row.get("x"), precision=1),
                _format_markdown_scalar(row.get("y"), precision=1),
                _format_markdown_scalar(row.get("z"), precision=1),
                _format_markdown_scalar(row.get("semi_major"), precision=1),
                _format_markdown_scalar(row.get("semi_minor"), precision=1),
                _format_markdown_scalar(row.get("azimuth"), precision=1),
                str(row.get("fit_metric_name", "n/a")),
                _format_markdown_scalar(row.get("fit_metric_value")),
                str(row.get("n_stations", "n/a")),
                str(row.get("n_phases", "n/a")),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "Method",
                "Estimate",
                "Latitude",
                "Longitude",
                "RD x",
                "RD y",
                "Depth z",
                "Semi-major",
                "Semi-minor",
                "Azimuth",
                "Fit metric",
                "Fit value",
                "Stations",
                "Phases",
            ],
            estimate_rows,
        )
    )
    lines.append("")

    if not event_shifts.empty:
        shift_rows: list[list[str]] = []
        for row in event_shifts.to_dict("records"):
            shift_rows.append(
                [
                    str(row.get("method_label", "n/a")),
                    str(row.get("estimate_label", "n/a")),
                    _format_markdown_scalar(
                        row.get("catalog_horizontal_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("catalog_depth_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("baseline_horizontal_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("baseline_depth_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("baseline_semi_major_change_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("baseline_standard_error_change"),
                    ),
                ]
            )
        lines.append("## Relative Shifts")
        lines.append("")
        lines.append(
            _markdown_table(
                [
                    "Method",
                    "Estimate",
                    "Horiz. vs catalogue (m)",
                    "Depth vs catalogue (m)",
                    "Horiz. vs baseline (m)",
                    "Depth vs baseline (m)",
                    "Semi-major Δ vs baseline (m)",
                    "Std err Δ vs baseline",
                ],
                shift_rows,
            )
        )
        lines.append("")

    return "\n".join(lines)


def write_event_method_reports(
    method_specs: list[SourceLocationMethodSpec],
    *,
    comparison_df: pd.DataFrame,
    estimate_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    event_set_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    comparison_lookup = {
        str(row["event_id"]): row for row in comparison_df.to_dict("records")
    }

    for event_row in event_set_df.to_dict("records"):
        event_id = str(event_row["event_id"])
        if "event_id" in estimate_df.columns:
            event_estimates = estimate_df.loc[
                estimate_df["event_id"] == event_id
            ].copy()
        else:
            event_estimates = pd.DataFrame()
        if "event_id" in shift_df.columns:
            event_shifts = shift_df.loc[shift_df["event_id"] == event_id].copy()
        else:
            event_shifts = pd.DataFrame()
        report_text = build_event_method_report(
            method_specs,
            event_row=event_row,
            event_estimates=event_estimates,
            event_shifts=event_shifts,
            comparison_row=comparison_lookup.get(event_id),
        )
        output_path = output_dir / f"event_{event_id}_source_location_method_report.md"
        output_path.write_text(report_text, encoding="utf-8")
        output_paths.append(output_path)

    return output_paths


def _markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def build_method_comparison_readme(
    *,
    event_set_name: str,
    metadata_df: pd.DataFrame,
    estimate_df: pd.DataFrame,
    shift_summary_df: pd.DataFrame,
) -> str:
    official_event_count = 0
    if (
        not metadata_df.empty
        and "official_catalogue_event_count" in metadata_df.columns
    ):
        official_event_count = int(metadata_df["official_catalogue_event_count"].max())

    lines = [f"# {event_set_name} Source Location Method Comparison", ""]
    lines.extend(
        [
            (
                "This export compares multiple source-location methods on the "
                "official event-set catalogue."
            ),
            (
                "Calibration methods may learn their delay and sigma terms on a "
                "larger solved residual inventory than the official report catalogue."
            ),
            (
                "The step numbering is the intended hierarchy: step 0 is the simplest "
                "baseline and higher step numbers add increasingly structured calibration."
            ),
            f"Official catalogue size: {official_event_count} events.",
            "",
            "## Methods",
            "",
        ]
    )
    method_rows = []
    for row in metadata_df.to_dict("records"):
        method_rows.append(
            [
                str(row["method_step"]),
                str(row["method_key"]),
                str(row["method_label"]),
                str(row["official_catalogue_solved_event_count"]),
                str(row["learning_inventory_event_count"]),
                str(row["learning_inventory_scope"]),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "Step",
                "Method key",
                "Method label",
                "Solved official events",
                "Learning inventory events",
                "Learning inventory scope",
            ],
            method_rows,
        )
    )
    lines.append("")
    lines.append("## Estimate Types")
    lines.append("")
    lines.append("- `map`: Bayesian maximum a posteriori point estimate")
    lines.append("- `posterior_mean`: posterior expectation of location")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(
        "- `*_method_metadata.csv`: method labels and learning-inventory scope"
    )
    lines.append(
        "- `*_method_catalogue.csv`: one row per official event with method-prefixed columns"
    )
    lines.append(
        "- `*_method_estimates.csv`: long-form solved location estimates with descriptive labels"
    )
    lines.append(
        "- `*_method_shifts.csv`: long-form per-event shifts relative to the "
        "official catalogue and baseline"
    )
    lines.append(
        "- `*_method_shift_summary.csv`: aggregated shift and uncertainty-change "
        "statistics by method and estimate type"
    )
    lines.append(
        "- `*_step0_baseline_posterior_mean_catalogue_overview.md`: baseline-only "
        "posterior-mean summary relative to the official catalogue"
    )
    lines.append(
        "- `*_step0_baseline_posterior_mean_catalogue_shift_map.png`: static RD map "
        "with catalogue points, shift arrows, baseline 95% ellipses, and unsolved events"
    )
    lines.append(
        "- `*_method_comparison.xml`: combined QuakeML catalogue with descriptive method IDs"
    )
    lines.append(
        "- `*_method_map.html`: interactive map with all official events, "
        "unsolved-event highlights, all method point estimates, and visible "
        "ellipse layers by default"
    )
    lines.append(
        "- `event_maps/`: one per-event interactive HTML map with the catalogue "
        "reference point plus all available method locations and ellipses on a "
        "single offline-safe map"
    )
    lines.append(
        "- `event_reports/`: one per-event markdown report with catalogue values, "
        "method status, and all available location/ellipse metrics"
    )
    lines.append(
        "- `quakeml/`: one per-event comparison QuakeML for every official "
        "catalogue event; unsolved events remain present with a status comment"
    )
    lines.append("")

    if not estimate_df.empty:
        solved_counts = (
            estimate_df.groupby(["method_label", "estimate_label"])["event_id"]
            .nunique()
            .reset_index(name="event_count")
        )
        lines.append("## Available Estimates")
        lines.append("")
        rows = [
            [
                str(row["method_label"]),
                str(row["estimate_label"]),
                str(int(row["event_count"])),
            ]
            for row in solved_counts.to_dict("records")
        ]
        lines.append(_markdown_table(["Method", "Estimate", "Events"], rows))
        lines.append("")

    if not shift_summary_df.empty:
        lines.append("## Shift Statistics Relative To Catalogue")
        lines.append("")
        catalogue_rows = []
        for row in shift_summary_df.to_dict("records"):
            catalogue_rows.append(
                [
                    str(row.get("method_label", "n/a")),
                    str(row.get("estimate_label", "n/a")),
                    str(int(row.get("event_count", 0))),
                    _format_markdown_scalar(
                        row.get("mean_catalog_horizontal_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("median_catalog_horizontal_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("p90_catalog_horizontal_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("mean_abs_catalog_depth_shift_m"),
                        precision=1,
                    ),
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "Method",
                    "Estimate",
                    "Events",
                    "Mean horiz. shift (m)",
                    "Median horiz. shift (m)",
                    "P90 horiz. shift (m)",
                    "Mean |depth shift| (m)",
                ],
                catalogue_rows,
            )
        )
        lines.append("")
        lines.append("## Shift Statistics Relative To Baseline")
        lines.append("")
        baseline_rows = []
        for row in shift_summary_df.to_dict("records"):
            baseline_rows.append(
                [
                    str(row.get("method_label", "n/a")),
                    str(row.get("estimate_label", "n/a")),
                    str(int(row.get("event_count", 0))),
                    _format_markdown_scalar(
                        row.get("mean_baseline_horizontal_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("median_baseline_horizontal_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("mean_abs_baseline_depth_shift_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("mean_baseline_semi_major_change_m"),
                        precision=1,
                    ),
                    _format_markdown_scalar(
                        row.get("mean_baseline_standard_error_change"),
                    ),
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "Method",
                    "Estimate",
                    "Events",
                    "Mean horiz. shift (m)",
                    "Median horiz. shift (m)",
                    "Mean |depth shift| (m)",
                    "Mean semi-major Δ (m)",
                    "Mean std err Δ",
                ],
                baseline_rows,
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
