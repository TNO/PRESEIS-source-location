"""Travel time computation utilities for seismic source location.

This module provides two layers of functionality:

Low-level (single function call, no caching)
--------------------------------------------
``compute_traveltimes`` runs the eikonal solver for a stationlist on a
pre-built grid and returns a ``(station, mode, x, y, z)`` DataArray directly.

Cache-based workflow (recommended for multi-event runs)
-------------------------------------------------------
Travel times are computed on a station-centred sub-grid (±max_distance_km)
and stored individually in ``cache/{station_code}.h5``.  Every per-station
grid uses the same *spacing* and the same global snapping rule as
``create_grid_from_bounds`` (coordinates rounded to multiples of *spacing*),
so grids from different stations share a common coordinate lattice.

``sample_velocity_grid`` samples the shared velocity model onto the study-area
grid once per location run.  ``assemble_event_traveltimes`` then loads only
the per-station cache files for the stations that contributed picks to a
specific event and reindexes them to that study-area grid — no permanent
combined file is written.

Cache hit/miss behaviour
------------------------
- Existing cache files are reused unless ``--force`` is passed to
  ``compute_traveltimes.py``.
- A ``UserWarning`` and a ``logger.warning`` entry are emitted for each
  station that is absent from the cache when ``assemble_event_traveltimes``
  is called, and for each station whose coverage does not fully span the
  study area.

Note
----
This module currently lives in ``preseis.source_location``.  In the future,
the caching and velocity-sampling layer may be split off into a dedicated
``preseis.modelling`` subpackage; the API will remain stable through that
move.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import preseis.dgm_velmod_sampler as dvs
import pyproj as prj
import xarray as xr
from preseis.processing.geometry import create_grid_from_bounds

from .source_location import eikonal_solve

logger = logging.getLogger(__name__)


def compute_traveltimes(
    stationlist: xr.DataArray,
    velocity_grid: xr.Dataset,
    grid: xr.Dataset,
) -> xr.DataArray:
    """Compute eikonal travel times from all stations to grid points.

    Uses fast marching method to solve eikonal equation for travel times
    from each station to all grid points for both P and S waves.

    Parameters
    ----------
    stationlist : xr.DataArray
        Station locations with dimensions (station, location)
        where location coords are [x, y, z] in grid CRS
    velocity_grid : xr.Dataset
        Velocity model on grid (must have Vinst variable with mode dimension)
    grid : xr.Dataset
        Computational grid with x, y, z coordinates

    Returns
    -------
    xr.DataArray
        Travel times with dimensions (station, mode, x, y, z) in seconds

    Notes
    -----
    The eikonal solver requires:
    - Uniform grid spacing (delta) in x, y, z
    - Grid origin coordinates
    - Velocity field matching grid dimensions

    The computation is vectorized over stations and modes for efficiency.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create simple station list
    >>> stations = xr.DataArray(
    ...     [[235000, 580000, -300]],
    ...     dims=["station", "location"],
    ...     coords={"station": ["NL.TEST"], "location": ["x", "y", "z"]}
    ... )  # doctest: +SKIP
    >>> # Compute travel times to grid
    >>> tt = compute_traveltimes(stations, vel_grid, grid)  # doctest: +SKIP
    >>> tt.shape  # doctest: +SKIP
    (1, 2, 51, 51, 31)  # 1 station, 2 modes (P,S), grid dimensions
    """
    origin = [
        float(grid["x"].values[0]),
        float(grid["y"].values[0]),
        float(grid["z"].values[0]),
    ]
    delta = [
        float(grid["x"].values[1] - grid["x"].values[0]),
        float(grid["y"].values[1] - grid["y"].values[0]),
        float(grid["z"].values[1] - grid["z"].values[0]),
    ]

    logger.info(f"Computing travel times: origin={origin}, delta={delta}")
    logger.info(f"  Stations: {len(stationlist.station)}")
    logger.info(f"  Grid points: {grid.sizes['x'] * grid.sizes['y'] * grid.sizes['z']}")

    velocity_field = velocity_grid["Vinst"]

    synth_tt = xr.apply_ufunc(
        eikonal_solve,
        stationlist,
        velocity_field,
        input_core_dims=[["location"], ["x", "y", "z"]],
        output_core_dims=[["x", "y", "z"]],
        exclude_dims={"location", "x", "y", "z"},
        kwargs={"origin": origin, "delta": delta},
        output_dtypes=[float],
        vectorize=True,
    ).assign_coords(velocity_field.coords)

    logger.info(
        f"Travel times computed: {synth_tt.shape}, "
        f"range=[{float(synth_tt.min()):.3f}, {float(synth_tt.max()):.3f}]s"
    )
    return synth_tt


# ---------------------------------------------------------------------------
# Cache-based workflow
# ---------------------------------------------------------------------------


def station_cache_path(cache_dir: Path, station_code: str) -> Path:
    """Return path to the cached TT file for *station_code*."""
    return cache_dir / f"{station_code}.h5"


def compute_and_cache_station(
    station_code: str,
    station_da: xr.DataArray,
    velocity_model: xr.Dataset,
    *,
    max_distance_km: float,
    z_min: float,
    z_max: float,
    spacing: float,
    crs: str,
    cache_path: Path,
) -> None:
    """Compute travel times for one station and write them to *cache_path*.

    The per-station grid is centred on the station and extends
    ``max_distance_km`` in ±x and ±y.  Grid coordinates are snapped to
    multiples of *spacing* (same rule as ``create_grid_from_bounds``), so
    all per-station grids share a common global coordinate lattice.

    Parameters
    ----------
    station_code:
        Station identifier, e.g. ``"NL.G054"``.
    station_da:
        ``xr.DataArray`` with dim ``location`` and coords ``["x", "y", "z"]``
        (in the target *crs*).  Typically obtained via
        ``stationlist.sel(station=code)``.
    velocity_model:
        Pre-loaded velocity model dataset (passed to
        ``dvs.sample_velocity_model``).
    max_distance_km:
        Half-extent of the per-station grid in kilometres.
    z_min, z_max:
        Depth range (metres, negative below surface), shared across all
        stations.
    spacing:
        Grid spacing in metres.  Must match the spacing used for the study
        area grid so that coordinates align exactly.
    crs:
        Coordinate reference system string (e.g. ``"EPSG:28992"``).
    cache_path:
        Output ``.h5`` path.  Parent directory is created if needed.

    Raises
    ------
    RuntimeError
        If the eikonal solver raises an unexpected error.
    """
    sx = float(station_da.sel(location="x"))
    sy = float(station_da.sel(location="y"))
    sz = float(station_da.sel(location="z"))

    d = max_distance_km * 1000.0  # km → m
    station_bounds = {"x": (sx - d, sx + d), "y": (sy - d, sy + d)}

    grid = create_grid_from_bounds(station_bounds, z_min, z_max, spacing, crs)
    logger.info(
        "  Station %s: sub-grid %dx%dx%d (±%.1f km)",
        station_code,
        grid.sizes["x"],
        grid.sizes["y"],
        grid.sizes["z"],
        max_distance_km,
    )

    vel_grid = dvs.sample_velocity_model(
        grid.x,
        grid.y,
        grid.z,
        velocity_model=velocity_model,
        crs=prj.CRS(crs),
    )

    # compute_traveltimes expects (station, location) stationlist; wrap the
    # single-station DataArray accordingly, then drop the station dim again.
    single_station = station_da.expand_dims("station").assign_coords(
        station=[station_code]
    )
    tt = compute_traveltimes(single_station, vel_grid, grid)
    tt_single = tt.isel(station=0)  # (mode, x, y, z)

    # Velocity at the station location → (mode,) DataArray
    station_velocity = (
        vel_grid["Vinst"]
        .interp(
            x=xr.DataArray(sx),
            y=xr.DataArray(sy),
            z=xr.DataArray(sz),
        )
        .drop_vars(["x", "y", "z"])
    )
    station_velocity.name = "station_velocity"

    ds = xr.Dataset(
        {
            "traveltime": tt_single,
            "station_velocity": station_velocity,
        }
    )
    ds.attrs.update(
        {
            "station_code": station_code,
            "station_x": sx,
            "station_y": sy,
            "station_z": sz,
            "spacing": spacing,
            "crs": crs,
            "max_distance_km": max_distance_km,
        }
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(cache_path, engine="h5netcdf")
    logger.info(
        "  Saved cache: %s  shape=%s",
        cache_path.name,
        tuple(tt_single.sizes.values()),
    )


def sample_velocity_grid(
    study_bounds: dict[str, tuple[float, float]],
    velocity_model: xr.Dataset,
    *,
    z_min: float,
    z_max: float,
    spacing: float,
    crs: str,
) -> xr.Dataset:
    """Sample the velocity model onto the study-area grid.

    This is a once-per-run operation whose result is passed to
    ``assemble_event_traveltimes`` for every event.  Separating it avoids
    re-sampling the velocity model for each event.

    Parameters
    ----------
    study_bounds:
        Study area bounds ``{"x": (x_min, x_max), "y": (y_min, y_max)}`` in
        *crs*.
    velocity_model:
        Pre-loaded velocity model dataset.
    z_min, z_max:
        Depth range (metres, negative below surface).
    spacing:
        Grid spacing in metres.
    crs:
        Coordinate reference system string.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``velocity(mode, x, y, z)`` and
        ``interface_depth(unit, x, y)``, on the snapped study-area coordinate
        lattice.
    """
    study_grid = create_grid_from_bounds(study_bounds, z_min, z_max, spacing, crs)
    vel_grid = dvs.sample_velocity_model(
        study_grid.x,
        study_grid.y,
        study_grid.z,
        velocity_model=velocity_model,
        crs=prj.CRS(crs),
    )

    interface_depth = vel_grid["depth"].rename("interface_depth")
    interface_depth.attrs.update(
        {
            "description": "Depth of each geological interface on the grid",
            "units": "m (NAP, negative down)",
        }
    )

    return xr.Dataset(
        {
            "velocity": vel_grid["Vinst"],
            "interface_depth": interface_depth,
        }
    )


def assemble_event_traveltimes(
    station_codes: list[str],
    cache_dir: Path,
    velocity_ds: xr.Dataset,
    *,
    nan_fill_value: float = 999.0,
) -> xr.Dataset:
    """Assemble per-station cached travel times for a specific set of stations.

    Loads the cached ``.h5`` file for each station in *station_codes*,
    reindexes it to the study-area grid defined by *velocity_ds*, and fills
    out-of-coverage cells with *nan_fill_value* (default 999 s, giving
    effectively zero likelihood during inference).  Stations with 100 % NaN
    (completely outside the study grid) are dropped with a warning.

    Station coordinates (x, y, z) are read from each cache file's attributes
    so no stationlist is required at call time.

    Parameters
    ----------
    station_codes:
        Station identifiers to include, e.g. ``["NL.G054", "NL.G144"]``.
    cache_dir:
        Directory containing per-station ``.h5`` cache files.
    velocity_ds:
        Output of ``sample_velocity_grid`` -- provides the reference grid and
        ``velocity`` / ``interface_depth`` variables.
    nan_fill_value:
        Sentinel travel time (seconds) for grid cells outside a station's
        cached extent.  Defaults to 999 s.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``traveltime(station, mode, x, y, z)``,
        ``velocity(mode, x, y, z)``, ``station_velocity(station, mode)``, and
        ``interface_depth(unit, x, y)``.  Station coordinates are stored as
        ``station_x``, ``station_y``, ``station_z`` 1-D coordinates.

    Raises
    ------
    RuntimeError
        If no cache files could be loaded for any of the requested stations.
    """
    ref = xr.DataArray(
        np.nan,
        coords={
            "x": velocity_ds["velocity"].x,
            "y": velocity_ds["velocity"].y,
            "z": velocity_ds["velocity"].z,
        },
        dims=["x", "y", "z"],
    )

    tt_list: list[xr.DataArray] = []
    sv_list: list[xr.DataArray] = []
    sx_vals: list[float] = []
    sy_vals: list[float] = []
    sz_vals: list[float] = []
    incomplete_coverage: list[str] = []

    for code in station_codes:
        cache_path = station_cache_path(cache_dir, code)
        if not cache_path.exists():
            logger.warning(
                "Station %s missing from cache; excluded from assembled dataset.", code
            )
            continue

        ds = xr.open_dataset(cache_path, engine="h5netcdf")
        tt_reindexed = ds["traveltime"].reindex_like(ref, fill_value=np.nan)
        sv = ds["station_velocity"]

        nan_fraction = float(np.isnan(tt_reindexed.values).mean())
        if nan_fraction == 1.0:
            logger.warning(
                "Station %s: 100%% of study grid outside TT coverage -- excluded.", code
            )
            continue
        elif nan_fraction > 0.0:
            logger.warning(
                "Station %s: %.1f%% of study grid outside coverage -- filled with %.0f s.",
                code,
                nan_fraction * 100,
                nan_fill_value,
            )
            tt_reindexed = tt_reindexed.fillna(nan_fill_value)
            incomplete_coverage.append(code)

        tt_list.append(tt_reindexed.expand_dims(station=[code]))
        sv_list.append(sv.expand_dims(station=[code]))
        sx_vals.append(float(ds.attrs["station_x"]))
        sy_vals.append(float(ds.attrs["station_y"]))
        sz_vals.append(float(ds.attrs["station_z"]))

    if not tt_list:
        raise RuntimeError(
            f"No cached TT files found in {cache_dir} for any of: {station_codes}"
        )

    synth_tt = xr.concat(tt_list, dim="station")
    station_velocity = xr.concat(sv_list, dim="station")

    ds_event = xr.Dataset(
        {
            "traveltime": synth_tt,
            "velocity": velocity_ds["velocity"],
            "station_velocity": station_velocity,
            "interface_depth": velocity_ds["interface_depth"],
        },
        coords={
            "station_x": ("station", sx_vals),
            "station_y": ("station", sy_vals),
            "station_z": ("station", sz_vals),
        },
    )

    if incomplete_coverage:
        ds_event.attrs["incomplete_coverage_stations"] = ",".join(incomplete_coverage)

    logger.debug(
        "Assembled TT for %d stations, grid shape %s",
        len(tt_list),
        tuple(synth_tt.sizes.values()),
    )
    return ds_event
