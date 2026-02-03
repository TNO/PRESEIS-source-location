"""Travel time computation utilities for seismic source location.

This module provides functions for computing eikonal travel times from
stations to grid points for use in Bayesian source location.
"""

import logging

import xarray as xr

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
