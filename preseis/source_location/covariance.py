"""Covariance structure utilities for Bayesian source location.

This module provides functions for:
- Spatial correlation models for travel time uncertainties
- Covariance matrix construction
- Weighted residual calculations for origin time optimization
- Uncertainty ellipse parameter extraction from covariance matrices
"""

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def spatial_correlation_coefficient(
    loc0: NDArray[np.floating], loc1: NDArray[np.floating], corr_len: float
) -> float:
    """Compute spatial correlation coefficient between two locations.

    Uses a Gaussian correlation model with characteristic length scale.

    Parameters
    ----------
    loc0 : NDArray
        First location coordinates (x, y, z)
    loc1 : NDArray
        Second location coordinates (x, y, z)
    corr_len : float
        Correlation length scale (same units as coordinates)

    Returns
    -------
    float
        Correlation coefficient in [0, 1]
    """
    reldist2 = np.sum(((loc1 - loc0) / corr_len) ** 2)
    return float(np.exp(-0.5 * reldist2))


def build_spatial_correlation_matrix(
    stationlist: xr.DataArray,
    correlation_length: float,
) -> xr.DataArray:
    """Build spatial correlation matrix between stations.

    Parameters
    ----------
    stationlist : xr.DataArray
        Station locations with dimensions (station, location) where
        location coordinates are ['x', 'y', 'z']
    correlation_length : float
        Correlation length scale in same units as station coordinates

    Returns
    -------
    xr.DataArray
        Correlation matrix with dimensions (station, station_T)
    """
    return xr.apply_ufunc(
        spatial_correlation_coefficient,
        stationlist,
        stationlist.rename({"station": "station_T"}),
        correlation_length,
        input_core_dims=[["location"], ["location"], []],
        output_core_dims=[[]],
        exclude_dims={"location"},
        vectorize=True,
    )


def build_mode_correlation_matrix(
    modes: list[str], correlation_coefficient: float = 0.2
) -> xr.DataArray:
    """Build correlation matrix between seismic modes (P, S).

    Parameters
    ----------
    modes : list[str]
        List of mode names, e.g., ['P', 'S']
    correlation_coefficient : float
        Off-diagonal correlation coefficient between modes

    Returns
    -------
    xr.DataArray
        Correlation matrix with dimensions (mode, mode_T)
    """
    n = len(modes)
    data = np.eye(n) + correlation_coefficient * (np.ones((n, n)) - np.eye(n))
    return xr.DataArray(
        data=data,
        coords={"mode": modes, "mode_T": modes},
    )


def build_covariance_structure(
    sigma: xr.DataArray,
    spatial_correlation_matrix: xr.DataArray,
    mode_correlation_matrix: xr.DataArray,
) -> xr.DataArray:
    """Build complete covariance structure for travel time uncertainties.

    Combines pick uncertainties, spatial correlation, and mode correlation
    into a full covariance structure.

    Parameters
    ----------
    sigma : xr.DataArray
        Pick uncertainties (standard deviations) with dimension 'mode'
    spatial_correlation_matrix : xr.DataArray
        Station-station correlation matrix with dimensions (station, station_T)
    mode_correlation_matrix : xr.DataArray
        Mode-mode correlation matrix with dimensions (mode, mode_T)

    Returns
    -------
    xr.DataArray
        Full covariance structure with dimensions (mode, mode_T, station, station_T)
    """
    return (
        sigma
        * sigma.rename({"mode": "mode_T"})
        * spatial_correlation_matrix
        * mode_correlation_matrix
    )


def weighted_rms_residual(
    obs_at: xr.DataArray,
    synth_at: xr.DataArray,
    sigma: xr.DataArray,
    time_shift: float = 0.0,
) -> float:
    """Calculate weighted RMS residual for a given origin time correction.

    Parameters
    ----------
    obs_at : xr.DataArray
        Observed arrival times with dimensions (mode, station)
    synth_at : xr.DataArray
        Synthetic arrival times at source location (mode, station)
    sigma : xr.DataArray
        Pick uncertainties (standard deviations) with dimension 'mode'
    time_shift : float
        Origin time correction to apply (subtracted from observed times)

    Returns
    -------
    float
        Weighted RMS residual in units of sigma
    """
    obs_corrected = obs_at - time_shift
    residuals = obs_corrected - synth_at
    weighted_residuals = residuals / sigma
    return float(np.sqrt((weighted_residuals**2).mean()))


def extract_ellipse_parameters(
    cov_matrix: NDArray[np.floating],
) -> dict[str, float]:
    """Extract horizontal uncertainty ellipse parameters from covariance matrix.

    Parameters
    ----------
    cov_matrix : NDArray
        3x3 covariance matrix for (x, y, z) coordinates

    Returns
    -------
    dict
        Dictionary with keys:
        - 'sigma_x': Standard deviation in x
        - 'sigma_y': Standard deviation in y
        - 'sigma_z': Standard deviation in z (depth)
        - 'semi_major': Semi-major axis of horizontal ellipse
        - 'semi_minor': Semi-minor axis of horizontal ellipse
        - 'azimuth': Azimuth of major axis (degrees from north, clockwise)
        - 'rms_horizontal': RMS horizontal uncertainty
    """
    # Standard deviations
    sigma_x = np.sqrt(cov_matrix[0, 0])
    sigma_y = np.sqrt(cov_matrix[1, 1])
    sigma_z = np.sqrt(cov_matrix[2, 2])

    # Check for finite covariance matrix
    if not np.isfinite(cov_matrix).all():
        raise ValueError(
            "Covariance matrix contains non-finite values (NaN or inf). "
            "This may indicate numerical issues in the source location calculation."
        )

    # Analyze horizontal uncertainty ellipse (x, y submatrix)
    cov_xy = cov_matrix[:2, :2]
    eigenvalues, eigenvectors = np.linalg.eig(cov_xy)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Semi-axes of uncertainty ellipse (1-sigma)
    semi_major = np.sqrt(eigenvalues[0])
    semi_minor = np.sqrt(eigenvalues[1])

    # Azimuth of major axis (degrees from north, clockwise)
    # eigenvector[0] is x (east), eigenvector[1] is y (north)
    azimuth = np.degrees(np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0])) % 360

    return {
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "sigma_z": float(sigma_z),
        "semi_major": float(semi_major),
        "semi_minor": float(semi_minor),
        "azimuth": float(azimuth),
        "rms_horizontal": float(np.sqrt((sigma_x**2 + sigma_y**2) / 2)),
    }
