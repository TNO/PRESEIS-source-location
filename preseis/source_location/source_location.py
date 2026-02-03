from functools import cache

import numpy as np
import pykonal
import xarray as xr
from findiff import FinDiff


def estimate_origin_time_correction(
    residuals: xr.DataArray,
    data_precision: xr.DataArray,
    data_dim: str = "data",
    data_dim_T: str = "data_T",
) -> xr.Dataset:
    """Estimate origin time correction using Generalized Least Squares (GLS).

    Computes the correction to the reference origin time that minimizes
    the weighted squared residuals, properly accounting for pick correlations.

    Terminology
    -----------
    - Reference origin time (t_ref): The assumed origin time used to compute
      observed arrival times. Typically from a catalog or initial estimate.
    - Origin time correction (dt): The shift needed to correct t_ref.
    - True origin time: t_true = t_ref + dt

    The arrival time model is:

        t_obs,i = t_ref + dt + traveltime_i(x) + ε_i

    where ε_i is measurement error. The residual relative to synthetics is:

        r_i = t_obs,i - traveltime_synth,i(x) = dt + ε_i  (if location correct)

    So the mean residual estimates dt, the origin time correction.

    Mathematical Background
    -----------------------
    The negative log-likelihood is:

        -log L(x, dt) = (1/2) * (r - dt·1)ᵀ P (r - dt·1)

    where P = C⁻¹ is the precision matrix. Minimizing over dt gives:

        dt_GLS = (1ᵀ P r) / (1ᵀ P 1) = Σᵢⱼ Pᵢⱼ rⱼ / Σᵢⱼ Pᵢⱼ

    This is the weighted mean with GLS weights wᵢ = Σⱼ Pᵢⱼ / Σᵢⱼ Pᵢⱼ.

    The variance is: Var(dt_GLS) = 1 / (1ᵀ P 1)

    Key Property (Linearity)
    ------------------------
    Since dt_GLS is LINEAR in the residuals r, and r is linear in synthetic
    traveltimes, when marginalizing over spatial uncertainty:

        E[dt] = dt(E[x], E[y], E[z])   (exact, not an approximation)

    Parameters
    ----------
    residuals : xr.DataArray
        Arrival time residuals r = t_obs - traveltime_synth with dimension 'data'.
        These are relative to the reference origin time.
    data_precision : xr.DataArray
        Precision matrix P = C⁻¹ with dimensions (data, data_T).
    data_dim : str
        Name of the data dimension (default: 'data')
    data_dim_T : str
        Name of the transposed data dimension (default: 'data_T')

    Returns
    -------
    xr.Dataset
        - origin_time_correction: GLS estimate of dt (add to t_ref to get t_true)
        - total_precision: 1ᵀP1, for uncertainty σ = 1/√(total_precision)
        - n_observations: Number of valid observations

    See Also
    --------
    marginalize_origin_time : Removes dt from residuals for spatial inference
    infer_source_location : Main entry point using this function
    """
    # GLS weights: wᵢ = Σⱼ Pᵢⱼ / Σᵢⱼ Pᵢⱼ (row sums, normalized)
    precision_row_sums = data_precision.sum(dim=data_dim_T)
    total_precision = data_precision.sum(dim=[data_dim, data_dim_T])
    gls_weights = precision_row_sums / total_precision

    # GLS estimate: dt = Σᵢ wᵢ rᵢ (origin time correction)
    origin_time_correction = (residuals * gls_weights).sum(dim=data_dim)

    # Count valid observations
    n_observations = (~residuals.isnull()).sum(dim=data_dim)

    return xr.Dataset(
        {
            "origin_time_correction": origin_time_correction,
            "total_precision": total_precision,
            "n_observations": n_observations,
        }
    )


def infer_source_location(
    obs_at: xr.DataArray,
    synth_tt: xr.DataArray,
    data_covariance: xr.DataArray,
    prior: xr.DataArray = xr.DataArray(1),
    spatial_dimensions: list[str] = ["x", "y", "z"],
    data_dimensions: list[str] = ["mode", "station"],
    input_dyad: list[str] = ["data", "data_T"],
    output_dyad: list[str] = ["space", "space_T"],
    verbose: bool = False,
) -> xr.Dataset:
    """Infer earthquake source location and origin time correction.

    Performs Bayesian inference of source location (x, y, z) and origin time
    correction (dt), properly accounting for pick correlations. The origin time
    correction is analytically marginalized, enabling efficient spatial inference.

    Input Data Model
    ----------------
    - obs_at: Observed arrival times RELATIVE TO A REFERENCE ORIGIN TIME (t_ref).
      Typically t_ref is from an initial catalog location or trigger time.
    - synth_tt: Synthetic TRAVELTIMES (source-to-station propagation time).

    The arrival time model is:

        obs_at_i = dt + traveltime_i(x) + ε_i

    where:
    - dt = t_true - t_ref is the origin time CORRECTION (what we estimate)
    - traveltime_i(x) is the propagation time from source at x to station i
    - ε_i is measurement error

    The residual r_i = obs_at_i - synth_tt_i(x) estimates dt when x is correct.

    Mathematical Framework
    ----------------------
    The joint posterior over location x and correction dt is:

        p(x, dt | d) ∝ p(d | x, dt) · p(x) · p(dt)

    Since the likelihood is quadratic in dt, we marginalize analytically:

        p(x | d) = ∫ p(x, dt | d) ddt

    This is equivalent to "demeaning" residuals with GLS weights.

    Linearity Property
    ------------------
    Since dt_GLS is linear in residuals:

        E[dt] = dt_GLS(E[x])   (exact equality)

    The posterior mean correction equals the GLS estimate at the posterior
    mean location—no integration needed for E[dt].

    Parameters
    ----------
    obs_at : xr.DataArray
        Observed arrival times RELATIVE TO REFERENCE ORIGIN TIME.
        Dimensions: (mode, station). Units should match synth_tt.
    synth_tt : xr.DataArray
        Synthetic TRAVELTIMES (not arrival times!).
        Dimensions: (mode, station, x, y, z)
    data_covariance : xr.DataArray
        Covariance matrix C with dimensions (data, data_T).
        Encodes pick uncertainties and inter-pick correlations.
    prior : xr.DataArray
        Prior p(x) over spatial dimensions (default: uniform)
    spatial_dimensions : list[str]
        Names of spatial dimensions ['x', 'y', 'z']
    data_dimensions : list[str]
        Names of data dimensions ['mode', 'station']
    input_dyad : list[str]
        Dimension names for covariance matrix ['data', 'data_T']
    output_dyad : list[str]
        Dimension names for spatial covariance ['space', 'space_T']
    verbose : bool
        Print progress messages

    Returns
    -------
    xr.Dataset
        Spatial inference results:
        - loglikelihood, logposterior: Log probability fields
        - location_ML, location_MAP, location_mean: Point estimates
        - covariance_differential_MAP, covariance_integral: Uncertainty

        Origin time correction results:
        - origin_time_correction_MAP: dt at MAP location (add to t_ref for t_true)
        - origin_time_correction_uncertainty: σ = 1/√(1ᵀP1)
        - origin_time_correction_posterior_mean: E[dt] by linearity
        - origin_time_correction_posterior_std: √Var[dt] from spatial uncertainty

    See Also
    --------
    estimate_origin_time_correction : GLS estimation of dt
    summarize_spatial_posterior : Spatial distribution characterization
    """
    if verbose:
        print("Computing residuals and precision matrix...")

    # Compute residuals ONCE
    residuals = obs_at - synth_tt  # broadcasts to (mode, station, x, y, z)

    # Compute precision matrix (inverse covariance) - used for both spatial and temporal
    data_precision = invert_covariance(data_covariance, input_dyad)

    # Stack residuals for characterization
    residuals_stacked = residuals.stack({"data": data_dimensions}).reset_index("data")

    if verbose:
        print("Characterizing spatial distribution...")

    # Get spatial distribution (with origin time marginalized out)
    spatial_result = summarize_spatial_posterior(
        data=residuals_stacked,
        data_covariance=data_covariance,
        prior=prior,
        spatial_dimensions=spatial_dimensions,
        input_dyad=input_dyad,
        output_dyad=output_dyad,
        verbose=verbose,
    )

    if verbose:
        print("Characterizing temporal distribution...")

    # Get MAP and mean locations
    loc_map = {
        dim: float(spatial_result["location_MAP"].sel(space=dim))
        for dim in spatial_dimensions
    }
    loc_mean = {
        dim: float(spatial_result["location_mean"].sel(space=dim))
        for dim in spatial_dimensions
    }

    # GLS estimate of origin time correction at MAP location
    residuals_at_map = residuals_stacked.interp(**loc_map)
    temporal_map = estimate_origin_time_correction(
        residuals_at_map, data_precision, input_dyad[0], input_dyad[1]
    )
    dt_map = temporal_map["origin_time_correction"]
    dt_map_uncertainty = 1.0 / np.sqrt(temporal_map["total_precision"])

    # GLS estimate at posterior mean location (= posterior mean by linearity)
    residuals_at_mean = residuals_stacked.interp(**loc_mean)
    temporal_mean = estimate_origin_time_correction(
        residuals_at_mean, data_precision, input_dyad[0], input_dyad[1]
    )
    dt_at_mean = temporal_mean["origin_time_correction"]

    # Posterior mean of origin time correction
    # By linearity: E[dt] = dt(E[x], E[y], E[z]) exactly
    dt_posterior_mean = float(dt_at_mean)

    # Posterior variance requires field evaluation: Var[dt] = E[dt²] - E[dt]²
    temporal_field = estimate_origin_time_correction(
        residuals_stacked, data_precision, input_dyad[0], input_dyad[1]
    )
    dt_field = temporal_field["origin_time_correction"]

    posterior = np.exp(spatial_result["logposterior"])
    posterior_norm = posterior / posterior.sum()

    dt_sq_mean = float((dt_field**2 * posterior_norm).sum())
    dt_posterior_std = float(np.sqrt(max(0, dt_sq_mean - dt_posterior_mean**2)))

    # Merge all results
    return xr.merge(
        [
            spatial_result,
            dt_map.rename("origin_time_correction_MAP"),
            dt_map_uncertainty.rename("origin_time_correction_uncertainty"),
            dt_at_mean.rename("origin_time_correction_at_mean_location"),
            xr.DataArray(dt_posterior_mean, name="origin_time_correction_posterior_mean"),
            xr.DataArray(dt_posterior_std, name="origin_time_correction_posterior_std"),
        ]
    )


def summarize_spatial_posterior(
    data,
    data_covariance,
    prior=xr.DataArray(1),
    spatial_dimensions=["x", "y", "z"],
    input_dyad=["data", "data_T"],
    output_dyad=["space", "space_T"],
    verbose=False,
):
    """Compute and summarize the spatial posterior distribution.

    Performs Bayesian inference for source location with origin time
    analytically marginalized. Returns point estimates (ML, MAP, mean)
    and uncertainty measures (Hessian-based and integral-based covariances).

    The origin time is analytically marginalized by demeaning residuals
    using GLS weights derived from the precision matrix.

    Parameters
    ----------
    data : xr.DataArray
        Stacked residuals with dimension 'data' and spatial dimensions
    data_covariance : xr.DataArray
        Covariance matrix C with dimensions (data, data_T)
    prior : xr.DataArray
        Prior p(x) over spatial dimensions (default: uniform)
    spatial_dimensions : list[str]
        Names of spatial dimensions
    input_dyad : list[str]
        Dimension names for data covariance
    output_dyad : list[str]
        Dimension names for spatial covariance output
    verbose : bool
        Print progress messages

    Returns
    -------
    xr.Dataset
        - loglikelihood: Log-likelihood field over spatial grid
        - logposterior: Log-posterior field (likelihood × prior, normalized)
        - location_ML: Maximum likelihood location
        - location_MAP: Maximum a posteriori location
        - location_mean: Posterior mean location
        - covariance_differential_*: Hessian-based covariance at ML/MAP
        - covariance_integral: Posterior covariance from second moments
        - active_stations: Which stations contributed data
    """
    # STAGE 1: determine likelihood and posterior
    if verbose:
        print("STAGE 1: determine posterior distribution")

    # Compute likelihood and posterior (with origin time marginalized)
    loglikelihood, logposterior = compute_marginal_likelihood(
        data, data_covariance, prior, spatial_dimensions, input_dyad, verbose
    )

    # STAGE 2: characterize distribution
    if verbose:
        print("STAGE 2: characterize posterior distribution")

    # find max likelihood (ML) locations
    if verbose:
        print("...determine maximum likelihood (ML) location and covariance")
    loc, cov = get_spatial_point_estimate(
        loglikelihood,
        spatial_dimensions,
        output_dyad,
    )
    location_ML = loc.rename("location_ML")
    covariance_differential_ML = cov.rename("covariance_differential_ML")
    determinant_differential_ML = xr.apply_ufunc(
        np.linalg.det,
        covariance_differential_ML,
        input_core_dims=[output_dyad],
        output_core_dims=[[]],
        exclude_dims=set(output_dyad),
    ).rename("determinant_differential_ML")

    # find max a posteriori (MAP) locations
    if verbose:
        print("...determine maximum a posterior (MAP) location and covariance")
    loc, cov = get_spatial_point_estimate(
        logposterior,
        spatial_dimensions,
        output_dyad,
    )
    location_MAP = loc.rename("location_MAP")
    covariance_differential_MAP = cov.rename("covariance_differential_MAP")
    determinant_differential_MAP = xr.apply_ufunc(
        np.linalg.det,
        covariance_differential_MAP,
        input_core_dims=[output_dyad],
        output_core_dims=[[]],
        exclude_dims=set(output_dyad),
    ).rename("determinant_differential_MAP")

    # determine mean and covariance, i.e., first and second order moments
    # of spatial distribution
    if verbose:
        print("...determine posterior moments")
    loc, cov = get_spatial_moments(
        logposterior,
        spatial_dimensions,
        output_dyad,
    )
    location_mean = loc.rename("location_mean")
    covariance_integral = cov.rename("covariance_integral")
    determinant_integral = xr.apply_ufunc(
        np.linalg.det,
        covariance_integral,
        input_core_dims=[output_dyad],
        output_core_dims=[[]],
        exclude_dims=set(output_dyad),
    ).rename("determinant_integral")

    # determine active stations
    active_stations = get_active_stations(data)

    # return all packaged in dataset
    return xr.merge(
        [
            loglikelihood,
            logposterior,
            location_ML,
            covariance_differential_ML,
            determinant_differential_ML,
            location_MAP,
            covariance_differential_MAP,
            determinant_differential_MAP,
            location_mean,
            covariance_integral,
            determinant_integral,
            active_stations,
        ]
    )


def compute_marginal_likelihood(
    data, data_covariance, prior, spatial_dimensions, input_dyad, verbose
):
    """Compute marginal likelihood with origin time analytically marginalized.

    The joint likelihood p(d | x, dt) is quadratic in dt, allowing analytical
    marginalization. The marginal likelihood p(d | x) is obtained by:

    1. Computing precision matrix P = C⁻¹
    2. "Demeaning" residuals: r_demean = r - dt_GLS(x)·1
       where dt_GLS uses GLS weights wᵢ = Σⱼ Pᵢⱼ / Σᵢⱼ Pᵢⱼ
    3. Computing χ² = r_demeanᵀ P r_demean

    This demeaning IS the analytical marginalization over dt.

    Parameters
    ----------
    data : xr.DataArray
        Stacked residuals with dimension 'data' and spatial dimensions
    data_covariance : xr.DataArray
        Covariance matrix C
    prior : xr.DataArray
        Prior p(x) over spatial dimensions
    spatial_dimensions : list[str]
        Names of spatial dimensions
    input_dyad : list[str]
        Dimension names for covariance matrix
    verbose : bool
        Print progress messages

    Returns
    -------
    loglikelihood : xr.DataArray
        Log marginal likelihood log p(d | x)
    logposterior : xr.DataArray
        Log posterior log p(x | d) (normalized)
    """
    data_precision = invert_covariance(data_covariance, input_dyad)

    # Demean residuals with GLS weights (analytically marginalizes origin time)
    if verbose:
        print("...demeaning residuals")
    data_demean = demean_residuals(data, data_precision, input_dyad)
    data_demean_T = data_demean.rename({input_dyad[0]: input_dyad[1]})

    # determine squared misfit, loglikelihood
    if verbose:
        print("...determine likelihood and posterior")
    squared_misfit = xr.dot(data_demean_T, data_precision, data_demean, dims=input_dyad)
    loglikelihood = (-0.5 * squared_misfit).rename("loglikelihood")

    # determine posterior
    # Bayes rule
    logposterior = loglikelihood + np.log(prior)
    total = np.exp(logposterior).sum(spatial_dimensions)
    logposterior = logposterior - np.log(total)
    logposterior = logposterior.rename("logposterior")

    return loglikelihood, logposterior


def get_active_stations(data):
    stations = np.unique(data["station"].data)
    station_status = np.full_like(stations, True, dtype=bool)
    active_stations = xr.DataArray(
        station_status, coords={"station": stations}, name="active_stations"
    )

    return active_stations


def invert_covariance(data_covariance, input_dyad):
    data_precision = xr.apply_ufunc(
        _safe_inv,
        data_covariance,
        input_core_dims=[input_dyad],
        output_core_dims=[input_dyad],
        exclude_dims=set(input_dyad),
        vectorize=True,
    )

    return data_precision


def get_spatial_moments(logposterior, spatial_dimensions, output_dyad):
    posterior = np.exp(logposterior)
    location_mean = (
        xr.concat(
            [
                posterior[dim].weighted(posterior).mean(spatial_dimensions)
                for dim in spatial_dimensions
            ],
            dim=output_dyad[0],
        )
        .rename("location_mean")
        .assign_coords({output_dyad[0]: spatial_dimensions})
    )

    # determine second order moment of spatial distribution for covariance matrix
    spdims = [posterior[id] for id in spatial_dimensions]
    distance = xr.concat(spdims, dim=output_dyad[0]) - location_mean
    distance_tensor = distance * distance.rename({output_dyad[0]: output_dyad[1]})
    covariance_integral = (
        distance_tensor.weighted(posterior)
        .mean(spatial_dimensions)
        .transpose(..., *output_dyad)
        .rename("covariance_integral")
    )

    return location_mean, covariance_integral


def get_spatial_point_estimate(loglikelihood, spatial_dimensions, output_dyad):
    # prep work for Finite Difference calculation of Hessian
    hessian_stencil = _get_hessian_stencil(
        loglikelihood, spatial_dimensions, output_dyad
    )
    space_index_dim = "space_index"
    index = loglikelihood.argmax(dim=spatial_dimensions)
    index_array = xr.Dataset(index).to_array(dim=space_index_dim)

    data = np.exp(loglikelihood.isel(index))
    location = (
        xr.concat([data[dim] for dim in spatial_dimensions], dim=output_dyad[0])
        .rename("location")
        .drop(spatial_dimensions)
    )
    hessian = _calculate_hessian(
        loglikelihood,
        index_array,
        hessian_stencil,
        spatial_dimensions,
        space_index_dim,
    )
    covariance_differential = xr.apply_ufunc(
        _safe_inv,
        -hessian,
        input_core_dims=[output_dyad],
        output_core_dims=[output_dyad],
        exclude_dims=set(output_dyad),
        vectorize=True,
    ).rename("covariance_differential")

    return location, covariance_differential


def eikonal_solve(source_location, velocity, origin, delta):
    xs, ys, zs = velocity.shape
    x0, y0, z0 = origin
    solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
    solver.velocity.min_coords = x0, y0, z0
    solver.velocity.npts = xs, ys, zs
    solver.velocity.node_intervals = delta
    solver.velocity.values = velocity.copy()

    solver.src_loc = source_location
    solver.solve()
    return solver.traveltime.values


def demean_residuals(data, data_precision, input_dyad):
    """Demean residuals using GLS weights from precision matrix.

    Subtracts a weighted mean from residuals, where weights are derived from
    the precision matrix. This operation is mathematically equivalent to
    analytically marginalizing out the origin time correction dt from the
    joint likelihood. The demeaned residuals represent pure measurement error
    after accounting for the optimal origin time correction at each location.

    Terminology
    -----------
    - Residual: r_i = obs_at_i - synth_tt_i(x) = dt + ε_i
    - GLS mean: dt_GLS = Σᵢ wᵢ rᵢ (estimates the origin time correction)
    - Demeaned: r_demean,i = r_i - dt_GLS = ε_i (pure measurement error)

    The GLS weights are: wᵢ = Σⱼ Pᵢⱼ / Σᵢⱼ Pᵢⱼ

    Parameters
    ----------
    data : xr.DataArray
        Residuals with data dimension
    data_precision : xr.DataArray
        Precision matrix P = C⁻¹
    input_dyad : list[str]
        Dimension names [data_dim, data_dim_T]

    Returns
    -------
    xr.DataArray
        Demeaned residuals (origin time correction removed)
    """
    # GLS weights: wᵢ = Σⱼ Pᵢⱼ / Σᵢⱼ Pᵢⱼ
    gls_weights = data_precision.sum([input_dyad[1]]) / data_precision.sum(input_dyad)
    # GLS mean = optimal origin time shift at each location
    gls_mean = data.weighted(gls_weights).mean(input_dyad[0])
    # Demeaned residuals = residuals with origin time removed
    data_demean = data - gls_mean

    return data_demean


def _safe_inv(arg):
    try:
        ret = np.linalg.inv(arg)
    except np.linalg.LinAlgError:
        print("unable to invert covariance matrix")
        ret = np.full_like(arg, np.nan)

    return ret


def _hessian_FD(i, j, spacing, shape):
    if i == j:
        stencil = FinDiff(i, spacing[i], 2).stencil(shape)
    else:
        stencil = FinDiff((i, spacing[i], 1), (j, spacing[j], 1)).stencil(shape)

    return stencil


def _get_hessian_stencil(
    data,
    spatial_dimensions=("x", "y", "z"),
    output_dyad=("space", "space_T"),
):
    spatial_shape = tuple(data[d].size for d in spatial_dimensions)
    spacing = tuple(data[d].diff(d).mean().values.item() for d in spatial_dimensions)
    num_dimensions = len(spatial_dimensions)

    stencil_xarray = xr.DataArray(
        _get_stencil(spatial_shape, spacing, num_dimensions),
        coords={output_dyad[0]: spatial_dimensions, output_dyad[1]: spatial_dimensions},
    )
    return stencil_xarray


@cache
def _get_stencil(spatial_shape, spacing, nd):
    stencil_array = [
        [_hessian_FD(i, j, spacing, spatial_shape) for j in range(nd)]
        for i in range(nd)
    ]

    return stencil_array


def _apply_stencil(ar, idx, st):
    ret = st.apply(ar, tuple(idx))
    return ret


def _calculate_hessian(
    data,
    index_array,
    stencil_array,
    spatial_dimensions=["x", "y", "z"],
    index_dim="space_index",
):
    hessian = xr.apply_ufunc(
        _apply_stencil,
        data,
        index_array,
        stencil_array,
        input_core_dims=[spatial_dimensions, [index_dim], []],
        exclude_dims=set((*spatial_dimensions, index_dim)),
        vectorize=True,
    )
    return hessian
