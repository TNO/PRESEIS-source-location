import xarray as xr
import numpy as np
from scipy.stats import chi2
from matplotlib.patches import Ellipse
import pykonal
from findiff import FinDiff
from functools import cache


def characterize_spatial_distribution(
    data,
    data_covariance,
    prior=xr.DataArray(1),  # can be an xarray structure
    spatial_dimensions=["x", "y", "z"],
    input_dyad=["data", "data_T"],  # specifies input axes for covariance
    output_dyad=["space", "space_T"],  # specifies output axes for covariance
    verbose=False,
):
    # STAGE 1: determine likelihood and posterior
    if verbose:
        print("STAGE 1: determine posterior distribution")

    # invert data covariance matrix C in data precision matrix P
    loglikelihood, logposterior = infer_spatial_distribution(
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

    # determine active stations
    active_stations = get_active_stations(data)

    # return all packaged in dataset
    return xr.merge(
        [
            loglikelihood,
            logposterior,
            location_ML,
            covariance_differential_ML,
            location_MAP,
            covariance_differential_MAP,
            location_mean,
            covariance_integral,
            active_stations,
        ]
    )


def infer_spatial_distribution(
    data, data_covariance, prior, spatial_dimensions, input_dyad, verbose
):
    data_precision = invert_covariance(data_covariance, input_dyad)

    # shift residuals to weighted mean: demean
    if verbose:
        print("...demean residuals")
    data_demean = _demean_data(data, data_precision, input_dyad)
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


def covariance_ellipse(centre, cov, fraction, **kwargs):
    """
    Adapted from https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)

    # width and height of ellipse to draw based on
    # fraction inside
    ndof = len(eigvals)
    nstd = np.sqrt(chi2.ppf(fraction, ndof))
    width, height = 2 * nstd * np.sqrt(eigvals)

    return Ellipse(
        xy=centre,
        width=width,
        height=height,
        angle=np.degrees(theta),
        lw=1,
        facecolor="none",
        **kwargs
    )


def _demean_data(data, data_precision, input_dyad):
    data_weights = data_precision.sum([input_dyad[1]]) / data_precision.sum(input_dyad)
    data_mean = data.weighted(data_weights).mean(input_dyad[0])
    data_demean = data - data_mean

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
