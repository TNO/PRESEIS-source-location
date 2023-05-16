import xarray as xr
import numpy as np
from scipy.stats import chi2
from matplotlib.patches import Ellipse
import pykonal

def safe_inv(arg):
    try: 
        ret = np.linalg.inv(arg)
    except:
        print("unable to invert covariance matrix")
        ret = np.full_like(arg, np.nan)
    return ret


def characterize_spatial_distribution(
        residuals, 
        covariance_matrix, 
        prior = xr.DataArray(1), # can be an xarray structure
        spatial_dimensions = ["x","y","z"], # specifies input
        data_dyad = ["station_mode", "station_mode_"], # specifies input
        coordinate_dyad = ["spatial_coordinates", "spatial_coordinates_"], # specifies output
        verbose = False
    ):

    # STAGE 0: prep
    # subselect relevant entries from covariance matrix
    # this allows use of a complete covariance_matrix with a 
    # limited set of residuals
    res, res_T, covmat = xr.align(
        residuals,
        residuals.rename({data_dyad[0]: data_dyad[1]}),
        covariance_matrix
    )

    # STAGE 1: determine likelihood
    if verbose: print("STAGE 1: determine posterior distribution")

    # invert covariance matrix in precision matrix P
    if verbose: print("...invert data covariance matrix")
    P = xr.apply_ufunc(
        safe_inv,
        covmat,
        input_core_dims = [data_dyad],
        output_core_dims = [data_dyad],
        exclude_dims = set(data_dyad),
        vectorize=True
    )

    # shift residuals to weighted mean: demean
    if verbose: print("...demean residuals")
    P_weights = P.sum([data_dyad[1]]) / P.sum(data_dyad)
    res_mean = res.weighted(P_weights).mean(data_dyad[0])
    res_demean = res - res_mean
    res_demean_T = res_demean.rename({data_dyad[0]: data_dyad[1]})

    # determine squared misfit, likelihood and posterior
    if verbose: print("...determine likelihood and posterior")
    sqmisfit = xr.dot(res_demean_T, P, res_demean, dims=data_dyad)
    likelihood = np.exp(-0.5 * sqmisfit).rename("likelihood")
    posterior = (prior * likelihood).rename("posterior")
    posterior = posterior / posterior.sum(spatial_dimensions)

    # STAGE 2: characterize distribution
    if verbose: print("STAGE 2: characterize posterior distribution")

    # find max a posteriori (MAP) locations
    if verbose: print("...determine maximum a posterior (MAP) location(s)")
    MAP_index = posterior.argmax(dim = spatial_dimensions)
    loc = posterior.isel(MAP_index)
    location_MAP = xr.concat(
        [loc[dim] for dim in spatial_dimensions],
        dim = "spatial_coordinates"
        ).rename("location_MAP").drop(spatial_dimensions)


    # determine slowness at MAP
    if verbose: print("...determine slowness vectors at MAP location(s)")
    # (note inefficiency - calculating all slowness vectors in the volume)
    slowness = xr.concat(
        [res_demean.differentiate(dim) for dim in spatial_dimensions],
        dim = "spatial_coordinates"
        ).assign_coords({"spatial_coordinates": spatial_dimensions})
    slowness = slowness.isel(MAP_index)\
        .drop(spatial_dimensions)\
        .transpose(...,"spatial_coordinates")
    slowness_T = slowness.rename({
        data_dyad[0]: data_dyad[1], 
        coordinate_dyad[0]: coordinate_dyad[1]
        })
    
    # translate temporal/data P into spatial P
    if verbose: print("...determine spatial precision matrix at MAP location(s)")
    spatial_P = xr.dot(slowness_T, P, slowness, dims=data_dyad)

    # invert to obtain spatial covariance matrix
    if verbose: print("...invert precision matrix to covariance matrix at MAP location(s)")
    covariance_differential = xr.apply_ufunc(
        safe_inv,
        spatial_P,
        input_core_dims = [coordinate_dyad],
        output_core_dims = [coordinate_dyad],
        exclude_dims = set(coordinate_dyad),
        vectorize=True
        ).rename("covariance_differential")
    
    # determine mean of spatial distribution
    if verbose: print("...determine mean posterior location(s)")
    location_mean = xr.concat(
        [posterior[dim].weighted(posterior).mean(spatial_dimensions) for dim in spatial_dimensions],
        dim = "spatial_coordinates"
        ).rename("location_mean").assign_coords({"spatial_coordinates": spatial_dimensions})

    # determine second order moment of spatial distribution for covariance matrix
    if verbose: print("...second order moment / integral covariance")
    spdims = [residuals[id] for id in spatial_dimensions]
    distance= xr.concat(spdims, dim = "spatial_coordinates") - location_mean
    distance_tensor = distance * distance.rename({"spatial_coordinates": "spatial_coordinates_"})
    covariance_integral = distance_tensor.weighted(posterior).mean(spatial_dimensions)\
        .transpose(...,*coordinate_dyad)\
        .rename("covariance_integral")

    # include list of contributing stations (for any mode)
    active_stations = xr.full_like(residuals["station"], True).unstack().any("mode").rename("active_stations")

    # return all packaged in dataset
    return xr.merge([
        posterior,
        location_MAP, 
        covariance_differential, 
        location_mean, 
        covariance_integral,
        active_stations
        ])


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
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # width and height of ellipse to draw based on 
    # fraction inside
    ndof = len(eigvals)
    nstd = np.sqrt(chi2.ppf(fraction, ndof))
    width, height = 2 * nstd * np.sqrt(eigvals)

    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), lw=1, facecolor='none', **kwargs)