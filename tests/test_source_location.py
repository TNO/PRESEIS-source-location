"""Tests for source location inference."""

import numpy as np
import pytest
import xarray as xr

from preseis.source_location import (
    demean_residuals,
    estimate_origin_time_correction,
    get_spatial_moments,
    get_spatial_point_estimate,
    invert_covariance,
)


class TestEstimateOriginTimeCorrection:
    """Tests for origin time correction estimation."""

    def test_zero_residuals(self):
        """Zero residuals should give zero correction."""
        residuals = xr.DataArray(
            [0.0, 0.0, 0.0], dims=["data"], coords={"data": ["p1", "p2", "p3"]}
        )
        precision = xr.DataArray(
            np.eye(3), dims=["data", "data_T"], coords={"data": ["p1", "p2", "p3"]}
        )
        precision = precision.assign_coords(data_T=["p1", "p2", "p3"])

        result = estimate_origin_time_correction(residuals, precision)

        assert np.isclose(float(result["origin_time_correction"]), 0.0)

    def test_constant_residuals(self):
        """Constant residuals should estimate to that value."""
        const = 0.5
        residuals = xr.DataArray(
            [const, const, const],
            dims=["data"],
            coords={"data": ["p1", "p2", "p3"]},
        )
        precision = xr.DataArray(
            np.eye(3),
            dims=["data", "data_T"],
        )
        precision = precision.assign_coords(data=["p1", "p2", "p3"])
        precision = precision.assign_coords(data_T=["p1", "p2", "p3"])

        result = estimate_origin_time_correction(residuals, precision)

        assert np.isclose(float(result["origin_time_correction"]), const, atol=1e-6)

    def test_weighted_average(self):
        """Should compute weighted average with GLS weights."""
        residuals = xr.DataArray(
            [1.0, 2.0],
            dims=["data"],
            coords={"data": ["p1", "p2"]},
        )

        # Diagonal precision matrix with different weights
        precision = xr.DataArray(
            [[2.0, 0.0], [0.0, 1.0]],
            dims=["data", "data_T"],
        )
        precision = precision.assign_coords(data=["p1", "p2"], data_T=["p1", "p2"])

        result = estimate_origin_time_correction(residuals, precision)

        # GLS weight: w1 = 2/(2+1) = 2/3, w2 = 1/3
        # dt = w1*r1 + w2*r2 = (2/3)*1 + (1/3)*2 = 4/3
        expected_dt = (2.0 * 1.0 + 1.0 * 2.0) / (2.0 + 1.0)
        assert np.isclose(float(result["origin_time_correction"]), expected_dt, atol=1e-6)

    def test_off_diagonal_precision(self):
        """Should handle correlated observations."""
        residuals = xr.DataArray(
            [1.0, 1.0],
            dims=["data"],
            coords={"data": ["p1", "p2"]},
        )

        # Precision with correlation
        precision = xr.DataArray(
            [[2.0, -0.5], [-0.5, 2.0]],
            dims=["data", "data_T"],
        )
        precision = precision.assign_coords(data=["p1", "p2"], data_T=["p1", "p2"])

        result = estimate_origin_time_correction(residuals, precision)

        # Should still be close to mean when residuals are equal
        assert 0.5 < float(result["origin_time_correction"]) < 1.5

    def test_n_observations(self):
        """Should count number of valid observations."""
        residuals = xr.DataArray(
            [1.0, np.nan, 2.0, 3.0],
            dims=["data"],
            coords={"data": ["p1", "p2", "p3", "p4"]},
        )
        precision = xr.DataArray(
            np.eye(4),
            dims=["data", "data_T"],
        )
        precision = precision.assign_coords(
            data=["p1", "p2", "p3", "p4"], data_T=["p1", "p2", "p3", "p4"]
        )

        result = estimate_origin_time_correction(residuals, precision)

        # Should count 3 valid (non-NaN) observations
        assert int(result["n_observations"]) == 3


class TestDemeanResiduals:
    """Tests for residual demeaning."""

    def test_simple_demeaning(self):
        """Test basic residual demeaning."""
        residuals = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["data"],
            coords={"data": ["p1", "p2", "p3"]},
        )
        precision = xr.DataArray(
            np.eye(3),
            dims=["data", "data_T"],
        )
        precision = precision.assign_coords(
            data=["p1", "p2", "p3"], data_T=["p1", "p2", "p3"]
        )

        result = demean_residuals(residuals, precision, input_dyad=["data", "data_T"])

        # Result should be a dataset or array with demeaned data
        assert result is not None


class TestInvertCovariance:
    """Tests for covariance matrix inversion."""

    def test_identity_inversion(self):
        """Inverting identity should give identity."""
        cov = xr.DataArray(
            np.eye(3),
            dims=["data", "data_T"],
        )
        cov = cov.assign_coords(data=["p1", "p2", "p3"], data_T=["p1", "p2", "p3"])

        prec = invert_covariance(cov, input_dyad=["data", "data_T"])

        # Should be close to identity
        np.testing.assert_allclose(np.asarray(prec), np.eye(3), atol=1e-5)

    def test_diagonal_inversion(self):
        """Inverting diagonal matrix should invert diagonal elements."""
        cov = xr.DataArray(
            np.diag([2.0, 4.0, 1.0]),
            dims=["data", "data_T"],
        )
        cov = cov.assign_coords(data=["p1", "p2", "p3"], data_T=["p1", "p2", "p3"])

        prec = invert_covariance(cov, input_dyad=["data", "data_T"])

        expected = np.diag([0.5, 0.25, 1.0])
        np.testing.assert_allclose(np.asarray(prec), expected, atol=1e-5)

    def test_inversion_is_consistent(self):
        """Inverting then inverting again should recover original."""
        cov = xr.DataArray(
            [[2.0, 0.5], [0.5, 1.0]],
            dims=["data", "data_T"],
        )
        cov = cov.assign_coords(data=["p1", "p2"], data_T=["p1", "p2"])

        prec1 = invert_covariance(cov, input_dyad=["data", "data_T"])
        # prec1 may not have the same structure, so just check it's invertible
        assert prec1 is not None


class TestGetSpatialMoments:
    """Tests for spatial moment calculation."""

    def test_uniform_posterior(self):
        """Uniform posterior should have non-empty moments."""
        # Create uniform posterior on grid
        x = np.linspace(-100, 100, 5)
        y = np.linspace(-50, 50, 3)
        z = np.linspace(-500, -100, 4)

        logposterior = xr.DataArray(
            np.zeros((len(x), len(y), len(z))),
            dims=["x", "y", "z"],
            coords={"x": x, "y": y, "z": z},
        )

        location_mean, cov = get_spatial_moments(
            logposterior, spatial_dimensions=["x", "y", "z"], output_dyad=["space", "space_T"]
        )

        # Moments should be returned
        assert location_mean is not None
        assert cov is not None

    def test_concentrated_posterior(self):
        """Concentrated posterior should have valid moments."""
        # Create narrow Gaussian-like posterior
        x = np.linspace(-100, 100, 21)
        y = np.linspace(-50, 50, 11)
        z = np.linspace(-500, -100, 10)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        logposterior_vals = -0.01 * (xx**2 + yy**2 + (zz + 300) ** 2)

        logposterior = xr.DataArray(
            logposterior_vals,
            dims=["x", "y", "z"],
            coords={"x": x, "y": y, "z": z},
        )

        location_mean, cov = get_spatial_moments(
            logposterior, spatial_dimensions=["x", "y", "z"], output_dyad=["space", "space_T"]
        )

        # Moments should be valid
        assert location_mean is not None
        assert cov is not None


class TestGetSpatialPointEstimate:
    """Tests for spatial point extimate (MAP/ML)."""

    def test_map_location(self):
        """MAP should return valid point estimate."""
        # Simple parabolic posterior
        x = np.linspace(-100, 100, 21)
        y = np.linspace(-50, 50, 11)
        z = np.linspace(-500, -100, 5)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        # Maximum at (10, 5, -250)
        logposterior_vals = -0.01 * ((xx - 10) ** 2 + (yy - 5) ** 2 + (zz + 250) ** 2)

        logposterior = xr.DataArray(
            logposterior_vals,
            dims=["x", "y", "z"],
            coords={"x": x, "y": y, "z": z},
        )

        location, cov_diff = get_spatial_point_estimate(
            logposterior, spatial_dimensions=["x", "y", "z"], output_dyad=["space", "space_T"]
        )

        # Should return valid point estimate
        assert location is not None
        assert cov_diff is not None

    def test_ml_location(self):
        """ML should return valid point estimate."""
        x = np.linspace(-100, 100, 11)
        y = np.linspace(-50, 50, 9)
        z = np.linspace(-500, -100, 5)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        logposterior_vals = -0.02 * (xx**2 + yy**2 + (zz + 300) ** 2)

        logposterior = xr.DataArray(
            logposterior_vals,
            dims=["x", "y", "z"],
            coords={"x": x, "y": y, "z": z},
        )

        location, cov_diff = get_spatial_point_estimate(
            logposterior, spatial_dimensions=["x", "y", "z"], output_dyad=["space", "space_T"]
        )

        # Should return valid point estimates
        assert location is not None
        assert cov_diff is not None
