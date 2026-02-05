"""Tests for covariance structure utilities."""

import numpy as np
import pytest
import xarray as xr

from preseis.source_location import (
    build_covariance_structure,
    build_mode_correlation_matrix,
    build_spatial_correlation_matrix,
    extract_ellipse_parameters,
    spatial_correlation_coefficient,
    weighted_rms_residual,
)


class TestSpatialCorrelationCoefficient:
    """Tests for spatial correlation coefficient."""

    def test_same_location(self):
        """Correlation at same location should be 1.0."""
        loc = np.array([1000.0, 2000.0, -500.0])
        corr = spatial_correlation_coefficient(loc, loc, corr_len=1000.0)
        assert np.isclose(corr, 1.0)

    def test_far_locations(self):
        """Correlation at far distances should be small."""
        loc0 = np.array([0.0, 0.0, 0.0])
        loc1 = np.array([10000.0, 10000.0, 0.0])
        corr = spatial_correlation_coefficient(loc0, loc1, corr_len=1000.0)
        assert corr < 0.1
        assert corr > 0

    def test_correlation_length_scale(self):
        """Correlation should decrease with distance normalized by length scale."""
        loc0 = np.array([0.0, 0.0, 0.0])
        loc1 = np.array([1000.0, 0.0, 0.0])

        # Larger correlation length -> higher correlation
        corr_long = spatial_correlation_coefficient(loc0, loc1, corr_len=5000.0)
        corr_short = spatial_correlation_coefficient(loc0, loc1, corr_len=500.0)
        assert corr_long > corr_short

    def test_symmetry(self):
        """Correlation should be symmetric in locations."""
        loc0 = np.array([100.0, 200.0, 300.0])
        loc1 = np.array([500.0, 600.0, 700.0])
        corr01 = spatial_correlation_coefficient(loc0, loc1, corr_len=1000.0)
        corr10 = spatial_correlation_coefficient(loc1, loc0, corr_len=1000.0)
        assert np.isclose(corr01, corr10)

    def test_output_range(self):
        """Correlation should always be in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            loc0 = np.random.randn(3) * 10000
            loc1 = np.random.randn(3) * 10000
            corr = spatial_correlation_coefficient(
                loc0, loc1, corr_len=np.random.uniform(100, 5000)
            )
            assert 0 <= corr <= 1


class TestBuildSpatialCorrelationMatrix:
    """Tests for spatial correlation matrix construction."""

    def test_diagonal_is_one(self):
        """Diagonal elements (self-correlation) should be 1.0."""
        stations = xr.DataArray(
            [[1000.0, 2000.0, -500.0], [3000.0, 4000.0, -1000.0]],
            dims=["station", "location"],
            coords={"station": ["STA1", "STA2"], "location": ["x", "y", "z"]},
        )
        corr_mat = build_spatial_correlation_matrix(stations, correlation_length=1000.0)

        # Check diagonal
        assert corr_mat.size == 4
        assert np.isclose(float(corr_mat.sel(station="STA1", station_T="STA1")), 1.0)
        assert np.isclose(float(corr_mat.sel(station="STA2", station_T="STA2")), 1.0)

    def test_symmetry(self):
        """Correlation matrix should be symmetric."""
        stations = xr.DataArray(
            [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0], [2000.0, 0.0, 0.0]],
            dims=["station", "location"],
            coords={
                "station": ["STA1", "STA2", "STA3"],
                "location": ["x", "y", "z"],
            },
        )
        corr_mat = build_spatial_correlation_matrix(stations, correlation_length=1000.0)

        # Check symmetry
        for i in range(len(corr_mat.station)):
            for j in range(len(corr_mat.station_T)):
                s1, s2 = corr_mat.station.values[i], corr_mat.station_T.values[j]
                val_ij = float(corr_mat.sel(station=s1, station_T=s2))
                val_ji = float(corr_mat.sel(station=s2, station_T=s1))
                assert np.isclose(val_ij, val_ji)

    def test_all_values_in_range(self):
        """All correlation values should be in [0, 1]."""
        stations = xr.DataArray(
            [[i * 1000, j * 1000, -500.0] for i in range(3) for j in range(3)],
            dims=["station", "location"],
            coords={"location": ["x", "y", "z"]},
        )
        corr_mat = build_spatial_correlation_matrix(stations, correlation_length=2000.0)

        assert (corr_mat >= 0).all()
        assert (corr_mat <= 1).all()


class TestBuildModeCorrelationMatrix:
    """Tests for mode correlation matrix construction."""

    def test_two_modes(self):
        """Test P-S mode correlation structure."""
        modes = ["P", "S"]
        corr_mat = build_mode_correlation_matrix(modes, correlation_coefficient=0.2)

        # Diagonal should be 1
        assert np.isclose(float(corr_mat.sel(mode="P", mode_T="P")), 1.0)
        assert np.isclose(float(corr_mat.sel(mode="S", mode_T="S")), 1.0)

        # Off-diagonal should be correlation coefficient
        assert np.isclose(float(corr_mat.sel(mode="P", mode_T="S")), 0.2)
        assert np.isclose(float(corr_mat.sel(mode="S", mode_T="P")), 0.2)

    def test_three_modes(self):
        """Test with three modes."""
        modes = ["P", "S1", "S2"]
        corr_mat = build_mode_correlation_matrix(modes, correlation_coefficient=0.3)

        # All diagonal should be 1
        for mode in modes:
            assert np.isclose(float(corr_mat.sel(mode=mode, mode_T=mode)), 1.0)

        # All off-diagonal should be 0.3
        for m1 in modes:
            for m2 in modes:
                if m1 != m2:
                    val = float(corr_mat.sel(mode=m1, mode_T=m2))
                    assert np.isclose(val, 0.3)

    def test_symmetry(self):
        """Mode correlation matrix should be symmetric."""
        modes = ["P", "S", "T"]
        corr_mat = build_mode_correlation_matrix(modes, correlation_coefficient=0.15)

        for m1 in modes:
            for m2 in modes:
                val1 = float(corr_mat.sel(mode=m1, mode_T=m2))
                val2 = float(corr_mat.sel(mode=m2, mode_T=m1))
                assert np.isclose(val1, val2)

    def test_positive_definite(self):
        """Mode correlation matrix should be positive definite."""
        modes = ["P", "S"]
        corr_mat = build_mode_correlation_matrix(modes, correlation_coefficient=0.2)

        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(corr_mat.values)
        assert (eigenvalues > 0).all()


class TestBuildCovarianceStructure:
    """Tests for complete covariance structure."""

    def test_output_shape(self):
        """Test output dimensions and shape."""
        sigma = xr.DataArray(
            [0.5, 0.8], dims=["mode"], coords={"mode": ["P", "S"]}
        )

        stations = xr.DataArray(
            [[0.0, 0.0, 0.0], [1000.0, 1000.0, 0.0]],
            dims=["station", "location"],
            coords={"station": ["STA1", "STA2"], "location": ["x", "y", "z"]},
        )
        spatial_corr = build_spatial_correlation_matrix(
            stations, correlation_length=1000.0
        )

        mode_corr = build_mode_correlation_matrix(["P", "S"], correlation_coefficient=0.2)

        cov = build_covariance_structure(sigma, spatial_corr, mode_corr)

        # Check dimensions exist
        assert "mode" in cov.dims or hasattr(cov, "dims")
        assert cov is not None
        assert len(cov.shape) >= 2

    def test_diagonal_values(self):
        """Diagonal should contain variance values."""
        sigma = xr.DataArray(
            [1.0, 2.0], dims=["mode"], coords={"mode": ["P", "S"]}
        )

        stations = xr.DataArray(
            [[0.0, 0.0, 0.0]],
            dims=["station", "location"],
            coords={"station": ["STA1"], "location": ["x", "y", "z"]},
        )
        spatial_corr = build_spatial_correlation_matrix(
            stations, correlation_length=1000.0
        )

        mode_corr = build_mode_correlation_matrix(["P", "S"], correlation_coefficient=0.0)

        cov = build_covariance_structure(sigma, spatial_corr, mode_corr)

        # Diagonal elements should be variances (sigma[i]^2)
        assert np.isclose(
            float(cov.sel(mode="P", mode_T="P", station="STA1", station_T="STA1")),
            1.0,
        )
        assert np.isclose(
            float(cov.sel(mode="S", mode_T="S", station="STA1", station_T="STA1")),
            4.0,
        )


class TestWeightedRmsResidual:
    """Tests for weighted RMS residual calculation."""

    def test_zero_residuals(self):
        """Zero residuals should give zero RMS."""
        obs_at = xr.DataArray(
            [[1.0, 1.5], [1.2, 1.7]],
            dims=["mode", "station"],
            coords={"mode": ["P", "S"], "station": ["STA1", "STA2"]},
        )
        synth_at = obs_at.copy()
        sigma = xr.DataArray([1.0, 1.0], dims=["mode"], coords={"mode": ["P", "S"]})

        rms = weighted_rms_residual(obs_at, synth_at, sigma, time_shift=0.0)
        assert np.isclose(rms, 0.0)

    def test_time_shift(self):
        """Time shift should affect residuals."""
        obs_at = xr.DataArray(
            [[1.0, 1.5], [1.2, 1.7]],
            dims=["mode", "station"],
            coords={"mode": ["P", "S"], "station": ["STA1", "STA2"]},
        )
        synth_at = xr.DataArray(
            [[0.5, 1.0], [0.7, 1.2]],
            dims=["mode", "station"],
            coords={"mode": ["P", "S"], "station": ["STA1", "STA2"]},
        )
        sigma = xr.DataArray([1.0, 1.0], dims=["mode"], coords={"mode": ["P", "S"]})

        # Without time shift
        rms_no_shift = weighted_rms_residual(obs_at, synth_at, sigma, time_shift=0.0)

        # With time shift that corrects most of residual
        rms_with_shift = weighted_rms_residual(obs_at, synth_at, sigma, time_shift=0.5)

        assert rms_with_shift < rms_no_shift

    def test_uncertainty_weighting(self):
        """Larger uncertainties should give smaller weighted residuals."""
        obs_at = xr.DataArray(
            [[1.5]],
            dims=["mode", "station"],
            coords={"mode": ["P"], "station": ["STA1"]},
        )
        synth_at = xr.DataArray(
            [[1.0]],
            dims=["mode", "station"],
            coords={"mode": ["P"], "station": ["STA1"]},
        )

        # Small uncertainty
        sigma_small = xr.DataArray([0.5], dims=["mode"], coords={"mode": ["P"]})
        rms_small_sigma = weighted_rms_residual(
            obs_at, synth_at, sigma_small, time_shift=0.0
        )

        # Large uncertainty
        sigma_large = xr.DataArray([5.0], dims=["mode"], coords={"mode": ["P"]})
        rms_large_sigma = weighted_rms_residual(
            obs_at, synth_at, sigma_large, time_shift=0.0
        )

        assert rms_small_sigma > rms_large_sigma


class TestExtractEllipseParameters:
    """Tests for ellipse parameter extraction from covariance."""

    def test_diagonal_covariance(self):
        """Diagonal covariance should give simple ellipse parameters."""
        cov = np.array([[4.0, 0, 0], [0, 1.0, 0], [0, 0, 9.0]])

        params = extract_ellipse_parameters(cov)

        # Standard deviations
        assert np.isclose(params["sigma_x"], 2.0)
        assert np.isclose(params["sigma_y"], 1.0)
        assert np.isclose(params["sigma_z"], 3.0)

        # Semi-major and semi-minor (from eigenvalues)
        assert np.isclose(params["semi_major"], 2.0)
        assert np.isclose(params["semi_minor"], 1.0)

        # Azimuth should be 0 or 90 for diagonal
        assert params["azimuth"] in [0, 90, 180, 270]

    def test_rms_horizontal(self):
        """RMS horizontal should be reasonable."""
        cov = np.array([[4.0, 0, 0], [0, 1.0, 0], [0, 0, 9.0]])

        params = extract_ellipse_parameters(cov)

        # RMS horizontal = sqrt((sigma_x^2 + sigma_y^2)/2)
        expected_rms = np.sqrt((4.0 + 1.0) / 2)
        assert np.isclose(params["rms_horizontal"], expected_rms)

    def test_off_diagonal_covariance(self):
        """Off-diagonal correlation should affect ellipse orientation."""
        # Strong positive correlation in x-y
        cov = np.array([[4.0, 3.0, 0], [3.0, 4.0, 0], [0, 0, 1.0]])

        params = extract_ellipse_parameters(cov)

        # Diagonal should still be recoverable
        assert np.isclose(params["sigma_x"], 2.0)
        assert np.isclose(params["sigma_y"], 2.0)

        # Semi-major and semi-minor should differ from diagonal values
        assert params["semi_major"] > 2.0
        assert params["semi_minor"] < 1.5

    def test_non_finite_covariance(self):
        """Non-finite values should raise error."""
        cov_nan = np.array([[np.nan, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        cov_inf = np.array([[np.inf, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

        with pytest.raises(ValueError, match="non-finite values"):
            extract_ellipse_parameters(cov_nan)

        with pytest.raises(ValueError, match="non-finite values"):
            extract_ellipse_parameters(cov_inf)

    def test_azimuth_range(self):
        """Azimuth should be in [0, 360)."""
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = np.radians(angle)
            # Rotation matrix
            rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            # Ellipse with semi-major 2 and semi-minor 1
            cov_eig = np.diag([4.0, 1.0])
            cov_xy = rot @ cov_eig @ rot.T
            cov = np.eye(3)
            cov[:2, :2] = cov_xy

            params = extract_ellipse_parameters(cov)
            assert 0 <= params["azimuth"] < 360
