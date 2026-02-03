"""Tests for travel time computation."""

import numpy as np
import pytest
import xarray as xr

from preseis.source_location import compute_traveltimes


class TestComputeTraveltimes:
    """Tests for compute_traveltimes function."""

    @pytest.fixture
    def simple_grid(self):
        """Create a simple computational grid."""
        x = np.linspace(0, 10000, 11)  # 11 points, 1km spacing
        y = np.linspace(0, 10000, 11)
        z = np.linspace(-3000, 0, 4)  # 4 points, 1km spacing
        return xr.Dataset(coords={"x": x, "y": y, "z": z})

    @pytest.fixture
    def velocity_grid(self, simple_grid):
        """Create a velocity grid with constant velocities."""
        # Create constant velocity fields: Vp=5000 m/s, Vs=3000 m/s
        vp = xr.full_like(simple_grid.x, 5000.0).expand_dims(
            {"y": simple_grid.y, "z": simple_grid.z}
        )
        vs = xr.full_like(simple_grid.x, 3000.0).expand_dims(
            {"y": simple_grid.y, "z": simple_grid.z}
        )

        # Stack P and S modes
        vinst = xr.concat([vp, vs], dim="mode").assign_coords(mode=["P", "S"])

        return xr.Dataset({"Vinst": vinst})

    @pytest.fixture
    def stationlist(self):
        """Create a simple station list."""
        stations = xr.DataArray(
            [[5000.0, 5000.0, 0.0]],  # Single station at center, surface
            dims=["station", "location"],
            coords={"station": ["TEST.STA1"], "location": ["x", "y", "z"]},
        )
        return stations

    def test_compute_traveltimes_shape(self, stationlist, velocity_grid, simple_grid):
        """Test that output has correct shape."""
        tt = compute_traveltimes(stationlist, velocity_grid, simple_grid)

        # Check dimensions
        assert tt.dims == ("station", "mode", "x", "y", "z")
        assert len(tt.station) == 1
        assert len(tt.mode) == 2  # P and S
        assert len(tt.x) == len(simple_grid.x)
        assert len(tt.y) == len(simple_grid.y)
        assert len(tt.z) == len(simple_grid.z)

    def test_compute_traveltimes_values(self, stationlist, velocity_grid, simple_grid):
        """Test that travel times are physically reasonable."""
        tt = compute_traveltimes(stationlist, velocity_grid, simple_grid)

        # Travel times should be positive
        assert (tt >= 0).all()

        # P waves should be faster (shorter times) than S waves
        tt_p = tt.sel(mode="P")
        tt_s = tt.sel(mode="S")
        # At most locations P should be faster (allowing for numerical edge effects)
        assert (tt_p <= tt_s).sum() > 0.9 * tt_p.size

    def test_compute_traveltimes_at_station(
        self, stationlist, velocity_grid, simple_grid
    ):
        """Test that travel time is small near station location."""
        tt = compute_traveltimes(stationlist, velocity_grid, simple_grid)

        # Get travel time at station location (5000, 5000, 0)
        tt_at_station = tt.sel(
            station="TEST.STA1", x=5000, y=5000, z=0, method="nearest"
        )

        # Get travel times at corners (far from station)
        tt_corner = tt.sel(station="TEST.STA1", x=0, y=0, z=0, method="nearest")

        # Travel time at station should be less than at corner
        assert float(tt_at_station.sel(mode="P")) < float(tt_corner.sel(mode="P"))
        assert float(tt_at_station.sel(mode="S")) < float(tt_corner.sel(mode="S"))

    def test_compute_traveltimes_multiple_stations(self, velocity_grid, simple_grid):
        """Test with multiple stations."""
        # Create multiple stations
        stations = xr.DataArray(
            [
                [2000.0, 2000.0, 0.0],  # Station 1
                [8000.0, 8000.0, 0.0],  # Station 2
            ],
            dims=["station", "location"],
            coords={
                "station": ["TEST.STA1", "TEST.STA2"],
                "location": ["x", "y", "z"],
            },
        )

        tt = compute_traveltimes(stations, velocity_grid, simple_grid)

        # Check shape
        assert len(tt.station) == 2
        assert len(tt.mode) == 2

        # Travel times should be different for different stations
        tt_sta1 = tt.sel(station="TEST.STA1")
        tt_sta2 = tt.sel(station="TEST.STA2")
        assert not np.allclose(tt_sta1.values, tt_sta2.values)

    def test_compute_traveltimes_preserves_coords(
        self, stationlist, velocity_grid, simple_grid
    ):
        """Test that coordinates are preserved."""
        tt = compute_traveltimes(stationlist, velocity_grid, simple_grid)

        # Check that spatial coordinates match input grid
        assert np.allclose(tt.x.values, simple_grid.x.values)
        assert np.allclose(tt.y.values, simple_grid.y.values)
        assert np.allclose(tt.z.values, simple_grid.z.values)

        # Check that mode coordinate exists
        assert "mode" in tt.coords
        assert list(tt.mode.values) == ["P", "S"]
