"""Pytest configuration for preseis.source_location tests."""

import pytest


@pytest.fixture
def simple_1d_grid():
    """Create a simple 1D grid for basic testing."""
    import numpy as np
    import xarray as xr

    x = np.linspace(0, 10000, 21)
    y = np.array([0.0])
    z = np.array([0.0])
    return xr.Dataset(coords={"x": x, "y": y, "z": z})


@pytest.fixture
def uniform_velocity():
    """Create uniform velocity field."""
    import xarray as xr

    # Returns a function that creates velocity grid for any grid
    def _create_velocity(grid, vp=5000.0, vs=3000.0):
        import numpy as np

        # Create velocity arrays matching grid shape
        shape = (len(grid.x), len(grid.y), len(grid.z))
        vp_field = np.full(shape, vp)
        vs_field = np.full(shape, vs)

        # Stack into xarray with mode dimension
        vp_da = xr.DataArray(
            vp_field,
            dims=["x", "y", "z"],
            coords={"x": grid.x, "y": grid.y, "z": grid.z},
        )
        vs_da = xr.DataArray(
            vs_field,
            dims=["x", "y", "z"],
            coords={"x": grid.x, "y": grid.y, "z": grid.z},
        )

        vinst = xr.concat([vp_da, vs_da], dim="mode").assign_coords(mode=["P", "S"])
        return xr.Dataset({"Vinst": vinst})

    return _create_velocity
