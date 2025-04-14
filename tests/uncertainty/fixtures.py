from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

from climakitae.core.data_interface import DataParameters


@pytest.fixture
def mock_multi_ens_dataset():
    """Create a mock dataset with simulation and member_id attributes"""
    ds = xr.Dataset(
        data_vars={"tas": (["time"], [1.0, 2.0, 3.0, 4.0, 5.0])},
        coords={
            "time": pd.date_range("2020-01-01", periods=5),
            "simulation": "EC-Earth3",
            "member_id": "r1i1p1f1",
        },
    )
    return ds


@pytest.fixture
def mock_data_for_warm_level():
    """Create a mock dataset with simulation attribute"""
    import numpy as np
    import xarray as xr

    # Create a simple dataset with simulation coordinate
    ds = xr.Dataset(
        data_vars={"tas": (["time"], np.random.rand(10))},
        coords={"time": pd.date_range("2020-01-01", periods=10)},
    )
    ds = ds.assign_coords({"simulation": "CESM2"})
    return ds


@pytest.fixture
def wrf_dataset():
    """Fixture to create a mock WRF dataset."""
    selections = DataParameters()
    selections.area_average = "No"
    selections.area_subset = "states"
    selections.cached_area = ["CA"]
    selections.downscaling_method = "Dynamical"
    selections.scenario_historical = ["Historical Climate"]
    selections.scenario_ssp = ["SSP 3-7.0"]
    selections.append_historical = True
    selections.variable = "Precipitation (total)"
    selections.time_slice = (1981, 2100)
    selections.resolution = "9 km"
    selections.timescale = "monthly"
    wrf_ds = selections.retrieve().squeeze()

    # WRF simulation names have additional information about activity ID and ensemble
    # Lookup dictionary used to rename simulation values
    wrf_cmip_lookup_dict = {
        "WRF_EC-Earth3-Veg_r1i1p1f1": "EC-Earth3-Veg",
        "WRF_EC-Earth3_r1i1p1f1": "EC-Earth3",
        "WRF_CESM2_r11i1p1f1": "CESM2",
        "WRF_CNRM-ESM2-1_r1i1p1f2": "CNRM-ESM2-1",
        "WRF_FGOALS-g3_r1i1p1f1": "FGOALS-g3",
        "WRF_MIROC6_r1i1p1f1": "MIROC6",
        "WRF_TaiESM1_r1i1p1f1": "TaiESM1",
        "WRF_MPI-ESM1-2-HR_r3i1p1f1": "MPI-ESM1-2-HR",
    }

    wrf_ds = wrf_ds.sortby("simulation")  # Sort simulations alphabetically
    wrf_ds["simulation"] = [
        wrf_cmip_lookup_dict[sim] for sim in wrf_ds.simulation.values
    ]  # Rename simulations
    wrf_ds = wrf_ds.clip(0.1)  # Remove values less than 0.1

    return wrf_ds


@pytest.fixture
def mock_data_for_clipping():
    """Create a mock dataset for testing clipping."""
    # Create a custom mock object instead of a real xarray Dataset
    mock_ds = MagicMock()

    # Add rio attribute with required methods
    mock_rio = MagicMock()
    mock_rio.write_crs.return_value = mock_ds
    mock_rio.clip.return_value = mock_ds
    mock_ds.rio = mock_rio

    return mock_ds


@pytest.fixture
def mock_geoms_for_clipping():
    """Create mock geometries for states and counties."""
    # Create a simple polygon for testing
    polygon = Polygon([(-120, 35), (-120, 36), (-119, 36), (-119, 35)])

    # Create mock GeoDataFrames
    states = gpd.GeoDataFrame(
        {"NAME": ["California", "Nevada"]},
        geometry=[polygon, Polygon()],
    )
    counties = gpd.GeoDataFrame(
        {"NAME": ["Los Angeles", "San Francisco"]},
        geometry=[polygon, Polygon()],
    )
    return {"states": states, "counties": counties}


@pytest.fixture
def mock_data_for_standardization():
    """Create a simple mock dataset with required attributes and structure."""
    # Create sample data
    time = pd.date_range("2020-01-01", periods=3)
    lats = np.array([0, 1])
    lons = np.array([0, 1])
    data = np.random.rand(3, 2, 2)

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            "tas": (["time", "lat", "lon"], data),
            "height": ([], 0),
        },
        coords={
            "time": time,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "source_id": "MODEL1",
            "experiment_id": "historical",
            "frequency": "mon",
        },
    )
    return ds


@pytest.fixture
def simple_dataset():
    """Create a simple 2x2 dataset with latitude values at 0° and 60°N."""
    ds = xr.Dataset(
        data_vars={"tas": (["y", "x"], np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "y": np.array([0.0, 60.0]),  # Equator and 60°N
            "x": np.array([0.0, 10.0]),
        },
    )
    return ds


@pytest.fixture
def multi_var_dataset():
    """Create a dataset with multiple variables for testing."""
    ds = xr.Dataset(
        data_vars={
            "tas": (["y", "x"], np.array([[1.0, 2.0], [3.0, 4.0]])),
            "pr": (["y", "x"], np.array([[5.0, 6.0], [7.0, 8.0]])),
        },
        coords={
            "y": np.array([0.0, 60.0]),
            "x": np.array([0.0, 10.0]),
        },
    )
    return ds


@pytest.fixture
def global_dataset():
    """Create a global dataset with latitudes from -90° to 90°."""
    lats = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    lons = np.array([0.0, 90.0, 180.0, 270.0])
    data = np.ones((len(lats), len(lons)))  # All values are 1.0

    ds = xr.Dataset(
        data_vars={"tas": (["y", "x"], data)},
        coords={
            "y": lats,
            "x": lons,
        },
    )
    return ds


@pytest.fixture
def mock_catalog():
    """Create a mock catalog for testing."""
    # Create a mock catalog
    mock_catalog = MagicMock()
    # Mock the search method to return a mock subset catalog
    mock_subset = MagicMock()
    mock_catalog.search.return_value = mock_subset

    # Create mock dataset dictionary with two simple datasets
    ds1 = xr.Dataset(
        data_vars={"tas": (["time"], [1.0, 2.0])},
        coords={"time": [0, 1]},
        attrs={"source_id": "MODEL1", "experiment_id": "historical"},
    )
    ds2 = xr.Dataset(
        data_vars={"tas": (["time"], [3.0, 4.0])},
        coords={"time": [0, 1]},
        attrs={"source_id": "MODEL2", "experiment_id": "historical"},
    )
    mock_data_dict = {"dataset1": ds1, "dataset2": ds2}

    # Mock the to_dataset_dict method to return our mock data
    mock_subset.to_dataset_dict.return_value = mock_data_dict

    return mock_catalog
