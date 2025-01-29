import pytest
import xarray as xr
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from climakitae.explore.warming import WarmingLevels, WarmingLevelChoose, get_sliced_data, clean_warm_data, clean_list, _drop_invalid_sims

@pytest.fixture
def mock_warming_levels():
    return WarmingLevels()

@pytest.fixture
def mock_catalog_data():
    time = pd.date_range("2000-01-01", periods=12, freq="MS")
    all_sims = ["sim1", "sim2"]
    data = np.random.rand(12, 2)
    da = xr.DataArray(data, coords={"time": time, "all_sims": all_sims}, dims=["time", "all_sims"]),
    return da

@pytest.fixture
def mock_gwl_times():
    return pd.DataFrame({"1.5": [2025, 2030]}, index=[("sim1", "r1i1p1f1", "ssp585"), ("sim2", "r1i1p1f1", "ssp370")])

# Test Initialization
def test_warming_levels_init(mock_warming_levels):
    assert isinstance(mock_warming_levels.wl_params, WarmingLevelChoose)
    assert isinstance(mock_warming_levels.sliced_data, dict)
    assert isinstance(mock_warming_levels.gwl_snapshots, xr.DataArray)
    assert mock_warming_levels.wl_params.window == 15

# Test find_warming_slice with edge cases
def test_find_warming_slice(mock_warming_levels, mock_catalog_data, mock_gwl_times):
    mock_warming_levels.catalog_data = mock_catalog_data
    mock_warming_levels.wl_params.months = np.arange(1, 13)
    mock_warming_levels.wl_params.window = 15
    mock_warming_levels.wl_params.anom = "No"
    
    level = "1.5"
    result = mock_warming_levels.find_warming_slice(level, mock_gwl_times)
    assert isinstance(result, xr.DataArray)
    assert "all_sims" in result.dims
    assert result.shape[0] > 0

# Test calculate
def test_calculate(mock_warming_levels):
    mock_warming_levels.wl_params.retrieve = MagicMock(return_value=xr.DataArray(np.random.rand(10, 2)))
    mock_warming_levels.calculate()
    assert isinstance(mock_warming_levels.catalog_data, xr.DataArray)
    assert isinstance(mock_warming_levels.sliced_data, dict)
    assert len(mock_warming_levels.sliced_data) > 0

# Test clean_warm_data with null values
def test_clean_warm_data():
    time = pd.date_range("2000-01-01", periods=12, freq="MS")
    data = np.random.rand(12, 2)
    data[0, 0] = np.nan  # Insert NaN
    all_sims = ["sim1", "sim2"]
    warm_data = xr.DataArray(data, coords={"time": time, "all_sims": all_sims}, dims=["time", "all_sims"])
    
    result = clean_warm_data(warm_data)
    assert isinstance(result, xr.DataArray)
    assert not np.all(np.isnan(result.values))

# Test clean_list handles missing simulations
def test_clean_list(mock_catalog_data, mock_gwl_times):
    modified_data = mock_catalog_data.sel(all_sims=["sim1"])  # Sim2 is missing
    result = clean_list(modified_data, mock_gwl_times)
    assert isinstance(result, xr.DataArray)
    assert "all_sims" in result.dims
    assert "sim2" not in result.all_sims.values

# Test get_sliced_data with invalid warming level
def test_get_sliced_data_invalid_level(mock_catalog_data, mock_gwl_times):
    level = "2.0"  # Not present in mock_gwl_times
    result = get_sliced_data(mock_catalog_data.isel(all_sims=0), level, mock_gwl_times)
    assert isinstance(result, xr.DataArray)
    assert np.isnan(result.values).all()

# Test _drop_invalid_sims removes incorrect simulations
def test_drop_invalid_sims():
    mock_ds = xr.Dataset({"var": ("all_sims", [1, 2, 3])}, coords={"all_sims": ["sim1", "sim2", "sim3"]})
    mock_selections = MagicMock()
    mock_selections.scenario_ssp = ["ssp585"]
    result = _drop_invalid_sims(mock_ds, mock_selections)
    assert isinstance(result, xr.Dataset)
    assert "all_sims" in result.dims
    assert len(result.all_sims) <= len(mock_ds.all_sims)
