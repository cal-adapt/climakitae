from climakitae.explore.typical_meteorological_year import * # TODO replace
import pytest
import xarray as xr
import numpy as np
import pandas as pd
import panel

def test_compute_cdf(self):
    """Test cdf function applied to single array."""
    # Create test data array
    test_data = np.arange(0,365*3,1)
    test = xr.DataArray(
        data=test_data,
        coords={
            "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
        })
    result = compute_cdf(test)
    assert result.shape == (2,1023)
    # Max bin is max value
    assert result[0].max() == pytest.approx(test_data.max(), abs=0.4)
    assert result[1][-1] == 1. # Max probability 1

def test_get_cdf_by_sim(self):
    """Test cdf computation by simulation."""
    # Create test data array
    test_data = np.arange(0,365*3,1)
    test_data = test_data * np.ones((2,len(test_data)))
    test = xr.DataArray(
        data=test_data,
        coords={
            "simulation": ["sim1","sim2"],
            "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
        })
    result = get_cdf_by_sim(test)
    
    # Correct shape
    assert result.shape == (2, 2, 1023)

    # Max of first simulation matches
    assert result[0][0].max() == pytest.approx(test_data.isel(simulation=0).max(), abs=0.4)

    # Max of second simulation matches
    assert result[1][0].max() == pytest.approx(test_data.isel(simulation=1).max(), abs=0.4)

    # Simulation list contains all sims
    assert (result.simulation == test.simulation).all()

def test_get_cdf_by_mon_and_sim(self):
    """Test cdf calculation by month and simulation."""
    # Create test data array
    test_data = np.arange(0,365*3,1)
    test_data = test_data * np.ones((2,len(test_data)))
    test = xr.DataArray(
        data=test_data,
        coords={
            "simulation": ["sim1","sim2"],
            "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
        })
    result = get_cdf_by_mon_and_sim(test)

    # Result contains all months
    assert (result.month == np.arange(1,13)).all()

    # Simulation list contains all sims
    assert (result.simulation == test.simulation).all()

    # Shape correct
    assert result.shape == (2, 12, 2, 1023)

    # Spot check the January max matches
    assert result[1][0][0].max() == pytest.approx(test.isel({"simulation":1}).groupby("time.month").max()[0], abs=0.4)

def test_get_cdf(self):
    """Test full cdf workflow with dataset."""
    # Create test dataset
    test_data = np.arange(0,365*3,1)
    test_data = test_data * np.ones((2,len(test_data)))
    test = xr.DataArray(
        name = "temperature",
        data=test_data,
        coords={
            "simulation": ["sim1","sim2"],
            "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
        }).to_dataset()
    test["wind speed"] = (["simulation","time"],test_data)
    result = get_cdf(test)

    assert "temperature" in result
    assert "wind speed" in result
    for coord in ["data","simulation","month"]:
        assert coord in result.coords

    assert result.data[0] == "bins"
    assert result.data[1] == "probability"

    # Spot check the July max matches
    assert result["temperature"].isel(simulation=0,month=6)[0].max() == pytest.approx(test["temperature"].isel({"simulation":1}).groupby("time.month").max()[6], abs=0.4)

def test_fs_statistic(self):
    """Test f-s statistic computation on cdf data."""
    test_data = np.arange(0,365*3,1)
    test_data = test_data * np.ones((2,len(test_data)))
    test = xr.DataArray(
        name = "temperature",
        data=test_data,
        coords={
            "simulation": ["sim1","sim2"],
            "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
        }).to_dataset()
    result = get_cdf(test)

    # Since datasets are identical, fs should be zero
    fs = fs_statistic(result,result)
    assert (fs["temperature"] == 0).all()

    test_data2 = np.ones((365*3))
    test_data2 = test_data2 * np.ones((2,len(test_data2)))
    test2 = xr.DataArray(
        name = "temperature",
        data=test_data2,
        coords={
            "simulation": ["sim1","sim2"],
            "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
        }).to_dataset()
    result2 = get_cdf(test2)

    # Should have non-zero differences now
    fs = fs_statistic(result,result2)
    assert (fs["temperature"] != 0).any()

def test_compute_weighted_fs(self):
    """Test weighing of f-s statistic."""
    test_data = np.array([20])
    test = xr.DataArray(
        name = "Daily max air temperature",
        data = test_data
    ).to_dataset()
    vars_list = [
        "Daily max air temperature",
        "Daily min air temperature",
        "Daily mean air temperature",
        "Daily max dewpoint temperature",
        "Daily min dewpoint temperature",
        "Daily mean dewpoint temperature",
        "Daily max wind speed",
        "Daily mean wind speed",
        "Global horizontal irradiance",
        "Direct normal irradiance",
    ]
    for item in vars_list[1:]:
        test[item] = test_data
    fs = compute_weighted_fs(test)

    # Check that results are correctly weighted
    values_list = [1,1,2,1,1,2,1,1,5,5]
    for variable,value in zip(vars_list,values_list):
        assert fs[variable] == value

def test_plot_one_var_cdf(self):
    """Test that plot runs and returns object."""
    # Create test dataset
    test_data = np.arange(0,365*3,1)
    test_data = test_data * np.ones((2,len(test_data)))
    test = xr.DataArray(
        name = "temperature",
        data=test_data,
        coords={
            "simulation": ["sim1","sim2"],
            "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
        }).to_dataset()
    test["temperature"].attrs["units"] = "C"
    result = get_cdf(test)

    test_plot = plot_one_var_cdf(result,"temperature")

    assert isinstance(test_plot,panel.layout.base.Column)