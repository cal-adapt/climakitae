"""
Test suite for climakitae/explore/typical_meteorological_year.py

Includes tests for the more general functions along with the TMY class.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.optimize import OptimizeWarning

from climakitae.core.constants import UNSET
from climakitae.explore.typical_meteorological_year import (
    TMY,
    _compute_cdf,
    _get_cdf_by_mon_and_sim,
    _get_cdf_by_sim,
    compute_weighted_fs,
    compute_weighted_fs_sum,
    fs_statistic,
    get_cdf,
    get_cdf_monthly,
    get_top_months,
    is_HadISD,
    match_str_to_wl,
    remove_pinatubo_years,
)


class TestFunctionsForTMY:
    """Test the general functions that are not part of the TMY class."""

    def test_match_str_to_wl(self):
        """Check the string returned for multiple warming levels."""
        test_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 2.4]
        expected = [
            "present-day",
            "near-future",
            "mid-century",
            "mid-late-century",
            "late-century",
            "warming-level-2.4",
        ]
        for test_val, exp_val in zip(test_levels, expected):
            assert match_str_to_wl(test_val) == exp_val

    def test_is_HadISD(self):
        """Check whether station is correctly ID'd as HadISD station."""
        assert is_HadISD("San Diego Lindbergh Field (KSAN)")
        assert not is_HadISD("San Diego")

    def test__compute_cdf(self):
        """Test cdf function applied to single array."""
        # Create test data array
        test_data = np.arange(0, 365 * 3, 1)
        test_da = xr.DataArray(
            data=test_data,
            coords={
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = _compute_cdf(test_da)
        assert result.shape == (2, 1023)
        # Max bin is max value
        assert result[0].max() == pytest.approx(test_data.max(), abs=1e-6)
        assert result[1][-1] == pytest.approx(1.0, abs=1e-6)  # Max probability 1

    def test_get_cdf_monthly(self):
        """Test CDF monthly calculation."""
        test_data = np.arange(0, 365 * 3 * 24, 1)
        test_data = np.expand_dims(test_data, [1]) * np.ones((1, 2))
        coords = {
            "time": pd.date_range(
                start="2001-01-01-00", end="2003-12-31-23", freq="1h"
            ),
            "simulation": ["sim1", "sim2"],
        }
        da = xr.DataArray(
            name="Daily max air temperature",
            dims=["time", "simulation"],
            data=test_data,
            coords=coords,
        ).to_dataset()
        result = get_cdf_monthly(da)
        assert isinstance(result, xr.Dataset)
        # Was cdf applied over simulation and months?
        for dim in ["year", "month", "bin_number"]:
            assert dim in result.dims

    def test__get_cdf_by_sim(self):
        """Test cdf computation by simulation."""
        # Create test data array
        test_data = np.arange(0, 365 * 3, 1)
        test_data = test_data * np.ones((2, len(test_data)))
        test_da = xr.DataArray(
            data=test_data,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = _get_cdf_by_sim(test_da)

        # Correct shape
        assert result.shape == (2, 2, 1023)

        # Max of first simulation matches
        assert result[0][0].max() == pytest.approx(
            test_da.isel(simulation=0).max(), abs=1e-6
        )

        # Max of second simulation matches
        assert result[1][0].max() == pytest.approx(
            test_da.isel(simulation=1).max(), abs=1e-6
        )

        # Simulation list contains all sims
        assert (result.simulation == test_da.simulation).all()

    def test__get_cdf_by_mon_and_sim(self):
        """Test cdf calculation by month and simulation."""
        # Create test data array
        test_data = np.arange(0, 365 * 3, 1)
        test_data = test_data * np.ones((2, len(test_data)))
        test_da = xr.DataArray(
            data=test_data,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = _get_cdf_by_mon_and_sim(test_da)

        # Result contains all months
        assert (result.month == np.arange(1, 13)).all()

        # Simulation list contains all sims
        assert (result.simulation == test_da.simulation).all()

        # Shape correct
        assert result.shape == (2, 12, 2, 1023)

        # Spot check the January max matches
        assert result[1][0][0].max() == pytest.approx(
            test_da.isel({"simulation": 1}).groupby("time.month").max()[0], abs=1e-6
        )

    def test_get_cdf(self):
        """Test full cdf workflow with dataset."""
        # Create test dataset
        test_data = np.arange(0, 365 * 3, 1)
        test_data = test_data * np.ones((2, len(test_data)))
        test_ds = xr.DataArray(
            name="temperature",
            data=test_data,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        ).to_dataset()
        test_ds["wind speed"] = (["simulation", "time"], test_data)
        result = get_cdf(test_ds)

        assert "temperature" in result
        assert "wind speed" in result
        for coord in ["data", "simulation", "month"]:
            assert coord in result.coords

        assert result.data[0] == "bins"
        assert result.data[1] == "probability"

        # Spot check the July max matches
        assert result["temperature"].isel(simulation=0, month=6)[
            0
        ].max() == pytest.approx(
            test_ds["temperature"]
            .isel({"simulation": 1})
            .groupby("time.month")
            .max()[6],
            abs=1e-6,
        )

    def test_fs_statistic(self):
        """Test F-S statistic computation on cdf data."""
        test_data = np.arange(0, 365 * 3, 1)
        test_data = test_data * np.ones((2, len(test_data)))
        test_ds = xr.DataArray(
            name="temperature",
            data=test_data,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        ).to_dataset()
        result = get_cdf(test_ds)

        # Since datasets are identical, fs should be zero
        fs = fs_statistic(result, result)
        assert (fs["temperature"] == 0).all()

        test_data2 = np.ones((365 * 3))
        test_data2 = test_data2 * np.ones((2, len(test_data2)))
        test_ds2 = xr.DataArray(
            name="temperature",
            data=test_data2,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        ).to_dataset()
        result2 = get_cdf(test_ds2)

        # Should have non-zero differences now
        fs = fs_statistic(result, result2)
        assert (fs["temperature"] != 0).any()

    def test_compute_weighted_fs(self):
        """Test weighing of F-S statistic."""
        test_data = np.array([20])
        test_ds = xr.DataArray(
            name="Daily max air temperature", data=test_data
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
            test_ds[item] = test_data
        fs = compute_weighted_fs(test_ds)

        # Check that results are correctly weighted
        values_list = [1, 1, 2, 1, 1, 2, 1, 1, 5, 5]
        for variable, value in zip(vars_list, values_list):
            assert fs[variable] == value

    def test_compute_weighted_fs_sum(self):
        """Check format and values of weighted F-S statistic sum."""
        # Fake cdf climatology data
        coords = {
            "data": ["bins", "probability"],
            "simulation": ["sim1", "sim2"],
            "month": list(range(1, 13)),
        }
        dims = {"data": 2, "simulation": 2, "month": 12, "bin_number": 10}
        probs = np.linspace(0.01, 1, 10)
        bins = np.array(range(1, 11))
        data = np.vstack((bins, probs))
        data = np.expand_dims(data, [1, 2]) * np.ones((1, 2, 12, 1))
        ds_clim = xr.DataArray(
            name="Daily max air temperature",
            data=data,
            dims=dims,
            coords=coords,
        ).to_dataset()

        # Fake cdf monthly data
        coords = {
            "data": ["bins", "probability"],
            "simulation": ["sim1", "sim2"],
            "month": list(range(1, 13)),
            "year": list(range(2001, 2004)),
        }
        dims = {"data": 2, "simulation": 2, "month": 12, "year": 3, "bin_number": 10}
        probs = np.linspace(0.05, 1, 10)
        bins = np.array(range(1, 11))
        data2 = np.vstack((bins, probs))
        data2 = np.expand_dims(data2, [1, 2, 3]) * np.ones((1, 2, 12, 3, 1))
        ds_month = xr.DataArray(
            name="Daily max air temperature",
            data=data2,
            dims=dims,
            coords=coords,
        ).to_dataset()

        # Populate required variables
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
            ds_clim[item] = (("data", "simulation", "month", "bin_number"), data)
            ds_month[item] = (
                ("data", "simulation", "month", "year", "bin_number"),
                data2,
            )

        # Get weighted F-S statistics
        result = compute_weighted_fs_sum(ds_clim, ds_month)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (2, 12, 3)
        assert "month" in result.dims
        # Spot check a row of values
        test = result.sel(simulation="sim1", month=1).data
        expected = np.array([0.00645161, 0.00645161, 0.00645161])
        assert np.allclose(test, expected, atol=1e-9)

    def test_get_top_months(self):
        """Check top months dataframe format and that month with lowest f-s value is chosen."""
        coords = {
            "simulation": ["sim1", "sim2"],
            "month": list(range(1, 13)),
            "year": list(range(2001, 2004)),
        }
        dims = {"simulation": 2, "month": 12, "year": 3}
        sim1 = np.linspace(0.05, 1, 12)
        sim2 = np.linspace(0.3, 1, 12)
        data = np.vstack((sim1, sim2))
        data = np.expand_dims(data, [2]) + np.ones((1, 12, 3)) * np.array(
            [0, 0.02, 0.04]
        )
        fs = xr.DataArray(
            name="Daily max air temperature",
            data=data,
            dims=dims,
            coords=coords,
        )
        result = get_top_months(fs)
        # Correctly formatted dataframe
        for col in ["month", "simulation", "year"]:
            assert col in result.columns
        assert (np.unique(result["simulation"]) == np.array(["sim1", "sim2"])).all()
        # Lowest stat value is in 2001 for all sims, months
        assert (result.year.values == [2001 for x in range(0, 24)]).all()

    def test_get_top_months_skip_last(self):
        """Check get_top_months excludes the final month as an option only when
        the skip_last flag is set to True."""
        coords = {
            "simulation": ["sim1", "sim2"],
            "month": list(range(1, 13)),
            "year": list(range(2001, 2004)),
        }
        dims = {"simulation": 2, "month": 12, "year": 3}
        sim1 = np.linspace(0.05, 1, 12)
        sim2 = np.linspace(0.3, 1, 12)
        data = np.vstack((sim1, sim2))
        data = np.expand_dims(data, [2]) + np.ones((1, 12, 3)) * np.array(
            [0, 0.02, 0.04]
        )
        fs = xr.DataArray(
            name="Daily max air temperature",
            data=data,
            dims=dims,
            coords=coords,
        )
        # Set last year/month to lowest stat value to be best match
        fs[:, -1, -1] = np.zeros((2,))
        result = get_top_months(fs)
        # Default is no skipping - so final year should get chosen for December
        assert (result.loc[result["month"] == 12]["year"] == [2003, 2003]).all()

        result = get_top_months(fs, skip_last=True)
        # Default is no skipping - so final year should get chosen for December
        assert (result.loc[result["month"] == 12]["year"] == [2001, 2001]).all()

    def test_remove_pinatubo_years(self):
        """Check that years immediately after eruption are removed from dataset."""
        test_data = np.arange(0, 10, 1)
        test_data = test_data * np.ones((1, len(test_data)))
        test_ds = xr.DataArray(
            name="temperature",
            data=test_data,
            coords={
                "simulation": ["sim1"],
                "year": range(1991, 2001),
            },
        ).to_dataset()
        result = remove_pinatubo_years(test_ds)
        # Check Pinatubo years gone
        for year in range(1991, 1995):
            assert year not in result.year
        # Check other years still present
        for year in range(1995, 2001):
            assert year in result.year


@pytest.fixture
def mock_t_hourly() -> xr.DataArray:
    """Fixture hourly data array."""
    test_data = np.arange(0, 365 * 3 * 24, 1)
    test_data = np.expand_dims(test_data, [1, 2])
    coords = {
        "x": 7.819e05,
        "y": -4.116e06,
        "lakemask": 0,
        "landmask": 0,
        "Lambert_Conformal": 0,
        "time": pd.date_range(start="2001-01-01-00", end="2003-12-31-23", freq="1h"),
        "scenario": ["Historical + SSP 3-7.0"],
        "simulation": ["WRF_EC-Earth3_r1i1p1f1"],
    }
    da = xr.DataArray(
        name="Air Temperature at 2m",
        dims=["time", "scenario", "simulation"],
        data=test_data,
        coords=coords,
    )
    da.attrs = {
        "variable_id": "t2",
        "extended_description": "Temperature of the air 2m above Earth's surface.",
        "units": "degC",
        "data_type": "Gridded",
        "resolution": "9 km",
        "frequency": "hourly",
        "location_subset": ["coordinate selection"],
        "approach": "Time",
        "downscaling_method": "Dynamical",
        "institution": "UCLA",
        "grid_mapping": "Lambert_Conformal",
        "timezone": "America/Los_Angeles",
    }
    yield da


def mock_t_ds() -> xr.Dataset:
    """Fake hourly dataset that can be manipulated to set up tests."""
    test_data = np.arange(0, 365 * 3 * 24, 1)
    test_data = np.expand_dims(test_data, [1, 2])
    coords = {
        "x": 7.819e05,
        "y": -4.116e06,
        "lakemask": 0,
        "landmask": 0,
        "Lambert_Conformal": 0,
        "time": pd.date_range(start="2001-01-01-00", end="2003-12-31-23", freq="1h"),
        "scenario": ["Historical + SSP 3-7.0"],
        "simulation": ["WRF_EC-Earth3_r1i1p1f1"],
    }
    da = xr.DataArray(
        name="Air Temperature at 2m",
        dims=["time", "scenario", "simulation"],
        data=test_data,
        coords=coords,
    )
    da.attrs = {
        "variable_id": "t2",
        "extended_description": "Temperature of the air 2m above Earth's surface.",
        "units": "degC",
        "data_type": "Gridded",
        "resolution": "9 km",
        "frequency": "hourly",
        "location_subset": ["coordinate selection"],
        "approach": "Time",
        "downscaling_method": "Dynamical",
        "institution": "UCLA",
        "grid_mapping": "Lambert_Conformal",
        "timezone": "America/Los_Angeles",
    }
    return da.to_dataset()


@pytest.mark.advanced
class TestTMYClass:
    """Test the TMY class with fake data."""

    @pytest.mark.integration
    def test_init_with_station(self):
        """Check class initialization with station."""
        # Use valid station name
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 1990
        end_year = 2020
        # Initialize TMY object
        tmy = TMY(start_year=start_year, end_year=end_year, station_name=stn_name)
        assert tmy.start_year == start_year
        assert tmy.end_year == end_year
        assert tmy.stn_name == stn_name
        # Check that we pull correct station coordinates
        assert tmy.lat_range == pytest.approx((33.57975, 33.77975), abs=1e-6)
        assert tmy.lon_range == pytest.approx((-117.967459, -117.76746), abs=1e-6)
        assert tmy.stn_state == "CA"
        assert tmy.stn_code == 72297793184
        assert tmy.verbose

        # Use invalid station name
        stn_name = "KSNA"
        with pytest.raises(ValueError):
            tmy = TMY(start_year=start_year, end_year=end_year, station_name=stn_name)

        # Don't provide any loc data:
        with pytest.raises(
            ValueError,
            match="No valid station name or latitude and longitude provided.",
        ):
            tmy = TMY(start_year, end_year)

    @pytest.mark.integration
    def test_init_with_coords(self):
        """Check class initialization with coordinates."""
        # Use valid station name
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        # Initialize TMY object
        tmy = TMY(start_year=start_year, end_year=end_year, latitude=lat, longitude=lon)
        assert tmy.lat_range == pytest.approx((33.46, 33.66), abs=1e-6)
        assert tmy.lon_range == pytest.approx((-117.91, -117.71), abs=1e-6)
        assert tmy.stn_code == "None"

    @pytest.mark.integration
    def test_init_with_custom_name(self):
        """Check class initialization with coordinates."""
        # Use valid station name
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        station_name = "custom_station"
        # Initialize TMY object
        tmy = TMY(
            start_year=start_year,
            end_year=end_year,
            latitude=lat,
            longitude=lon,
            station_name=station_name,
        )
        assert tmy.lat_range == pytest.approx((33.46, 33.66), abs=1e-6)
        assert tmy.lon_range == pytest.approx((-117.91, -117.71), abs=1e-6)
        assert tmy.stn_name == station_name
        assert tmy.warming_level is UNSET

        # not allowed to use HadISD station name with lat/lon
        with pytest.raises(
            ValueError,
            match="Do not set `latitude` and `longitude` when using a HadISD station for `station_name`. Change `station_name` value if using custom location.",
        ):
            station_name = "Santa Ana John Wayne Airport (KSNA)"
            tmy = TMY(
                start_year=start_year,
                end_year=end_year,
                latitude=lat,
                longitude=lon,
                station_name=station_name,
            )

    @pytest.mark.integration
    def test_init_with_warming_level(self):
        """Check class initialization with coordinates."""
        # Use valid station name
        lat = 33.56
        lon = -117.81
        warming_level = 2.0
        # Initialize TMY object
        tmy = TMY(warming_level=warming_level, latitude=lat, longitude=lon)
        assert tmy.warming_level == warming_level
        assert tmy.start_year is UNSET
        assert tmy.end_year is UNSET

        # Can't use years with GWL
        with pytest.raises(
            ValueError,
            match="Variables `start_year` and `end_year` cannot be paired with `warming_level`. Set either `start_year` and `end_year` OR `warming_level.",
        ):
            tmy = TMY(
                start_year=2000,
                end_year=2020,
                warming_level=warming_level,
                latitude=lat,
                longitude=lon,
            )

    @pytest.mark.integration
    def test__load_single_variable_time(self):
        """Load data for a single variable."""
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        # Initialize TMY object
        tmy = TMY(start_year=start_year, end_year=end_year, latitude=lat, longitude=lon)
        # Actually going to load data for a single point and check results
        result = tmy._load_single_variable("Air Temperature at 2m", "degC")
        assert isinstance(result, xr.DataArray)
        assert result.name == "Air Temperature at 2m"
        assert result.attrs["units"] == "degC"
        assert result.lat.shape == ()
        assert result.lon.shape == ()
        assert result.lat.data == 33.55938
        assert result.lon.data == -117.80269
        assert (result.simulation.values == tmy.simulations).all()

    def test__load_single_variable_warming_level(self):
        """Load data for a single variable."""
        lat = 33.56
        lon = -117.81
        warming_level = 2.0
        # Initialize TMY object
        tmy = TMY(warming_level=warming_level, latitude=lat, longitude=lon)
        # Actually going to load data for a single point and check results
        result = tmy._load_single_variable("Air Temperature at 2m", "degC")
        assert isinstance(result, xr.DataArray)
        assert result.name == "Air Temperature at 2m"
        assert result.attrs["units"] == "degC"
        assert result.lat.shape == ()
        assert result.lon.shape == ()
        assert result.lat.data == 33.55938
        assert result.lon.data == -117.80269
        assert (result.warming_level.values == [2.0]).all()
        simulations = [s + "_historical+ssp370" for s in tmy.simulations]
        assert (result.simulation.values == simulations).all()

    def test__get_tmy_variable(self, mock_t_hourly):
        """Check that daily stat gets returned and values match expected."""
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        # Initialize TMY object
        tmy = TMY(start_year=start_year, end_year=end_year, latitude=lat, longitude=lon)
        # Actually going to load data for a single point and check results
        with patch.object(tmy, "_load_single_variable", return_value=mock_t_hourly):
            result = tmy._get_tmy_variable(
                "Air Temperature at 2m", "degC", ["max", "min", "mean", "sum"]
            )
            assert isinstance(result, list)
            # Check attributes of first result
            assert result[0].name == "Air Temperature at 2m"
            assert result[0].attrs["units"] == "degC"
            assert (result[0].simulation.values == mock_t_hourly.simulation).all()
            assert result[0].attrs["frequency"] == "daily"
            # Check all stats match
            assert result[0].equals(mock_t_hourly.resample(time="1D").max())
            assert result[1].equals(mock_t_hourly.resample(time="1D").min())
            assert result[2].equals(mock_t_hourly.resample(time="1D").mean())
            assert result[3].equals(mock_t_hourly.resample(time="1D").sum())

    def test_load_all_variables(self, mock_t_hourly):
        """Check that data load gets called and results merged."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        # Initialize TMY object
        varlist = [
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
        tmy = TMY(start_year, end_year, station_name=stn_name)
        with patch.object(tmy, "_load_single_variable", return_value=mock_t_hourly):
            tmy.load_all_variables()
            assert isinstance(tmy.all_vars, xr.Dataset)
            for varname in varlist:
                assert varname in tmy.all_vars

    def test_set_cdf_climatology(self):
        """Check that data load and get_cdf get called."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        # Initialize TMY object
        tmy = TMY(start_year, end_year, station_name=stn_name)
        with (
            patch.object(tmy, "load_all_variables") as mock_load,
            patch(
                "climakitae.explore.typical_meteorological_year.get_cdf",
                return_value=xr.Dataset(),
            ) as mock_get_cdf,
        ):
            tmy.set_cdf_climatology()
            # Check correct methods called
            mock_load.assert_called_once()
            mock_get_cdf.assert_called_once()
            assert tmy.cdf_climatology is not UNSET

    @patch("climakitae.explore.typical_meteorological_year.get_cdf_monthly")
    @patch("climakitae.explore.typical_meteorological_year.remove_pinatubo_years")
    def test_cdf_monthly(self, mock_get_cdf_monthly, mock_remove_pinatubo):
        """Check that data load and get_cdf_monthly get called."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        tmy = TMY(start_year=start_year, end_year=end_year, station_name=stn_name)
        with patch.object(tmy, "load_all_variables") as mock_load:
            tmy.set_cdf_monthly()
            # Check correct methods called
            mock_load.assert_called_once()
            mock_get_cdf_monthly.assert_called_once()
            mock_remove_pinatubo.assert_called_once()
            assert tmy.cdf_monthly is not UNSET

    def test_generate_tmy(self):
        """Test that all steps called in full workflow."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        tmy = TMY(start_year=start_year, end_year=end_year, station_name=stn_name)
        with (
            patch.object(tmy, "load_all_variables") as mock_load,
            patch.object(tmy, "get_candidate_months") as mock_get_months,
            patch.object(tmy, "run_tmy_analysis") as mock_run_tmy,
            patch.object(tmy, "export_tmy_data") as mock_export,
        ):
            tmy.generate_tmy()
            # Check correct methods called
            mock_load.assert_called_once()
            mock_get_months.assert_called_once()
            mock_run_tmy.assert_called_once()
            mock_export.assert_called_once()

    def test_get_candidate_months(self):
        """Test the TMY workflow calls up to set_top_months."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        # Initialize TMY object
        tmy = TMY(start_year=start_year, end_year=end_year, station_name=stn_name)
        with (
            patch.object(tmy, "set_cdf_monthly") as mock_month,
            patch.object(tmy, "set_cdf_climatology") as mock_clim,
            patch.object(tmy, "set_weighted_statistic") as mock_weight,
            patch.object(tmy, "set_top_months") as mock_top_months,
        ):
            tmy.get_candidate_months()
            # Check correct methods called
            mock_clim.assert_called_once()
            mock_month.assert_called_once()
            mock_weight.assert_called_once()
            mock_top_months.assert_called_once()

    def test__make_8760_tables(self):
        """Check that dataframe of 8760 values returned."""
        data = {
            "month": list(range(1, 13)),
            "simulation": ["WRF_EC-Earth3_r1i1p1f1" for x in range(0, 12)],
            "year": [2001 for x in range(0, 12)],
        }
        df = pd.DataFrame.from_dict(data)
        all_vars_ds = mock_t_ds()

        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        tmy = TMY(start_year=start_year, end_year=end_year, station_name=stn_name)
        result = tmy._make_8760_tables(all_vars_ds, df)
        # Check result dict of dataframes (only 1 for 1 simulation in test)
        assert list(result.keys()) == list(all_vars_ds.simulation.values)
        assert (
            result["WRF_EC-Earth3_r1i1p1f1"].columns
            == [
                "time",
                "scenario",
                "simulation",
                "x",
                "y",
                "lakemask",
                "landmask",
                "Lambert_Conformal",
                "Air Temperature at 2m",
            ]
        ).all()
        assert len(result["WRF_EC-Earth3_r1i1p1f1"].index) == 8760

    def test__smooth_month_transition_hours(self):
        """Check that smoothed data returned for variables in list."""
        variable_list = [
            "Air Temperature at 2m",
            "Dew point temperature",
            "Wind speed at 10m",
            "Wind direction at 10m",
            "Surface Pressure",
            "Water Vapor Mixing Ratio at 2m",
            "Relative humidity",
        ]
        data = {
            "time": pd.date_range("2000-01-01-00", "2000-03-31-23", freq="1h"),
            "simulation": ["WRF_EC-Earth3_r1i1p1f1" for x in range(0, 2184)],
        }
        # Add data with a transition from 0 to 1 at midnight on
        # Jan 31/Feb 1 to be smoothed
        for varname in variable_list:
            data[varname] = np.zeros(len(data["time"]))
            data[varname][31 * 24 : 32 * 24] = 1
        df = pd.DataFrame.from_dict(data)

        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        tmy = TMY(start_year, end_year, station_name=stn_name)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=OptimizeWarning)
            result = tmy._smooth_month_transition_hours(df.copy())

        # Check that data was altered in the smoothing window for all listed variables
        for varname in variable_list:
            assert not pytest.approx(df[varname][738], 1e-6) == pytest.approx(
                result[varname][738], 1e-6
            )
            assert not pytest.approx(df[varname][746], 1e-6) == pytest.approx(
                result[varname][746], 1e-6
            )
        # Check that values outside the window were not changed
        for varname in variable_list:
            assert pytest.approx(df[varname][780], 1e-6) == pytest.approx(
                result[varname][780], 1e-6
            )
            assert pytest.approx(df[varname][720], 1e-6) == pytest.approx(
                result[varname][720], 1e-6
            )

    @patch("climakitae.explore.typical_meteorological_year.get_top_months")
    def test_set_top_months(self, mock_top_months):
        """Check that set_top_months calls correct functions."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        # Initialize TMY object
        tmy = TMY(start_year=start_year, end_year=end_year, station_name=stn_name)
        with patch.object(tmy, "set_weighted_statistic") as mock_fs:
            tmy.set_top_months()
            # Check correct methods called
            mock_fs.assert_called_once()
            mock_top_months.assert_called_once()
