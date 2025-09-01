from unittest.mock import patch

import numpy as np
import pandas as pd
import panel
import pytest
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.explore.typical_meteorological_year import (
    TMY,
    compute_cdf,
    compute_weighted_fs,
    fs_statistic,
    get_cdf,
    get_cdf_by_mon_and_sim,
    get_cdf_by_sim,
    get_cdf_monthly,
    remove_pinatubo_years,
)


class TestFunctionsForTMY:
    def test_compute_cdf(self):
        """Test cdf function applied to single array."""
        # Create test data array
        test_data = np.arange(0, 365 * 3, 1)
        test = xr.DataArray(
            data=test_data,
            coords={
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = compute_cdf(test)
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
            name="Air Temperature at 2m",
            dims=["time", "simulation"],
            data=test_data,
            coords=coords,
        ).to_dataset()
        result = get_cdf_monthly(da)
        assert isinstance(result, xr.Dataset)
        # Was cdf applied over simulation and months?
        for dim in ["year", "month", "bin_number"]:
            assert dim in result.dims

    def test_get_cdf_by_sim(self):
        """Test cdf computation by simulation."""
        # Create test data array
        test_data = np.arange(0, 365 * 3, 1)
        test_data = test_data * np.ones((2, len(test_data)))
        test = xr.DataArray(
            data=test_data,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = get_cdf_by_sim(test)

        # Correct shape
        assert result.shape == (2, 2, 1023)

        # Max of first simulation matches
        assert result[0][0].max() == pytest.approx(
            test.isel(simulation=0).max(), abs=1e-6
        )

        # Max of second simulation matches
        assert result[1][0].max() == pytest.approx(
            test.isel(simulation=1).max(), abs=1e-6
        )

        # Simulation list contains all sims
        assert (result.simulation == test.simulation).all()

    def test_get_cdf_by_mon_and_sim(self):
        """Test cdf calculation by month and simulation."""
        # Create test data array
        test_data = np.arange(0, 365 * 3, 1)
        test_data = test_data * np.ones((2, len(test_data)))
        test = xr.DataArray(
            data=test_data,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = get_cdf_by_mon_and_sim(test)

        # Result contains all months
        assert (result.month == np.arange(1, 13)).all()

        # Simulation list contains all sims
        assert (result.simulation == test.simulation).all()

        # Shape correct
        assert result.shape == (2, 12, 2, 1023)

        # Spot check the January max matches
        assert result[1][0][0].max() == pytest.approx(
            test.isel({"simulation": 1}).groupby("time.month").max()[0], abs=1e-6
        )

    def test_get_cdf(self):
        """Test full cdf workflow with dataset."""
        # Create test dataset
        test_data = np.arange(0, 365 * 3, 1)
        test_data = test_data * np.ones((2, len(test_data)))
        test = xr.DataArray(
            name="temperature",
            data=test_data,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        ).to_dataset()
        test["wind speed"] = (["simulation", "time"], test_data)
        result = get_cdf(test)

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
            test["temperature"].isel({"simulation": 1}).groupby("time.month").max()[6],
            abs=1e-6,
        )

    def test_fs_statistic(self):
        """Test f-s statistic computation on cdf data."""
        test_data = np.arange(0, 365 * 3, 1)
        test_data = test_data * np.ones((2, len(test_data)))
        test = xr.DataArray(
            name="temperature",
            data=test_data,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        ).to_dataset()
        result = get_cdf(test)

        # Since datasets are identical, fs should be zero
        fs = fs_statistic(result, result)
        assert (fs["temperature"] == 0).all()

        test_data2 = np.ones((365 * 3))
        test_data2 = test_data2 * np.ones((2, len(test_data2)))
        test2 = xr.DataArray(
            name="temperature",
            data=test_data2,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        ).to_dataset()
        result2 = get_cdf(test2)

        # Should have non-zero differences now
        fs = fs_statistic(result, result2)
        assert (fs["temperature"] != 0).any()

    def test_compute_weighted_fs(self):
        """Test weighing of f-s statistic."""
        test_data = np.array([20])
        test = xr.DataArray(
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
            test[item] = test_data
        fs = compute_weighted_fs(test)

        # Check that results are correctly weighted
        values_list = [1, 1, 2, 1, 1, 2, 1, 1, 5, 5]
        for variable, value in zip(vars_list, values_list):
            assert fs[variable] == value

    def test_remove_pinatubo_years(self):
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
def mock_t_hourly():
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
    return da


def mock_t_ds():
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


class TestTMYClass:
    @pytest.fixture
    def mock_da_hourly(self):
        test_data = np.arange(0, 365 * 3 * 24, 1)
        test_data = test_data * np.ones((2, len(test_data)))
        test = xr.DataArray(
            name="temperature",
            data=test_data,
            coords={
                "simulation": ["sim1", "sim2"],
                "time": pd.date_range(
                    start="2001-01-01-00", end="2003-12-31-23", freq="1h"
                ),
            },
        )
        return test

    def test_TMY_init_station(self):
        """Check class initialization with station."""
        # Use valid station name
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 1990
        end_year = 2020
        # Initialize TMY object
        tmy = TMY(start_year, end_year, station_name=stn_name)
        assert tmy.start_year == start_year
        assert tmy.end_year == end_year
        assert tmy.stn_name == stn_name
        assert tmy.latitude == pytest.approx((33.62975, 33.729749999999996), abs=1e-6)
        assert tmy.longitude == pytest.approx(
            (-117.92746, -117.80745999999999), abs=1e-6
        )
        assert tmy.stn_state == "CA"
        assert tmy.stn_code == 72297793184
        assert tmy.verbose == False

        # Use invalid station name
        stn_name = "KSNA"
        with pytest.raises(ValueError):
            tmy = TMY(start_year, end_year, station_name=stn_name)

    def test_TMY_init_coords(self):
        """Check class initialization with coordinates."""
        # Use valid station name
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        # Initialize TMY object
        tmy = TMY(start_year, end_year, latitude=lat, longitude=lon)
        assert tmy.latitude == pytest.approx((33.510000000000005, 33.61), abs=1e-6)
        assert tmy.longitude == pytest.approx((-117.87, -117.75), abs=1e-6)
        assert tmy.stn_code == "None"

    def test__load_single_variable(self):
        """Load data for a single variable."""
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        # Initialize TMY object
        tmy = TMY(start_year, end_year, latitude=lat, longitude=lon)
        # Actually going to load data for a single point and check results
        result = tmy._load_single_variable("Air Temperature at 2m", "degC")
        assert isinstance(result, xr.DataArray)
        assert result.name == "Air Temperature at 2m"
        assert result.attrs["units"] == "degC"
        assert result.lat.shape == ()
        assert result.lon.shape == ()
        assert result.lat.data == 33.544018
        assert result.lon.data == -117.829834
        assert (result.simulation.values == tmy.data_models).all()

    def test__get_tmy_variable(self, mock_t_hourly):
        """Check that daily stat gets returned and values match expected."""
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        # Initialize TMY object
        tmy = TMY(start_year, end_year, latitude=lat, longitude=lon)
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
            # load_all_variables called since we just initialized this TMY object
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
        # Initialize TMY object
        tmy = TMY(start_year, end_year, station_name=stn_name)
        with patch.object(tmy, "load_all_variables") as mock_load:
            tmy.set_cdf_monthly()
            # load_all_variables called since we just initialized this TMY object
            mock_load.assert_called_once()
            mock_get_cdf_monthly.assert_called_once()
            mock_remove_pinatubo.assert_called_once()
            assert tmy.cdf_monthly is not UNSET

    def test_generate_tmy(self):
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        # Initialize TMY object
        tmy = TMY(start_year, end_year, station_name=stn_name)
        with (
            patch.object(tmy, "load_all_variables") as mock_load,
            patch.object(tmy, "get_candidate_months") as mock_get_months,
            patch.object(tmy, "run_tmy_analysis") as mock_run_tmy,
            patch.object(tmy, "export_tmy_data") as mock_export,
        ):
            tmy.generate_tmy()
            mock_load.assert_called_once()
            mock_get_months.assert_called_once()
            mock_run_tmy.assert_called_once()
            mock_export.assert_called_once()

    def test_get_candidate_months(self):
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        # Initialize TMY object
        tmy = TMY(start_year, end_year, station_name=stn_name)
        with (
            patch.object(tmy, "set_cdf_monthly") as mock_month,
            patch.object(tmy, "set_cdf_climatology") as mock_clim,
            patch.object(tmy, "set_weighted_statistic") as mock_weight,
            patch.object(tmy, "calculate_top_df") as mock_top_df,
        ):
            tmy.get_candidate_months()
            mock_clim.assert_called_once()
            mock_month.assert_called_once()
            mock_weight.assert_called_once()
            mock_top_df.assert_called_once()

    def test__make_tables(self):
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
        # Initialize TMY object
        tmy = TMY(start_year, end_year, station_name=stn_name)
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

    def test_calculate_top_df(self):
        # self.weighted_fs xarray.DataArray (simulation: 4, year: 27, month: 12)
        pass
