"""
Test suite for climakitae/explore/shock_extreme_meteorological_year.py

Includes tests for the more general functions along with the shock_XMY class.
Exclude tests for functions from typical_meteorological_year.py that are used in shock_extreme_meteorological_year.py
"""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.optimize import OptimizeWarning

from climakitae.core.constants import UNSET
from climakitae.explore.shock_extreme_meteorological_year import (
    shock_XMY,
    generate_candidate_months,
    find_hot_cold_extreme_from_median,
)


class TestFunctionsForXMY:
    """Test the general functions that are not part of the shock_XMY class."""

    def test_get_top_months_cold(self):
        """Check top months dataframe format and that month with lowest f-s value is chosen for cold shock XMY."""

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
        extreme = "cold"
        result = generate_candidate_months(extreme, fs)
        # Correctly formatted dataframe
        for col in ["month", "simulation", "year"]:
            assert col in result.columns
        assert (np.unique(result["simulation"]) == np.array(["sim1", "sim2"])).all()
        # Lowest stat value is in 2001 for all sims, months
        assert (result.year.values == [2001 for x in range(0, 24)]).all()

    def test_get_top_months_hot(self):
        """Check top months dataframe format and that month with highest f-s value is chosen for hot shock XMY."""

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
        extreme = "hot"
        result = shock_get_top_months(extreme, fs)
        # Correctly formatted dataframe
        for col in ["month", "simulation", "year"]:
            assert col in result.columns
        assert (np.unique(result["simulation"]) == np.array(["sim1", "sim2"])).all()
        # Highest stat value is in 2001 for all sims, months
        assert (result.year.values == [2003 for x in range(0, 24)]).all()

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
        extreme = "cold"
        # Set last year/month to lowest stat value to be best match
        fs[:, -1, -1] = np.zeros((2,))
        result = shock_get_top_months(extreme, fs)
        # Default is no skipping - so final year should get chosen for December
        assert (result.loc[result["month"] == 12]["year"] == [2003, 2003]).all()

        result = shock_get_top_months(extreme, fs, skip_last=True)
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


def mock_complete_hourly_ds() -> xr.Dataset:
    """Fake hourly dataset with all TMY variables for run_xmy_analysis tests."""
    n_hours = 365 * 3 * 24
    time = pd.date_range(start="2001-01-01-00", end="2003-12-31-23", freq="1h")
    sims = ["WRF_EC-Earth3_r1i1p1f1"]
    coords = {
        "time": time,
        "simulation": sims,
    }
    dims = ["time", "simulation"]
    ds = xr.Dataset(coords=coords)

    # All variables expected by _smooth_month_transition_hours + RH + mixing ratio
    variables = {
        "Air Temperature at 2m": 20.0,
        "Dew point temperature": 10.0,
        "Relative humidity": 50.0,
        "Wind speed at 10m": 5.0,
        "Wind direction at 10m": 180.0,
        "Surface Pressure": 101325.0,
        "Water Vapor Mixing Ratio at 2m": 5.0,
        "Instantaneous downwelling shortwave flux at bottom": 200.0,
        "Shortwave surface downward direct normal irradiance": 150.0,
        "Shortwave surface downward diffuse irradiance": 50.0,
        "Instantaneous downwelling longwave flux at bottom": 300.0,
    }
    for varname, val in variables.items():
        ds[varname] = (dims, np.full((n_hours, len(sims)), val))
    return ds


@pytest.mark.advanced
class TestXMYClass:
    """Test the shock_XMY class with fake data."""

    @pytest.mark.integration
    def test_init_with_station(self):
        """Check class initialization with station."""
        # Use valid station name
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 1990
        end_year = 2020
        extreme = "hot"
        # Initialize shock_XMY object
        xmy = shock_XMY(
            extreme=extreme,
            start_year=start_year,
            end_year=end_year,
            station_name=stn_name,
        )
        assert xmy.start_year == start_year
        assert xmy.end_year == end_year
        assert xmy.stn_name == stn_name
        # Check that we pull correct station coordinates
        assert xmy.lat_range == pytest.approx((33.57975, 33.77975), abs=1e-6)
        assert xmy.lon_range == pytest.approx((-117.967459, -117.76746), abs=1e-6)
        assert xmy.stn_state == "CA"
        assert xmy.stn_code == 72297793184
        assert xmy.extreme == extreme
        assert xmy.verbose

        # Use invalid station name
        stn_name = "KSNA"
        with pytest.raises(ValueError):
            xmy = shock_XMY(
                extreme=extreme,
                start_year=start_year,
                end_year=end_year,
                station_name=stn_name,
            )

        # Don't provide any loc data:
        with pytest.raises(
            ValueError,
            match="No valid station name or latitude and longitude provided.",
        ):
            xmy = shock_XMY(extreme, start_year, end_year)

    @pytest.mark.integration
    def test_init_with_coords(self):
        """Check class initialization with coordinates."""
        # Use valid station name
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        extreme = "hot"
        # Initialize shock_XMY object
        xmy = shock_XMY(
            extreme=extreme,
            start_year=start_year,
            end_year=end_year,
            latitude=lat,
            longitude=lon,
        )
        assert xmy.lat_range == pytest.approx((33.46, 33.66), abs=1e-6)
        assert xmy.lon_range == pytest.approx((-117.91, -117.71), abs=1e-6)
        assert xmy.stn_code == "None"

    @pytest.mark.integration
    def test_init_with_custom_name(self):
        """Check class initialization with coordinates."""
        # Use valid station name
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        station_name = "custom_station"
        extreme = "hot"
        # Initialize shock_XMY object
        xmy = shock_XMY(
            extreme=extreme,
            start_year=start_year,
            end_year=end_year,
            latitude=lat,
            longitude=lon,
            station_name=station_name,
        )
        assert xmy.lat_range == pytest.approx((33.46, 33.66), abs=1e-6)
        assert xmy.lon_range == pytest.approx((-117.91, -117.71), abs=1e-6)
        assert xmy.stn_name == station_name
        assert xmy.warming_level is UNSET

        # not allowed to use HadISD station name with lat/lon
        with pytest.raises(
            ValueError,
            match="Do not set `latitude` and `longitude` when using a HadISD station for `station_name`. Change `station_name` value if using custom location.",
        ):
            station_name = "Santa Ana John Wayne Airport (KSNA)"
            xmy = shock_XMY(
                extreme=extreme,
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
        extreme = "hot"
        # Initialize shock_XMY object
        xmy = shock_XMY(
            extreme=extreme, warming_level=warming_level, latitude=lat, longitude=lon
        )
        assert xmy.warming_level == warming_level
        assert xmy.start_year is UNSET
        assert xmy.end_year is UNSET

        # Can't use years with GWL
        with pytest.raises(
            ValueError,
            match="Variables `start_year` and `end_year` cannot be paired with `warming_level`. Set either `start_year` and `end_year` OR `warming_level.",
        ):
            xmy = shock_XMY(
                extreme="hot",
                start_year=2000,
                end_year=2020,
                warming_level=warming_level,
                latitude=lat,
                longitude=lon,
            )

    @pytest.mark.integration
    def test__fetch_raw_variable_time(self):
        """Fetch a single variable via _fetch_raw_variable (time mode)."""
        lat = 33.56
        lon = -117.81
        start_year = 1990
        end_year = 2020
        extreme = "hot"
        xmy = shock_XMY(
            extreme=extreme,
            start_year=start_year,
            end_year=end_year,
            latitude=lat,
            longitude=lon,
        )
        result = xmy._fetch_raw_variable("t2", table_id="1hr")
        assert isinstance(result, xr.DataArray)
        assert "time" in result.dims
        assert "simulation" in result.dims
        assert (result.simulation.values == xmy.simulations).all()

    def test__fetch_raw_variable_warming_level(self):
        """Fetch a single variable via _fetch_raw_variable (warming level mode)."""
        lat = 33.56
        lon = -117.81
        warming_level = 2.0
        extreme = "hot"
        xmy = shock_XMY(
            extreme=extreme, warming_level=warming_level, latitude=lat, longitude=lon
        )
        result = xmy._fetch_raw_variable("t2", table_id="1hr")
        assert isinstance(result, xr.DataArray)
        assert "time" in result.dims
        assert "simulation" in result.dims
        # Warming level mode still has centered_year before cleaning
        if "centered_year" in result.coords:
            assert result.centered_year.shape[0] == len(result.simulation)

    def test_load_all_variables(self):
        """Check that load_all_variables creates expected datasets."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        extreme = "hot"
        xmy = shock_XMY(extreme, start_year, end_year, station_name=stn_name)

        # Expected daily variable names in all_vars
        daily_varlist = [
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

        # Create mock DataArrays for _fetch_raw_variable
        time_hourly = pd.date_range("2001-01-01", "2003-12-31 23:00", freq="1h")
        time_daily = pd.date_range("2001-01-01", "2003-12-31", freq="1D")
        sims = xmy.simulations

        # Reasonable mock values per variable to avoid numerical issues
        hourly_values = {
            "t2": 290.0,  # K
            "q2": 0.005,  # kg/kg
            "psfc": 101325.0,  # Pa
            "u10": 3.0,  # m/s
            "v10": 4.0,  # m/s
            "swdnb": 200.0,  # W/m2
            "swddni": 150.0,  # W/m2
            "swddif": 50.0,  # W/m2
            "lwdnb": 300.0,  # W/m2
        }
        daily_values = {
            "t2max": 295.0,
            "t2min": 285.0,
            "t2": 290.0,
            "wspd10max": 8.0,
            "wspd10mean": 4.0,
            "rh": 60.0,
            "sw_dwn": 200.0,
        }

        def mock_fetch(variable_id, table_id="1hr"):
            if table_id == "1hr":
                time = time_hourly
                val = hourly_values.get(variable_id, 1.0)
            else:
                time = time_daily
                val = daily_values.get(variable_id, 1.0)
            data = np.full((len(time), len(sims)), val)
            da = xr.DataArray(
                data=data,
                dims=["time", "simulation"],
                coords={"time": time, "simulation": sims},
            )
            da.name = variable_id
            return da

        with patch.object(xmy, "_fetch_raw_variable", side_effect=mock_fetch):
            xmy.load_all_variables()
            # Daily CDF dataset
            assert isinstance(xmy.all_vars, xr.Dataset)
            for varname in daily_varlist:
                assert varname in xmy.all_vars
            # Hourly 8760 dataset
            assert hasattr(xmy, "_hourly_data")
            assert isinstance(xmy._hourly_data, xr.Dataset)

    def test_load_all_variables_warming_level_captures_centered_year(self):
        """Check that _sim_centered_years is populated in warming level mode."""
        lat = 33.56
        lon = -117.81
        warming_level = 2.0
        extreme = "hot"
        xmy = shock_XMY(
            extreme=extreme, warming_level=warming_level, latitude=lat, longitude=lon
        )

        time_hourly = pd.date_range("2001-01-01", "2003-12-31 23:00", freq="1h")
        time_daily = pd.date_range("2001-01-01", "2003-12-31", freq="1D")
        sims = xmy.simulations

        hourly_values = {
            "t2": 290.0,
            "q2": 0.005,
            "psfc": 101325.0,
            "u10": 3.0,
            "v10": 4.0,
            "swdnb": 200.0,
            "swddni": 150.0,
            "swddif": 50.0,
            "lwdnb": 300.0,
        }
        daily_values = {
            "t2max": 295.0,
            "t2min": 285.0,
            "t2": 290.0,
            "wspd10max": 8.0,
            "wspd10mean": 4.0,
            "rh": 60.0,
            "sw_dwn": 200.0,
        }

        # Track which call is first (for centered_year injection)
        call_count = {"n": 0}

        def mock_fetch(variable_id, table_id="1hr"):
            call_count["n"] += 1
            if table_id == "1hr":
                time = time_hourly
                val = hourly_values.get(variable_id, 1.0)
            else:
                time = time_daily
                val = daily_values.get(variable_id, 1.0)
            data = np.full((len(time), len(sims)), val)
            da = xr.DataArray(
                data=data,
                dims=["time", "simulation"],
                coords={"time": time, "simulation": sims},
            )
            # First hourly fetch returns raw data with centered_year
            if call_count["n"] == 1:
                centered_years = [2040, 2045, 2038, 2042]
                da = da.assign_coords(
                    centered_year=("simulation", centered_years[: len(sims)])
                )
            da.name = variable_id
            return da

        with patch.object(xmy, "_fetch_raw_variable", side_effect=mock_fetch):
            xmy.load_all_variables()
            # _sim_centered_years should be populated
            assert hasattr(xmy, "_sim_centered_years")
            assert isinstance(xmy._sim_centered_years, dict)
            assert len(xmy._sim_centered_years) == len(sims)
            for sim in sims:
                assert sim in xmy._sim_centered_years

    def test_set_cdf_climatology(self):
        """Check that data load and get_cdf get called."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        extreme = "hot"
        # Initialize shock_XMY object
        xmy = shock_XMY(extreme, start_year, end_year, station_name=stn_name)
        with (
            patch.object(xmy, "load_all_variables") as mock_load,
            patch(
                "climakitae.explore.shock_extreme_meteorological_year.get_cdf",
                return_value=xr.Dataset(),
            ) as mock_get_cdf,
        ):
            xmy.set_cdf_climatology()
            # Check correct methods called
            mock_load.assert_called_once()
            mock_get_cdf.assert_called_once()
            assert xmy.cdf_climatology is not UNSET

    @patch("climakitae.explore.shock_extreme_meteorological_year.get_cdf_monthly")
    @patch("climakitae.explore.shock_extreme_meteorological_year.remove_pinatubo_years")
    def test_cdf_monthly(self, mock_get_cdf_monthly, mock_remove_pinatubo):
        """Check that data load and get_cdf_monthly get called."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        extreme = "hot"
        xmy = shock_XMY(
            extreme=extreme,
            start_year=start_year,
            end_year=end_year,
            station_name=stn_name,
        )
        with patch.object(xmy, "load_all_variables") as mock_load:
            xmy.set_cdf_monthly()
            # Check correct methods called
            mock_load.assert_called_once()
            mock_get_cdf_monthly.assert_called_once()
            mock_remove_pinatubo.assert_called_once()
            assert xmy.cdf_monthly is not UNSET

    def test_generate_xmy(self):
        """Test that all steps called in full workflow."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        extreme = "hot"
        xmy = shock_XMY(
            extreme=extreme,
            start_year=start_year,
            end_year=end_year,
            station_name=stn_name,
        )
        with (
            patch.object(xmy, "load_all_variables") as mock_load,
            patch.object(xmy, "get_candidate_months") as mock_get_months,
            patch.object(xmy, "run_xmy_analysis") as mock_run_xmy,
            patch.object(xmy, "export_xmy_data") as mock_export,
        ):
            xmy.generate_xmy()
            # Check correct methods called
            mock_load.assert_called_once()
            mock_get_months.assert_called_once()
            mock_run_xmy.assert_called_once()
            mock_export.assert_called_once()

    def test_get_candidate_months(self):
        """Test the shock_XMY workflow calls up to set_top_months."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        extreme = "hot"
        # Initialize shock_XMY object
        xmy = shock_XMY(
            extreme=extreme,
            start_year=start_year,
            end_year=end_year,
            station_name=stn_name,
        )
        with (
            patch.object(xmy, "set_cdf_monthly") as mock_month,
            patch.object(xmy, "set_cdf_climatology") as mock_clim,
            patch.object(xmy, "set_top_months") as mock_top_months,
        ):
            xmy.get_candidate_months()
            # Check correct methods called
            mock_clim.assert_called_once()
            mock_month.assert_called_once()
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
        extreme = "hot"
        xmy = shock_XMY(
            extreme=extreme,
            start_year=start_year,
            end_year=end_year,
            station_name=stn_name,
        )
        result = xmy._make_8760_tables(all_vars_ds, df)
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
        extreme = "hot"
        xmy = shock_XMY(extreme, start_year, end_year, station_name=stn_name)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=OptimizeWarning)
            result = xmy._smooth_month_transition_hours(df.copy())

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

    @patch("climakitae.explore.shock_extreme_meteorological_year.shock_get_top_months")
    # def test_set_top_months(self, mock_top_months):
    #     """Check that set_top_months calls correct functions."""
    #     stn_name = "Santa Ana John Wayne Airport (KSNA)"
    #     start_year = 2001
    #     end_year = 2003
    #     extreme = "hot"
    #     # Initialize shock_XMY object
    #     xmy = shock_XMY(
    #         extreme=extreme,
    #         start_year=start_year,
    #         end_year=end_year,
    #         station_name=stn_name,
    #     )
    #     with patch.object(xmy, "set_weighted_statistic") as mock_fs:
    #         xmy.set_top_months()
    #         # Check correct methods called
    #         mock_fs.assert_called_once()
    #         mock_top_months.assert_called_once()

    def test_run_xmy_analysis_adds_scenario_column(self):
        """Check that run_xmy_analysis adds 'scenario' column in time mode."""
        stn_name = "Santa Ana John Wayne Airport (KSNA)"
        start_year = 2001
        end_year = 2003
        extreme = "hot"
        xmy = shock_XMY(
            extreme=extreme,
            start_year=start_year,
            end_year=end_year,
            station_name=stn_name,
        )

        # Build a mock _hourly_data and top_months
        sim = "WRF_EC-Earth3_r1i1p1f1"
        hourly_ds = mock_complete_hourly_ds()
        xmy._hourly_data = hourly_ds
        xmy.top_months = pd.DataFrame(
            {
                "month": list(range(1, 13)),
                "simulation": [sim] * 12,
                "year": [2001] * 12,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=OptimizeWarning)
            xmy.run_xmy_analysis()

        # Time mode → scenario column, not warming_level
        assert "scenario" in xmy.xmy_data_to_export[sim].columns
        assert "warming_level" not in xmy.xmy_data_to_export[sim].columns
        assert (xmy.xmy_data_to_export[sim]["scenario"] == "historical+ssp370").all()

    def test_run_xmy_analysis_adds_warming_level_column(self):
        """Check that run_xmy_analysis adds 'warming_level' column in GWL mode."""
        lat = 33.56
        lon = -117.81
        warming_level = 2.0
        extreme = "hot"
        xmy = shock_XMY(
            extreme=extreme, warming_level=warming_level, latitude=lat, longitude=lon
        )
        # In warming level mode, start_year/end_year are set during load
        xmy.start_year = 2001
        xmy.end_year = 2003

        sim = "WRF_EC-Earth3_r1i1p1f1"
        hourly_ds = mock_complete_hourly_ds()
        xmy._hourly_data = hourly_ds
        xmy.top_months = pd.DataFrame(
            {
                "month": list(range(1, 13)),
                "simulation": [sim] * 12,
                "year": [2001] * 12,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=OptimizeWarning)
            xmy.run_xmy_analysis()

        # GWL mode → warming_level column, not scenario
        assert "warming_level" in xmy.xmy_data_to_export[sim].columns
        assert "scenario" not in xmy.xmy_data_to_export[sim].columns
        assert (xmy.xmy_data_to_export[sim]["warming_level"] == 2.0).all()

    def test_export_xmy_data_uses_sim_centered_years(self):
        """Check that export_xmy_data reads centered_year from _sim_centered_years dict."""
        lat = 33.56
        lon = -117.81
        warming_level = 2.0
        extreme = "hot"
        xmy = shock_XMY(
            extreme=extreme, warming_level=warming_level, latitude=lat, longitude=lon
        )
        xmy.start_year = 2001
        xmy.end_year = 2003

        sim = "WRF_EC-Earth3_r1i1p1f1"
        xmy.xmy_data_to_export = {
            sim: pd.DataFrame(
                {
                    "time": pd.date_range("2000-01-01", periods=8760, freq="1h"),
                    "warming_level": [warming_level] * 8760,
                    "simulation": [sim] * 8760,
                    "Air Temperature at 2m": np.ones(8760),
                }
            )
        }
        xmy._sim_centered_years = {sim: 2040}

        with patch(
            "climakitae.explore.shock_extreme_meteorological_year.write_tmy_file"
        ) as mock_write:
            xmy.export_xmy_data()
            mock_write.assert_called_once()
            # write_xmy_file(filename, df, years, stn_name, ...)
            call_args = mock_write.call_args
            # Year range derived from centered_year 2040: (2025, 2054)
            assert call_args[0][2] == (2025, 2054)
            # Filename should contain warming level label appended to sim name
            assert "mid-century" in call_args[0][0]
