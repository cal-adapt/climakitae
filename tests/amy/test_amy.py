import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch

from climakitae.core.data_interface import DataParameters
from climakitae.explore.amy import (
    _format_meteo_yr_df,
    _set_amy_year_inputs,
    compute_amy,
    compute_mean_monthly_meteo_yr,
    compute_profile,
    compute_severe_yr,
    get_climate_profile,
    retrieve_meteo_yr_data,
    retrieve_profile_data,
)


@pytest.mark.advanced
def test__set_amy_year_inputs() -> None:
    """
    Test the _set_amy_year_inputs function.
    """
    # Test with a valid year
    year1, year2 = 2000, 2030
    result = _set_amy_year_inputs(year1, year2)
    assert result == (2000, 2030)

    # Test with a year too far in the future
    year2 = 2200
    result = _set_amy_year_inputs(year1, year2)
    assert result == (2000, 2100)

    # Test with an invalid year
    year1, year2 = 1979, 2030
    # should raise ValueError exception
    with pytest.raises(ValueError) as excinfo:
        _set_amy_year_inputs(year1, year2)
    assert (
        str(excinfo.value)
        == """You've input an invalid start year. The start year must be 1980 or later."""
    )

    # test with years too close together
    year1, year2 = 2030, 2031
    with pytest.raises(ValueError) as excinfo:
        _set_amy_year_inputs(year1, year2)
    assert (
        str(excinfo.value)
        == """To compute an Average Meteorological Year, you must input a date range with a difference
            of at least 5 years, where the end year is no later than 2100 and the start year is no later than
            2095."""
    )


@pytest.mark.advanced
def test_retrieve_meteo_yr_data() -> None:
    """
    Test the retrieve_meteo_yr_data function.
    """

    data_params = DataParameters()

    # Test historical-only case (year_end < 2015)
    result = retrieve_meteo_yr_data(data_params, year_start=1980, year_end=2014)
    assert data_params.scenario_ssp == []
    assert data_params.scenario_historical == ["Historical Climate"]

    # Test multiple SSPs case
    data_params = DataParameters()
    data_params.scenario_ssp = ["SSP 2-4.5", "SSP 3-7.0"]
    retrieve_meteo_yr_data(data_params)
    assert data_params.scenario_ssp == ["SSP 2-4.5"]  # Should select first one

    # test modern climate data
    data_params = DataParameters()
    retrieve_meteo_yr_data(data_params, year_start=2016, year_end=2030)
    assert data_params.scenario_ssp == ["SSP 3-7.0"]

    # test valid ssp
    data_params = DataParameters()
    met_yr_data = retrieve_meteo_yr_data(
        data_params,
        ssp="SSP 2-4.5",
    )
    assert isinstance(met_yr_data, xr.DataArray)

    # test invalid ssp
    with pytest.raises(KeyError) as excinfo:
        retrieve_meteo_yr_data(
            data_params,
            ssp="invalid_ssp",
        )
    assert str(excinfo.value) == "'invalid_ssp'"


@pytest.mark.advanced
@patch("climakitae.explore.amy.read_catalog_from_select")
def test_retrieve_meteo_yr_data_no_data_available(mock_read_catalog):
    """Test error when no data is available for the given parameters."""
    # Mock read_catalog_from_select to return None (no data available)
    mock_read_catalog.return_value = None

    data_params = DataParameters()

    # This should raise the specific ValueError for insufficient data
    with pytest.raises(ValueError, match="COULD NOT RETRIEVE DATA"):
        retrieve_meteo_yr_data(data_params)


@pytest.mark.advanced
def create_mock_hourly_da() -> xr.DataArray:
    # Create mock data: 365 days × 24 hours with random temperature values

    mock_values = np.random.normal(
        20, 5, size=(365, 24)
    )  # Mean 20°C with 5°C standard deviation

    # Create the DataArray with proper dimensions - removed "time" from dims
    mock_da = xr.DataArray(
        data=mock_values,
        dims=["dayofyear", "hour"],  # Previously had "time" as a third dim
        coords={
            "dayofyear": np.arange(1, 366),
            "hour": np.arange(1, 25),
        },
    )

    return mock_da


@pytest.mark.advanced
def test__format_meteo_yr_data() -> None:
    """
    Test the _format_meteo_yr_data function.
    """

    # test empty data
    with pytest.raises(ValueError) as excinfo:
        _format_meteo_yr_df(
            pd.DataFrame(),
        )

    hourly_da = create_mock_hourly_da()

    # test valid data
    df = _format_meteo_yr_df(
        pd.DataFrame(
            hourly_da,
            columns=np.arange(1, 25, 1),
            index=np.arange(1, 366, 1),
        )
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (365, 24)
    assert df.index.name == "Day of Year"

    df = pd.DataFrame(
        np.random.rand(365, 24), columns=np.arange(1, 25), index=np.arange(1, 366)
    )
    formatted = _format_meteo_yr_df(df)
    assert len(formatted) == 365
    assert formatted.index.name == "Day of Year"


@pytest.mark.advanced
def create_mock_amy_data(days: int = 366) -> xr.DataArray:
    """Create a mock DataArray for testing compute_amy with minimal data size"""
    # Determine if we're creating leap year or non-leap year data
    leap_year = days == 366
    year = "2020" if leap_year else "2021"  # 2020 was a leap year, 2021 was not

    # Create exactly the right number of days (either 365 or 366)
    end_date = f"{year}-12-31" if days == 365 else f"{year}-12-31"
    times = pd.date_range(start=f"{year}-01-01", end=end_date, freq="h")

    # Generate simple temperature data with seasonal pattern
    data = 15 + 10 * np.sin(np.pi * np.arange(len(times)) / 12)

    # Create DataArray with same structure as the real one
    mock_da = xr.DataArray(
        data=data,
        dims=["time"],
        coords={"time": times},
        name="temperature",
    )

    # Add required additional coordinates
    mock_da = mock_da.assign_coords(
        {
            "simulation": "WRF_TaiESM1_r1i1p1f1",
            "Lambert_Conformal": 1,
            "scenario": "SSP 3-7.0",
        }
    )

    # Verify we have the expected number of unique days
    assert len(np.unique(mock_da.time.dt.dayofyear)) == days

    return mock_da


@pytest.mark.advanced
@pytest.mark.xfail(
    reason="compute_amy has a bug where it passes 'allofit' as quantile value instead of numeric"
)
def test_compute_amy() -> None:
    """
    Test the compute_amy function.

    NOTE: This test is expected to fail due to a bug in compute_amy where
    it tries to use 'allofit' as a quantile value (should be numeric).
    """
    # Instead of testing with xr.DataArray()
    empty_da = xr.DataArray(
        data=[], coords={"time": pd.DatetimeIndex([])}, dims=["time"]
    )

    # test with empty data
    with pytest.raises(ValueError):
        compute_amy(empty_da)

    # test with mock data
    mock_data = create_mock_amy_data()
    amy = compute_amy(
        mock_data,
    )
    assert isinstance(amy, pd.DataFrame)
    assert len(amy) == 366

    # Test with non-default days_in_year
    mock_data = create_mock_amy_data(days=365)
    amy = compute_amy(mock_data, days_in_year=365)
    assert isinstance(amy, pd.DataFrame)
    assert len(amy) == 365


@pytest.mark.advanced
def test_compute_severe_yr() -> None:
    """
    Test the compute_severe_yr function.
    """
    # Instead of testing with xr.DataArray()
    empty_da = xr.DataArray(
        data=[], coords={"time": pd.DatetimeIndex([])}, dims=["time"]
    )

    # test with empty data
    with pytest.raises(ValueError) as excinfo:
        compute_amy(empty_da)

    # Test with non-default days_in_year
    mock_data = create_mock_amy_data(days=365)
    sev_yr_data = compute_severe_yr(mock_data, days_in_year=365)
    assert isinstance(sev_yr_data, pd.DataFrame)
    assert sev_yr_data.shape == (365, 24)

    # test with mock data
    mock_data = create_mock_amy_data()
    sev_yr_data = compute_severe_yr(
        mock_data,
    )
    assert isinstance(sev_yr_data, pd.DataFrame)
    assert sev_yr_data.shape == (366, 24)
    assert sev_yr_data.index.name == "Day of Year"


@pytest.mark.advanced
def test_compute_mean_monthly_meteo_yr() -> None:
    """
    Test the compute_mean_monthly_meteo_yr function.
    """

    def _create_mock_tmy_df() -> pd.DataFrame:
        """
        Create a mock DataFrame that mimics the output of compute_amy or compute_severe_yr
        with realistic hourly temperature data following a diurnal and seasonal pattern.
        """
        # Create date range for a full year (using non-leap year for simplicity)
        dates = pd.date_range(start="2023-01-01", end="2023-12-31")

        # Create hour columns in the expected format (12-hour clock with am/pm)
        hour_columns = []
        for ampm in ["am", "pm"]:
            for h in range(1, 13):
                hour_columns.append(f"{h}{ampm}")

        # Create a proper index
        index = pd.Index(dates.strftime("%b-%d"), name="Day of Year")

        # Create MultiIndex columns with 'Hour' as name - this is key to the fix
        columns = pd.MultiIndex.from_product([["Hour"], hour_columns])

        # Initialize DataFrame with proper structure
        mock_df = pd.DataFrame(index=index, columns=hour_columns)

        # Fill with realistic data
        for i, date in enumerate(dates):
            seasonal_base = 15 - 15 * np.cos(2 * np.pi * (i / 365))

            for j, hour_label in enumerate(hour_columns):
                hour = j % 12 + 1
                if "pm" in hour_label and hour != 12:
                    hour += 12
                if "am" in hour_label and hour == 12:
                    hour = 0

                diurnal_effect = 5 * np.sin(np.pi * (hour - 6) / 12)
                noise = np.random.normal(0, 1)

                temp = seasonal_base + diurnal_effect + noise
                mock_df.loc[dates[i].strftime("%b-%d"), hour_label] = temp

        # Ensure columns have the name "Hour" - simpler alternative approach
        mock_df.columns.name = "Hour"

        return mock_df

    # mock data
    mock_df = _create_mock_tmy_df()

    # test with default parameters
    mean_monthly_data = compute_mean_monthly_meteo_yr(
        mock_df,
    )
    assert isinstance(mean_monthly_data, pd.DataFrame)
    assert mean_monthly_data.shape == (12, 1)
    assert mean_monthly_data.index.name == "Month"
    assert all(
        month in mean_monthly_data.index
        for month in [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )

    def _generate_day_from_index(i: int) -> str:
        """
        Helper function to generate a day string from the index.
        Uses correct days per month for a leap year.
        """
        # Days in each month (for leap year)
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        day = i
        month_idx = 0

        # Find the correct month
        while day > days_in_month[month_idx]:
            day -= days_in_month[month_idx]
            month_idx += 1
            if month_idx >= 12:  # Safety check
                month_idx = 11
                day = days_in_month[11]
                break

        # Format the day string
        return f"{month_names[month_idx]}-{day:02d}"

    # Create mock data
    mock_df = pd.DataFrame(
        np.random.rand(366, 24),
        columns=[f"{h}{ap}" for ap in ["am", "pm"] for h in range(1, 13)],
        index=pd.Index(
            [(_generate_day_from_index(i)) for i in range(1, 367)],
            name="Day of Year",
        ),
    )
    mock_df.columns.name = "Hour"

    # Test with custom column name
    mean_monthly_data = compute_mean_monthly_meteo_yr(mock_df, col_name="temperature")
    assert isinstance(mean_monthly_data, pd.DataFrame)
    assert "temperature" in mean_monthly_data.columns


class TestGetClimateProfile:
    """Test class for get_climate_profile function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock profile data structures
        self.mock_future_profile = pd.DataFrame(
            np.random.rand(365, 24),
            columns=np.arange(1, 25),
            index=pd.Index(
                [f"Jan-{i:02d}" for i in range(1, 32)]
                + [f"Feb-{i:02d}" for i in range(1, 29)]
                + [f"Mar-{i:02d}" for i in range(1, 32)]
                + [f"Apr-{i:02d}" for i in range(1, 31)]
                + [f"May-{i:02d}" for i in range(1, 32)]
                + [f"Jun-{i:02d}" for i in range(1, 31)]
                + [f"Jul-{i:02d}" for i in range(1, 32)]
                + [f"Aug-{i:02d}" for i in range(1, 32)]
                + [f"Sep-{i:02d}" for i in range(1, 31)]
                + [f"Oct-{i:02d}" for i in range(1, 32)]
                + [f"Nov-{i:02d}" for i in range(1, 31)]
                + [f"Dec-{i:02d}" for i in range(1, 32)][:365],
                name="Day of Year",
            ),
        )
        self.mock_future_profile.attrs = {"units": "degF", "variable_name": "tasmax"}

        self.mock_historic_profile = pd.DataFrame(
            np.random.rand(365, 24),
            columns=np.arange(1, 25),
            index=self.mock_future_profile.index,
        )
        self.mock_historic_profile.attrs = {"units": "degF", "variable_name": "tasmax"}

        # Create mock xarray data
        times = pd.date_range("2020-01-01", periods=8760, freq="h")
        self.mock_future_data = xr.Dataset(
            {"tasmax": (["time"], np.random.rand(8760))}, coords={"time": times}
        )
        self.mock_future_data.attrs = {"units": "degF"}

        self.mock_historic_data = xr.Dataset(
            {"tasmax": (["time"], np.random.rand(8760))}, coords={"time": times}
        )
        self.mock_historic_data.attrs = {"units": "degF"}

    @patch("climakitae.explore.amy.compute_profile")
    @patch("climakitae.explore.amy.retrieve_profile_data")
    @patch("builtins.print")
    def test_get_climate_profile_default_params(
        self, mock_print, mock_retrieve, mock_compute
    ):
        """Test get_climate_profile with default parameters.

        Tests basic functionality with default warming level,
        verifying data retrieval and profile computation workflow.
        """
        # Mock the retrieve_profile_data function
        mock_retrieve.return_value = (self.mock_historic_data, self.mock_future_data)

        # Mock the compute_profile function
        mock_compute.side_effect = [
            self.mock_future_profile,
            self.mock_historic_profile,
        ]

        # Call the function with minimal parameters (warming_level is required)
        result = get_climate_profile(warming_level=[2.0])

        # Verify data retrieval was called with correct parameters
        mock_retrieve.assert_called_once()
        call_args = mock_retrieve.call_args[1]  # Get kwargs
        assert call_args["warming_level"] == [2.0]

        # Verify compute_profile was called twice (future and historic)
        assert mock_compute.call_count == 2

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (365, 24)  # Default days_in_year=365, 24 hours

        # Verify difference calculation (future - historic)
        expected_diff = self.mock_future_profile - self.mock_historic_profile
        pd.testing.assert_frame_equal(result, expected_diff)

    @patch("climakitae.explore.amy.compute_profile")
    @patch("climakitae.explore.amy.retrieve_profile_data")
    @patch("builtins.print")
    def test_get_climate_profile_custom_params(
        self, mock_print, mock_retrieve, mock_compute
    ):
        """Test get_climate_profile with custom parameters.

        Tests functionality with various custom parameters including
        variable, resolution, units, and other optional parameters.
        """
        # Mock the retrieve_profile_data function
        mock_retrieve.return_value = (self.mock_historic_data, self.mock_future_data)

        # Mock the compute_profile function
        mock_compute.side_effect = [
            self.mock_future_profile,
            self.mock_historic_profile,
        ]

        # Call the function with custom parameters
        custom_params = {
            "variable": "Air Temperature at 2m",
            "resolution": "45 km",
            "warming_level": [1.5],
            "units": "degC",
            "days_in_year": 366,
            "q": 0.75,
            "cached_area": "Los Angeles County",
            "latitude": 34.0522,
            "longitude": -118.2437,
        }

        result = get_climate_profile(**custom_params)

        # Verify data retrieval was called with correct parameters
        mock_retrieve.assert_called_once()
        call_args = mock_retrieve.call_args[1]  # Get kwargs
        assert call_args["variable"] == "Air Temperature at 2m"
        assert call_args["resolution"] == "45 km"
        assert call_args["warming_level"] == [1.5]
        assert call_args["units"] == "degC"
        assert call_args["cached_area"] == "Los Angeles County"
        assert call_args["latitude"] == 34.0522
        assert call_args["longitude"] == -118.2437

        # Verify compute_profile was called with custom parameters
        assert mock_compute.call_count == 2
        # Check first call (future data)
        first_call_args = mock_compute.call_args_list[0][
            1
        ]  # Get kwargs from first call
        assert first_call_args["days_in_year"] == 366
        assert first_call_args["q"] == 0.75

        # Verify result is a DataFrame with correct shape
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (365, 24)  # Mock data shape

        # Verify difference calculation
        expected_diff = self.mock_future_profile - self.mock_historic_profile
        pd.testing.assert_frame_equal(result, expected_diff)

    @patch("climakitae.explore.amy.compute_profile")
    @patch("climakitae.explore.amy.retrieve_profile_data")
    @patch("builtins.print")
    def test_get_climate_profile_no_delta(
        self, mock_print, mock_retrieve, mock_compute
    ):
        """Test get_climate_profile with no_delta=True.

        Tests that when no_delta=True, the function returns raw future
        profile without baseline subtraction.
        """
        # Mock the retrieve_profile_data function
        mock_retrieve.return_value = (self.mock_historic_data, self.mock_future_data)

        # Mock the compute_profile function - should only be called once for future data
        mock_compute.return_value = self.mock_future_profile

        # Call the function with no_delta=True
        result = get_climate_profile(warming_level=[2.0], no_delta=True)

        # Verify data retrieval was called
        mock_retrieve.assert_called_once()

        # Verify compute_profile was called only once (for future data only)
        assert mock_compute.call_count == 1

        # Verify the function returned the future profile directly
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, self.mock_future_profile)

        # Verify print message about no baseline subtraction
        printed_messages = [str(call) for call in mock_print.call_args_list]
        found_no_delta_message = any(
            "No baseline subtraction requested" in msg for msg in printed_messages
        )
        assert (
            found_no_delta_message
        ), "Expected no baseline subtraction message not found"

    @patch("climakitae.explore.amy.compute_profile")
    @patch("climakitae.explore.amy.retrieve_profile_data")
    @patch("builtins.print")
    def test_get_climate_profile_multiple_warming_levels(
        self, mock_print, mock_retrieve, mock_compute
    ):
        """Test get_climate_profile with multiple warming levels.

        Tests functionality when multiple warming levels are provided,
        verifying proper handling of MultiIndex DataFrame structure.
        """
        # Create mock profiles with MultiIndex columns for multiple warming levels
        hours = list(range(1, 25))
        warming_levels = ["WL_1.5", "WL_2.0", "WL_3.0"]

        # Create MultiIndex columns (Hour, Warming_Level)
        multi_cols = pd.MultiIndex.from_product(
            [hours, warming_levels], names=["Hour", "Warming_Level"]
        )

        mock_future_multiindex = pd.DataFrame(
            np.random.rand(365, len(multi_cols)),
            columns=multi_cols,
            index=self.mock_future_profile.index,
        )
        mock_future_multiindex.attrs = {"units": "degF", "variable_name": "tasmax"}

        # Mock the retrieve_profile_data function
        mock_retrieve.return_value = (self.mock_historic_data, self.mock_future_data)

        # Mock compute_profile to return MultiIndex future, single-level historic
        mock_compute.side_effect = [mock_future_multiindex, self.mock_historic_profile]

        # Call the function with multiple warming levels
        result = get_climate_profile(warming_level=[1.5, 2.0, 3.0])

        # Verify data retrieval was called
        mock_retrieve.assert_called_once()
        call_args = mock_retrieve.call_args[1]
        assert call_args["warming_level"] == [1.5, 2.0, 3.0]

        # Verify compute_profile was called twice
        assert mock_compute.call_count == 2

        # Verify result is a DataFrame with MultiIndex columns
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)

        # Verify column structure
        assert "Hour" in result.columns.names
        assert "Warming_Level" in result.columns.names

        # Verify shape - should have same number of rows as input
        assert result.shape[0] == 365

        # The difference calculation should work with the MultiIndex structure
        # Each warming level column should be (future - historic) for corresponding hour
        for col in result.columns:
            hour = col[0]  # Hour is first level
            if hour in self.mock_historic_profile.columns:
                expected_val = (
                    mock_future_multiindex[col] - self.mock_historic_profile[hour]
                )
                pd.testing.assert_series_equal(
                    result[col], expected_val, check_names=False
                )

    @patch("climakitae.explore.amy.retrieve_profile_data")
    @patch("builtins.print")
    def test_get_climate_profile_data_retrieval_error(self, mock_print, mock_retrieve):
        """Test get_climate_profile when data retrieval fails.

        Tests error handling when retrieve_profile_data raises an exception.
        """
        # Mock retrieve_profile_data to raise an exception
        mock_retrieve.side_effect = ValueError("Failed to retrieve climate data")

        # Call the function and expect it to raise the same exception
        with pytest.raises(ValueError, match="Failed to retrieve climate data"):
            get_climate_profile(warming_level=[2.0])

        # Verify data retrieval was attempted
        mock_retrieve.assert_called_once()

    @patch("climakitae.explore.amy.compute_profile")
    @patch("climakitae.explore.amy.retrieve_profile_data")
    @patch("builtins.print")
    def test_get_climate_profile_compute_profile_error(
        self, mock_print, mock_retrieve, mock_compute
    ):
        """Test get_climate_profile when compute_profile fails.

        Tests error handling when compute_profile raises an exception.
        """
        # Mock successful data retrieval
        mock_retrieve.return_value = (self.mock_historic_data, self.mock_future_data)

        # Mock compute_profile to raise an exception on first call
        mock_compute.side_effect = ValueError("Profile computation failed")

        # Call the function and expect it to raise the same exception
        with pytest.raises(ValueError, match="Profile computation failed"):
            get_climate_profile(warming_level=[2.0])

        # Verify data retrieval succeeded
        mock_retrieve.assert_called_once()

        # Verify compute_profile was attempted
        mock_compute.assert_called_once()

    @patch("climakitae.explore.amy.retrieve_profile_data")
    def test_get_climate_profile_invalid_params(self, mock_retrieve):
        """Test get_climate_profile with invalid parameters.

        Tests error handling for various invalid parameter scenarios
        including missing required parameters, invalid keys, and wrong types.
        """
        # Test missing required parameter (warming_level)
        mock_retrieve.side_effect = ValueError(
            "Missing required input: 'warming_level'"
        )
        with pytest.raises(ValueError, match="Missing required input: 'warming_level'"):
            get_climate_profile(variable="Air Temperature at 2m")

        # Test invalid parameter key
        mock_retrieve.side_effect = ValueError(
            "Invalid input(s): ['invalid_param']. Allowed inputs are: ['variable', 'resolution', 'warming_level', 'cached_area', 'units', 'latitude', 'longitude']"
        )
        with pytest.raises(
            ValueError, match="Invalid input\\(s\\): \\['invalid_param'\\]"
        ):
            get_climate_profile(warming_level=[2.0], invalid_param="test")

        # Test invalid parameter type - warming_level should be list
        mock_retrieve.side_effect = TypeError(
            "Parameter 'warming_level' must be of type list, got float"
        )
        with pytest.raises(
            TypeError, match="Parameter 'warming_level' must be of type list"
        ):
            get_climate_profile(warming_level=2.0)

        # Test invalid parameter type - variable should be str
        mock_retrieve.side_effect = TypeError(
            "Parameter 'variable' must be of type str, got int"
        )
        with pytest.raises(TypeError, match="Parameter 'variable' must be of type str"):
            get_climate_profile(warming_level=[2.0], variable=123)

        # Test invalid parameter type - resolution should be str
        mock_retrieve.side_effect = TypeError(
            "Parameter 'resolution' must be of type str, got int"
        )
        with pytest.raises(
            TypeError, match="Parameter 'resolution' must be of type str"
        ):
            get_climate_profile(warming_level=[2.0], resolution=45)

        # Verify that retrieve_profile_data was called in all error cases
        assert mock_retrieve.call_count == 5


class TestRetrieveProfileData:
    """Test class for retrieve_profile_data function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock xarray datasets that would be returned by get_data
        times = pd.date_range("2020-01-01", periods=8760, freq="h")
        
        self.mock_historic_dataset = xr.Dataset(
            {"tasmax": (["time"], np.random.rand(8760))}, 
            coords={"time": times}
        )
        self.mock_historic_dataset.attrs = {
            "units": "degF", 
            "variable_id": "tasmax",
            "warming_level": 1.2
        }
        
        self.mock_future_dataset = xr.Dataset(
            {"tasmax": (["time"], np.random.rand(8760))}, 
            coords={"time": times}
        )
        self.mock_future_dataset.attrs = {
            "units": "degF", 
            "variable_id": "tasmax", 
            "warming_level": 2.0
        }

    @patch('climakitae.explore.amy.get_data')
    def test_retrieve_profile_data_default_params(self, mock_get_data):
        """Test retrieve_profile_data with default parameters.
        
        Tests that the function returns both historic and future data
        with proper default parameter handling and warming level setup.
        """
        # Mock get_data to return different datasets for historic vs future calls
        mock_get_data.side_effect = [self.mock_historic_dataset, self.mock_future_dataset]
        
        # Call function with minimal required parameter
        historic_data, future_data = retrieve_profile_data(warming_level=[2.0])
        
        # Verify return values are the expected datasets
        assert historic_data is self.mock_historic_dataset
        assert future_data is self.mock_future_dataset
        
        # Verify get_data was called twice (historic + future)
        assert mock_get_data.call_count == 2
        
        # Verify historic call used warming_level=1.2 and other defaults
        historic_call_kwargs = mock_get_data.call_args_list[0][1]  # First call kwargs
        assert historic_call_kwargs['warming_level'] == [1.2]
        assert historic_call_kwargs['variable'] == "Air Temperature at 2m"
        assert historic_call_kwargs['resolution'] == "3 km"
        assert historic_call_kwargs['downscaling_method'] == "Dynamical"
        assert historic_call_kwargs['timescale'] == "hourly"
        assert historic_call_kwargs['area_average'] == "Yes"
        assert historic_call_kwargs['approach'] == "Warming Level"
        
        # Verify future call used user-provided warming_level
        future_call_kwargs = mock_get_data.call_args_list[1][1]  # Second call kwargs  
        assert future_call_kwargs['warming_level'] == [2.0]
        
        # Verify both calls have consistent base parameters
        for call_kwargs in [historic_call_kwargs, future_call_kwargs]:
            assert call_kwargs['variable'] == "Air Temperature at 2m"
            assert call_kwargs['resolution'] == "3 km"
            assert call_kwargs['downscaling_method'] == "Dynamical"

    @patch('climakitae.explore.amy.get_data')
    def test_retrieve_profile_data_custom_params(self, mock_get_data):
        """Test retrieve_profile_data with custom parameters.
        
        Tests that all custom parameters are properly passed through
        to the get_data function calls for both historic and future data.
        """
        # Mock get_data to return different datasets
        mock_get_data.side_effect = [self.mock_historic_dataset, self.mock_future_dataset]
        
        # Call function with comprehensive custom parameters
        custom_params = {
            'variable': 'Air Temperature at 2m',
            'resolution': '45 km',
            'warming_level': [1.5, 2.0, 3.0],
            'units': 'degC',
            'cached_area': 'Los Angeles County',
            'latitude': 34.0522,
            'longitude': -118.2437
        }
        
        historic_data, future_data = retrieve_profile_data(**custom_params)
        
        # Verify return values are the expected datasets
        assert historic_data is self.mock_historic_dataset
        assert future_data is self.mock_future_dataset
        
        # Verify get_data was called twice
        assert mock_get_data.call_count == 2
        
        # Verify historic call parameters (always uses warming_level=1.2)
        historic_call_kwargs = mock_get_data.call_args_list[0][1]
        assert historic_call_kwargs['warming_level'] == [1.2]  # Always 1.2 for historic
        assert historic_call_kwargs['variable'] == 'Air Temperature at 2m'
        assert historic_call_kwargs['resolution'] == '45 km'
        assert historic_call_kwargs['units'] == 'degC'
        assert historic_call_kwargs['cached_area'] == 'Los Angeles County'
        assert historic_call_kwargs['latitude'] == 34.0522
        assert historic_call_kwargs['longitude'] == -118.2437
        
        # Verify future call parameters (uses user-provided warming levels)
        future_call_kwargs = mock_get_data.call_args_list[1][1]
        assert future_call_kwargs['warming_level'] == [1.5, 2.0, 3.0]
        assert future_call_kwargs['variable'] == 'Air Temperature at 2m'
        assert future_call_kwargs['resolution'] == '45 km'
        assert future_call_kwargs['units'] == 'degC'
        assert future_call_kwargs['cached_area'] == 'Los Angeles County'
        assert future_call_kwargs['latitude'] == 34.0522
        assert future_call_kwargs['longitude'] == -118.2437
        
        # Verify standard parameters are always set correctly
        for call_kwargs in [historic_call_kwargs, future_call_kwargs]:
            assert call_kwargs['downscaling_method'] == "Dynamical"
            assert call_kwargs['timescale'] == "hourly"
            assert call_kwargs['area_average'] == "Yes"
            assert call_kwargs['approach'] == "Warming Level"
