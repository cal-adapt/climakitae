import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.data_interface import DataParameters
from climakitae.explore.amy import (
    _format_meteo_yr_df,
    _set_amy_year_inputs,
    compute_amy,
    compute_mean_monthly_meteo_yr,
    compute_severe_yr,
    retrieve_meteo_yr_data,
)

@pytest.mark.elevated
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

@pytest.mark.elevated
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

@pytest.mark.elevated
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

@pytest.mark.elevated
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

@pytest.mark.elevated
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

@pytest.mark.elevated
def test_compute_amy() -> None:
    """
    Test the compute_amy function.
    """
    # Instead of testing with xr.DataArray()
    empty_da = xr.DataArray(
        data=[], coords={"time": pd.DatetimeIndex([])}, dims=["time"]
    )

    # test with empty data
    with pytest.raises(ValueError) as excinfo:
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

@pytest.mark.elevated
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

@pytest.mark.elevated
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
