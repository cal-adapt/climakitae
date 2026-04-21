"""
Test the available extreme value theory functions for calculating return
values and periods. These tests do not check the correctness of the
calculations; they just ensure that the functions run without error, or raise
the expected error messages for invalid argument specifications.
"""

import os

import numpy as np
import pytest
import xarray as xr

from climakitae.explore import threshold_tools

# ------------- Data for testing -----------------------------------------------


@pytest.fixture
def T2_ams(rootdir: str) -> xr.DataArray:
    """Generate an annual maximum series (ams) datarray for testing."""
    # This data is generated in "create_test_data.py"
    test_filename = "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = (
        xr.open_dataset(test_filepath)
        .T2.mean(dim=("x", "y"), skipna=True)
        .isel(scenario=0, simulation=0)
    )
    return threshold_tools.get_block_maxima(test_data, block_size=1, check_ess=False)


# ------------- Test return values and periods ----------------------------------


@pytest.mark.advanced
def test_return_value(T2_ams: xr.DataArray):
    """Test Return Values."""
    rvs = threshold_tools.get_return_value(
        T2_ams, return_period=10, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    assert not np.isnan(rvs["return_value"].values[()])


@pytest.mark.advanced
def test_return_value_dropna(T2_ams: xr.DataArray):
    """Test Return Values with dropna_time option."""
    T2_ams.data[2] = np.nan
    rvs = threshold_tools.get_return_value(
        T2_ams,
        return_period=10,
        distr="gev",
        bootstrap_runs=1,
        multiple_points=False,
        dropna_time=True,
    )
    assert not np.isnan(rvs["return_value"].values[()])


@pytest.mark.advanced
def test_return_value_invalid_distr(T2_ams: xr.DataArray):
    """# Test invalid distribution argument for Return Values."""
    with pytest.raises(ValueError, match="invalid distribution type"):
        rvs = threshold_tools.get_return_value(
            T2_ams,
            return_period=10,
            distr="foo",
            bootstrap_runs=1,
            multiple_points=False,
        )


@pytest.mark.advanced
def test_return_period(T2_ams: xr.DataArray):
    """Test Return Periods."""
    rvs = threshold_tools.get_return_period(
        T2_ams,
        return_value=290,
        distr="gumbel",
        bootstrap_runs=1,
        multiple_points=False,
    )
    assert not np.isnan(rvs["return_period"].values[()])


@pytest.mark.advanced
def test_return_period_dropna(T2_ams: xr.DataArray):
    """Test Return Periods."""
    T2_ams.data[2] = np.nan
    rvs = threshold_tools.get_return_period(
        T2_ams,
        return_value=290,
        distr="gumbel",
        bootstrap_runs=1,
        multiple_points=False,
        dropna_time=True,
    )
    assert not np.isnan(rvs["return_period"].values[()])


@pytest.mark.advanced
def test_return_period_invalid_distr(T2_ams: xr.DataArray):
    """Test invalid distribution argument for Return Periods."""
    with pytest.raises(ValueError, match="invalid distribution type"):
        _ = threshold_tools.get_return_period(
            T2_ams,
            return_value=290,
            distr="foo",
            bootstrap_runs=1,
            multiple_points=False,
        )


@pytest.mark.advanced
def test_return_values_block_size(T2_ams: xr.DataArray):
    """Test return values for different block sizes."""
    rvs1 = threshold_tools.get_return_value(
        T2_ams, return_period=10, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # set different block size attribute to test that the calculation is handled differently:
    T2_ams.attrs["block size"] = "2 year"
    rvs2 = threshold_tools.get_return_value(
        T2_ams, return_period=10, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # test that the return values from longer block sizes should be smaller:
    assert rvs1["return_value"].values[()] >= rvs2["return_value"].values[()]


@pytest.mark.advanced
def test_return_periods_block_size(T2_ams: xr.DataArray):
    """Test return periods for different block sizes."""
    rps1 = threshold_tools.get_return_period(
        T2_ams, return_value=290, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # set different block size attribute to test that the calculation is handled differently:
    T2_ams.attrs["block size"] = "2 year"
    rps2 = threshold_tools.get_return_period(
        T2_ams, return_value=290, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # test that the return periods from longer block sizes should be larger:
    assert rps1["return_period"].values[()] <= rps2["return_period"].values[()]


@pytest.mark.advanced
def test_return_probs_block_size(T2_ams: xr.DataArray):
    """Test return probabilities for different block sizes."""
    rps1 = threshold_tools.get_return_prob(
        T2_ams, threshold=290, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # set different block size attribute to test that the calculation is handled differently:
    T2_ams.attrs["block size"] = "2 year"
    rps2 = threshold_tools.get_return_prob(
        T2_ams, threshold=290, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # test that the return probs from longer block sizes should be smaller:
    assert rps1["return_prob"].values[()] >= rps2["return_prob"].values[()]


# -------------- Test AMS block maxima calculations for complex extreme events


@pytest.mark.advanced
def test_ams_ex1(T2_hourly: xr.DataArray):
    """Test that the AMS (block maxima) for a 3-day grouped event are lower than
    the simple AMS (single hottest value in each year).
    """
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    ams = threshold_tools.get_block_maxima(T2_hourly, check_ess=False)
    ams_3d = threshold_tools.get_block_maxima(
        T2_hourly, groupby=(1, "day"), grouped_duration=(3, "day"), check_ess=False
    )
    assert (ams >= ams_3d).all()


@pytest.mark.advanced
def test_ams_ex2(T2_hourly: xr.DataArray):
    """Test that the AMS (block maxima) for a 3-day continous event are lower than
    the AMS for a grouped 3-day event.
    """
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    ams_3d = threshold_tools.get_block_maxima(
        T2_hourly, groupby=(1, "day"), grouped_duration=(3, "day"), check_ess=False
    )
    ams_72h = threshold_tools.get_block_maxima(
        T2_hourly, duration=(72, "hour"), check_ess=False
    )
    assert (ams_3d >= ams_72h).all()


@pytest.mark.advanced
def test_ams_ex3(T2_hourly: xr.DataArray):
    """Test that the AMS (block maxima) for a 4-hour per day for 3 days are lower
    than the AMS for a grouped 3-day event.
    """
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    ams_3d = threshold_tools.get_block_maxima(
        T2_hourly, groupby=(1, "day"), grouped_duration=(3, "day"), check_ess=False
    )
    ams_3d_4h = threshold_tools.get_block_maxima(
        T2_hourly,
        duration=(4, "hour"),
        groupby=(1, "day"),
        grouped_duration=(3, "day"),
        check_ess=False,
    )
    assert (ams_3d >= ams_3d_4h).all()


def test_ams_ex4(T2_hourly: xr.DataArray):
    """Test that the AMS (block maxima) for a 4-hour per day for 3 days are greater
    than the AMS for a grouped 3-day event.
    """
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    ams_3d = threshold_tools.get_block_maxima(
        T2_hourly,
        extremes_type="min",
        groupby=(1, "day"),
        grouped_duration=(3, "day"),
        check_ess=False,
    )
    ams_3d_4h = threshold_tools.get_block_maxima(
        T2_hourly,
        extremes_type="min",
        duration=(4, "hour"),
        groupby=(1, "day"),
        grouped_duration=(3, "day"),
        check_ess=False,
    )
    assert (ams_3d <= ams_3d_4h).all()


def test_block_maxima_value_error(T2_hourly: xr.DataArray):
    """Test the configurations that should raise ValueError."""
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly, extremes_type="mx", check_ess=False
        )

    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly, duration=(4, "day"), check_ess=False
        )

    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly, groupby=(4, "hr"), check_ess=False
        )

    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly, grouped_duration=(3, "day"), check_ess=False
        )

    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly,
            duration=(4, "month"),
            grouped_duration=(3, "day"),
            check_ess=False,
        )

    with pytest.raises(ValueError, match="invalid rolling_agg"):
        _ = threshold_tools.get_block_maxima(
            T2_hourly, rolling_agg="sum", check_ess=False
        )


class TestGetBlockMaximaRollingAgg:
    """Tests for the rolling_agg parameter of get_block_maxima.

    Covers all three valid modes ('sustained', 'cumulative', 'average') across
    the duration and groupby/grouped_duration parameter combinations.

    Value-comparison tests rely on the hourly temperature fixture (~300 K).
    With groupby=(1, 'day'):
      - 'cumulative' sums 24 hourly values  → daily total ≈ 7 200 K
      - 'average'    averages 24 hourly values → daily mean ≈ 300 K
      - 'sustained'  takes the daily max       → daily max  ≈ 305 K
    After a 3-day grouped_duration window, cumulative AMS >> average AMS,
    so the ordering is guaranteed regardless of natural variability in the data.
    """

    def test_rolling_agg_cumulative_shape(self, T2_hourly: xr.DataArray):
        """rolling_agg='cumulative' returns a DataArray with one value per year."""
        da = T2_hourly.isel(scenario=0, simulation=0)
        ams = threshold_tools.get_block_maxima(
            da,
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            check_ess=False,
            rolling_agg="cumulative",
        )
        assert isinstance(ams, xr.DataArray)
        assert "time" in ams.dims
        assert len(ams.time) == len(
            np.unique(da.time.dt.year.values)
        )  # one block per year

    def test_rolling_agg_average_shape(self, T2_hourly: xr.DataArray):
        """rolling_agg='average' returns a DataArray with one value per year."""
        da = T2_hourly.isel(scenario=0, simulation=0)
        ams = threshold_tools.get_block_maxima(
            da,
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            check_ess=False,
            rolling_agg="average",
        )
        assert isinstance(ams, xr.DataArray)
        assert "time" in ams.dims

    def test_rolling_agg_sustained_shape(self, T2_hourly: xr.DataArray):
        """rolling_agg='sustained' (explicit) returns a DataArray matching existing behavior."""
        da = T2_hourly.isel(scenario=0, simulation=0)
        ams_default = threshold_tools.get_block_maxima(
            da, groupby=(1, "day"), grouped_duration=(3, "day"), check_ess=False
        )
        ams_explicit = threshold_tools.get_block_maxima(
            da,
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            check_ess=False,
            rolling_agg="sustained",
        )
        xr.testing.assert_equal(ams_default, ams_explicit)

    @pytest.mark.advanced
    def test_rolling_agg_cumulative_exceeds_average(self, T2_hourly: xr.DataArray):
        """Cumulative groupby AMS is larger than average groupby AMS.

        With groupby=(1, 'day'), cumulative sums 24 hourly values while average
        takes the mean of 24 values, so cumulative = 24 * average per day.
        After a 3-day grouped_duration the ratio is 72x, making the inequality strict.
        """
        da = T2_hourly.isel(scenario=0, simulation=0)
        ams_cumulative = threshold_tools.get_block_maxima(
            da,
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            check_ess=False,
            rolling_agg="cumulative",
        )
        ams_average = threshold_tools.get_block_maxima(
            da,
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            check_ess=False,
            rolling_agg="average",
        )
        assert (ams_cumulative > ams_average).all()

    @pytest.mark.advanced
    def test_rolling_agg_cumulative_exceeds_sustained(self, T2_hourly: xr.DataArray):
        """Cumulative daily-sum AMS is larger than sustained daily-max AMS.

        The daily sum (~7 200 K) is much larger than the daily maximum (~305 K),
        so the cumulative 3-day total AMS will always exceed the sustained AMS.
        """
        da = T2_hourly.isel(scenario=0, simulation=0)
        ams_cumulative = threshold_tools.get_block_maxima(
            da,
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            check_ess=False,
            rolling_agg="cumulative",
        )
        ams_sustained = threshold_tools.get_block_maxima(
            da,
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            check_ess=False,
            rolling_agg="sustained",
        )
        assert (ams_cumulative > ams_sustained).all()

    @pytest.mark.advanced
    def test_rolling_agg_cumulative_with_duration(self, T2_hourly: xr.DataArray):
        """rolling_agg='cumulative' with duration returns larger AMS than 'sustained'.

        A 4-hour rolling sum (~1 200 K) is much larger than a 4-hour rolling
        minimum (~295 K), so the inequality is guaranteed.
        """
        da = T2_hourly.isel(scenario=0, simulation=0)
        ams_cumulative = threshold_tools.get_block_maxima(
            da, duration=(4, "hour"), check_ess=False, rolling_agg="cumulative"
        )
        ams_sustained = threshold_tools.get_block_maxima(
            da, duration=(4, "hour"), check_ess=False, rolling_agg="sustained"
        )
        assert (ams_cumulative > ams_sustained).all()

    @pytest.mark.advanced
    def test_rolling_agg_average_with_duration(self, T2_hourly: xr.DataArray):
        """rolling_agg='average' with duration returns larger AMS than 'sustained'.

        A 4-hour rolling mean (~300 K) is larger than a 4-hour rolling
        minimum (~295 K) for temperature data, so the inequality holds.
        """
        da = T2_hourly.isel(scenario=0, simulation=0)
        ams_average = threshold_tools.get_block_maxima(
            da, duration=(4, "hour"), check_ess=False, rolling_agg="average"
        )
        ams_sustained = threshold_tools.get_block_maxima(
            da, duration=(4, "hour"), check_ess=False, rolling_agg="sustained"
        )
        assert (ams_average >= ams_sustained).all()
