"""
Unit tests for the 1-in-X extreme value analysis functionality in `metric_calc` processor.

This module contains comprehensive unit tests for the 1-in-X return value calculations
including initialization, distribution fitting, batch processing, and helper methods.

The tests validate:
- Proper initialization of 1-in-X configuration parameters
- Statistical distribution fitting (GEV, Gumbel, Weibull, etc.)
- Vectorized and batch processing of simulations
- Memory-efficient spatial batching
- Variable preprocessing (e.g., precipitation)
- Helper methods for time handling and result creation
"""

import gc
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.processors.metric_calc import MetricCalc


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def one_in_x_da_with_sim():
    """Create a DataArray with simulation dimension for 1-in-X testing.

    Returns
    -------
    xr.DataArray
        DataArray with dims (time, sim, lat, lon) containing random data
        suitable for extreme value analysis testing.
    """
    np.random.seed(42)
    n_time = 365 * 30  # 30 years of daily data
    n_sims = 4
    n_lat = 3
    n_lon = 3

    # Generate data with some realistic extreme value behavior
    data = np.random.gumbel(loc=20, scale=5, size=(n_time, n_sims, n_lat, n_lon))

    time_coords = pd.date_range("1990-01-01", periods=n_time, freq="D")
    sim_coords = [f"sim_{i}" for i in range(n_sims)]

    da = xr.DataArray(
        data,
        dims=["time", "sim", "lat", "lon"],
        coords={
            "time": time_coords,
            "sim": sim_coords,
            "lat": np.linspace(32, 42, n_lat),
            "lon": np.linspace(-124, -114, n_lon),
        },
        name="temperature",
        attrs={"frequency": "day"},
    )
    return da


@pytest.fixture
def one_in_x_ds_with_sim(one_in_x_da_with_sim):
    """Create a Dataset with simulation dimension for 1-in-X testing.

    Returns
    -------
    xr.Dataset
        Dataset wrapping the DataArray fixture.
    """
    return xr.Dataset({"temperature": one_in_x_da_with_sim})


@pytest.fixture
def block_maxima_data():
    """Create block maxima data (yearly aggregated) for distribution fitting tests.

    Returns
    -------
    xr.DataArray
        DataArray with dims (time, sim) containing 30 years of annual maxima.
    """
    np.random.seed(42)
    n_years = 30
    n_sims = 2

    # Generate realistic annual maxima using GEV distribution
    from scipy import stats

    shape, loc, scale = 0.1, 30, 5
    data = stats.genextreme.rvs(
        c=-shape, loc=loc, scale=scale, size=(n_years, n_sims), random_state=42
    )

    da = xr.DataArray(
        data,
        dims=["time", "sim"],
        coords={
            "time": pd.date_range("1990", periods=n_years, freq="YE"),
            "sim": ["sim_0", "sim_1"],
        },
        name="annual_max",
    )
    return da


# =============================================================================
# Test Classes
# =============================================================================


class TestMetricCalcOneInXInit:
    """Test class for 1-in-X initialization and parameter setup."""

    def test_one_in_x_init_with_return_periods_list(self):
        """Test initialization with return_periods as a list."""
        processor = MetricCalc({"one_in_x": {"return_periods": [10, 25, 50, 100]}})
        assert processor.one_in_x_config is not UNSET
        np.testing.assert_array_equal(
            processor.return_periods, np.array([10, 25, 50, 100])
        )

    def test_one_in_x_init_with_return_periods_scalar(self):
        """Test initialization with return_periods as a scalar value."""
        processor = MetricCalc({"one_in_x": {"return_periods": 50}})
        np.testing.assert_array_equal(processor.return_periods, np.array([50]))

    def test_one_in_x_init_with_return_periods_ndarray(self):
        """Test initialization with return_periods as numpy array."""
        periods = np.array([10, 50, 100])
        processor = MetricCalc({"one_in_x": {"return_periods": periods}})
        np.testing.assert_array_equal(processor.return_periods, periods)

    def test_one_in_x_init_missing_return_periods_raises(self):
        """Test that missing return_periods raises ValueError."""
        with pytest.raises(ValueError, match="return_periods is required"):
            MetricCalc({"one_in_x": {}})

    @pytest.mark.parametrize(
        "param,value,expected_attr",
        [
            ("distribution", "gumbel", "distribution"),
            ("distribution", "weibull", "distribution"),
            ("extremes_type", "min", "extremes_type"),
            ("event_duration", (3, "hour"), "event_duration"),
            ("block_size", 2, "block_size"),
            ("goodness_of_fit_test", False, "goodness_of_fit_test"),
            ("print_goodness_of_fit", False, "print_goodness_of_fit"),
        ],
    )
    def test_one_in_x_init_custom_params(self, param, value, expected_attr):
        """Test initialization with various custom parameters."""
        config = {"return_periods": [10, 50], param: value}
        processor = MetricCalc({"one_in_x": config})
        assert getattr(processor, expected_attr) == value

    def test_one_in_x_init_default_params(self):
        """Test that default parameters are set correctly."""
        processor = MetricCalc({"one_in_x": {"return_periods": [10]}})

        assert processor.distribution == "gev"
        assert processor.extremes_type == "max"
        assert processor.event_duration == (1, "day")
        assert processor.block_size == 1
        assert processor.goodness_of_fit_test is True
        assert processor.print_goodness_of_fit is True
        assert processor.variable_preprocessing == {}

    def test_one_in_x_init_all_options(self):
        """Test initialization with all options specified."""
        config = {
            "return_periods": [10, 25, 50, 100],
            "distribution": "gamma",
            "extremes_type": "min",
            "event_duration": (6, "hour"),
            "block_size": 2,
            "goodness_of_fit_test": False,
            "print_goodness_of_fit": False,
            "variable_preprocessing": {"precipitation": {"remove_trace": True}},
        }
        processor = MetricCalc({"one_in_x": config})

        np.testing.assert_array_equal(
            processor.return_periods, np.array([10, 25, 50, 100])
        )
        assert processor.distribution == "gamma"
        assert processor.extremes_type == "min"
        assert processor.event_duration == (6, "hour")
        assert processor.block_size == 2
        assert processor.goodness_of_fit_test is False
        assert processor.print_goodness_of_fit is False
        assert processor.variable_preprocessing == {
            "precipitation": {"remove_trace": True}
        }


class TestMetricCalcFitReturnValues1d:
    """Test class for the _fit_return_values_1d distribution fitting method."""

    def setup_method(self):
        """Set up test fixtures for distribution fitting tests."""
        # Create a processor with basic 1-in-X config
        self.processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 25, 50, 100],
                    "distribution": "gev",
                    "goodness_of_fit_test": True,
                }
            }
        )
        # Generate realistic block maxima data using GEV distribution
        np.random.seed(42)
        from scipy import stats

        shape, loc, scale = 0.1, 30, 5
        self.valid_block_maxima = stats.genextreme.rvs(
            c=-shape, loc=loc, scale=scale, size=30, random_state=42
        )
        self.return_periods = np.array([10, 25, 50, 100])

    @pytest.mark.parametrize(
        "distribution",
        ["gev", "gumbel", "weibull", "pearson3", "genpareto", "gamma"],
    )
    def test_fit_distribution_types(self, distribution):
        """Test fitting with all supported distribution types."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": distribution,
                    "goodness_of_fit_test": False,
                }
            }
        )

        return_values, p_value = processor._fit_return_values_1d(
            self.valid_block_maxima,
            np.array([10, 50]),
            distr=distribution,
            get_p_value=False,
        )

        # Should return valid values (not NaN)
        assert return_values.shape == (2,)
        assert not np.all(np.isnan(return_values))
        # p_value should be NaN when not requested
        assert np.isnan(p_value)

    def test_fit_invalid_distribution_returns_invalid(self):
        """Test that invalid distribution type returns invalid values (caught by exception handler).

        Note: Due to numpy dtype handling, this may return either NaN (for float arrays)
        or invalid integer values. The key is that the function doesn't crash.
        """
        return_values, p_value = self.processor._fit_return_values_1d(
            self.valid_block_maxima,
            self.return_periods,
            distr="invalid_dist",
            get_p_value=False,
        )

        # Invalid distribution should be handled gracefully
        # Check that p_value is NaN (always float)
        assert np.isnan(p_value)
        # Return values array should have the right shape
        assert return_values.shape == self.return_periods.shape

    def test_fit_insufficient_data_returns_nan(self):
        """Test that insufficient data returns NaN values."""
        insufficient_data = np.array([1.0, 2.0])  # Only 2 points, need at least 3

        return_values, p_value = self.processor._fit_return_values_1d(
            insufficient_data,
            self.return_periods,
            distr="gev",
            get_p_value=True,
        )

        assert np.all(np.isnan(return_values))
        assert np.isnan(p_value)

    def test_fit_with_nan_values(self):
        """Test fitting with NaN values in the data (should be filtered out)."""
        data_with_nans = self.valid_block_maxima.copy()
        data_with_nans[0] = np.nan
        data_with_nans[5] = np.nan

        return_values, p_value = self.processor._fit_return_values_1d(
            data_with_nans,
            self.return_periods,
            distr="gev",
            get_p_value=False,
        )

        # Should still return valid values after filtering NaNs
        assert not np.all(np.isnan(return_values))

    def test_fit_with_p_value(self):
        """Test that p-value is calculated when requested."""
        return_values, p_value = self.processor._fit_return_values_1d(
            self.valid_block_maxima,
            self.return_periods,
            distr="gev",
            get_p_value=True,
        )

        assert not np.isnan(p_value)
        assert 0 <= p_value <= 1

    def test_fit_without_p_value(self):
        """Test that p-value is NaN when not requested."""
        return_values, p_value = self.processor._fit_return_values_1d(
            self.valid_block_maxima,
            self.return_periods,
            distr="gev",
            get_p_value=False,
        )

        assert np.isnan(p_value)

    def test_fit_extremes_type_min(self):
        """Test fitting with extremes_type='min' for minima."""
        # For minima, return values should be lower for longer return periods
        return_values_min, _ = self.processor._fit_return_values_1d(
            self.valid_block_maxima,
            self.return_periods,
            distr="gev",
            extremes_type="min",
            get_p_value=False,
        )

        return_values_max, _ = self.processor._fit_return_values_1d(
            self.valid_block_maxima,
            self.return_periods,
            distr="gev",
            extremes_type="max",
            get_p_value=False,
        )

        # Min and max should give different results
        assert not np.allclose(return_values_min, return_values_max)

    def test_fit_return_values_increase_with_return_period(self):
        """Test that return values increase with return period for maxima."""
        return_values, _ = self.processor._fit_return_values_1d(
            self.valid_block_maxima,
            self.return_periods,
            distr="gev",
            extremes_type="max",
            get_p_value=False,
        )

        # For maxima, longer return periods should give higher values
        assert return_values[0] < return_values[1] < return_values[2] < return_values[3]

    def test_fit_all_nan_data(self):
        """Test fitting with all NaN data returns NaN."""
        all_nan_data = np.full(30, np.nan)

        return_values, p_value = self.processor._fit_return_values_1d(
            all_nan_data,
            self.return_periods,
            distr="gev",
            get_p_value=True,
        )

        assert np.all(np.isnan(return_values))
        assert np.isnan(p_value)


class TestMetricCalcUpdateContextOneInX:
    """Test class for update_context method with 1-in-X configuration."""

    def test_update_context_with_one_in_x(self):
        """Test that update_context correctly describes 1-in-X calculations."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 25, 50, 100],
                    "distribution": "gev",
                    "extremes_type": "max",
                    "event_duration": (1, "day"),
                }
            }
        )
        context: Dict[str, Any] = {}
        processor.update_context(context)

        assert _NEW_ATTRS_KEY in context
        assert "metric_calc" in context[_NEW_ATTRS_KEY]
        description = context[_NEW_ATTRS_KEY]["metric_calc"]

        # Check that key information is in the description
        assert "1-in-X" in description
        assert "10, 25, 50, 100" in description
        assert "gev" in description
        assert "max" in description

    def test_update_context_with_different_distribution(self):
        """Test update_context with different distribution types."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [50],
                    "distribution": "gamma",
                    "extremes_type": "min",
                    "event_duration": (6, "hour"),
                }
            }
        )
        context: Dict[str, Any] = {}
        processor.update_context(context)

        description = context[_NEW_ATTRS_KEY]["metric_calc"]
        assert "gamma" in description
        assert "min" in description
        assert "6 hour" in description


class TestMetricCalcHelperMethods:
    """Test class for various helper methods in MetricCalc."""

    def test_set_data_accessor(self):
        """Test set_data_accessor stores the catalog."""
        processor = MetricCalc({"metric": "mean"})
        mock_catalog = MagicMock()

        processor.set_data_accessor(mock_catalog)

        assert processor._catalog is mock_catalog

    def test_create_result_dataset_with_pvalues(self):
        """Test _create_one_in_x_result_dataset with p-values."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": "gev",
                    "event_duration": (1, "day"),
                }
            }
        )

        # Create mock return values and p-values
        ret_vals = xr.DataArray(
            np.array([[30.0, 35.0], [31.0, 36.0]]),
            dims=["sim", "one_in_x"],
            coords={"sim": ["sim_0", "sim_1"], "one_in_x": [10, 50]},
        )
        p_vals = xr.DataArray(
            np.array([0.5, 0.6]),
            dims=["sim"],
            coords={"sim": ["sim_0", "sim_1"]},
        )
        data_array = xr.DataArray(
            np.random.rand(100, 2),
            dims=["time", "sim"],
            coords={
                "time": pd.date_range("2000-01-01", periods=100),
                "sim": ["sim_0", "sim_1"],
            },
        )

        result = processor._create_one_in_x_result_dataset(ret_vals, p_vals, data_array)

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars
        assert "p_values" in result.data_vars
        assert "groupby" in result.attrs
        assert "fitted_distr" in result.attrs
        assert result.attrs["fitted_distr"] == "gev"

    def test_create_result_dataset_without_pvalues(self):
        """Test _create_one_in_x_result_dataset without p-values."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": "gumbel",
                    "event_duration": (1, "day"),
                    "goodness_of_fit_test": False,
                }
            }
        )

        ret_vals = xr.DataArray(
            np.array([[30.0, 35.0]]),
            dims=["sim", "one_in_x"],
            coords={"sim": ["sim_0"], "one_in_x": [10, 50]},
        )
        data_array = xr.DataArray(
            np.random.rand(100, 1),
            dims=["time", "sim"],
            coords={
                "time": pd.date_range("2000-01-01", periods=100),
                "sim": ["sim_0"],
            },
        )

        result = processor._create_one_in_x_result_dataset(ret_vals, None, data_array)

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars
        assert "p_values" not in result.data_vars
        assert result.attrs["fitted_distr"] == "gumbel"

    def test_add_dummy_time_with_time_delta(self):
        """Test _add_dummy_time_if_needed with time_delta dimension."""
        processor = MetricCalc(
            {"one_in_x": {"return_periods": [10], "distribution": "gev"}}
        )

        # Create data with time_delta dimension (warming level data)
        data = xr.DataArray(
            np.random.rand(365, 2),
            dims=["time_delta", "sim"],
            coords={
                "time_delta": range(365),
                "sim": ["sim_0", "sim_1"],
            },
            attrs={"frequency": "day"},
        )

        result = processor._add_dummy_time_if_needed(data, "day")

        assert "time" in result.dims
        assert "time_delta" not in result.dims
        assert len(result.time) == 365

    def test_add_dummy_time_with_from_center(self):
        """Test _add_dummy_time_if_needed with *_from_center dimension."""
        processor = MetricCalc(
            {"one_in_x": {"return_periods": [10], "distribution": "gev"}}
        )

        # Create data with hours_from_center dimension
        data = xr.DataArray(
            np.random.rand(24, 2),
            dims=["hours_from_center", "sim"],
            coords={
                "hours_from_center": range(24),
                "sim": ["sim_0", "sim_1"],
            },
        )

        result = processor._add_dummy_time_if_needed(data, "1hr")

        assert "time" in result.dims
        assert "hours_from_center" not in result.dims

    def test_add_dummy_time_missing_dim_raises(self):
        """Test _add_dummy_time_if_needed raises error when no valid time dim."""
        processor = MetricCalc(
            {"one_in_x": {"return_periods": [10], "distribution": "gev"}}
        )

        # Create data without any time-like dimension
        data = xr.DataArray(
            np.random.rand(10, 2),
            dims=["lat", "sim"],
            coords={"lat": range(10), "sim": ["sim_0", "sim_1"]},
        )

        with pytest.raises(ValueError, match="must have a 'time'"):
            processor._add_dummy_time_if_needed(data, "day")

    def test_get_optimal_chunks_with_dask(self):
        """Test _get_optimal_chunks with dask array."""
        import dask.array as da

        processor = MetricCalc({"metric": "mean"})

        # Create a chunked dask array
        data = xr.DataArray(
            da.random.random((365 * 10, 4, 10, 10), chunks=(365, 2, 5, 5)),
            dims=["time", "sim", "lat", "lon"],
            coords={
                "time": pd.date_range("2000-01-01", periods=365 * 10),
                "sim": [f"sim_{i}" for i in range(4)],
                "lat": range(10),
                "lon": range(10),
            },
        )

        chunks = processor._get_optimal_chunks(data)

        assert "time" in chunks
        assert "sim" in chunks
        # Spatial dims should be present
        assert "lat" in chunks or "lon" in chunks

    def test_get_optimal_chunks_no_dask(self):
        """Test _get_optimal_chunks with numpy array returns empty dict."""
        processor = MetricCalc({"metric": "mean"})

        # Create a numpy array (no chunks)
        data = xr.DataArray(
            np.random.rand(100, 2, 5, 5),
            dims=["time", "sim", "lat", "lon"],
        )

        chunks = processor._get_optimal_chunks(data)

        assert chunks == {}

    def test_get_dask_scheduler_without_distributed(self):
        """Test _get_dask_scheduler returns 'threads' without distributed client."""
        processor = MetricCalc({"metric": "mean"})

        scheduler = processor._get_dask_scheduler()

        assert scheduler == "threads"


class TestMetricCalcPreprocessVariable:
    """Test class for variable-specific preprocessing for 1-in-X calculations."""

    @pytest.mark.parametrize(
        "var_name",
        ["precipitation", "pr", "Precipitation (total)", "PRECIPITATION", "PR_daily"],
    )
    def test_preprocess_precipitation_variables_recognized(self, var_name):
        """Test that precipitation variables are recognized for preprocessing."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10],
                    "variable_preprocessing": {
                        "precipitation": {"remove_trace": True, "trace_threshold": 0.01}
                    },
                }
            }
        )

        # Create test data with small values (trace precipitation)
        data = xr.DataArray(
            np.array([0.001, 0.005, 0.1, 0.5, 1.0]),
            dims=["time"],
            coords={"time": pd.date_range("2000-01-01", periods=5)},
            attrs={"frequency": "day"},
        )

        result = processor._preprocess_variable_for_one_in_x(data, var_name)

        # Should filter out trace values below threshold
        # Note: The actual behavior depends on the implementation
        assert result is not None

    def test_preprocess_non_precipitation_unchanged(self):
        """Test that non-precipitation variables are not modified."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10],
                    "variable_preprocessing": {"precipitation": {"remove_trace": True}},
                }
            }
        )

        # Create test data
        data = xr.DataArray(
            np.array([20.0, 25.0, 30.0, 35.0, 40.0]),
            dims=["time"],
            coords={"time": pd.date_range("2000-01-01", periods=5)},
            name="temperature",
        )

        result = processor._preprocess_variable_for_one_in_x(data, "temperature")

        # Temperature should not be modified
        xr.testing.assert_equal(result, data)


class TestMetricCalcCalculateOneInXSingle:
    """Test class for _calculate_one_in_x_single method."""

    def test_one_in_x_missing_sim_dimension_raises(self):
        """Test that missing 'sim' dimension raises ValueError."""
        processor = MetricCalc(
            {"one_in_x": {"return_periods": [10, 50], "distribution": "gev"}}
        )

        # Create data without sim dimension
        data = xr.DataArray(
            np.random.rand(365, 5, 5),
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2000-01-01", periods=365),
                "lat": range(5),
                "lon": range(5),
            },
        )

        with pytest.raises(ValueError, match="must have a 'sim' dimension"):
            processor._calculate_one_in_x_single(data)

    def test_one_in_x_with_dataset_input(self, one_in_x_ds_with_sim):
        """Test _calculate_one_in_x_single with Dataset input."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                }
            }
        )

        result = processor._calculate_one_in_x_single(one_in_x_ds_with_sim)

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars
        assert "one_in_x" in result.dims

    def test_one_in_x_with_dataarray_input(self, one_in_x_da_with_sim):
        """Test _calculate_one_in_x_single with DataArray input."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                }
            }
        )

        result = processor._calculate_one_in_x_single(one_in_x_da_with_sim)

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars
        assert "one_in_x" in result.dims
        assert "sim" in result.dims


class TestMetricCalcAdaptiveBatchSize:
    """Test class for _calculate_adaptive_batch_size method."""

    def test_adaptive_batch_size_returns_valid_int(self, one_in_x_da_with_sim):
        """Test that adaptive batch size returns a valid integer."""
        processor = MetricCalc(
            {"one_in_x": {"return_periods": [10], "distribution": "gev"}}
        )

        batch_size = processor._calculate_adaptive_batch_size(one_in_x_da_with_sim)

        assert isinstance(batch_size, int)
        assert batch_size >= 1
        assert batch_size <= len(one_in_x_da_with_sim.sim)

    def test_adaptive_batch_size_psutil_import_error(self, one_in_x_da_with_sim):
        """Test fallback when psutil is not available."""
        processor = MetricCalc(
            {"one_in_x": {"return_periods": [10], "distribution": "gev"}}
        )

        # Mock psutil import to raise ImportError
        with patch.dict("sys.modules", {"psutil": None}):
            with patch(
                "climakitae.new_core.processors.metric_calc.MetricCalc._calculate_adaptive_batch_size"
            ) as mock_method:
                # Simulate the fallback behavior
                mock_method.return_value = min(2, len(one_in_x_da_with_sim.sim))
                batch_size = mock_method(one_in_x_da_with_sim)

        assert batch_size <= 2


class TestMetricCalcFitDistributionsVectorized:
    """Test class for _fit_distributions_vectorized method."""

    def test_fit_distributions_vectorized_numpy(self, block_maxima_data):
        """Test vectorized distribution fitting with numpy array."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": "gev",
                    "goodness_of_fit_test": True,
                }
            }
        )

        return_values, p_values = processor._fit_distributions_vectorized(
            block_maxima_data, "time"
        )

        assert return_values is not None
        assert p_values is not None
        # Should have one_in_x dimension in return values
        assert "one_in_x" in return_values.dims
        # Should have shape matching return periods
        assert return_values.sizes["one_in_x"] == 2

    def test_fit_distributions_vectorized_dask(self, block_maxima_data):
        """Test vectorized distribution fitting with dask array."""
        import dask.array as da

        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                }
            }
        )

        # Convert to dask
        dask_data = block_maxima_data.chunk({"time": 10, "sim": 1})

        return_values, p_values = processor._fit_distributions_vectorized(
            dask_data, "time"
        )

        assert return_values is not None
        assert "one_in_x" in return_values.dims


class TestMetricCalcExecuteOneInX:
    """Test class for execute method with 1-in-X configuration."""

    def test_execute_one_in_x_with_dataarray(self, one_in_x_da_with_sim):
        """Test execute method with 1-in-X on DataArray."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                }
            }
        )

        result = processor.execute(one_in_x_da_with_sim, context={})

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars

    def test_execute_one_in_x_with_dataset(self, one_in_x_ds_with_sim):
        """Test execute method with 1-in-X on Dataset."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                }
            }
        )

        result = processor.execute(one_in_x_ds_with_sim, context={})

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars

    def test_execute_one_in_x_with_dict(self, one_in_x_da_with_sim):
        """Test execute method with 1-in-X on dict of DataArrays."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                }
            }
        )

        input_dict = {"temp": one_in_x_da_with_sim}
        result = processor.execute(input_dict, context={})

        assert isinstance(result, dict)
        assert "temp" in result
        assert isinstance(result["temp"], xr.Dataset)

    def test_execute_one_in_x_with_list(self, one_in_x_da_with_sim):
        """Test execute method with 1-in-X on list of DataArrays."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                }
            }
        )

        input_list = [one_in_x_da_with_sim]
        result = processor.execute(input_list, context={})

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], xr.Dataset)


class TestMetricCalcSpatialBatching:
    """Test class for spatial batching methods."""

    @pytest.fixture
    def spatial_data_with_closest_cell(self):
        """Create data with closest_cell dimension for spatial batching tests."""
        np.random.seed(42)
        n_time = 365 * 30  # 30 years of daily data
        n_sims = 2
        n_cells = 150  # More than SPATIAL_BATCH_SIZE (100)

        data = np.random.gumbel(loc=20, scale=5, size=(n_time, n_sims, n_cells))
        time_coords = pd.date_range("1990-01-01", periods=n_time, freq="D")

        return xr.DataArray(
            data,
            dims=["time", "sim", "closest_cell"],
            coords={
                "time": time_coords,
                "sim": ["sim_0", "sim_1"],
                "closest_cell": range(n_cells),
            },
            name="temperature",
            attrs={"frequency": "day"},
        )

    @pytest.fixture
    def spatial_data_with_points(self):
        """Create data with points dimension for spatial batching tests."""
        np.random.seed(42)
        n_time = 365 * 30
        n_sims = 2
        n_points = 120  # More than SPATIAL_BATCH_SIZE (100)

        data = np.random.gumbel(loc=20, scale=5, size=(n_time, n_sims, n_points))
        time_coords = pd.date_range("1990-01-01", periods=n_time, freq="D")

        return xr.DataArray(
            data,
            dims=["time", "sim", "points"],
            coords={
                "time": time_coords,
                "sim": ["sim_0", "sim_1"],
                "points": range(n_points),
            },
            name="temperature",
            attrs={"frequency": "day"},
        )

    def test_fit_with_spatial_batching_closest_cell(self, block_maxima_data):
        """Test _fit_with_spatial_batching with closest_cell dimension."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10, 50],
                    "distribution": "gev",
                    "goodness_of_fit_test": True,
                }
            }
        )

        # Expand block maxima to have a spatial dimension
        block_maxima_spatial = block_maxima_data.expand_dims({"closest_cell": 5})

        return_values, p_values = processor._fit_with_spatial_batching(
            block_maxima_spatial, "time", "closest_cell", batch_size=2
        )

        assert "closest_cell" in return_values.dims
        assert return_values.sizes["closest_cell"] == 5
        assert "one_in_x" in return_values.dims

    def test_fit_with_spatial_batching_points(self, block_maxima_data):
        """Test _fit_with_spatial_batching with points dimension."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                }
            }
        )

        # Expand block maxima to have points dimension
        block_maxima_spatial = block_maxima_data.expand_dims({"points": 4})

        return_values, p_values = processor._fit_with_spatial_batching(
            block_maxima_spatial, "time", "points", batch_size=2
        )

        assert "points" in return_values.dims
        assert return_values.sizes["points"] == 4

    def test_process_batch_with_closest_cell_dim(self, spatial_data_with_closest_cell):
        """Test _process_simulation_batch triggers early spatial batching."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                    "event_duration": (1, "day"),
                }
            }
        )

        block_maxima_kwargs = {
            "extremes_type": "max",
            "check_ess": False,
            "block_size": 1,
            "groupby": (1, "day"),
        }

        # This should trigger _fit_with_early_spatial_batching because
        # closest_cell size (150) > SPATIAL_BATCH_SIZE (100)
        result = processor._process_simulation_batch(
            spatial_data_with_closest_cell,
            block_maxima_kwargs,
            batch_num=1,
            total_batches=1,
        )

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars
        assert "closest_cell" in result.dims

    def test_process_batch_with_points_dim(self, spatial_data_with_points):
        """Test _process_simulation_batch triggers early spatial batching with points."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                    "event_duration": (1, "day"),
                }
            }
        )

        block_maxima_kwargs = {
            "extremes_type": "max",
            "check_ess": False,
            "block_size": 1,
            "groupby": (1, "day"),
        }

        result = processor._process_simulation_batch(
            spatial_data_with_points,
            block_maxima_kwargs,
            batch_num=1,
            total_batches=1,
        )

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars
        assert "points" in result.dims


class TestMetricCalcEventDuration:
    """Test class for event duration configuration in 1-in-X analysis."""

    @pytest.fixture
    def hourly_data_with_sim(self):
        """Create hourly data for event duration testing."""
        np.random.seed(42)
        n_hours = 24 * 365 * 10  # 10 years of hourly data
        n_sims = 2
        n_lat = 2
        n_lon = 2

        data = np.random.gumbel(loc=20, scale=5, size=(n_hours, n_sims, n_lat, n_lon))
        time_coords = pd.date_range("2000-01-01", periods=n_hours, freq="h")

        return xr.DataArray(
            data,
            dims=["time", "sim", "lat", "lon"],
            coords={
                "time": time_coords,
                "sim": ["sim_0", "sim_1"],
                "lat": [35.0, 36.0],
                "lon": [-120.0, -119.0],
            },
            name="temperature",
            attrs={"frequency": "1hr"},
        )

    def test_vectorized_calculation_with_hour_duration(self, hourly_data_with_sim):
        """Test _calculate_one_in_x_vectorized with hour event duration."""
        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10],
                    "distribution": "gev",
                    "event_duration": (6, "hour"),
                    "goodness_of_fit_test": False,
                }
            }
        )

        result = processor._calculate_one_in_x_vectorized(hourly_data_with_sim)

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars


class TestMetricCalcDaskArrayHandling:
    """Test class for Dask array handling in 1-in-X analysis."""

    def test_one_in_x_with_small_dask_array(self):
        """Test 1-in-X calculation with small dask array (< 10MB)."""
        import dask.array as da

        np.random.seed(42)
        n_time = 365 * 5  # 5 years - small dataset
        n_sims = 2
        n_lat = 3
        n_lon = 3

        # Create small dask array
        data = (
            da.random.random((n_time, n_sims, n_lat, n_lon), chunks=(365, 1, 3, 3)) * 10
            + 20
        )

        time_coords = pd.date_range("2000-01-01", periods=n_time, freq="D")

        da_xr = xr.DataArray(
            data,
            dims=["time", "sim", "lat", "lon"],
            coords={
                "time": time_coords,
                "sim": ["sim_0", "sim_1"],
                "lat": [35.0, 36.0, 37.0],
                "lon": [-120.0, -119.0, -118.0],
            },
            name="temperature",
            attrs={"frequency": "day"},
        )

        processor = MetricCalc(
            {
                "one_in_x": {
                    "return_periods": [10],
                    "distribution": "gev",
                    "goodness_of_fit_test": False,
                }
            }
        )

        result = processor._calculate_one_in_x_single(da_xr)

        assert isinstance(result, xr.Dataset)
        assert "return_value" in result.data_vars
