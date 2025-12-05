"""
Units tests for the `metric_calc` processor.

This processor calculates various statistical metrics (e.g., mean, median, percentiles)
over specified dimensions of the input data.

The tests validate the correct functionality of the processor, including handling of
different metrics, percentiles, dimensions, and edge cases.
"""

import logging
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.processors.metric_calc import MetricCalc


@pytest.fixture
def test_da():
    """Set up a sample xarray DataArray for testing."""
    data = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        dims=["time", "lat"],
        coords={"time": [0, 1, 2], "lat": [10, 20, 30]},
        name="t2max",
    )
    yield data


@pytest.fixture
def test_da_with_nan():
    """Set up a sample xarray DataArray with NaN values for testing."""
    data = xr.DataArray(
        np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]]),
        dims=["time", "lat"],
        coords={"time": [0, 1, 2], "lat": [10, 20, 30]},
        name="t2max",
    )
    yield data


@pytest.fixture
def test_ds():
    """Set up a sample xarray Dataset for testing."""
    data = xr.Dataset(
        {
            "t2max": xr.DataArray(
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                dims=["time", "lat"],
                coords={"time": [0, 1, 2], "lat": [10, 20, 30]},
            )
        }
    )
    yield data


class TestMetricCalcProcessorInit:
    """Unit tests for the MetricCalc processor initialization."""

    def test_default_initialization(self):
        """Test MetricCalc initialization with default parameters."""
        processor = MetricCalc({})
        assert processor.metric == "mean"
        assert processor.percentiles is UNSET
        assert processor.percentiles_only is False
        assert processor.dim == "time"
        assert processor.skipna is True

    def test_custom_initialization(self):
        """Test MetricCalc initialization with custom parameters."""
        value = {
            "metric": "median",
            "percentiles": [10, 50, 90],
            "percentiles_only": True,
            "dim": ["time", "lat"],
            "skipna": False,
        }
        processor = MetricCalc(value=value)
        assert processor.metric == value["metric"]
        assert processor.percentiles == value["percentiles"]
        assert processor.percentiles_only == value["percentiles_only"]
        assert processor.dim == value["dim"]
        assert processor.skipna == value["skipna"]


class TestMetricCalcCalculateMetricsSingle:
    """Unit tests for the MetricCalc processor's calculate_metrics method (single metric)."""

    def test_invalid_dims(self, test_da, caplog):
        """Test handling of invalid dimensions."""
        processor = MetricCalc(value={"metric": "mean", "dim": "invalid_dim"})
        with caplog.at_level(logging.WARNING):
            retval = processor._calculate_metrics_single(test_da)
            assert retval is test_da
            assert "None of the specified dimensions" in caplog.text

    def test_invalid_dim_with_valid_dim(self, test_da):
        """Test handling of partially invalid dimensions."""
        processor = MetricCalc(value={"metric": "mean", "dim": ["time", "invalid_dim"]})
        result = processor._calculate_metrics_single(test_da)
        expected = xr.DataArray(
            np.array([4.0, 5.0, 6.0]),
            dims=["lat"],
            coords={"lat": [10, 20, 30]},
            name="t2max",
        )
        xr.testing.assert_equal(result, expected)
        assert "invalid_dim" not in result.dims

    def test_invalid_dim_with_multiple_valid_dims(self, test_da):
        """Test handling of multiple valid dimensions with an invalid one."""
        processor = MetricCalc(
            value={"metric": "mean", "dim": ["time", "lat", "invalid_dim"]}
        )
        result = processor._calculate_metrics_single(test_da)
        expected = xr.DataArray(
            np.array(5.0),
            dims=[],
            name="t2max",
        )
        xr.testing.assert_equal(result, expected)
        assert "invalid_dim" not in result.dims

    def test_percentiles(self, test_da):
        """Test calculation of percentiles."""
        processor = MetricCalc(
            value={
                "percentiles": [25, 50, 75],
                "percentiles_only": True,
                "dim": "time",
            }
        )
        result = processor._calculate_metrics_single(test_da)
        expected = xr.DataArray(
            np.array([[2.5, 3.5, 4.5], [4.0, 5.0, 6.0], [5.5, 6.5, 7.5]]),
            dims=["percentile", "lat"],
            coords={"percentile": [25, 50, 75], "lat": [10, 20, 30]},
        )
        xr.testing.assert_equal(result, expected)

    def test_percentiles_only_true(self, test_da):
        """Test calculation of percentiles with percentiles_only=True."""
        processor = MetricCalc(
            value={
                "percentiles": [0, 50, 100],
                "percentiles_only": True,
                "dim": "time",
            }
        )
        result = processor._calculate_metrics_single(test_da)
        expected = xr.DataArray(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            dims=["percentile", "lat"],
            coords={"percentile": [0, 50, 100], "lat": [10, 20, 30]},
        )
        xr.testing.assert_equal(result, expected)

    def test_percentiles_only_false_with_ds(self, test_ds):
        """Test calculation of percentiles with percentiles_only=False."""
        processor = MetricCalc(
            value={
                "metric": "mean",
                "percentiles": [0, 25, 60, 100],
                "percentiles_only": False,
                "dim": "time",
            }
        )
        result = processor._calculate_metrics_single(test_ds)
        expected = xr.Dataset(
            {
                "t2max_p0": xr.DataArray(
                    np.array([1.0, 2.0, 3.0]),
                    dims=["lat"],
                    coords={"lat": [10, 20, 30]},
                ),
                "t2max_p25": xr.DataArray(
                    np.array([2.5, 3.5, 4.5]),
                    dims=["lat"],
                    coords={"lat": [10, 20, 30]},
                ),
                "t2max_p60": xr.DataArray(
                    np.array([4.6, 5.6, 6.6]),
                    dims=["lat"],
                    coords={"lat": [10, 20, 30]},
                ),
                "t2max_p100": xr.DataArray(
                    np.array([7.0, 8.0, 9.0]),
                    dims=["lat"],
                    coords={"lat": [10, 20, 30]},
                ),
                "t2max_mean": xr.DataArray(
                    np.array([4.0, 5.0, 6.0]),
                    dims=["lat"],
                    coords={"lat": [10, 20, 30]},
                ),
            }
        )
        # Assert type of result
        assert isinstance(result, xr.Dataset)
        # Assert equality of datasets
        xr.testing.assert_equal(result, expected)

    def test_percentiles_only_false_with_da(self, test_da):
        """Test calculation of percentiles with percentiles_only=False."""
        processor = MetricCalc(
            value={
                "metric": "mean",
                "percentiles": [0, 25, 60, 100],
                "percentiles_only": False,
                "dim": "time",
            }
        )
        result = processor._calculate_metrics_single(test_da)
        expected = xr.DataArray(
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.5, 3.5, 4.5],
                    [4.6, 5.6, 6.6],
                    [7.0, 8.0, 9.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
            dims=["statistic", "lat"],
            coords={
                "statistic": ["p0", "p25", "p60", "p100", "mean"],
                "lat": [10, 20, 30],
            },
        )
        # Assert type of result
        assert isinstance(result, xr.DataArray)
        # Assert equality of datasets
        xr.testing.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "metric,expected",
        [
            ("min", np.array([1, 2, 3])),
            ("max", np.array([7, 8, 9])),
            ("mean", np.array([4.0, 5.0, 6.0])),
            ("median", np.array([4.0, 5.0, 6.0])),
        ],
    )
    def test_calculate_metric(self, test_da, metric, expected):
        """Test calculation of various metrics."""
        processor = MetricCalc(value={"metric": metric, "dim": "time"})
        result = processor._calculate_metrics_single(test_da)
        expected_da = xr.DataArray(expected, dims=["lat"], coords={"lat": [10, 20, 30]})
        xr.testing.assert_equal(result, expected_da)

    def test_calculate_metric_skipna(self, test_da_with_nan):
        """Test calculation of metrics with skipna parameter."""
        processor = MetricCalc(value={"metric": "mean", "dim": "time", "skipna": True})
        result = processor._calculate_metrics_single(test_da_with_nan)
        expected = xr.DataArray(
            np.array([4.0, 5.0, 7.5]),
            dims=["lat"],
            coords={"lat": [10, 20, 30]},
            name="t2max",
        )
        xr.testing.assert_equal(result, expected)

    def test_calculate_metric_no_skipna(self, test_da_with_nan):
        """Test calculation of metrics with skipna=False parameter."""
        processor = MetricCalc(value={"metric": "mean", "dim": "time", "skipna": False})
        result = processor._calculate_metrics_single(test_da_with_nan)
        expected = xr.DataArray(
            np.array([4.0, np.nan, np.nan]),
            dims=["lat"],
            coords={"lat": [10, 20, 30]},
            name="t2max",
        )
        xr.testing.assert_equal(result, expected)


class TestMetricUpdateContext:
    """Unit tests for the MetricCalc processor's update_context method."""

    def test_update_context(self):
        """Test that update_context correctly updates the context dictionary."""
        processor = MetricCalc(
            value={
                "metric": "median",
                "percentiles": [10, 50, 90],
                "percentiles_only": False,
                "dim": ["time", "lat"],
                "skipna": False,
            }
        )
        context: Dict[str, Any] = {}
        processor.update_context(context)
        assert "metric_calc" in context[_NEW_ATTRS_KEY]
        assert "Percentiles [10, 50, 90]" in context[_NEW_ATTRS_KEY]["metric_calc"]
        assert "median" in context[_NEW_ATTRS_KEY]["metric_calc"]


class TestMetricCalcExecute:
    """Unit tests for the MetricCalc processor's execute method."""

    @pytest.mark.parametrize("input_data", [[], (), {}])
    def test_execute_on_empty_iterable(self, input_data):
        """Test the execute method of MetricCalc processor with an empty iterable input."""
        processor = MetricCalc(
            value={
                "metric": "mean",
                "dim": "time",
            }
        )
        with pytest.raises(
            ValueError,
            match="Metric calculation operation failed to produce valid results on empty arguments.",
        ):
            result = processor.execute(input_data, context={})

    def test_execute_da(self, test_da):
        """Test the execute method of MetricCalc processor with a DataArray input."""
        processor = MetricCalc(
            value={
                "metric": "mean",
                "dim": "time",
            }
        )
        result = processor.execute(test_da, context={})
        expected = xr.DataArray(
            np.array([4.0, 5.0, 6.0]),
            dims=["lat"],
            coords={"lat": [10, 20, 30]},
            name="t2max",
        )
        xr.testing.assert_equal(result, expected)

    def test_execute_ds(self, test_ds):
        """Test the execute method of MetricCalc processor with a Dataset input."""
        processor = MetricCalc(
            value={
                "metric": "max",
                "dim": "time",
            }
        )
        result = processor.execute(test_ds, context={})
        expected = xr.Dataset(
            {
                "t2max": xr.DataArray(
                    np.array([7, 8, 9]),
                    dims=["lat"],
                    coords={"lat": [10, 20, 30]},
                )
            }
        )
        xr.testing.assert_equal(result, expected)

    def test_execute_with_list_of_da(self, test_da):
        """Test the execute method of MetricCalc processor with a list of DataArray inputs."""
        processor = MetricCalc(
            value={
                "metric": "min",
                "dim": "time",
            }
        )
        result = processor.execute([test_da, test_da], context={})
        expected = [
            xr.DataArray(
                np.array([1, 2, 3]),
                dims=["lat"],
                coords={"lat": [10, 20, 30]},
                name="t2max",
            ),
            xr.DataArray(
                np.array([1, 2, 3]),
                dims=["lat"],
                coords={"lat": [10, 20, 30]},
                name="t2max",
            ),
        ]
        for res, exp in zip(result, expected):
            xr.testing.assert_equal(res, exp)

    def test_execute_with_invalid_type(self, caplog):
        """Test the execute method of MetricCalc processor with an invalid input type."""
        processor = MetricCalc(
            value={
                "metric": "mean",
                "dim": "time",
            }
        )
        with pytest.raises(
            TypeError, match="Expected xr.Dataset, xr.DataArray, dict, list, or tuple."
        ):
            processor.execute(result="invalid_input_type", context={})

    @patch(
        "climakitae.new_core.processors.metric_calc.MetricCalc._calculate_metrics_single",
        return_value=None,
    )
    def test_execute_catch_invalid_results(self, retval, caplog):
        """Test that execute method raises an error for invalid results from _calculate_metrics_single."""
        processor = MetricCalc(
            value={
                "metric": "mean",
                "dim": "time",
            }
        )
        with pytest.raises(
            ValueError,
            match="Metric calculation operation failed to produce valid results.",
        ):
            processor.execute(result=xr.DataArray(np.array([1, 2, 3])), context={})
