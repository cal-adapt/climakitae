"""
Units tests for the `metric_calc` processor.

This processor calculates various statistical metrics (e.g., mean, median, percentiles)
over specified dimensions of the input data.

The tests validate the correct functionality of the processor, including handling of
different metrics, percentiles, dimensions, and edge cases.
"""

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

    def test_invalid_dims(self, test_da):
        """Test handling of invalid dimensions."""
        processor = MetricCalc(value={"metric": "mean", "dim": "invalid_dim"})
        with pytest.warns(
            UserWarning,
            match="None of the specified dimensions",
        ):
            retval = processor._calculate_metrics_single(test_da)
            assert retval is test_da

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

    def test_calculate_mean(self, test_da):
        """Test calculation of mean metric."""
        processor = MetricCalc(value={"metric": "mean", "dim": "time"})
        result = processor._calculate_metrics_single(test_da)
        expected = xr.DataArray(
            np.array([4.0, 5.0, 6.0]),
            dims=["lat"],
            coords={"lat": [10, 20, 30]},
        )
        xr.testing.assert_equal(result, expected)

    def test_calculate_median(self, test_da):
        """Test calculation of median metric."""
        processor = MetricCalc(value={"metric": "median", "dim": "time"})
        result = processor._calculate_metrics_single(test_da)
        expected = xr.DataArray(
            np.array([4.0, 5.0, 6.0]),
            dims=["lat"],
            coords={"lat": [10, 20, 30]},
        )
        xr.testing.assert_equal(result, expected)

    def test_calculate_min(self, test_da):
        """Test calculation of min metric."""
        processor = MetricCalc(value={"metric": "min", "dim": "time"})
        result = processor._calculate_metrics_single(test_da)
        expected = xr.DataArray(
            np.array([1, 2, 3]),
            dims=["lat"],
            coords={"lat": [10, 20, 30]},
        )
        xr.testing.assert_equal(result, expected)

    def test_calculate_max(self, test_da):
        """Test calculation of max metric."""
        processor = MetricCalc(value={"metric": "max", "dim": "time"})
        result = processor._calculate_metrics_single(test_da)
        expected = xr.DataArray(
            np.array([7, 8, 9]),
            dims=["lat"],
            coords={"lat": [10, 20, 30]},
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
        assert (
            context[_NEW_ATTRS_KEY]["metric_calc"]
            == "Process 'metric_calc' applied to the data. Percentiles [10, 50, 90] were calculated and Metric 'median' was calculated along dimension(s): ['time', 'lat']."
        )


class TestMetricCalcExecute:
    """Unit tests for the MetricCalc processor's execute method."""

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

    def test_execute_with_invalid_type(self):
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
    def test_execute_catch_invalid_results(self, _):
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
