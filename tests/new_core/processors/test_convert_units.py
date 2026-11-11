"""
Unit tests for climakitae/new_core/processors/convert_units.py

This module tests the ConvertUnits processor to ensure it correctly converts
data units as specified. It verifies that the processor modifies the data
array in the expected manner and handles edge cases appropriately.
"""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.processors.convert_units import (
    ConvertUnits,
    _handle_flux_to_precipitation,
    _handle_precipitation_to_flux,
)


@pytest.fixture
def sample_dataset():
    """Fixture providing a sample xr.Dataset for testing."""
    data = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.array([[0, 10, 20], [30, 40, 50]], dtype=float),
                dims=["lat", "lon"],
                coords={"lat": [0, 1], "lon": [0, 1, 2]},
                name="temperature",
                attrs={"units": "degC"},
            )
        }
    )
    yield data


@pytest.fixture
def sample_dataset2():
    """Fixture providing another sample xr.Dataset for testing."""
    data = xr.Dataset(
        {
            "precipitation": xr.DataArray(
                np.array([[0, 10, 20], [30, 40, 50]], dtype=float),
                dims=["lat", "lon"],
                coords={"lat": [0, 1], "lon": [0, 1, 2]},
                name="temperature",
                attrs={"units": "degF"},
            )
        }
    )
    yield data


@pytest.fixture
def processor():
    """Fixture providing a ConvertUnits processor instance for testing."""
    yield ConvertUnits(value="K")


class TestConvertUnitsProcessorInitialization:
    """Tests for initialization of ConvertUnits processor."""

    def test_initialization_with_valid_unit(self):
        """Test initialization with a valid target unit."""
        processor = ConvertUnits(value="K")
        assert processor.value == "K"
        assert processor.name == "convert_units"


class TestConvertUnitsConvertUnitsHelper:
    """Tests for the ConvertUnits processor's `_convert_units` functionality."""

    @pytest.mark.parametrize(
        "target_unit,expected_data",
        [
            ("K", np.array([[273.15, 283.15, 293.15], [303.15, 313.15, 323.15]])),
            ("degF", np.array([[32.0, 50.0, 68.0], [86.0, 104.0, 122.0]])),
            # ("mm", np.array([[0.0, 10.0, 20.0], [30.0, 40.0, 50.0]])),  # No conversion expected
        ],
    )
    def test_valid_conversion_str(
        self,
        sample_dataset,
        processor,
        target_unit,
        expected_data,
    ):
        """Test unit conversion to various target units."""
        converted_dataset = processor._convert_units(sample_dataset, target_unit)
        var = list(converted_dataset.data_vars.keys())[0]

        np.testing.assert_allclose(
            converted_dataset[var].data,
            expected_data,
            rtol=1e-5,
        )
        assert converted_dataset[var].attrs["units"] == target_unit

    @pytest.mark.parametrize(
        "target_units,warning_match",
        [
            (["K", "degF"], "The provided value is not the correct type"),
            (
                ("degF", "this should not be reached"),
                "The provided value is not the correct type",
            ),
            (
                ("invalid_units", "also invalid here"),
                "The provided value is not the correct type",
            ),
        ],
    )
    def test_invalid_conversion_list_tuple(
        self,
        sample_dataset,
        processor,
        target_units,
        warning_match,
    ):
        """Test unit conversion when target units are provided as a list or tuple (not supported)."""
        with pytest.warns(UserWarning, match=warning_match):
            converted_dataset = processor._convert_units(sample_dataset, target_units)
            assert processor.success is False
            var = list(converted_dataset.data_vars.keys())[0]
            # Data should be unchanged when conversion fails
            np.testing.assert_array_equal(
                converted_dataset[var].data,
                sample_dataset[var].data,
            )

    def test_no_conversion_needed(
        self,
        sample_dataset,
        processor,
    ):
        """Test that no conversion occurs when target unit matches source unit."""
        processor.value = "degC"  # Same as source unit
        converted_dataset = processor._convert_units(sample_dataset, processor.value)
        var = list(converted_dataset.data_vars.keys())[0]

        np.testing.assert_array_equal(
            converted_dataset[var].data,
            sample_dataset[var].data,
        )
        assert converted_dataset[var].attrs["units"] == "degC"

    def test_no_units_in_data(
        self,
        sample_dataset,
        processor,
    ):
        """Test that missing units in data raises a warning and leaves data unchanged."""
        var = list(sample_dataset.data_vars.keys())[0]
        del sample_dataset[var].attrs["units"]  # Remove units

        with pytest.warns(
            UserWarning, match="This variable does not have identifiable native units"
        ):
            converted_dataset = processor._convert_units(sample_dataset, "K")

        assert processor.success is False
        np.testing.assert_array_equal(
            converted_dataset[var].data,
            sample_dataset[var].data,
        )
        assert "units" not in converted_dataset[var].attrs

    def test_unsupported_conversion(
        self,
        sample_dataset,
        processor,
    ):
        """Test that unsupported unit conversions raise a ValueError."""
        with pytest.warns(
            UserWarning,
            match="The selected units 'unknown_unit' are not valid for 'degC'",
        ):
            processor._convert_units(sample_dataset, "unknown_unit")
            assert processor.success is False

    def test_incorrect_value_type(
        self,
        sample_dataset,
        processor,
    ):
        """Test that incorrect value types raise a TypeError."""
        with pytest.warns(
            UserWarning, match="The provided value is not the correct type."
        ):
            processor._convert_units(sample_dataset, 123)
            assert processor.success is False

    def test_no_valid_units(
        self,
        sample_dataset,
        processor,
    ):
        """Test that invalid units in data raise a warning and leave data unchanged."""
        var = list(sample_dataset.data_vars.keys())[0]
        sample_dataset[var].attrs["units"] = "invalid_unit"

        with pytest.warns(
            UserWarning,
            match="No valid unit conversions implemented for 'invalid_unit'",
        ):
            converted_dataset = processor._convert_units(sample_dataset, "K")

        assert processor.success is False
        np.testing.assert_array_equal(
            converted_dataset[var].data,
            sample_dataset[var].data,
        )
        assert converted_dataset[var].attrs["units"] == "invalid_unit"


class TestConvertUnitsExecute:
    """Tests for the ConvertUnits processor's `execute` method."""

    def test_execute_no_conversion_when_unset(
        self,
        sample_dataset,
    ):
        """Test the execute method leaves data unchanged when value is UNSET."""
        processor = ConvertUnits(value=UNSET)
        converted_dataset = processor.execute(sample_dataset, context={})
        var = list(converted_dataset.data_vars.keys())[0]

        np.testing.assert_array_equal(
            converted_dataset[var].data,
            sample_dataset[var].data,
        )
        assert converted_dataset[var].attrs["units"] == "degC"

    @pytest.mark.parametrize("data_type", [dict, list, tuple])
    def test_execute_with_different_result_types(
        self, sample_dataset, sample_dataset2, processor, data_type
    ):
        """Test the execute method performs unit conversion correct with different data types."""
        if data_type is dict:
            result = {"key": sample_dataset}
        else:
            result = data_type([sample_dataset, sample_dataset2])
        converted_dataset = processor.execute(result, context={})

        if data_type is dict:
            var = list(converted_dataset["key"].data_vars.keys())[0]
            expected_data = np.array(
                [[273.15, 283.15, 293.15], [303.15, 313.15, 323.15]]
            )
            np.testing.assert_allclose(
                converted_dataset["key"][var].data,
                expected_data,
                rtol=1e-5,
            )
            assert converted_dataset["key"][var].attrs["units"] == "K"
        else:
            for i, ds in enumerate(converted_dataset):
                var = list(ds.data_vars.keys())[0]
                expected_data = (
                    np.array([[273.15, 283.15, 293.15], [303.15, 313.15, 323.15]])
                    if i == 0
                    else np.array(
                        [[255.372, 260.928, 266.484], [272.04, 277.596, 283.152]]
                    )
                )
                np.testing.assert_allclose(
                    ds[var].data,
                    expected_data,
                    rtol=1e-5,
                )
                assert ds[var].attrs["units"] == "K"

    def test_execute_conversion(
        self,
        sample_dataset,
        processor,
    ):
        """Test the execute method performs unit conversion correctly."""
        converted_dataset = processor.execute(sample_dataset, context={})
        var = list(converted_dataset.data_vars.keys())[0]

        expected_data = np.array([[273.15, 283.15, 293.15], [303.15, 313.15, 323.15]])
        np.testing.assert_allclose(
            converted_dataset[var].data,
            expected_data,
            rtol=1e-5,
        )
        assert converted_dataset[var].attrs["units"] == "K"

    def test_execute_shared_target_converts_all_compatible_variables(self):
        """Test a shared target is applied to every compatible measurement field."""
        dataset = xr.Dataset(
            {
                "tasmax": ("time", [300.0]),
                "tasmin": ("time", [50.0]),
            },
            coords={"time": pd.date_range("2020-01-01", periods=1)},
        )
        dataset["tasmax"].attrs["units"] = "K"
        dataset["tasmin"].attrs["units"] = "degF"

        result = ConvertUnits("degC").execute(dataset, context={})

        np.testing.assert_allclose(result["tasmax"].values, [26.85])
        np.testing.assert_allclose(result["tasmin"].values, [10.0])
        assert result["tasmax"].attrs["units"] == "degC"
        assert result["tasmin"].attrs["units"] == "degC"

    def test_execute_by_variable_preserves_omitted_and_crs_variables(self):
        """Test targeted conversion leaves omitted measurements and CRS fields intact."""
        dataset = xr.Dataset(
            {
                "tasmax": ("time", [300.0]),
                "pr": ("time", [2.0]),
                "spatial_ref": xr.DataArray(0, attrs={"grid_mapping_name": "crs"}),
            },
            coords={"time": pd.date_range("2020-01-01", periods=1)},
        )
        dataset["tasmax"].attrs.update({"units": "K", "grid_mapping": "spatial_ref"})
        dataset["pr"].attrs["units"] = "mm/d"
        dataset["spatial_ref"].attrs["units"] = "K"

        result = ConvertUnits({"tasmax": "degC"}).execute(
            dataset,
            context={"variable_id": ["tasmax", "pr"]},
        )

        np.testing.assert_allclose(result["tasmax"].values, [26.85])
        np.testing.assert_array_equal(result["pr"].values, [2.0])
        assert result["tasmax"].attrs["units"] == "degC"
        assert result["pr"].attrs["units"] == "mm/d"
        assert result["spatial_ref"].attrs["units"] == "K"
        assert result["tasmax"].attrs["grid_mapping"] == "spatial_ref"

    def test_execute_shared_incompatible_target_is_atomic(self):
        """Test incompatible shared targets leave all measurement fields unchanged."""
        dataset = xr.Dataset(
            {
                "tasmax": ("time", [300.0]),
                "pr": ("time", [2.0]),
            },
            coords={"time": pd.date_range("2020-01-01", periods=1)},
        )
        dataset["tasmax"].attrs["units"] = "K"
        dataset["pr"].attrs["units"] = "mm/d"
        context = {}

        with pytest.warns(UserWarning, match="no variables were converted"):
            result = ConvertUnits("degC").execute(dataset, context)

        assert result is dataset
        np.testing.assert_array_equal(result["tasmax"].values, [300.0])
        np.testing.assert_array_equal(result["pr"].values, [2.0])
        assert result["tasmax"].attrs["units"] == "K"
        assert result["pr"].attrs["units"] == "mm/d"
        assert "_new_attrs" not in context

    @pytest.mark.parametrize(
        "native_units,target_unit,warning_match",
        [
            (None, "degC", "identifiable native units"),
            ("unknown", "degC", "No valid unit conversions implemented"),
        ],
    )
    def test_execute_invalid_targeted_conversion_is_atomic(
        self, native_units, target_unit, warning_match
    ):
        """Leave all fields unchanged if any targeted variable cannot convert."""
        dataset = xr.Dataset(
            {
                "tasmax": ("time", [300.0]),
                "pr": ("time", [2.0]),
            },
            coords={"time": pd.date_range("2020-01-01", periods=1)},
        )
        dataset["tasmax"].attrs["units"] = "K"
        if native_units is not None:
            dataset["pr"].attrs["units"] = native_units

        with pytest.warns(UserWarning, match=warning_match):
            result = ConvertUnits({"tasmax": "degC", "pr": target_unit}).execute(
                dataset, context={}
            )

        assert result is dataset
        np.testing.assert_array_equal(result["tasmax"].values, [300.0])
        np.testing.assert_array_equal(result["pr"].values, [2.0])
        assert result["tasmax"].attrs["units"] == "K"
        assert result["pr"].attrs.get("units") == native_units

    def test_execute_by_variable_preserves_monthly_precipitation_conversion(self):
        """Test targeted flux conversion uses monthly calendar-day scaling."""
        dataset = xr.Dataset(
            {
                "precipitation": ("time", [1.0, 1.0]),
                "tasmax": ("time", [300.0, 301.0]),
            },
            coords={"time": pd.to_datetime(["2020-01-01", "2020-02-01"])},
            attrs={"frequency": "mon"},
        )
        dataset["precipitation"].attrs["units"] = "kg m-2 s-1"
        dataset["tasmax"].attrs["units"] = "K"

        result = ConvertUnits({"precipitation": "mm"}).execute(dataset, context={})

        np.testing.assert_allclose(
            result["precipitation"].values, np.array([31, 29]) * 86400
        )
        np.testing.assert_array_equal(result["tasmax"].values, [300.0, 301.0])
        assert result["precipitation"].attrs["units"] == "mm"
        assert result["tasmax"].attrs["units"] == "K"

    def test_execute_keeps_multivariable_conversions_lazy(self):
        """Test unit conversion keeps Dask-backed measurement arrays lazy."""
        dataset = xr.Dataset(
            {
                "tasmax": (
                    "time",
                    da.from_array(np.array([300.0, 301.0]), chunks=1),
                ),
                "tasmin": (
                    "time",
                    da.from_array(np.array([280.0, 281.0]), chunks=1),
                ),
            },
            coords={"time": pd.date_range("2020-01-01", periods=2)},
        )
        dataset["tasmax"].attrs["units"] = "K"
        dataset["tasmin"].attrs["units"] = "K"

        result = ConvertUnits("degC").execute(dataset, context={})

        assert isinstance(result["tasmax"].data, da.Array)
        assert isinstance(result["tasmin"].data, da.Array)
        np.testing.assert_allclose(result["tasmax"].compute().values, [26.85, 27.85])
        np.testing.assert_allclose(result["tasmin"].compute().values, [6.85, 7.85])

    def test_execute_converts_dataarray(self):
        """Test unit conversion handles a named DataArray directly."""
        data = xr.DataArray(
            [300.0],
            dims=["time"],
            coords={"time": pd.date_range("2020-01-01", periods=1)},
            name="tasmax",
            attrs={"units": "K", "standard_name": "air_temperature"},
        )

        result = ConvertUnits("degC").execute(data, context={})

        np.testing.assert_allclose(result.values, [26.85])
        assert result.attrs["units"] == "degC"
        assert result.attrs["standard_name"] == "air_temperature"

    def test_execute_shared_target_skips_unselected_unit_bearing_variables(self):
        """Test selected climate fields convert without touching support fields."""
        dataset = xr.Dataset(
            {
                "tasmax": ("time", [300.0]),
                "tasmin": ("time", [280.0]),
                "elevation": ("time", [1000.0]),
            },
            coords={"time": pd.date_range("2020-01-01", periods=1)},
        )
        dataset["tasmax"].attrs["units"] = "K"
        dataset["tasmin"].attrs["units"] = "K"
        dataset["elevation"].attrs["units"] = "m"

        result = ConvertUnits("degC").execute(
            dataset,
            context={"variable_id": ["tasmax", "tasmin"]},
        )

        np.testing.assert_allclose(result["tasmax"].values, [26.85])
        np.testing.assert_allclose(result["tasmin"].values, [6.85])
        np.testing.assert_array_equal(result["elevation"].values, [1000.0])
        assert result["elevation"].attrs["units"] == "m"


class TestHelperConversionFunctions:
    """Tests for helper conversion functions within ConvertUnits processor."""

    def test_precipitation_to_flux(self):
        """Test conversion from precipitation units to flux units."""
        precip_data = xr.DataArray(
            np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype=float),
            dims=["time"],
            coords={"time": pd.date_range("2000-01-01", periods=12, freq="ME")},
            name="precipitation",
            attrs={"units": "mm/d"},
        )
        expected_flux_data = np.array(
            precip_data / (86400 * precip_data["time"].dt.days_in_month)
        )
        flux_data = _handle_precipitation_to_flux(precip_data, frequency="mon")
        np.testing.assert_allclose(
            flux_data.data,
            expected_flux_data,
            rtol=1e-5,
        )

    def test_flux_to_precipitation(self):
        """Test conversion from flux units to precipitation units."""
        flux_data = xr.DataArray(
            np.array(
                [
                    0,
                    0.00011574,
                    0.00023148,
                    0.00034722,
                    0.00046296,
                    0.0005787,
                    0.00069444,
                    0.00081018,
                    0.00092592,
                    0.00104166,
                    0.0011574,
                    0.00127314,
                ],
                dtype=float,
            ),
            dims=["time"],
            coords={"time": pd.date_range("2000-01-01", periods=12, freq="ME")},
            name="flux",
            attrs={"units": "kg m-2 s-1"},
        )
        expected_precip_data = np.array(
            flux_data * (86400 * flux_data["time"].dt.days_in_month)
        )

        precip_data = _handle_flux_to_precipitation(flux_data, frequency="mon")
        np.testing.assert_allclose(
            precip_data.data,
            expected_precip_data,
            rtol=1e-5,
        )
