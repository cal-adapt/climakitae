"""
Unit tests for climakitae/new_core/processors/concatenate.py.

This module contains comprehensive unit tests for the Concat processor class
that concatenates multiple datasets along a specified dimension, with special
handling for time dimension concatenation and member_id processing.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.new_core.processors.concatenate import Concat


class TestConcatInitialization:
    """Test class for Concat initialization."""

    def setup_method(self):
        """Set up test fixtures."""

    def test_init_default_dimension(self):
        """Test initialization with default dimension."""
        processor = Concat()
        assert processor.dim_name == "time"
        assert processor._original_dim_name == "time"
        assert processor.name == "concat"
        assert processor.catalog is None
        assert processor.needs_catalog is True

    def test_init_custom_string_dimension(self):
        """Test initialization with custom string dimension."""
        processor = Concat("simulation")
        assert processor.dim_name == "simulation"
        assert processor._original_dim_name == "simulation"

    @pytest.mark.parametrize(
        "invalid_input,expected",
        [
            (123, "time"),
            ([], "time"),
            ({}, "time"),
            (None, "time"),
            (True, "time"),
        ],
    )
    def test_init_invalid_types_fallback_to_time(self, invalid_input, expected):
        """Test initialization with invalid types falls back to time."""
        processor = Concat(invalid_input)
        assert processor.dim_name == expected

    def test_init_empty_string_fallback_to_time(self):
        """Test initialization with empty string."""
        processor = Concat("")
        assert processor.dim_name == ""  # Empty string is still a string


class TestConcatDataAccessor:
    """Test class for data accessor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat()
        self.mock_catalog = MagicMock()

    def test_set_data_accessor_valid_catalog(self):
        """Test setting data accessor with valid catalog."""
        self.processor.set_data_accessor(self.mock_catalog)
        assert self.processor.catalog is self.mock_catalog

    def test_set_data_accessor_none(self):
        """Test setting data accessor with None."""
        self.processor.set_data_accessor(None)  # type: ignore
        assert self.processor.catalog is None


class TestConcatExecuteBasicCases:
    """Test class for basic execute method functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat("sim")
        self.context = {}

    def test_execute_single_dataset_passthrough(self):
        """Test execute with single xarray Dataset passes through unchanged."""
        dataset = xr.Dataset({"temp": (["time"], [1, 2, 3])})
        result = self.processor.execute(dataset, self.context)
        assert result is dataset

    def test_execute_single_dataarray_passthrough(self):
        """Test execute with single xarray DataArray passes through unchanged."""
        dataarray = xr.DataArray([1, 2, 3], dims=["time"])
        result = self.processor.execute(dataarray, self.context)
        assert result is dataarray

    def test_execute_empty_dict_raises_error(self):
        """Test execute with empty dictionary raises ValueError."""
        with pytest.raises(
            ValueError, match="No valid datasets found for concatenation"
        ):
            self.processor.execute({}, self.context)

    def test_execute_empty_list_raises_error(self):
        """Test execute with empty list raises ValueError."""
        with pytest.raises(
            ValueError, match="No valid datasets found for concatenation"
        ):
            self.processor.execute([], self.context)

    def test_execute_dict_with_non_xarray_objects_raises_error(self):
        """Test execute with dict containing no valid xarray objects raises ValueError."""
        invalid_data = {"key1": "not_xarray", "key2": 123}
        with pytest.raises(
            ValueError, match="No valid datasets found for concatenation"
        ):
            self.processor.execute(invalid_data, self.context)


class TestConcatExecuteAttributeHandling:
    """Test class for attribute handling in execute method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat("sim")
        self.context = {}

    def create_mock_dataset(self, attrs, dims=None, expand_dims_return=None):
        """Create a mock dataset with specified attributes and dimensions."""
        dataset = MagicMock(spec=xr.Dataset)
        dataset.attrs = attrs
        dataset.dims = dims or {}
        dataset.expand_dims = MagicMock(return_value=expand_dims_return or dataset)
        return dataset

    def test_execute_dict_with_standard_attributes(self):
        """Test execute with dictionary containing datasets with standard intake attributes."""
        dataset1 = self.create_mock_dataset(
            {
                "intake_esm_attrs:activity_id": "LOCA2",
                "intake_esm_attrs:institution_id": "CNRM",
                "intake_esm_attrs:source_id": "CNRM-CM6-1",
                "intake_esm_attrs:experiment_id": "ssp245",
                "intake_esm_attrs:member_id": "r1i1p1f2",
            }
        )

        dataset2 = self.create_mock_dataset(
            {
                "intake_esm_attrs:activity_id": "LOCA2",
                "intake_esm_attrs:institution_id": "GFDL",
                "intake_esm_attrs:source_id": "GFDL-ESM4",
                "intake_esm_attrs:experiment_id": "ssp245",
                "intake_esm_attrs:member_id": "r1i1p1f1",
            }
        )

        result_dict = {"cnrm": dataset1, "gfdl": dataset2}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated) as mock_concat:
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        assert result is concatenated
        mock_concat.assert_called_once()

    def test_execute_dict_with_missing_attributes(self):
        """Test execute with datasets missing some attributes uses 'unknown'."""
        dataset1 = self.create_mock_dataset(
            {
                "intake_esm_attrs:source_id": "CNRM-CM6-1",
                # Missing other attributes
            }
        )

        result_dict = {"test": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        assert result is concatenated

    def test_execute_dict_with_renewable_energy_catalog_attributes(self):
        """Test execute with renewable energy catalog uses different attribute keys."""
        self.processor.catalog = MagicMock()
        self.processor.catalog.catalog_key = "renewable energy generation"

        dataset1 = self.create_mock_dataset(
            {
                "installation": "wind_farm_1",
                "institution_id": "NREL",
                "source_id": "WRF",
                "experiment_id": "historical",
                "member_id": "run1",
            }
        )

        result_dict = {"wind": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        assert result is concatenated

    def test_execute_dict_with_spaces_in_attributes(self):
        """Test execute handles attributes with spaces correctly."""
        dataset1 = self.create_mock_dataset(
            {
                "intake_esm_attrs:source_id": "Model Name With Spaces",
                "intake_esm_attrs:institution_id": "Institution Name",
            }
        )

        result_dict = {"test": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        assert result is concatenated


class TestConcatExecuteMemberIdHandling:
    """Test class for member_id dimension handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat("sim")
        self.context = {}

    def create_mock_dataset_with_member_id(self, attrs, member_ids):
        """Create a mock dataset with member_id dimension."""
        dataset = MagicMock(spec=xr.Dataset)
        dataset.attrs = attrs
        dataset.dims = {"member_id": len(member_ids), "time": 10}
        dataset.member_id = MagicMock()
        dataset.member_id.values = member_ids

        # Mock sel and drop_vars methods for member processing
        def mock_sel(member_id):
            _ = member_id  # Mark as intentionally unused
            selected = MagicMock(spec=xr.Dataset)
            selected.drop_vars = MagicMock(return_value=selected)
            selected.expand_dims = MagicMock(return_value=selected)
            return selected

        dataset.sel = MagicMock(side_effect=mock_sel)
        return dataset

    def test_execute_dict_with_member_id_dimension(self):
        """Test execute handles member_id dimension correctly."""
        dataset1 = self.create_mock_dataset_with_member_id(
            {
                "intake_esm_attrs:source_id": "CNRM-CM6-1",
                "intake_esm_attrs:experiment_id": "ssp245",
            },
            ["r1i1p1f1", "r2i1p1f1"],
        )

        result_dict = {"cnrm": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        # Should call sel for each member
        assert dataset1.sel.call_count == 2
        assert result is concatenated

    def test_execute_dict_with_member_id_already_in_attr_id(self):
        """Test execute when member_id is already part of the attribute ID."""
        dataset1 = self.create_mock_dataset_with_member_id(
            {
                "intake_esm_attrs:source_id": "CNRM-CM6-1",
                "intake_esm_attrs:member_id": "r1i1p1f1",  # This will be in attr_id
            },
            ["r1i1p1f1"],
        )

        result_dict = {"cnrm": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        assert result is concatenated

    def test_execute_list_with_member_id_dimension(self):
        """Test execute handles member_id dimension in list input."""
        dataset1 = self.create_mock_dataset_with_member_id(
            {
                "intake_esm_attrs:source_id": "CNRM-CM6-1",
            },
            ["r1i1p1f1"],
        )

        result_list = [dataset1]
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_list, self.context)

        assert result is concatenated

    def test_execute_dict_with_key_containing_member_id(self):
        """Test execute handles keys that contain member_id information."""
        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {"intake_esm_attrs:source_id": "CNRM-CM6-1"}
        dataset1.dims = {}  # No member_id dimension
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        # Key ending with member ID pattern
        result_dict = {"model.run.r1i1p1f1": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        # Should use the key with dots replaced by underscores
        dataset1.expand_dims.assert_called_once()
        assert result is concatenated


class TestConcatExecuteTimeDimension:
    """Test class for time dimension special handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat("time")  # Start with time dimension
        self.context = {}

    def test_execute_time_dimension_calls_extend_time_domain(self):
        """Test execute with time dimension calls extend_time_domain."""
        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {"intake_esm_attrs:source_id": "CNRM-CM6-1"}
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {"ssp245": dataset1}
        extended_dict = {"ssp245_extended": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch(
            "climakitae.new_core.processors.concatenate.extend_time_domain",
            return_value=extended_dict,
        ) as mock_extend:
            with patch("xarray.concat", return_value=concatenated):
                result = self.processor.execute(result_dict, self.context)  # type: ignore

        # Should call extend_time_domain
        mock_extend.assert_called_once_with(result_dict)
        # Dimension should change to "sim" after time processing
        assert self.processor.dim_name == "sim"
        # Original dimension should still be tracked
        assert self.processor._original_dim_name == "time"
        assert result is concatenated

    def test_execute_non_time_dimension_skips_extend_time_domain(self):
        """Test execute with non-time dimension skips extend_time_domain."""
        processor = Concat("sim")  # Non-time dimension
        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {"intake_esm_attrs:source_id": "CNRM-CM6-1"}
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {"model1": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch(
            "climakitae.new_core.processors.concatenate.extend_time_domain"
        ) as mock_extend:
            with patch("xarray.concat", return_value=concatenated):
                result = processor.execute(result_dict, self.context)  # type: ignore

        # Should NOT call extend_time_domain
        mock_extend.assert_not_called()
        # Dimension should remain unchanged
        assert processor.dim_name == "sim"
        assert result is concatenated


class TestConcatExecuteResolutionHandling:
    """Test class for resolution attribute handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat("sim")
        self.context = {}

    @pytest.mark.parametrize(
        "key,expected_resolution",
        [
            ("model.d01.experiment", "45 km"),
            ("model.d02.experiment", "9 km"),
            ("model.d03.experiment", "3 km"),
            ("model.d04.experiment", None),  # No resolution for d04
            ("model.experiment", None),  # No resolution key
        ],
    )
    def test_execute_sets_resolution_attribute(self, key, expected_resolution):
        """Test execute sets resolution attribute based on key."""
        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {"intake_esm_attrs:source_id": "CNRM-CM6-1"}
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {key: dataset1}
        concatenated = MagicMock(spec=xr.Dataset)
        concatenated.attrs = {}

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)

        if expected_resolution:
            assert concatenated.attrs["resolution"] == expected_resolution
        else:
            assert "resolution" not in concatenated.attrs


class TestConcatExecuteErrorHandling:
    """Test class for error handling in execute method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat("sim")
        self.context = {}

    def test_execute_xarray_concat_failure_raises_with_warning(self):
        """Test execute raises ValueError and warns when xarray.concat fails."""
        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {"intake_esm_attrs:source_id": "CNRM-CM6-1"}
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {"model1": dataset1}

        with patch("xarray.concat", side_effect=ValueError("Dimension mismatch")):
            with patch("builtins.print") as mock_print:
                with pytest.warns(UserWarning, match="Failed to concatenate datasets"):
                    with pytest.raises(ValueError, match="Dimension mismatch"):
                        self.processor.execute(result_dict, self.context)  # type: ignore

                # Should print debugging info
                mock_print.assert_called()

    def test_execute_mixed_valid_invalid_objects_processes_valid_only(self):
        """Test execute processes only valid xarray objects from mixed input."""
        valid_dataset = MagicMock(spec=xr.Dataset)
        valid_dataset.attrs = {"intake_esm_attrs:source_id": "CNRM-CM6-1"}
        valid_dataset.dims = {}
        valid_dataset.expand_dims = MagicMock(return_value=valid_dataset)

        result_dict = {
            "valid": valid_dataset,
            "invalid1": "not_xarray",
            "invalid2": 123,
            "invalid3": None,
        }
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)

        # Should only process the valid dataset
        assert result is concatenated

    def test_execute_list_with_mixed_objects(self):
        """Test execute with list containing mixed valid/invalid objects."""
        valid_dataset = MagicMock(spec=xr.Dataset)
        valid_dataset.attrs = {"intake_esm_attrs:source_id": "CNRM-CM6-1"}
        valid_dataset.dims = {}
        valid_dataset.expand_dims = MagicMock(return_value=valid_dataset)

        result_list = [valid_dataset, "invalid", None, 123]
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_list, self.context)

        assert result is concatenated

    def test_execute_returns_first_valid_dataset_on_concat_fail_with_single_dataset(
        self,
    ):
        """Test execute returns first valid dataset when concatenation fails and only one dataset exists."""
        valid_dataset = MagicMock(spec=xr.Dataset)
        valid_dataset.attrs = {"intake_esm_attrs:source_id": "CNRM-CM6-1"}
        valid_dataset.dims = {}
        valid_dataset.expand_dims = MagicMock(return_value=valid_dataset)

        result_dict = {"invalid1": "not_xarray", "valid": valid_dataset}

        # This should not reach xarray.concat because only one valid dataset
        concatenated = MagicMock(spec=xr.Dataset)
        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)

        assert result is concatenated


class TestConcatUpdateContext:
    """Test class for update_context method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat("sim")
        self.context = {}

    def test_update_context_creates_new_attrs_key(self):
        """Test update_context creates new_attrs key if not present."""
        source_ids = ["model1", "model2"]

        self.processor.update_context(self.context, source_ids)

        assert "new_attrs" in self.context
        assert "concat" in self.context["new_attrs"]
        assert (
            "Multiple datasets were concatenated" in self.context["new_attrs"]["concat"]
        )

    def test_update_context_appends_to_existing_attrs(self):
        """Test update_context appends to existing new_attrs."""
        # Pre-populate context
        self.context["new_attrs"] = {"existing": "data"}
        source_ids = ["model1"]

        self.processor.update_context(self.context, source_ids)

        assert "existing" in self.context["new_attrs"]
        assert "concat" in self.context["new_attrs"]

    def test_update_context_with_time_dimension_mentions_extension(self):
        """Test update_context mentions time domain extension for time dimension."""
        processor = Concat("time")
        processor._original_dim_name = "time"  # Simulate time processing
        source_ids = ["ssp245_model1"]

        processor.update_context(self.context, source_ids)

        context_msg = self.context["new_attrs"]["concat"]
        assert "Time domain extension was performed" in context_msg
        assert "prepending historical data to SSP scenarios" in context_msg

    def test_update_context_without_time_dimension_no_extension_mention(self):
        """Test update_context doesn't mention time extension for non-time dimensions."""
        processor = Concat("sim")
        processor._original_dim_name = "sim"  # No time processing
        source_ids = ["model1", "model2"]

        processor.update_context(self.context, source_ids)

        context_msg = self.context["new_attrs"]["concat"]
        assert "Time domain extension" not in context_msg
        assert "Multiple datasets were concatenated" in context_msg

    def test_update_context_with_unset_source_ids(self):
        """Test update_context works with UNSET source_ids."""
        from climakitae.core.constants import UNSET

        self.processor.update_context(self.context, UNSET)

        assert "new_attrs" in self.context
        assert "concat" in self.context["new_attrs"]


class TestConcatIntegration:
    """Integration tests for complete Concat workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat("sim")

    @pytest.mark.integration
    def test_full_concat_workflow_with_real_structure(self):
        """Test complete concatenation workflow with realistic data structures."""
        # Create realistic mock datasets
        time_coords = pd.date_range("2020-01-01", periods=365)
        lat_coords = np.linspace(32, 42, 10)
        lon_coords = np.linspace(-124, -114, 10)

        dataset1 = xr.Dataset(
            {
                "tasmax": (["time", "lat", "lon"], np.random.rand(365, 10, 10)),
            },
            coords={"time": time_coords, "lat": lat_coords, "lon": lon_coords},
            attrs={
                "intake_esm_attrs:source_id": "CNRM-CM6-1",
                "intake_esm_attrs:institution_id": "CNRM-CERFACS",
                "intake_esm_attrs:experiment_id": "ssp245",
            },
        )

        dataset2 = xr.Dataset(
            {
                "tasmax": (["time", "lat", "lon"], np.random.rand(365, 10, 10)),
            },
            coords={"time": time_coords, "lat": lat_coords, "lon": lon_coords},
            attrs={
                "intake_esm_attrs:source_id": "GFDL-ESM4",
                "intake_esm_attrs:institution_id": "NOAA-GFDL",
                "intake_esm_attrs:experiment_id": "ssp245",
            },
        )

        result_dict = {"cnrm.ssp245": dataset1, "gfdl.ssp245": dataset2}

        context = {}

        # Execute the concatenation
        result = self.processor.execute(result_dict, context)  # type: ignore

        # Verify results
        assert isinstance(result, xr.Dataset)
        assert "sim" in result.dims
        assert result.sizes["sim"] == 2
        assert "tasmax" in result.data_vars
        assert "new_attrs" in context
        assert "concat" in context["new_attrs"]

    @pytest.mark.integration
    def test_time_dimension_integration_with_extend_time_domain(self):
        """Test time dimension integration with actual extend_time_domain function."""
        processor = Concat("time")

        # Create mock datasets representing SSP scenario
        dataset_ssp = xr.Dataset(
            {"tas": (["time"], [20.0, 21.0, 22.0])},
            coords={"time": pd.date_range("2015-01-01", periods=3)},
            attrs={
                "intake_esm_attrs:source_id": "CNRM-CM6-1",
                "intake_esm_attrs:experiment_id": "ssp245",
            },
        )

        result_dict = {"ssp245.cnrm": dataset_ssp}
        context = {}

        # Mock extend_time_domain to return extended data
        extended_dataset = dataset_ssp.copy()
        extended_dataset.attrs["historical_prepended"] = True
        extended_dict = {"ssp245.cnrm": extended_dataset}

        with patch(
            "climakitae.new_core.processors.concatenate.extend_time_domain",
            return_value=extended_dict,
        ):
            result = processor.execute(result_dict, context)  # type: ignore

        # Verify time processing occurred
        assert processor.dim_name == "sim"  # Changed from "time"
        assert processor._original_dim_name == "time"  # Original tracked
        assert isinstance(result, xr.Dataset)
        assert "new_attrs" in context
        assert "Time domain extension" in context["new_attrs"]["concat"]


class TestConcatEdgeCases:
    """Test class for edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = Concat("sim")
        self.context = {}

    def test_execute_with_dataarray_input(self):
        """Test execute handles xarray DataArrays correctly."""
        # Create mock DataArrays
        dataarray1 = MagicMock(spec=xr.DataArray)
        dataarray1.attrs = {"intake_esm_attrs:source_id": "model1"}
        dataarray1.dims = {}
        dataarray1.expand_dims = MagicMock(return_value=dataarray1)

        dataarray2 = MagicMock(spec=xr.DataArray)
        dataarray2.attrs = {"intake_esm_attrs:source_id": "model2"}
        dataarray2.dims = {}
        dataarray2.expand_dims = MagicMock(return_value=dataarray2)

        result_dict = {"model1": dataarray1, "model2": dataarray2}
        concatenated = MagicMock(spec=xr.DataArray)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        assert result is concatenated

    def test_execute_with_extremely_long_attribute_names(self):
        """Test execute handles very long attribute names gracefully."""
        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {
            "intake_esm_attrs:source_id": "Very_Long_Model_Name_That_Exceeds_Normal_Limits",
            "intake_esm_attrs:institution_id": "Very_Long_Institution_Name_With_Multiple_Words",
        }
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {"long_name": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        assert result is concatenated

    def test_execute_with_special_characters_in_attributes(self):
        """Test execute handles special characters in attributes."""
        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {
            "intake_esm_attrs:source_id": "Model-Name_With.Special@Characters",
            "intake_esm_attrs:experiment_id": "ssp2.6",
        }
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {"special": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        assert result is concatenated

    def test_execute_with_unicode_characters_in_attributes(self):
        """Test execute handles unicode characters in attributes."""
        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {
            "intake_esm_attrs:source_id": "Modèle_Climatique_Français",
            "intake_esm_attrs:institution_id": "CNRM-CERFACS",
        }
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {"unicode": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = self.processor.execute(result_dict, self.context)  # type: ignore

        assert result is concatenated

    def test_execute_preserves_print_statements(self):
        """Test execute preserves print statements for user feedback."""
        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {"intake_esm_attrs:source_id": "model1"}
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {"model1": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            with patch("builtins.print") as mock_print:
                _ = self.processor.execute(result_dict, self.context)  # type: ignore

        # Should print success message
        mock_print.assert_called()
        print_args = str(mock_print.call_args_list)
        assert "Concatenated datasets along" in print_args


class TestConcatParameterizedCases:
    """Test class for parameterized test cases."""

    @pytest.mark.parametrize("dim_name", ["sim", "model", "ensemble", "member"])
    def test_concat_with_different_dimension_names(self, dim_name):
        """Test concatenation with different dimension names."""
        processor = Concat(dim_name)

        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = {"intake_esm_attrs:source_id": "model1"}
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {"model1": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = processor.execute(result_dict, {})  # type: ignore

        # Should use the specified dimension name
        dataset1.expand_dims.assert_called_with({dim_name: ["model1"]})
        assert result is concatenated

    @pytest.mark.parametrize(
        "catalog_key",
        [
            "renewable energy generation",
            "CMIP6",
            None,
        ],
    )
    def test_catalog_specific_attribute_handling(self, catalog_key):
        """Test different attribute sets based on catalog type."""
        processor = Concat("sim")
        processor.catalog = MagicMock()
        processor.catalog.catalog_key = catalog_key

        # Create dataset with both sets of attributes
        all_attrs = {
            "installation": "test_install",
            "institution_id": "test_institution",
            "source_id": "test_source",
            "experiment_id": "test_exp",
            "member_id": "test_member",
            "intake_esm_attrs:activity_id": "test_activity",
            "intake_esm_attrs:institution_id": "test_esm_institution",
            "intake_esm_attrs:source_id": "test_esm_source",
            "intake_esm_attrs:experiment_id": "test_esm_exp",
            "intake_esm_attrs:member_id": "test_esm_member",
        }

        dataset1 = MagicMock(spec=xr.Dataset)
        dataset1.attrs = all_attrs
        dataset1.dims = {}
        dataset1.expand_dims = MagicMock(return_value=dataset1)

        result_dict = {"test": dataset1}
        concatenated = MagicMock(spec=xr.Dataset)

        with patch("xarray.concat", return_value=concatenated):
            result = processor.execute(result_dict, {})  # type: ignore

        assert result is concatenated

    @pytest.mark.parametrize(
        "member_ids,expected_count",
        [
            (["r1i1p1f1"], 1),
            (["r1i1p1f1", "r2i1p1f1"], 2),
            (["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"], 3),
            ([], 0),  # Edge case: no members
        ],
    )
    def test_member_id_processing_with_different_counts(
        self, member_ids, expected_count
    ):
        """Test member_id processing with different numbers of members."""
        processor = Concat("sim")

        if expected_count > 0:
            dataset1 = MagicMock(spec=xr.Dataset)
            dataset1.attrs = {"intake_esm_attrs:source_id": "model1"}
            dataset1.dims = {"member_id": len(member_ids)}
            dataset1.member_id = MagicMock()
            dataset1.member_id.values = member_ids

            def mock_sel(member_id):
                selected = MagicMock(spec=xr.Dataset)
                selected.drop_vars = MagicMock(return_value=selected)
                selected.expand_dims = MagicMock(return_value=selected)
                return selected

            dataset1.sel = MagicMock(side_effect=mock_sel)

            result_dict = {"model1": dataset1}
            concatenated = MagicMock(spec=xr.Dataset)

            with patch("xarray.concat", return_value=concatenated):
                result = processor.execute(result_dict, {})  # type: ignore

            # Should call sel for each member
            assert dataset1.sel.call_count == expected_count
            assert result is concatenated
        else:
            # Edge case with no members - this would be an unusual real-world case
            dataset1 = MagicMock(spec=xr.Dataset)
            dataset1.attrs = {"intake_esm_attrs:source_id": "model1"}
            dataset1.dims = {}  # No member_id dimension when empty
            dataset1.expand_dims = MagicMock(return_value=dataset1)

            result_dict = {"model1": dataset1}
            concatenated = MagicMock(spec=xr.Dataset)

            with patch("xarray.concat", return_value=concatenated):
                result = processor.execute(result_dict, {})  # type: ignore

            # Should use standard processing when no member_id dimension
            dataset1.expand_dims.assert_called_once()
            assert result is concatenated
