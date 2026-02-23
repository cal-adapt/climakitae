"""
Unit tests for climakitae/new_core/processors/warming_param_validator.py.

This module contains comprehensive unit tests for the Warming Level processor
parameter validation functionality.
"""

import warnings
from unittest.mock import patch

import pandas as pd
import pytest

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.warming_param_validator import (
    _check_catalog,
    _check_input_types,
    _check_query,
    _check_wl_values,
    validate_warming_level_param,
)


@pytest.fixture
def patched_data():
    """Fixture to patch DataCatalog and read_csv_file for tests."""
    with (
        patch(
            "climakitae.new_core.param_validation.warming_param_validator.DataCatalog"
        ) as mock_catalog,
        patch(
            "climakitae.new_core.param_validation.warming_param_validator.read_csv_file"
        ) as mock_read_csv,
    ):
        mock_catalog.return_value.catalog_df = pd.DataFrame(
            {
                "activity_id": ["WRF", "LOCA2"],
                "source_id": ["ModelA", "ModelB"],
                "member_id": ["r1i1p1f1", "r1i1p1f1"],
                "experiment_id": ["ssp245", "ssp585"],
            }
        )
        mock_read_csv.return_value = pd.DataFrame(
            {
                "ModelA_r1i1p1f1_ssp245": [0.5, 1.0, 2.0, 3.0],
                "ModelB_r1i1p1f1_ssp585": [0.8, 1.5, 2.5, 3.5],
            }
        )
        yield


class TestCheckInputTypes:
    """Test class for _check_input_types function."""

    def test_check_input_types_valid_dict(self):
        """Test _check_input_types with a valid dictionary input."""
        value = {
            "warming_levels": [1.5, 2.0],
            "warming_level_months": [1, 2, 3],
            "warming_level_window": 15,
            "add_dummy_time": True,
        }
        result = _check_input_types(value)
        assert result is True

    @pytest.mark.parametrize("container_type", [list, set, tuple])
    def test_check_input_types_invalid_type(self, container_type: type):
        """Test _check_input_types with an invalid type input."""
        value = container_type(["1.5", "2.0"])
        with pytest.warns(
            UserWarning,
            match="Warming Level Processor expects a dictionary of parameters.",
        ):
            result = _check_input_types(value)
            assert result is False

    @pytest.mark.parametrize("wrong_type", [None, "1.5, 2.0", [1.5, "two"]])
    def test_check_input_types_missing_warming_levels(self, wrong_type: object):
        """Test _check_input_types with missing 'warming_levels' key."""
        value = {
            "warming_levels": wrong_type,
        }
        with pytest.warns(UserWarning, match="Invalid 'warming_levels' parameter."):
            result = _check_input_types(value)
            assert result is False

    @pytest.mark.parametrize("wrong_months", [[0, 13], ["January", 2], [1.5, 2.5]])
    def test_check_input_types_invalid_warming_level_months(self, wrong_months):
        """Test _check_input_types with invalid 'warming_level_months'."""
        value = {
            "warming_levels": [1.5, 2.0],
            "warming_level_months": wrong_months,
        }
        with pytest.warns(
            UserWarning,
            match="Invalid 'warming_level_months' parameter.",
        ):
            result = _check_input_types(value)
            assert result is False

    @pytest.mark.parametrize("wrong_window", [-1, "fifteen", 2.5])
    def test_check_input_types_invalid_warming_level_window(self, wrong_window):
        """Test _check_input_types with invalid 'warming_level_window'."""
        value = {
            "warming_levels": [1.5, 2.0],
            "warming_level_window": wrong_window,
        }
        with pytest.warns(
            UserWarning,
            match="Invalid 'warming_level_window' parameter.",
        ):
            result = _check_input_types(value)
            assert result is False

    @pytest.mark.parametrize("wrong_type", ["True", 0, -1.5, {}])
    def test_check_input_types_invalid_add_dummy_time(self, wrong_type):
        """Test _check_input_types with invalid 'warming_level_window'."""
        value = {
            "warming_levels": [1.5, 2.0],
            "add_dummy_time": wrong_type,
        }
        with pytest.warns(
            UserWarning,
            match="Invalid 'add_dummy_time' parameter.",
        ):
            result = _check_input_types(value)
            assert result is False


class TestCheckCatalog:
    """Test class for _check_catalog function."""

    def test_check_catalog_valid_cadcat(self):
        """Test _check_catalog with valid 'cadcat' catalog."""
        query = {"catalog": "cadcat"}
        result = _check_catalog(query)
        assert result is True

    def test_check_catalog_unset_defaults_to_valid(self):
        """Test _check_catalog with UNSET catalog (assumes cadcat default)."""
        query = {"activity_id": "WRF"}  # No catalog key
        result = _check_catalog(query)
        assert result is True

    def test_check_catalog_invalid_renewables(self):
        """Test _check_catalog with invalid 'renewable energy generation' catalog."""
        query = {"catalog": "renewable energy generation"}
        with pytest.warns(
            UserWarning,
            match="Warming level processor is not supported for 'renewable energy generation' catalog",
        ):
            result = _check_catalog(query)
            assert result is False

    def test_check_catalog_invalid_hdp(self):
        """Test _check_catalog with invalid 'hdp' catalog."""
        query = {"catalog": "hdp"}
        with pytest.warns(
            UserWarning,
            match="Warming level processor is not supported for 'hdp' catalog",
        ):
            result = _check_catalog(query)
            assert result is False

    def test_check_catalog_invalid_not_dict(self):
        """Test _check_catalog with non-dictionary input."""
        result = _check_catalog("not a dict")
        assert result is False


class TestCheckQuery:
    """Test class for _check_query function."""

    @pytest.mark.parametrize("wrong_type", [[], (), "invalid", 123, None])
    def test_invalid_type(self, wrong_type):
        """Test _check_query with an invalid data type."""
        result = _check_query(wrong_type)
        assert result is False

    def test_experiment_id_not_set(self):
        """Test that experiment_id is not set."""
        value = {"experiment_id": "i am something"}
        with pytest.warns(
            UserWarning,
            match="Warming level approach requires 'experiment_id' to be UNSET.",
        ):
            result = _check_query(value)
            assert result is False

    def test_time_slice_not_set(self, caplog):
        """Test that time_slice is not set when using warming level approach."""
        value = {"processes": {"time_slice": "do not set me"}}
        with caplog.at_level("ERROR"):
            result = _check_query(value)
            assert result is False
            assert (
                "The warming_level and time_slice processors cannot be used concurrently."
                in caplog.text
            )

    @pytest.mark.parametrize("correct_activity_id", [UNSET, "WRF", "LOCA2"])
    def test_activity_id_is_valid(self, correct_activity_id):
        """Test that activity_id is valid."""
        value = {"activity_id": correct_activity_id}
        result = _check_query(value)
        assert result is True

    def test_activity_id_is_invalid(self):
        """Test that activity_id is False and throws a warning for incorrect parameters."""
        value = {"activity_id": "this is a wrong parameter type"}
        with pytest.warns(UserWarning, match="Invalid 'activity_id' parameter."):
            result = _check_query(value)
            assert result is False


class TestCheckWLValues:
    """Test class for _check_wl_values function."""

    def test_check_wl_values_activity_id_unset(self, patched_data):
        """Test _check_wl_values works with activity_id UNSET."""
        value = {"warming_levels": [2.0]}
        result = _check_wl_values(value)
        assert result is True

    def test_check_wl_values_all_within_range(self, patched_data):
        """Test _check_wl_values with all warming levels within range."""
        value = {"warming_levels": [1.0, 2.0, 3.0]}
        query = {"activity_id": "WRF"}
        result = _check_wl_values(value, query)
        assert result is True

    def test_check_wl_values_out_of_range(self, patched_data):
        """Test _check_wl_values with a warming level out of range."""
        value = {"warming_levels": [4.5]}
        query = {"activity_id": "LOCA2"}
        with pytest.warns(UserWarning, match="outside the range"):
            result = _check_wl_values(value, query)
            assert result is False

    def test_check_wl_values_some_out_of_range(self, patched_data):
        """Test _check_wl_values with some warming levels out of range."""
        value = {"warming_levels": [1.0, 4.0]}
        query = {"activity_id": "WRF"}
        with pytest.warns(UserWarning, match="4.0 is outside the range"):
            result = _check_wl_values(value, query)
            assert result is False


class TestValidateWarmingLevelParam:
    """Test class for validate_warming_level_param function."""

    def test_validate_warming_level_param_valid(self, patched_data):
        """Test validate_warming_level_param with valid parameters."""
        value = {
            "warming_levels": [1.5, 2.0],
            "warming_level_months": [1, 2, 3],
            "warming_level_window": 15,
        }
        query = {"activity_id": "WRF", "experiment_id": UNSET}
        result = validate_warming_level_param(value, query=query)
        assert result is True

    def test_validate_warming_level_param_missing_query(self):
        """Test validate_warming_level_param with missing query parameter."""
        value = {
            "warming_levels": [1.5, 2.0],
            "warming_level_months": [1, 2, 3],
            "warming_level_window": 15,
        }
        query = UNSET
        with pytest.warns(
            UserWarning, match="Warming Level Processor requires a 'query' parameter."
        ):
            result = validate_warming_level_param(value, query=query)
            assert result is False

    def test_validate_warming_level_param_invalid_input_types(self):
        """Test validate_warming_level_param with invalid input types."""
        value = ["1.5", "2.0"]
        query = {"activity_id": "WRF", "experiment_id": UNSET}
        with pytest.warns(
            UserWarning,
            match="Warming Level Processor expects a dictionary of parameters.",
        ):
            result = validate_warming_level_param(value, query=query)
            assert result is False

    def test_validate_warming_level_param_invalid_query(self):
        """Test validate_warming_level_param with invalid query parameters."""
        value = {
            "warming_levels": [1.5, 2.0],
            "warming_level_months": [1, 2, 3],
            "warming_level_window": 15,
        }
        query = {"activity_id": "INVALID", "experiment_id": "something"}
        with pytest.warns(UserWarning, match="Invalid 'activity_id' parameter."):
            result = validate_warming_level_param(value, query=query)
            assert result is False

    def test_validate_warming_level_param_invalid_wl_values(self, patched_data):
        """Test validate_warming_level_param with invalid warming level values."""
        value = {
            "warming_levels": [5.0],
            "warming_level_months": [1, 2, 3],
            "warming_level_window": 15,
        }
        query = {"activity_id": "WRF", "experiment_id": UNSET}
        with pytest.warns(UserWarning, match="outside the range"):
            result = validate_warming_level_param(value, query=query)
            assert result is False
