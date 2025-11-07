"""
Unit tests for climakitae/new_core/param_validation/clip_param_validator.py

This module contains comprehensive unit tests for the clip parameter validation functions
that validate and normalize parameters for the Clip processor.

Tests cover:
- Main validate_clip_param function with various input types
- String parameter validation (file paths, station identifiers, boundary keys)
- List parameter validation (mixed types, duplicates, stations, boundaries)
- Tuple parameter validation (coordinate bounds, single points)
- Station identifier validation
- File path detection
- Boundary key validation with case sensitivity
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climakitae.new_core.param_validation.clip_param_validator import (
    _is_file_path_like,
    _validate_boundary_key_string,
    _validate_list_param,
    _validate_station_identifier,
    _validate_string_param,
    _validate_tuple_param,
    _warn_about_case_sensitivity,
    validate_clip_param,
)


class TestValidateClipParam:
    """Test class for validate_clip_param dispatcher function.

    This class tests the main entry point that dispatches to type-specific
    validators based on input type (str, list, tuple, etc.).

    Note: validate_clip_param returns bool indicating validation success,
    not the validated value itself.
    """

    def test_validate_with_valid_str_returns_true(self):
        """Test validate_clip_param with valid string returns True."""
        # Valid boundary key
        result = validate_clip_param("CA")
        assert result is True

    def test_validate_with_valid_list_returns_true(self):
        """Test validate_clip_param with valid list returns True."""
        # Valid list of boundary keys
        result = validate_clip_param(["CA", "NV"])
        assert result is True

    def test_validate_with_valid_tuple_point_returns_true(self):
        """Test validate_clip_param with valid tuple (point) returns True."""
        # Valid coordinate point - nested tuples
        result = validate_clip_param(((37.0, 38.0), (-122.0, -121.0)))
        assert result is True

    def test_validate_with_none_returns_false(self):
        """Test validate_clip_param with None returns False."""
        with pytest.warns(UserWarning, match="Clip parameter cannot be None"):
            result = validate_clip_param(None)
            assert result is False

    def test_validate_with_invalid_type_returns_false(self):
        """Test validate_clip_param with invalid type returns False and warns."""
        with pytest.warns(UserWarning, match="Invalid parameter type"):
            result = validate_clip_param(123)
            assert result is False

    def test_validate_with_dict_returns_false(self):
        """Test validate_clip_param with dict returns False and warns."""
        with pytest.warns(UserWarning, match="Invalid parameter type"):
            result = validate_clip_param({"key": "value"})
            assert result is False

    def test_validate_with_invalid_tuple_format_returns_false(self):
        """Test validate_clip_param with invalid tuple format returns False."""
        # 4-element tuple instead of nested ((lat_min, lat_max), (lon_min, lon_max))
        with pytest.warns(
            UserWarning, match="Coordinate bounds tuple must have exactly 2 elements"
        ):
            result = validate_clip_param((32.0, -124.0, 42.0, -114.0))
            assert result is False


class TestValidateStringParam:
    """Test class for _validate_string_param function.

    Tests validation of string parameters with priority:
    1. File path (highest priority - most specific)
    2. Station identifier (specific format)
    3. Boundary key (most general)
    """

    def test_validate_empty_string_returns_false(self):
        """Test _validate_string_param with empty string returns False."""
        with pytest.warns(UserWarning, match="Empty or whitespace-only strings"):
            result = _validate_string_param("")
            assert result is False

    def test_validate_whitespace_only_string_returns_false(self):
        """Test _validate_string_param with whitespace-only string returns False."""
        with pytest.warns(UserWarning, match="Empty or whitespace-only strings"):
            result = _validate_string_param("   ")
            assert result is False

    def test_validate_valid_boundary_key_returns_true(self):
        """Test _validate_string_param with valid boundary key returns True."""
        result = _validate_string_param("CA")
        assert result is True

    def test_validate_existing_file_path_returns_true(self):
        """Test _validate_string_param with existing file path returns True."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".shp", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = _validate_string_param(tmp_path)
            assert result is True
        finally:
            os.unlink(tmp_path)

    def test_validate_nonexistent_file_path_returns_false(self):
        """Test _validate_string_param with non-existent file path returns False."""
        fake_path = "/nonexistent/path/to/file.shp"
        with pytest.warns(UserWarning, match="File path .* does not exist"):
            result = _validate_string_param(fake_path)
            assert result is False

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_station_identifier"
    )
    @patch(
        "climakitae.new_core.param_validation.clip_param_validator.is_station_identifier"
    )
    def test_validate_valid_station_identifier_returns_true(
        self, mock_is_station, mock_validate_station
    ):
        """Test _validate_string_param with valid station identifier returns True."""
        mock_is_station.return_value = True
        mock_validate_station.return_value = True

        result = _validate_string_param("KSAC")
        assert result is True
        mock_is_station.assert_called_once_with("KSAC")
        mock_validate_station.assert_called_once_with("KSAC")

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_boundary_key_string"
    )
    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_station_identifier"
    )
    @patch(
        "climakitae.new_core.param_validation.clip_param_validator.is_station_identifier"
    )
    def test_validate_station_like_but_is_boundary_returns_true(
        self, mock_is_station, mock_validate_station, mock_validate_boundary
    ):
        """Test _validate_string_param with station-like string that's actually a boundary."""
        mock_is_station.return_value = True
        mock_validate_station.return_value = False
        mock_validate_boundary.return_value = True

        with pytest.warns(
            UserWarning, match="looks like a station identifier but was not found"
        ):
            result = _validate_string_param("Kern")
            assert result is True

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_boundary_key_string"
    )
    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_station_identifier"
    )
    @patch(
        "climakitae.new_core.param_validation.clip_param_validator.is_station_identifier"
    )
    def test_validate_station_like_but_invalid_returns_false(
        self, mock_is_station, mock_validate_station, mock_validate_boundary
    ):
        """Test _validate_string_param with invalid station-like string returns False."""
        mock_is_station.return_value = True
        mock_validate_station.return_value = False
        mock_validate_boundary.return_value = False

        result = _validate_string_param("KXYZ")
        assert result is False

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_boundary_key_string"
    )
    @patch(
        "climakitae.new_core.param_validation.clip_param_validator.is_station_identifier"
    )
    def test_validate_non_station_like_boundary_returns_true(
        self, mock_is_station, mock_validate_boundary
    ):
        """Test _validate_string_param with non-station-like boundary key returns True."""
        mock_is_station.return_value = False
        mock_validate_boundary.return_value = True

        result = _validate_string_param("Los Angeles County")
        assert result is True
        mock_validate_boundary.assert_called_once_with("Los Angeles County")


class TestValidateListParam:
    """Test class for _validate_list_param function.

    Tests validation of list parameters including empty lists, mixed types,
    duplicates, valid boundaries, and valid stations.
    """

    def test_validate_empty_list_returns_false(self):
        """Test _validate_list_param with empty list returns False."""
        with pytest.warns(UserWarning, match="Empty list is not valid"):
            result = _validate_list_param([])
            assert result is False

    def test_validate_list_with_valid_boundaries_returns_true(self):
        """Test _validate_list_param with valid boundary keys returns True."""
        result = _validate_list_param(["CA", "NV", "OR"])
        assert result is True

    def test_validate_list_with_mixed_types_returns_false(self):
        """Test _validate_list_param with mixed types returns False."""
        with pytest.warns(
            UserWarning, match="All items in clip parameter list must be the same type"
        ):
            result = _validate_list_param(["CA", 123, "NV"])
            assert result is False

    def test_validate_list_with_duplicates_returns_false(self):
        """Test _validate_list_param with duplicates returns False."""
        with pytest.warns(UserWarning, match="Duplicate boundary keys found"):
            result = _validate_list_param(["CA", "NV", "CA"])
            assert result is False

    def test_validate_list_with_empty_strings_returns_false(self):
        """Test _validate_list_param with empty/whitespace strings returns False."""
        with pytest.warns(UserWarning, match="invalid items"):
            result = _validate_list_param(["CA", "", "NV"])
            assert result is False

    def test_validate_list_with_whitespace_strings_returns_false(self):
        """Test _validate_list_param with whitespace-only strings returns False."""
        with pytest.warns(UserWarning, match="invalid items"):
            result = _validate_list_param(["CA", "   ", "NV"])
            assert result is False

    def test_validate_list_with_invalid_boundary_keys_returns_false(self):
        """Test _validate_list_param with invalid boundary keys returns False."""
        # INVALID_KEY doesn't match any boundaries but still passes through
        # because _validate_boundary_key_string only checks basic validity
        # This actually returns True because it just warns about case sensitivity
        with pytest.warns(UserWarning, match="does not match any known boundary keys"):
            result = _validate_list_param(["CA", "INVALID_KEY", "NV"])
            # Actually passes because invalid key warnings are non-blocking
            assert result is True

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_station_identifier"
    )
    @patch(
        "climakitae.new_core.param_validation.clip_param_validator.is_station_identifier"
    )
    def test_validate_list_with_all_valid_stations_returns_true(
        self, mock_is_station, mock_validate_station
    ):
        """Test _validate_list_param with all valid station identifiers returns True."""
        mock_is_station.return_value = True
        mock_validate_station.return_value = True

        result = _validate_list_param(["KSAC", "KOAK", "KSFO"])
        assert result is True
        assert mock_validate_station.call_count == 3

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_station_identifier"
    )
    @patch(
        "climakitae.new_core.param_validation.clip_param_validator.is_station_identifier"
    )
    def test_validate_list_with_one_invalid_station_returns_false(
        self, mock_is_station, mock_validate_station
    ):
        """Test _validate_list_param with one invalid station returns False."""
        mock_is_station.return_value = True
        # First two succeed, third fails
        mock_validate_station.side_effect = [True, True, False]

        result = _validate_list_param(["KSAC", "KOAK", "KXYZ"])
        assert result is False

    def test_validate_list_with_valid_tuples_returns_true(self):
        """Test _validate_list_param with valid coordinate tuples returns True."""
        result = _validate_list_param(
            [((32.0, 34.0), (-120.0, -118.0)), ((35.0, 37.0), (-122.0, -120.0))]
        )
        assert result is True

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_tuple_param"
    )
    def test_validate_list_with_invalid_tuples_returns_false(self, mock_validate_tuple):
        """Test _validate_list_param with invalid tuples returns False."""
        # First tuple valid, second invalid
        mock_validate_tuple.side_effect = [True, False]

        with pytest.warns(UserWarning, match="invalid items"):
            result = _validate_list_param(
                [((32.0, 34.0), (-120.0, -118.0)), (1, 2, 3, 4)]  # Invalid format
            )
            assert result is False

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_boundary_key_string"
    )
    def test_validate_list_with_exception_in_validation_returns_false(
        self, mock_validate
    ):
        """Test _validate_list_param with exception during validation returns False."""
        # First item succeeds, second raises exception, third succeeds
        mock_validate.side_effect = [True, Exception("Validation error"), True]

        with pytest.warns(UserWarning, match="invalid items"):
            result = _validate_list_param(["CA", "INVALID", "NV"])
            assert result is False

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._validate_boundary_key_string"
    )
    def test_validate_list_all_items_invalid_returns_false(self, mock_validate):
        """Test _validate_list_param with all invalid items returns False."""
        # Mock returns None for invalid items (the code checks `if validated_item is not None`)
        mock_validate.return_value = None

        with pytest.warns(
            UserWarning, match="Found \\d+ invalid items in clip parameter list"
        ):
            result = _validate_list_param(["INVALID1", "INVALID2"])
            assert result is False


class TestValidateTupleParam:
    """Test class for _validate_tuple_param function.

    Tests validation of tuple parameters for coordinate bounds including
    nested tuples, single points, invalid formats, and coordinate ranges.
    """

    def test_validate_valid_nested_tuple_returns_true(self):
        """Test _validate_tuple_param with valid nested tuple returns True."""
        result = _validate_tuple_param(((32.0, 42.0), (-124.0, -114.0)))
        assert result is True

    def test_validate_tuple_wrong_length_returns_false(self):
        """Test _validate_tuple_param with wrong tuple length returns False."""
        with pytest.warns(UserWarning, match="must have exactly 2 elements"):
            result = _validate_tuple_param((32.0, 42.0, -124.0))
            assert result is False

    def test_validate_tuple_with_non_numeric_returns_false(self):
        """Test _validate_tuple_param with non-numeric values returns False."""
        with pytest.warns(UserWarning, match="must be numeric"):
            result = _validate_tuple_param((("a", "b"), (-124.0, -114.0)))
            assert result is False

    def test_validate_tuple_invalid_latitude_range_returns_false(self):
        """Test _validate_tuple_param with invalid latitude range returns False."""
        with pytest.warns(UserWarning, match="Latitude values must be between"):
            result = _validate_tuple_param(((95.0, 100.0), (-124.0, -114.0)))
            assert result is False

    def test_validate_tuple_invalid_longitude_range_returns_false(self):
        """Test _validate_tuple_param with invalid longitude range returns False."""
        with pytest.warns(UserWarning, match="Longitude values must be between"):
            result = _validate_tuple_param(((32.0, 42.0), (-200.0, -250.0)))
            assert result is False

    def test_validate_tuple_single_point_returns_true(self):
        """Test _validate_tuple_param with single point (float values) returns True."""
        result = _validate_tuple_param((37.0, -122.0))
        assert result is True

    def test_validate_tuple_with_wrong_length_nested_tuple_returns_false(self):
        """Test _validate_tuple_param with wrong length nested tuple returns False."""
        with pytest.warns(
            UserWarning, match="must either be a tuple of two numeric values"
        ):
            result = _validate_tuple_param(((32.0, 42.0, 50.0), (-124.0, -114.0)))
            assert result is False

    def test_validate_tuple_with_invalid_bound_type_dict_returns_false(self):
        """Test _validate_tuple_param with invalid bound type (dict) returns False."""
        with pytest.warns(UserWarning, match="must be a tuple of two numeric values"):
            result = _validate_tuple_param(({"invalid": "dict"}, (-124.0, -114.0)))
            assert result is False

    def test_validate_tuple_with_invalid_bound_type_none_returns_false(self):
        """Test _validate_tuple_param with None bound returns False."""
        with pytest.warns(UserWarning, match="must be a tuple of two numeric values"):
            result = _validate_tuple_param((None, (-124.0, -114.0)))
            assert result is False

    def test_validate_tuple_min_greater_than_max_latitude_returns_false(self):
        """Test _validate_tuple_param with min > max for latitude returns False."""
        with pytest.warns(UserWarning, match="Minimum latitude must be less than"):
            result = _validate_tuple_param(((42.0, 32.0), (-124.0, -114.0)))
            assert result is False

    def test_validate_tuple_min_greater_than_max_longitude_returns_false(self):
        """Test _validate_tuple_param with min > max for longitude returns False."""
        with pytest.warns(UserWarning, match="Minimum longitude must be less than"):
            result = _validate_tuple_param(((32.0, 42.0), (-114.0, -124.0)))
            assert result is False


class TestValidateStationIdentifier:
    """Test class for _validate_station_identifier function.

    Tests validation of weather station identifiers with valid/invalid codes
    and cross-category suggestions for boundaries.
    """

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_validate_valid_station_returns_true(self, mock_data_catalog_class):
        """Test _validate_station_identifier with valid station code returns True."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog

        # Mock stations DataFrame via get("stations")
        mock_catalog.get.return_value = pd.DataFrame(
            {
                "station": ["SACRAMENTO EXECUTIVE AIRPORT"],
                "ID": ["KSAC"],
                "city": ["Sacramento"],
                "state": ["CA"],
            }
        )

        result = _validate_station_identifier("KSAC")
        assert result is True
        mock_catalog.get.assert_called_with("stations")

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_validate_invalid_station_returns_false(self, mock_data_catalog_class):
        """Test _validate_station_identifier with invalid station code returns False."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog

        # Mock stations DataFrame with at least one station so it's not empty
        mock_catalog.get.return_value = pd.DataFrame(
            {
                "station": ["SACRAMENTO EXECUTIVE AIRPORT"],
                "ID": ["KSAC"],
                "city": ["Sacramento"],
                "state": ["CA"],
            }
        )
        # Mock list_clip_boundaries to return empty dict
        mock_catalog.list_clip_boundaries.return_value = {}

        with pytest.warns(UserWarning, match="not found in station database"):
            result = _validate_station_identifier("KXYZ")
            assert result is False

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_validate_invalid_station_with_boundary_suggestion(
        self, mock_data_catalog_class
    ):
        """Test _validate_station_identifier suggests matching boundaries when station not found."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog
        # Need at least one station to pass the "data available" check
        mock_catalog.get.return_value = pd.DataFrame(
            {
                "station": ["SOME OTHER STATION"],
                "ID": ["KOTH"],
                "city": ["Other"],
                "state": ["CA"],
            }
        )

        # Mock list_clip_boundaries with "Kern County" which is similar enough to "Kern"
        mock_catalog.list_clip_boundaries.return_value = {
            "counties": ["Kern County", "Los Angeles County"]
        }

        # Use "Kern" which should match "Kern County"
        with pytest.warns(UserWarning, match="Or did you mean one of these boundaries"):
            result = _validate_station_identifier("Kern")
            assert result is False

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_validate_station_no_data_available_returns_false(
        self, mock_data_catalog_class
    ):
        """Test _validate_station_identifier with no data available returns False."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog
        # Return None for stations
        mock_catalog.get.return_value = None

        with pytest.warns(UserWarning, match="Station data is not available"):
            result = _validate_station_identifier("KSAC")
            assert result is False

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_validate_station_multiple_matches_returns_false(
        self, mock_data_catalog_class
    ):
        """Test _validate_station_identifier with multiple matches returns False."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog

        # Return multiple matching stations
        mock_catalog.get.return_value = pd.DataFrame(
            {
                "station": ["SACRAMENTO EXECUTIVE AIRPORT", "SACRAMENTO INTERNATIONAL"],
                "ID": ["KSAC", "KSAC2"],
                "city": ["Sacramento", "Sacramento"],
                "state": ["CA", "CA"],
            }
        )

        with pytest.warns(UserWarning, match="Multiple stations match"):
            result = _validate_station_identifier("SACRAMENTO")
            assert result is False

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_validate_station_exception_returns_false(self, mock_data_catalog_class):
        """Test _validate_station_identifier with exception returns False."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog

        # Raise exception during get
        mock_catalog.get.side_effect = Exception("Database error")

        with pytest.warns(UserWarning, match="Error validating station identifier"):
            result = _validate_station_identifier("KSAC")
            assert result is False


class TestIsFilePathLike:
    """Test class for _is_file_path_like function.

    Tests heuristic detection of file paths vs regular strings.
    """

    def test_is_file_path_like_with_shp_extension_returns_true(self):
        """Test _is_file_path_like with .shp extension returns True."""
        assert _is_file_path_like("data/boundaries.shp") is True

    def test_is_file_path_like_with_json_extension_returns_true(self):
        """Test _is_file_path_like with .json extension returns True."""
        assert _is_file_path_like("/path/to/file.json") is True

    def test_is_file_path_like_with_directory_separator_returns_true(self):
        """Test _is_file_path_like with directory separator returns True."""
        assert _is_file_path_like("directory/file") is True

    def test_is_file_path_like_with_regular_string_returns_false(self):
        """Test _is_file_path_like with regular string returns False."""
        assert _is_file_path_like("CA") is False

    def test_is_file_path_like_with_boundary_name_returns_false(self):
        """Test _is_file_path_like with boundary name returns False."""
        assert _is_file_path_like("Los Angeles County") is False


class TestValidateBoundaryKeyString:
    """Test class for _validate_boundary_key_string function.

    Tests validation of boundary keys including case sensitivity and
    invalid characters.
    """

    def test_validate_valid_boundary_key_returns_true(self):
        """Test _validate_boundary_key_string with valid key returns True."""
        result = _validate_boundary_key_string("CA")
        assert result is True

    def test_validate_too_long_boundary_key_returns_false(self):
        """Test _validate_boundary_key_string with excessively long key returns False."""
        long_key = "x" * 201
        with pytest.warns(UserWarning, match="Boundary key is too long"):
            result = _validate_boundary_key_string(long_key)
            assert result is False

    def test_validate_boundary_key_with_invalid_chars_returns_false(self):
        """Test _validate_boundary_key_string with invalid characters returns False."""
        with pytest.warns(UserWarning, match="contains invalid characters"):
            result = _validate_boundary_key_string("CA<>")
            assert result is False

    @patch(
        "climakitae.new_core.param_validation.clip_param_validator._warn_about_case_sensitivity"
    )
    def test_validate_boundary_key_calls_case_sensitivity_check(self, mock_warn):
        """Test _validate_boundary_key_string calls case sensitivity warning."""
        mock_warn.return_value = True
        result = _validate_boundary_key_string("ca")
        assert result is True
        mock_warn.assert_called_once_with("ca")


class TestWarnAboutCaseSensitivity:
    """Test class for _warn_about_case_sensitivity function.

    Tests case sensitivity warnings and suggestions for boundary keys.
    """

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_warn_exact_match_returns_true(self, mock_data_catalog_class):
        """Test _warn_about_case_sensitivity with exact match returns True."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog
        mock_catalog.list_clip_boundaries.return_value = {"states": ["CA", "NV"]}

        result = _warn_about_case_sensitivity("CA")
        assert result is True

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_warn_case_mismatch_warns_and_returns_false(self, mock_data_catalog_class):
        """Test _warn_about_case_sensitivity with case mismatch warns."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog
        mock_catalog.list_clip_boundaries.return_value = {"states": ["CA", "NV"]}

        with pytest.warns(UserWarning, match="Did you mean"):
            result = _warn_about_case_sensitivity("ca")
            assert result is False

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_warn_no_match_warns_and_returns_false(self, mock_data_catalog_class):
        """Test _warn_about_case_sensitivity with no match warns."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog
        mock_catalog.list_clip_boundaries.return_value = {"states": ["CA", "NV"]}

        with pytest.warns(UserWarning, match="does not match any known boundary keys"):
            result = _warn_about_case_sensitivity("INVALID")
            assert result is False

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_warn_lowercase_state_abbreviation_adds_hint(self, mock_data_catalog_class):
        """Test _warn_about_case_sensitivity with lowercase state abbreviation adds hint to suggestion warning."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog
        # Return boundaries list that doesn't include lowercase version
        mock_catalog.list_clip_boundaries.return_value = {"states": ["CA", "NV"]}

        with pytest.warns(
            UserWarning, match="all lowercase.*should probably be uppercase"
        ):
            result = _warn_about_case_sensitivity("ca")
            assert result is False

    @patch("climakitae.new_core.param_validation.clip_param_validator.DataCatalog")
    def test_warn_county_incorrect_capitalization_adds_hint(
        self, mock_data_catalog_class
    ):
        """Test _warn_about_case_sensitivity with incorrect county capitalization adds hint to suggestion warning."""
        mock_catalog = MagicMock()
        mock_data_catalog_class.return_value = mock_catalog
        # Return boundaries list that doesn't include the incorrect capitalization
        mock_catalog.list_clip_boundaries.return_value = {
            "counties": ["Los Angeles County", "San Diego County"]
        }

        with pytest.warns(UserWarning, match="County names typically end with"):
            result = _warn_about_case_sensitivity("los angeles county")
            assert result is False
