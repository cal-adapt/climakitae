"""Unit tests for climakitae/new_core/param_validation/cadcat_preview_param_validator.py.

Tests focus on:
- Validator registration and lookup via the catalog validator registry
- Initialization against a mock DataCatalog
- Required-key validation (activity_id, table_id, grid_label, variable_id)
- Rejection of unsupported processors (warming_level, bias_adjust_model_to_station,
  filter_unadjusted_models)
- get_default_processors behavior for time vs sim concatenation
"""

import warnings
from unittest.mock import MagicMock

from climakitae.core.constants import CATALOG_CADCAT_PREVIEW, UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    _CATALOG_VALIDATOR_REGISTRY,
)
from climakitae.new_core.param_validation.cadcat_preview_param_validator import (
    CadcatPreviewValidator,
)

warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=DeprecationWarning
)


def _make_catalog():
    """Build a mock DataCatalog exposing a ``preview`` attribute."""
    mock_data_catalog = MagicMock()
    mock_preview = MagicMock()
    mock_preview.df.columns = [
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "member_id",
        "table_id",
        "variable_id",
        "grid_label",
    ]
    mock_data_catalog.preview = mock_preview
    return mock_data_catalog, mock_preview


class TestRegistration:
    """Validator is wired into the catalog validator registry."""

    def test_registered_under_cadcat_preview(self):
        assert CATALOG_CADCAT_PREVIEW in _CATALOG_VALIDATOR_REGISTRY
        assert _CATALOG_VALIDATOR_REGISTRY[CATALOG_CADCAT_PREVIEW] is (
            CadcatPreviewValidator
        )


class TestInit:
    """Initialization wires the preview catalog and keys correctly."""

    def test_init_sets_catalog_and_keys(self):
        mock_data_catalog, mock_preview = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        assert validator.catalog is mock_preview
        expected_keys = {
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "table_id",
            "grid_label",
            "variable_id",
        }
        assert set(validator.all_catalog_keys.keys()) == expected_keys
        assert all(v is UNSET for v in validator.all_catalog_keys.values())

    def test_init_sets_invalid_processors(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        assert validator.invalid_processors == [
            "warming_level",
            "bias_adjust_model_to_station",
            "filter_unadjusted_models",
        ]


class TestRequiredKeys:
    """is_valid_query enforces required keys."""

    def test_valid_query_passes_required_check(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        query = {
            "activity_id": "sup3r",
            "table_id": "1hr",
            "grid_label": "conus4km",
            "variable_id": "ws100",
            "processes": {},
        }
        assert validator._check_query_for_required_keys(query) is True

    def test_missing_required_key_fails(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        query = {
            "activity_id": "sup3r",
            "table_id": "1hr",
            "variable_id": "ws100",
            "processes": {},
        }
        assert validator._check_query_for_required_keys(query) is False

    def test_unset_sentinel_treated_as_missing(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        query = {
            "activity_id": UNSET,
            "table_id": "1hr",
            "grid_label": "conus4km",
            "variable_id": "ws100",
            "processes": {},
        }
        assert validator._check_query_for_required_keys(query) is False


class TestInvalidProcessors:
    """Preview validator rejects processors that aren't safe yet."""

    def test_warming_level_rejected(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        query = {"processes": {"warming_level": {"warming_levels": [2.0]}}}
        assert validator._check_query_invalid_processors(query) is False

    def test_bias_adjust_rejected(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        query = {"processes": {"bias_adjust_model_to_station": {"stations": ["KSAC"]}}}
        assert validator._check_query_invalid_processors(query) is False

    def test_filter_unadjusted_rejected(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        query = {"processes": {"filter_unadjusted_models": True}}
        assert validator._check_query_invalid_processors(query) is False

    def test_time_slice_allowed(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        query = {"processes": {"time_slice": ("2015-01-01", "2015-12-31")}}
        assert validator._check_query_invalid_processors(query) is True

    def test_empty_processes_allowed(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        assert validator._check_query_invalid_processors({"processes": {}}) is True
        assert validator._check_query_invalid_processors({}) is True


class TestDefaultProcessors:
    """get_default_processors produces the right concat dimension."""

    def test_ssp245_defaults_to_time_concat(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        defaults = validator.get_default_processors({"experiment_id": "ssp245"})
        assert defaults["concat"] == "time"
        assert defaults["drop_leap_days"] == "yes"

    def test_historical_defaults_to_sim_concat(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        defaults = validator.get_default_processors({"experiment_id": "historical"})
        assert defaults["concat"] == "sim"

    def test_historical_list_defaults_to_sim_concat(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        defaults = validator.get_default_processors(
            {"experiment_id": ["historical", "reanalysis"]}
        )
        assert defaults["concat"] == "sim"

    def test_mixed_list_with_ssp_defaults_to_time_concat(self):
        mock_data_catalog, _ = _make_catalog()
        validator = CadcatPreviewValidator(mock_data_catalog)
        defaults = validator.get_default_processors(
            {"experiment_id": ["historical", "ssp245"]}
        )
        assert defaults["concat"] == "time"
