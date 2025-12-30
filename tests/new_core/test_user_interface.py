"""
Unit tests for climakitae/new_core/user_interface.py

This module contains comprehensive unit tests for the ClimateData class
that provide the high-level interface for accessing climate data.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climakitae.core.constants import UNSET
from climakitae.new_core.user_interface import ClimateData


class TestClimateDataInit:
    """Test class for ClimateData initialization."""

    @patch("climakitae.new_core.user_interface.read_csv_file")
    @patch("climakitae.new_core.user_interface.DatasetFactory")
    def test_init_successful(self, mock_factory, mock_read_csv):
        """Test successful initialization."""
        mock_factory_instance = MagicMock()
        mock_factory.return_value = mock_factory_instance
        mock_read_csv.return_value = pd.DataFrame()

        with patch("climakitae.new_core.user_interface.logger") as mock_logger:
            climate_data = ClimateData()

        assert hasattr(climate_data, "_factory")
        assert hasattr(climate_data, "_query")
        assert hasattr(climate_data, "var_desc")
        # The implementation uses logger.info for the ready message
        mock_logger.info.assert_any_call("✅ Ready to query!")

    @patch("climakitae.new_core.user_interface.read_csv_file")
    @patch("climakitae.new_core.user_interface.DatasetFactory")
    def test_init_with_factory_error(self, mock_factory, mock_read_csv):
        """Test initialization when DatasetFactory raises an exception."""
        mock_factory.side_effect = Exception("Factory error")

        with patch("climakitae.new_core.user_interface.logger") as mock_logger:
            climate_data = ClimateData()

        # Should handle the error gracefully using logger.error
        # The implementation uses logger.error with exc_info=True
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "❌ Setup failed:" in call_args[0][0]
        assert "Factory error" in str(call_args)


class TestClimateDataParameterSetters:
    """Test class for parameter setting methods."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()

    def test_catalog_valid(self):
        """Test catalog setter with valid input."""
        result = self.climate_data.catalog("test_catalog")
        assert self.climate_data._query["catalog"] == "test_catalog"
        assert result is self.climate_data

    def test_catalog_invalid_empty_string(self):
        """Test catalog setter with empty string."""
        try:
            self.climate_data.catalog("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Catalog must be a non-empty string" in str(e)

    def test_installation_valid(self):
        """Test installation setter with valid input."""
        result = self.climate_data.installation("pv_utility")
        assert self.climate_data._query["installation"] == "pv_utility"
        assert result is self.climate_data

    def test_installation_invalid(self):
        """Test installation setter with invalid input."""
        try:
            self.climate_data.installation("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Installation must be a non-empty string" in str(e)

    def test_activity_id_valid(self):
        """Test activity_id setter with valid input."""
        result = self.climate_data.activity_id("CMIP6")
        assert self.climate_data._query["activity_id"] == "CMIP6"
        assert result is self.climate_data

    def test_activity_id_invalid(self):
        """Test activity_id setter with invalid input."""
        try:
            self.climate_data.activity_id("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Activity ID must be a non-empty string" in str(e)

    def test_institution_id_valid(self):
        """Test institution_id setter with valid input."""
        result = self.climate_data.institution_id("CNRM")
        assert self.climate_data._query["institution_id"] == "CNRM"
        assert result is self.climate_data

    def test_institution_id_invalid(self):
        """Test institution_id setter with invalid input."""
        try:
            self.climate_data.institution_id("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Institution ID must be a non-empty string" in str(e)

    def test_source_id_valid(self):
        """Test source_id setter with valid input."""
        result = self.climate_data.source_id("GCM")
        assert self.climate_data._query["source_id"] == "GCM"
        assert result is self.climate_data

    def test_source_id_invalid(self):
        """Test source_id setter with invalid input."""
        try:
            self.climate_data.source_id("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Source ID must be a non-empty string" in str(e)

    def test_table_id_valid(self):
        """Test table_id setter with valid input."""
        result = self.climate_data.table_id("day")
        assert self.climate_data._query["table_id"] == "day"
        assert result is self.climate_data

    def test_table_id_invalid(self):
        """Test table_id setter with invalid input."""
        try:
            self.climate_data.table_id("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Table ID must be a non-empty string" in str(e)

    def test_grid_label_valid(self):
        """Test grid_label setter with valid input."""
        result = self.climate_data.grid_label("d03")
        assert self.climate_data._query["grid_label"] == "d03"
        assert result is self.climate_data

    def test_grid_label_invalid(self):
        """Test grid_label setter with invalid input."""
        try:
            self.climate_data.grid_label("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Grid label must be a non-empty string" in str(e)

    def test_variable_valid(self):
        """Test variable setter with valid input."""
        result = self.climate_data.variable("tasmax")
        assert self.climate_data._query["variable_id"] == "tasmax"
        assert result is self.climate_data

    def test_variable_invalid(self):
        """Test variable setter with invalid input."""
        try:
            self.climate_data.variable("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Variable must be a non-empty string" in str(e)

    def test_experiment_id_string_valid(self):
        """Test experiment_id setter with valid string."""
        result = self.climate_data.experiment_id("historical")
        assert self.climate_data._query["experiment_id"] == ["historical"]
        assert result is self.climate_data

    def test_experiment_id_list_valid(self):
        """Test experiment_id setter with valid list."""
        result = self.climate_data.experiment_id(["historical", "ssp245"])
        assert self.climate_data._query["experiment_id"] == ["historical", "ssp245"]
        assert result is self.climate_data

    def test_experiment_id_invalid_type(self):
        """Test experiment_id setter with invalid type."""
        try:
            self.climate_data.experiment_id(123)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Experiment ID must be a non-empty string or list of strings" in str(
                e
            )

    def test_station_id_string_valid(self):
        """Test station_id setter with valid string."""
        result = self.climate_data.station_id("CWOP_E4369")
        assert self.climate_data._query["station_id"] == ["CWOP_E4369"]
        assert result is self.climate_data

    def test_station_id_list_valid(self):
        """Test station_id setter with valid list."""
        result = self.climate_data.station_id(["CWOP_E4369", "CWOP_F5680"])
        assert self.climate_data._query["station_id"] == ["CWOP_E4369", "CWOP_F5680"]
        assert result is self.climate_data

    def test_station_id_invalid_type(self):
        """Test station_id setter with invalid type."""
        try:
            self.climate_data.station_id(123)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Station ID must be a non-empty string or list of strings" in str(e)

    def test_network_id_string_valid(self):
        """Test network_id setter with valid string."""
        result = self.climate_data.network_id("ASOSAWOS")
        assert self.climate_data._query["network_id"] == "ASOSAWOS"
        assert result is self.climate_data

    def test_network_id_list_valid(self):
        """Test network_id setter with valid list."""
        result = self.climate_data.network_id(["ASOSAWOS", "SCAN"])
        assert self.climate_data._query["network_id"] == ["ASOSAWOS", "SCAN"]
        assert result is self.climate_data

    def test_network_id_invalid_type(self):
        """Test network_id setter with invalid type."""
        try:
            self.climate_data.network_id(123)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Network ID must be a non-empty string or list of strings" in str(e)

    def test_processes_valid(self):
        """Test processes setter with valid input."""
        processes = {"spatial_avg": "region", "temporal_avg": "monthly"}
        result = self.climate_data.processes(processes)
        assert self.climate_data._query["processes"] == processes
        assert result is self.climate_data

    def test_processes_invalid_type(self):
        """Test processes setter with invalid type."""
        try:
            self.climate_data.processes("not_a_dict")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Processes must be a dictionary" in str(e)


class TestClimateDataGet:
    """Test class for get method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()
            self.climate_data._factory = mock_factory_instance

    def test_get_missing_required_parameters(self):
        """Test get method when required parameters are missing."""
        with patch("builtins.print") as mock_print:
            result = self.climate_data.get()

        # Correctly returns None when validation fails
        assert result is None
        # Should print error about missing parameters
        printed_text = "".join(str(call) for call in mock_print.call_args_list)
        assert "Missing required parameters" in printed_text

    def test_get_successful_execution(self):
        """Test successful get execution."""
        # Set required parameters
        self.climate_data._query = {
            "catalog": "climate",
            "installation": UNSET,
            "activity_id": UNSET,
            "institution_id": UNSET,
            "source_id": UNSET,
            "experiment_id": UNSET,
            "station_id": UNSET,
            "network_id": UNSET,
            "table_id": "day",
            "grid_label": "d03",
            "variable_id": "tas",
            "processes": UNSET,
        }

        # Mock dataset and execution
        mock_dataset = MagicMock()
        expected_data = MagicMock()
        expected_data.nbytes = 100  # Non-empty data
        mock_dataset.execute.return_value = expected_data
        self.climate_data._factory.create_dataset.return_value = mock_dataset

        with patch("climakitae.new_core.user_interface.logger") as mock_logger:
            result = self.climate_data.get()

        assert result is expected_data
        # The implementation uses logger.info for success message
        mock_logger.info.assert_any_call("✅ Data retrieval successful!")


class TestClimateDataValidation:
    """Test class for parameter validation."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()

    def test_validate_all_required_present(self):
        """Test validation when all required parameters are present."""
        query = {
            "catalog": "climate",
            "installation": UNSET,
            "activity_id": UNSET,
            "institution_id": UNSET,
            "source_id": UNSET,
            "experiment_id": UNSET,
            "table_id": "day",
            "grid_label": "d03",
            "variable_id": "tas",
            "processes": UNSET,
        }

        result = self.climate_data._validate_required_parameters(query)
        assert result is True


class TestClimateDataOptionMethods:
    """Test class for option exploration methods."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()
            self.climate_data._factory = mock_factory_instance

    def test_show_catalog_options(self):
        """Test show_catalog_options method."""
        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_catalog_options()

        mock_show.assert_called_once_with(
            "catalog", "catalog options (Cloud data collections)", limit_per_group=None
        )

    @pytest.mark.parametrize(
        "catalog,expected_processors,should_call_get_valid",
        [
            (UNSET, ["clip", "concat"], False),  # No catalog set
            ("hdp", ["concat", "export"], True),  # Catalog set
        ],
    )
    def test_show_processors(self, catalog, expected_processors, should_call_get_valid):
        """Test show_processors method with and without catalog."""
        if catalog is UNSET:
            # Mock processor registry for UNSET case
            self.climate_data._factory._processing_step_registry = {
                proc: (MagicMock(), 10) for proc in expected_processors
            }
        else:
            # Set catalog and mock get_valid_processors
            self.climate_data._query["catalog"] = catalog
            self.climate_data._factory.get_valid_processors = MagicMock(
                return_value=expected_processors
            )

        with patch("builtins.print") as mock_print:
            self.climate_data.show_processors()

        # Verify get_valid_processors was called appropriately
        if should_call_get_valid:
            self.climate_data._factory.get_valid_processors.assert_called_once_with(
                catalog
            )

        # Verify printed output
        printed_text = "".join(str(call) for call in mock_print.call_args_list)
        assert "Processors" in printed_text
        for proc in expected_processors:
            assert proc in printed_text


class TestClimateDataConvenienceMethods:
    """Test class for convenience methods."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()

    def test_reset(self):
        """Test reset convenience method."""
        # Set some parameters
        self.climate_data._query["catalog"] = "test"

        result = self.climate_data.reset()

        # Should reset and return self
        assert result is self.climate_data
        assert self.climate_data._query["catalog"] is UNSET

    def test_copy_query(self):
        """Test copy_query method."""
        self.climate_data._query["catalog"] = "climate"
        self.climate_data._query["variable_id"] = "tas"

        result = self.climate_data.copy_query()

        expected = {"catalog": "climate", "variable_id": "tas"}
        assert result == expected


class TestClimateDataChaining:
    """Test class for method chaining functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()

    def test_method_chaining(self):
        """Test that methods can be chained together."""
        result = (
            self.climate_data.catalog("climate")
            .variable("tas")
            .table_id("day")
            .grid_label("d03")
        )

        # Should return the same instance
        assert result is self.climate_data

        # Parameters should be set correctly
        assert self.climate_data._query["catalog"] == "climate"
        assert self.climate_data._query["variable_id"] == "tas"
        assert self.climate_data._query["table_id"] == "day"
        assert self.climate_data._query["grid_label"] == "d03"


class TestClimateDataAdditionalShowMethods:
    """Test class for additional show methods."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()

    def test_show_query(self):
        """Test show_query method."""
        self.climate_data._query["catalog"] = "test_catalog"
        self.climate_data._query["variable_id"] = "tasmax"

        with patch("builtins.print") as mock_print:
            self.climate_data.show_query()

        mock_print.assert_called()
        printed_text = "".join(str(call) for call in mock_print.call_args_list)
        assert "Current Query" in printed_text

    def test_show_installation_options(self):
        """Test show_installation_options method."""
        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_installation_options()
            mock_show.assert_called_once_with(
                "installation",
                "installation options (Renewable energy generation types)",
                limit_per_group=None,
            )

    def test_show_activity_id_options(self):
        """Test show_activity_id_options method."""
        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_activity_id_options()
            mock_show.assert_called_once_with(
                "activity_id",
                "activity_id options (Downscaling methods)",
                limit_per_group=None,
            )

    def test_show_institution_id_options(self):
        """Test show_institution_id_options method."""
        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_institution_id_options()
            mock_show.assert_called_once_with(
                "institution_id",
                "institution_id options (Data producers)",
                limit_per_group=None,
            )

    def test_show_source_id_options(self):
        """Test show_source_id_options method."""
        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_source_id_options()
            mock_show.assert_called_once_with(
                "source_id",
                "source_id options (Climate model simulations)",
                limit_per_group=None,
            )

    def test_show_experiment_id_options(self):
        """Test show_experiment_id_options method."""
        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_experiment_id_options()
            mock_show.assert_called_once_with(
                "experiment_id",
                "experiment_id options (Simulation runs)",
                limit_per_group=None,
            )

    def test_show_table_id_options(self):
        """Test show_table_id_options method."""
        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_table_id_options()
            mock_show.assert_called_once_with(
                "table_id",
                "table_id options (Temporal resolutions)",
                limit_per_group=None,
            )

    def test_show_grid_label_options(self):
        """Test show_grid_label_options method."""
        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_grid_label_options()
            mock_show.assert_called_once_with(
                "grid_label",
                "grid_label options (Spatial resolutions)",
                limit_per_group=None,
            )

    def test_show_variable_options(self):
        """Test show_variable_options method."""
        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_variable_options()
            mock_show.assert_called_once_with(
                "variable_id", "Variables", limit_per_group=None
            )

    def test_show_station_options(self):
        """Test show_station_options method."""
        with patch.object(
            self.climate_data._factory,
            "get_stations",
            return_value=["station1", "station2"],
        ) as mock_get_stations:
            with patch("builtins.print") as mock_print:
                self.climate_data.show_station_options()

            mock_get_stations.assert_called_once()
            mock_print.assert_called()

    def test_show_boundary_options(self):
        """Test show_boundary_options method."""
        with patch.object(
            self.climate_data._factory, "get_boundaries", return_value=["CA", "US"]
        ) as mock_get_boundaries:
            with patch("builtins.print") as mock_print:
                self.climate_data.show_boundary_options()

            mock_get_boundaries.assert_called_once()
            mock_print.assert_called()

    def test_show_all_options(self):
        """Test show_all_options method."""
        with (
            patch.object(self.climate_data, "show_catalog_options") as mock_catalog,
            patch.object(
                self.climate_data, "show_installation_options"
            ) as mock_installation,
            patch.object(
                self.climate_data, "show_activity_id_options"
            ) as mock_activity,
            patch.object(
                self.climate_data, "show_institution_id_options"
            ) as mock_institution,
            patch.object(self.climate_data, "show_source_id_options") as mock_source,
            patch.object(
                self.climate_data, "show_experiment_id_options"
            ) as mock_experiment,
            patch.object(self.climate_data, "show_table_id_options") as mock_table,
            patch.object(self.climate_data, "show_grid_label_options") as mock_grid,
            patch.object(self.climate_data, "show_variable_options") as mock_variable,
            patch.object(
                self.climate_data, "show_station_id_options"
            ) as mock_station_id,
            patch.object(
                self.climate_data, "show_network_id_options"
            ) as mock_network_id,
        ):

            self.climate_data.show_all_options()

            mock_catalog.assert_called_once()
            mock_installation.assert_called_once()
            mock_activity.assert_called_once()
            mock_institution.assert_called_once()
            mock_source.assert_called_once()
            mock_experiment.assert_called_once()
            mock_table.assert_called_once()
            mock_grid.assert_called_once()
            mock_variable.assert_called_once()
            mock_station_id.assert_called_once()
            mock_network_id.assert_called_once()

    def test_show_options_private_method(self):
        """Test _show_options private method."""
        with patch.object(
            self.climate_data._factory,
            "get_catalog_options",
            return_value=["option1", "option2"],
        ) as mock_get_options:
            with patch("builtins.print") as mock_print:
                self.climate_data._show_options("test_field", "test options")

            mock_get_options.assert_called_once_with("test_field", {})
            mock_print.assert_called()

    def test_format_option_string(self):
        """Test _format_option with string input."""
        result = self.climate_data._format_option("test_option", "test")
        assert result == "test_option"

    def test_format_option_list(self):
        """Test _format_option with list input."""
        # Note: _format_option expects a string, but testing edge case behavior
        result = self.climate_data._format_option("option1,option2", "test")
        assert result == "option1,option2"

    def test_format_option_with_spacing(self):
        """Test _format_option with spacing parameter."""
        result = self.climate_data._format_option("test_option", "test", spacing=4)
        assert result == "test_option"


class TestClimateDataAdditionalMethods:
    """Test class for additional methods like load_query and _reset_query."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()

    def test_load_query(self):
        """Test load_query method."""
        test_query = {
            "catalog": "climate",
            "variable_id": "tas",
            "experiment_id": ["historical"],
        }

        result = self.climate_data.load_query(test_query)

        assert result is self.climate_data
        assert self.climate_data._query["catalog"] == "climate"
        assert self.climate_data._query["variable_id"] == "tas"
        assert self.climate_data._query["experiment_id"] == ["historical"]

    def test_load_query_empty_dict(self):
        """Test load_query with empty dictionary."""
        result = self.climate_data.load_query({})
        assert result is self.climate_data

    def test_reset_query_private(self):
        """Test _reset_query private method."""
        # Set some query parameters first
        self.climate_data._query["catalog"] = "test"
        self.climate_data._query["variable_id"] = "tasmax"

        # Call private reset method
        self.climate_data._reset_query()

        # All parameters should be reset to UNSET
        assert self.climate_data._query["catalog"] is UNSET
        assert self.climate_data._query["variable_id"] is UNSET


class TestClimateDataVerbosity:
    """Test class for verbosity and logging configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()

    def test_verbosity_invalid_type(self):
        """Test verbosity setter with invalid type raises ValueError."""
        try:
            self.climate_data.verbosity("not_an_int")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Verbosity level must be an integer" in str(e)

    def test_verbosity_valid_levels(self):
        """Test verbosity setter with valid integer levels."""
        # Test various valid levels
        result = self.climate_data.verbosity(-2)
        assert result is self.climate_data

        result = self.climate_data.verbosity(-1)
        assert result is self.climate_data

        result = self.climate_data.verbosity(0)
        assert result is self.climate_data

        result = self.climate_data.verbosity(1)
        assert result is self.climate_data


class TestClimateDataExperimentIdEdgeCases:
    """Test class for experiment_id edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()

    def test_experiment_id_empty_string(self):
        """Test experiment_id with empty string raises ValueError."""
        try:
            self.climate_data.experiment_id("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Experiment ID must be a non-empty string" in str(e)

    def test_experiment_id_list_with_empty_string(self):
        """Test experiment_id list containing empty string raises ValueError."""
        try:
            self.climate_data.experiment_id(["historical", ""])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Each experiment ID must be a non-empty string" in str(e)

    def test_experiment_id_list_with_non_string(self):
        """Test experiment_id list containing non-string raises ValueError."""
        try:
            self.climate_data.experiment_id(["historical", 123])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Each experiment ID must be a non-empty string" in str(e)


class TestClimateDataGetErrorHandling:
    """Test class for get() error handling paths."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()
            self.climate_data._factory = mock_factory_instance

    def test_get_dataset_creation_error(self):
        """Test get() when dataset creation raises an error."""
        self.climate_data._query = {
            "catalog": "climate",
            "installation": UNSET,
            "activity_id": UNSET,
            "institution_id": UNSET,
            "source_id": UNSET,
            "experiment_id": UNSET,
            "table_id": "day",
            "grid_label": "d03",
            "variable_id": "tas",
            "processes": UNSET,
        }

        self.climate_data._factory.create_dataset.side_effect = ValueError(
            "Dataset creation failed"
        )

        result = self.climate_data.get()

        assert result is None

    def test_get_dataset_execute_error(self):
        """Test get() when dataset execution raises an error."""
        self.climate_data._query = {
            "catalog": "climate",
            "installation": UNSET,
            "activity_id": UNSET,
            "institution_id": UNSET,
            "source_id": UNSET,
            "experiment_id": UNSET,
            "table_id": "day",
            "grid_label": "d03",
            "variable_id": "tas",
            "processes": UNSET,
        }

        mock_dataset = MagicMock()
        mock_dataset.execute.side_effect = RuntimeError("Execution failed")
        self.climate_data._factory.create_dataset.return_value = mock_dataset

        result = self.climate_data.get()

        # Should return None (data not assigned before exception)
        assert result is None


class TestClimateDataFormatOption:
    """Test class for _format_option method."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()

    def test_format_option_grid_label_d01(self):
        """Test _format_option with grid_label d01."""
        result = self.climate_data._format_option("d01", "grid_label")
        assert result == "d01 (45 km)"

    def test_format_option_grid_label_d02(self):
        """Test _format_option with grid_label d02."""
        result = self.climate_data._format_option("d02", "grid_label")
        assert result == "d02 ( 9 km)"

    def test_format_option_grid_label_d03(self):
        """Test _format_option with grid_label d03."""
        result = self.climate_data._format_option("d03", "grid_label")
        assert result == "d03 ( 3 km)"

    def test_format_option_grid_label_unknown(self):
        """Test _format_option with unknown grid_label."""
        result = self.climate_data._format_option("d99", "grid_label")
        assert result == "d99 (Unknown)"

    def test_format_option_variable_id_found(self):
        """Test _format_option with variable_id that exists in var_desc."""
        self.climate_data.var_desc = pd.DataFrame(
            {"variable_id": ["tas"], "display_name": ["Air Temperature"]}
        )
        result = self.climate_data._format_option("tas", "variable_id", spacing=4)
        assert "tas:" in result
        assert "Air Temperature" in result

    def test_format_option_variable_id_not_found(self):
        """Test _format_option with variable_id not in var_desc."""
        self.climate_data.var_desc = pd.DataFrame(
            {"variable_id": ["tas"], "display_name": ["Air Temperature"]}
        )
        result = self.climate_data._format_option("unknown_var", "variable_id")
        assert "unknown_var:" in result
        assert "No description available" in result


class TestClimateDataShowOptionsExceptionHandling:
    """Test class for exception handling in show_* methods."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with (
            patch(
                "climakitae.new_core.user_interface.DatasetFactory",
                return_value=mock_factory_instance,
            ),
            patch(
                "climakitae.new_core.user_interface.read_csv_file",
                return_value=pd.DataFrame(),
            ),
            patch("builtins.print"),
        ):
            self.climate_data = ClimateData()
            self.climate_data._factory = mock_factory_instance

    def test_show_processors_exception(self):
        """Test show_processors handles exception gracefully."""
        self.climate_data._factory.get_processors.side_effect = Exception(
            "Processor error"
        )

        # Should not raise, just log the error
        self.climate_data.show_processors()

    def test_show_station_options_exception(self):
        """Test show_station_options handles exception gracefully."""
        self.climate_data._factory.get_stations.side_effect = Exception("Station error")

        # Should not raise, just log the error
        self.climate_data.show_station_options()

    def test_show_boundary_options_with_type(self):
        """Test show_boundary_options with specific boundary_type parameter."""
        self.climate_data._factory.get_boundaries.return_value = ["CA", "NV"]

        with patch("builtins.print"):
            self.climate_data.show_boundary_options(boundary_type="us_states")

        self.climate_data._factory.get_boundaries.assert_called_with("us_states")

    def test_show_boundary_options_exception(self):
        """Test show_boundary_options handles exception gracefully."""
        self.climate_data._factory.get_boundaries.side_effect = Exception(
            "Boundary error"
        )

        # Should not raise, just log the error
        self.climate_data.show_boundary_options()

    def test_show_options_exception(self):
        """Test _show_options handles exception gracefully."""
        self.climate_data._factory.get_catalog_options.side_effect = Exception(
            "Options error"
        )

        # Should not raise, just log the error
        with patch("builtins.print"):
            self.climate_data._show_options("catalog", "Test Options")

    def test_show_variable_options_with_query(self):
        """Test show_variable_options with existing query parameters."""
        self.climate_data._query["catalog"] = "cadcat"

        with patch.object(self.climate_data, "_show_options") as mock_show:
            self.climate_data.show_variable_options()

        mock_show.assert_called_once_with(
            "variable_id",
            "Variables (constrained by current query):",
            limit_per_group=None,
        )
