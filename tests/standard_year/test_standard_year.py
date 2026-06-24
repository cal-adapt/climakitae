"""
Unit tests for climakitae/explore/standard_year_profile.py

This module contains comprehensive unit tests for the Standard Year and climate
profile computation functions that provide climate profile analysis.

"""

from typing import Tuple
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.explore.standard_year_profile import (
    _compute_difference_profile,
    _compute_paired_difference,
    _construct_profile_dataframe,
    _convert_stations_to_lat_lon,
    _create_multi_wl_multi_sim_dataframe,
    _create_multi_wl_single_sim_dataframe,
    _create_simple_dataframe,
    _create_single_wl_multi_sim_dataframe,
    _get_buffer_from_resolution,
    _get_clean_standardyr_filename,
    _get_station_coordinates,
    _handle_location_params,
    _stack_profile_data,
    compute_profile,
    export_profile_to_csv,
    get_climate_profile,
    get_profile_metadata,
    get_profile_units,
    retrieve_profile_data,
    set_profile_metadata,
)


class TestGetClimateProfile:
    """Test class for get_climate_profile function.

    Tests the high-level function that combines data retrieval and profile
    computation to produce climate profiles as DataFrames.

    Attributes
    ----------
    mock_retrieve_profile_data : MagicMock
        Mock for retrieve_profile_data function.
    mock_compute_profile : MagicMock
        Mock for compute_profile function.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_retrieve_patcher = patch(
            "climakitae.explore.standard_year_profile.retrieve_profile_data"
        )
        self.mock_compute_patcher = patch(
            "climakitae.explore.standard_year_profile.compute_profile"
        )
        self.mock_print_patcher = patch("builtins.print")

        self.mock_retrieve_profile_data = self.mock_retrieve_patcher.start()
        self.mock_compute_profile = self.mock_compute_patcher.start()
        self.mock_print = self.mock_print_patcher.start()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.mock_retrieve_patcher.stop()
        self.mock_compute_patcher.stop()
        self.mock_print_patcher.stop()

    def test_get_climate_profile_returns_dataframe(self):
        """Test that get_climate_profile returns a pandas DataFrame."""
        # Setup mock datasets with proper data_vars
        mock_historic_data = MagicMock(spec=xr.Dataset)
        mock_historic_data.data_vars = {"tasmax": MagicMock()}
        mock_future_data = MagicMock(spec=xr.Dataset)
        mock_future_data.data_vars = {"tasmax": MagicMock()}

        self.mock_retrieve_profile_data.return_value = (
            mock_historic_data,
            mock_future_data,
        )

        # Create mock profile DataFrames
        mock_future_profile = pd.DataFrame(np.random.rand(8760, 7))
        mock_historic_profile = pd.DataFrame(np.random.rand(8760, 7))
        self.mock_compute_profile.side_effect = [
            mock_future_profile,
            mock_historic_profile,
        ]

        # Execute function
        result = get_climate_profile(warming_level=[2.0])

        # Verify outcome: returns a DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] > 0, "DataFrame should have rows"
        assert result.shape[1] > 0, "DataFrame should have columns"

    def test_get_climate_profile_with_no_delta_returns_raw_future(self):
        """Test that get_climate_profile returns raw future profile when no_delta=True."""
        # Setup mock datasets with proper data_vars
        mock_future_data = MagicMock(spec=xr.Dataset)
        mock_future_data.data_vars = {"tasmax": MagicMock()}

        # When no_delta=True, only future data is returned, historic is None
        self.mock_retrieve_profile_data.return_value = (None, mock_future_data)

        # Create mock future profile
        mock_future_profile = pd.DataFrame(np.random.rand(8760, 7))
        self.mock_compute_profile.return_value = mock_future_profile

        # Execute function with no_delta=True
        result = get_climate_profile(warming_level=[2.0], no_delta=True)

        # Verify outcome: returns the raw future profile (no difference calculation)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == mock_future_profile.shape
        ), "Should return the original future profile shape"
        # Verify compute_profile was called only once (for future data only)
        assert (
            self.mock_compute_profile.call_count == 1
        ), "Should call compute_profile only once for future data"

    def test_get_climate_profile_raises_error_when_no_data_returned(self):
        """Test that get_climate_profile raises ValueError when no data is retrieved."""
        # Setup scenario where both datasets are None
        self.mock_retrieve_profile_data.return_value = (None, None)

        # Execute and verify outcome: should raise ValueError
        with pytest.raises(
            ValueError,
            match="No data returned for either historical or future datasets",
        ):
            get_climate_profile(warming_level=[2.0])


@pytest.mark.advanced
class TestComputeProfile:
    """Test class for compute_profile function.

    Tests the core function that computes climate profiles from xarray DataArrays
    using quantile-based analysis across multiple years of data.

    Attributes
    ----------
    sample_data : xr.DataArray
        Sample climate data for testing.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create smaller sample for faster testing (just enough for the algorithm)
        time_delta = pd.date_range(
            "2020-01-01", periods=8760, freq="h"
        )  # 1 year of hourly data
        warming_levels = [1.5]
        simulations = ["sim1"]

        # Create test data with proper dimensions
        data = np.random.rand(len(warming_levels), len(time_delta), len(simulations))

        self.sample_data = xr.DataArray(
            data,
            dims=["warming_level", "time_delta", "simulation"],
            coords={
                "warming_level": warming_levels,
                "time_delta": time_delta,
                "simulation": simulations,
            },
            attrs={"units": "degC", "variable_id": "tasmax"},
        )

    def test_compute_profile_returns_dataframe_with_correct_shape(self):
        """Test that compute_profile returns DataFrame with expected dimensions."""
        # Execute function
        result = compute_profile(self.sample_data, q=0.5)

        # Verify outcome: returns DataFrame with correct shape
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == 8760, "Should have 8760 rows (hours)"
        assert result.shape[1] > 0, "Should have columns for hours and other dimensions"

        # Verify the DataFrame has proper index and column structure
        assert result.index.dtype == int or isinstance(
            result.index, pd.Index
        ), "Should have proper index"
        assert hasattr(result, "attrs"), "Should preserve metadata in attrs"

    def test_compute_profile_respects_days_in_year_parameter(self):
        """Test that compute_profile creates DataFrame with specified number of days."""
        # Execute function with regular year (365 days) - this should work with 8760 hours
        result_8760 = compute_profile(self.sample_data, q=0.5)

        # Verify outcome: correct number of rows based on days_in_year
        assert result_8760.shape[0] == 8760, "Should have 365 rows for regular year"
        assert result_8760.shape[1] == 1, "Should have 1 column for 1 simulation"

    def test_compute_profile_preserves_metadata_from_input(self):
        """Test that compute_profile preserves important metadata from input DataArray."""
        # Execute function
        result = compute_profile(self.sample_data, q=0.75)

        # Verify outcome: metadata is preserved and enhanced
        assert "units" in result.attrs, "Should preserve units from input data"
        assert "quantile" in result.attrs, "Should include quantile information"
        assert "method" in result.attrs, "Should include method description"
        assert (
            result.attrs["quantile"] == 0.75
        ), "Should record the correct quantile used"
        assert result.attrs["units"] == "degC", "Should preserve original units"


@pytest.mark.advanced
class TestGetSimulationLabel:
    """Test class for _get_simulation_label helper function behavior.

    Tests the simulation label extraction logic that handles WRF simulation
    name parsing, including GCM, parameters, and SSP scenario extraction.
    Since _get_simulation_label is a nested function, we test it through
    compute_profile's behavior with different simulation naming patterns.

    Attributes
    ----------
    base_data_shape : tuple
        Shape for creating test data arrays.
    time_delta : pd.DatetimeIndex
        Time coordinate for test data.
    warming_levels : list
        Warming level values for test data.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create base components for test data
        self.time_delta = pd.date_range("2020-01-01", periods=8760, freq="h")
        self.warming_levels = [1.5]
        self.base_data_shape = (len(self.warming_levels), len(self.time_delta))

    def _create_test_dataarray(self, simulations):
        """Helper to create test DataArray with specified simulations."""
        data = np.random.rand(
            len(self.warming_levels), len(self.time_delta), len(simulations)
        )
        return xr.DataArray(
            data,
            dims=["warming_level", "time_delta", "simulation"],
            coords={
                "warming_level": self.warming_levels,
                "time_delta": self.time_delta,
                "simulation": simulations,
            },
            attrs={"units": "degC"},
        )

    def test_wrf_simulation_label_full_format(self):
        """Test WRF simulation name parsing with full format: WRF_GCM_params_scenario."""
        # Create data with multiple WRF simulations to get MultiIndex
        simulations = [
            "WRF_CESM2_r11i1p1f1_historical+ssp245",
            "WRF_CESM2_r11i1p1f1_historical+ssp370",
        ]
        test_data = self._create_test_dataarray(simulations)

        # Execute function
        result = compute_profile(test_data, q=0.5)

        # Verify outcome: simulation labels are correctly parsed
        assert isinstance(result.columns, pd.Index), "Should have simple Index"
        sim_names = result.columns.unique()
        assert len(sim_names) == 2, "Should have two simulations"

        #! sim names parsed incorrectly
        # Should extract: CESM2-r11i1p1f1-ssp245 and CESM2-r11i1p1f1-ssp370
        sim_labels = sorted(sim_names)
        assert (
            "CESM2-r11i1p1f1-ssp245" in sim_labels
        ), "Should parse first simulation correctly"
        assert (
            "CESM2-r11i1p1f1-ssp370" in sim_labels
        ), "Should parse second simulation correctly"
        assert all("CESM2" in label for label in sim_labels), "Should extract GCM name"
        assert all(
            "r11i1p1f1" in label for label in sim_labels
        ), "Should extract parameters"

    def test_wrf_simulation_label_with_cnrm_esm(self):
        """Test WRF simulation parsing with hyphenated GCM name."""
        # Create data with CNRM-ESM2-1 GCM (contains hyphens) - use multiple sims
        simulations = [
            "WRF_CNRM-ESM2-1_r1i1p1f2_historical+ssp370",
            "WRF_GFDL-CM4_r1i1p1f1_historical+ssp245",
        ]
        test_data = self._create_test_dataarray(simulations)

        # Execute function
        result = compute_profile(test_data, q=0.5)

        # Verify outcome: handles hyphenated GCM names correctly
        sim_names = result.columns.unique()
        assert len(sim_names) == 2, "Should have two simulations"
        sim_labels = list(sim_names)

        # Find the CNRM label
        cnrm_label = [l for l in sim_labels if "CNRM" in l][0]
        assert "CNRM-ESM2-1" in cnrm_label, "Should preserve hyphens in GCM name"
        assert "r1i1p1f2" in cnrm_label, "Should extract parameters"
        assert "ssp370" in cnrm_label, "Should extract ssp370"
        assert (
            cnrm_label == "CNRM-ESM2-1-r1i1p1f2-ssp370"
        ), "Should format correctly with hyphenated GCM"

    def test_wrf_simulation_label_historical_only(self):
        """Test WRF simulation parsing with historical-only scenario (no SSP)."""
        # Create data with historical-only scenarios - use multiple
        simulations = [
            "WRF_GFDL-CM4_r1i1p1f1_historical",
            "WRF_MPI-ESM1-2-HR_r3i1p1f1_historical",
        ]
        test_data = self._create_test_dataarray(simulations)

        # Execute function
        result = compute_profile(test_data, q=0.5)

        # Verify outcome: uses 'hist' fallback for historical-only
        sim_names = result.columns.unique()
        assert len(sim_names) == 2, "Should have two simulations"

        # Check both use hist fallback
        gfdl_label = [label for label in sim_names if "GFDL" in label][0]
        assert "GFDL-CM4" in gfdl_label, "Should extract GCM name"
        assert "r1i1p1f1" in gfdl_label, "Should extract parameters"
        assert "hist" in gfdl_label, "Should use 'hist' fallback for historical-only"
        assert gfdl_label == "GFDL-CM4-r1i1p1f1-hist", "Should format with hist suffix"

    def test_wrf_simulation_label_ssp585(self):
        """Test WRF simulation parsing extracts ssp585 correctly."""
        # Create data with ssp585 and ssp370 scenarios - use multiple
        simulations = [
            "WRF_EC-Earth3_r4i1p1f1_historical+ssp585",
            "WRF_EC-Earth3_r4i1p1f1_historical+ssp370",
        ]
        test_data = self._create_test_dataarray(simulations)

        # Execute function
        result = compute_profile(test_data, q=0.5)

        # Verify outcome: correctly extracts ssp585
        sim_names = result.columns.unique()
        assert len(sim_names) == 2, "Should have two simulations"

        # Find ssp585 label
        ssp585_label = [label for label in sim_names if "ssp585" in label][0]
        assert "ssp585" in ssp585_label, "Should extract ssp585 from scenario"
        assert ssp585_label == "EC-Earth3-r4i1p1f1-ssp585", "Should format with ssp585"

    def test_wrf_simulation_label_short_format_fallback(self):
        """Test WRF simulation with fewer than 4 parts uses fallback format."""
        # Create data with shorter WRF formats (fewer than 4 parts) - use multiple
        simulations = ["WRF_CESM2_run1", "WRF_GFDL_run2"]
        test_data = self._create_test_dataarray(simulations)

        # Execute function
        result = compute_profile(test_data, q=0.5)

        # Verify outcome: uses fallback format with index
        sim_names = result.columns.unique()
        assert len(sim_names) == 2, "Should have two simulations"
        sim_labels = sorted(sim_names)

        # Check fallback format: GCM-index
        assert any(
            "CESM2" in label for label in sim_labels
        ), "Should extract CESM2 from second part"
        assert any(
            "GFDL" in label for label in sim_labels
        ), "Should extract GFDL from second part"
        assert all(
            "-" in label for label in sim_labels
        ), "Should append index with hyphen"

    def test_non_wrf_simulation_label(self):
        """Test non-WRF simulation names use base name with index."""
        # Create data with non-WRF simulation names
        simulations = ["LOCA_simulation_v1", "custom_model_run"]
        test_data = self._create_test_dataarray(simulations)

        # Execute function
        result = compute_profile(test_data, q=0.5)

        # Verify outcome: uses base name plus index for non-WRF
        sim_names = result.columns.unique()
        assert len(sim_names) == 2, "Should have two simulations"

        # Check that base names are extracted and indices appended
        sim_labels = sorted(sim_names)
        assert (
            "LOCA" in sim_labels[0] or "custom" in sim_labels[0]
        ), "Should extract base name"
        assert any(
            "-1" in label or "-2" in label for label in sim_labels
        ), "Should append index"

    def test_none_simulation_uses_fallback(self):
        """Test that None simulation value uses Sim_N fallback format."""
        # Create data without explicit simulation names - use multiple None values
        simulations = [None, None]
        test_data = self._create_test_dataarray(simulations)

        # Execute function
        result = compute_profile(test_data, q=0.5)

        # Verify outcome: uses Sim_N format for None
        sim_names = result.columns.unique()
        assert len(sim_names) == 2, "Should have two simulations"
        # Should use fallback format like Sim_1, Sim_2 or None-1, None-2
        for sim_label in sim_names:
            assert (
                "Sim_" in str(sim_label)
                or "None" in str(sim_label)
                or "-" in str(sim_label)
            ), f"Should use fallback format for None simulation, got {sim_label}"

    def test_multiple_wrf_simulations_parsed_correctly(self):
        """Test multiple WRF simulations are all parsed correctly."""
        # Create data with multiple WRF simulations
        simulations = [
            "WRF_CESM2_r11i1p1f1_historical+ssp245",
            "WRF_CNRM-ESM2-1_r1i1p1f2_historical+ssp370",
            "WRF_EC-Earth3_r4i1p1f1_historical+ssp585",
        ]
        test_data = self._create_test_dataarray(simulations)

        # Execute function
        result = compute_profile(test_data, q=0.5)

        # Verify outcome: all simulations parsed correctly
        sim_names = result.columns.unique()
        assert len(sim_names) == 3, "Should have three unique simulations"

        sim_labels = sorted(sim_names)
        assert any(
            "CESM2" in label and "ssp245" in label for label in sim_labels
        ), "Should parse CESM2 with ssp245"
        assert any(
            "CNRM-ESM2-1" in label and "ssp370" in label for label in sim_labels
        ), "Should parse CNRM-ESM2-1 with ssp370"
        assert any(
            "EC-Earth3" in label and "ssp585" in label for label in sim_labels
        ), "Should parse EC-Earth3 with ssp585"


class TestProfileUtilityFunctions:
    """Test class for profile utility functions.

    Tests the utility functions that handle metadata extraction, units,
    and profile DataFrame formatting operations.

    Attributes
    ----------
    sample_profile : pd.DataFrame
        Sample profile DataFrame for testing.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample profile DataFrame with metadata
        self.sample_profile = pd.DataFrame(
            np.random.rand(365, 24), index=range(1, 366), columns=range(1, 25)
        )
        # Add metadata attributes
        self.sample_profile.attrs = {
            "units": "degF",
            "variable_id": "tasmax",
            "quantile": 0.5,
            "method": "8760 analysis",
            "description": "Test climate profile",
        }

    def test_get_profile_units_returns_correct_units(self):
        """Test that get_profile_units extracts units from DataFrame metadata."""
        # Execute function
        units = get_profile_units(self.sample_profile)

        # Verify outcome: returns the correct units
        assert units == "degF", "Should return the units from DataFrame metadata"

    def test_get_profile_units_returns_unknown_when_missing(self):
        """Test that get_profile_units returns 'Unknown' when units are not present."""
        # Create DataFrame without units
        profile_no_units = pd.DataFrame(np.random.rand(10, 5))

        # Execute function
        units = get_profile_units(profile_no_units)

        # Verify outcome: returns 'Unknown' for missing units
        assert (
            units == "Unknown"
        ), "Should return 'Unknown' when units metadata is missing"

    def test_get_profile_metadata_returns_all_attributes(self):
        """Test that get_profile_metadata returns complete metadata dictionary."""
        # Execute function
        metadata = get_profile_metadata(self.sample_profile)

        # Verify outcome: returns dictionary with all metadata
        assert isinstance(metadata, dict), "Should return a dictionary"
        assert metadata["units"] == "degF", "Should include units"
        assert metadata["variable_id"] == "tasmax", "Should include variable_id"
        assert metadata["quantile"] == 0.5, "Should include quantile"
        assert metadata["method"] == "8760 analysis", "Should include method"
        assert len(metadata) == 5, "Should return all metadata attributes"

    def test_set_profile_metadata_updates_dataframe_attrs(self):
        """Test that set_profile_metadata properly updates DataFrame attributes."""
        # Setup new metadata to add
        new_metadata = {
            "author": "Test User",
            "created_date": "2023-01-01",
            "notes": "Test profile data",
        }

        # Execute function
        set_profile_metadata(self.sample_profile, new_metadata)

        # Verify outcome: DataFrame attrs are updated
        assert "author" in self.sample_profile.attrs, "Should add new author attribute"
        assert (
            "created_date" in self.sample_profile.attrs
        ), "Should add new created_date attribute"
        assert "notes" in self.sample_profile.attrs, "Should add new notes attribute"
        assert (
            self.sample_profile.attrs["author"] == "Test User"
        ), "Should set correct author value"

        # Original attributes should still exist
        assert (
            self.sample_profile.attrs["units"] == "degF"
        ), "Should preserve original units"

    def test_set_profile_metadata_raises_error_for_non_dict_input(self):
        """Test that set_profile_metadata raises ValueError for non-dictionary input."""
        # Execute and verify outcome: should raise ValueError for non-dict input
        with pytest.raises(
            ValueError, match="Metadata must be provided as a dictionary"
        ):
            set_profile_metadata(self.sample_profile, "not_a_dict")


class TestComputeDifferenceProfile:
    """Test class for _compute_difference_profile function.

    Tests the function that computes differences between future and historic
    climate profiles, handling various DataFrame column structures including
    simple indexes and MultiIndex columns.

    Attributes
    ----------
    simple_future_profile : pd.DataFrame
        Simple future profile with single-level columns.
    simple_historic_profile : pd.DataFrame
        Simple historic profile with single-level columns.
    multi_future_profile : pd.DataFrame
        Future profile with MultiIndex columns.
    multi_historic_profile : pd.DataFrame
        Historic profile with MultiIndex columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple profiles (single-level columns)
        self.simple_future_profile = pd.DataFrame(
            np.random.rand(8760, 1) + 20.0,  # Future is warmer
            index=range(1, 8761),
            columns=range(1,2),
        )
        self.simple_historic_profile = pd.DataFrame(
            np.random.rand(8760, 1) + 15.0,  # Future is cooler
            index=range(1, 8761),
            columns=range(1, 2),
        )

        # Create MultiIndex profiles
        wl_levels = [1.5, 2.0]
        simulations = ["sim1", "sim2"]

        # Future with MultiIndex (Hour, Warming_Level, Simulation)
        multi_cols_future = pd.MultiIndex.from_product(
            [wl_levels, simulations],
            names=["Warming_Level", "Simulation"],
        )
        self.multi_future_profile = pd.DataFrame(
            np.random.rand(8760, len(multi_cols_future)) + 20.0,
            index=range(1, 8761),
            columns=multi_cols_future,
        )

        # Historic with MultiIndex (Hour, Simulation)
        multi_cols_historic = pd.MultiIndex.from_product(
            [wl_levels, simulations],
            names=["Warming_Level", "Simulation"],
        )
        self.multi_historic_profile = pd.DataFrame(
            np.random.rand(8760, len(multi_cols_historic)) + 15.0,
            index=range(1, 8761),
            columns=multi_cols_historic,
        )

    def test_compute_difference_profile_with_simple_columns(self):
        """Test _compute_difference_profile with simple single-level columns."""
        # Execute function
        result = _compute_difference_profile(
            self.simple_future_profile, self.simple_historic_profile
        )

        # Verify outcome: returns DataFrame with difference values
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == self.simple_future_profile.shape
        ), "Shape should match future profile"

        # Check that differences are computed correctly (future - historic)
        # Since future values are ~20 and historic are ~15, differences should be ~5
        assert (
            result.mean().mean() > 0
        ), "Future should be warmer than historic on average"
        assert (
            result.mean().mean() < 10
        ), "Difference should be reasonable (< 10 degrees)"

    def test_compute_difference_profile_with_multiindex_columns(self):
        """Test _compute_difference_profile with MultiIndex columns."""
        # Create matched MultiIndex profiles for testing
        wl_levels = [1.5, 2.0]
        simulations = ["sim1", "sim2"]

        # Both profiles have (Hour, Simulation) structure
        multi_cols = pd.MultiIndex.from_product(
            [wl_levels,simulations], names=["Warming_Level","Simulation"]
        )

        future_multi = pd.DataFrame(
            np.random.rand(8760, len(multi_cols)) + 20.0,
            index=range(1, 8761),
            columns=multi_cols,
        )
        historic_multi = pd.DataFrame(
            np.random.rand(8760, len(multi_cols)) + 15.0,
            index=range(1, 8761),
            columns=multi_cols,
        )

        # Execute function
        result = _compute_difference_profile(future_multi, historic_multi)

        # Verify outcome: returns DataFrame with MultiIndex structure preserved
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should preserve MultiIndex structure"
        assert result.columns.names == [
            "Warming_Level",
            "Simulation",
        ], "Should preserve column level names"
        assert result.shape == future_multi.shape, "Shape should match future profile"

    def test_compute_difference_profile_with_mixed_index_types(self):
        """Test _compute_difference_profile when future has MultiIndex and historic has simple columns."""
        # Execute function with mixed index types
        result = _compute_difference_profile(
            self.multi_future_profile, self.simple_historic_profile
        )

        # Verify outcome: returns DataFrame preserving future structure
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should preserve future MultiIndex structure"
        assert (
            result.shape == self.multi_future_profile.shape
        ), "Shape should match future profile"
        assert (
            result.columns.names == self.multi_future_profile.columns.names
        ), "Should preserve column level names"

    def test_compute_difference_profile_handles_empty_dataframes(self):
        """Test _compute_difference_profile with empty DataFrames."""
        # Create empty DataFrames
        empty_future = pd.DataFrame()
        empty_historic = pd.DataFrame()

        # Execute function
        result = _compute_difference_profile(empty_future, empty_historic)

        # Verify outcome: returns empty DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.empty, "Should return empty DataFrame for empty inputs"

    def test_compute_difference_profile_preserves_metadata(self):
        """Test that _compute_difference_profile preserves metadata from future profile."""
        # Add metadata to future profile
        self.simple_future_profile.attrs = {
            "units": "degC",
            "variable_id": "tasmax",
            "description": "Future temperature profile",
        }

        # Execute function
        result = _compute_difference_profile(
            self.simple_future_profile, self.simple_historic_profile
        )

        # Verify outcome: metadata is preserved
        assert hasattr(result, "attrs"), "Result should have attrs attribute"
        assert result.attrs["units"] == "degC", "Should preserve units metadata"
        assert (
            result.attrs["variable_id"] == "tasmax"
        ), "Should preserve variable_id metadata"


class TestComputePairedDifference:
    """Test class for _compute_paired_difference function.

    Tests the function that computes paired differences when both profiles
    have simulation dimensions, matching simulations between future and historic
    profiles for accurate comparison.

    Attributes
    ----------
    future_profile : pd.DataFrame
        Future profile with simulation columns.
    historic_profile : pd.DataFrame
        Historic profile with simulation columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        wl_levels = [1.5,2.0]
        simulations = ["sim1", "sim2", "sim3"]

        # Create future profile with (Hour, Simulation)
        future_cols = pd.MultiIndex.from_product(
            [wl_levels, simulations], names=["Warming_Level", "Simulation"]
        )
        self.future_profile = pd.DataFrame(
            np.random.rand(8760, len(future_cols)) + 20.0,
            index=range(1, 8761),
            columns=future_cols,
        )

        # Create historic profile with (Hour, Simulation) - same simulations
        self.historic_profile = pd.DataFrame(
            np.random.rand(8760, len(future_cols)) + 15.0,
            index=range(1, 8761),
            columns=future_cols,
        )

    def test_compute_paired_difference_matches_common_simulations(self):
        """Test _compute_paired_difference matches common simulations correctly."""
        # Execute function with matching simulation sets
        result = _compute_paired_difference(
            self.future_profile, self.historic_profile
        )

        # Verify outcome: returns DataFrame with paired differences
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should maintain MultiIndex structure"
        assert result.columns.names == [
            "Warming_Level",
            "Simulation",
        ], "Should preserve column level names"
        assert (
            result.shape == self.future_profile.shape
        ), "Shape should match future profile"

        # Check that differences are computed (future - historic)
        # Values should be positive since future (20+) > historic (15+)
        assert (
            result.mean().mean() > 0
        ), "Differences should be positive (future > historic)"

    def test_compute_paired_difference_with_no_common_simulations(self):
        """Test _compute_paired_difference when no simulations match."""
        # Create historic profile with different simulations
        wl_levels = [1.5,2.0]
        different_sims = ["sim4", "sim5", "sim6"]  # No overlap with future sims

        historic_cols = pd.MultiIndex.from_product(
            [wl_levels, different_sims], names=["Warming_Level", "Simulation"]
        )
        historic_different = pd.DataFrame(
            np.random.rand(8760, len(historic_cols)) + 15.0,
            index=range(1, 8761),
            columns=historic_cols,
        )

        # Execute function with no matching simulations
        result = _compute_paired_difference(
            self.future_profile, historic_different
        )

        # Verify outcome: returns DataFrame when no matches (computes average difference)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == self.future_profile.shape
        ), "Shape should match future profile"
        # When no common simulations, function computes difference using averages
        # Results should still be meaningful (positive since future > historic)
        assert (
            result.mean().mean() > 0
        ), "Should compute positive differences even without matched sims"

    def test_compute_paired_difference_with_duplicate_columns(self):
        """Test _compute_paired_difference handles duplicate columns correctly."""
        # Create future profile with duplicate columns
        wl_levels = [1.5, 1.5, 2.0, 2.0]  # Duplicate hours
        sims = ["sim1", "sim1"]  # Duplicate simulations

        dup_cols = pd.MultiIndex.from_product(
            [wl_levels, sims], names=["Warming_Level", "Simulation"]
        )

        future_dup = pd.DataFrame(
            np.random.rand(8760, len(dup_cols)) + 20.0,
            index=range(1, 8761),
            columns=dup_cols,
        )
        historic_dup = pd.DataFrame(
            np.random.rand(8760, len(dup_cols)) + 15.0,
            index=range(1, 8761),
            columns=dup_cols,
        )

        # Execute function
        result = _compute_paired_difference(
            future_dup, historic_dup
        )

        # Verify outcome: handles duplicates gracefully
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == future_dup.shape[0], "Should preserve row count"

    def test_compute_paired_difference_missing_historic_column_uses_mean(
        self,
    ):
        """Test fallback to historic hour mean when matching column not found.

        This tests the else branch in _compute_paired_difference where
        historic_col is None or not in historic_profile.columns, triggering the
        fallback to _get_historic_hour_mean.
        """
        # Create future profile with some simulations
        wl_levels = [1.5,2.0]
        future_sims = ["sim1", "sim2", "sim3"]

        future_cols = pd.MultiIndex.from_product(
            [wl_levels, future_sims], names=["Warming_Level", "Simulation"]
        )
        future_profile = pd.DataFrame(
            np.random.rand(8760, len(future_cols)) + 20.0,
            index=range(1, 8761),
            columns=future_cols,
        )

        # Create historic profile with PARTIALLY matching simulations
        # This will cause some columns to not match
        historic_sims = ["sim1", "sim4"]  # Only sim1 matches, sim2 and sim3 don't

        historic_cols = pd.MultiIndex.from_product(
            [wl_levels, historic_sims], names=["Warming_Level", "Simulation"]
        )
        historic_profile = pd.DataFrame(
            np.random.rand(8760, len(historic_cols)) + 15.0,
            index=range(1, 8761),
            columns=historic_cols,
        )

        # Execute function
        result = _compute_paired_difference(
            future_profile, historic_profile
        )

        # Verify outcome: returns DataFrame with differences
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == future_profile.shape, "Shape should match future profile"

        # For sim1: should use direct pairing (matching historic column exists)
        # For sim2 and sim3: should use mean of historic for each hour (fallback path)

        # Check that sim1 columns exist and have valid values
        sim1_cols = [col for col in result.columns if col[1] == "sim1"]
        assert len(sim1_cols) == 1, "Should have 1 columns for sim1"
        assert all(
            not pd.isna(result[col]).any() for col in sim1_cols
        ), "Sim1 columns should have no NaN values"

        # Check that sim2 and sim3 use fallback (mean) - they should also have valid values
        sim2_cols = [col for col in result.columns if col[1] == "sim2"]
        sim3_cols = [col for col in result.columns if col[1] == "sim3"]
        assert len(sim2_cols) == 1, "Should have 1 column for sim2"
        assert len(sim3_cols) == 1, "Should have 1 column for sim3"

        # All values should be numeric (using mean fallback)
        # Note: The fallback uses _get_historic_hour_mean which returns mean across simulations
        # This tests the else branch where historic_col is not found

        # Check that sim2 values exist (may be 0 if mean returns scalar 0)
        sim2_values = result[sim2_cols].values
        sim3_values = result[sim3_cols].values

        # Values should be numeric (not NaN in most cases, but could be if mean computation fails)
        # The key test is that the else branch was executed, which we verify by checking
        # that sim2 and sim3 exist and have numerical values
        assert sim2_values.shape == (
            8760,
            1,
        ), "Sim2 should have correct shape (8760 hours)"
        assert sim3_values.shape == (
            8760,
            1,
        ), "Sim3 should have correct shape (8760 hours)"

        # Most importantly: verify the function completed successfully and returned
        # a DataFrame with the expected structure, which means the else branch
        # (lines 466-474) was executed for sim2 and sim3
        assert (
            result.shape == future_profile.shape
        ), "Final result shape should match input"

class TestConstructProfileDataframe:
    """Test class for _construct_profile_dataframe function.

    Tests the function that constructs climate profile DataFrames with appropriate
    column structures based on warming level and simulation dimensions.

    Attributes
    ----------
    profile_data : dict
        Sample profile data dictionary for testing.
    warming_levels : np.ndarray
        Array of warming level values.
    simulations : list
        List of simulation identifiers.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.warming_levels = np.array([1.5, 2.0])
        self.simulations = ["sim1", "sim2"]

        # Create sample profile data dictionary
        self.profile_data = {}
        for wl in self.warming_levels:
            for i, sim in enumerate(self.simulations):
                wl_key = f"WL_{wl}"
                sim_key = f"Sim{i+1}"  # Simple simulation labels
                # Create 8760x2 profile matrix
                self.profile_data[(wl_key, sim_key)] = np.random.rand(8760, 2) + 20.0

        # Simple function to get simulation labels
        def sim_label_func(sim, sim_idx):
            return f"Sim{sim_idx + 1}"

        self.sim_label_func = sim_label_func

    def test_construct_profile_dataframe_single_wl_single_sim(self):
        """Test _construct_profile_dataframe with single warming level and single simulation."""
        # Use only first warming level and simulation
        single_wl = np.array([1.5])
        single_sim = ["sim1"]

        # Create appropriate profile data
        profile_data = {("Sim1"): np.random.rand(8760, 1) + 20.0}

        # Execute function
        result = _construct_profile_dataframe(
            profile_data=profile_data,
            warming_levels=single_wl,
            simulations=single_sim,
            sim_label_func=self.sim_label_func,
            hours_per_year=8760,
        )
        # Verify outcome: MultiIndex structure with (Hour, Simulation)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (
            8760,
            1,
        ), "Should have 8760 rows and 1 column (8760*1 sims)"
        assert isinstance(
            result.columns, pd.Index
        ), "Should have a simple Index column structure"

    def test_construct_profile_dataframe_single_wl_multi_sim(self):
        """Test _construct_profile_dataframe with single warming level and multiple simulations."""
        # Use single warming level but multiple simulations
        single_wl = np.array([1.5])
        multi_sim = ["sim1", "sim2"]

        # Create appropriate profile data
        profile_data = {
            ("Sim1"): np.random.rand(8760, 1) + 20.0,
            ("Sim2"): np.random.rand(8760, 1) + 20.0,
        }

        # Execute function
        result = _construct_profile_dataframe(
            profile_data=profile_data,
            warming_levels=single_wl,
            simulations=multi_sim,
            sim_label_func=self.sim_label_func,
            hours_per_year=8760,
        )

        # Verify outcome: MultiIndex structure with (Hour, Simulation)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (
            8760,
            2,
        ), "Should have 8760 rows and 2 columns"
        assert isinstance(
            result.columns, pd.Index
        ), "Should have a simple Index column structure"

    def test_construct_profile_dataframe_multi_wl_single_sim(self):
        """Test _construct_profile_dataframe with multiple warming levels and single simulation."""
        # Use multiple warming levels but single simulation
        multi_wl = np.array([1.5, 2.0])
        single_sim = ["sim1"]

        # Create appropriate profile data
        profile_data = {
            ("WL_1.5", "Sim1"): np.random.rand(8760, 1) + 20.0,
            ("WL_2.0", "Sim1"): np.random.rand(8760, 1) + 22.0,
        }

        # Execute function
        result = _construct_profile_dataframe(
            profile_data=profile_data,
            warming_levels=multi_wl,
            simulations=single_sim,
            sim_label_func=self.sim_label_func,
            hours_per_year=8760,
        )

        # Verify outcome: MultiIndex structure with (Hour, Warming_Level)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (
            8760,
            2,
        ), "Should have 8760 rows and 2 columns"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should have MultiIndex column structure"
        assert result.columns.names == [
            "Warming_Level",
            "Simulation"
        ], "Should have Hour and Warming_Level levels"

    def test_construct_profile_dataframe_multi_wl_multi_sim(self):
        """Test _construct_profile_dataframe with multiple warming levels and multiple simulations."""
        # Use multiple warming levels and multiple simulations
        multi_wl = np.array([1.5, 2.0])
        multi_sim = ["sim1", "sim2"]

        # Create appropriate profile data for all combinations
        profile_data = {
            ("WL_1.5", "Sim1"): np.random.rand(8760, 1) + 20.0,
            ("WL_1.5", "Sim2"): np.random.rand(8760, 1) + 20.0,
            ("WL_2.0", "Sim1"): np.random.rand(8760, 1) + 22.0,
            ("WL_2.0", "Sim2"): np.random.rand(8760, 1) + 22.0,
        }

        # Execute function
        result = _construct_profile_dataframe(
            profile_data=profile_data,
            warming_levels=multi_wl,
            simulations=multi_sim,
            sim_label_func=self.sim_label_func,
            hours_per_year=8760,
        )

        # Verify outcome: MultiIndex structure with (Hour, Warming_Level, Simulation)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (
            8760,
            4,
        ), "Should have 8760 rows and 4 columns"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should have MultiIndex column structure"
        # The function creates a 3-level MultiIndex for this scenario
        assert len(result.columns.levels) >= 2, "Should have at least 2 column levels"


class TestStackProfileData:
    """Test class for _stack_profile_data function.

    Tests the function that stacks profile data arrays into the appropriate
    format for DataFrame creation, handling different ordering and dimensions.

    Attributes
    ----------
    profile_data : dict
        Sample profile data dictionary for testing.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample profile data - 365 days x 24 hours
        self.profile_data = {
            ("WL_1.5", "Sim1"): np.random.rand(8760, 1) + 20.0,
            ("WL_1.5", "Sim2"): np.random.rand(8760, 1) + 20.0,
            ("WL_2.0", "Sim1"): np.random.rand(8760, 1) + 22.0,
            ("WL_2.0", "Sim2"): np.random.rand(8760, 1) + 22.0,
        }

    def test_stack_profile_data_hour_first_two_level(self):
        """Test _stack_profile_data with hour_first=True and two levels."""
        # Execute function with hour-first ordering
        result = _stack_profile_data(
            profile_data=self.profile_data,
            wl_names=["WL_1.5", "WL_2.0"],
            sim_names=["Sim1", "Sim2"],
        )

        # Verify outcome: returns 2D array with correct shape
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        assert result.shape == (
            8760,
            4,
        ), "Should have 8760 rows and 4 columns"
        assert not np.isnan(result).any(), "Should not contain NaN values"

        # Check that all values are positive (since input data is 20+ and 22+)
        assert np.all(result > 0), "All values should be positive"

    def test_stack_profile_data_three_level_structure(self):
        """Test _stack_profile_data with three_level=True."""
        # Execute function with three-level structure
        result = _stack_profile_data(
            profile_data=self.profile_data,
            wl_names=["WL_1.5", "WL_2.0"],
            sim_names=["Sim1", "Sim2"],
        )

        # Verify outcome: returns 2D array with proper dimensions
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        assert result.shape == (365, 96), "Should have correct total column count"
        assert result.dtype.kind == "f", "Should contain floating point data"

    def test_stack_profile_data_handles_single_simulation(self):
        """Test _stack_profile_data with single simulation."""
        # Create single simulation profile data
        single_sim_data = {
            ("WL_1.5", "Sim1"): np.random.rand(8760, 1) + 20.0,
            ("WL_2.0", "Sim1"): np.random.rand(8760, 1) + 22.0,
        }

        # Execute function
        result = _stack_profile_data(
            profile_data=single_sim_data,
            wl_names=["WL_1.5", "WL_2.0"],
            sim_names=["Sim1"],
        )

        # Verify outcome: correct dimensions for single simulation
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        assert result.shape == (8760, 2), "Should have 48 columns (24*2WL*1sim)"
        assert np.all(result >= 20), "Values should be in expected range"

    def test_stack_profile_data_preserves_data_values(self):
        """Test _stack_profile_data preserves original data values correctly."""
        # Create simple test data to verify preservation
        simple_data = {
            ("WL_1.5", "Sim1"): np.ones((8760, 1)) * 25.0,  # All values = 25
            ("WL_2.0", "Sim1"): np.ones((8760, 1)) * 30.0,  # All values = 30
        }

        # Execute function
        result = _stack_profile_data(
            profile_data=simple_data,
            wl_names=["WL_1.5", "WL_2.0"],
            sim_names=["Sim1"],
        )

        # Verify outcome: preserves data values correctly
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        # Check that we have both 25.0 and 30.0 values in the result
        unique_values = np.unique(result)
        assert 25.0 in unique_values, "Should preserve 25.0 values from WL_1.5"
        assert 30.0 in unique_values, "Should preserve 30.0 values from WL_2.0"


class TestCreateSimpleDataframe:
    """Test class for _create_simple_dataframe function.

    Tests the function that creates a simple DataFrame for single warming level
    and single simulation scenarios, handling profile data dictionary structure
    and proper DataFrame construction with hour columns.

    Attributes
    ----------
    profile_data : dict
        Sample profile data dictionary for testing.
    warming_level : float
        Sample warming level value.
    simulation : str
        Sample simulation identifier.
    sim_label_func : callable
        Function to get simulation labels.
    """

    def setup_method(self):
        """Set up test fixtures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.warming_level = 2.0
        self.simulation = "Sim1"

        # Create sample profile data dictionary
        wl_key = "WL_2.0"
        sim_key = "Sim1"

        # Create 365x24 profile matrix with realistic climate data
        self.profile_data = {
            (sim_key): np.random.rand(8760, 1) + 20.0  # Temperature-like data
        }

        # Simple function to get simulation labels
        def sim_label_func(sim, sim_idx):
            return f"Sim{sim_idx + 1}"

        self.sim_label_func = sim_label_func
        self.hours_per_year = 8760

    def test_create_simple_dataframe_returns_dataframe(self):
        """Test _create_simple_dataframe returns pd.DataFrame."""
        # Execute function
        result = _create_simple_dataframe(
            profile_data=self.profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            hours_per_year=8760,
        )

        # Verify outcome: returns a pandas DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] > 0, "DataFrame should have rows"
        assert result.shape[1] > 0, "DataFrame should have columns"

    def test_create_simple_dataframe_with_proper_structure(self):
        """Test that the DataFrame has correct MultiIndex column structure."""
        # Execute function
        result = _create_simple_dataframe(
            profile_data=self.profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            hours_per_year=8760,
        )

        # Verify outcome: correct MultiIndex column structure
        assert isinstance(result.columns, pd.Index), "Columns should be a simple Index"

        # Verify expected dimensions: 365 rows, 24 columns
        expected_rows = 8760
        expected_cols = 1
        assert result.shape == (
            expected_rows,
            expected_cols,
        ), f"Should have {expected_rows} rows and {expected_cols} columns"

        # Verify index structure (hour number)
        expected_index = np.arange(1, self.hours_per_year + 1)
        np.testing.assert_array_equal(
            result.index.values,
            expected_index,
            err_msg="Index should be day numbers from 1 to hours_per_year",
        )

    def test_create_simple_dataframe_with_different_scenarios(self):
        """Test _create_simple_dataframe with different warming level and simulation scenarios."""
        # Test different warming level
        different_wl = 1.5
        different_sim = "Sim1"
        different_wl_data = {"Sim1": np.random.rand(8760, 1) + 15.0}

        # Execute function with different warming level
        result_wl = _create_simple_dataframe(
            profile_data=different_wl_data,
            warming_level=different_wl,
            simulation=different_sim,
            sim_label_func=self.sim_label_func,
            hours_per_year=8760,
        )

        # Verify outcome: maintains same structure with different data
        assert isinstance(
            result_wl, pd.DataFrame
        ), "Should return DataFrame for different WL"
        assert result_wl.shape == (8760, 1), "Should maintain same shape"

        # Test different simulation identifier
        different_sim = "sim2"
        # Note: sim_label_func always uses index 0, so key will be "Sim1" regardless of simulation value
        different_sim_data = {"Sim1": np.random.rand(8760, 1) + 25.0}

        # Execute function with different simulation
        result_sim = _create_simple_dataframe(
            profile_data=different_sim_data,
            warming_level=self.warming_level,
            simulation=different_sim,
            sim_label_func=self.sim_label_func,
            hours_per_year=8760,
        )

        # Verify outcome: handles different simulation correctly
        assert isinstance(
            result_sim, pd.DataFrame
        ), "Should return DataFrame for different sim"
        assert result_sim.shape == (8760,1), "Should maintain same shape"

        # Verify all results have different data but same structure
        assert not result_wl.equals(
            result_sim
        ), "Different scenarios should produce different data"
        assert list(result_wl.columns) == list(
            result_sim.columns
        ), "Columns should be identical"

    def test_create_simple_dataframe_preserves_data(self):
        """Test _create_simple_dataframe preserves data values correctly."""
        # Create profile data with known values for verification
        test_data = np.ones((8760, 1)) * 42.5  # All values set to 42.5
        test_profile_data = {"Sim1": test_data}

        # Execute function
        result = _create_simple_dataframe(
            profile_data=test_profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            hours_per_year=8760,
        )

        # Verify outcome: data values are preserved correctly
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

        # Check that all values in the DataFrame match the original data
        assert np.all(result.values == 42.5), "All values should be 42.5"

        # Verify shape matches original profile matrix
        assert (
            result.shape == test_data.shape
        ), "Shape should match original profile matrix"

        # Test with different profile matrix size (leap year)
        leap_year_data = np.ones((8768, 1)) * 33.7  # Leap year with different values
        leap_year_profile_data = {"Sim1": leap_year_data}

        # Execute function with leap year data
        result_leap = _create_simple_dataframe(
            profile_data=leap_year_profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            hours_per_year=8768,  # Leap year
        )

        # Verify outcome: handles different matrix sizes correctly
        assert result_leap.shape == (
            8768,
            1,
        ), "Should handle leap year shape (8768 hours)"
        assert np.all(result_leap.values == 33.7), "All leap year values should be 33.7"

        # Verify proper index for leap year
        expected_leap_index = np.arange(1, 367, 1)  # 1 to 366
        np.testing.assert_array_equal(result_leap.index.values, expected_leap_index)

    def test_create_simple_dataframe_with_different_year_lengths(self):
        """Test _create_simple_dataframe with different days_in_year parameter values."""
        # Test with various year lengths
        year_length_scenarios = [
            {"hours": 8760, "description": "regular year"},
            {"hours": 8768, "description": "leap year"},
            {"hours": 8800, "description": "simplified calendar year"},
            {"hours": 8800, "description": "partial year"},
        ]

        for scenario in year_length_scenarios:
            hours = scenario["hours"]
            description = scenario["description"]

            # Create profile data matching the year length
            profile_matrix = np.random.rand(hours, 1) + 18.0
            test_data = {("WL_2.0", "Sim1"): profile_matrix}

            # Execute function
            result = _create_simple_dataframe(
                profile_data=test_data,
                warming_level=self.warming_level,
                simulation=self.simulation,
                sim_label_func=self.sim_label_func,
                hours_per_year=hours,
            )

            # Verify outcome: correct dimensions for each scenario
            assert isinstance(
                result, pd.DataFrame
            ), f"Should return DataFrame for {description}"
            assert result.shape == (
                hours,
                1,
            ), f"Should have {hours} rows for {description}"
            assert result.shape[1] == 1, f"Should have 1 column for {description}"

            # Verify proper index generation
            expected_index = np.arange(1, hours + 1, 1)
            np.testing.assert_array_equal(
                result.index.values,
                expected_index,
                err_msg=f"Index should be 1 to {hours} for {description}",
            )

            # Verify columns remain consistent regardless of year length

            np.testing.assert_array_equal(
                result.columns.values,
                "Sim1",
                err_msg=f"Columns should always be the simulation names for {description}",
            )

            # Verify data values are preserved
            np.testing.assert_array_equal(
                result.values,
                profile_matrix,
                err_msg=f"Data values should be preserved for {description}",
            )


class TestCreateSingleWlMultiSimDataframe:
    """Test class for _create_single_wl_multi_sim_dataframe function.

    Tests the function that creates DataFrames for single warming level with
    multiple simulations, validating MultiIndex column structure and data handling.

    Attributes
    ----------
    sample_profile_data : dict
        Sample profile data dictionary with (warming_level, simulation) keys.
    mock_sim_label_func : MagicMock
        Mock function for generating simulation labels.
    warming_level : float
        Sample warming level for testing.
    simulations : list
        Sample list of simulations.
    hours : np.ndarray
        Array of hour values for columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock simulation label function
        self.mock_sim_label_func = MagicMock()
        self.mock_sim_label_func.side_effect = lambda sim, idx: f"sim_{sim}_{idx}"

        # Test parameters
        self.warming_level = 2.0
        self.simulations = ["model_A", "model_B", "model_C"]
        self.hours_per_year = 8760

        # Create sample profile data dictionary
        # The function expects data for each (WL_X, sim_label) combination
        self.sample_profile_data = {}
        for i, sim in enumerate(self.simulations):
            sim_key = f"sim_{sim}_{i}"
            # Create random data for each simulation (365 days x 24 hours)
            profile_matrix = np.random.rand(8760, 1) + 20.0
            self.sample_profile_data[sim_key] = profile_matrix

    def test_create_single_wl_multi_sim_dataframe_returns_dataframe(self):
        """Test that _create_single_wl_multi_sim_dataframe returns a pandas DataFrame."""
        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_level=self.warming_level,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: returns a pandas DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] > 0, "DataFrame should have rows"
        assert result.shape[1] > 0, "DataFrame should have columns"

    def test_create_single_wl_multi_sim_dataframe_multiindex_structure(self):
        """Test that the DataFrame has correct MultiIndex column structure."""
        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_level=self.warming_level,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: correct MultiIndex column structure
        assert isinstance(result.columns, pd.iIndex), "Columns should be a simple Index"


        # Verify expected dimensions: 365 rows, (24 hours × 3 simulations) columns
        expected_rows = 8760
        expected_cols = len(self.simulations)  
        assert result.shape == (
            expected_rows,
            expected_cols,
        ), f"Should have {expected_rows} rows and {expected_cols} columns"

        # Verify index structure (day numbers)
        expected_index = np.arange(1, self.hours_per_year + 1)
        np.testing.assert_array_equal(
            result.index.values,
            expected_index,
            err_msg="Index should be day numbers from 1 to hours_per_year",
        )

    def test_create_single_wl_multi_sim_dataframe_handles_multiple_simulations(self):
        """Test function handles multiple simulations correctly."""
        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_level=self.warming_level,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: a column for each simulation
        unique_simulations = result.columns.unique()

        # Should have one column for each (hour, simulation) combination
        expected_sim_names = ["sim_model_A_0", "sim_model_B_1", "sim_model_C_2"]
        assert len(unique_simulations) == len(
            self.simulations
        ), f"Should have {len(self.simulations)} simulations"

        # Verify simulation names match expected pattern from mock function
        for expected_sim in expected_sim_names:
            assert (
                expected_sim in unique_simulations
            ), f"Should contain simulation {expected_sim}"

    def test_create_single_wl_multi_sim_dataframe_duplicate_simulation_names(self):
        """Test function handles duplicate simulation names with uniqueness suffixes."""
        # Create mock sim_label_func that returns duplicate names
        mock_dup_sim_func = MagicMock()
        mock_dup_sim_func.side_effect = (
            lambda sim, idx: "duplicate_name"
        )  # All return same name

        # Create profile data with keys matching what the function will generate
        # The function creates unique names: first is "duplicate_name",
        # second is "duplicate_name_v1", third is "duplicate_name_v2"
        duplicate_profile_data = {}
        simulations_with_dups = ["model_A", "model_B", "model_C"]

        # Add data for original and uniquified names
        for i, unique_suffix in enumerate(
            ["duplicate_name", "duplicate_name_v1", "duplicate_name_v2"]
        ):
            profile_matrix = (
                np.random.rand(8760, 1) + 20.0 + i
            )  # Slightly different data
            duplicate_profile_data[unique_suffix] = profile_matrix

        # Execute function and verify warning is printed
        with patch("builtins.print") as mock_print:
            result = _create_single_wl_multi_sim_dataframe(
                profile_data=duplicate_profile_data,
                warming_level=self.warming_level,
                simulations=simulations_with_dups,
                sim_label_func=mock_dup_sim_func,
                hours_per_year=self.hours_per_year,
            )

        # Verify outcome: warning message was printed about duplicates
        printed_calls = [str(call) for call in mock_print.call_args_list]
        printed_output = " ".join(printed_calls)
        assert (
            "duplicate simulation names" in printed_output.lower()
        ), "Should warn about duplicate simulation names"
        assert (
            "uniqueness suffixes" in printed_output.lower()
        ), "Should mention adding uniqueness suffixes"

        # Verify the sim_label_func was called for each simulation
        assert mock_dup_sim_func.call_count == len(
            simulations_with_dups
        ), "Should call sim_label_func for each simulation"

        # Verify the result has the uniquified names in columns
        assert isinstance(result, pd.DataFrame), "Should return a DataFrame"
        unique_sims = result.columns.unique()

        # Should have 3 unique simulation names after de-duplication
        assert (
            len(unique_sims) == 3
        ), "Should have 3 unique simulations after de-duplication"
        assert "duplicate_name" in unique_sims, "Should contain original duplicate_name"
        assert "duplicate_name_v1" in unique_sims, "Should contain duplicate_name_v1"
        assert "duplicate_name_v2" in unique_sims, "Should contain duplicate_name_v2"

    def test_create_single_wl_multi_sim_dataframe_preserves_data_integrity(self):
        """Test that profile data values are correctly preserved in MultiIndex structure."""
        # Create specific test data with known values for verification
        test_simulations = ["test_sim_A", "test_sim_B"]

        # Create mock sim_label_func for predictable names
        test_sim_func = MagicMock()
        test_sim_func.side_effect = lambda sim, idx: f"test_{sim}_{idx}"
        test_hours = 8760

        # Create test profile data with known values
        test_profile_data = {}
        expected_values = {}

        for i, sim in enumerate(test_simulations):
            sim_key = f"test_{sim}_{i}"
            wl_key = f"WL_{self.warming_level}"
            # Create known test data: hour i has value (i+1)*10 + hr
            profile_matrix = np.zeros((test_hours, 1))
            for hr in range(test_hours):
                profile_matrix[hr, i] = (hr + 1) * 10 + i

            test_profile_data[sim_key] = profile_matrix
            expected_values[sim_key] = profile_matrix

        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=test_profile_data,
            warming_level=self.warming_level,
            simulations=test_simulations,
            sim_label_func=test_sim_func,
            hours_per_year=test_hours,
        )

        # Verify outcome: data integrity is preserved
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.shape == (
            test_hours,
            len(test_simulations),
        ), "Should have correct dimensions"

        # Verify specific data values are preserved for each (hour, simulation) combination
        for hour in test_hours:
            for i, sim in enumerate(test_simulations):
                sim_key = f"test_{sim}_{i}"
                expected_matrix = expected_values[sim_key]

                # Get column data for this simulation combination
                column_data = result[sim_key]
                expected_column = expected_matrix[:, list(test_hours).index(hour)]

                # Verify data values match
                np.testing.assert_array_equal(
                    column_data.values,
                    expected_column,
                    err_msg=f"Data mismatch for hour {hour}, simulation {sim_key}",
                )

        # Verify specific known values at expected positions
        # Day 1 (index 0), Hour 0, Sim A should be 10 (day 1 * 10 + hour 0)
        sim_a_key = "test_test_sim_A_0"
        assert (
            result.loc[1, sim_a_key] == 10.0
        ), "Day 1, Hour 0, Sim A should be 10"

        # Day 2 (index 1), Hour 1, Sim B should be 21 (day 2 * 10 + hour 1)
        sim_b_key = "test_test_sim_B_1"
        assert (
            result.loc[2, sim_b_key] == 21.0
        ), "Day 2, Hour 1, Sim B should be 21"


class TestCreateMultiWlSingleSimDataframe:
    """Test class for _create_multi_wl_single_sim_dataframe function.

    Tests the function that creates DataFrames for multiple warming levels with
    single simulation, validating MultiIndex column structure and data handling.

    Attributes
    ----------
    sample_profile_data : dict
        Sample profile data dictionary with (warming_level, simulation) keys.
    mock_sim_label_func : MagicMock
        Mock function for generating simulation labels.
    warming_levels : np.ndarray
        Array of warming levels for testing.
    simulation : str
        Sample simulation identifier.
    hours : np.ndarray
        Array of hour values for columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock simulation label function
        self.mock_sim_label_func = MagicMock()
        self.mock_sim_label_func.return_value = "test_simulation"

        # Test parameters
        self.warming_levels = np.array([1.5, 2.0, 3.0])
        self.simulation = "model_X"
        self.hours_per_year = 8760

        # Create sample profile data dictionary
        # The function expects data for each (WL_X, sim_label) combination
        self.sample_profile_data = {}
        sim_key = "test_simulation"

        for wl in self.warming_levels:
            wl_key = f"WL_{wl}"
            # Create random data for each warming level (365 days x 24 hours)
            profile_matrix = (
                np.random.rand(365, 24) + 20.0 + wl
            )  # Add WL to make different
            self.sample_profile_data[(wl_key, sim_key)] = profile_matrix

    def test_create_multi_wl_single_sim_dataframe_returns_dataframe(self):
        """Test that _create_multi_wl_single_sim_dataframe returns a pandas DataFrame."""
        # Execute function
        result = _create_multi_wl_single_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_levels=self.warming_levels,
            simulation=self.simulation,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: returns a pandas DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] > 0, "DataFrame should have rows"
        assert result.shape[1] > 0, "DataFrame should have columns"

    def test_create_multi_wl_single_sim_dataframe_multiindex_structure(self):
        """Test that the DataFrame has correct MultiIndex column structure."""
        # Execute function
        result = _create_multi_wl_single_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_levels=self.warming_levels,
            simulation=self.simulation,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: correct MultiIndex column structure
        assert isinstance(result.columns, pd.MultiIndex), "Columns should be MultiIndex"
        assert result.columns.names == [
            "Hour",
            "Warming_Level",
        ], "Column levels should be named Hour and Warming_Level"

        # Verify expected dimensions: 365 rows, (24 hours × 3 warming levels) columns
        expected_rows = 365
        expected_cols = 24 * len(
            self.warming_levels
        )  # 24 hours × 3 warming levels = 72
        assert result.shape == (
            expected_rows,
            expected_cols,
        ), f"Should have {expected_rows} rows and {expected_cols} columns"

        # Verify index structure (day numbers)
        expected_index = np.arange(1, self.days_in_year + 1)
        np.testing.assert_array_equal(
            result.index.values,
            expected_index,
            err_msg="Index should be day numbers from 1 to days_in_year",
        )

    def test_create_multi_wl_single_sim_dataframe_handles_multiple_warming_levels(self):
        """Test function handles multiple warming levels correctly."""
        # Execute function
        result = _create_multi_wl_single_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_levels=self.warming_levels,
            simulation=self.simulation,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: each warming level creates columns for all hours
        # Expected structure: (hour, wl) for each hour and each warming level
        unique_warming_levels = result.columns.get_level_values(
            "Warming_Level"
        ).unique()
        unique_hours = result.columns.get_level_values("Hour").unique()

        # Should have one column for each (hour, warming_level) combination
        expected_wl_names = ["WL_1.5", "WL_2.0", "WL_3.0"]
        assert len(unique_warming_levels) == len(
            self.warming_levels
        ), f"Should have {len(self.warming_levels)} warming levels"
        assert len(unique_hours) == len(
            self.hours
        ), f"Should have {len(self.hours)} hours"

        # Verify warming level names match expected pattern
        for expected_wl in expected_wl_names:
            assert (
                expected_wl in unique_warming_levels
            ), f"Should contain warming level {expected_wl}"

        # Verify each hour appears for each warming level (24 hours × 3 WLs = 72 columns)
        for hour in self.hours:
            for wl_name in expected_wl_names:
                assert (
                    hour,
                    wl_name,
                ) in result.columns, (
                    f"Should have column for hour {hour}, warming level {wl_name}"
                )

    def test_create_multi_wl_single_sim_dataframe_preserves_data_integrity(self):
        """Test that profile data values are correctly preserved in MultiIndex structure."""
        # Create specific test data with known values for verification
        test_warming_levels = np.array([1.0, 2.0])
        test_hours = np.array([0, 1, 2])  # Use smaller subset for easier verification
        test_days = 3  # Use smaller dataset for precise testing
        #! revisit

        # Create mock sim_label_func for predictable names
        test_sim_func = MagicMock()
        test_sim_func.return_value = "test_sim"

        # Create test profile data with known values
        test_profile_data = {}
        expected_values = {}

        for wl in test_warming_levels:
            wl_key = f"WL_{wl}"
            sim_key = "test_sim"
            # Create known test data: day i, hour j has value (day+1)*10 + hour + wl
            profile_matrix = np.zeros((test_days, len(test_hours)))
            for day in range(test_days):
                for hour_idx, hour in enumerate(test_hours):
                    profile_matrix[day, hour_idx] = (day + 1) * 10 + hour + wl

            test_profile_data[(wl_key, sim_key)] = profile_matrix
            expected_values[wl_key] = profile_matrix

        # Execute function
        result = _create_multi_wl_single_sim_dataframe(
            profile_data=test_profile_data,
            warming_levels=test_warming_levels,
            simulation="test_simulation",
            sim_label_func=test_sim_func,
            days_in_year=test_hours
        )

        # Verify outcome: data integrity is preserved
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.shape == (
            test_days,
            len(test_hours) * len(test_warming_levels),
        ), "Should have correct dimensions"

        # Verify specific data values are preserved for each (hour, warming_level) combination
        for hour in test_hours:
            for wl in test_warming_levels:
                wl_key = f"WL_{wl}"
                expected_matrix = expected_values[wl_key]

                # Get column data for this (hour, warming_level) combination
                column_data = result[(hour, wl_key)]
                expected_column = expected_matrix[:, list(test_hours).index(hour)]

                # Verify data values match
                np.testing.assert_array_equal(
                    column_data.values,
                    expected_column,
                    err_msg=f"Data mismatch for hour {hour}, warming level {wl_key}",
                )

        # Verify specific known values at expected positions
        # Day 1 (index 0), Hour 0, WL 1.0 should be 10 + 0 + 1.0 = 11.0
        assert (
            result.loc[1, (0, "WL_1.0")] == 11.0
        ), "Day 1, Hour 0, WL 1.0 should be 11.0"

        # Day 2 (index 1), Hour 1, WL 2.0 should be 20 + 1 + 2.0 = 23.0
        assert (
            result.loc[2, (1, "WL_2.0")] == 23.0
        ), "Day 2, Hour 1, WL 2.0 should be 23.0"

    def test_create_multi_wl_single_sim_dataframe_different_warming_level_configs(self):
        """Test function with different warming level configurations."""
        # Test scenarios with different warming level configurations
        test_scenarios = [
            {
                "name": "single_warming_level",
                "warming_levels": np.array([2.0]),
                "expected_cols": 24 * 1,  # 24 hours × 1 WL
                "expected_wl_names": ["WL_2.0"],
            },
            {
                "name": "two_warming_levels",
                "warming_levels": np.array([1.5, 3.0]),
                "expected_cols": 24 * 2,  # 24 hours × 2 WLs
                "expected_wl_names": ["WL_1.5", "WL_3.0"],
            },
            {
                "name": "many_warming_levels",
                "warming_levels": np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
                "expected_cols": 24 * 6,  # 24 hours × 6 WLs
                "expected_wl_names": [
                    "WL_1.0",
                    "WL_1.5",
                    "WL_2.0",
                    "WL_2.5",
                    "WL_3.0",
                    "WL_4.0",
                ],
            },
        ]

        for scenario in test_scenarios:
            # Create profile data for this scenario
            scenario_profile_data = {}
            sim_key = "test_simulation"

            for wl in scenario["warming_levels"]:
                wl_key = f"WL_{wl}"
                profile_matrix = np.random.rand(365, 24) + 20.0 + wl
                scenario_profile_data[(wl_key, sim_key)] = profile_matrix

            # Execute function
            result = _create_multi_wl_single_sim_dataframe(
                profile_data=scenario_profile_data,
                warming_levels=scenario["warming_levels"],
                simulation=self.simulation,
                sim_label_func=self.mock_sim_label_func,
                hours_per_year=self.hours_per_year,
            )

            # Verify outcome for this scenario
            assert isinstance(
                result, pd.DataFrame
            ), f"Should return DataFrame for {scenario['name']}"
            assert (
                result.shape[0] == 365
            ), f"Should have 365 rows for {scenario['name']}"
            assert (
                result.shape[1] == scenario["expected_cols"]
            ), f"Should have {scenario['expected_cols']} columns for {scenario['name']}"

            # Verify MultiIndex structure
            assert isinstance(
                result.columns, pd.MultiIndex
            ), f"Should have MultiIndex columns for {scenario['name']}"
            assert result.columns.names == [
                "Hour",
                "Warming_Level",
            ], f"Should have correct level names for {scenario['name']}"

            # Verify warming level names
            unique_wls = result.columns.get_level_values("Warming_Level").unique()
            assert len(unique_wls) == len(
                scenario["warming_levels"]
            ), f"Should have {len(scenario['warming_levels'])} unique warming levels for {scenario['name']}"

            for expected_wl in scenario["expected_wl_names"]:
                assert (
                    expected_wl in unique_wls
                ), f"Should contain {expected_wl} for {scenario['name']}"

            # Verify all hours are present for each warming level
            unique_hours = result.columns.get_level_values("Hour").unique()
            assert (
                len(unique_hours) == 24
            ), f"Should have 24 hours for {scenario['name']}"


class TestCreateMultiWlMultiSimDataframe:
    """Test class for _create_multi_wl_multi_sim_dataframe function.

    Tests the function that creates DataFrames for climate profiles with
    multiple warming levels and multiple simulations together, producing
    (Hour, Warming_Level, Simulation) MultiIndex column structure.

    Attributes
    ----------
    warming_levels : np.ndarray
        Array of warming level values for testing.
    simulations : list
        List of simulation identifiers for testing.
    mock_sim_label_func : MagicMock
        Mock simulation label function.
    hours_per_year : int
        Hours per year (8760).
    profile_data : dict
        Sample profile data dictionary.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.warming_levels = np.array([1.5, 2.0, 3.0])
        self.simulations = ["sim1", "sim2"]
        self.mock_sim_label_func = MagicMock(side_effect=lambda x, i: f"Simulation_{x}")
        self.hours_per_year = 8760

        # Create sample profile data with (warming_level, simulation_label) keys
        # The keys must use the labeled names that sim_label_func produces
        self.profile_data = {}
        for wl in self.warming_levels:
            wl_key = f"WL_{wl}"
            for i, sim in enumerate(self.simulations):
                sim_label = f"Simulation_{sim}"  # Match what sim_label_func returns
                # Create 365x24 matrix for each combination
                profile_matrix = np.random.rand(365, 24) + 20.0 + wl
                self.profile_data[(wl_key, sim_label)] = profile_matrix

    def test_returns_dataframe(self):
        """Test that _create_multi_wl_multi_sim_dataframe returns a pandas DataFrame."""
        # Execute function
        result = _create_multi_wl_multi_sim_dataframe(
            profile_data=self.profile_data,
            warming_levels=self.warming_levels,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: returns DataFrame with correct shape
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == 365, "Should have 365 rows (days)"

        # With 3 warming levels, 2 simulations, and 24 hours: 24 * 3 * 2 = 144 columns
        expected_cols = 24 * len(self.warming_levels) * len(self.simulations)
        assert result.shape[1] == expected_cols, f"Should have {expected_cols} columns"

    def test_multiindex_structure(self):
        """Test that DataFrame has proper (Hour, Warming_Level, Simulation) MultiIndex structure."""
        # Execute function
        result = _create_multi_wl_multi_sim_dataframe(
            profile_data=self.profile_data,
            warming_levels=self.warming_levels,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: proper MultiIndex structure
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should have MultiIndex columns"
        assert result.columns.names == [
            "Hour",
            "Warming_Level",
            "Simulation",
        ], "Should have three levels: Hour, Warming_Level, Simulation"

        # Verify all hours are present
        unique_hours = result.columns.get_level_values("Hour").unique()
        assert len(unique_hours) == 24, "Should have 24 unique hours"
        assert all(h in unique_hours for h in range(1, 25)), "Should have hours 1-24"

        # Verify all warming levels are present
        unique_wls = result.columns.get_level_values("Warming_Level").unique()
        expected_wl_names = [f"WL_{wl}" for wl in self.warming_levels]
        assert len(unique_wls) == len(
            self.warming_levels
        ), f"Should have {len(self.warming_levels)} warming levels"
        for wl_name in expected_wl_names:
            assert wl_name in unique_wls, f"Should contain warming level {wl_name}"

        # Verify all simulations are present
        unique_sims = result.columns.get_level_values("Simulation").unique()
        expected_sim_names = [f"Simulation_{sim}" for sim in self.simulations]
        assert len(unique_sims) == len(
            self.simulations
        ), f"Should have {len(self.simulations)} simulations"
        for sim_name in expected_sim_names:
            assert sim_name in unique_sims, f"Should contain simulation {sim_name}"

    def test_handles_multiple_warming_levels_and_simulations(self):
        """Test proper handling of multiple warming levels and simulations together."""
        # Execute function with multiple warming levels and simulations
        result = _create_multi_wl_multi_sim_dataframe(
            profile_data=self.profile_data,
            warming_levels=self.warming_levels,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: each hour has all combinations of warming levels and simulations
        for hour in range(1, 25):
            hour_cols = result.loc[:, hour]

            # Should have wl_count * sim_count columns for each hour
            expected_cols_per_hour = len(self.warming_levels) * len(self.simulations)
            assert (
                hour_cols.shape[1] == expected_cols_per_hour
            ), f"Hour {hour} should have {expected_cols_per_hour} columns"

            # Verify all warming levels present for this hour
            if isinstance(hour_cols.columns, pd.MultiIndex):
                wls = hour_cols.columns.get_level_values("Warming_Level").unique()
                assert len(wls) == len(
                    self.warming_levels
                ), f"Hour {hour} should have all {len(self.warming_levels)} warming levels"

                # Verify all simulations present for this hour
                sims = hour_cols.columns.get_level_values("Simulation").unique()
                assert len(sims) == len(
                    self.simulations
                ), f"Hour {hour} should have all {len(self.simulations)} simulations"

    def test_preserves_data_integrity(self):
        """Test that data values are preserved correctly in the transformation."""
        # Create specific test data to verify data preservation
        test_warming_levels = np.array([1.5, 2.0])
        test_simulations = ["sim1", "sim2"]
        test_profile_data = {}

        # Create known data patterns for each combination
        for wl in test_warming_levels:
            wl_key = f"WL_{wl}"
            for sim in test_simulations:
                sim_label = f"Simulation_{sim}"
                # Create a matrix where values = day + hour + wl*10 + sim_index
                sim_index = test_simulations.index(sim)
                profile_matrix = np.zeros((365, 24))
                for day in range(365):
                    for hour in range(24):
                        profile_matrix[day, hour] = day + hour + wl * 10 + sim_index
                test_profile_data[(wl_key, sim_label)] = profile_matrix

        # Execute function
        result = _create_multi_wl_multi_sim_dataframe(
            profile_data=test_profile_data,
            warming_levels=test_warming_levels,
            simulations=test_simulations,
            sim_label_func=self.mock_sim_label_func,
            hours_per_year=self.hours_per_year,
        )

        # Verify outcome: data values are preserved correctly
        # Check specific values for a sample of combinations
        for day in [1, 100, 365]:  # Check first, middle, and last day
            for hour in [1, 12, 24]:  # Check different hours
                for wl in test_warming_levels:
                    for sim_idx, sim in enumerate(test_simulations):
                        wl_name = f"WL_{wl}"
                        sim_name = f"Simulation_{sim}"

                        # Get value from result DataFrame
                        result_value = result.loc[day, (hour, wl_name, sim_name)]

                        # Calculate expected value
                        expected_value = (day - 1) + (hour - 1) + wl * 10 + sim_idx

                        assert (
                            abs(result_value - expected_value) < 0.001
                        ), f"Value mismatch at day={day}, hour={hour}, wl={wl}, sim={sim}: expected {expected_value}, got {result_value}"

    def test_complex_combinations(self):
        """Test various complex combinations of warming levels and simulations."""
        # Test different scenarios with varying numbers of warming levels and simulations
        test_scenarios = [
            {
                "name": "single_wl_single_sim",
                "warming_levels": np.array([2.0]),
                "simulations": ["sim1"],
                "expected_cols": 24,  # 24 hours * 1 WL * 1 sim
            },
            {
                "name": "many_wls_many_sims",
                "warming_levels": np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
                "simulations": ["sim1", "sim2", "sim3", "sim4"],
                "expected_cols": 480,  # 24 hours * 5 WLs * 4 sims
            },
            {
                "name": "two_wls_three_sims",
                "warming_levels": np.array([1.5, 3.0]),
                "simulations": ["sim1", "sim2", "sim3"],
                "expected_cols": 144,  # 24 hours * 2 WLs * 3 sims
            },
        ]

        for scenario in test_scenarios:
            # Create profile data for this scenario
            scenario_profile_data = {}

            for wl in scenario["warming_levels"]:
                wl_key = f"WL_{wl}"
                for sim in scenario["simulations"]:
                    sim_label = f"Simulation_{sim}"
                    profile_matrix = np.random.rand(365, 24) + 20.0 + wl
                    scenario_profile_data[(wl_key, sim_label)] = profile_matrix

            # Execute function
            result = _create_multi_wl_multi_sim_dataframe(
                profile_data=scenario_profile_data,
                warming_levels=scenario["warming_levels"],
                simulations=scenario["simulations"],
                sim_label_func=self.mock_sim_label_func,
                hours_per_year=self.hours_per_year,
            )

            # Verify outcome for this scenario
            assert isinstance(
                result, pd.DataFrame
            ), f"Should return DataFrame for {scenario['name']}"
            assert (
                result.shape[0] == 365
            ), f"Should have 365 rows for {scenario['name']}"
            assert (
                result.shape[1] == scenario["expected_cols"]
            ), f"Should have {scenario['expected_cols']} columns for {scenario['name']}"

            # Verify MultiIndex structure
            assert isinstance(
                result.columns, pd.MultiIndex
            ), f"Should have MultiIndex columns for {scenario['name']}"
            assert result.columns.names == [
                "Hour",
                "Warming_Level",
                "Simulation",
            ], f"Should have correct level names for {scenario['name']}"

            # Verify correct number of unique values in each level
            unique_hours = result.columns.get_level_values("Hour").unique()
            assert (
                len(unique_hours) == 24
            ), f"Should have 24 hours for {scenario['name']}"

            unique_wls = result.columns.get_level_values("Warming_Level").unique()
            assert len(unique_wls) == len(
                scenario["warming_levels"]
            ), f"Should have {len(scenario['warming_levels'])} warming levels for {scenario['name']}"

            unique_sims = result.columns.get_level_values("Simulation").unique()
            assert len(unique_sims) == len(
                scenario["simulations"]
            ), f"Should have {len(scenario['simulations'])} simulations for {scenario['name']}"

    def test_create_multi_wl_multi_sim_dataframe_duplicate_simulation_names(self):
        """Test function handles duplicate simulation names with uniqueness suffixes."""
        # Create mock sim_label_func that returns duplicate names
        mock_dup_sim_func = MagicMock()
        mock_dup_sim_func.side_effect = (
            lambda sim, idx: "duplicate_name"
        )  # All return same name

        # Create profile data with keys matching what the function will generate
        # The function creates unique names: first is "duplicate_name",
        # second is "duplicate_name_v1", third is "duplicate_name_v2"
        duplicate_profile_data = {}
        simulations_with_dups = ["model_A", "model_B", "model_C"]
        test_warming_levels = np.array([1.5, 2.0])

        # Add data for original and uniquified names for each warming level
        for wl in test_warming_levels:
            wl_key = f"WL_{wl}"
            for i, unique_suffix in enumerate(
                ["duplicate_name", "duplicate_name_v1", "duplicate_name_v2"]
            ):
                profile_matrix = (
                    np.random.rand(365, 24) + 20.0 + wl + i
                )  # Slightly different data
                duplicate_profile_data[(wl_key, unique_suffix)] = profile_matrix

        # Execute function and verify warning is printed
        with patch("builtins.print") as mock_print:
            result = _create_multi_wl_multi_sim_dataframe(
                profile_data=duplicate_profile_data,
                warming_levels=test_warming_levels,
                simulations=simulations_with_dups,
                sim_label_func=mock_dup_sim_func,
                hours_per_year=self.hours_per_year,
            )

        # Verify outcome: warning message was printed about duplicates
        printed_calls = [str(call) for call in mock_print.call_args_list]
        printed_output = " ".join(printed_calls)
        assert (
            "duplicate simulation names" in printed_output.lower()
        ), "Should warn about duplicate simulation names"
        assert (
            "uniqueness suffixes" in printed_output.lower()
        ), "Should mention adding uniqueness suffixes"

        # Verify the sim_label_func was called for each simulation
        assert mock_dup_sim_func.call_count == len(
            simulations_with_dups
        ), "Should call sim_label_func for each simulation"

        # Verify the result has the uniquified names in columns
        assert isinstance(result, pd.DataFrame), "Should return a DataFrame"
        unique_sims = result.columns.get_level_values("Simulation").unique()

        # Should have 3 unique simulation names after de-duplication
        assert (
            len(unique_sims) == 3
        ), "Should have 3 unique simulations after de-duplication"
        assert "duplicate_name" in unique_sims, "Should contain original duplicate_name"
        assert "duplicate_name_v1" in unique_sims, "Should contain duplicate_name_v1"
        assert "duplicate_name_v2" in unique_sims, "Should contain duplicate_name_v2"

        # Verify data exists for all combinations of warming levels and unique simulation names
        for wl in test_warming_levels:
            wl_name = f"WL_{wl}"
            for unique_sim in [
                "duplicate_name",
                "duplicate_name_v1",
                "duplicate_name_v2",
            ]:
                # Each hour-wl-sim combination should have a column
                for hour in [1, 12, 24]:  # Sample a few hours
                    # Check if the column exists in MultiIndex
                    col_tuple = (hour, wl_name, unique_sim)
                    assert (
                        col_tuple in result.columns
                    ), f"Should have column for (hour={hour}, wl={wl_name}, sim={unique_sim})"

                    # Verify data length for this column
                    col_data = result[col_tuple]
                    assert (
                        len(col_data) == 365
                    ), f"Should have 365 days for hour={hour}, wl={wl_name}, sim={unique_sim}"

class TestGetStationCoordinates:
    """Test class for _get_station_coordinates function.

    Tests the function that looks up latitude and longitude coordinates
    for a given weather station name from the DataInterface.

    Attributes
    ----------
    mock_data_interface : MagicMock
        Mock DataInterface instance.
    mock_stations_gdf : MagicMock
        Mock GeoDataFrame with station data.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock stations GeoDataFrame
        self.mock_stations_gdf = pd.DataFrame(
            {
                "station": [
                    "San Diego Lindbergh Field (KSAN)",
                    "Los Angeles International Airport (KLAX)",
                    "San Francisco International Airport (KSFO)",
                ],
                "LAT_Y": [32.7336, 33.9416, 37.6213],
                "LON_X": [-117.1831, -118.4085, -122.3790],
            }
        )

    def test_get_station_coordinates_returns_tuple(self):
        """Test that _get_station_coordinates returns a tuple of coordinates."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            result = _get_station_coordinates("San Diego Lindbergh Field (KSAN)")

            # Verify outcome: returns tuple of (lat, lon)
            assert isinstance(result, tuple), "Should return a tuple"
            assert len(result) == 2, "Tuple should contain exactly 2 elements"
            lat, lon = result
            assert isinstance(
                lat, (int, float, np.number)
            ), "Latitude should be numeric"
            assert isinstance(
                lon, (int, float, np.number)
            ), "Longitude should be numeric"

    def test_get_station_coordinates_returns_correct_values(self):
        """Test that _get_station_coordinates returns correct coordinate values."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            lat, lon = _get_station_coordinates("San Diego Lindbergh Field (KSAN)")

            # Verify outcome: returns correct coordinates
            assert lat == 32.7336, "Should return correct latitude"
            assert lon == -117.1831, "Should return correct longitude"

    def test_get_station_coordinates_with_different_stations(self):
        """Test _get_station_coordinates with multiple different stations."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            # Test LAX station
            lat_lax, lon_lax = _get_station_coordinates(
                "Los Angeles International Airport (KLAX)"
            )
            assert lat_lax == 33.9416, "Should return correct LAX latitude"
            assert lon_lax == -118.4085, "Should return correct LAX longitude"

            # Test SFO station
            lat_sfo, lon_sfo = _get_station_coordinates(
                "San Francisco International Airport (KSFO)"
            )
            assert lat_sfo == 37.6213, "Should return correct SFO latitude"
            assert lon_sfo == -122.3790, "Should return correct SFO longitude"

    def test_get_station_coordinates_raises_error_for_invalid_station(self):
        """Test that _get_station_coordinates raises ValueError for invalid station name."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            with pytest.raises(ValueError, match="Station .* not found"):
                _get_station_coordinates("Invalid Station Name (XXX)")

    def test_get_station_coordinates_raises_error_for_empty_string(self):
        """Test that _get_station_coordinates raises ValueError for empty station name."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            with pytest.raises(ValueError, match="Station .* not found"):
                _get_station_coordinates("")


class TestConvertStationsToLatLon:
    """Test class for _convert_stations_to_lat_lon function.

    Tests the function that converts a list of station names to lat/lon
    bounds with a specified buffer for spatial subsetting.

    Attributes
    ----------
    mock_stations_gdf : pd.DataFrame
        Mock GeoDataFrame with station data.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock stations GeoDataFrame
        self.mock_stations_gdf = pd.DataFrame(
            {
                "station": [
                    "San Diego Lindbergh Field (KSAN)",
                    "Los Angeles International Airport (KLAX)",
                    "San Francisco International Airport (KSFO)",
                ],
                "LAT_Y": [32.7336, 33.9416, 37.6213],
                "LON_X": [-117.1831, -118.4085, -122.3790],
            }
        )

    def test_convert_stations_to_lat_lon_returns_tuple_of_tuples(self):
        """Test that _convert_stations_to_lat_lon returns tuple of (lat_bounds, lon_bounds)."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            result = _convert_stations_to_lat_lon(
                ["San Diego Lindbergh Field (KSAN)"], buffer=0.02
            )

            # Verify outcome: returns tuple of two tuples
            assert isinstance(result, tuple), "Should return a tuple"
            assert (
                len(result) == 2
            ), "Should contain 2 elements (lat_bounds, lon_bounds)"
            lat_bounds, lon_bounds = result
            assert isinstance(lat_bounds, tuple), "lat_bounds should be a tuple"
            assert isinstance(lon_bounds, tuple), "lon_bounds should be a tuple"
            assert len(lat_bounds) == 2, "lat_bounds should have min and max"
            assert len(lon_bounds) == 2, "lon_bounds should have min and max"

    def test_convert_stations_to_lat_lon_single_station_with_buffer(self):
        """Test _convert_stations_to_lat_lon with single station applies buffer correctly."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            # Execute function with default buffer (0.02)
            lat_bounds, lon_bounds = _convert_stations_to_lat_lon(
                ["San Diego Lindbergh Field (KSAN)"], buffer=0.02
            )

            # Verify outcome: bounds are station coordinates +/- buffer
            # San Diego: lat=32.7336, lon=-117.1831
            expected_lat_min = 32.7336 - 0.02
            expected_lat_max = 32.7336 + 0.02
            expected_lon_min = -117.1831 - 0.02
            expected_lon_max = -117.1831 + 0.02

            assert (
                abs(lat_bounds[0] - expected_lat_min) < 1e-6
            ), "Should apply buffer to min latitude"
            assert (
                abs(lat_bounds[1] - expected_lat_max) < 1e-6
            ), "Should apply buffer to max latitude"
            assert (
                abs(lon_bounds[0] - expected_lon_min) < 1e-6
            ), "Should apply buffer to min longitude"
            assert (
                abs(lon_bounds[1] - expected_lon_max) < 1e-6
            ), "Should apply buffer to max longitude"

    def test_convert_stations_to_lat_lon_multiple_stations(self):
        """Test _convert_stations_to_lat_lon with multiple stations creates bounding box."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            # Execute function with two stations
            lat_bounds, lon_bounds = _convert_stations_to_lat_lon(
                [
                    "San Diego Lindbergh Field (KSAN)",
                    "Los Angeles International Airport (KLAX)",
                ],
                buffer=0.02,
            )

            # Verify outcome: bounds encompass both stations with buffer
            # San Diego: lat=32.7336, lon=-117.1831
            # LAX: lat=33.9416, lon=-118.4085
            expected_lat_min = 32.7336 - 0.02  # Min from San Diego
            expected_lat_max = 33.9416 + 0.02  # Max from LAX
            expected_lon_min = -118.4085 - 0.02  # Min (most negative) from LAX
            expected_lon_max = -117.1831 + 0.02  # Max (least negative) from San Diego

            assert (
                abs(lat_bounds[0] - expected_lat_min) < 1e-6
            ), "Should use minimum latitude with buffer"
            assert (
                abs(lat_bounds[1] - expected_lat_max) < 1e-6
            ), "Should use maximum latitude with buffer"
            assert (
                abs(lon_bounds[0] - expected_lon_min) < 1e-6
            ), "Should use minimum longitude with buffer"
            assert (
                abs(lon_bounds[1] - expected_lon_max) < 1e-6
            ), "Should use maximum longitude with buffer"

    def test_convert_stations_to_lat_lon_with_custom_buffer(self):
        """Test _convert_stations_to_lat_lon with custom buffer value."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            # Execute function with custom buffer
            custom_buffer = 0.05
            lat_bounds, lon_bounds = _convert_stations_to_lat_lon(
                ["San Diego Lindbergh Field (KSAN)"], buffer=custom_buffer
            )

            # Verify outcome: uses custom buffer value
            expected_lat_min = 32.7336 - custom_buffer
            expected_lat_max = 32.7336 + custom_buffer
            expected_lon_min = -117.1831 - custom_buffer
            expected_lon_max = -117.1831 + custom_buffer

            assert (
                abs(lat_bounds[0] - expected_lat_min) < 1e-6
            ), "Should apply custom buffer to min latitude"
            assert (
                abs(lat_bounds[1] - expected_lat_max) < 1e-6
            ), "Should apply custom buffer to max latitude"
            assert (
                abs(lon_bounds[0] - expected_lon_min) < 1e-6
            ), "Should apply custom buffer to min longitude"
            assert (
                abs(lon_bounds[1] - expected_lon_max) < 1e-6
            ), "Should apply custom buffer to max longitude"

    def test_convert_stations_to_lat_lon_raises_error_for_empty_list(self):
        """Test that _convert_stations_to_lat_lon raises ValueError for empty station list."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            with pytest.raises(
                ValueError, match="No stations provided for coordinate conversion"
            ):
                _convert_stations_to_lat_lon([], buffer=0.02)

    def test_convert_stations_to_lat_lon_raises_error_for_invalid_station(self):
        """Test that _convert_stations_to_lat_lon raises ValueError when station not found."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            with pytest.raises(ValueError, match="Station .* not found"):
                _convert_stations_to_lat_lon(
                    ["Invalid Station Name (XXX)"], buffer=0.02
                )

    def test_convert_stations_to_lat_lon_with_three_stations(self):
        """Test _convert_stations_to_lat_lon creates correct bounding box for three stations."""
        with patch(
            "climakitae.explore.standard_year_profile.pd.read_csv",
            return_value=self.mock_stations_gdf,
        ):
            # Execute function with all three stations
            lat_bounds, lon_bounds = _convert_stations_to_lat_lon(
                [
                    "San Diego Lindbergh Field (KSAN)",
                    "Los Angeles International Airport (KLAX)",
                    "San Francisco International Airport (KSFO)",
                ],
                buffer=0.02,
            )

            # Verify outcome: bounds encompass all three stations
            # San Diego: lat=32.7336 (min), lon=-117.1831 (max/least negative)
            # LAX: lat=33.9416, lon=-118.4085
            # SFO: lat=37.6213 (max), lon=-122.3790 (min/most negative)
            expected_lat_min = 32.7336 - 0.02
            expected_lat_max = 37.6213 + 0.02
            expected_lon_min = -122.3790 - 0.02
            expected_lon_max = -117.1831 + 0.02

            assert (
                abs(lat_bounds[0] - expected_lat_min) < 1e-6
            ), "Should use minimum latitude from all stations"
            assert (
                abs(lat_bounds[1] - expected_lat_max) < 1e-6
            ), "Should use maximum latitude from all stations"
            assert (
                abs(lon_bounds[0] - expected_lon_min) < 1e-6
            ), "Should use minimum longitude from all stations"
            assert (
                abs(lon_bounds[1] - expected_lon_max) < 1e-6
            ), "Should use maximum longitude from all stations"


class TestHelperFunctions:
    """Test class for any helper functions not explictly tested in other classes."""

    def test__handle_location_params_station(self):
        """Test a valid station configuration for the location parameter."""
        settings = {
            "location": "Ontario International Airport (KONT)",
        }
        new_settings = _handle_location_params(**settings)
        assert "stations" in new_settings
        assert isinstance(new_settings["stations"], list)
        assert new_settings["stations"] == ["Ontario International Airport (KONT)"]

    def test__handle_location_params_cached_area(self):
        """Test a valid cached_area configuration for the location parameter."""
        settings = {
            "location": "Alpine County",
        }
        new_settings = _handle_location_params(**settings)
        assert "cached_area" in new_settings
        assert isinstance(new_settings["cached_area"], str)
        assert new_settings["cached_area"] == "Alpine County"

    def test__handle_location_params_lat_lon(self):
        """Test a valid lat/lon configuration for the location parameter."""
        settings = {
            "location": (-120.9, 35.8),
            "resolution": "9 km",
        }
        new_settings = _handle_location_params(**settings)
        for coord in ["latitude", "longitude"]:
            assert coord in new_settings
            assert isinstance(new_settings[coord], Tuple)
        assert new_settings["longitude"] == pytest.approx(
            (-120.9 - 0.08, -120.9 + 0.08), 1e-6
        )
        assert new_settings["latitude"] == pytest.approx(
            (35.8 - 0.08, 35.8 + 0.08), 1e-6
        )

    def test__handle_location_params_invalid_lat_lon(self):
        """Check three invalid location parameter settings with lat/lon."""
        # Longtiude not a negative value
        settings = {
            "location": (220, 35.8),
            "resolution": "9 km",
        }
        with pytest.raises(
            ValueError,
            match="Expected a positive-valued latitude coordinate and negative-valued longitude coordinate.",
        ):
            _ = _handle_location_params(**settings)

        # Bad coordinate list
        settings["location"] = [(-120, 35), (-121, 35)]
        with pytest.raises(
            TypeError,
            match="The location list may only contain string or numeric values, not tuple.",
        ):
            _ = _handle_location_params(**settings)

        # Too many coordinates
        settings["location"] = [-120, -120, 35]
        with pytest.raises(
            ValueError,
            match="Length of `location` parameter must be two if providing a coordinate pair.",
        ):
            _ = _handle_location_params(**settings)

    def test__handle_location_params_invalid_type(self):
        """Test setting location parameter with invalid dictionary type."""
        settings = {
            "location": {"sacramento": (38.59373314007944, -121.49188107549107)},
            "resolution": "3 km",
        }
        with pytest.raises(
            TypeError,
            match="The `location` parameter type should str, List, or Tuple if set. Got type dict.",
        ):
            _ = _handle_location_params(**settings)

    @pytest.mark.parametrize(
        "resolution,expected_buffer",
        [
            pytest.param("3 km", 0.02),
            pytest.param("9 km", 0.08),
            pytest.param("45 km", 0.35),
            pytest.param(None, 0.02),
        ],
    )
    def test__get_buffer_from_resolution(self, resolution, expected_buffer):
        """Test that correct buffer size is returned for each resolution."""
        test_buffer = _get_buffer_from_resolution(resolution)
        assert test_buffer == expected_buffer
