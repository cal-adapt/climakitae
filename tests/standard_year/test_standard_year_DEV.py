"""
Unit tests for climakitae/explore/standard_year_profile.py

This module contains comprehensive unit tests for the Standard Year and climate
profile computation functions that provide climate profile analysis.

"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.explore.standard_year_profile import (
    _compute_difference_profile,
    _compute_mixed_index_difference,
    _compute_multiindex_difference,
    _compute_simple_difference,
    _compute_simulation_paired_difference,
    _compute_warming_level_difference,
    _construct_profile_dataframe,
    _convert_stations_to_lat_lon,
    _create_multi_wl_multi_sim_dataframe,
    _create_multi_wl_single_sim_dataframe,
    _create_simple_dataframe,
    _create_single_wl_multi_sim_dataframe,
    _find_matching_historic_column,
    _find_matching_historic_value,
    _format_based_on_structure,
    _format_meteo_yr_df,
    _get_clean_standardyr_filename,
    _get_historic_hour_mean,
    _get_station_coordinates,
    _stack_profile_data,
    compute_profile,
    export_profile_to_csv,
    get_climate_profile,
    get_profile_metadata,
    get_profile_units,
    retrieve_profile_data,
    set_profile_metadata,
)


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
            (wl_key, sim_key): np.random.rand(365, 24) + 20.0  # Temperature-like data
        }

        # Simple function to get simulation labels
        def sim_label_func(sim, sim_idx):
            return f"Sim{sim_idx + 1}"

        self.sim_label_func = sim_label_func
        self.days_in_year = 365
        self.hours = np.arange(1, 25, 1)  # Hours 1-24
        self.hours_per_day = 24

    def test_create_simple_dataframe_returns_dataframe(self):
        """Test _create_simple_dataframe returns pd.DataFrame."""
        # Execute function
        result = _create_simple_dataframe(
            profile_data=self.profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
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
            days_in_year=365,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: correct MultiIndex column structure
        assert isinstance(result.columns, pd.MultiIndex), "Columns should be MultiIndex"
        assert result.columns.names == [
            "Hour",
            "Simulation",
        ], "Column levels should be named Hour and Simulation"

        # Verify expected dimensions: 365 rows, 24 columns
        expected_rows = 365
        expected_cols = 24
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

    def test_create_simple_dataframe_with_different_scenarios(self):
        """Test _create_simple_dataframe with different warming level and simulation scenarios."""
        # Test different warming level
        different_wl = 1.5
        different_sim = "Sim1"
        different_wl_data = {("WL_1.5", "Sim1"): np.random.rand(365, 24) + 15.0}

        # Execute function with different warming level
        result_wl = _create_simple_dataframe(
            profile_data=different_wl_data,
            warming_level=different_wl,
            simulation=different_sim,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: maintains same structure with different data
        assert isinstance(
            result_wl, pd.DataFrame
        ), "Should return DataFrame for different WL"
        assert result_wl.shape == (365, 24), "Should maintain same shape"

        # Test different simulation identifier
        different_sim = "sim2"
        # Note: sim_label_func always uses index 0, so key will be "Sim1" regardless of simulation value
        different_sim_data = {("WL_2.0", "Sim1"): np.random.rand(365, 24) + 25.0}

        # Execute function with different simulation
        result_sim = _create_simple_dataframe(
            profile_data=different_sim_data,
            warming_level=self.warming_level,
            simulation=different_sim,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: handles different simulation correctly
        assert isinstance(
            result_sim, pd.DataFrame
        ), "Should return DataFrame for different sim"
        assert result_sim.shape == (365, 24), "Should maintain same shape"

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
        test_data = np.ones((365, 24)) * 42.5  # All values set to 42.5
        test_profile_data = {("WL_2.0", "Sim1"): test_data}

        # Execute function
        result = _create_simple_dataframe(
            profile_data=test_profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
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
        leap_year_data = np.ones((366, 24)) * 33.7  # Leap year with different values
        leap_year_profile_data = {("WL_2.0", "Sim1"): leap_year_data}

        # Execute function with leap year data
        result_leap = _create_simple_dataframe(
            profile_data=leap_year_profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            days_in_year=366,  # Leap year
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: handles different matrix sizes correctly
        assert result_leap.shape == (
            366,
            24,
        ), "Should handle leap year shape (366 days)"
        assert np.all(result_leap.values == 33.7), "All leap year values should be 33.7"

        # Verify proper index for leap year
        expected_leap_index = np.arange(1, 367, 1)  # 1 to 366
        np.testing.assert_array_equal(result_leap.index.values, expected_leap_index)

    def test_create_simple_dataframe_with_different_year_lengths(self):
        """Test _create_simple_dataframe with different days_in_year parameter values."""
        # Test with various year lengths
        year_length_scenarios = [
            {"days": 365, "description": "regular year"},
            {"days": 366, "description": "leap year"},
            {"days": 360, "description": "simplified calendar year"},
            {"days": 300, "description": "partial year"},
        ]

        for scenario in year_length_scenarios:
            days = scenario["days"]
            description = scenario["description"]

            # Create profile data matching the year length
            profile_matrix = np.random.rand(days, 24) + 18.0
            test_data = {("WL_2.0", "Sim1"): profile_matrix}

            # Execute function
            result = _create_simple_dataframe(
                profile_data=test_data,
                warming_level=self.warming_level,
                simulation=self.simulation,
                sim_label_func=self.sim_label_func,
                days_in_year=days,
                hours=self.hours,
                hours_per_day=self.hours_per_day,
            )

            # Verify outcome: correct dimensions for each scenario
            assert isinstance(
                result, pd.DataFrame
            ), f"Should return DataFrame for {description}"
            assert result.shape == (
                days,
                24,
            ), f"Should have {days} rows for {description}"
            assert result.shape[1] == 24, f"Should have 24 columns for {description}"

            # Verify proper index generation
            expected_index = np.arange(1, days + 1, 1)
            np.testing.assert_array_equal(
                result.index.values,
                expected_index,
                err_msg=f"Index should be 1 to {days} for {description}",
            )

            # Verify columns remain consistent regardless of year length
            expected_columns = pd.MultiIndex.from_tuples(
                [(i, "Sim1") for i in range(1, 25)]
            )
            np.testing.assert_array_equal(
                result.columns.values,
                expected_columns,
                err_msg=f"Columns should always be 1-24 for {description}",
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
        self.hours = np.arange(0, 24)
        self.days_in_year = 365
        self.hours_per_day = 24

        # Create sample profile data dictionary
        # The function expects data for each (WL_X, sim_label) combination
        self.sample_profile_data = {}
        for i, sim in enumerate(self.simulations):
            sim_key = f"sim_{sim}_{i}"
            wl_key = f"WL_{self.warming_level}"
            # Create random data for each simulation (365 days x 24 hours)
            profile_matrix = np.random.rand(365, 24) + 20.0
            self.sample_profile_data[(wl_key, sim_key)] = profile_matrix

    def test_create_single_wl_multi_sim_dataframe_returns_dataframe(self):
        """Test that _create_single_wl_multi_sim_dataframe returns a pandas DataFrame."""
        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_level=self.warming_level,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
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
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: correct MultiIndex column structure
        assert isinstance(result.columns, pd.MultiIndex), "Columns should be MultiIndex"
        assert result.columns.names == [
            "Hour",
            "Simulation",
        ], "Column levels should be named Hour and Simulation"

        # Verify expected dimensions: 365 rows, (24 hours × 3 simulations) columns
        expected_rows = 365
        expected_cols = 24 * len(self.simulations)  # 24 hours × 3 simulations = 72
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

    def test_create_single_wl_multi_sim_dataframe_handles_multiple_simulations(self):
        """Test function handles multiple simulations correctly."""
        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_level=self.warming_level,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: each simulation creates columns for all hours
        # Expected structure: (hour, sim) for each hour and each simulation
        unique_simulations = result.columns.get_level_values("Simulation").unique()
        unique_hours = result.columns.get_level_values("Hour").unique()

        # Should have one column for each (hour, simulation) combination
        expected_sim_names = ["sim_model_A_0", "sim_model_B_1", "sim_model_C_2"]
        assert len(unique_simulations) == len(
            self.simulations
        ), f"Should have {len(self.simulations)} simulations"
        assert len(unique_hours) == len(
            self.hours
        ), f"Should have {len(self.hours)} hours"

        # Verify simulation names match expected pattern from mock function
        for expected_sim in expected_sim_names:
            assert (
                expected_sim in unique_simulations
            ), f"Should contain simulation {expected_sim}"

        # Verify each hour appears for each simulation (24 hours × 3 sims = 72 columns)
        for hour in self.hours:
            for sim_name in expected_sim_names:
                assert (
                    hour,
                    sim_name,
                ) in result.columns, (
                    f"Should have column for hour {hour}, simulation {sim_name}"
                )

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
        wl_key = f"WL_{self.warming_level}"

        # Add data for original and uniquified names
        for i, unique_suffix in enumerate(
            ["duplicate_name", "duplicate_name_v1", "duplicate_name_v2"]
        ):
            profile_matrix = (
                np.random.rand(365, 24) + 20.0 + i
            )  # Slightly different data
            duplicate_profile_data[(wl_key, unique_suffix)] = profile_matrix

        # Execute function and verify warning is printed
        with patch("builtins.print") as mock_print:
            result = _create_single_wl_multi_sim_dataframe(
                profile_data=duplicate_profile_data,
                warming_level=self.warming_level,
                simulations=simulations_with_dups,
                sim_label_func=mock_dup_sim_func,
                days_in_year=self.days_in_year,
                hours=self.hours,
                hours_per_day=self.hours_per_day,
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

    def test_create_single_wl_multi_sim_dataframe_preserves_data_integrity(self):
        """Test that profile data values are correctly preserved in MultiIndex structure."""
        # Create specific test data with known values for verification
        test_simulations = ["test_sim_A", "test_sim_B"]
        test_hours = np.array([0, 1, 2])  # Use smaller subset for easier verification
        test_days = 3  # Use smaller dataset for precise testing

        # Create mock sim_label_func for predictable names
        test_sim_func = MagicMock()
        test_sim_func.side_effect = lambda sim, idx: f"test_{sim}_{idx}"

        # Create test profile data with known values
        test_profile_data = {}
        expected_values = {}

        for i, sim in enumerate(test_simulations):
            sim_key = f"test_{sim}_{i}"
            wl_key = f"WL_{self.warming_level}"
            # Create known test data: day i, hour j has value (i+1)*10 + j
            profile_matrix = np.zeros((test_days, len(test_hours)))
            for day in range(test_days):
                for hour_idx, hour in enumerate(test_hours):
                    profile_matrix[day, hour_idx] = (day + 1) * 10 + hour

            test_profile_data[(wl_key, sim_key)] = profile_matrix
            expected_values[sim_key] = profile_matrix

        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=test_profile_data,
            warming_level=self.warming_level,
            simulations=test_simulations,
            sim_label_func=test_sim_func,
            days_in_year=test_days,
            hours=test_hours,
            hours_per_day=len(test_hours),
        )

        # Verify outcome: data integrity is preserved
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.shape == (
            test_days,
            len(test_hours) * len(test_simulations),
        ), "Should have correct dimensions"

        # Verify specific data values are preserved for each (hour, simulation) combination
        for hour in test_hours:
            for i, sim in enumerate(test_simulations):
                sim_key = f"test_{sim}_{i}"
                expected_matrix = expected_values[sim_key]

                # Get column data for this (hour, simulation) combination
                column_data = result[(hour, sim_key)]
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
            result.loc[1, (0, sim_a_key)] == 10.0
        ), "Day 1, Hour 0, Sim A should be 10"

        # Day 2 (index 1), Hour 1, Sim B should be 21 (day 2 * 10 + hour 1)
        sim_b_key = "test_test_sim_B_1"
        assert (
            result.loc[2, (1, sim_b_key)] == 21.0
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
        self.hours = np.arange(0, 24)
        self.days_in_year = 365
        self.hours_per_day = 24

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
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
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
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
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
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
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
            days_in_year=test_days,
            hours=test_hours,
            hours_per_day=len(test_hours),
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
                days_in_year=self.days_in_year,
                hours=self.hours,
                hours_per_day=self.hours_per_day,
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
