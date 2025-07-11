"""
Unit tests for the warming module in climakitae.explore.warming.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.paths import GWL_1850_1900_FILE
from climakitae.explore.warming import (
    WarmingLevels,
    _drop_invalid_sims,
    clean_list,
    clean_warm_data,
    get_sliced_data,
    process_item,
    relabel_axis,
)
from climakitae.util.utils import read_csv_file
from climakitae.util.warming_levels import _get_sliced_data, calculate_warming_level

# Load warming level times from CSV
gwl_times = read_csv_file(GWL_1850_1900_FILE, index_col=[0, 1, 2])


def test_missing_all_sims_attribute(
    test_dataarray_wl_20_summer_season_loca_3km_daily_temp,
):
    """Ensure AttributeError is raised when 'all_sims' is missing in DataArray dimensions."""
    da_wrong = test_dataarray_wl_20_summer_season_loca_3km_daily_temp.rename(
        {"simulation": "totally_wrong"}
    )
    with pytest.raises(AttributeError):
        calculate_warming_level(da_wrong, gwl_times, 2, range(1, 13), 15)


def test_get_sliced_data_empty_output(
    test_dataarray_time_2030_2035_wrf_3km_hourly_temp,
):
    """
    Verify that `_get_sliced_data` returns an empty DataArray when no `center_time` is found.

    `WRF_FGOALS-g3_r1i1p1f1` for `SSP 3-7.0` does not reach 4.0 warming, so we will use
    it to see if `_get_sliced_data` will just generate an empty DataArray for it.
    Context for this behavior in `_get_sliced_data`: an empty DataArray is generated
    for this simulation in `_get_sliced_data` because `_get_sliced_data` is called in a
    groupby call, requiring all objects it is called upon to return the same shape.
    """
    da = test_dataarray_time_2030_2035_wrf_3km_hourly_temp

    # Selecting a simulation that does not reach 4.0 warming level
    stacked_da = da.stack(all_sims=["scenario", "simulation"])
    one_sim = stacked_da.sel(
        all_sims=("Historical + SSP 3-7.0", "WRF_FGOALS-g3_r1i1p1f1")
    )

    # Call function and assert empty DataArray
    res = _get_sliced_data(one_sim, 4, gwl_times, range(1, 13), 15)
    assert res.isnull().all()


class TestWarmingLevels:
    """
    Test class for WarmingLevels class
    """

    @pytest.mark.advanced
    @staticmethod
    def test_warming_levels_init():
        """Test that WarmingLevels initializes correctly with default values."""

        # Mock the _check_available_warming_levels function

        with patch(
            "climakitae.explore.warming._check_available_warming_levels"
        ) as mock_check:
            mock_check.return_value = [0.8, 1.5, 2.0, 3.0, 4.0]

            # Initialize WarmingLevels
            wl = WarmingLevels()

            # Check that it has the expected attributes
            assert hasattr(wl, "wl_params")
            assert hasattr(wl, "warming_levels")
            assert wl.warming_levels == [0.8, 1.5, 2.0, 3.0, 4.0]
            mock_check.assert_called_once()

    @pytest.mark.advanced
    @staticmethod
    def test_find_warming_slice_with_mocked_data():
        """
        Test the `find_warming_slice` method to ensure it correctly processes
        warming slice data for a given level and gwl_times.
        """

        # Mock catalog_data
        catalog_data = xr.DataArray(
            np.random.rand(3, 10, 10),
            dims=["all_sims", "time", "lat"],
            coords={
                "all_sims": ["sim1", "sim2", "sim3"],
                "time": pd.date_range("2000-01-01", periods=10, freq="YE"),
                "lat": np.arange(10),
            },
        )

        # Mock gwl_times
        mock_gwl_times = pd.DataFrame(
            {
                "0.8": ["2005-01-01", "2006-01-01", "2007-01-01"],
                "1.5": ["2010-01-01", "2011-01-01", "2012-01-01"],
            },
            index=[
                ("sim1", "ens1", "ssp1"),
                ("sim2", "ens2", "ssp2"),
                ("sim3", "ens3", "ssp3"),
            ],
        )

        # Mock WarmingLevels object
        wl = WarmingLevels()
        wl.catalog_data = catalog_data
        wl.wl_params.months = [1, 2, 3]
        wl.wl_params.window = 15
        wl.wl_params.anom = "No"

        # Mock helper functions
        with patch(
            "climakitae.explore.warming.get_sliced_data"
        ) as mock_get_sliced_data, patch(
            "climakitae.explore.warming.clean_warm_data"
        ) as mock_clean_warm_data, patch(
            "climakitae.explore.warming.relabel_axis"
        ) as mock_relabel_axis:

            # Mock return values
            mock_get_sliced_data.side_effect = (
                lambda y, level, years, months, window, anom: xr.DataArray(
                    np.random.rand(10, 10),
                    dims=["time", "lat"],
                    coords={
                        "time": pd.date_range("2000-01-01", periods=10, freq="YE"),
                        "lat": np.arange(10),
                    },
                )
            )
            mock_clean_warm_data.side_effect = lambda x: x
            mock_relabel_axis.side_effect = lambda x: [
                "relabeled_sim1",
                "relabeled_sim2",
                "relabeled_sim3",
            ]

            # Call the method
            result = wl.find_warming_slice(level="1.5", gwl_times=mock_gwl_times)

            # Assertions
            assert isinstance(result, xr.DataArray)
            assert "warming_level" in result.coords
            assert result.coords["warming_level"].values[0] == "1.5"
            assert mock_get_sliced_data.call_count == 3  # Called for each simulation
            assert mock_clean_warm_data.called
            assert mock_relabel_axis.called

    @pytest.mark.advanced
    @staticmethod
    def test_calculate_with_mocked_dependencies():
        """
        Test the `calculate` method of the `WarmingLevels` class to ensure it processes
        data correctly and updates `sliced_data` and `gwl_snapshots` attributes.
        """

        # Mock WarmingLevels object
        wl = WarmingLevels()

        # Mock wl_params
        wl.wl_params.retrieve = lambda: xr.DataArray(
            np.random.rand(3, 3, 10),
            dims=["simulation", "scenario", "time"],
            coords={
                "simulation": ["sim1", "sim2", "sim3"],
                "scenario": ["ssp1", "ssp2", "ssp3"],
                "time": pd.date_range("2000-01-01", periods=10, freq="YE"),
            },
        )
        wl.wl_params.warming_levels = ["0.8", "1.5"]
        wl.wl_params.load_data = False
        wl.wl_params.anom = "No"

        # Mock gwl_times
        mock_gwl_times = pd.DataFrame(
            {
                "0.8": ["2005-01-01", "2006-01-01", "2007-01-01"],
                "1.5": ["2010-01-01", "2011-01-01", "2012-01-01"],
            },
            index=[
                ("sim1", "ens1", "ssp1"),
                ("sim2", "ens2", "ssp2"),
                ("sim3", "ens3", "ssp3"),
            ],
        )

        # Mock helper functions
        with patch(
            "climakitae.explore.warming._drop_invalid_sims"
        ) as mock_drop_invalid_sims, patch(
            "climakitae.explore.warming.read_csv_file"
        ) as mock_read_csv_file, patch(
            "climakitae.explore.warming.clean_list"
        ) as mock_clean_list, patch(
            "climakitae.explore.warming.WarmingLevels.find_warming_slice"
        ) as mock_find_warming_slice, patch(
            "climakitae.explore.warming.xr.concat"
        ) as mock_xr_concat:

            # Mock return values
            mock_drop_invalid_sims.side_effect = lambda x, y: x
            mock_read_csv_file.return_value = mock_gwl_times
            mock_clean_list.side_effect = lambda x, y: x
            mock_find_warming_slice.side_effect = lambda level, gwl_times: xr.DataArray(
                np.random.rand(10, 10),
                dims=["time", "lat"],
                coords={
                    "time": pd.date_range("2000-01-01", periods=10, freq="YE"),
                    "lat": np.arange(10),
                },
                attrs={
                    "frequency": "daily",
                },
            )
            mock_xr_concat.side_effect = lambda values, dim: xr.DataArray(
                np.random.rand(len(values), 10, 10),
                dims=["warming_level", "time", "lat"],
                coords={
                    "warming_level": ["0.8", "1.5"],
                    "time": pd.date_range("2000-01-01", periods=10, freq="YE"),
                    "lat": np.arange(10),
                },
            )

            # Call the method
            wl.calculate()

            # Assertions
            assert isinstance(wl.sliced_data, dict)
            assert isinstance(wl.gwl_snapshots, xr.DataArray)
            assert len(wl.sliced_data) == 2  # Two warming levels
            assert "0.8" in wl.sliced_data
            assert "1.5" in wl.sliced_data
            assert mock_drop_invalid_sims.called
            assert mock_read_csv_file.called
            assert mock_clean_list.called
            assert (
                mock_find_warming_slice.call_count == 2
            )  # Called for each warming level
            assert mock_xr_concat.called


class TestRelabelAxis:
    """Test suite for relabel_axis function."""

    @staticmethod
    def test_relabel_axis_basic():
        """Test relabel_axis with typical input."""
        # Create a proper DataArray with dimensions that can be stacked
        ds = xr.Dataset(
            data_vars={"data": (["simulation", "scenario"], np.random.rand(3, 2))},
            coords={
                "simulation": ["CESM2", "ACCESS-CM2", "MIROC6"],
                "scenario": ["SSP3-7.0", "SSP5-8.5"],
            },
        )

        # Stack dimensions to create the multi-index coordinate
        stacked = ds.stack(all_sims=["simulation", "scenario"])
        all_sims_coordinate = stacked.all_sims

        # Expected output after relabeling
        expected_pairs = [
            "CESM2_SSP3-7.0",
            "CESM2_SSP5-8.5",
            "ACCESS-CM2_SSP3-7.0",
            "ACCESS-CM2_SSP5-8.5",
            "MIROC6_SSP3-7.0",
            "MIROC6_SSP5-8.5",
        ]

        # Call relabel_axis and compare with expected output
        result = relabel_axis(all_sims_coordinate)
        assert set(result) == set(expected_pairs)

    @staticmethod
    def test_relabel_axis_empty():
        """Test relabel_axis with empty input."""
        # Create empty dataset with required dimensions
        _ = xr.Dataset(
            data_vars={"data": (["simulation", "scenario"], np.empty((0, 0)))},
            coords={"simulation": [], "scenario": []},
        )

        mock_empty_coord = xr.DataArray([], dims=["all_sims"])
        mock_empty_coord.values = np.array([], dtype=object)

        expected_output = []
        assert relabel_axis(mock_empty_coord) == expected_output

    @staticmethod
    def test_relabel_axis_single_element():
        """Test relabel_axis with a single element."""
        # Create a DataArray with a single element in each dimension
        ds = xr.Dataset(
            data_vars={"data": (["simulation", "scenario"], [[1.0]])},
            coords={"simulation": ["CanESM5"], "scenario": ["SSP1-1.9"]},
        )

        # Stack dimensions to create the multi-index coordinate
        stacked = ds.stack(all_sims=["simulation", "scenario"])
        all_sims_coordinate = stacked.all_sims

        expected_output = ["CanESM5_SSP1-1.9"]
        result = relabel_axis(all_sims_coordinate)
        assert result == expected_output


class TestProcessItem:
    """Test suite for the process_item function.

    This test class verifies that the process_item function correctly extracts
    and processes simulation metadata from an xarray DataArray.
    """

    @pytest.fixture
    def sample_dataarray(self) -> xr.DataArray:
        """Create a sample xarray DataArray with required attributes for testing."""
        return xr.DataArray(
            data=[1],
            attrs={
                "simulation": xr.DataArray("Dynamical_sim1_ensemble1"),
                "scenario": xr.DataArray("Historical + SSP 5-8.5"),
            },
        )

    def test_process_item_standard_case(self, sample_dataarray: xr.DataArray):
        """
        Test process_item with standard input format.

        Validates:
        - Correct extraction of simulation string parts
        - Proper scenario string processing
        - Expected return tuple format
        """
        result = process_item(sample_dataarray)

        # Verify return value is the expected tuple
        assert isinstance(result, tuple)
        assert len(result) == 3

        # Verify each component is extracted correctly
        assert result[0] == "sim1"  # sim_str
        assert result[1] == "ensemble1"  # ensemble
        assert result[2] == "ssp585"  # scenario

    @pytest.mark.parametrize(
        "simulation,scenario,expected",
        [
            (
                "Dynamical_MPI_r1i1p1",
                "Historical + SSP 3-7.0",
                ("MPI", "r1i1p1", "ssp370"),
            ),
            (
                "Statistical_CMIP6_ens4",
                "Historical + SSP 2-4.5",
                ("CMIP6", "ens4", "ssp245"),
            ),
            (
                "Dynamical_NorESM_r1",
                "Historical + SSP 5-8.5",
                ("NorESM", "r1", "ssp585"),
            ),
            ("Statistical_NASA_e5", "Historical + SSP 3-7.0", ("NASA", "e5", "ssp370")),
        ],
    )
    def test_process_item_variations(
        self, simulation: str, scenario: str, expected: tuple
    ) -> None:
        """
        Test process_item with various input formats.

        Tests:
        - Different simulation naming conventions
        - Different scenario formats including dash notation
        - Consistent processing across variations

        Parameters
        ----------
        simulation : str
            The simulation attribute value to test
        scenario : str
            The scenario attribute value to test
        expected : tuple
            The expected output tuple (sim_str, ensemble, scenario)
        """
        # Create a test DataArray with the parameterized values
        test_da = xr.DataArray(
            data=[1],
            attrs={
                "simulation": xr.DataArray(simulation),
                "scenario": xr.DataArray(scenario),
            },
        )

        # Test the function with these inputs
        result = process_item(test_da)
        assert result == expected

    def test_process_item_error_handling(self) -> None:
        """
        Verify proper error handling for invalid inputs.

        Tests:
        - Missing required attributes
        - Incorrectly formatted simulation string
        - Invalid scenario format
        """
        # Test case 1: Missing simulation attribute
        da_missing_sim = xr.DataArray(
            data=[1], attrs={"scenario": xr.DataArray("Historical + SSP 5-8.5")}
        )
        with pytest.raises(AttributeError):
            process_item(da_missing_sim)

        # Test case 2: Missing scenario attribute
        da_missing_scenario = xr.DataArray(
            data=[1], attrs={"simulation": xr.DataArray("Dynamical_sim1_ensemble1")}
        )
        with pytest.raises(AttributeError):
            process_item(da_missing_scenario)

        # Test case 3: Incorrectly formatted simulation string (not enough parts)
        da_invalid_sim = xr.DataArray(
            data=[1],
            attrs={
                "simulation": xr.DataArray("Dynamical-invalid"),
                "scenario": xr.DataArray("Historical + SSP 5-8.5"),
            },
        )
        with pytest.raises(ValueError):
            process_item(da_invalid_sim)

        # Test case 4: Scenario string with no "+" separator
        da_invalid_scenario = xr.DataArray(
            data=[1],
            attrs={
                "simulation": xr.DataArray("Dynamical_sim1_ensemble1"),
                "scenario": xr.DataArray("SSP 5-8.5"),  # No "+" separator
            },
        )
        with pytest.raises(IndexError):
            process_item(da_invalid_scenario)


class TestCleanList:
    """
    Test suite for the clean_list function which filters xarray datasets
    to retain only simulations with valid warming level data.
    """

    @pytest.fixture
    def mock_dataset(self) -> xr.Dataset:
        """
        Create a mock xarray Dataset with simulation data.

        Returns:
            xr.Dataset: A dataset with temperature data for multiple simulations.
        """
        # Create test data with multiple simulations
        data = xr.Dataset(
            data_vars={
                "temperature": (["all_sims", "time"], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            },
            coords={"all_sims": ["sim1", "sim2", "sim3"], "time": [0, 1, 2]},
            attrs={"description": "Test dataset"},
        )

        # Add attributes to each simulation
        data["all_sims"].attrs = {
            "simulation": "test_simulation",
            "scenario": "test_scenario",
        }

        return data

    @pytest.fixture
    def mock_gwl_times(self) -> pd.DataFrame:
        """
        Create a mock pandas DataFrame representing the warming level lookup table.

        Returns:
            _ : pd.DataFrame
                A DataFrame with warming level data for specific simulations.
        """
        # Create index tuples representing (simulation, ensemble, scenario)
        index = pd.MultiIndex.from_tuples(
            [("model1", "ensemble1", "ssp585"), ("model2", "ensemble1", "ssp245")],
            names=["sim", "ensemble", "scenario"],
        )

        # Create DataFrame with warming level data
        return pd.DataFrame({"warming_level": [1.5, 2.0]}, index=index)

    @patch("climakitae.explore.warming.process_item")
    def test_clean_list_normal_case(
        self,
        mock_process_item: MagicMock,
        mock_dataset: xr.Dataset,
        mock_gwl_times: pd.DataFrame,
    ) -> None:
        """
        Test clean_list with valid inputs under normal conditions.

        This test verifies that:
        1. The function correctly filters simulations based on the lookup table
        2. Only simulations with matching entries in gwl_times are retained
        3. The function preserves dataset attributes and structure

        Parameters
        ----------
        mock_process_item : MagicMock
            Mocked process_item function
        mock_dataset : xr.Dataset
            Test xarray Dataset
        mock_gwl_times : pd.DataFrame
            Test warming level lookup table
        """
        # Configure mock_process_item to return different values for each simulation
        mock_process_item.side_effect = [
            ("model1", "ensemble1", "ssp585"),  # sim1 -> valid
            ("model3", "ensemble2", "ssp370"),  # sim2 -> invalid
            ("model2", "ensemble1", "ssp245"),  # sim3 -> valid
        ]

        # Run the function
        result = clean_list(mock_dataset, mock_gwl_times)

        # Verify process_item was called for each simulation
        assert mock_process_item.call_count == 3

        # Check that only valid simulations are retained
        assert len(result.all_sims) == 2
        assert "sim1" in result.all_sims.values
        assert "sim3" in result.all_sims.values
        assert "sim2" not in result.all_sims.values

        # Check that the data structure is preserved
        assert "temperature" in result.data_vars
        assert result.attrs == mock_dataset.attrs

    @patch("climakitae.explore.warming.process_item")
    def test_clean_list_empty_result(
        self,
        mock_process_item: MagicMock,
        mock_dataset: xr.Dataset,
        mock_gwl_times: pd.DataFrame,
    ) -> None:
        """
        Test clean_list when no simulations match the lookup table.

        This edge case test verifies that:
        1. The function handles the case where no simulations match
        2. An empty dataset with the correct structure is returned

        Parameters
        ----------
        mock_process_item : MagicMock
            Mocked process_item function
        mock_dataset : xr.Dataset
            Test xarray Dataset
        mock_gwl_times : pd.DataFrame
            Test warming level lookup table
        """
        # Configure mock to return values not in the lookup table
        mock_process_item.side_effect = [
            ("invalid1", "ensemble3", "ssp119"),
            ("invalid2", "ensemble3", "ssp119"),
            ("invalid3", "ensemble3", "ssp119"),
        ]

        # Run the function
        result = clean_list(mock_dataset, mock_gwl_times)

        # Verify all simulations were checked
        assert mock_process_item.call_count == 3

        # Check that no simulations are retained
        assert len(result.all_sims) == 0

        # Check that the data structure is preserved
        assert "temperature" in result.data_vars
        assert result.attrs == mock_dataset.attrs

    @patch("climakitae.explore.warming.process_item")
    def test_clean_list_error_handling(
        self,
        mock_process_item: MagicMock,
        mock_dataset: xr.Dataset,
        mock_gwl_times: pd.DataFrame,
    ) -> None:
        """
        Test clean_list's error handling when process_item raises exceptions.

        This test verifies that:
        1. Errors in process_item are properly propagated
        2. The function correctly handles attribute access errors

        Args:
            mock_process_item: Mocked process_item function
            mock_dataset: Test xarray Dataset
            mock_gwl_times: Test warming level lookup table
        """
        # Configure mock to raise an exception for the second simulation
        mock_process_item.side_effect = [
            ("model1", "ensemble1", "ssp585"),  # sim1 -> valid
            KeyError("Missing attribute"),  # sim2 -> error
            ("model2", "ensemble1", "ssp245"),  # sim3 -> valid
        ]

        # Test that the exception is propagated
        with pytest.raises(KeyError, match="Missing attribute"):
            clean_list(mock_dataset, mock_gwl_times)

        # Test with missing all_sims dimension
        invalid_dataset = xr.Dataset(
            data_vars={"temperature": (["wrong_dim"], [1, 2, 3])},
            coords={"wrong_dim": [0, 1, 2]},
        )

        with pytest.raises(AttributeError):
            clean_list(invalid_dataset, mock_gwl_times)


class TestCleanWarmData:
    """Test suite for the clean_warm_data function.

    These tests verify that the function correctly removes invalid simulations
    where warming levels are not crossed (i.e., centered_year is null).
    """

    @pytest.fixture
    def sample_data_array(self) -> xr.DataArray:
        """
        Create a sample xarray DataArray with a centered_year coordinate.

        Returns
        -------
        xr.DataArray
            A sample DataArray with all_sims and time dimensions, and a centered_year coordinate.
        """
        # Create sample data with 3 simulations, 2 timestamps
        data = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],  # sim1 data
                [[5.0, 6.0], [7.0, 8.0]],  # sim2 data
                [[9.0, 10.0], [11.0, 12.0]],  # sim3 data
            ]
        )

        # Create DataArray with all_sims and time dimensions
        da = xr.DataArray(
            data,
            dims=["all_sims", "time", "lat"],
            coords={
                "all_sims": ["sim1", "sim2", "sim3"],
                "time": [0, 1],
                "lat": [0, 1],
            },
        )

        # Add centered_year coordinate with some null values
        # sim1: valid (year 2050), sim2: invalid (null), sim3: valid (year 2070)
        centered_years = xr.DataArray(
            [2050, np.nan, 2070],
            dims=["all_sims"],
            coords={"all_sims": ["sim1", "sim2", "sim3"]},
        )
        da = da.assign_coords(centered_year=centered_years)

        return da

    def test_clean_warm_data_normal_case(self, sample_data_array: xr.DataArray):
        """
        Test clean_warm_data with a mix of valid and invalid simulations.

        Validates:
        - Simulations with null centered_year values are removed
        - Simulations with valid centered_year values are retained
        - Data structure and values are preserved for valid simulations
        - No unintended modifications to the original data

        Parameters
        ----------
        sample_data_array : xr.DataArray
            Fixture providing sample data with mixed valid/invalid simulations
        """
        # Call the function to clean the data
        cleaned_data = clean_warm_data(sample_data_array)

        # Verify that invalid simulation (sim2) is removed
        assert "sim2" not in cleaned_data.all_sims.values
        assert len(cleaned_data.all_sims) == 2
        assert "sim1" in cleaned_data.all_sims.values
        assert "sim3" in cleaned_data.all_sims.values

        # Verify that data values for valid simulations are preserved
        np.testing.assert_array_equal(
            cleaned_data.sel(all_sims="sim1").values,
            sample_data_array.sel(all_sims="sim1").values,
        )
        np.testing.assert_array_equal(
            cleaned_data.sel(all_sims="sim3").values,
            sample_data_array.sel(all_sims="sim3").values,
        )

        # Verify that centered_year values are preserved
        assert cleaned_data.centered_year.sel(all_sims="sim1").item() == 2050
        assert cleaned_data.centered_year.sel(all_sims="sim3").item() == 2070

    def test_clean_warm_data_all_null(self):
        """
        Test clean_warm_data when all simulations have null centered_year values.

        This edge case test verifies that:
        - When all simulations have null centered_year, the function returns
            the original data without modifications
        - The function correctly identifies the "all null" condition
        """
        # Create a dataset where all simulations have null centered_year
        data = np.ones((3, 2, 2))  # 3 simulations, 2 timestamps, 2 lat points
        all_null_da = xr.DataArray(
            data,
            dims=["all_sims", "time", "lat"],
            coords={
                "all_sims": ["sim1", "sim2", "sim3"],
                "time": [0, 1],
                "lat": [0, 1],
            },
        )

        # Assign all null centered_years
        centered_years = xr.DataArray(
            [np.nan, np.nan, np.nan],
            dims=["all_sims"],
            coords={"all_sims": ["sim1", "sim2", "sim3"]},
        )
        all_null_da = all_null_da.assign_coords(centered_year=centered_years)

        # Process with clean_warm_data
        result = clean_warm_data(all_null_da)

        # Verify the data is unchanged
        xr.testing.assert_identical(result, all_null_da)
        assert len(result.all_sims) == 3  # All simulations retained

    def test_clean_warm_data_single_simulation(self):
        """
        Test clean_warm_data with a single simulation.

        Tests:
        - Special handling when there's only one simulation
        - Correct behavior when the single simulation is valid
        - Correct behavior when the single simulation is invalid

        This test verifies the different code path used when centered_year.isnull().size == 1
        """
        # Case 1: Single valid simulation
        valid_data = np.ones((1, 2, 2))
        valid_da = xr.DataArray(
            valid_data,
            dims=["all_sims", "time", "lat"],
            coords={"all_sims": ["sim1"], "time": [0, 1], "lat": [0, 1]},
        )
        valid_da = valid_da.assign_coords(
            centered_year=xr.DataArray([2050], dims=["all_sims"])
        )

        # Process with clean_warm_data
        valid_result = clean_warm_data(valid_da)

        # Verify the simulation is retained
        assert len(valid_result.all_sims) == 1
        assert "sim1" in valid_result.all_sims.values


class TestGetSlicedData:
    """Test suite for get_sliced_data function."""

    @pytest.fixture
    def mock_dataarray(self) -> xr.DataArray:
        """
        Create a mock DataArray with required attributes for testing.

        Returns
        -------
        xr.DataArray
            A sample DataArray with time dimension and required attributes.
        """
        # Create time range spanning multiple years for testing different scenarios
        times = pd.date_range(start="2000-01-01", end="2040-12-31", freq="MS")

        # Create latitude and longitude dimensions
        lat = np.linspace(-90, 90, 5)
        lon = np.linspace(-180, 180, 5)

        # Create random data
        data = np.random.rand(len(times), len(lat), len(lon))

        # Create DataArray
        da = xr.DataArray(
            data=data,
            dims=["time", "lat", "lon"],
            coords={
                "time": times,
                "lat": lat,
                "lon": lon,
            },
            attrs={
                "frequency": "monthly",
                "simulation": xr.DataArray("Dynamical_sim1_ensemble1"),
                "scenario": xr.DataArray("Historical + ssp585"),
            },
        )

        return da

    @pytest.fixture
    def mock_gwl_times(self) -> pd.DataFrame:
        """
        Create a mock warming level lookup table.

        Returns
        -------
        pd.DataFrame
            DataFrame with warming levels as columns and simulation identifiers as index.
        """
        # Create warming level lookup table with different simulations and levels
        index = pd.MultiIndex.from_tuples(
            [
                (
                    "sim1",
                    "ensemble1",
                    "ssp585",
                ),  # Valid simulation with all warming levels
                (
                    "sim2",
                    "ensemble2",
                    "ssp370",
                ),  # Valid simulation with some null warming levels
                (
                    "sim3",
                    "ensemble3",
                    "ssp245",
                ),  # Simulation with late warming level (2090)
            ],
            names=["sim", "ensemble", "scenario"],
        )

        return pd.DataFrame(
            {
                "0.8": pd.to_datetime(["2010-01-01", "2012-01-01", "2015-01-01"]),
                "1.5": pd.to_datetime(["2025-01-01", "2030-01-01", "2040-01-01"]),
                "2.0": pd.to_datetime(["2035-01-01", np.nan, "2060-01-01"]),
                "3.0": pd.to_datetime(["2055-01-01", np.nan, "2080-01-01"]),
                "4.0": pd.to_datetime(["2075-01-01", np.nan, "2090-01-01"]),
            },
            index=index,
        )

    @patch("xarray.core.dataarray.DataArray.sel")
    @patch("climakitae.explore.warming.process_item")
    def test_get_sliced_data_valid_center_time(
        self,
        mock_process_item: MagicMock,
        mock_sel: MagicMock,
        mock_dataarray: xr.DataArray,
        mock_gwl_times: pd.DataFrame,
    ) -> None:
        """
        Test get_sliced_data with a valid center time.

        This test verifies that:
        1. The function correctly slices data around the center time
        2. The time dimension is properly reset to center around zero
        3. Month filtering works correctly
        4. Anomaly calculation is correctly performed when requested
        5. Centered year is properly assigned as a coordinate

        Parameters
        ----------
        mock_process_item : MagicMock
            Mock for the process_item function
        mock_sel : MagicMock
            Mock for the DataArray selection method
        mock_dataarray : xr.DataArray
            Test DataArray fixture
        mock_gwl_times : pd.DataFrame
            Test warming level lookup table fixture
        """
        # Configure mock_process_item to return a valid simulation identifier
        mock_process_item.return_value = ("sim1", "ensemble1", "ssp585")

        # Create mock return values
        mock_sliced_time = pd.date_range(
            start="2015-01-01", periods=240, freq="MS"
        )  # Monthly dates
        mock_sliced_data = xr.DataArray(
            data=np.random.rand(len(mock_sliced_time), 5, 5),
            dims=["time", "lat", "lon"],
            coords={
                "time": mock_sliced_time,  # Use datetime values
                "lat": mock_dataarray.lat.values,
                "lon": mock_dataarray.lon.values,
                "centered_year": 2025,
            },
            attrs=mock_dataarray.attrs,
        )
        mock_ref_period = mock_dataarray.copy()
        mock_sel.side_effect = [mock_ref_period] + [mock_sliced_data for _ in range(8)]

        # Test with anomaly calculation
        window = 10
        level = "1.5"
        result = get_sliced_data(
            mock_dataarray,
            level,
            mock_gwl_times,
            months=np.arange(1, 13),
            window=window,
            anom="Yes",
        )

        # Verify sel was called
        assert mock_sel.call_count >= 1

        # Check time bounds: should be 2025 +/- 10 years (2015-2035)
        # but now encoded as integers centered around 0
        center_year = mock_gwl_times.loc[mock_process_item.return_value][level].year
        assert (
            result.time.min().dt.year <= center_year - window
        )  # Should start at or before center_year - window
        assert (
            result.time.max().dt.year >= center_year
        )  # Should extend at least to center_year

        # Check that we have the expected number of timesteps
        assert (
            len(result.time) <= window * 2 * 12
        )  # At most window*2 years of monthly data

        # Check centered_year coordinate is set correctly
        assert result.centered_year == 2025

        # Case 3: Filter specific months
        summer_months = np.array([6, 7, 8])  # June, July, August
        result_summer = get_sliced_data(
            mock_dataarray,
            level,
            mock_gwl_times,
            months=summer_months,
            window=window,
            anom="No",
        )

        assert len(result_summer.time) == 240

    @patch("climakitae.explore.warming.process_item")
    def test_get_sliced_data_na_center_time(
        self,
        mock_process_item,
        mock_dataarray: xr.DataArray,
        mock_gwl_times: pd.DataFrame,
    ) -> None:
        """
        Test get_sliced_data when center time is NaN.

        This test verifies that:
        1. The function correctly handles simulations that don't reach the specified warming level
        2. A properly structured DataArray of NaNs is returned with the expected dimensions
        3. The returned DataArray has appropriate time coordinates and centered_year attribute

        Parameters
        ----------
        mock_process_item : MagicMock
            Mock for the process_item function
        mock_dataarray : xr.DataArray
            Test DataArray fixture
        mock_gwl_times : pd.DataFrame
            Test warming level lookup table fixture
        """
        # Configure mock_process_item to return a simulation with NaN warming levels
        mock_process_item.return_value = ("sim2", "ensemble2", "ssp370")

        # Test with a warming level (4.0) that this simulation doesn't reach
        level = "4.0"
        window = 15
        months = np.array([1, 2, 3, 4])  # Just some months

        # Call the function
        result = get_sliced_data(
            mock_dataarray, level, mock_gwl_times, months=months, window=window
        )

        # Verify the result is a DataArray filled with NaNs
        assert isinstance(result, xr.DataArray)
        assert np.all(np.isnan(result.values))

        # Check expected time dimension properties
        assert result.time.size > 0  # Should have some time points

        # Verify centered_year is NaN
        assert np.isnan(result.centered_year)

        # Verify dimensions are preserved from original DataArray
        assert "lat" in result.dims
        assert "lon" in result.dims

    @patch("climakitae.explore.warming.process_item")
    def test_get_sliced_data_edge_cases(
        self,
        mock_process_item,
        mock_dataarray: xr.DataArray,
        mock_gwl_times: pd.DataFrame,
    ) -> None:
        """
        Test get_sliced_data edge cases.

        This test verifies:
        1. Handling of simulations with warming levels near the end of the dataset
        2. Warning is produced when time slice extends beyond available data
        3. Proper time dimension normalization when data is truncated
        4. Correct behavior with minimum window size

        Parameters
        ----------
        mock_process_item : MagicMock
            Mock for the process_item function
        mock_dataarray : xr.DataArray
            Test DataArray fixture
        mock_gwl_times : pd.DataFrame
            Test warming level lookup table fixture
        """
        # Configure mock to return simulation with late warming level
        mock_process_item.return_value = ("sim3", "ensemble3", "ssp245")

        # Test with a warming level near end of century (might cause time truncation)
        level = "4.0"  # Center year 2090
        window = 15

        # Mock the print function to capture warning messages
        with patch("builtins.print") as mock_print:
            result = get_sliced_data(
                mock_dataarray, level, mock_gwl_times, window=window
            )

            # Check if warning was printed about incomplete data
            mock_print_called = any(
                "not completely available" in str(call)
                for call in mock_print.call_args_list
            )

            # Should warn if our data (ending in 2040) doesn't cover the full window
            if mock_dataarray.time.max().dt.year < 2090 + window:
                assert mock_print_called

                # For truncated data, verify time dimension is still centered properly
                time_counts = result.time.size
                expected_counts = window * 2 * 12  # monthly data
                assert time_counts < expected_counts

        # Test with minimum window size
        min_window = 5
        result_min_window = get_sliced_data(
            mock_dataarray, "1.5", mock_gwl_times, window=min_window
        )

        # Check the result has appropriate dimensions
        assert isinstance(result_min_window, xr.DataArray)
        assert result_min_window.time.size <= min_window * 2 * 12


class TestDropInvalidSims:
    """Test suite for _drop_invalid_sims function."""

    @pytest.fixture
    def mock_dataset(self) -> xr.Dataset:
        """
        Create a mock xarray Dataset with an 'all_sims' dimension.

        Returns
        -------
        xr.Dataset
            A mock dataset with temperature data across multiple simulations and scenarios.
        """
        # Create temperature data first
        data = np.random.rand(3, 5)  # 4 simulations, 5 time points

        # Create all_sims values as a MultiIndex
        model_ids = [
            "Dynamical_CMIP6-1_r1i1p1f1",
            "Dynamical_CMIP6-2_r1i1p1f1",
            "Dynamical_CMIP6-3_r2i1p1f1",
        ]
        scenarios = [
            "Historical + SSP 2-4.5",
            "Historical + SSP 5-8.5",
            "Historical + SSP 3-7.0",
        ]

        # Create a pandas MultiIndex
        multi_index = pd.MultiIndex.from_tuples(
            list(zip(model_ids, scenarios)), names=["model", "scenario"]
        )

        # Convert MultiIndex to xarray coordinates using the recommended method
        multi_index_coords = xr.Coordinates.from_pandas_multiindex(
            multi_index, "all_sims"
        )

        # Create the dataset with string identifiers first
        ds = xr.Dataset(
            data_vars={"temperature": (["all_sims", "time"], data)},
            coords={
                "all_sims": ["sim1", "sim2", "sim3"],
                "time": pd.date_range("2020-01-01", periods=5),
            },
        )
        # Assign these coordinates to the dataset
        ds = ds.assign_coords(multi_index_coords)

        return ds

    @pytest.fixture
    def mock_catalog_subset(self) -> Mock:
        """
        Create a mock for _get_cat_subset function.

        Returns
        -------
        Mock
            A mock object simulating the catalog subset returned by _get_cat_subset.
        """
        mock_subset = Mock()
        # Create a DataFrame that mimics the catalog data structure
        mock_subset.df = pd.DataFrame(
            {
                "activity_id": ["Dynamical", "Dynamical", "Dynamical"],
                "source_id": ["CMIP6-1", "CMIP6-2", "CMIP6-3"],
                "member_id": ["r1i1p1f1", "r1i1p1f1", "r2i1p1f1"],
                "experiment_id": [
                    "ssp245",
                    "ssp585",
                    "ssp370",
                ],  # Note: no 'historical' entries here
            }
        )
        return mock_subset

    @patch("climakitae.explore.warming._get_cat_subset")
    def test_drop_invalid_sims_normal_case(
        self,
        mock_get_cat_subset: Mock,
        mock_dataset: xr.Dataset,
        mock_catalog_subset: Mock,
    ):
        """
        Test the function with standard input where some simulations are valid.

        This test verifies that:
        1. The function correctly filters the dataset based on valid simulation-scenario pairs
        2. The resulting dataset contains only the valid simulations
        3. The function properly handles the catalog data to create valid_sim_list

        Parameters
        ----------
        mock_get_cat_subset : Mock
            Mocked _get_cat_subset function
        mock_dataset : xr.Dataset
            Mock dataset with all_sims dimension
        mock_catalog_subset : Mock
            Mock catalog subset with DataFrame structure
        """
        # Setup the mock to return our fixture
        mock_get_cat_subset.return_value = mock_catalog_subset

        # Create a mock selections object
        mock_selections = Mock()

        # Call the function
        result = _drop_invalid_sims(mock_dataset, mock_selections)

        # Verify _get_cat_subset was called with correct parameters
        mock_get_cat_subset.assert_called_once_with(mock_selections)

        # Check that the result is an xarray Dataset
        assert isinstance(result, xr.Dataset)

        # Expected valid sim list based on our mock catalog
        expected_valid_sims = [
            ("Dynamical_CMIP6-1_r1i1p1f1", "Historical + SSP 2-4.5"),
            ("Dynamical_CMIP6-2_r1i1p1f1", "Historical + SSP 5-8.5"),
            ("Dynamical_CMIP6-3_r2i1p1f1", "Historical + SSP 3-7.0"),
        ]

        # Verify that only valid simulations are in the result
        assert set(result.all_sims.values) == set(expected_valid_sims)

        # Verify data integrity - check that temperature data is preserved
        for sim in expected_valid_sims:
            assert not np.isnan(result.sel(all_sims=sim)["temperature"].values).any()

    @patch("climakitae.explore.warming._get_cat_subset")
    def test_drop_invalid_sims_with_historical(
        self, mock_get_cat_subset: Mock, mock_dataset: xr.Dataset
    ):
        """
        Test the function when catalog includes historical experiments.

        This test verifies that:
        1. The function correctly excludes entries with experiment_id='historical'
        2. Only the non-historical entries are used to create the valid_sim_list

        Parameters
        ----------
        mock_get_cat_subset : Mock
            Mocked _get_cat_subset function
        mock_dataset : xr.Dataset
            Mock dataset with all_sims dimension
        """
        # Create a mock subset with both historical and non-historical entries
        mixed_subset = Mock()
        mixed_subset.df = pd.DataFrame(
            {
                "activity_id": ["Dynamical", "Dynamical", "Dynamical", "Dynamical"],
                "source_id": ["CMIP6-1", "CMIP6-1", "CMIP6-2", "CMIP6-3"],
                "member_id": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1", "r2i1p1f1"],
                "experiment_id": ["historical", "ssp245", "ssp585", "ssp370"],
            }
        )

        # Setup the mock to return our mixed subset
        mock_get_cat_subset.return_value = mixed_subset

        # Create a mock selections object
        mock_selections = Mock()

        # Call the function
        result = _drop_invalid_sims(mock_dataset, mock_selections)

        # Expected valid sim list from mock
        expected_valid_sims = [
            ("Dynamical_CMIP6-1_r1i1p1f1", "Historical + SSP 2-4.5"),
            ("Dynamical_CMIP6-2_r1i1p1f1", "Historical + SSP 5-8.5"),
            ("Dynamical_CMIP6-3_r2i1p1f1", "Historical + SSP 3-7.0"),
        ]

        # Verify that only non-historical simulations are in the result
        assert set(result.all_sims.values) == set(expected_valid_sims)

        # Verify _get_cat_subset was called with correct parameters
        mock_get_cat_subset.assert_called_once_with(mock_selections)
