"""
Unit tests for the warming module in climakitae.explore.warming.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.paths import gwl_1850_1900_file
from climakitae.explore.warming import (
    WarmingLevels,
    _check_available_warming_levels,
    clean_list,
    clean_warm_data,
    process_item,
    relabel_axis,
)
from climakitae.util.utils import read_csv_file
from climakitae.util.warming_levels import _calculate_warming_level, _get_sliced_data

# Load warming level times from CSV
gwl_times = read_csv_file(gwl_1850_1900_file, index_col=[0, 1, 2])


def test_missing_all_sims_attribute(
    test_dataarray_wl_20_summer_season_loca_3km_daily_temp,
):
    """Ensure AttributeError is raised when 'all_sims' is missing in DataArray dimensions."""
    da_wrong = test_dataarray_wl_20_summer_season_loca_3km_daily_temp.rename(
        {"simulation": "totally_wrong"}
    )
    with pytest.raises(AttributeError):
        _calculate_warming_level(da_wrong, gwl_times, 2, range(1, 13), 15)


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
                "time": pd.date_range("2000-01-01", periods=10, freq="Y"),
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
                        "time": pd.date_range("2000-01-01", periods=10, freq="Y"),
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
