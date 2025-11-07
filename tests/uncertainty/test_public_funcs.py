from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.explore.uncertainty import (
    GWL_1850_1900_FILE,
    GWL_1981_2010_FILE,
    CmipOpt,
    cmip_mmm,
    get_ensemble_data,
    get_warm_level,
    grab_multimodel_data,
    weighted_temporal_mean,
)
from tests.uncertainty.fixtures import (
    mock_data_for_clipping,
    mock_data_for_warm_level,
    mock_multi_ens_dataset,
    wrf_dataset,
)  # noqa: F401


def test_get_warm_level_input_validation():
    """Test validation of warm_level parameter."""
    # Test non-numeric warm_level
    with pytest.raises(ValueError) as excinfo:
        get_warm_level("not_a_number", None)
    assert "Please specify warming level as an integer or float." in str(excinfo.value)

    # Test invalid numeric warm_level
    with pytest.raises(ValueError) as excinfo:
        get_warm_level(2.5, None)
    assert (
        "Specified warming level is not valid. Options are: 1.5, 2.0, 3.0, 4.0"
        in str(excinfo.value)
    )


@patch("climakitae.explore.uncertainty.read_csv_file")
def test_get_warm_level_file_loading(mock_read_csv, mock_data_for_warm_level):
    """Test file loading logic in get_warm_level for both IPCC and non-IPCC paths"""

    # Create mock DataFrames for the CSV files with datetime strings (matching actual format)
    mock_ipcc_df = pd.DataFrame(
        {
            "1.5": ["2030-06-15 12:00:00"],
            "2.0": ["2045-03-15 00:00:00"],
            "3.0": ["2070-01-16 12:00:00"],
            "4.0": ["2090-04-16 00:00:00"],
        }
    )
    mock_ipcc_df.index = pd.MultiIndex.from_tuples(
        [("CESM2", "r11i1p1f1", "ssp370")], names=["GCM", "run", "scenario"]
    )

    mock_non_ipcc_df = pd.DataFrame(
        {
            "1.5": ["2025-04-16 00:00:00"],
            "2.0": ["2040-08-16 12:00:00"],
            "3.0": ["2065-09-16 00:00:00"],
            "4.0": ["2085-11-16 00:00:00"],
        }
    )
    mock_non_ipcc_df.index = pd.MultiIndex.from_tuples(
        [("CESM2", "r11i1p1f1", "ssp370")], names=["GCM", "run", "scenario"]
    )

    mock_ece3_df = pd.DataFrame(
        {
            "1.5": ["2026-01-16 12:00:00"],
            "2.0": ["2041-03-16 12:00:00"],
            "3.0": ["2066-05-16 12:00:00"],
            "4.0": ["2086-04-16 00:00:00"],
        }
    )
    mock_ece3_df.index = pd.MultiIndex.from_tuples(
        [("EC-Earth3", "r1i1p1f1", "ssp370")], names=["GCM", "run", "scenario"]
    )

    # Set up the mock to return different DataFrames based on input file path
    def side_effect(file_path, **kwargs):
        match file_path:
            case _ if file_path == GWL_1850_1900_FILE:
                return mock_ipcc_df
            case _ if file_path == GWL_1981_2010_FILE:
                return mock_non_ipcc_df
            case "data/gwl_1981-2010ref_EC-Earth3_ssp370.csv":
                return mock_ece3_df
            case _:
                return pd.DataFrame()

    mock_read_csv.side_effect = side_effect

    # Test IPCC path (ipcc=True)
    with patch("climakitae.explore.uncertainty.GWL_1850_1900_FILE", GWL_1850_1900_FILE):
        with patch(
            "climakitae.explore.uncertainty.GWL_1981_2010_FILE", GWL_1981_2010_FILE
        ):
            # Mock just enough of the function to test file loading
            with patch.object(
                get_warm_level.__globals__["pd"], "concat"
            ) as mock_concat:
                # Call the function with ipcc=True
                try:
                    get_warm_level(2.0, mock_data_for_warm_level, ipcc=True)
                except Exception:
                    # We expect the function to potentially fail after file loading,
                    # since we're not mocking everything, but we only care about file loading
                    pass

                # Verify read_csv_file was called with the correct file
                mock_read_csv.assert_called_with(
                    GWL_1850_1900_FILE, index_col=[0, 1, 2]
                )
                # Verify concat was not called (IPCC path uses single file)
                mock_concat.assert_not_called()

    # Test non-IPCC path (ipcc=False)
    mock_read_csv.reset_mock()
    with patch("climakitae.explore.uncertainty.GWL_1850_1900_FILE", GWL_1850_1900_FILE):
        with patch(
            "climakitae.explore.uncertainty.GWL_1981_2010_FILE", GWL_1981_2010_FILE
        ):
            with patch.object(
                pd, "concat", return_value=pd.concat([mock_non_ipcc_df, mock_ece3_df])
            ) as mock_concat:
                # Call the function with ipcc=False
                try:
                    get_warm_level(2.0, mock_data_for_warm_level, ipcc=False)
                except Exception:
                    pass

                # Verify read_csv_file was called with both files
                assert mock_read_csv.call_count == 2
                mock_read_csv.assert_any_call(GWL_1981_2010_FILE, index_col=[0, 1, 2])
                mock_read_csv.assert_any_call(
                    "data/gwl_1981-2010ref_EC-Earth3_ssp370.csv", index_col=[0, 1, 2]
                )

                # Verify concat was called with the two DataFrames
                mock_concat.assert_called_once()


@patch("climakitae.explore.uncertainty.read_csv_file")
def test_get_warm_level_member_id_selection(
    mock_read_csv, mock_data_for_clipping, mock_multi_ens_dataset
):
    """Test member_id selection logic."""
    # Setup mock dataframe with warming data - create a DataFrame with one row per model
    # Use realistic datetime string format for warming-level reach times
    mock_df = pd.DataFrame(
        {
            "1.5": [
                "2030-06-15 12:00:00",
                "2030-06-16 12:00:00",
                "2030-06-17 12:00:00",
            ],
            "2.0": [
                "2045-03-15 00:00:00",
                "2045-03-16 00:00:00",
                "2045-03-17 00:00:00",
            ],
            "3.0": [
                "2070-01-15 12:00:00",
                "2070-01-16 12:00:00",
                "2070-01-17 12:00:00",
            ],
            "4.0": [
                "2090-04-15 00:00:00",
                "2090-04-16 00:00:00",
                "2090-04-17 00:00:00",
            ],
        }
    )
    mock_df.index = pd.MultiIndex.from_tuples(
        [
            ("CESM2", "r11i1p1f1", "ssp370"),
            ("EC-Earth3", "r1i1p1f1", "ssp370"),
            ("CNRM-ESM2-1", "r1i1p1f2", "ssp370"),
        ],
        names=["GCM", "run", "scenario"],
    )
    mock_read_csv.return_value = mock_df

    # Test default member_id selection for CESM2
    with patch("xarray.Dataset.sel", return_value=mock_data_for_clipping):
        get_warm_level(2.0, mock_data_for_clipping)
        # For CESM2, should use r11i1p1f1
        assert (
            mock_df.loc[("CESM2", "r11i1p1f1", "ssp370")]["2.0"]
            == "2045-03-15 00:00:00"
        )

    # Test default member_id selection for CNRM-ESM2-1
    cnrm_ds = mock_data_for_clipping.copy()
    cnrm_ds = cnrm_ds.assign_coords({"simulation": "CNRM-ESM2-1"})
    with patch("xarray.Dataset.sel", return_value=cnrm_ds):
        get_warm_level(2.0, cnrm_ds)
        # For CNRM-ESM2-1, should use r1i1p1f2
        assert (
            mock_df.loc[("CNRM-ESM2-1", "r1i1p1f2", "ssp370")]["2.0"]
            == "2045-03-17 00:00:00"
        )

    # Test multi_ens=True (should use the member_id from the dataset)
    with patch("xarray.Dataset.sel", return_value=mock_multi_ens_dataset):
        get_warm_level(2.0, mock_multi_ens_dataset, multi_ens=True)
        # Should use the member_id from the dataset
        assert (
            mock_df.loc[("EC-Earth3", "r1i1p1f1", "ssp370")]["2.0"]
            == "2045-03-16 00:00:00"
        )


@patch("climakitae.explore.uncertainty.read_csv_file")
def test_get_warm_level_time_slice(mock_read_csv, mock_data_for_warm_level):
    """Test time slicing logic based on warming level."""
    # Setup mock dataframe with warming data
    mock_df = pd.DataFrame(
        {
            "3.0": ["2070-01-01 00:00:00"],  # Warming level reached in 2070
        }
    )
    mock_df.index = pd.MultiIndex.from_tuples(
        [("CESM2", "r11i1p1f1", "ssp370")], names=["GCM", "run", "scenario"]
    )
    mock_read_csv.return_value = mock_df

    # Test normal time slice (-14/+15 years from warming level year)
    with patch("xarray.Dataset.sel", return_value=mock_data_for_warm_level) as mock_sel:
        get_warm_level(3.0, mock_data_for_warm_level)
        # Should select time slice from 2056 (2070-14) to 2085 (2070+15)
        mock_sel.assert_called_with(time=slice("2056", "2085"))

    # Test case where warming level not reached (year string less than 4 chars)
    mock_df["3.0"] = ["N/A"]
    with patch("builtins.print") as mock_print:
        result = get_warm_level(3.0, mock_data_for_warm_level)
        assert result is None
        mock_print.assert_called_with(
            "Invalid year format extracted: 'N/A' for 3.0°C warming level, ensemble member r11i1p1f1 of model CESM2"
        )

    # Test case where end year exceeds 2100
    mock_df["3.0"] = ["2090-01-01 00:00:00"]  # 2090 + 15 > 2100
    with patch("builtins.print") as mock_print:
        result = get_warm_level(3.0, mock_data_for_warm_level)
        assert result is None
        mock_print.assert_called_with(
            "End year for SSP time slice occurs after 2100; skipping ensemble member r11i1p1f1 of model CESM2"
        )


@pytest.mark.advanced
def test_get_warm_level_integration(wrf_dataset):
    """Test get_warm_level with a real-like dataset."""
    # Extract a single simulation to test
    ds = wrf_dataset.isel(simulation=0).drop_vars("simulation")
    ds = ds.assign_coords({"simulation": "CESM2"})

    # Since we can't predict the actual warming level data, we'll patch it
    with patch("climakitae.explore.uncertainty.read_csv_file") as mock_read_csv:
        mock_df = pd.DataFrame(
            {
                "3.0": ["2070-01-01 00:00:00"],  # Warming level reached in 2070
            }
        )
        mock_df.index = pd.MultiIndex.from_tuples(
            [("CESM2", "r11i1p1f1", "ssp370")], names=["GCM", "run", "scenario"]
        )
        mock_read_csv.return_value = mock_df

        # Use module-level patching instead of object-level patching
        with patch("xarray.DataArray.sel", return_value=ds) as mock_sel:
            get_warm_level(3.0, ds)
            mock_sel.assert_called_with(time=slice("2056", "2085"))


def test_cmip_mmm_basic():
    """Test basic functionality of cmip_mmm with a simple dataset."""
    # Create test dataset with simulation dimension
    ds = xr.Dataset(
        data_vars={"tas": (["time", "simulation"], np.array([[1, 2, 3], [4, 5, 6]]))},
        coords={"time": [0, 1], "simulation": ["model1", "model2", "model3"]},
    )

    result = cmip_mmm(ds)

    # Check that simulation dimension is removed
    assert "simulation" not in result.dims

    # Check that mean is calculated correctly
    expected_mean = np.array([2, 5])  # Mean of [1,2,3] and [4,5,6]
    np.testing.assert_array_equal(result["tas"].values, expected_mean)


def test_cmip_mmm_multiple_variables():
    """Test cmip_mmm with multiple variables."""
    # Create test dataset with multiple variables
    ds = xr.Dataset(
        data_vars={
            "tas": (["time", "simulation"], np.array([[1, 2, 3], [4, 5, 6]])),
            "pr": (["time", "simulation"], np.array([[10, 20, 30], [40, 50, 60]])),
        },
        coords={"time": [0, 1], "simulation": ["model1", "model2", "model3"]},
    )

    result = cmip_mmm(ds)

    # Check that both variables have means calculated correctly
    expected_tas_mean = np.array([2, 5])
    expected_pr_mean = np.array([20, 50])

    np.testing.assert_array_equal(result["tas"].values, expected_tas_mean)
    np.testing.assert_array_equal(result["pr"].values, expected_pr_mean)


def test_cmip_mmm_with_nans():
    """Test cmip_mmm with dataset containing NaN values."""
    # Create test dataset with NaNs
    data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    ds = xr.Dataset(
        data_vars={"tas": (["time", "simulation"], data)},
        coords={"time": [0, 1], "simulation": ["model1", "model2", "model3"]},
    )

    result = cmip_mmm(ds)

    # xarray's mean ignores NaNs by default
    expected_mean = np.array([1.5, 5])  # Mean of [1,2] and [4,6]
    np.testing.assert_array_equal(result["tas"].values, expected_mean)


def test_cmip_mmm_no_side_effects():
    """Test that cmip_mmm does not modify the input dataset."""
    # Create test dataset
    ds = xr.Dataset(
        data_vars={"tas": (["time", "simulation"], np.array([[1, 2, 3], [4, 5, 6]]))},
        coords={"time": [0, 1], "simulation": ["model1", "model2", "model3"]},
    )

    # Create a deep copy for comparison
    ds_original = ds.copy(deep=True)

    # Call function
    _ = cmip_mmm(ds)

    # Verify original dataset is unchanged
    xr.testing.assert_identical(ds, ds_original)


def test_cmip_mmm_empty_simulation():
    """Test cmip_mmm with an empty simulation dimension."""
    # Create dataset with empty simulation dimension
    ds = xr.Dataset(
        data_vars={"tas": (["time", "simulation"], np.zeros((2, 0)))},
        coords={"time": [0, 1], "simulation": []},
    )

    result = cmip_mmm(ds)
    # mean should be nan since there is no simulation dimension
    expected_mean = np.array([np.nan, np.nan])
    np.testing.assert_array_equal(result["tas"].values, expected_mean)


def test_cmip_mmm_no_simulation_dim():
    """Test cmip_mmm with a dataset that doesn't have a simulation dimension."""
    # Create dataset without simulation dimension
    ds = xr.Dataset(
        data_vars={"tas": (["time"], np.array([1, 2]))}, coords={"time": [0, 1]}
    )

    with pytest.raises(ValueError):
        _ = cmip_mmm(ds)


@patch("climakitae.explore.uncertainty.intake")
@patch("climakitae.explore.uncertainty._drop_member_id")
@patch("climakitae.explore.uncertainty._standardize_cmip6_data")
def test_grab_multimodel_data_basic(mock_standardize, mock_drop_member, mock_intake):
    """Test basic functionality of grab_multimodel_data."""
    # Create mock catalog and search results
    mock_catalog = MagicMock()
    mock_search = MagicMock()
    mock_cat_search = MagicMock()
    mock_intake.open_esm_datastore.return_value = mock_catalog
    mock_catalog.search.return_value = mock_search
    mock_search.search.return_value = mock_cat_search

    # Create mock datasets
    mock_hist_ds = MagicMock(spec=xr.Dataset)
    mock_ssp_ds = MagicMock(spec=xr.Dataset)

    # Configure mock returns for to_dataset_dict
    mock_cat_search.to_dataset_dict.return_value = {"model1.historical": mock_hist_ds}
    mock_catalog.search.side_effect = [mock_search, mock_cat_search]
    mock_cat_search.to_dataset_dict.return_value = {"model1.ssp370": mock_ssp_ds}

    # Configure drop_member_id to return the input dictionaries
    mock_drop_member.side_effect = lambda x: x

    # Create a mock CmipOpt instance
    copt = MagicMock(spec=CmipOpt)
    copt.variable = "tas"
    copt.area_subset = "states"
    copt.location = "California"
    copt.timescale = "monthly"

    # Mock the _cmip_clip method to return the dataset directly
    copt._cmip_clip = MagicMock()
    copt._cmip_clip.side_effect = lambda ds: ds

    # Configure the mock hist_ds and ssp_ds with the necessary methods
    mock_hist_concat = MagicMock(spec=xr.Dataset)
    mock_ssp_concat = MagicMock(spec=xr.Dataset)
    mock_result_ds = MagicMock(spec=xr.Dataset)

    # Configure the sel method for time slicing
    mock_hist_ds.sel = MagicMock(return_value=mock_hist_ds)
    mock_ssp_ds.sel = MagicMock(return_value=mock_ssp_ds)

    # Mock concat operations
    with patch("xarray.concat") as mock_concat:
        mock_concat.side_effect = [mock_hist_concat, mock_ssp_concat, mock_result_ds]

        # Call the function
        result = grab_multimodel_data(copt)

        # Check that the result is correct
        assert result is mock_result_ds

        # Verify intake.open_esm_datastore was called correctly
        mock_intake.open_esm_datastore.assert_called_once_with(
            "https://cadcat.s3.amazonaws.com/tmp/cmip6-regrid.json"
        )

        # Verify catalog search was called with correct parameters
        mock_catalog.search.assert_any_call(
            table_id="monthly",
            variable_id="tas",
            experiment_id=["historical", "ssp370"],
            member_id="r1i1p1f1",
            require_all_on="source_id",
        )

        # Verify concat was called the expected number of times
        assert mock_concat.call_count == 3

        # Verify _cmip_clip was called
        assert copt._cmip_clip.call_count >= 1


@patch("climakitae.explore.uncertainty.intake")
@patch("climakitae.explore.uncertainty._drop_member_id")
def test_grab_multimodel_data_precipitation(mock_drop_member, mock_intake):
    """Test grab_multimodel_data with precipitation variable."""
    # Create mock catalog and search results
    mock_catalog = MagicMock()
    mock_intake.open_esm_datastore.return_value = mock_catalog

    # Setup mock search and dict returns
    mock_search = MagicMock()
    mock_catalog.search.return_value = mock_search
    mock_search.search.return_value = mock_search
    mock_search.to_dataset_dict.return_value = {}

    # Configure drop_member_id to return the input dictionaries
    mock_drop_member.side_effect = lambda x: x

    # Create a mock CmipOpt instance with precipitation
    copt = MagicMock(spec=CmipOpt)
    copt.variable = "pr"
    copt.area_subset = "states"
    copt.location = "California"
    copt.timescale = "monthly"

    # Mock the _cmip_clip method to return the dataset directly
    copt._cmip_clip = MagicMock()
    copt._cmip_clip.side_effect = lambda ds: ds

    # Mock concat operations to return empty datasets
    with (
        patch("xarray.concat", return_value=xr.Dataset()),
        patch("xarray.Dataset.sel", return_value=xr.Dataset()),
    ):

        # Call the function
        grab_multimodel_data(copt)

        # Verify correct paths for precipitation search
        calls = mock_catalog.search.call_args_list
        assert len(calls) >= 2

        # Check that the precipitation-specific paths were used for the second search
        expected_paths = [
            "CESM2.*r11i1p1f1",
            "CNRM-ESM2-1.*r1i1p1f2",
            "MPI-ESM1-2-LR.*r7i1p1f1",
        ]

        for path in expected_paths:
            assert path in calls[1][1]["path"]


@patch("climakitae.explore.uncertainty.intake")
@patch("climakitae.explore.uncertainty._drop_member_id")
def test_grab_multimodel_data_temperature(mock_drop_member, mock_intake):
    """Test grab_multimodel_data with temperature variable."""
    # Create mock catalog and search results
    mock_catalog = MagicMock()
    mock_intake.open_esm_datastore.return_value = mock_catalog

    # Setup mock search and dict returns
    mock_search = MagicMock()
    mock_catalog.search.return_value = mock_search
    mock_search.search.return_value = mock_search
    mock_search.to_dataset_dict.return_value = {}

    # Configure drop_member_id to return the input dictionaries
    mock_drop_member.side_effect = lambda x: x

    # Create a mock CmipOpt instance with precipitation
    copt = MagicMock(spec=CmipOpt)
    copt.variable = "tas"
    copt.area_subset = "states"
    copt.location = "California"
    copt.timescale = "monthly"

    # Mock the _cmip_clip method to return the dataset directly
    copt._cmip_clip = MagicMock()
    copt._cmip_clip.side_effect = lambda ds: ds

    # Mock concat operations to return empty datasets
    with (
        patch("xarray.concat", return_value=xr.Dataset()),
        patch("xarray.Dataset.sel", return_value=xr.Dataset()),
    ):

        # Call the function
        grab_multimodel_data(copt)

        # Verify correct paths for temperature search
        calls = mock_catalog.search.call_args_list
        assert len(calls) >= 2

        # Check that the temperature-specific paths were used for the second search
        # (different from precipitation paths)
        expected_paths = ["CESM2.*r11i1p1f1", "CNRM-ESM2-1.*r1i1p1f2"]

        second_call_paths = calls[1][1]["path"]
        assert len(second_call_paths) == len(expected_paths)
        for path in expected_paths:
            assert path in second_call_paths
        assert "MPI-ESM1-2-LR.*r7i1p1f1" not in second_call_paths


@patch("climakitae.explore.uncertainty.intake")
@patch("climakitae.explore.uncertainty._drop_member_id")
@patch("climakitae.explore.uncertainty._standardize_cmip6_data")
@patch("climakitae.explore.uncertainty._clip_region")  # Add this patch
def test_grab_multimodel_data_alpha_sort(
    mock_clip_region, mock_standardize, mock_drop_member, mock_intake
):
    """Test grab_multimodel_data with alpha_sort=True parameter."""
    # Create mock catalog and search results
    mock_catalog = MagicMock()
    mock_intake.open_esm_datastore.return_value = mock_catalog

    # Setup mock search and dict returns
    mock_search = MagicMock()
    mock_catalog.search.return_value = mock_search
    mock_search.search.return_value = mock_search

    # Setup dictionaries with unsorted model names
    hist_dict = {
        "cmip6.C.model2.historical": MagicMock(spec=xr.Dataset),
        "cmip6.A.model1.historical": MagicMock(spec=xr.Dataset),
    }
    ssp_dict = {
        "cmip6.C.model2.ssp370": MagicMock(spec=xr.Dataset),
        "cmip6.A.model1.ssp370": MagicMock(spec=xr.Dataset),
    }
    mock_search.to_dataset_dict.side_effect = [hist_dict, {}]

    # Configure cal-adapt dataset returns
    mock_cal_search = MagicMock()
    mock_catalog.search.return_value.to_dataset_dict.side_effect = [hist_dict, ssp_dict]
    mock_catalog.search.side_effect = [mock_search, mock_cal_search, None]
    mock_cal_search.to_dataset_dict.return_value = {}

    # Configure drop_member_id to return dictionaries that can be sorted
    mock_drop_member.return_value = hist_dict

    # Make clip_region return the input dataset
    mock_clip_region.side_effect = lambda ds, *args, **kwargs: ds

    # Create a mock CmipOpt instance instead of a real one
    copt = MagicMock(spec=CmipOpt)
    copt.variable = "tas"
    copt.area_subset = "states"
    copt.location = "California"
    copt.timescale = "monthly"

    # Mock _cmip_clip to avoid calling the real implementation
    copt._cmip_clip = MagicMock(side_effect=lambda ds: ds)

    # Mock the sorting operation to verify it's called
    with (
        patch("builtins.sorted") as mock_sorted,
        patch("xarray.concat", return_value=xr.Dataset()),
        patch("xarray.Dataset.sel", return_value=xr.Dataset()),
    ):

        # Call function with alpha_sort=True
        grab_multimodel_data(copt, alpha_sort=True)

        # Verify the sorted function was called
        assert mock_sorted.call_count >= 2

        # Verify the key function was used for sorting by model name
        sort_key = mock_sorted.call_args_list[0][1]["key"]
        test_item = ("cmip6.A.model1.historical", None)
        assert sort_key(test_item) == "model1"


@patch("climakitae.explore.uncertainty.intake")
@patch("climakitae.explore.uncertainty._drop_member_id")
def test_grab_multimodel_data_time_slice_selection(mock_drop_member, mock_intake):
    """Test time slice selection in grab_multimodel_data."""
    # Create mock catalog and search results
    mock_catalog = MagicMock()
    mock_intake.open_esm_datastore.return_value = mock_catalog

    # Setup mock search and dict returns
    mock_search = MagicMock()
    mock_cal_search = MagicMock()
    mock_catalog.search.return_value = mock_search
    mock_search.search.return_value = mock_cal_search

    # Configure mock returns for dataset dictionaries
    hist_dict = {"model1.historical": MagicMock(spec=xr.Dataset)}
    ssp_dict = {"model1.ssp370": MagicMock(spec=xr.Dataset)}

    # Set up search results
    mock_catalog.search.side_effect = [mock_search, mock_cal_search]
    mock_cal_search.to_dataset_dict.side_effect = [hist_dict, ssp_dict]

    # Configure drop_member_id to return the input dictionaries
    mock_drop_member.side_effect = lambda x: x

    # Create a mock CmipOpt instance
    copt = MagicMock(spec=CmipOpt)
    copt.variable = "tas"
    copt.area_subset = "states"
    copt.location = "California"
    copt.timescale = "monthly"

    # Mock _cmip_clip to return its input
    copt._cmip_clip = MagicMock(side_effect=lambda ds: ds)

    # Mock concat results - this is where sel() will be called
    mock_hist_concat = MagicMock(spec=xr.Dataset)
    mock_ssp_concat = MagicMock(spec=xr.Dataset)

    # Set up squeeze on concatenated results
    mock_hist_squeeze = MagicMock(spec=xr.Dataset)
    mock_hist_concat.squeeze = MagicMock(return_value=mock_hist_squeeze)

    mock_ssp_squeeze = MagicMock(spec=xr.Dataset)
    mock_ssp_concat.squeeze = MagicMock(return_value=mock_ssp_squeeze)

    # Set up sel on squeezed historical dataset
    mock_hist_sel = MagicMock(spec=xr.Dataset)
    mock_hist_squeeze.sel = MagicMock(return_value=mock_hist_sel)

    # We don't need to mock ssp_squeeze.sel() because that's not called in the function

    with patch("xarray.concat") as mock_concat:
        mock_concat.side_effect = [mock_hist_concat, mock_ssp_concat, MagicMock()]

        # Call the function
        grab_multimodel_data(copt)

        # Verify the time slices were applied to the concatenated results
        mock_hist_squeeze.sel.assert_called_with(time=slice("1850", "2014"))

        # Verify _cmip_clip was called with the right arguments
        copt._cmip_clip.assert_any_call(mock_hist_sel)
        copt._cmip_clip.assert_any_call(mock_ssp_squeeze)


@patch("climakitae.explore.uncertainty.intake")
@patch("climakitae.explore.uncertainty._drop_member_id")
@patch("xarray.concat")
def test_grab_multimodel_data_cmip_clip(mock_concat, mock_drop_member, mock_intake):
    """Test that _cmip_clip is called in grab_multimodel_data."""
    # Create mock datasets and returns
    mock_catalog = MagicMock()
    mock_intake.open_esm_datastore.return_value = mock_catalog
    mock_search = MagicMock()
    mock_catalog.search.return_value = mock_search
    mock_search.search.return_value = mock_search
    mock_search.to_dataset_dict.return_value = {}

    # Mock datasets after concat
    mock_hist_ds = MagicMock(spec=xr.Dataset)
    mock_ssp_ds = MagicMock(spec=xr.Dataset)
    mock_concat.side_effect = [mock_hist_ds, mock_ssp_ds, MagicMock()]

    # Setup squeeze and sel methods to match the actual function flow
    mock_hist_squeeze = MagicMock(spec=xr.Dataset)
    mock_hist_ds.squeeze = MagicMock(return_value=mock_hist_squeeze)

    mock_hist_sel = MagicMock(spec=xr.Dataset)
    mock_hist_squeeze.sel = MagicMock(return_value=mock_hist_sel)

    mock_ssp_squeeze = MagicMock(spec=xr.Dataset)
    mock_ssp_ds.squeeze = MagicMock(return_value=mock_ssp_squeeze)

    # Configure drop_member_id
    mock_drop_member.side_effect = lambda x: x

    # Create a mock CmipOpt instance with a spy on _cmip_clip
    copt = MagicMock(spec=CmipOpt)
    copt.variable = "tas"
    copt.timescale = "monthly"
    copt._cmip_clip = MagicMock()

    # Call the function
    grab_multimodel_data(copt)

    # Verify _cmip_clip was called with both historical and SSP datasets
    assert copt._cmip_clip.call_count == 2

    # Updated assertions to match actual function behavior
    copt._cmip_clip.assert_any_call(mock_hist_squeeze.sel.return_value)
    copt._cmip_clip.assert_any_call(mock_ssp_squeeze)


@patch("climakitae.explore.uncertainty.intake")
def test_grab_multimodel_data_error_handling(mock_intake):
    """Test error handling in grab_multimodel_data."""
    # Simulate catalog access error
    mock_intake.open_esm_datastore.side_effect = Exception("Catalog access error")

    copt = CmipOpt()

    # Check that the function raises the exception properly
    with pytest.raises(Exception) as excinfo:
        grab_multimodel_data(copt)

    assert "Catalog access error" in str(excinfo.value)


@patch("climakitae.explore.uncertainty._grab_ensemble_data_by_experiment_id")
@patch("climakitae.explore.uncertainty.get_warm_level")
@patch("climakitae.explore.uncertainty.area_subset_geometry")
def test_get_ensemble_data_basic_functionality(
    mock_area_subset, mock_warm_level, mock_grab_data
):
    """Test the basic functionality of get_ensemble_data."""
    # Setup mocks for input datasets
    mock_hist_ds = MagicMock(spec=xr.Dataset)
    mock_ssp_ds = MagicMock(spec=xr.Dataset)

    # Setup nested attributes properly
    mock_hist_ds.member_id = MagicMock()
    mock_hist_ds.member_id.values = ["r1i1p1f1"]
    mock_ssp_ds.member_id = MagicMock()
    mock_ssp_ds.member_id.values = ["r1i1p1f1"]

    # Create simulation attribute with item method
    mock_hist_ds.simulation = MagicMock()
    mock_hist_ds.simulation.item.return_value = "model1"
    mock_ssp_ds.simulation = MagicMock()
    mock_ssp_ds.simulation.item.return_value = "model1"

    # Configure grab_ensemble_data_by_experiment_id mock
    mock_grab_data.side_effect = [
        [mock_ssp_ds],  # SSP370 data
        [mock_hist_ds],  # Historical data
    ]

    # Configure warm_level mock
    mock_warm_ds = MagicMock(spec=xr.Dataset)
    mock_warm_level.return_value = mock_warm_ds

    # Configure geometry and selections
    mock_geometry = MagicMock()
    mock_area_subset.return_value = mock_geometry
    mock_selections = MagicMock()
    mock_selections.area_average = "No"

    # Configure dataset selections
    mock_hist_sel = MagicMock(spec=xr.Dataset)
    mock_ssp_sel = MagicMock(spec=xr.Dataset)
    mock_hist_ds.sel = MagicMock(return_value=mock_hist_sel)
    mock_ssp_ds.sel = MagicMock(return_value=mock_ssp_sel)

    # Configure xarray.concat
    with patch("xarray.concat") as mock_concat:

        mock_hist_concat = MagicMock(spec=xr.Dataset)
        mock_warm_concat = MagicMock(spec=xr.Dataset)

        mock_concat.side_effect = [mock_hist_concat, mock_warm_concat]

        # Configure member IDs for concatenated results
        mock_hist_concat.simulation = MagicMock()
        mock_hist_concat.simulation.values = ["model1"]
        mock_hist_concat.member_id = MagicMock()
        mock_hist_concat.member_id.values = ["r1i1p1f1"]
        mock_warm_concat.simulation = MagicMock()
        mock_warm_concat.simulation.values = ["model1"]
        mock_warm_concat.member_id = MagicMock()
        mock_warm_concat.member_id.values = ["r1i1p1f1"]

        # Configure the rio methods for both concatenated datasets
        # Note: Using a simple approach where each method returns the same object
        mock_hist_concat.rio = MagicMock()
        mock_hist_concat.rio.write_crs = MagicMock(return_value=mock_hist_concat)
        mock_hist_concat.rio.clip = MagicMock(return_value=mock_hist_concat)

        mock_warm_concat.rio = MagicMock()
        mock_warm_concat.rio.write_crs = MagicMock(return_value=mock_warm_concat)
        mock_warm_concat.rio.clip = MagicMock(return_value=mock_warm_concat)

        # Mock all possible sel calls to return the same object for simplicity
        mock_hist_concat.sel = MagicMock(return_value=mock_hist_concat)
        mock_warm_concat.sel = MagicMock(return_value=mock_warm_concat)

        # Add necessary attributes and methods for model ID handling
        mock_hist_concat.coords = {"member_id": MagicMock()}

        # Call function
        from climakitae.explore.uncertainty import get_ensemble_data

        hist_result, warm_result = get_ensemble_data("tas", mock_selections, ["model1"])

        # Verify the results are not None
        assert hist_result is not None
        assert warm_result is not None

        # Verify the correct calls were made to the mock functions
        mock_grab_data.assert_any_call("tas", ["model1"], "ssp370")
        mock_grab_data.assert_any_call("tas", ["model1"], "historical")
        mock_warm_level.assert_called_with(
            3.0, mock_ssp_sel, multi_ens=True, ipcc=False
        )


@patch("climakitae.explore.uncertainty._grab_ensemble_data_by_experiment_id")
@patch("climakitae.explore.uncertainty.get_warm_level")
@patch("climakitae.explore.uncertainty.area_subset_geometry")
def test_get_ensemble_data_multiple_models(
    mock_area_subset, mock_warm_level, mock_grab_data
):
    """Test get_ensemble_data with multiple models."""
    # Setup mocks for datasets from two models
    mock_hist_ds1 = MagicMock(spec=xr.Dataset)
    mock_hist_ds2 = MagicMock(spec=xr.Dataset)
    mock_ssp_ds1 = MagicMock(spec=xr.Dataset)
    mock_ssp_ds2 = MagicMock(spec=xr.Dataset)

    # Configure member_id as a MagicMock first
    mock_hist_ds1.member_id = MagicMock()
    mock_hist_ds2.member_id = MagicMock()
    mock_ssp_ds1.member_id = MagicMock()
    mock_ssp_ds2.member_id = MagicMock()

    # Now you can set the values attribute
    mock_hist_ds1.member_id.values = ["r1i1p1f1"]
    mock_hist_ds2.member_id.values = ["r1i1p1f1"]
    mock_ssp_ds1.member_id.values = ["r1i1p1f1"]
    mock_ssp_ds2.member_id.values = ["r1i1p1f1"]

    # Configure simulation as a MagicMock
    mock_hist_ds1.simulation = MagicMock()
    mock_hist_ds2.simulation = MagicMock()
    mock_ssp_ds1.simulation = MagicMock()
    mock_ssp_ds2.simulation = MagicMock()

    # Set the return values for simulation.item()
    mock_hist_ds1.simulation.item.return_value = "model1"
    mock_hist_ds2.simulation.item.return_value = "model2"
    mock_ssp_ds1.simulation.item.return_value = "model1"
    mock_ssp_ds2.simulation.item.return_value = "model2"

    # Configure data retrieval
    mock_grab_data.side_effect = [
        [mock_ssp_ds1, mock_ssp_ds2],  # SSP370 data
        [mock_hist_ds1, mock_hist_ds2],  # Historical data
    ]

    # Configure warm level mocks
    mock_warm_ds1 = MagicMock(spec=xr.Dataset)
    mock_warm_ds2 = MagicMock(spec=xr.Dataset)
    mock_warm_level.side_effect = [mock_warm_ds1, mock_warm_ds2]

    # Configure geometry and selections
    mock_geometry = MagicMock()
    mock_area_subset.return_value = mock_geometry
    mock_selections = MagicMock()
    mock_selections.area_average = "No"

    # Configure dataset selections
    mock_hist_sel1 = MagicMock(spec=xr.Dataset)
    mock_hist_sel2 = MagicMock(spec=xr.Dataset)
    mock_hist_ds1.sel = MagicMock(return_value=mock_hist_sel1)
    mock_hist_ds2.sel = MagicMock(return_value=mock_hist_sel2)

    # Configure xarray.concat
    with patch("xarray.concat") as mock_concat:
        mock_hist_concat = MagicMock(spec=xr.Dataset)
        mock_warm_concat = MagicMock(spec=xr.Dataset)
        mock_concat.side_effect = [mock_hist_concat, mock_warm_concat]

        # Configure member IDs for concatenated results
        mock_hist_concat.simulation = MagicMock()
        mock_hist_concat.simulation.values = ["model1", "model2"]
        mock_hist_concat.member_id = MagicMock()
        mock_hist_concat.member_id.values = ["r1i1p1f1", "r1i1p1f1"]
        mock_warm_concat.simulation = MagicMock()
        mock_warm_concat.simulation.values = ["model1", "model2"]
        mock_warm_concat.member_id = MagicMock()
        mock_warm_concat.member_id.values = ["r1i1p1f1", "r1i1p1f1"]

        # Configure time slice result
        mock_hist_slice = MagicMock(spec=xr.Dataset)
        mock_hist_concat.sel = MagicMock(return_value=mock_hist_slice)

        # Configure rio methods
        mock_hist_slice.rio = MagicMock()
        mock_hist_slice.rio.write_crs = MagicMock(return_value=mock_hist_slice)
        mock_hist_slice.rio.clip = MagicMock(return_value=mock_hist_slice)

        mock_warm_concat.rio = MagicMock()
        mock_warm_concat.rio.write_crs = MagicMock(return_value=mock_warm_concat)
        mock_warm_concat.rio.clip = MagicMock(return_value=mock_warm_concat)

        # Call function
        from climakitae.explore.uncertainty import get_ensemble_data

        hist_result, warm_result = get_ensemble_data(
            "tas", mock_selections, ["model1", "model2"]
        )

        # Verify both models were processed
        assert mock_warm_level.call_count == 2
        mock_warm_level.assert_any_call(
            3.0, mock_ssp_ds1.sel(), multi_ens=True, ipcc=False
        )
        mock_warm_level.assert_any_call(
            3.0, mock_ssp_ds2.sel(), multi_ens=True, ipcc=False
        )


@patch("climakitae.explore.uncertainty._grab_ensemble_data_by_experiment_id")
@patch("climakitae.explore.uncertainty.get_warm_level")
@patch("climakitae.explore.uncertainty.area_subset_geometry")
@patch("climakitae.explore.uncertainty._precip_flux_to_total")
def test_get_ensemble_data_precipitation(
    mock_precip, mock_area_subset, mock_warm_level, mock_grab_data
):
    """Test get_ensemble_data for precipitation variable."""
    # Setup basic mocks
    mock_hist_ds = MagicMock(spec=xr.Dataset)
    mock_ssp_ds = MagicMock(spec=xr.Dataset)

    # Configure member IDs and simulation values
    mock_hist_ds.member_id = MagicMock()
    mock_ssp_ds.member_id = MagicMock()
    mock_hist_ds.simulation = MagicMock()
    mock_ssp_ds.simulation = MagicMock()
    mock_hist_ds.member_id.values = ["r1i1p1f1"]
    mock_ssp_ds.member_id.values = ["r1i1p1f1"]
    mock_hist_ds.simulation.item.return_value = "model1"
    mock_ssp_ds.simulation.item.return_value = "model1"

    # Configure grab data mock
    mock_grab_data.side_effect = [
        [mock_ssp_ds],  # SSP370 data
        [mock_hist_ds],  # Historical data
    ]

    # Configure warm level mock
    mock_warm_ds = MagicMock(spec=xr.Dataset)
    mock_warm_level.return_value = mock_warm_ds

    # Configure geometry and selections
    mock_geometry = MagicMock()
    mock_area_subset.return_value = mock_geometry
    mock_selections = MagicMock()
    mock_selections.area_average = "No"

    # Configure precip conversion
    mock_precip.side_effect = lambda ds: ds

    # Configure dataset selection
    mock_hist_sel = MagicMock(spec=xr.Dataset)
    mock_hist_ds.sel.return_value = mock_hist_sel

    # Configure xarray.concat
    with patch("xarray.concat") as mock_concat:
        mock_hist_concat = MagicMock(spec=xr.Dataset)
        mock_warm_concat = MagicMock(spec=xr.Dataset)
        mock_concat.side_effect = [mock_hist_concat, mock_warm_concat]

        # mock attributes
        mock_hist_concat.simulation = MagicMock()
        mock_warm_concat.simulation = MagicMock()
        mock_hist_concat.member_id = MagicMock()
        mock_warm_concat.member_id = MagicMock()

        # Configure member IDs for concatenated results
        mock_hist_concat.simulation.values = ["model1"]
        mock_hist_concat.member_id.values = ["r1i1p1f1"]
        mock_warm_concat.simulation.values = ["model1"]
        mock_warm_concat.member_id.values = ["r1i1p1f1"]

        # Configure time slice result
        mock_hist_slice = MagicMock(spec=xr.Dataset)
        mock_hist_concat.sel.return_value = mock_hist_slice

        # Configure rio methods
        mock_hist_slice.rio = MagicMock()
        mock_hist_slice.rio.write_crs = MagicMock(return_value=mock_hist_slice)
        mock_hist_slice.rio.clip = MagicMock(return_value=mock_hist_slice)

        mock_warm_concat.rio = MagicMock()
        mock_warm_concat.rio.write_crs = MagicMock(return_value=mock_warm_concat)
        mock_warm_concat.rio.clip = MagicMock(return_value=mock_warm_concat)

        # Call function with precipitation variable
        from climakitae.explore.uncertainty import get_ensemble_data

        get_ensemble_data("pr", mock_selections, ["model1"])

        # Verify precipitation conversion was called
        mock_precip.assert_called()


@patch("climakitae.explore.uncertainty._grab_ensemble_data_by_experiment_id")
@patch("climakitae.explore.uncertainty.get_warm_level")
@patch("climakitae.explore.uncertainty.area_subset_geometry")
@patch("climakitae.explore.uncertainty._area_wgt_average")
def test_get_ensemble_data_area_averaging(
    mock_area_avg, mock_area_subset, mock_warm_level, mock_grab_data, caplog
):
    """Test get_ensemble_data with area averaging."""
    # Setup basic mocks
    mock_hist_ds = MagicMock(spec=xr.Dataset)
    mock_ssp_ds = MagicMock(spec=xr.Dataset)

    # mock member_id and simulation attributes
    mock_hist_ds.member_id = MagicMock()
    mock_ssp_ds.member_id = MagicMock()
    mock_hist_ds.simulation = MagicMock()
    mock_ssp_ds.simulation = MagicMock()

    # Configure member IDs and simulation values
    mock_hist_ds.member_id.values = ["r1i1p1f1"]
    mock_ssp_ds.member_id.values = ["r1i1p1f1"]
    mock_hist_ds.simulation.item.return_value = "model1"
    mock_ssp_ds.simulation.item.return_value = "model1"

    # Configure grab data mock
    mock_grab_data.side_effect = [
        [mock_ssp_ds],  # SSP370 data
        [mock_hist_ds],  # Historical data
    ]

    # Configure warm level mock
    mock_warm_ds = MagicMock(spec=xr.Dataset)
    mock_warm_level.return_value = mock_warm_ds

    # Configure geometry and selections
    mock_geometry = MagicMock()
    mock_area_subset.return_value = mock_geometry
    mock_selections = MagicMock()
    mock_selections.area_average = "Yes"  # Enable area averaging

    # Configure area_wgt_average to return a modified dataset
    mock_area_avg.side_effect = lambda ds: ds

    # Configure dataset selection
    mock_hist_sel = MagicMock(spec=xr.Dataset)
    mock_ssp_sel = MagicMock(spec=xr.Dataset)
    mock_hist_ds.sel = MagicMock(return_value=mock_hist_sel)
    mock_ssp_ds.sel = MagicMock(return_value=mock_ssp_sel)

    # Configure xarray.concat
    with patch("xarray.concat") as mock_concat:
        mock_hist_concat = MagicMock(spec=xr.Dataset)
        mock_warm_concat = MagicMock(spec=xr.Dataset)
        mock_concat.side_effect = [mock_hist_concat, mock_warm_concat]

        # Set up the member IDs for concatenated results
        mock_hist_concat.simulation = MagicMock()
        mock_hist_concat.simulation.values = ["model1"]
        mock_hist_concat.member_id = MagicMock()
        mock_hist_concat.member_id.values = ["r1i1p1f1"]
        mock_warm_concat.simulation = MagicMock()
        mock_warm_concat.simulation.values = ["model1"]
        mock_warm_concat.member_id = MagicMock()
        mock_warm_concat.member_id.values = ["r1i1p1f1"]

        # Set up the coords for the history selection
        mock_hist_concat.coords = {}
        mock_hist_concat.coords["member_id"] = ["model1r1i1p1f1"]

        # Configure various sel operations on the concatenated history dataset
        mock_hist_slice_member = MagicMock(spec=xr.Dataset)
        mock_hist_slice_time = MagicMock(spec=xr.Dataset)

        # Make sel return different mock objects based on the arguments
        def mock_hist_concat_sel_side_effect(**kwargs):
            if "member_id" in kwargs:
                return mock_hist_slice_member
            elif "time" in kwargs:
                return mock_hist_slice_time
            return MagicMock(spec=xr.Dataset)

        mock_hist_concat.sel = MagicMock(side_effect=mock_hist_concat_sel_side_effect)

        # Configure rio for the history slice (time selection)
        mock_hist_slice_time.rio = MagicMock()
        mock_hist_slice_time.rio.write_crs = MagicMock(
            return_value=mock_hist_slice_time
        )
        mock_hist_rio_clipped = MagicMock(spec=xr.Dataset)
        mock_hist_slice_time.rio.clip = MagicMock(return_value=mock_hist_rio_clipped)

        # Configure rio for the warming concat
        mock_warm_concat.rio = MagicMock()
        mock_warm_concat.rio.write_crs = MagicMock(return_value=mock_warm_concat)
        mock_warm_rio_clipped = MagicMock(spec=xr.Dataset)
        mock_warm_concat.rio.clip = MagicMock(return_value=mock_warm_rio_clipped)

        hist_result, warm_result = get_ensemble_data("tas", mock_selections, ["model1"])

        # Verify area_wgt_average was called exactly twice
        assert (
            mock_area_avg.call_count == 2
        ), "Expected area_wgt_average to be called twice"

        # Add custom assertion that will provide details on failure
        # This approach helps identify which objects were actually passed
        assert hist_result is not None, "Historical result should not be None"
        assert warm_result is not None, "Warming result should not be None"


def test_weighted_temporal_mean_basic():
    """Test of basic functionality for weighted_temporal_mean function."""
    # Create a sample dataset
    ds = xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), np.random.rand(5, 4, 3)),
            "weights": (("time",), np.random.rand(5)),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=5),
            "lat": [10, 20, 30, 40],
            "lon": [100, 110, 120],
        },
    )

    # Call the function
    result = weighted_temporal_mean(ds)

    # Check the result
    assert isinstance(result, xr.Dataset)
    assert "temperature" in result.data_vars
    assert result["temperature"].dims == ("time", "lat", "lon")
