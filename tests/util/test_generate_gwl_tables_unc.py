from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.util.generate_gwl_tables_unc import (
    GWLGenerator,
    main,
    make_weighted_timeseries,
)


@pytest.fixture
def mock_cmip6_df():
    """Creates a mock CMIP6 DataFrame for testing."""
    data = {
        "zstore": [
            "s3://bucket/path1",  # hist r1
            "s3://bucket/path2",  # hist r2
            "s3://bucket/path3",  # ssp370 r1
            "s3://bucket/path4",  # ssp370 r2
            "s3://bucket/path5",  # ssp585 r1
            "s3://bucket/path6",  # ssp585 r2
            "s3://bucket/path7",  # ssp126 r1
            "s3://bucket/path8",  # ssp245 r1
        ],
        "table_id": ["Amon"] * 8,
        "variable_id": ["tas"] * 8,
        "experiment_id": [
            "historical",
            "historical",
            "ssp370",
            "ssp370",
            "ssp585",
            "ssp585",
            "ssp126",
            "ssp245",
        ],
        "source_id": ["EC-Earth3"] * 8,
        "member_id": [
            "r1i1p1f1",
            "r2i1p1f1",
            "r1i1p1f1",
            "r2i1p1f1",
            "r1i1p1f1",
            "r2i1p1f1",
            "r1i1p1f1",
            "r1i1p1f1",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_generator(mock_cmip6_df):
    """Creates a GWLGenerator instance with mocked dependencies."""
    with patch("s3fs.S3FileSystem") as mock_s3fs:
        # Create a mock instance for the S3FileSystem
        mock_fs_instance = MagicMock()
        # Configure the mock S3FileSystem class to return our instance
        mock_s3fs.return_value = mock_fs_instance

        # Mock get_sims_on_aws to avoid complex mocking within __init__
        with patch.object(
            GWLGenerator, "get_sims_on_aws", autospec=True
        ) as mock_get_sims:
            # Define a simple mock sims_on_aws DataFrame structure
            sims_data = {
                "historical": [["r1i1p1f1", "r2i1p1f1"]],
                "ssp585": [["r1i1p1f1", "r2i1p1f1"]],
                "ssp370": [["r1i1p1f1", "r2i1p1f1"]],
                "ssp245": [["r1i1p1f1"]],
                "ssp126": [["r1i1p1f1"]],
            }
            # Use a specific index name for clarity if needed, otherwise default is fine
            mock_sims_on_aws = pd.DataFrame(sims_data, index=["EC-Earth3"])
            mock_get_sims.return_value = mock_sims_on_aws

            # Now instantiate GWLGenerator. __init__ will use the mocked s3fs and get_sims_on_aws
            generator = GWLGenerator(mock_cmip6_df)

            # Although __init__ uses the mocks, let's explicitly ensure the instance
            # has the correct mock objects assigned for clarity in tests.
            # __init__ should have already assigned the mock_fs_instance.
            # If sims_on_aws=None was passed (default), __init__ calls get_sims_on_aws
            # and assigns its return value (mock_sims_on_aws).
            assert generator.fs is mock_fs_instance
            assert generator.sims_on_aws is mock_sims_on_aws

            # Make the mock fs instance's get_mapper return a unique mock per call
            # to allow asserting calls with different arguments if needed,
            # or just return a single mock if the return value doesn't matter.
            # For simplicity here, let's assume the return value itself isn't critical
            # for the build_timeseries logic, just that it's called.
            mock_fs_instance.get_mapper = MagicMock(return_value="mock_mapper_path")

            return generator


class TestGWLGenerator:

    @patch("s3fs.S3FileSystem")
    @patch("climakitae.util.generate_gwl_tables_unc.GWLGenerator.get_sims_on_aws")
    def test_init(self, mock_get_sims_on_aws, mock_s3fs):
        """
        Test of init method.

        Test that df is assigned correctly.
        Test that sims_on_aws is assigned correctly.
        Test that fs is assigned correctly.
        """

        # Create a mock DataFrame
        df = {"GWL": [1.5, 2.0, 2.5], "Year": [2000, 2001, 2002]}

        # Mock the return value of get_sims_on_aws
        mock_get_sims_on_aws.return_value = "mock_sims_on_aws"
        # Mock the S3FileSystem instance
        mock_s3fs.return_value = "mock_fs"

        # Create an instance of GWLGenerator
        generator = GWLGenerator(df)

        # Assert that the attributes are assigned correctly
        assert generator.df == df
        assert generator.sims_on_aws == "mock_sims_on_aws"
        assert generator.fs == "mock_fs"

    def test_get_sims_on_aws(self):
        """Test the get_sims_on_aws method for filtering and structuring simulation data."""
        # --- Mock Input DataFrame (self.df) ---
        # More complex df to test filtering logic
        data = {
            "zstore": [f"s3://path{i}" for i in range(15)],
            "table_id": ["Amon"] * 14 + ["Omon"],  # Include one wrong table_id
            "variable_id": ["tas"] * 13 + ["tos", "tas"],  # Include wrong variable_id
            "experiment_id": [
                "historical",
                "historical",
                "ssp370",
                "ssp585",  # ModelA (hist+ssp, r1 overlap)
                "historical",
                "ssp126",
                "ssp245",  # ModelB (hist+ssp, no overlap)
                "historical",
                "historical",  # ModelC (hist only)
                "ssp370",
                "ssp585",  # ModelD (ssp only)
                "historical",
                "ssp370",  # ModelE (hist+ssp, r1 overlap, tas)
                "historical",  # ModelF (hist only, Omon)
                "historical",  # ModelA (hist only, Omon)
            ],
            "source_id": [
                "ModelA",
                "ModelA",
                "ModelA",
                "ModelA",
                "ModelB",
                "ModelB",
                "ModelB",
                "ModelC",
                "ModelC",
                "ModelD",
                "ModelD",
                "ModelE",
                "ModelE",  # ModelE tas
                "ModelF",  # ModelF Omon
                "ModelA",  # ModelA tas again for Omon test
            ],
            "member_id": [
                "r1",
                "r2",
                "r1",
                "r1",  # ModelA
                "r1",
                "r2",
                "r2",  # ModelB
                "r1",
                "r2",  # ModelC
                "r1",
                "r2",  # ModelD
                "r1",
                "r1",  # ModelE tas
                "r1",  # ModelF Omon
                "r1",  # ModelA Omon tas
            ],
        }
        mock_df = pd.DataFrame(data)

        # --- Instantiate Generator (without mocking get_sims_on_aws itself) ---
        # We need to call the actual method, so we only mock s3fs if needed,
        # but here we don't even need s3fs for this specific method test.
        # We pass sims_on_aws=True temporarily to prevent __init__ from calling
        # the method we want to test. Then we set it back to None.
        generator = GWLGenerator(mock_df, sims_on_aws=True)
        generator.sims_on_aws = None  # Reset so the method runs

        # --- Call the Method ---
        result_df = generator.get_sims_on_aws()

        # --- Assertions ---
        assert isinstance(result_df, pd.DataFrame)

        # Expected models: A, B, E (C dropped - no SSP, D dropped - no hist, F ignored - wrong table/var)
        expected_index = pd.Index(
            ["ModelA", "ModelB", "ModelE"], name="source_id"
        ).sort_values()
        pd.testing.assert_index_equal(result_df.index.sort_values(), expected_index)

        expected_columns = ["historical", "ssp585", "ssp370", "ssp245", "ssp126"]
        pd.testing.assert_index_equal(result_df.columns, pd.Index(expected_columns))

        # Check cell values (lists should be sorted for consistent comparison)
        # ModelA: hist r1, r2; ssp370 r1; ssp585 r1. Keep hist r1 only.
        assert sorted(result_df.loc["ModelA", "historical"]) == ["r1"]
        assert sorted(result_df.loc["ModelA", "ssp585"]) == ["r1"]
        assert sorted(result_df.loc["ModelA", "ssp370"]) == ["r1"]
        assert sorted(result_df.loc["ModelA", "ssp245"]) == []
        assert sorted(result_df.loc["ModelA", "ssp126"]) == []

        # ModelB: hist r1; ssp126 r2; ssp245 r2. No overlap, hist should be empty.
        assert sorted(result_df.loc["ModelB", "historical"]) == []
        assert sorted(result_df.loc["ModelB", "ssp585"]) == []
        assert sorted(result_df.loc["ModelB", "ssp370"]) == []
        assert sorted(result_df.loc["ModelB", "ssp245"]) == ["r2"]
        assert sorted(result_df.loc["ModelB", "ssp126"]) == ["r2"]

        # ModelE: hist r1; ssp370 r1. Keep hist r1.
        assert sorted(result_df.loc["ModelE", "historical"]) == ["r1"]
        assert sorted(result_df.loc["ModelE", "ssp585"]) == []
        assert sorted(result_df.loc["ModelE", "ssp370"]) == ["r1"]
        assert sorted(result_df.loc["ModelE", "ssp245"]) == []
        assert sorted(result_df.loc["ModelE", "ssp126"]) == []

    def test_build_timeseries_basic(self, mock_generator):
        """Test the basic functionality of build_timeseries, including isel."""
        model_config = {
            "variable": "tas",
            "model": "EC-Earth3",
            "ens_mem": "r1i1p1f1",  # Added required ens_mem
            "scenarios": [
                "ssp370",
                "ssp585",
            ],  # Use scenarios from mock_gwl_generator fixture
        }

        # --- Mock Data ---
        # Mock data arrays returned by make_weighted_timeseries
        historical_time = pd.date_range("1850-01-01", periods=50, freq="MS")
        # Use a length that will be affected by isel(time=slice(0, 1032)) for testing
        ssp_time_full = pd.date_range("2015-01-01", periods=1100, freq="MS")
        ssp_time_sliced = ssp_time_full[:1032]  # The expected time coord after isel

        historical_array = xr.DataArray(
            np.random.rand(len(historical_time)),
            coords=[historical_time],
            dims=["time"],
            name="hist_ts",
        ).sortby("time")
        # These represent the data *after* make_weighted_timeseries is called
        ssp370_array = xr.DataArray(
            np.random.rand(len(ssp_time_sliced)) * 2,
            coords=[ssp_time_sliced],
            dims=["time"],
            name="ssp370_ts",
        ).sortby("time")
        ssp585_array = xr.DataArray(
            np.random.rand(len(ssp_time_sliced)) * 3,
            coords=[ssp_time_sliced],
            dims=["time"],
            name="ssp585_ts",
        ).sortby("time")

        # Mock zarr open results (need __enter__ for context manager)
        # We mock the object returned by the context manager directly
        mock_historical_temp_ctx = MagicMock(spec=xr.Dataset)  # Mock as Dataset
        mock_historical_temp_ctx.__getitem__.return_value = (
            "mock_hist_data"  # What make_weighted_timeseries receives
        )

        mock_ssp370_temp_ctx = MagicMock(spec=xr.Dataset)
        # Mock the isel call specifically to return itself for chaining
        mock_ssp370_temp_ctx.isel.return_value = mock_ssp370_temp_ctx
        mock_ssp370_temp_ctx.__getitem__.return_value = "mock_ssp370_data"

        mock_ssp585_temp_ctx = MagicMock(spec=xr.Dataset)
        mock_ssp585_temp_ctx.isel.return_value = mock_ssp585_temp_ctx
        mock_ssp585_temp_ctx.__getitem__.return_value = "mock_ssp585_data"

        # Mock the context manager return values
        mock_historical_open = MagicMock()
        mock_historical_open.__enter__.return_value = mock_historical_temp_ctx
        mock_ssp370_open = MagicMock()
        mock_ssp370_open.__enter__.return_value = mock_ssp370_temp_ctx
        mock_ssp585_open = MagicMock()
        mock_ssp585_open.__enter__.return_value = mock_ssp585_temp_ctx

        # Mock concatenated results
        # Note: The actual concat happens inside the loop, one scenario at a time
        mock_concatenated_370 = xr.concat([historical_array, ssp370_array], dim="time")
        mock_concatenated_585 = xr.concat([historical_array, ssp585_array], dim="time")

        # --- Patching ---
        with patch(
            "xarray.open_zarr",
            # Order matters: historical, ssp370, ssp585
            side_effect=[mock_historical_open, mock_ssp370_open, mock_ssp585_open],
        ) as mock_open_zarr, patch(
            "climakitae.util.generate_gwl_tables_unc.make_weighted_timeseries",
            # Order matters: historical, ssp370, ssp585
            side_effect=[historical_array, ssp370_array, ssp585_array],
        ) as mock_make_ts, patch(
            "xarray.decode_cf",
            side_effect=lambda ds: ds,  # Pass through mock dataset context
        ) as mock_decode, patch(
            "xarray.concat",
            # Order matters: ssp370 concat, ssp585 concat
            side_effect=[mock_concatenated_370, mock_concatenated_585],
        ) as mock_concat:

            # --- Call Method ---
            result_ds = mock_generator.build_timeseries(model_config)

            # --- Assertions ---
            # Check calls to fs.get_mapper (using the mock_gwl_generator's fs)
            # Paths correspond to mock_cmip6_df fixture data
            assert mock_generator.fs.get_mapper.call_count == 3
            mock_generator.fs.get_mapper.assert_has_calls(
                [
                    call("s3://bucket/path1"),  # historical r1i1p1f1
                    call("s3://bucket/path3"),  # ssp370 r1i1p1f1
                    call("s3://bucket/path5"),  # ssp585 r1i1p1f1
                ],
                any_order=False,
            )

            # Check open_zarr calls
            assert mock_open_zarr.call_count == 3
            # Check args for each call (hist doesn't decode_times=False)
            mock_open_zarr.assert_has_calls(
                [
                    call(mock_generator.fs.get_mapper.return_value),  # hist call
                    call(
                        mock_generator.fs.get_mapper.return_value,
                        decode_times=False,
                    ),  # ssp370 call
                    call(
                        mock_generator.fs.get_mapper.return_value,
                        decode_times=False,
                    ),  # ssp585 call
                ]
            )

            # Check isel calls (only on scenario data context managers)
            mock_ssp370_temp_ctx.isel.assert_called_once_with(time=slice(0, 1032))
            mock_ssp585_temp_ctx.isel.assert_called_once_with(time=slice(0, 1032))
            mock_historical_temp_ctx.isel.assert_not_called()  # Should not be called on historical

            # Check make_weighted_timeseries calls
            assert mock_make_ts.call_count == 3
            mock_make_ts.assert_has_calls(
                [
                    call("mock_hist_data"),
                    call("mock_ssp370_data"),
                    call("mock_ssp585_data"),
                ]
            )

            # Check decode_cf calls (only on scenario data context managers, after isel)
            assert mock_decode.call_count == 2
            mock_decode.assert_has_calls(
                [
                    call(mock_ssp370_temp_ctx),  # Called on the result of isel
                    call(mock_ssp585_temp_ctx),  # Called on the result of isel
                ]
            )

            # Check concat calls
            assert mock_concat.call_count == 2
            # Check first concat call (ssp370)
            concat_args_370, concat_kwargs_370 = mock_concat.call_args_list[0]
            assert len(concat_args_370[0]) == 2  # List of two arrays
            xr.testing.assert_equal(concat_args_370[0][0], historical_array)
            # The second item should be the *sorted* ssp370_array (sorting happens in the code)
            xr.testing.assert_equal(concat_args_370[0][1], ssp370_array)
            assert concat_kwargs_370["dim"] == "time"
            # Check second concat call (ssp585)
            concat_args_585, concat_kwargs_585 = mock_concat.call_args_list[1]
            assert len(concat_args_585[0]) == 2
            xr.testing.assert_equal(concat_args_585[0][0], historical_array)
            xr.testing.assert_equal(concat_args_585[0][1], ssp585_array)
            assert concat_kwargs_585["dim"] == "time"

            # Check the final result dataset
            assert isinstance(result_ds, xr.Dataset)
            assert list(result_ds.data_vars.keys()) == [
                "ssp370",
                "ssp585",
            ]  # Check data variables
            xr.testing.assert_equal(result_ds["ssp370"], mock_concatenated_370)
            xr.testing.assert_equal(result_ds["ssp585"], mock_concatenated_585)
