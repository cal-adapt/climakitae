"""Tests capability to read CMIP6 simulation information from aws"""

from typing import Dict, List, Tuple
from unittest.mock import MagicMock, PropertyMock, call, patch

import intake
import intake_esm
import numpy as np
import pandas as pd
import pytest
import s3fs  # Import needed for type hinting
import xarray as xr

from climakitae.core.constants import WARMING_LEVELS
from climakitae.util.generate_gwl_tables import (
    GWLGenerator,
    main,
    make_weighted_timeseries,
)

TEST_MODEL = "EC-Earth3"
TEST_REFERENCE_PERIOD = {"start_year": "19810101", "end_year": "20101231"}


@pytest.fixture
def mock_cmip6_df() -> pd.DataFrame:
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
def cesm_catalog() -> intake_esm.core.esm_datastore:
    catalog_cesm = intake.open_esm_datastore(
        "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json"
    )
    return catalog_cesm


@pytest.fixture
def mock_cesm2_lens() -> xr.Dataset:
    return xr.Dataset()


@pytest.fixture
def mock_generator(mock_cmip6_df: pd.DataFrame) -> GWLGenerator:
    """Creates a GWLGenerator instance with mocked S3FileSystem and get_sims_on_aws."""
    with patch("s3fs.S3FileSystem") as mock_s3fs:
        mock_fs_instance = mock_s3fs.return_value

        with patch.object(
            GWLGenerator, "get_sims_on_aws", autospec=True
        ) as mock_get_sims:
            # Define a mock sims_on_aws DataFrame structure
            sims_data = {
                "historical": [["r1i1p1f1", "r2i1p1f1"]],
                "ssp585": [["r1i1p1f1", "r2i1p1f1"]],
                "ssp370": [["r1i1p1f1", "r2i1p1f1"]],
                "ssp245": [["r1i1p1f1"]],
                "ssp126": [["r1i1p1f1"]],
            }
            mock_sims_on_aws = pd.DataFrame(sims_data, index=["EC-Earth3"])
            mock_get_sims.return_value = mock_sims_on_aws

            with patch(
                "climakitae.util.generate_gwl_tables.GWLGenerator._set_cesm2_lens"
            ):
                # Instantiate GWLGenerator; __init__ uses the mocks
                generator = GWLGenerator(mock_cmip6_df, {})

                # Verify mocks are assigned correctly within the instance
                assert generator.fs is mock_fs_instance
                assert generator.sims_on_aws is mock_sims_on_aws

                # Mock the get_mapper method
                mock_fs_instance.get_mapper = MagicMock(return_value="mock_mapper_path")

            return generator


class TestGWLGenerator:
    """Tests for the GWLGenerator class methods."""

    @patch("s3fs.S3FileSystem")
    @patch("climakitae.util.generate_gwl_tables.GWLGenerator.get_sims_on_aws")
    @patch("climakitae.util.generate_gwl_tables.GWLGenerator._set_cesm2_lens")
    def test_init(
        self,
        mock_cesm2_lens: MagicMock,
        mock_get_sims_on_aws: MagicMock,
        mock_s3fs: MagicMock,
    ):
        """
        Test the __init__ method of GWLGenerator.
        Verifies correct assignment of df, sims_on_aws, and fs attributes.
        """
        mock_df_data = {"GWL": [1.5, 2.0, 2.5], "Year": [2000, 2001, 2002]}
        mock_df = pd.DataFrame(mock_df_data)
        mock_sims_return = pd.DataFrame({"historical": [["r1"]]}, index=["ModelA"])

        mock_fs_instance = mock_s3fs.return_value

        mock_get_sims_on_aws.return_value = mock_sims_return

        generator = GWLGenerator(mock_df, {})

        pd.testing.assert_frame_equal(generator.df, mock_df)
        pd.testing.assert_frame_equal(generator.sims_on_aws, mock_sims_return)
        assert (
            generator.fs is mock_fs_instance
        )  # Check __init__ received the correct instance
        mock_get_sims_on_aws.assert_called_once()  # Called because sims_on_aws was None
        mock_s3fs.assert_called_once_with(
            anon=True
        )  # Check S3FileSystem was instantiated

    @patch("climakitae.util.generate_gwl_tables.GWLGenerator._set_cesm2_lens")
    def test_get_sims_on_aws(self, mock_cesm2_lens: MagicMock):
        """Test the get_sims_on_aws method for filtering and structuring simulation data."""
        # More complex df to test filtering logic
        data = {
            "zstore": [f"s3://path{i}" for i in range(15)],
            "table_id": ["Amon"] * 14 + ["Omon"],
            "variable_id": ["tas"] * 13 + ["tos", "tas"],
            "experiment_id": [
                "historical",
                "historical",
                "ssp370",
                "ssp585",  # ModelA
                "historical",
                "ssp126",
                "ssp245",  # ModelB
                "historical",
                "historical",  # ModelC
                "ssp370",
                "ssp585",  # ModelD
                "historical",
                "ssp370",  # ModelE
                "historical",  # ModelF (Omon)
                "historical",  # ModelA (Omon)
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
                "ModelE",
                "ModelF",
                "ModelA",
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
                "r1",  # ModelE
                "r1",  # ModelF
                "r1",  # ModelA
            ],
        }
        mock_df = pd.DataFrame(data)
        mock_cesm_catalog = MagicMock()

        # Instantiate Generator, bypassing __init__'s call to the method
        generator = GWLGenerator(
            mock_df, mock_cesm_catalog, sims_on_aws=True
        )  # Pass dummy value
        generator.sims_on_aws = None  # Reset so the actual method runs

        result_df = generator.get_sims_on_aws()

        assert isinstance(result_df, pd.DataFrame)

        # Expected models: A, B, E (C dropped - no SSP, D dropped - no hist, F ignored)
        expected_index = pd.Index(["ModelA", "ModelB", "ModelE"]).sort_values()
        pd.testing.assert_index_equal(result_df.index.sort_values(), expected_index)

        expected_columns = ["historical", "ssp585", "ssp370", "ssp245", "ssp126"]
        pd.testing.assert_index_equal(result_df.columns, pd.Index(expected_columns))

        # Check cell values (lists should be sorted for consistent comparison)
        # ModelA: hist r1, r2; ssp370 r1; ssp585 r1. Keep hist r1 only (present in SSPs).
        assert sorted(result_df.loc["ModelA", "historical"]) == ["r1"]
        assert sorted(result_df.loc["ModelA", "ssp585"]) == ["r1"]
        assert sorted(result_df.loc["ModelA", "ssp370"]) == ["r1"]
        assert result_df.loc["ModelA", "ssp245"] == []  # Check empty list directly
        assert result_df.loc["ModelA", "ssp126"] == []

        # ModelB: hist r1; ssp126 r2; ssp245 r2. No overlap, hist should be empty.
        assert result_df.loc["ModelB", "historical"] == []
        assert result_df.loc["ModelB", "ssp585"] == []
        assert result_df.loc["ModelB", "ssp370"] == []
        assert sorted(result_df.loc["ModelB", "ssp245"]) == ["r2"]
        assert sorted(result_df.loc["ModelB", "ssp126"]) == ["r2"]

        # ModelE: hist r1; ssp370 r1. Keep hist r1.
        assert sorted(result_df.loc["ModelE", "historical"]) == ["r1"]
        assert result_df.loc["ModelE", "ssp585"] == []
        assert sorted(result_df.loc["ModelE", "ssp370"]) == ["r1"]
        assert result_df.loc["ModelE", "ssp245"] == []
        assert result_df.loc["ModelE", "ssp126"] == []

    def test_build_timeseries_loading_and_processing(
        self, mock_generator: GWLGenerator
    ):
        """Test basic functionality of build_timeseries, including data loading and concatenation."""
        model_config: Dict = {
            "variable": "tas",
            "model": TEST_MODEL,
            "ens_mem": "r1i1p1f1",
            "scenarios": ["ssp370", "ssp585"],
        }

        # Mock data arrays returned by make_weighted_timeseries
        historical_time = pd.date_range("1850-01-01", periods=50, freq="MS")
        ssp_time_full = pd.date_range("2015-01-01", periods=1100, freq="MS")
        ssp_time_sliced = ssp_time_full[:1032]  # Expected time coord after isel

        historical_array = xr.DataArray(
            np.random.rand(len(historical_time)),
            coords=[historical_time],
            dims=["time"],
            name="hist_ts",
        ).sortby("time")
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

        # Mock zarr open results
        mock_historical_ds = MagicMock(spec=xr.Dataset)
        mock_historical_ds.__getitem__.return_value = "mock_hist_data_array"

        mock_ssp370_ds_raw = MagicMock(spec=xr.Dataset)  # Before isel/decode
        mock_ssp370_ds_processed = MagicMock(spec=xr.Dataset)  # After isel/decode
        mock_ssp370_ds_raw.isel.return_value = mock_ssp370_ds_processed
        mock_ssp370_ds_processed.__getitem__.return_value = "mock_ssp370_data_array"

        mock_ssp585_ds_raw = MagicMock(spec=xr.Dataset)
        mock_ssp585_ds_processed = MagicMock(spec=xr.Dataset)
        mock_ssp585_ds_raw.isel.return_value = mock_ssp585_ds_processed
        mock_ssp585_ds_processed.__getitem__.return_value = "mock_ssp585_data_array"

        # Mock concatenated results
        mock_concatenated_370 = xr.concat([historical_array, ssp370_array], dim="time")
        mock_concatenated_585 = xr.concat([historical_array, ssp585_array], dim="time")

        with (
            patch(
                "xarray.open_zarr",
                side_effect=[
                    mock_historical_ds,
                    mock_ssp370_ds_raw,
                    mock_ssp585_ds_raw,
                ],
            ) as mock_open_zarr,
            patch(
                "climakitae.util.generate_gwl_tables.make_weighted_timeseries",
                side_effect=[historical_array, ssp370_array, ssp585_array],
            ) as mock_make_ts,
            patch(
                "xarray.decode_cf", side_effect=lambda ds: ds
            ) as mock_decode,  # Pass through the already-mocked dataset
            patch(
                "xarray.concat",
                side_effect=[mock_concatenated_370, mock_concatenated_585],
            ) as mock_concat,
        ):
            result_ds = mock_generator.build_timeseries(model_config)

            # Assertions
            # Check xr.open_zarr calls - uses direct URL with storage_options instead of fs.get_mapper
            assert mock_open_zarr.call_count == 3
            mock_open_zarr.assert_has_calls(
                [
                    call(
                        "s3://bucket/path1",
                        consolidated=True,
                        storage_options={"anon": True},
                    ),  # hist
                    call(
                        "s3://bucket/path3",
                        decode_times=False,
                        consolidated=True,
                        storage_options={"anon": True},
                    ),  # ssp370
                    call(
                        "s3://bucket/path5",
                        decode_times=False,
                        consolidated=True,
                        storage_options={"anon": True},
                    ),  # ssp585
                ],
                any_order=False,
            )

            # Check make_weighted_timeseries calls
            assert mock_make_ts.call_count == 3
            mock_make_ts.assert_has_calls(
                [
                    call("mock_hist_data_array"),
                    call("mock_ssp370_data_array"),
                    call("mock_ssp585_data_array"),
                ]
            )

            # Verify the rest of the test assertions as before
            # Check isel calls (only on raw scenario datasets)
            mock_ssp370_ds_raw.isel.assert_called_once_with(time=slice(0, 1032))
            mock_ssp585_ds_raw.isel.assert_called_once_with(time=slice(0, 1032))

            # Check decode_cf calls (only on processed scenario datasets)
            assert mock_decode.call_count == 2
            mock_decode.assert_has_calls(
                [
                    call(mock_ssp370_ds_processed),
                    call(mock_ssp585_ds_processed),
                ]
            )

            # Check concat calls
            assert mock_concat.call_count == 2
            # Check first concat call (ssp370)
            concat_args_370, concat_kwargs_370 = mock_concat.call_args_list[0]
            assert len(concat_args_370[0]) == 2
            xr.testing.assert_equal(concat_args_370[0][0], historical_array)
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
            assert list(result_ds.data_vars.keys()) == ["ssp370", "ssp585"]
            xr.testing.assert_equal(result_ds["ssp370"], mock_concatenated_370)
            xr.testing.assert_equal(result_ds["ssp585"], mock_concatenated_585)

    def test_build_timeseries_historical_load_error(self, mock_generator: GWLGenerator):
        """Test error handling when loading historical data fails."""
        model_config = {
            "variable": "tas",
            "model": TEST_MODEL,
            "ens_mem": "r1i1p1f1",
            "scenarios": ["ssp370"],
        }
        # Patch xr.open_zarr to raise an exception
        with patch("xarray.open_zarr", side_effect=Exception("Simulated error")):
            result = mock_generator.build_timeseries(model_config)
            assert isinstance(result, xr.Dataset)
            assert not result.data_vars  # Should be empty due to error

    def test_build_timeseries_no_historical_data(self, mock_generator: GWLGenerator):
        """
        Test build_timeseries returns empty Dataset if no historical data is found.
        Tests that lines 236-237 in generate_gwl_tables_unc.py work correctly.
        """
        # Use an existing model but with a non-existent ensemble member
        model_config = {
            "variable": "tas",
            "model": TEST_MODEL,  # This model exists in the mock
            "ens_mem": "non_existent_member",  # But this ensemble member doesn't
            "scenarios": ["ssp370"],
        }

        # Mock the DataFrame.empty property to return True for any filtering
        # that tries to find historical data for this model/ensemble member
        with patch("pandas.DataFrame.empty", new_callable=PropertyMock) as mock_empty:
            # Make any filtered DataFrame appear empty when checking for historical data
            mock_empty.return_value = True

            # Call the method
            result = mock_generator.build_timeseries(model_config)

            # Verify the result is an empty Dataset
            assert isinstance(result, xr.Dataset)
            assert not result.data_vars  # Should be empty due to no historical data

    def test_build_timeseries_scenario_data_error(self, mock_generator: GWLGenerator):
        """
        Test build_timeseries error handling when scenario data fails.
        """
        model_config = {
            "variable": "tas",
            "model": TEST_MODEL,
            "ens_mem": "r1i1p1f1",
            "scenarios": ["ssp370", "ssp585"],
        }

        # Create mock arrays
        historical_array = xr.DataArray(
            np.random.rand(10),
            coords=[pd.date_range("1850-01-01", periods=10, freq="MS")],
            dims=["time"],
            name="historical",
        )

        # Create mock objects for historical and scenario datasets
        mock_historical_ds = MagicMock(spec=xr.Dataset)
        mock_historical_ds.__getitem__.return_value = "mock_historical_data"

        mock_historical_open = MagicMock()
        mock_historical_open.__enter__.return_value = mock_historical_ds

        # This mock will raise an exception for scenario in open_zarr to test the exception handling
        def side_effect_open_zarr(mapper, **kwargs):
            if "path1" in str(mapper):  # historical path
                return mock_historical_open
            else:  # scenario paths
                raise Exception("Simulated scenario data error")

        with (
            patch("xarray.open_zarr", side_effect=side_effect_open_zarr),
            patch(
                "climakitae.util.generate_gwl_tables_unc.make_weighted_timeseries",
                return_value=historical_array,
            ),
        ):
            result = mock_generator.build_timeseries(model_config)

            # Should return a Dataset with no data vars since all scenarios failed
            assert isinstance(result, xr.Dataset)
            assert len(result.data_vars) == 0

    def test_get_gwl_crossing_timestamps(self):
        """
        Test the static get_gwl method.

        Verifies correct timestamp identification for GWL crossing and
        handling of scenarios that never cross (returning NaT).
        """
        index = pd.date_range(start="2000-01-01", periods=5, freq="YE")
        data = {
            "scenario1": pd.Series(
                [0.8, 1.2, 1.6, 1.9, 2.1], index=index
            ),  # Crosses 1.0, 1.5
            "scenario2": pd.Series(
                [0.7, 0.9, 1.1, 1.3, 1.4], index=index
            ),  # Crosses 1.0
        }
        smoothed = pd.DataFrame(data)

        # Test level crossed by both
        result1 = GWLGenerator.get_gwl(smoothed, 1.0)
        assert isinstance(result1, pd.Series)
        assert result1["scenario1"] == pd.Timestamp("2001-12-31")
        assert result1["scenario2"] == pd.Timestamp("2002-12-31")

        # Test level crossed by only one
        result2 = GWLGenerator.get_gwl(smoothed, 1.5)
        assert result2["scenario1"] == pd.Timestamp("2002-12-31")
        assert pd.isna(result2["scenario2"])  # Check for NaT/nan

        # Test level crossed by none
        result3 = GWLGenerator.get_gwl(smoothed, 2.5)
        assert pd.isna(result3["scenario1"])
        assert pd.isna(result3["scenario2"])

    def test_get_gwl_table_for_single_model_and_ensemble(
        self, mock_generator: GWLGenerator
    ):
        """
        Test get_gwl_table_for_single_model_and_ensemble method.

        Covers basic functionality, anomaly calculation, smoothing, GWL detection,
        and exception handling for date selection.
        """
        model_config: Dict = {
            "variable": "tas",
            "model": "TestModel",
            "ens_mem": "r1i1p1f1",
            "scenarios": ["ssp245", "ssp585"],
        }
        reference_period: Dict[str, str] = {
            "start_year": "18500101",
            "end_year": "19001231",
        }  # Use YYYYMMDD

        # Mock data for the Dataset returned by build_timeseries
        time_range = pd.date_range(start="1850-01-01", end="2100-12-31", freq="MS")
        ssp245_data = np.linspace(0, 3, len(time_range))  # Linear warming 0 to 3 deg
        ssp585_data = np.linspace(0, 5, len(time_range))  # Linear warming 0 to 5 deg
        mock_built_ds = xr.Dataset(
            {"ssp245": (["time"], ssp245_data), "ssp585": (["time"], ssp585_data)},
            coords={"time": time_range},
        )

        # Create mock data that correctly matches the structure after xarray to_pandas conversion
        # After to_array() and to_pandas(), the DataFrame would have scenarios as index and time as columns
        mock_time_index = pd.date_range(start="2010-01-01", periods=20, freq="YE")
        mock_smoothed_df = pd.DataFrame(
            index=pd.Index(
                ["ssp245", "ssp585"], name="scenario"
            ),  # Add name="scenario" here
            columns=mock_time_index,
            data=[
                np.linspace(0.5, 2.5, 20),  # ssp245 data
                np.linspace(0.8, 4.0, 20),  # ssp585 data
            ],
        )
        # This creates a DataFrame with scenarios as index and timestamps as columns
        # When transposed in the method, it will have timestamps as index and scenarios as columns

        # Mock the return value of the static get_gwl method
        def mock_gwl_function(df: pd.DataFrame, level: float) -> pd.Series:
            # Simulate realistic return values based on mock_smoothed_df
            timestamps = {
                1.5: {
                    "ssp245": pd.Timestamp("2015-12-31"),
                    "ssp585": pd.Timestamp("2013-12-31"),
                },
                2.0: {
                    "ssp245": pd.Timestamp("2018-12-31"),
                    "ssp585": pd.Timestamp("2015-12-31"),
                },
                2.5: {
                    "ssp245": pd.Timestamp("2020-12-31"),
                    "ssp585": pd.Timestamp("2017-12-31"),
                },
                3.0: {"ssp245": pd.NaT, "ssp585": pd.Timestamp("2019-12-31")},
                4.0: {"ssp245": pd.NaT, "ssp585": pd.Timestamp("2022-12-31")},
            }
            if level in timestamps:
                # Ensure the Series index matches the input DataFrame columns
                return pd.Series(timestamps[level], index=df.columns)
            else:
                return pd.Series([pd.NaT] * len(df.columns), index=df.columns)

        with (
            patch.object(
                mock_generator, "build_timeseries", return_value=mock_built_ds
            ) as mock_build_ts,
            # Patch the static method via the class
            patch.object(
                GWLGenerator, "get_gwl", side_effect=mock_gwl_function
            ) as mock_get_gwl_static,
            # Mock the intermediate conversion to pandas after smoothing
            patch.object(
                xr.DataArray, "to_pandas", return_value=mock_smoothed_df
            ) as mock_to_pandas,
        ):
            # --- Call 1: Normal reference period ---
            gwlevels, final_model = (
                mock_generator.get_gwl_table_for_single_model_and_ensemble(
                    model_config, reference_period
                )
            )

            # Assertions for Call 1
            mock_build_ts.assert_called_once_with(model_config)
            mock_to_pandas.assert_called_once()  # Check conversion after smoothing
            assert mock_get_gwl_static.call_count == len(
                WARMING_LEVELS
            )  # Called for 5 levels

            # Check gwlevels DataFrame structure and content
            assert isinstance(gwlevels, pd.DataFrame)
            expected_gwl_cols = WARMING_LEVELS
            pd.testing.assert_index_equal(
                gwlevels.columns, pd.Index(expected_gwl_cols, dtype=float)
            )
            expected_gwl_index = pd.Index(
                ["ssp245", "ssp585"], name="scenario"
            )  # Should be indexed by scenario
            pd.testing.assert_index_equal(gwlevels.index, expected_gwl_index)
            assert gwlevels.loc["ssp245", 1.5] == pd.Timestamp("2015-12-31")
            assert pd.isna(gwlevels.loc["ssp245", 3.0])
            assert gwlevels.loc["ssp585", 4.0] == pd.Timestamp("2022-12-31")

            # Check final_model DataFrame structure (smoothed anomalies)
            assert isinstance(final_model, pd.DataFrame)
            expected_final_cols = ["r1i1p1f1_ssp245", "r1i1p1f1_ssp585"]
            pd.testing.assert_index_equal(
                final_model.columns, pd.Index(expected_final_cols, name="scenario")
            )

    def test_get_gwl_table_for_single_model_and_ensemble_reference_period_error(
        self, mock_generator: GWLGenerator
    ):
        """
        Test error handling when reference period selection fails with calendar issues.
        This covers lines 413-416 in generate_gwl_tables_unc.py.
        """
        model_config = {
            "variable": "tas",
            "model": TEST_MODEL,
            "ens_mem": "r1i1p1f1",
            "scenarios": ["ssp370"],
        }
        reference_period = TEST_REFERENCE_PERIOD

        # Create mocked dataset with time coordinates that will trigger calendar adjustment
        time_range = pd.date_range(start="1850-01-01", end="2100-12-31", freq="MS")
        mock_ds = xr.Dataset(
            {"ssp370": (["time"], np.random.rand(len(time_range)))},
            coords={"time": time_range},
        )

        # Instead of replacing mock_ds.sel, patch xarray's Dataset.sel method
        with (
            patch.object(mock_generator, "build_timeseries", return_value=mock_ds),
            patch.object(
                xr.Dataset,
                "sel",
                side_effect=[
                    KeyError("Invalid date"),  # First sel fails
                    mock_ds.isel(time=slice(0, 5)),  # Second sel succeeds
                ],
            ),
        ):
            # Call the method
            gwlevels, final_model = (
                mock_generator.get_gwl_table_for_single_model_and_ensemble(
                    model_config, reference_period
                )
            )

            # Verify the first call failed and second worked
            assert not gwlevels.empty  # Should have results since second sel worked

    def test_get_gwl_table_for_single_model_and_ensemble_gwl_error(
        self, mock_generator: GWLGenerator
    ):
        """Test error handling when calculating GWL crossings fails."""
        model_config = {
            "variable": "tas",
            "model": TEST_MODEL,
            "ens_mem": "r1i1p1f1",
            "scenarios": ["ssp370"],
        }
        reference_period = {
            "start_year": "19810101",
            "end_year": "20101231",
        }

        # Mock non-empty smoothed DataFrame
        smoothed_df = pd.DataFrame(
            {"ssp370": [1.0, 1.5, 2.0]}, index=pd.date_range("2000-01-01", periods=3)
        )

        with (
            patch.object(mock_generator, "build_timeseries", return_value=xr.Dataset()),
            patch.object(xr.DataArray, "to_pandas", return_value=smoothed_df),
            patch.object(
                GWLGenerator, "get_gwl", side_effect=Exception("GWL calculation error")
            ),
        ):
            gwlevels, final_model = (
                mock_generator.get_gwl_table_for_single_model_and_ensemble(
                    model_config, reference_period
                )
            )

            # Should return potentially partial gwlevels and empty final_model
            assert isinstance(gwlevels, pd.DataFrame)
            assert isinstance(final_model, pd.DataFrame)
            assert final_model.empty  # final_model should be empty on error

    def test_get_gwl_table(self, mock_generator: GWLGenerator):
        """
        Test the get_gwl_table method for aggregating results across ensemble members.

        Covers:
        1. Basic aggregation with multiple members.
        2. Handling exceptions during single member processing.
        3. Returning empty DataFrames when no members succeed or are found.
        """
        model_config: Dict = {
            "variable": "tas",
            "model": "TestModel",
            "scenarios": ["historical", "ssp585"],
        }
        reference_period: Dict[str, str] = {
            "start_year": "18500101",
            "end_year": "19001231",
        }

        # Mock sims_on_aws to list members for the model
        mock_generator.sims_on_aws = pd.DataFrame(
            {
                "historical": [
                    ["r1", "r2", "r3"]
                ],  # Need historical for consistency check in get_sims_on_aws if called
                "ssp370": [["r1", "r2", "r3"]],
                "ssp585": [["r1", "r2", "r3"]],
                "ssp126": [[]],
                "ssp245": [[]],
            },
            index=pd.Index(["TestModel"], name="source_id"),
        )

        # Ensure the lookup scenario 'ssp370' exists
        assert "historical" in mock_generator.sims_on_aws.columns

        # Mock return values for get_gwl_table_for_single_model_and_ensemble
        # Member r1 (success)
        gwlevels_r1 = pd.DataFrame(
            {
                1.5: [pd.Timestamp("2030-01-01"), pd.Timestamp("2025-01-01")],
                2.0: [pd.Timestamp("2045-01-01"), pd.Timestamp("2040-01-01")],
            },
            index=pd.Index(["historical", "ssp585"], name="scenario"),
        )
        timeseries_r1 = pd.DataFrame(
            {
                "r1_historical": np.linspace(0, 3, 20),
                "r1_ssp585": np.linspace(0, 4, 20),
            },
            index=pd.date_range(start="2000-01-01", periods=20, freq="YE"),
        )

        # Member r2 (failure) - simulated by raising Exception

        # Member r3 (success)
        gwlevels_r3 = pd.DataFrame(
            {
                1.5: [pd.Timestamp("2032-01-01"), pd.Timestamp("2027-01-01")],
                2.0: [pd.Timestamp("2047-01-01"), pd.Timestamp("2042-01-01")],
            },
            index=pd.Index(["historical", "ssp585"], name="scenario"),
        )
        timeseries_r3 = pd.DataFrame(
            {
                "r3_historical": np.linspace(0, 3.2, 20),
                "r3_ssp585": np.linspace(0, 4.2, 20),
            },
            index=pd.date_range(start="2000-01-01", periods=20, freq="YE"),
        )

        # --- Test Case 1: Basic aggregation with one failure ---
        with patch.object(
            mock_generator,
            "get_gwl_table_for_single_model_and_ensemble",
            side_effect=[
                (gwlevels_r1, timeseries_r1),
                Exception("Simulated processing failure for r2"),
                (gwlevels_r3, timeseries_r3),
            ],
        ) as mock_get_single:

            with patch("climakitae.util.generate_gwl_tables.test", False):
                gwlevels_agg, timeseries_agg = mock_generator.get_gwl_table(
                    model_config, reference_period
                )

            # Assert calls to the single-member function
            assert mock_get_single.call_count == 3
            expected_calls = [
                call(
                    {
                        "variable": "tas",
                        "model": "TestModel",
                        "scenarios": ["historical", "ssp585"],
                        "ens_mem": "r1",
                    },
                    reference_period,
                ),
                call(
                    {
                        "variable": "tas",
                        "model": "TestModel",
                        "scenarios": ["historical", "ssp585"],
                        "ens_mem": "r2",
                    },
                    reference_period,
                ),
                call(
                    {
                        "variable": "tas",
                        "model": "TestModel",
                        "scenarios": ["historical", "ssp585"],
                        "ens_mem": "r3",
                    },
                    reference_period,
                ),
            ]
            mock_get_single.assert_has_calls(expected_calls, any_order=True)

            # Check aggregated gwlevels DataFrame
            assert isinstance(gwlevels_agg, pd.DataFrame)
            expected_gwl_index = pd.MultiIndex.from_tuples(
                [
                    ("r1", "historical"),
                    ("r1", "ssp585"),
                    ("r3", "historical"),
                    ("r3", "ssp585"),
                ],
                names=[None, "scenario"],
            )
            pd.testing.assert_index_equal(gwlevels_agg.index, expected_gwl_index)
            expected_gwl_cols = [1.5, 2.0]  # Based on mock data
            pd.testing.assert_index_equal(
                gwlevels_agg.columns, pd.Index(expected_gwl_cols, dtype=float)
            )
            assert gwlevels_agg.loc[("r1", "historical"), 1.5] == pd.Timestamp(
                "2030-01-01"
            )
            assert gwlevels_agg.loc[("r3", "ssp585"), 2.0] == pd.Timestamp("2042-01-01")
            # Check aggregated timeseries DataFrame
            assert isinstance(timeseries_agg, pd.DataFrame)
            # Check that column prefixes are properly added
            assert all(col.startswith("TestModel_") for col in timeseries_agg.columns)

            # Extract expected columns (with TestModel_ prefix)
            expected_cols = [
                "TestModel_r1_historical",
                "TestModel_r1_ssp585",
                "TestModel_r3_historical",
                "TestModel_r3_ssp585",
            ]
            assert sorted(timeseries_agg.columns.tolist()) == sorted(expected_cols)

            # Check values to ensure data is preserved
            assert timeseries_agg["TestModel_r3_historical"].iloc[0] == pytest.approx(
                0.0
            )
            assert timeseries_agg["TestModel_r1_ssp585"].iloc[-1] == pytest.approx(4.0)

        # --- Test Case 2: All members fail ---
        with patch.object(
            mock_generator,
            "get_gwl_table_for_single_model_and_ensemble",
            side_effect=Exception("All failed"),  # All calls raise an exception
        ) as mock_get_single_fail:
            # Set the test global variable to prevent limiting to 10 members
            with patch("climakitae.util.generate_gwl_tables.test", False):
                empty_gwlevels, empty_timeseries = mock_generator.get_gwl_table(
                    model_config, reference_period
                )

            assert mock_get_single_fail.call_count == 3  # Still attempts all members
            assert empty_gwlevels.empty
            assert empty_timeseries.empty

        # --- Test Case 3: No members found initially ---
        # Modify sims_on_aws to have no members for the lookup scenario
        mock_generator.sims_on_aws.loc["TestModel", "historical"] = []

        with patch.object(
            mock_generator, "get_gwl_table_for_single_model_and_ensemble"
        ) as mock_get_single_no_members:
            no_member_gwlevels, no_member_timeseries = mock_generator.get_gwl_table(
                model_config, reference_period
            )

            mock_get_single_no_members.assert_not_called()  # Should not be called if list is empty
            assert no_member_gwlevels.empty
            assert no_member_timeseries.empty

    def test_get_gwl_table_concat_error(self, mock_generator: GWLGenerator):
        """
        Test error handling in get_gwl_table during final concatenation.
        """
        model_config = {
            "variable": "tas",
            "model": TEST_MODEL,
            "scenarios": ["ssp370"],
        }
        reference_period = TEST_REFERENCE_PERIOD

        # Mock sims_on_aws to list members for the model
        mock_generator.sims_on_aws = pd.DataFrame(
            {
                "historical": [["r1", "r2"]],
                "ssp370": [["r1", "r2"]],
            },
            index=pd.Index([TEST_MODEL], name="source_id"),
        )

        # Create mock return values
        gwlevels_r1 = pd.DataFrame(
            {
                1.5: [pd.Timestamp("2030-01-01")],
            },
            index=pd.Index(["ssp370"], name="scenario"),
        )
        timeseries_r1 = pd.DataFrame(
            {"r1_ssp370": np.linspace(0, 3, 10)},
            index=pd.date_range(start="2000-01-01", periods=10, freq="YE"),
        )

        gwlevels_r2 = pd.DataFrame(
            {
                1.5: [pd.Timestamp("2032-01-01")],
            },
            # Different column name will cause concat to fail
            index=pd.Index(["different_scenario"], name="scenario"),
        )
        timeseries_r2 = pd.DataFrame(
            {"r2_ssp370": np.linspace(0, 3.2, 10)},
            index=pd.date_range(start="2000-01-01", periods=10, freq="YE"),
        )

        # Patch pd.concat to simulate a failure during concatenation
        with (
            patch.object(
                mock_generator,
                "get_gwl_table_for_single_model_and_ensemble",
                side_effect=[
                    (gwlevels_r1, timeseries_r1),
                    (gwlevels_r2, timeseries_r2),
                ],
            ),
            patch(
                "pandas.concat",
                side_effect=Exception("Concat error: inconsistent indexes"),
            ),
        ):
            gwlevels_agg, timeseries_agg = mock_generator.get_gwl_table(
                model_config, reference_period
            )

            # Should return empty DataFrames when concatenation fails
            assert isinstance(gwlevels_agg, pd.DataFrame)
            assert gwlevels_agg.empty
            assert isinstance(timeseries_agg, pd.DataFrame)
            assert timeseries_agg.empty

    def test_generate_gwl_file(self, mock_generator: GWLGenerator):
        """Test the generate_gwl_file method for orchestrating and writing results."""
        # Input configuration
        models_list: List[str] = [TEST_MODEL]
        scenarios_list: List[str] = ["historical"]
        ref_periods_list: List[Dict[str, str]] = [TEST_REFERENCE_PERIOD]
        ref_period_str = "1981-2010"  # Expected string format

        # Mock return value for get_gwl_table (aggregated results)
        # GWL table (MultiIndex: member, scenario)
        mock_agg_gwlevels_index = pd.MultiIndex.from_tuples(
            [("r1i1p1f1", "historical")], names=["ensemble_member", "scenario"]
        )
        mock_agg_gwlevels = pd.DataFrame(
            {1.5: [pd.Timestamp("2030-01-01")], 2.0: [pd.Timestamp("2045-01-01")]},
            index=mock_agg_gwlevels_index,
        )
        # Smoothed timeseries (columns are scenarios) - this is ignored by generate_gwl_file
        mock_agg_timeseries = pd.DataFrame(
            {"historical": np.linspace(0, 3, 20)},
            index=pd.date_range(start="2000-01-01", periods=20, freq="YE"),
        )

        # Expected final DataFrame structure to be written to CSV
        expected_final_gwl_index = pd.MultiIndex.from_tuples(
            [(TEST_MODEL, "r1i1p1f1", "historical")],
            names=["GCM", "run", "scenario"],
        )
        expected_final_gwl_df = pd.DataFrame(
            {1.5: [pd.Timestamp("2030-01-01")], 2.0: [pd.Timestamp("2045-01-01")]},
            index=expected_final_gwl_index,
        )
        expected_filename = f"data/gwl_{ref_period_str}ref.csv"

        with (
            patch.object(
                mock_generator,
                "get_gwl_table",
                return_value=(mock_agg_gwlevels, mock_agg_timeseries),
            ) as mock_get_gwl_table,
            patch.object(
                mock_generator,
                "get_table_cesm2",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ) as mock_get_table_cesm2,
            patch(
                "climakitae.util.generate_gwl_tables.write_csv_file"
            ) as mock_write_csv,
        ):
            mock_generator.generate_gwl_file(models_list, ref_periods_list)

            # Assert get_gwl_table was called correctly
            mock_get_gwl_table.assert_called_once()
            call_args, call_kwargs = mock_get_gwl_table.call_args
            expected_model_config_call = {
                "variable": "tas",  # Hardcoded in generate_gwl_file
                "model": TEST_MODEL,
                "scenarios": ["ssp585", "ssp370", "ssp245"],
            }
            assert call_args[0] == expected_model_config_call
            assert call_args[1] == ref_periods_list[0]

            # Assert write_csv_file was called correctly
            assert mock_write_csv.call_count == 2
            write_call_args, write_call_kwargs = mock_write_csv.call_args
            # Check the DataFrame passed to write_csv_file
            pd.testing.assert_frame_equal(write_call_args[0], expected_final_gwl_df)
            # Check the filename passed to write_csv_file
            assert write_call_args[1] == expected_filename

    def test_generate_gwl_file_empty_results(self, mock_generator: GWLGenerator):
        """Test generate_gwl_file when get_gwl_table returns empty DataFrames."""
        models_list = [TEST_MODEL]
        scenarios_list = ["ssp370"]
        ref_periods_list = [TEST_REFERENCE_PERIOD]

        # Mock get_gwl_table to return empty DataFrames
        with (
            patch.object(
                mock_generator,
                "get_gwl_table",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ) as mock_get_gwl_table,
            patch.object(
                mock_generator,
                "get_table_cesm2",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ) as mock_get_table_cesm2,
            patch(
                "climakitae.util.generate_gwl_tables.write_csv_file"
            ) as mock_write_csv,
        ):
            # Should not raise any exceptions, just print message about no data
            mock_generator.generate_gwl_file(models_list, ref_periods_list)
            mock_get_gwl_table.assert_called_once()

    def test_generate_gwl_file_csv_write_error(self, mock_generator: GWLGenerator):
        """Test generate_gwl_file handling CSV writing errors."""
        models_list = [TEST_MODEL]
        scenarios_list = ["ssp370"]
        ref_periods_list = [TEST_REFERENCE_PERIOD]

        # Create valid GWL table to test CSV writing path
        mock_agg_gwlevels_index = pd.MultiIndex.from_tuples(
            [("r1i1p1f1", "ssp370")], names=["ensemble_member", "scenario"]
        )
        mock_agg_gwlevels = pd.DataFrame(
            {1.5: [pd.Timestamp("2030-01-01")]},
            index=mock_agg_gwlevels_index,
        )

        with (
            patch.object(
                mock_generator,
                "get_gwl_table",
                return_value=(mock_agg_gwlevels, pd.DataFrame()),
            ),
            patch.object(
                mock_generator,
                "get_table_cesm2",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch(
                "climakitae.util.generate_gwl_tables.write_csv_file",
                side_effect=Exception("CSV write error"),
            ) as mock_write_csv,
        ):
            # Should catch the exception and print error message
            mock_generator.generate_gwl_file(models_list, ref_periods_list)
            assert mock_write_csv.call_count == 2

    def test_generate_gwl_file_multiple_periods(self, mock_generator: GWLGenerator):
        """Test generate_gwl_file with multiple reference periods."""
        models_list = [TEST_MODEL, "CMCC-ESM2"]
        scenarios_list = ["ssp370"]
        ref_periods_list = [
            TEST_REFERENCE_PERIOD,
            {"start_year": "19510101", "end_year": "19801231"},
        ]

        # Create valid GWL table to test the processing path
        mock_agg_gwlevels_index = pd.MultiIndex.from_tuples(
            [("r1i1p1f1", "ssp370")], names=["ensemble_member", "scenario"]
        )
        mock_agg_gwlevels = pd.DataFrame(
            {1.5: [pd.Timestamp("2030-01-01")]},
            index=mock_agg_gwlevels_index,
        )

        with (
            patch.object(
                mock_generator,
                "get_gwl_table",
                return_value=(mock_agg_gwlevels, pd.DataFrame()),
            ) as mock_get_gwl_table,
            patch.object(
                mock_generator,
                "get_table_cesm2",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch(
                "climakitae.util.generate_gwl_tables.write_csv_file"
            ) as mock_write_csv,
        ):
            mock_generator.generate_gwl_file(models_list, ref_periods_list)

            # Should be called for each model and reference period
            assert mock_get_gwl_table.call_count == 4  # 2 models × 2 periods

    def test_generate_gwl_file_error_handling(self, mock_generator: GWLGenerator):
        """
        Test error handling in generate_gwl_file method.
        This covers line 501 in generate_gwl_tables_unc.py.
        """
        models_list = [TEST_MODEL]
        scenarios_list = ["ssp370"]
        ref_periods_list = [TEST_REFERENCE_PERIOD]

        # Create a model_config dictionary to match what would be created in generate_gwl_file
        model_config = {
            "variable": "tas",
            "model": TEST_MODEL,
            "scenarios": ["ssp585", "ssp370", "ssp245"],
        }

        # Mock get_gwl_table to throw an exception on call
        with (
            patch.object(
                mock_generator,
                "get_gwl_table",
            ) as mock_get_table,
            patch.object(
                mock_generator,
                "get_table_cesm2",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
        ):
            # Configure the mock to raise an exception
            mock_get_table.side_effect = Exception("Error in get_gwl_table")

            # Since the method doesn't catch this exception (which is what we're trying to test)
            # we need to catch it in the test to verify it's thrown
            try:
                mock_generator.generate_gwl_file(models_list, ref_periods_list)
            except Exception as e:
                # Verify that the exception came from our mocked method
                assert str(e) == "Error in get_gwl_table"

            # Verify our mock was called
            mock_get_table.assert_called_once_with(model_config, ref_periods_list[0])

    def test_generate_gwl_file_csv_write_specific_error(
        self, mock_generator: GWLGenerator
    ):
        """
        Test specific CSV writing error in generate_gwl_file.
        This covers lines 532-534 in generate_gwl_tables_unc.py.
        """
        models_list = [TEST_MODEL]
        scenarios_list = ["ssp370"]
        ref_periods_list = [TEST_REFERENCE_PERIOD]

        # Create valid GWL tables for concatenation
        mock_agg_gwlevels_index = pd.MultiIndex.from_tuples(
            [("r1i1p1f1", "ssp370")], names=["ensemble_member", "scenario"]
        )
        mock_agg_gwlevels = pd.DataFrame(
            {1.5: [pd.Timestamp("2030-01-01")]},
            index=mock_agg_gwlevels_index,
        )

        with (
            # Return valid data from get_gwl_table
            patch.object(
                mock_generator,
                "get_gwl_table",
                return_value=(mock_agg_gwlevels, pd.DataFrame()),
            ),
            patch.object(
                mock_generator,
                "get_table_cesm2",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            # But make write_csv_file fail with a specific error
            patch(
                "climakitae.util.generate_gwl_tables.write_csv_file",
                side_effect=IOError("Permission denied"),
            ),
        ):
            # Should catch the IOError and print an error message
            mock_generator.generate_gwl_file(models_list, ref_periods_list)


class TestMainGWLGenerator:
    """Test the main function of the GWLGenerator module."""

    @patch("climakitae.util.generate_gwl_tables.pd.read_csv")
    @patch("intake_esm.core.esm_datastore")
    @patch("climakitae.util.generate_gwl_tables.GWLGenerator")
    def test_main(
        self,
        mock_gwl_generator_class: MagicMock,
        mock_esm_datastore: MagicMock,
        mock_read_csv: MagicMock,
    ):
        """Test the main execution function."""
        actual_reference_period = [
            {"start_year": "18500101", "end_year": "19000101"},
            {"start_year": "19810101", "end_year": "20101231"},
        ]

        mock_df = MagicMock(spec=pd.DataFrame)
        mock_read_csv.return_value = mock_df
        mock_esm_datastore.return_value = {}

        mock_gwl_instance = MagicMock(spec=GWLGenerator)
        mock_gwl_generator_class.return_value = mock_gwl_instance

        main()

        mock_read_csv.assert_called_once_with(
            "https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv"
        ),
        mock_esm_datastore.assert_called_once_with(
            "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json"
        )
        mock_gwl_generator_class.assert_called_once_with(
            mock_df, mock_esm_datastore.return_value
        )
        mock_gwl_instance.generate_gwl_file.assert_called_once_with(
            [],
            actual_reference_period,
        )

    @patch("climakitae.util.generate_gwl_tables.pd.read_csv")
    def test_main_csv_error(self, mock_read_csv: MagicMock):
        """Test the main function when CSV loading fails."""
        mock_read_csv.side_effect = Exception("Network error")

        # The main function should catch the exception and return early
        main()
        mock_read_csv.assert_called_once_with(
            "https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv"
        )

    @patch("climakitae.util.generate_gwl_tables.pd.read_csv")
    @patch("intake_esm.core.esm_datastore")
    @patch("climakitae.util.generate_gwl_tables.GWLGenerator")
    def test_main_gwl_generator_init_error(
        self,
        mock_gwl_generator_class: MagicMock,
        mock_esm_datastore: MagicMock,
        mock_read_csv: MagicMock,
    ):
        """Test the main function when GWLGenerator initialization fails."""
        mock_df = MagicMock(spec=pd.DataFrame)
        mock_read_csv.return_value = mock_df
        mock_esm_datastore.return_value = {}

        # Simulate an error initializing the GWLGenerator
        mock_gwl_generator_class.side_effect = Exception("Initialization error")

        # The main function should catch the exception and return early
        main()
        mock_read_csv.assert_called_once_with(
            "https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv"
        ),
        mock_esm_datastore.assert_called_once_with(
            "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json"
        )
        mock_gwl_generator_class.assert_called_once_with(
            mock_df, mock_esm_datastore.return_value
        )

    @patch("climakitae.util.generate_gwl_tables.pd.read_csv")
    @patch("intake_esm.core.esm_datastore")
    @patch("climakitae.util.generate_gwl_tables.GWLGenerator")
    def test_main_generate_gwl_file_error(
        self,
        mock_gwl_generator_class: MagicMock,
        mock_esm_datastore: MagicMock,
        mock_read_csv: MagicMock,
    ):
        """Test the main function when generate_gwl_file raises an exception."""
        actual_reference_period = [
            {"start_year": "18500101", "end_year": "19000101"},
            {"start_year": "19810101", "end_year": "20101231"},
        ]

        mock_df = MagicMock(spec=pd.DataFrame)
        mock_read_csv.return_value = mock_df
        mock_esm_datastore.return_value = {}

        mock_gwl_instance = MagicMock(spec=GWLGenerator)
        mock_gwl_instance.generate_gwl_file.side_effect = Exception(
            "GWL generation error"
        )
        mock_gwl_generator_class.return_value = mock_gwl_instance

        # The main function should handle this exception internally (no exception raised)
        # Wrap in try/except to catch any unhandled exceptions
        try:
            main()
        except Exception as e:
            pytest.fail(f"main() raised {type(e).__name__} unexpectedly: {e}")

        # Verify the mocks were called correctly
        mock_read_csv.assert_called_once_with(
            "https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv"
        ),
        mock_esm_datastore.assert_called_once_with(
            "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json"
        )
        mock_gwl_generator_class.assert_called_once_with(
            mock_df, mock_esm_datastore.return_value
        )
        mock_gwl_instance.generate_gwl_file.assert_called_once_with(
            [],
            actual_reference_period,
        )

    @patch("climakitae.util.generate_gwl_tables.pd.read_csv")
    @patch("climakitae.util.generate_gwl_tables.GWLGenerator")
    def test_main_catalog_load_error(
        self, mock_gwl_generator_class: MagicMock, mock_read_csv: MagicMock
    ):
        """
        Test edge case error in main function where the catalog fails to load.
        This covers line 673 in generate_gwl_tables_unc.py.
        """
        # When the CSV load raises a specific exception (IOError)
        mock_read_csv.side_effect = IOError("Failed to connect to S3")

        # The function should handle this gracefully
        main()  # No exception should propagate out

        # Verify that GWLGenerator was never constructed since the load failed
        mock_gwl_generator_class.assert_not_called()


# Test for make_weighted_timeseries utility function
def test_make_weighted_timeseries():
    """Test the make_weighted_timeseries utility function."""
    # Create a sample DataArray
    times = pd.date_range("2000-01-01", periods=3)
    lats = np.arange(0, 90, 30)
    lons = np.arange(0, 120, 60)
    data = np.random.rand(len(times), len(lats), len(lons))
    temp_da = xr.DataArray(
        data, coords=[times, lats, lons], dims=["time", "lat", "lon"], name="tas"
    )

    # Expected weights (sqrt(cos(lat))) - unnormalized
    # cos(0)=1, cos(30)=sqrt(3)/2, cos(60)=1/2
    # sqrt(1)=1, sqrt(sqrt(3)/2)~0.93, sqrt(1/2)~0.707
    # Sum ~ 2.637
    # Normalized weights ~ [0.379, 0.353, 0.268]

    result_ts = make_weighted_timeseries(temp_da)

    # Check output type and shape
    assert isinstance(result_ts, xr.DataArray)
    assert result_ts.dims == ("time",)
    assert len(result_ts["time"]) == len(times)

    # Check calculation for the first time step (manual verification)
    expected_t0 = (
        (data[0, 0, :] * np.sqrt(np.cos(np.deg2rad(lats[0])))).mean()
        + (data[0, 1, :] * np.sqrt(np.cos(np.deg2rad(lats[1])))).mean()
        + (data[0, 2, :] * np.sqrt(np.cos(np.deg2rad(lats[2])))).mean()
    )
    # Need to apply normalization from the function
    weights = np.sqrt(np.cos(np.deg2rad(lats)))
    normalized_weights = weights / np.sum(weights)
    expected_t0_normalized = (
        (data[0, 0, :].mean() * normalized_weights[0])
        + (data[0, 1, :].mean() * normalized_weights[1])
        + (data[0, 2, :].mean() * normalized_weights[2])
    )

    np.testing.assert_allclose(result_ts[0].item(), expected_t0_normalized)

    # Test with 'latitude' and 'longitude' coordinate names
    temp_da_renamed = temp_da.rename({"lat": "latitude", "lon": "longitude"})
    result_ts_renamed = make_weighted_timeseries(temp_da_renamed)
    assert result_ts_renamed.dims == ("time",)
    np.testing.assert_allclose(result_ts_renamed[0].item(), expected_t0_normalized)

    # Test raises ValueError if coords are missing
    temp_da_no_coords = xr.DataArray(data, dims=["time", "y", "x"])
    with pytest.raises(
        ValueError,
        match="Input DataArray must have latitude and longitude coordinates.",
    ):
        make_weighted_timeseries(temp_da_no_coords)
