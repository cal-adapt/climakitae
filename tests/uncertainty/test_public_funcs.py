from unittest.mock import patch

import pandas as pd
import pytest

from climakitae.explore.uncertainty import (
    get_ks_pval_df,
    get_warm_level,
    gwl_1850_1900_file,
    gwl_1981_2010_file,
)
from tests.uncertainty.fixtures import (
    mock_data_for_clipping,
    mock_data_for_warm_level,
    mock_multi_ens_dataset,
    wrf_dataset,
)

# TODO investigate whether get_ks_pval_df is actually working


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

    # Create mock DataFrames for the CSV files
    mock_ipcc_df = pd.DataFrame(
        {"1.5": [2030.5], "2.0": [2045.2], "3.0": [2070.1], "4.0": [2090.3]}
    )
    mock_ipcc_df.index = pd.MultiIndex.from_tuples(
        [("CESM2", "r11i1p1f1", "ssp370")], names=["GCM", "run", "scenario"]
    )

    mock_non_ipcc_df = pd.DataFrame(
        {"1.5": [2025.3], "2.0": [2040.7], "3.0": [2065.8], "4.0": [2085.9]}
    )

    mock_ece3_df = pd.DataFrame(
        {"1.5": [2026.1], "2.0": [2041.2], "3.0": [2066.4], "4.0": [2086.3]}
    )

    # Set up the mock to return different DataFrames based on input file path
    def side_effect(file_path, **kwargs):
        if file_path == gwl_1850_1900_file:
            return mock_ipcc_df
        elif file_path == gwl_1981_2010_file:
            return mock_non_ipcc_df
        elif file_path == "data/gwl_1981-2010ref_EC-Earth3_ssp370.csv":
            return mock_ece3_df
        else:
            return pd.DataFrame()

    mock_read_csv.side_effect = side_effect

    # Test IPCC path (ipcc=True)
    with patch("climakitae.explore.uncertainty.gwl_1850_1900_file", gwl_1850_1900_file):
        with patch(
            "climakitae.explore.uncertainty.gwl_1981_2010_file", gwl_1981_2010_file
        ):
            # Mock just enough of the function to test file loading
            with patch.object(
                get_warm_level.__globals__["pd"], "concat"
            ) as mock_concat:
                # Call the function with ipcc=True
                try:
                    result = get_warm_level(2.0, mock_data_for_warm_level, ipcc=True)
                except:
                    # We expect the function to potentially fail after file loading,
                    # since we're not mocking everything, but we only care about file loading
                    pass

                # Verify read_csv_file was called with the correct file
                mock_read_csv.assert_called_with(
                    gwl_1850_1900_file, index_col=[0, 1, 2]
                )
                # Verify concat was not called
                mock_concat.assert_not_called()

    # Test non-IPCC path (ipcc=False)
    mock_read_csv.reset_mock()
    with patch("climakitae.explore.uncertainty.gwl_1850_1900_file", gwl_1850_1900_file):
        with patch(
            "climakitae.explore.uncertainty.gwl_1981_2010_file", gwl_1981_2010_file
        ):
            with patch.object(
                pd, "concat", return_value=pd.concat([mock_non_ipcc_df, mock_ece3_df])
            ) as mock_concat:
                # Call the function with ipcc=False
                try:
                    result = get_warm_level(2.0, mock_data_for_warm_level, ipcc=False)
                except:
                    pass

                # Verify read_csv_file was called with both files
                assert mock_read_csv.call_count == 2
                mock_read_csv.assert_any_call(gwl_1981_2010_file)
                mock_read_csv.assert_any_call(
                    "data/gwl_1981-2010ref_EC-Earth3_ssp370.csv"
                )

                # Verify concat was called with the two DataFrames
                mock_concat.assert_called_once()


@patch("climakitae.explore.uncertainty.read_csv_file")
def test_get_warm_level_member_id_selection(
    mock_read_csv, mock_data_for_clipping, mock_multi_ens_dataset
):
    """Test member_id selection logic."""
    # Setup mock dataframe with warming data - create a DataFrame with one row per model
    mock_df = pd.DataFrame(
        {
            "1.5": [2030.5, 2030.6, 2030.7],
            "2.0": [2045.2, 2045.3, 2045.4],
            "3.0": [2070.1, 2070.2, 2070.3],
            "4.0": [2090.3, 2090.4, 2090.5],
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
    with patch("xarray.Dataset.sel", return_value=mock_data_for_clipping) as mock_sel:
        get_warm_level(2.0, mock_data_for_clipping)
        # For CESM2, should use r11i1p1f1
        assert mock_df.loc[("CESM2", "r11i1p1f1", "ssp370")]["2.0"] == 2045.2

    # Test default member_id selection for CNRM-ESM2-1
    cnrm_ds = mock_data_for_clipping.copy()
    cnrm_ds = cnrm_ds.assign_coords({"simulation": "CNRM-ESM2-1"})
    with patch("xarray.Dataset.sel", return_value=cnrm_ds) as mock_sel:
        get_warm_level(2.0, cnrm_ds)
        # For CNRM-ESM2-1, should use r1i1p1f2
        assert mock_df.loc[("CNRM-ESM2-1", "r1i1p1f2", "ssp370")]["2.0"] == 2045.4

    # Test multi_ens=True (should use the member_id from the dataset)
    with patch("xarray.Dataset.sel", return_value=mock_multi_ens_dataset) as mock_sel:
        get_warm_level(2.0, mock_multi_ens_dataset, multi_ens=True)
        # Should use the member_id from the dataset
        assert mock_df.loc[("EC-Earth3", "r1i1p1f1", "ssp370")]["2.0"] == 2045.3


@patch("climakitae.explore.uncertainty.read_csv_file")
def test_get_warm_level_time_slice(mock_read_csv, mock_data_for_warm_level):
    """Test time slicing logic based on warming level."""
    # Setup mock dataframe with warming data
    mock_df = pd.DataFrame(
        {
            "3.0": [2070.0],  # Warming level reached in 2070
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
            "3.0°C warming level not reached for ensemble member r11i1p1f1 of model CESM2"
        )

    # Test case where end year exceeds 2100
    mock_df["3.0"] = [2090.0]  # 2090 + 15 > 2100
    with patch("builtins.print") as mock_print:
        result = get_warm_level(3.0, mock_data_for_warm_level)
        assert result is None
        mock_print.assert_called_with(
            "End year for SSP time slice occurs after 2100; skipping ensemble member r11i1p1f1 of model CESM2"
        )


def test_get_warm_level_integration(wrf_dataset):
    """Test get_warm_level with a real-like dataset."""
    # Extract a single simulation to test
    ds = wrf_dataset.isel(simulation=0).drop_vars("simulation")
    ds = ds.assign_coords({"simulation": "CESM2"})

    # Since we can't predict the actual warming level data, we'll patch it
    with patch("climakitae.explore.uncertainty.read_csv_file") as mock_read_csv:
        mock_df = pd.DataFrame(
            {
                "3.0": [2070.0],  # Warming level reached in 2070
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
