"""Test CMIOpt class."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.explore.uncertainty import CmipOpt


def test_cmipopt_init():
    """Test CMIOpt class initialization."""
    # Test with default parameters
    cmiopt = CmipOpt()
    assert cmiopt.variable == "tas"
    assert cmiopt.area_subset == "states"
    assert cmiopt.location == "California"
    assert cmiopt.timescale == "monthly"
    assert cmiopt.area_average == True


def test_cmipopt_init_with_params():
    """Test CMIOpt class initialization with parameters."""
    # Test with custom parameters
    cmiopt = CmipOpt(
        variable="pr",
        area_subset="countries",
        location="USA",
        timescale="annual",
        area_average=False,
    )
    assert cmiopt.variable == "pr"
    assert cmiopt.area_subset == "countries"
    assert cmiopt.location == "USA"
    assert cmiopt.timescale == "annual"
    assert cmiopt.area_average == False


@pytest.fixture
def sample_dataset():
    """Create a sample dataset with multiple variables."""
    # Create a simple dataset with tas, pr, and other variables
    ds = xr.Dataset(
        data_vars={
            "tas": (["time", "lat", "lon"], np.random.rand(5, 3, 4)),
            "pr": (["time", "lat", "lon"], np.random.rand(5, 3, 4)),
            "extra_var": (["time", "lat", "lon"], np.random.rand(5, 3, 4)),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=5),
            "lat": np.linspace(34, 42, 3),
            "lon": np.linspace(-124, -118, 4),
        },
    )
    return ds


def test_cmip_clip_basic(sample_dataset):
    """Test basic functionality of _cmip_clip."""
    # Create CmipOpt with default settings (variable='tas', area_average=False)
    copt = CmipOpt(variable="tas", area_average=False)

    # Create a filtered dataset that should be returned by _clip_region
    filtered_dataset = sample_dataset.drop_vars(["pr", "extra_var"])

    # Mock the _clip_region function to return the filtered dataset
    with patch(
        "climakitae.explore.uncertainty._clip_region", return_value=filtered_dataset
    ) as mock_clip:
        result = copt._cmip_clip(sample_dataset)

        # Check that _clip_region was called with correct arguments
        mock_clip.assert_called_once_with(
            filtered_dataset,
            copt.area_subset,
            copt.location,
        )

        # Check that only 'tas' remains in the result
        assert len(result.data_vars) == 1
        assert "tas" in result.data_vars
        assert "pr" not in result.data_vars
        assert "extra_var" not in result.data_vars


def test_cmip_clip_area_average(sample_dataset):
    """Test _cmip_clip with area_average=True."""
    # Create CmipOpt with area_average=True
    copt = CmipOpt(variable="tas", area_average=True)

    # Mock both _clip_region and _area_wgt_average functions
    with patch(
        "climakitae.explore.uncertainty._clip_region", return_value=sample_dataset
    ) as mock_clip:
        with patch(
            "climakitae.explore.uncertainty._area_wgt_average",
            return_value=sample_dataset.isel(lat=0, lon=0),
        ) as mock_avg:
            result = copt._cmip_clip(sample_dataset)

            # Check that _area_wgt_average was called
            mock_avg.assert_called_once()

            # Check result shape (should be reduced after area averaging)
            assert "lat" not in result.dims
            assert "lon" not in result.dims


def test_cmip_clip_precipitation(sample_dataset):
    """Test _cmip_clip with variable='pr' to check precipitation conversion."""
    # Create CmipOpt with variable='pr'
    copt = CmipOpt(variable="pr", area_average=False)

    # Create a filtered dataset that only contains 'pr'
    filtered_dataset = sample_dataset.drop_vars(["tas", "extra_var"])

    # Mock _clip_region and _precip_flux_to_total functions
    with patch(
        "climakitae.explore.uncertainty._clip_region", return_value=filtered_dataset
    ) as mock_clip:
        with patch(
            "climakitae.explore.uncertainty._precip_flux_to_total",
            return_value=filtered_dataset,
        ) as mock_precip:
            result = copt._cmip_clip(sample_dataset)

            # Check that _precip_flux_to_total was called
            mock_precip.assert_called_once()

            # Check that only 'pr' remains in the result
            assert len(result.data_vars) == 1
            assert "pr" in result.data_vars
            assert "tas" not in result.data_vars
            assert "extra_var" not in result.data_vars


def test_cmip_clip_full_workflow(sample_dataset):
    """Test full workflow of _cmip_clip with precipitation variable and area averaging."""
    # Create CmipOpt with variable='pr' and area_average=True
    copt = CmipOpt(variable="pr", area_average=True)

    # Define mock return values
    clipped_ds = sample_dataset.copy()
    precip_ds = sample_dataset.copy()
    area_avg_ds = sample_dataset.isel(lat=0, lon=0).copy()

    # Mock all dependent functions in sequence
    with patch(
        "climakitae.explore.uncertainty._clip_region", return_value=clipped_ds
    ) as mock_clip:
        with patch(
            "climakitae.explore.uncertainty._precip_flux_to_total",
            return_value=precip_ds,
        ) as mock_precip:
            with patch(
                "climakitae.explore.uncertainty._area_wgt_average",
                return_value=area_avg_ds,
            ) as mock_avg:
                result = copt._cmip_clip(sample_dataset)

                # Check that all functions were called in correct order
                mock_clip.assert_called_once()
                mock_precip.assert_called_once_with(clipped_ds)
                mock_avg.assert_called_once_with(precip_ds)

                # Check result matches expected output
                assert result == area_avg_ds


def test_cmip_clip_no_matching_variable(sample_dataset):
    """Test _cmip_clip with a variable not present in the dataset."""
    # Create CmipOpt with variable not in the dataset
    copt = CmipOpt(variable="nonexistent_var", area_average=False)

    # Mock _clip_region function
    with patch("climakitae.explore.uncertainty._clip_region") as mock_clip:
        # The dataset should be empty after dropping all variables
        mock_clip.return_value = sample_dataset.drop_vars(["tas", "pr", "extra_var"])

        result = copt._cmip_clip(sample_dataset)

        # Check that _clip_region was called with an empty dataset
        mock_clip.assert_called_once()
        assert len(result.data_vars) == 0
