"""
Units tests for the `warming_level` processor.

This module contains comprehensive unit tests for the `warming_level` processor,
which is responsible for adjusting climate data based on specified global warming levels.
These tests ensure that the processor behaves as expected under various scenarios,
including different warming levels, time windows, and edge cases.

The tests cover:
- Basic functionality: Verifying that the processor correctly adjusts data for standard warming levels.
- Boundary conditions: Ensuring correct behavior at the edges of the time windows.
- Error handling: Confirming that appropriate exceptions are raised for invalid inputs.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.new_core.processors.warming_level import WarmingLevel


@pytest.fixture
def processor():
    """Fixture for an empty WarmingLevel processor with mock warming level time DataFrames."""
    wl_processor = WarmingLevel(value={})
    wl_time_data = [
        [
            "GCM",
            "run",
            "scenario",
            "0.8",
            "1.0",
            "1.2",
        ],
        [
            "ACCESS-CM2",
            "r3i1p1f1",
            "ssp585",
            "2005-09-16 00:00:00",
            "2012-01-16 12:00:00",
            "2017-05-16 12:00:00",
        ],
        [
            "ACCESS-ESM1-5",
            "r3i1p1f1",
            "ssp585",
            "2010-11-16 00:00:00",
            "2016-09-16 00:00:00",
            "2021-06-16 00:00:00",
        ],
        [
            "CNRM-ESM2-1",
            "r1i1p1f2",
            "ssp585",
            "2006-08-16 12:00:00",
            "2014-07-16 12:00:00",
            "2020-05-16 12:00:00",
        ],
    ]
    wl_processor.warming_level_times = pd.DataFrame(
        wl_time_data[1:], columns=wl_time_data[0]
    ).set_index(["GCM", "run", "scenario"])

    wl_time_idx_data = [
        ["time", "ACCESS-CM2_r3i1p1f1_ssp585", "ACCESS-CM2_r3i1p1f1_ssp370"],
        ["2025-1", 1.5, 1.6],
        ["2035-1", 1.7, 1.8],
        ["2045-1", 1.9, 2.0],
    ]
    wl_processor.warming_level_times_idx = pd.DataFrame(
        wl_time_idx_data[1:], columns=wl_time_idx_data[0]
    ).set_index("time")

    yield wl_processor


@pytest.fixture
def full_processor():
    """Fixture for a full WarmingLevel processor with default warming level time DataFrames."""
    wl_processor = WarmingLevel(value={})
    yield wl_processor


class TestWarmingLevelProcessorInitialization:
    """Tests for the initialization of the WarmingLevel DataProcessor."""

    def test_default_initialization(self, processor):
        """Test default initialization."""
        assert processor.warming_levels == [2.0]
        assert processor.warming_level_window == 15
        assert isinstance(processor.warming_level_times, pd.DataFrame)
        assert isinstance(processor.warming_level_times_idx, pd.DataFrame)

    @pytest.mark.parametrize("invalid_type", [None, 123, "string", 5.5, []])
    def test_invalid_initialization(self, invalid_type):
        """Test invalid initialization with wrong types for value parameter."""
        with pytest.raises(
            TypeError, match="Expected dictionary for warming level configuration"
        ):
            WarmingLevel(value=invalid_type)

    def test_custom_initialization(self):
        """Test custom initialization."""
        processor = WarmingLevel(
            value={"warming_levels": [1.5, 3.0], "warming_level_window": 10}
        )
        assert processor.warming_levels == [1.5, 3.0]
        assert processor.warming_level_window == 10


class TestWarmingLevelUpdateContext:
    """Tests for the update_context method of WarmingLevel DataProcessor."""

    def test_update_context_adds_warming_levels(self, processor):
        """Test that update_context adds warming level information to context."""
        context = {_NEW_ATTRS_KEY: {}}
        updated_context = processor.update_context(context)
        assert (
            f"Process '{processor.name}'"
            in updated_context[_NEW_ATTRS_KEY][processor.name]
        )
        assert "warming_levels" in updated_context[_NEW_ATTRS_KEY][processor.name]
        assert "warming_level_window" in updated_context[_NEW_ATTRS_KEY][processor.name]
        assert (
            str(processor.warming_levels)
            in updated_context[_NEW_ATTRS_KEY][processor.name]
        )
        assert (
            str(processor.warming_level_window)
            in updated_context[_NEW_ATTRS_KEY][processor.name]
        )


class TestWarmingReformatMemberIds:
    """Tests for the reformat_member_ids method of WarmingLevel DataProcessor."""

    def test_reformat_member_ids_empty(self, processor):
        """Test that reformat_member_ids returns empty dict for empty input."""
        ret = processor.reformat_member_ids({})
        assert ret == {}

    def test_reformat_member_ids_no_member_id(self, processor):
        """Test that reformat_member_ids returns input dict if no member_id dimension."""
        no_member_id_da = {
            "ssp245": xr.DataArray(
                [1, 2, 3],
                dims="time",
                coords=[("time", pd.date_range("2000-01-01", periods=3))],
                attrs={"units": "K"},
            )
        }
        with pytest.warns(
            UserWarning, match="No member_id found in data for key ssp245"
        ):
            ret = processor.reformat_member_ids(no_member_id_da)
            assert ret == no_member_id_da

    def test_reformat_member_ids_multiple_no_member_id(self, processor):
        """Test that reformat_member_ids returns input dict if no member_id dimension."""
        no_member_id_da = xr.DataArray(
            [1, 2, 3],
            dims="time",
            coords=[("time", pd.date_range("2000-01-01", periods=3))],
            attrs={"units": "K"},
        )
        no_member_id_dict = {
            "ssp245": no_member_id_da,
            "ssp585": no_member_id_da,
        }

        with pytest.warns(UserWarning) as record:
            ret = processor.reformat_member_ids(no_member_id_dict)
            assert ret == no_member_id_dict
            messages = [str(w.message) for w in record]
            assert len(messages) == 2
            assert "No member_id found in data for key ssp245" in messages[0]
            assert "No member_id found in data for key ssp585" in messages[1]

    def test_reformat_member_ids_correct(self, processor):
        """Test that reformat_member_ids correctly reformats member IDs in an xr.Dataset with one variable."""
        member_id_ds = {
            "ssp245": xr.Dataset(
                {
                    "var1": xr.DataArray(
                        [1, 2],
                        dims="member_id",
                        coords=[("member_id", [1, 2])],
                        attrs={"units": "K"},
                    ),
                }
            )
        }
        ret = processor.reformat_member_ids(member_id_ds)
        assert len(ret) == 2
        for key in ret:
            assert key.startswith("ssp245.")
            assert key in ret
            assert type(ret[key]) == xr.Dataset
            assert "member_id" not in ret[key]["var1"].dims
            # Check that original attributes are preserved
            assert member_id_ds["ssp245"]["var1"].attrs == ret[key]["var1"].attrs


class TestWarmingLevelExtendTimeDomain:
    """Tests for the extend_time_domain method of WarmingLevel DataProcessor."""

    def test_extend_time_domain_empty(self, processor):
        """Test that extend_time_domain returns empty dict for empty input."""
        ret = processor.extend_time_domain({})
        assert ret == {}

    def test_extend_time_domain_no_ssp_key(self, processor):
        """Test that extend_time_domain returns input dict if no ssp key."""
        no_ssp_key_da = {"this is not a key with S S P (historical)": xr.DataArray()}
        ret = processor.extend_time_domain(no_ssp_key_da)
        assert ret == {}

    def test_extend_time_domain_one_ssp_no_historical(self, processor):
        """Test that extend_time_domain correctly does not extend if no historical key."""
        time_domain_dict = {
            "ssp245": xr.DataArray([1, 2, 3], dims="time"),
            # "historical": xr.DataArray([4, 5, 6], dim="time"),
        }
        with pytest.warns(
            UserWarning, match="No historical data found for ssp245 with key historical"
        ):
            ret = processor.extend_time_domain(time_domain_dict)
            assert ret == {}

    def test_extend_time_domain_one_ssp_with_historical(self, processor):
        """Test that extend_time_domain correctly extends time domain if historical key exists."""
        time_domain_dict = {
            "ssp245": xr.DataArray([1, 2, 3], dims="time", attrs={"units": "K"}),
            "historical": xr.DataArray(
                [4, 5, 6], dims="time", attrs={"units": "not K"}
            ),
        }
        ret = processor.extend_time_domain(time_domain_dict)
        assert len(ret) == 1
        assert "ssp245" in ret
        assert isinstance(ret["ssp245"], xr.DataArray)
        assert ret["ssp245"].sizes["time"] == 6
        assert ret["ssp245"].attrs == time_domain_dict["ssp245"].attrs

    def test_extend_time_domain_no_time_dim(self, processor):
        """Test that extend_time_domain does not extend if no time dimension."""
        time_domain_dict = {
            "ssp245": xr.DataArray([1, 2, 3], dims="timez", attrs={"units": "K"}),
            "historical": xr.DataArray(
                [4, 5, 6], dims="timez", attrs={"units": "not K"}
            ),
        }
        with pytest.warns(
            UserWarning,
            match="No time dimension found in data for key ssp245 or historical",
        ):
            ret = processor.extend_time_domain(time_domain_dict)
            assert ret == {}


class TestWarmingLevelGetCenterYears:
    """Tests for the get_center_years method of WarmingLevel DataProcessor."""

    def test_no_valid_member_id(self, processor):
        """Test that get_center_years returns empty dict if all member_ids are None."""
        member_ids = [None, None]
        keys = ["key1", "key2"]
        ret = processor.get_center_years(member_ids, keys)
        assert ret == {}

    def test_valid_member_id_and_keys(self, processor):
        """Test that get_center_years returns np.nan for keys not in warming level table at a common WL."""
        processor.warming_levels = [1.0]
        member_ids = ["r3i1p1f1", "r1i1p1f2"]
        keys = [
            "WRF.UCLA.ACCESS-CM2.ssp585.mon.d01.r3i1p1f1",
            "WRF.UCLA.CNRM-ESM2-1.ssp585.mon.d01.r1i1p1f2",
        ]
        ret = processor.get_center_years(member_ids, keys)
        assert len(ret) == 2
        assert ret["WRF.UCLA.ACCESS-CM2.ssp585.mon.d01.r3i1p1f1"] == [2012]
        assert ret["WRF.UCLA.CNRM-ESM2-1.ssp585.mon.d01.r1i1p1f2"] == [2014]

    def test_custom_wl_key_not_found(self, processor):
        """Test that get_center_years returns np.nan for keys not in warming level table at a common WL."""
        processor.warming_levels = [1.2345]
        member_ids = ["r1i1p1f1"]
        keys = [
            "WRF.UCLA.MIROC6.ssp370.mon.d01.r1i1p1f1",  # This key is not in the wl table
        ]
        with pytest.warns(
            UserWarning, match="Warming level table does not contain data"
        ):
            ret = processor.get_center_years(member_ids, keys)
            assert len(ret) == 1
            assert ret["WRF.UCLA.MIROC6.ssp370.mon.d01.r1i1p1f1"] == [np.nan]

    def test_valid_custom_wl(self, processor):
        """Test that get_center_years returns correct year for keys in warming level table at a custom WL."""
        processor.warming_levels = [1.78987]
        member_ids = ["r3i1p1f1"]
        keys = [
            "WRF.UCLA.ACCESS-CM2.ssp585.mon.d01.r3i1p1f1",  # This key is in the wl table
        ]
        ret = processor.get_center_years(member_ids, keys)
        assert len(ret) == 1
        assert ret["WRF.UCLA.ACCESS-CM2.ssp585.mon.d01.r3i1p1f1"] == ["2045-1"]

    def test_invalid_custom_wl(self, processor):
        """Test that get_center_years returns np.nan for keys in warming level table at a custom WL not found."""
        processor.warming_levels = [2.5678]
        member_ids = ["r3i1p1f1"]
        keys = [
            "WRF.UCLA.ACCESS-CM2.ssp585.mon.d01.r3i1p1f1",  # This key is in the wl table
        ]
        with pytest.warns(
            UserWarning,
            match=f"\n\nNo warming level data found for ACCESS-CM2_r3i1p1f1_ssp585 at 2.5678C. \nPlease pick a warming level less than 1.2C.",
        ):
            ret = processor.get_center_years(member_ids, keys)
            assert len(ret) == 1
            assert ret["WRF.UCLA.ACCESS-CM2.ssp585.mon.d01.r3i1p1f1"] == [np.nan]


class TestWarmingLevelExecute:
    """Tests for the execute method of WarmingLevel DataProcessor."""

    @patch("climakitae.new_core.processors.warming_level.read_csv_file")
    def test_empty_warming_level_times_error(self, mock_read_csv_file, processor):
        """Test that execute raises error if warming_level_times is empty and cannot be found."""
        processor.warming_level_times = None
        mock_read_csv_file.side_effect = FileNotFoundError("File not found")

        with pytest.raises(
            RuntimeError, match="Failed to load warming level times table"
        ):
            processor.execute(result=None, context={})

    def test_execute_updates_context(self, request, full_processor):
        """Test that execute updates context with warming level information."""
        test_result = request.getfixturevalue("test_dataarray_dict")
        context = {}
        _ = full_processor.execute(result=test_result, context=context)
        assert full_processor.name in context[_NEW_ATTRS_KEY][full_processor.name]
        assert str(full_processor.value) in context[_NEW_ATTRS_KEY][full_processor.name]

    def test_execute_skips_missing_center_years(self, request, full_processor):
        """Test that execute skips keys with no valid center years and warns."""
        data = request.getfixturevalue("test_dataarray_dict")
        ssp_key = "WRF.UCLA.EC-Earth3.ssp370.day.d03.r1i1p1f1"

        with (
            patch.object(
                full_processor,
                "get_center_years",
                # return_value={hist_key: [np.nan], ssp_key: [np.nan]},
                return_value={ssp_key: [np.nan]},
            ),
            pytest.warns(
                UserWarning, match=f"No warming level data found for {ssp_key}"
            ),
            pytest.warns(UserWarning, match=f"No valid slices found for {ssp_key}."),
        ):
            assert full_processor.execute(data, context={}) == {}

    def test_execute_skips_warming_level(self, request, full_processor):
        """Test that execute skips warming levels if there is no warming level found for a certain key."""
        data = request.getfixturevalue("test_dataarray_dict")
        # Editing warming levels to include one that will be skipped
        full_processor.warming_levels = [1.5, 5.8, 2.0]
        with (
            pytest.warns(
                UserWarning,
                match=f"No warming level data found",
            ),
        ):
            ret = full_processor.execute(data, context={})
            for key in ret:
                assert len(ret[key].warming_level) == 2
                assert 5.8 not in ret[key].warming_level.values

    def test_execute_dims_correct(self, request, full_processor):
        """Test that execute returns a dict with expected keys and types."""
        test_result = request.getfixturevalue("test_dataarray_dict")
        ret = full_processor.execute(result=test_result, context={})
        for key in ret:
            assert isinstance(ret[key], xr.Dataset)
            assert "warming_level" in ret[key].dims
            assert "time_delta" in ret[key].dims
            assert "centered_year" in ret[key].coords

    def test_execute_years_correct(self, request, full_processor):
        """Test that execute manipulates the data to have correct dims and years."""
        test_result = request.getfixturevalue("test_dataarray_dict")
        test_key = "WRF.UCLA.EC-Earth3.ssp370.day.d03"
        ret = full_processor.execute(result=test_result, context={})
        ret_key = "WRF.UCLA.EC-Earth3.ssp370.day.d03.r1i1p1f1"

        # Check that the warming_level coordinate matches the processor's warming_levels
        assert (
            ret[ret_key].warming_level.values == full_processor.warming_levels
        ).all()
        # Check the length of the time_delta dimension
        first_year = str(test_result[test_key].isel(time=0).time.dt.year.item())
        # Find the number of elements in the first year of `ret[key]`
        timesteps_per_year = (
            test_result[test_key].sel(time=slice(first_year, first_year)).time.size
        )
        assert (
            len(ret[ret_key].time_delta)
            == timesteps_per_year * full_processor.warming_level_window * 2
        )
        assert isinstance(ret[ret_key].centered_year.item(), int)
        # Check that the centered_year is within expected range
        assert 1981 <= ret[ret_key].centered_year.item() <= 2100
