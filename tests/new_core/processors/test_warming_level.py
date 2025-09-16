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

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.new_core.processors.warming_level import WarmingLevel


@pytest.fixture
def processor():
    """Fixture for an empty WarmingLevel processor."""
    return WarmingLevel(value={})


class TestWarmingLevelProcessorInitialization:
    """Tests for the initialization of the WarmingLevel DataProcessor."""

    def test_default_initialization(self, processor):
        """Test default initialization."""
        assert processor.warming_levels == [2.0]
        assert processor.warming_level_window == 15
        assert isinstance(processor.warming_level_times, pd.DataFrame)
        assert isinstance(processor.warming_level_times_idx, pd.DataFrame)

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
    pass


class TestWarmingLevelExecute:
    pass
