from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from climakitae.core.data_interface import DataParameters
from climakitae.core.data_load import (_check_valid_unit_selection,
                                       _get_data_attributes, _get_eff_temp,
                                       _get_fosberg_fire_index,
                                       _get_hourly_dewpoint, _get_hourly_rh,
                                       _get_monthly_daily_dewpoint,
                                       _get_noaa_heat_index,
                                       _override_unit_defaults)


# Used as a return value for mocking data download
def mock_data():
    da = xr.DataArray(np.ones((1)),coords={"time":np.zeros((1))})
    da.attrs = {"units":""}
    da.name = "data"
    return da

class TestDataLoad:

    def test__override_unit_defaults(self):
        da = xr.DataArray()
        da.attrs = {}

        result1 = _override_unit_defaults(da,"pr")
        assert result1.attrs == da.attrs

        result2 = _override_unit_defaults(da,"huss")
        assert result2.attrs == {"units": "kg/kg"}

        result3 = _override_unit_defaults(da,"rsds")
        assert result3.attrs == {"units": "W/m2"}

    def test__check_valid_unit_selection(self):
        selections = DataParameters()
        result = _check_valid_unit_selection(selections)
        assert result is None

        # Edit a unit to cause exception
        selections.variable_options_df["unit"].iloc[0] = "C"
        with pytest.raises(ValueError):
            result = _check_valid_unit_selection(selections)

    def test__get_data_attributes(self):
        selections = DataParameters()
        result = _get_data_attributes(selections)

        # Check that dict with correct keys in returned.
        kwlist = ["variable_id", "extended_description", "units", "data_type", "resolution", "frequency", "location_subset", "approach", "downscaling_method"]
        for item in kwlist:
            assert item in result


class TestDerivedDataLoad:

    @patch("climakitae.core.data_load._get_data_one_var",return_value=mock_data())
    def test__get_hourly_rh(self,mock_get_data_one_var):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        selections = DataParameters()
        result = _get_hourly_rh(selections)

        assert isinstance(result,xr.core.dataarray.DataArray)

    @patch("climakitae.core.data_load._get_data_one_var",return_value=mock_data())
    def test__get_monthly_daily_dewpoint(self,mock_get_data_one_var):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        selections = DataParameters()
        result = _get_monthly_daily_dewpoint(selections)

        assert isinstance(result,xr.core.dataarray.DataArray)
        assert result.name == "dew_point_derived"

    @patch("climakitae.core.data_load._get_data_one_var",return_value=mock_data())
    def test__get_hourly_dewpoint(self,mock_get_data_one_var):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        selections = DataParameters()
        result = _get_hourly_dewpoint(selections)

        assert isinstance(result,xr.core.dataarray.DataArray)
        assert result.name == "dew_point_derived"

    @patch("climakitae.core.data_load._get_data_one_var",return_value=mock_data())
    def test__get_noaa_heat_index(self,mock_get_data_one_var):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        selections = DataParameters()
        result = _get_noaa_heat_index(selections)

        assert isinstance(result,xr.core.dataarray.DataArray)

    @patch("climakitae.core.data_load._get_data_one_var",return_value=mock_data())
    def test__get_eff_temp(self,mock_get_data_one_var):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        selections = DataParameters()
        result = _get_eff_temp(selections)

        assert isinstance(result,xr.core.dataarray.DataArray)

    @patch("climakitae.core.data_load._get_data_one_var",return_value=mock_data())
    def test__get_fosberg_fire_index(self,mock_get_data_one_var):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        selections = DataParameters()
        result = _get_fosberg_fire_index(selections)

        assert isinstance(result,xr.core.dataarray.DataArray)
        assert result.name == "Fosberg Fire Weather Index"