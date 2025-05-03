### Test Module for `warming_levels.py`

import pytest
import xarray as xr

from climakitae.core.paths import GWL_1850_1900_FILE
from climakitae.util.utils import read_csv_file
from climakitae.util.warming_levels import _calculate_warming_level, _get_sliced_data

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
        _calculate_warming_level(da_wrong, gwl_times, 2, range(1, 13), 15)


def test_get_sliced_data_empty_output(
    test_dataarray_time_2030_2035_wrf_3km_hourly_temp,
):
    """
    Verify that `_get_sliced_data` returns an empty DataArray when no `center_time` is found.

    `WRF_FGOALS-g3_r1i1p1f1` for `SSP 3-7.0` does not reach 4.0 warming, so we will use it to see if `_get_sliced_data` will just generate an empty DataArray for it.
        Context for this behavior in `_get_sliced_data`: an empty DataArray is generated for this simulation in `_get_sliced_data` because `_get_sliced_data`
        is called in a groupby call, requiring all objects it is called upon to return the same shape.
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
