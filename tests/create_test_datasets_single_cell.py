"""
Create test datasets for a single cell of hourly data, for all variations between:
    1. WL
        1.1 Different WL seasons
    2. Time
    3. LOCA
    4. WRF
"""

import numpy as np
from climakitae.core.data_interface import get_data
from climakitae.util.utils import get_closest_gridcell

##### ----- SET GLOBAL PARAMS -------

variable = "Air Temperature at 2m"
# downscaling_method='Dynamical',
resolution = "3 km"
timescale = "hourly"
# approach='Warming Level',
scenario = ["SSP 3-7.0 -- Business as Usual"]
units = "degF"
# warming_level=[2],
area_subset = "none"
latitude = (35.43424 - 0.02, 35.43424 + 0.02)
longitude = (-119.05524 - 0.02, -119.05524 + 0.02)
cached_area = ["entire domain"]
area_average = "No"
# time_slice=None,
# warming_level_window=15,
# warming_level_months=list(range(1,13))

##### --------------------------------

##### ----- HELPER FUNCTIONS ---------


def _get_params(
    downscaling_method,
    approach="Time",
    warming_level=None,
    time_slice=None,
    warming_level_window=15,
    warming_level_months=list(range(1, 13)),
):
    params = {
        "variable": variable,
        "downscaling_method": downscaling_method,
        "resolution": resolution,
        "timescale": timescale,
        "approach": approach,
        "scenario": scenario,
        "units": units,
        "warming_level": warming_level,
        "area_subset": area_subset,
        "latitude": latitude,
        "longitude": longitude,
        "time_slice": time_slice if approach == "Time" else None,
        "warming_level_window": warming_level_window,
        "warming_level_months": warming_level_months,
    }
    return params


_get_approach_str = lambda approach: "wl" if approach == "Warming Level" else "time"
_get_downscaling_str = lambda downscaling: (
    "wrf" if downscaling == "Dynamical" else "loca"
)


def _get_duration_str(approach, warming_level, time_slice):
    if approach == "Time":
        return f"{time_slice[0]}_{time_slice[1]}"
    elif approach == "Warming Level":
        return str(warming_level).replace(".", "")


def _get_filename(approach, downscaling, warming_level, time_slice):
    """
    Gets the filename for a given dataset with a given set of params.

    Example structures looks like:
        - WL WRF:     'test_data/test_dataset_wl_20_wrf_3km_hourly'
        - Time LOCA:  'test_data/test_dataset_time_2030_2035_loca_3km_hourly'

    """
    return (
        f"test_data/test_dataset_"
        f"{_get_approach_str(approach)}_"
        f"{_get_duration_str(approach, warming_level, time_slice)}_"
        f"{_get_downscaling_str(downscaling)}_"
        f"{resolution}_{timescale}"
    )


def _get_data_and_export(
    downscaling_method,
    approach,
    time_slice,
    warming_level,
    warming_level_months=list(range(1, 13)),
):
    """Helper function wrapper around `_get_params`, `_get_filename`, `get_data`, `get_closest_gridcell`, and exporting to netCDF."""
    params = _get_params(
        downscaling_method=downscaling_method,
        approach=approach,
        time_slice=time_slice,
        warming_level_months=warming_level_months,
    )
    filename = _get_filename(approach, downscaling_method, warming_level, time_slice)

    # Get data with validated params
    da = get_data(**params)
    da = get_closest_gridcell(
        da, lat=np.mean(latitude), lon=np.mean(longitude)
    )  # Make sure just one gridcell is pulled

    # Export data
    da.to_netcdf(filename)

    return params, filename


### --------------------------------

### 1. Time, Dynamical approach

# Set specific params
downscaling_method = "Dynamical"
approach = "Time"
time_slice = (2030, 2035)
warming_level = None

da = _get_data_and_export(downscaling_method, approach, time_slice, warming_level)


### 2. Time, Statistical approach

downscaling_method = "Statistical"
da = _get_data_and_export(downscaling_method, approach, time_slice, warming_level)

### 3. WL, Dynamical approach

downscaling_method = "Dynamical"
approach = "Warming Level"
warming_level = 2.0

da = _get_data_and_export(downscaling_method, approach, time_slice, warming_level)

### 4. WL, Statistical approach

downscaling_method = "Statistical"
warming_level_months = list(range(1, 13))
da = _get_data_and_export(downscaling_method, approach, time_slice, warming_level)

#### 5. WL, Dynamical, summer approach

warming_level_months = [6, 7, 8]
da = _get_data_and_export(
    downscaling_method, approach, time_slice, warming_level, warming_level_months
)
