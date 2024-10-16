"""Create test dataset for Warming Level data"""

import numpy as np
import climakitae as ck
import xarray as xr
from climakitae.explore import warming_levels

wl = warming_levels()
wl.wl_params.timescale = "hourly"
wl.wl_params.resolution = "45 km"
wl.wl_params.downscaling_method = "Dynamical"
wl.wl_params.variable_type = "Variable"
wl.wl_params.variable = "Air Temperature at 2m"
wl.wl_params.warming_levels = ["1.5", "2.0", "3.0", "4.0"]
wl.wl_params.units = "degF"
wl.wl_params.resolution = "3 km"
wl.wl_params.anom = "No"
wl.wl_params.months = np.arange(1, 13)
wl.wl_params.area_subset = "CA counties"
wl.wl_params.cached_area = ["Alameda County"]

# Compute warming levels
wl.calculate()

ds = xr.concat(wl.sliced_data.values(), dim="warming_level")
ds.to_netcdf("test_data/test_dataset_WL_Alamedacounty_45km_hourly.nc.nc")
