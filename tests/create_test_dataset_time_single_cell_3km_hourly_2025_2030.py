from climakitae.core.data_interface import DataParameters

selections = DataParameters()
selections.variable = "Air Temperature at 2m"
selections.scenario_historical = ["Historical Climate"]
selections.scenario_ssp = ["SSP 3-7.0 -- Business as Usual"]
selections.downscaling_method = "Dynamical"
selections.units = "degF"
selections.timescale = "hourly"
selections.resolution = "3 km"
selections.area_subset = "lat/lon"
selections.latitude = (35.43424 - 0.02, 35.43424 + 0.02)
selections.longitude = (-119.05524 - 0.02, -119.05524 + 0.02)
selections.time_slice = (2030, 2035)

# Compute data
da = selections.retrieve()

# Export data
da.to_netcdf("test_data/test_dataset_time_single_cell_3km_hourly_2030_2035.nc")
