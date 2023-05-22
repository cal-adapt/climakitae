"""Backend functions for retrieving and subsetting data from the AE catalog"""

import xarray as xr
import dask
import rioxarray
import intake
import numpy as np
import pandas as pd
import pkg_resources
import psutil
import warnings
import fnmatch
from ast import literal_eval
from shapely.geometry import box

from xclim.core.calendar import convert_calendar
from xclim.sdba.adjustment import QuantileDeltaMapping
from .catalog_convert import (
    _downscaling_method_to_activity_id,
    _resolution_to_gridlabel,
    _timescale_to_table_id,
    _scenario_to_experiment_id,
)
from .unit_conversions import _convert_units
from .utils import _readable_bytes, get_closest_gridcell
from .derive_variables import (
    _compute_relative_humidity,
    _compute_wind_mag,
    _compute_dewpointtemp,
    _compute_specific_humidity,
)

# Set options
xr.set_options(keep_attrs=True)
dask.config.set({"array.slicing.split_large_chunks": True})

# Import stations names and coordinates file
stations = pkg_resources.resource_filename("climakitae", "data/hadisd_stations.csv")
stations_df = pd.read_csv(stations)


# ============================ Read data into memory ================================


def _compute(xr_da):
    """Read data into memory"""

    # Check if data is already loaded into memory
    if xr_da.chunks is None:
        print("Your data is already loaded into memory")
        return xr_da

    # Get memory information
    avail_mem = psutil.virtual_memory().available  # Available system memory
    xr_data_nbytes = xr_da.nbytes  # Memory of data

    # If it will cause the system to have less than 256MB after loading the data, do not allow the compute to proceed.
    if avail_mem - xr_data_nbytes < 268435456:
        print("Available memory: {0}".format(_readable_bytes(avail_mem)))
        print("Total memory of input data: {0}".format(_readable_bytes(xr_data_nbytes)))
        raise MemoryError("Your input dataset is too large to read into memory!")

    else:
        print(
            "Reading {0} of data into memory... ".format(
                _readable_bytes(xr_data_nbytes)
            ),
            end="",
        )
        da_computed = xr_da.compute()
        print("complete!")
        return da_computed  # Load data into memory and return


# ============================ Helper functions ================================


def _get_as_shapely(selections):
    """
    Takes the location data, and turns it into a
    shapely object. Just doing polygons for now. Later other point/station data
    will be available too.

    Args:
        selections (_DataSelector): Data settings (variable, unit, timescale, etc)

    Returns:
        shapely_geom (shapely.geometry)

    """
    # Box is formed using the following shape:
    #   shapely.geometry.box(minx, miny, maxx, maxy)
    shapely_geom = box(
        selections.longitude[0],  # minx
        selections.latitude[0],  # miny
        selections.longitude[1],  # maxx
        selections.latitude[1],  # maxy
    )
    return shapely_geom


# ============= Main functions used in data reading/processing =================


def _get_cat_subset(selections, cat):
    """For an input set of data selections, get the catalog subset.

    Args:
        selections (_DataSelector): object holding user's selections
        cat (intake_esm.core.esm_datastore): catalog

    Returns:
        cat_subset (intake_esm.core.esm_datastore): catalog subset

    """

    scenario_selections = selections.scenario_ssp + selections.scenario_historical

    # Get catalog keys
    # Convert user-friendly names to catalog names (i.e. "45 km" to "d01")
    activity_id = [
        _downscaling_method_to_activity_id(dm) for dm in selections.downscaling_method
    ]
    table_id = _timescale_to_table_id(selections.timescale)
    grid_label = _resolution_to_gridlabel(selections.resolution)
    experiment_id = [_scenario_to_experiment_id(x) for x in scenario_selections]
    source_id = selections.simulation
    variable_id = selections.variable_id

    cat_subset = cat.search(
        activity_id=activity_id,
        table_id=table_id,
        grid_label=grid_label,
        variable_id=variable_id,
        experiment_id=experiment_id,
        source_id=source_id,
    )

    # Get just data that's on the LOCA grid
    # This will include LOCA data and WRF data on the LOCA native grid
    # Both datasets are tagged with UCSD as the institution_id, so we can use "UCSD" to further subset the catalog data
    if "Statistical" in selections.downscaling_method:
        cat_subset = cat_subset.search(institution_id="UCSD")
    # If only dynamical is selected, we need to remove UCSD from the WRF query
    else:
        wrf_on_native_grid = [
            institution
            for institution in cat.df.institution_id.unique()
            if institution != "UCSD"
        ]
        cat_subset = cat_subset.search(institution_id=wrf_on_native_grid)

    return cat_subset


def _get_area_subset(area_subset, cached_area, selections):
    """Get geometry to perform area subsetting with.

    Args:
        selections (_DataSelector): object holding user's selections

    Returns:
        ds_region (shapely.geometry): geometry to use for subsetting

    """

    def set_subarea(boundary_dataset):
        return boundary_dataset[boundary_dataset.index == shape_index].iloc[0].geometry

    if area_subset == "lat/lon":
        geom = _get_as_shapely(selections)
        if not geom.is_valid:
            raise ValueError(
                "Please go back to 'select' and choose" + " a valid lat/lon range."
            )
        ds_region = [geom]
    elif area_subset != "none":
        shape_index = int(selections._geography_choose[area_subset][cached_area])
        if area_subset == "states":
            shape = set_subarea(selections._geographies._us_states)
        elif area_subset == "CA counties":
            shape = set_subarea(selections._geographies._ca_counties)
        elif area_subset == "CA watersheds":
            shape = set_subarea(selections._geographies._ca_watersheds)
        elif area_subset == "CA Electric Load Serving Entities (IOU & POU)":
            shape = set_subarea(selections._geographies._ca_utilities)
        elif area_subset == "CA Electricity Demand Forecast Zones":
            shape = set_subarea(selections._geographies._ca_forecast_zones)
        elif area_subset == "CA Electric Balancing Authority Areas":
            shape = set_subarea(selections._geographies._ca_electric_balancing_areas)
        ds_region = [shape]
    else:
        ds_region = None
    return ds_region


def _process_and_concat(selections, dsets, cat_subset):
    """Process all data; merge all datasets into one.

    Args:
        selections (DataLoaders): object holding user's selections
        dsets (dictionary): dictionary of zarrs from catalog, with each key
            being its name and each item the zarr store
        cat_subset (intake_esm.core.esm_datastore): catalog subset

    Returns:
        da (xr.DataArray): output data

    """
    da_list = []
    scenario_list = selections.scenario_historical + selections.scenario_ssp
    append_historical = False

    if True in ["SSP" in one for one in selections.scenario_ssp]:
        if "Historical Climate" in selections.scenario_historical:
            if selections.time_slice[0] <= 2015:
                # Historical climate will be appended to the SSP data
                append_historical = True
            scenario_list.remove("Historical Climate")
        if "Historical Reconstruction (ERA5-WRF)" in selections.scenario_historical:
            # We are not allowing users to select historical reconstruction data and SSP data at the same time,
            # due to the memory restrictions at the moment
            scenario_list.remove("Historical Reconstruction (ERA5-WRF)")
    for scenario in scenario_list:
        scen_name = _scenario_to_experiment_id(scenario)
        sim_list = []
        for downscaling_method in selections.downscaling_method:
            activity_id = _downscaling_method_to_activity_id(downscaling_method)
            for simulation in selections.simulation:
                if append_historical:
                    # Reset name
                    scen_name = "Historical + " + scenario

                    # Get filenames
                    try:
                        historical_filename = fnmatch.filter(
                            list(dsets.keys()),
                            "*{0}.*{1}.*historical*".format(activity_id, simulation),
                        )[0]
                        if (  # Need to get CESM2 data if ensmean is selected for ssp2-4.5 or ssp5-8.5
                            simulation == "ensmean"
                        ) and (
                            scenario
                            in [
                                "SSP 2-4.5 -- Middle of the Road",
                                "SSP 5-8.5 -- Burn it All",
                            ]
                        ):
                            ssp_filename = fnmatch.filter(
                                list(dsets.keys()),
                                "*{0}.*CESM2.*{1}*".format(
                                    activity_id, _scenario_to_experiment_id(scenario)
                                ),
                            )[0]
                        else:
                            ssp_filename = fnmatch.filter(
                                list(dsets.keys()),
                                "*{0}.*{1}.*{2}*".format(
                                    activity_id,
                                    simulation,
                                    _scenario_to_experiment_id(scenario),
                                ),
                            )[0]
                    except:  # Some simulation + ssp options are not available. Just continue with the loop if no filename is found
                        continue
                    # Grab data
                    historical_data = dsets[historical_filename]
                    ssp_data = dsets[ssp_filename]

                    # Concatenate data. Rename scenario attribute
                    # This will append the SSP data to the historical data, both with the same simulation
                    ds_sim = xr.concat(
                        [historical_data, ssp_data],
                        dim="time",
                        coords="minimal",
                        compat="override",
                        join="inner",
                    )

                else:
                    try:
                        if (  # Need to get CESM2 data if ensmean is selected for ssp2-4.5 or ssp5-8.5
                            simulation == "ensmean"
                        ) and (
                            scenario
                            in [
                                "SSP 2-4.5 -- Middle of the Road",
                                "SSP 5-8.5 -- Burn it All",
                            ]
                        ):
                            filename = fnmatch.filter(
                                list(dsets.keys()),
                                "*{0}.*CESM2.*{1}*".format(
                                    activity_id, _scenario_to_experiment_id(scenario)
                                ),
                            )[0]
                        else:
                            filename = fnmatch.filter(
                                list(dsets.keys()),
                                "*{0}.*{1}.*{2}*".format(
                                    activity_id,
                                    simulation,
                                    _scenario_to_experiment_id(scenario),
                                ),
                            )[0]
                        ds_sim = dsets[filename]
                    except:
                        continue
                # Get the name of the variable id
                var_id = list(ds_sim.data_vars)[0]

                # Convert units
                da_sim = ds_sim[var_id]
                if var_id == "huss":
                    # Units for LOCA specific humidity are set to 1
                    # Reset to kg/kg so they can be converted if neccessary to g/kg
                    da_sim.attrs["units"] = "kg/kg"
                da_sim = _convert_units(da=da_sim, selected_units=selections.units)
                for member_id in da_sim.member_id.values:
                    da_sim_member_id = da_sim.sel(member_id=member_id).drop("member_id")
                    da_sim_member_id["simulation"] = "{0}_{1}_{2}".format(
                        activity_id, simulation, member_id
                    )
                    sim_list.append(da_sim_member_id)

        # Raise an appropriate error if no data found
        if len(sim_list) == 0:
            raise ValueError(
                "You've encountered a bug in the source code. The data selections you've set do not correspond to a valid data option in the Analytics Engine catalog."
            )

        # Concatenate along simulation dimension
        else:
            da = xr.concat(
                sim_list, dim="simulation", coords="minimal", compat="broadcast_equals"
            )
            da = da.assign_coords({"scenario": scen_name})
            da_list.append(da)

    da_final = xr.concat(
        da_list, dim="scenario", coords="minimal", compat="broadcast_equals"
    )
    return da_final


# ============ Read from catalog function used by ck.Application ===============


def _get_data_one_var(selections, cat):
    """Get data for one variable"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Silence warning if empty dataset returned

        # Get catalog subset for a set of user selections
        cat_subset = _get_cat_subset(selections=selections, cat=cat)

        # Read data from AWS
        data_dict = cat_subset.to_dataset_dict(
            zarr_kwargs={"consolidated": True},
            storage_options={"anon": True},
            progressbar=False,
        )

    # If SSP 2-4.5 or SSP 5-8.5 are selected, along with ensmean as the simulation,
    # We want to return the single available CESM2 model
    if ("ensmean" in selections.simulation) and (
        {"SSP 2-4.5 -- Middle of the Road", "SSP 5-8.5 -- Burn it All"}.intersection(
            set(selections.scenario_ssp)
        )
    ):
        cat_subset2 = cat.search(
            activity_id=[
                _downscaling_method_to_activity_id(dm)
                for dm in selections.downscaling_method
            ],
            table_id=_timescale_to_table_id(selections.timescale),
            grid_label=_resolution_to_gridlabel(selections.resolution),
            variable_id=selections.variable_id,
            experiment_id=[
                _scenario_to_experiment_id(x) for x in selections.scenario_ssp
            ],
            source_id=["CESM2"],
        )
        data_dict2 = cat_subset2.to_dataset_dict(
            zarr_kwargs={"consolidated": True},
            storage_options={"anon": True},
            progressbar=False,
        )
        data_dict = {**data_dict, **data_dict2}

    # Perform subsetting operations
    for dname, dset in data_dict.items():
        # Break down name into component parts
        (
            activity_id,
            institution_id,
            source_id,
            experiment_id,
            table_id,
            grid_label,
        ) = dname.split(".")

        # Add simulation and downscaling method as coordinates
        dset = dset.assign_coords({"simulation": source_id})

        # Time slice
        dset = dset.sel(
            time=slice(str(selections.time_slice[0]), str(selections.time_slice[1]))
        )

        # Perform area subsetting

        # You need to retrieve the entire domain because the shapefiles will cut out the ocean grid cells
        # But the some station's closest gridcells are the ocean!
        if selections.data_type == "Station":
            area_subset = "none"
            cached_area = "entire domain"
        else:
            area_subset = selections.area_subset
            cached_area = selections.cached_area
        ds_region = _get_area_subset(area_subset, cached_area, selections)
        if ds_region is not None:  # Perform subsetting
            if selections.downscaling_method == ["Dynamical"]:
                # all_touched: If True, all pixels touched by geometries will be burned in.  If false, only pixel whose center is within the polygon or that are selected by Bresenham's line algorithm will be burned in.
                # drop: If True, drop the data outside of the extent of the mask geoemtries. Otherwise, it will return the same raster with the data masked.
                dset = dset.rio.clip(
                    geometries=ds_region, crs=4326, drop=True, all_touched=False
                )
            else:
                # LOCA does not have x,y coordinates. rioxarray hates this
                # rioxarray will raise this error: MissingSpatialDimensionError: x dimension not found. 'rio.set_spatial_dims()' or using 'rename()' to change the dimension name to 'x' can address this.
                # Therefore I need to rename the lat, lon dimensions to x,y, and then reset them after clipping.
                # I also need to write a CRS to the dataset
                dset = dset.rename({"lon": "x", "lat": "y"})
                dset = dset.rio.write_crs("EPSG:4326")
                dset = dset.rio.clip(geometries=ds_region, crs=4326, drop=True)
                dset = dset.rename({"x": "lon", "y": "lat"}).drop("spatial_ref")

        # Perform area averaging
        if selections.area_average == "Yes":
            weights = np.cos(np.deg2rad(dset.lat))
            if set(["x", "y"]).issubset(set(dset.dims)):
                # WRF data has x,y
                dset = dset.weighted(weights).mean("x").mean("y")
            elif set(["lat", "lon"]).issubset(set(dset.dims)):
                # LOCA data has x,y
                dset = dset.weighted(weights).mean("lat").mean("lon")

        # Update dataset in dictionary
        data_dict.update({dname: dset})

    # Merge individual Datasets into one DataArray object.
    da = _process_and_concat(
        selections=selections, dsets=data_dict, cat_subset=cat_subset
    )

    # Assign data type attribute
    da_new_attrs = {  # Add descriptive attributes to DataArray
        "variable_id": ", ".join(
            selections.variable_id
        ),  # Convert list to comma separated string
        "extended_description": selections.extended_description,
        "units": selections.units,
        "data_type": selections.data_type,
        "resolution": selections.resolution,
        "frequency": selections.timescale,
        "location_subset": selections.cached_area,
        "institution": institution_id,
        "data_history": "Data has been accessed through the Cal-Adapt: Analytics Engine using the open-source climakitae python package.",
    }
    if "grid_mapping" in da.attrs:
        da_new_attrs = da_new_attrs | {"grid_mapping": da.attrs["grid_mapping"]}
    da.attrs = da_new_attrs
    da.name = selections.variable
    return da


def _read_catalog_from_select(selections, cat, loop=False):
    """The primary and first data loading method, called by
    core.Application.retrieve, it returns a DataArray (which can be quite large)
    containing everything requested by the user (which is stored in 'selections').

    Args:
        selections (DataLoaders): object holding user's selections
        cat (intake_esm.core.esm_datastore): catalog

    Returns:
        da (xr.DataArray): output data
    """

    if (selections.scenario_ssp != []) and (
        "Historical Reconstruction" in selections.scenario_historical
    ):
        raise ValueError(
            "Historical Reconstruction data is not available with SSP data. Please modify your selections and try again."
        )

    # Raise error if no scenarios are selected
    scenario_selections = selections.scenario_ssp + selections.scenario_historical
    if scenario_selections == []:
        raise ValueError("Please select as least one dataset.")

    # Raise error if station data selected, but no station is selected
    if (selections.data_type == "Station") and (
        selections.station in [[], ["No stations available at this location"]]
    ):
        raise ValueError(
            "Please select at least one weather station, or retrieve gridded data."
        )

    # For station data, need to expand time slice to ensure the historical period is included
    # At the end, the data will be cut back down to the user's original selection
    if selections.data_type == "Station":
        original_time_slice = selections.time_slice  # Preserve original user selections
        original_scenario_historical = selections.scenario_historical.copy()
        if "Historical Climate" not in selections.scenario_historical:
            selections.scenario_historical.append("Historical Climate")
        obs_data_bounds = (
            1980,
            2010,
        )  # Bounds of the observational data used in bias-correction
        if original_time_slice[0] > obs_data_bounds[0]:
            selections.time_slice = (obs_data_bounds[0], original_time_slice[1])
        if original_time_slice[1] < obs_data_bounds[1]:
            selections.time_slice = (selections.time_slice[0], obs_data_bounds[1])

    # Deal with derived variables
    orig_var_id_selection = selections.variable_id[0]
    orig_variable_selection = selections.variable
    if "_derived" in orig_var_id_selection:
        if "wind_speed_derived" in orig_var_id_selection:
            # Load u10 data
            selections.variable_id = ["u10"]
            u10_da = _get_data_one_var(selections, cat)

            # Load v10 data
            selections.variable_id = ["v10"]
            v10_da = _get_data_one_var(selections, cat)

            # Derive wind magnitude
            da = _compute_wind_mag(u10=u10_da, v10=v10_da)

        else:
            # Load temperature data
            selections.variable_id = ["t2"]
            t2_da = _get_data_one_var(selections, cat)

            # Load mixing ratio data
            selections.variable_id = ["q2"]
            q2_da = _get_data_one_var(selections, cat)

            # Load pressure data
            selections.variable_id = ["psfc"]
            pressure_da = _get_data_one_var(selections, cat)

            # Derive relative humidity
            rh_da = _compute_relative_humidity(
                pressure=pressure_da, temperature=t2_da, mixing_ratio=q2_da
            )

            if "rh_derived" in orig_var_id_selection:
                da = rh_da

            else:
                # Need to figure out how to silence divide by zero runtime warning
                # Derive dew point temperature
                dew_pnt_da = _compute_dewpointtemp(temperature=t2_da, rel_hum=rh_da)

                if "dew_point_derived" in orig_var_id_selection:
                    da = dew_pnt_da

                elif "q2_derived" in orig_var_id_selection:
                    # Derive specific humidity
                    da = _compute_specific_humidity(
                        tdps=dew_pnt_da, pressure=pressure_da
                    )

                else:
                    raise ValueError(
                        "You've encountered a bug. No data available for selected derived variable."
                    )

        selections.variable_id = [orig_var_id_selection]
        da.attrs["variable_id"] = orig_var_id_selection  # Reset variable ID attribute
        da.name = orig_variable_selection  # Set name of DataArray

    else:
        da = _get_data_one_var(selections, cat)

    if selections.data_type == "Station":
        if loop:
            # print("Retrieving station data using a for loop")
            da = _station_loop(selection, da, stations_df, original_time_slice)
        else:
            # print("Retrieving station data using xr.apply")
            da = _station_apply(selections, da, stations_df, original_time_slice)
        # Reset original selections
        if "Historical Climate" not in original_scenario_historical:
            selections.scenario_historical.remove("Historical Climate")
            da["scenario"] = [x.split("Historical + ")[1] for x in da.scenario.values]
        selections.time_slice = original_time_slice

    return da


# USE XR APPLY TO GET BIAS CORRECTED DATA TO STATION


def _station_apply(selections, da, stations_df, original_time_slice):
    # Grab zarr data
    station_subset = stations_df.loc[stations_df["station"].isin(selections.station)]
    filepaths = [
        "s3://cadcat/tmp/hadisd/HadISD_{}.zarr".format(s_id)
        for s_id in station_subset["station id"]
    ]

    station_ds = xr.open_mfdataset(
        filepaths,
        preprocess=_preprocess_hadisd,
        engine="zarr",
        consolidated=False,
        parallel=True,
        backend_kwargs=dict(storage_options={"anon": True}),
    )
    apply_output = station_ds.apply(
        _get_bias_corrected_closest_gridcell,
        keep_attrs=False,
        gridded_da=da,
        time_slice=original_time_slice,
    )
    return apply_output


def _get_bias_corrected_closest_gridcell(station_da, gridded_da, time_slice):
    """Get the closest gridcell to a weather station.
    Bias correct the data using historical station data
    """
    # Get the closest gridcell to the station
    station_lat, station_lon = station_da.attrs["coordinates"]
    gridded_da_closest_gridcell = get_closest_gridcell(
        gridded_da, station_lat, station_lon, print_coords=False
    )

    # Droop any coordinates in the output dataset that are not also dimensions
    # This makes merging all the stations together easier and drops superfluous coordinates
    gridded_da_closest_gridcell = gridded_da_closest_gridcell.drop(
        [
            i
            for i in gridded_da_closest_gridcell.coords
            if i not in gridded_da_closest_gridcell.dims
        ]
    )

    # Bias correct the model data using the station data
    # Cut the output data back to the user's selected time slice
    bias_corrected = _bias_correct_model_data(
        station_da, gridded_da_closest_gridcell, time_slice
    )

    # Add descriptive coordinates to the bias corrected data
    bias_corrected.attrs["station_coordinates"] = station_da.attrs[
        "coordinates"
    ]  # Coordinates of station
    bias_corrected.attrs["station_elevation"] = station_da.attrs[
        "elevation"
    ]  # Elevation of station
    return bias_corrected


def _bias_correct_model_data(
    obs_da, gridded_da, time_slice, nquantiles=20, group="time.dayofyear", kind="+"
):
    """Bias correct model data using observational station data
    Converts units of the station data to whatever the input model data's units are
    Converts calendars of both datasets to a no leap calendar
    Time slices the data
    Performs bias correction

    Args:
        obs_da (xr.DataArray): station data, preprocessed with the function _preprocess_hadisd
        gridded_da (xr.DataArray): input model data
        time_slice (tuple): temporal slice to cut gridded_da to, after bias correction

    """
    # Convert units to whatever the gridded data units are
    obs_da = _convert_units(obs_da, gridded_da.units)
    # Rechunk data. Cannot be chunked along time dimension
    # Error raised by xclim: ValueError: Multiple chunks along the main adjustment dimension time is not supported.
    gridded_da = gridded_da.chunk(dict(time=-1))
    obs_da = obs_da.chunk(dict(time=-1))
    # Convert calendar to no leap year
    obs_da = convert_calendar(obs_da, "noleap")
    gridded_da = convert_calendar(gridded_da, "noleap")
    # Data at the desired time slice
    data_sliced = gridded_da.sel(time=slice(str(time_slice[0]), str(time_slice[1])))
    # Get QDS
    QDM = QuantileDeltaMapping.train(
        obs_da,
        # Input data, sliced to time period of observational data
        gridded_da.sel(
            time=slice(
                str(obs_da.time.values[0].year), str(obs_da.time.values[-1].year)
            )
        ),
        nquantiles=nquantiles,
        group=group,
        kind=kind,
    )
    # Bias correct the data
    da_adj = QDM.adjust(data_sliced)
    da_adj.name = gridded_da.name  # Rename it to get back to original name
    return da_adj


def _preprocess_hadisd(ds):
    """
    Preprocess station data so that it can be more seamlessly integrated into the wrangling process
    Get name of station id and station name
    Rename data variable to the station name; this allows the return of a Dataset object, with each unique station as a data variable
    Convert celcius to kelvin
    Assign descriptive attributes
    Drop unneccessary coordinates that can cause issues when bias correcting with the model data

    Args:
        ds (xr.Dataset): data for a single HadISD station

    Returns:
        xr.Dataset

    """
    # Get station ID from file name
    station_id = ds.encoding["source"].split("HadISD_")[1].split(".zarr")[0]
    # Get name of station from station_id
    station_name = stations_df.loc[stations_df["station id"] == int(station_id)][
        "station"
    ].item()
    # Rename data variable to station name
    ds = ds.rename({"tas": station_name})
    # Convert Celcius to Kelvin
    ds[station_name] = ds[station_name] + 273.15
    # Assign descriptive attributes to the data variable
    ds[station_name] = ds[station_name].assign_attrs(
        {
            "coordinates": (
                ds.latitude.values.item(),
                ds.longitude.values.item(),
            ),
            "elevation": "{0} {1}".format(
                ds.elevation.item(), ds.elevation.attrs["units"]
            ),
            "units": "K",
        }
    )
    # Drop all coordinates except time
    ds = ds.drop(["elevation", "latitude", "longitude"])
    return ds


# ============ Retrieve data from a csv input ===============


def _read_catalog_from_csv(selections, cat, csv, merge=True):
    """Retrieve data from csv input.

    Allows user to bypass app.select GUI and allows developers to
    pre-set inputs in a csv file for ease of use in a notebook.
    selections: DataSelector
        Data settings (variable, unit, timescale, etc)
    Parameters
    ----------
    selections: DataLoaders
        Data settings (variable, unit, timescale, etc).
    cat: intake_esm.core.esm_datastore
        AE data catalog.
    csv: str
        Filepath to local csv file.
    merge: bool, optional
        If multiple datasets desired, merge to form a single object?
        Default to True.

    Returns: one of the following, depending on csv input and merge
        xr_ds (xr.Dataset): if multiple rows are in the csv, each row is a data_variable
        xr_da (xr.DataArray): if csv only has one row
        xr_list (list of xr.DataArrays): if multiple rows are in the csv and merge=True,
            multiple DataArrays are returned in a single list.
    """

    df = pd.read_csv(csv)
    df = df.fillna("")  # Replace empty cells (set to NaN by read_csv) with empty string
    df = df.apply(
        lambda x: x.str.strip()
    )  # Strip any accidental white space before or after each input
    xr_list = []
    for index, row in df.iterrows():
        selections.variable = row.variable
        selections.scenario_historical = (
            []
            if (row.scenario_historical == "")
            else [
                # This fancy list comprehension deals with the fact that scenario_historical
                # can be set to an empty list, which would coded as an empty string in the csv
                item.strip()
                for item in row.scenario_historical.split(",")
            ]
        )
        selections.scenario_ssp = (
            []
            if (row.scenario_ssp == "")
            else [item.strip() for item in row.scenario_ssp.split(",")]
        )
        selections.area_average = row.area_average
        selections.timescale = row.timescale
        selections.resolution = row.resolution
        # Evaluate string time slice as tuple... i.e "(1980,2000)" --> (1980,2000)
        selections.time_slice = literal_eval(row.time_slice)
        selections.units = row.units
        selections.area_subset = row.area_subset
        selections.cached_area = row.cached_area

        # Retrieve data
        xr_da = _read_catalog_from_select(selections, cat)
        xr_list.append(xr_da)

    if len(xr_list) > 1:  # If there's more than one element in the list
        if merge:  # Should we merge each element in the list?
            try:  # Try to merge
                xr_ds = xr.merge(
                    xr_list, combine_attrs="drop_conflicts"
                )  # Merge to form one Dataset object
                return xr_ds
            except:  # If data is incompatable with merging
                print(
                    "Unable to merge datasets. Function returning a list of each item"
                )
                pass
        else:  # If user does not want to merge elements, return a list of DataArrays
            return xr_list
    else:  # If only one DataArray is in the list, just return the single DataArray
        return xr_da
