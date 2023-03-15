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


def _get_as_shapely(location):
    """
    Takes the location data in the 'location' parameter, and turns it into a
    shapely object. Just doing polygons for now. Later other point/station data
    will be available too.

    Args:
        location (climakitae.selectors.LocSelectorArea): location selection

    Returns:
        shapely_geom (shapely.geometry)

    """
    # Box is formed using the following shape:
    #   shapely.geometry.box(minx, miny, maxx, maxy)
    shapely_geom = box(
        location.longitude[0],  # minx
        location.latitude[0],  # miny
        location.longitude[1],  # maxx
        location.latitude[1],  # maxy
    )
    return shapely_geom


# ============= Main functions used in data reading/processing =================


def _get_cat_subset(selections, cat):
    """For an input set of data selections, get the catalog subset.

    Args:
        selections (DataLoaders): object holding user's selections
        cat (intake_esm.core.esm_datastore): catalog

    Returns:
        cat_subset (intake_esm.core.esm_datastore): catalog subset

    """

    scenario_selections = selections.scenario_ssp + selections.scenario_historical

    # Get catalog keys
    # Convert user-friendly names to catalog names (i.e. "45km" to "d01")
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
    return cat_subset


def _get_area_subset(location):
    """Get geometry to perform area subsetting with.

    Args:
        location (climakitae.selectors.LocSelectorArea): location selection

    Returns:
        ds_region (shapely.geometry): geometry to use for subsetting

    """

    def set_subarea(boundary_dataset):
        return boundary_dataset[boundary_dataset.index == shape_index].iloc[0].geometry

    if location.area_subset == "lat/lon":
        geom = _get_as_shapely(location)
        if not geom.is_valid:
            raise ValueError(
                "Please go back to 'select' and choose" + " a valid lat/lon range."
            )
        ds_region = [geom]
    elif location.area_subset != "none":
        shape_index = int(
            location._geography_choose[location.area_subset][location.cached_area]
        )
        if location.area_subset == "states":
            shape = set_subarea(location._geographies._us_states)
        elif location.area_subset == "CA counties":
            shape = set_subarea(location._geographies._ca_counties)
        elif location.area_subset == "CA watersheds":
            shape = set_subarea(location._geographies._ca_watersheds)
        elif location.area_subset == "CA Electric Load Serving Entities (IOU & POU)":
            shape = set_subarea(location._geographies._ca_utilities)
        elif location.area_subset == "CA Electricity Demand Forecast Zones":
            shape = set_subarea(location._geographies._ca_forecast_zones)
        ds_region = [shape]
    else:
        ds_region = None
    return ds_region


def _process_and_concat(selections, location, dsets, cat_subset):
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
            # Historical climate will be appended to the SSP data
            append_historical = True
            scenario_list.remove("Historical Climate")
        if "Historical Reconstruction (ERA5-WRF)" in selections.scenario_historical:
            # We are not allowing users to select historical reconstruction data and SSP data at the same time,
            # due to the memory restrictions at the moment
            scenario_list.remove("Historical Reconstruction (ERA5-WRF)")

    for scenario in scenario_list:
        sim_list = []
        da_name = _scenario_to_experiment_id(scenario)
        for simulation in selections.simulation:
            if append_historical:
                # Reset name
                da_name = "Historical + " + scenario

                # Get filenames
                try:
                    historical_filename = [
                        name
                        for name in dsets.keys()
                        if simulation + "." + "historical" in name
                    ][0]
                    if (  # Need to get CESM2 data if ensmean is selected for ssp2-4.5 or ssp5-8.5
                        simulation == "ensmean"
                    ) and (
                        scenario
                        in [
                            "SSP 2-4.5 -- Middle of the Road",
                            "SSP 5-8.5 -- Burn it All",
                        ]
                    ):
                        ssp_filename = [
                            name
                            for name in dsets.keys()
                            if "CESM2." + _scenario_to_experiment_id(scenario) in name
                        ][0]
                    else:
                        ssp_filename = [
                            name
                            for name in dsets.keys()
                            if simulation + "." + _scenario_to_experiment_id(scenario)
                            in name
                        ][0]
                except:  # Some simulation + ssp options are not available. Just continue with the loop if no filename is found
                    continue
                # Grab data
                historical_data = dsets[historical_filename][selections.variable_id]
                ssp_data = dsets[ssp_filename][selections.variable_id]

                # Concatenate data. Rename scenario attribute
                historical_appended = xr.concat(
                    [historical_data, ssp_data],
                    dim="time",
                    coords="minimal",
                    compat="override",
                    join="inner",
                )
                sim_list.append(historical_appended)

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
                        filename = [
                            name
                            for name in dsets.keys()
                            if "CESM2." + _scenario_to_experiment_id(scenario) in name
                        ][0]
                    else:
                        filename = [
                            name
                            for name in dsets.keys()
                            if simulation + "." + _scenario_to_experiment_id(scenario)
                            in name
                        ][0]
                except:
                    continue
                sim_list.append(dsets[filename][selections.variable_id])

        # Concatenate along simulation dimension
        da = xr.concat(sim_list, dim="simulation", coords="minimal", compat="override")
        da_list.append(da.assign_coords({"scenario": da_name}))

    # Concatenate along scenario dimension
    da_final = xr.concat(da_list, dim="scenario", coords="minimal", compat="override")

    # Rename
    da_final.name = selections.variable

    # Add attributes
    orig_attrs = dsets[list(dsets.keys())[0]].attrs
    da_final.attrs = {  # Add descriptive attributes to DataArray
        "institution": orig_attrs["institution"],
        "source": orig_attrs["source"],
        "location_subset": location.cached_area,
        "resolution": selections.resolution,
        "frequency": selections.timescale,
        "grid_mapping": da_final.attrs["grid_mapping"],
        "location_subset": location.cached_area,
        "variable_id": selections.variable_id,
        "extended_description": selections.extended_description,
        "units": da_final.attrs["units"],
    }
    return da_final


# ============ Read from catalog function used by ck.Application ===============


def _get_data_one_var(selections, location, cat):
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
        # Add simulation as a coord
        dset = dset.assign_coords({"simulation": dset.attrs["source_id"]})

        # Time slice
        dset = dset.sel(
            time=slice(str(selections.time_slice[0]), str(selections.time_slice[1]))
        )

        # Perform area subsetting
        ds_region = _get_area_subset(location=location)
        if ds_region is not None:  # Perform subsetting
            dset = dset.rio.clip(geometries=ds_region, crs=4326, drop=True)

        # Perform area averaging
        if selections.area_average == "Yes":
            weights = np.cos(np.deg2rad(dset.lat))
            dset = dset.weighted(weights).mean("x").mean("y")

        # Update dataset in dictionary
        data_dict.update({dname: dset})

    # Merge individual Datasets into one DataArray object.
    da = _process_and_concat(
        selections=selections, location=location, dsets=data_dict, cat_subset=cat_subset
    )

    # Assign data type attribute
    da = da.assign_attrs({"data_type": location.data_type})

    return da


def _read_catalog_from_select(selections, location, cat, loop=False):
    """The primary and first data loading method, called by
    core.Application.retrieve, it returns a DataArray (which can be quite large)
    containing everything requested by the user (which is stored in 'selections'
    and 'location').

    Args:
        selections (DataLoaders): object holding user's selections
        location (LocSelectorArea): object holding user's location selections
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
    if (location.data_type == "Station") and (
        location.station in [[], ["No stations available at this location"]]
    ):
        raise ValueError(
            "Please select at least one weather station, or retrieve gridded data."
        )

    # For station data, need to expand time slice to ensure the historical period is included
    # At the end, the data will be cut back down to the user's original selection
    if location.data_type == "Station":
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
    orig_var_id_selection = selections.variable_id
    orig_variable_selection = selections.variable
    if orig_var_id_selection in [
        "wind_speed_derived",
        "rh_derived",
        "dew_point_derived",
        "specific_humid_derived",
    ]:
        if orig_var_id_selection == "wind_speed_derived":
            # Load u10 data
            selections.variable_id = "u10"
            u10_da = _get_data_one_var(selections, location, cat)

            # Load v10 data
            selections.variable_id = "v10"
            v10_da = _get_data_one_var(selections, location, cat)

            # Derive wind magnitude
            da = _compute_wind_mag(u10=u10_da, v10=v10_da)

        else:
            # Load pressure data
            selections.variable_id = "psfc"
            pressure_da = _get_data_one_var(selections, location, cat)

            # Load temperature data
            selections.variable_id = "t2"
            t2_da = _get_data_one_var(selections, location, cat)

            # Load mixing ratio data
            selections.variable_id = "q2"
            q2_da = _get_data_one_var(selections, location, cat)

            # Load dewpoint temperature data
            selections.variable_id = "tdps"
            tdps_da = _get_data_one_var(selections, location, cat)

            # Derive variables
            rh_da = _compute_relative_humidity(
                pressure=pressure_da, temperature=t2_da, mixing_ratio=q2_da
            )
            if orig_var_id_selection == "dew_point_derived":
                da = _compute_dewpointtemp(temperature=t2_da, rel_hum=rh_da)
            elif orig_var_id_selection == "specific_humid_derived":
                da = _compute_specific_humidity(tdps=tdps_da, pressure=pressure_da)
            elif orig_var_id_selection == "rh_derived":
                da = rh_da

        selections.variable_id = orig_var_id_selection
        da.attrs["variable_id"] = orig_var_id_selection  # Reset variable ID attribute
        da.name = orig_variable_selection  # Set name of DataArray

    else:
        da = _get_data_one_var(selections, location, cat)

    da = _convert_units(da=da, selected_units=selections.units)  # Convert units
    if location.data_type == "Station":
        if loop:
            print("Retrieving station data using a for loop")
            da = _station_loop(location, da, stations_df, original_time_slice)
        else:
            print("Retrieving station data using xr.apply")
            da = _station_apply(location, da, stations_df, original_time_slice)
        # Reset original selections
        if "Historical Climate" not in original_scenario_historical:
            selections.scenario_historical.remove("Historical Climate")
            da["scenario"] = [x.split("Historical + ")[1] for x in da.scenario.values]
        selections.time_slice = original_time_slice

    return da


# USE XR APPLY TO GET BIAS CORRECTED DATA TO STATION


def _station_apply(location, da, stations_df, original_time_slice):
    # Grab zarr data
    station_subset = stations_df.loc[stations_df["station"].isin(location.station)]
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
        _bias_corrected_closest_gridcell,
        keep_attrs=False,
        gridded_da=da,
        time_slice=original_time_slice,
    )
    return apply_output


def _bias_corrected_closest_gridcell(station_da, gridded_da, time_slice):
    """Get the closest gridcell to a weather station.
    Bias correct the data using historical station data
    """
    gridded_da_closest_gridcell = _closest_gridcell(station_da, gridded_da)
    bias_corrected = _bias_correct(station_da, gridded_da_closest_gridcell, time_slice)
    bias_corrected.attrs["station_coordinates"] = station_da.attrs["coordinates"]
    bias_corrected.attrs["station_elevation"] = station_da.attrs["elevation"]
    return bias_corrected


def _closest_gridcell(station_da, gridded_da):
    """Find the closest gridcell to station coordinates"""
    lat, lon = station_da.attrs["coordinates"]
    da_closest_gridcell = get_closest_gridcell(gridded_da, lat, lon, print_coords=False)
    da_closest_gridcell = da_closest_gridcell.drop(
        [i for i in da_closest_gridcell.coords if i not in da_closest_gridcell.dims]
    )
    return da_closest_gridcell


def _bias_correct(
    obs_da, gridded_da, time_slice, nquantiles=20, group="time.dayofyear", kind="+"
):
    """Bias correct model data using observational station data"""
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


#### LOOP THROUGH EACH STATION INSTEAD


def _station_loop(location, da, stations_df, original_time_slice):
    """Get the closest gridcell, perform bias correction using a for loop"""
    stations_da_list = []
    for station in location.station:
        # Retrieve the closest gridcell to the weather station
        da_closest_gridcell = _retrieve_and_format_closest_gridcell(
            stations_df, station, da
        )
        # Bias correct using historical station data
        bias_correct = True
        if bias_correct == True:
            da_closest_gridcell = _bias_correct_backend(
                stations_df,
                station,
                da_closest_gridcell,
                original_time_slice,
            )
        # Drop coordinates that aren't also dimensions
        da_closest_gridcell = da_closest_gridcell.drop(
            [i for i in da_closest_gridcell.coords if i not in da_closest_gridcell.dims]
        )
        stations_da_list.append(da_closest_gridcell)

    # Merge all stations to form a single object
    da = xr.merge(stations_da_list, combine_attrs="drop_conflicts")
    return da


def _bias_correct_backend(
    stations_df,
    station_name,
    da,
    time_slice,
    nquantiles=20,
    group="time.dayofyear",
    kind="+",
):
    """Bias correct the gridded data using observational station data.
    Retrieves the appropriate zarr store corresponding to the station name from the AE bucket.

    Parameters
    ----------
    stations_df: pd.DataFrame
        Weather station names and coordinates
    station_name: str
        Name of weather station to use
        Must directly correspond to a value in the "station" column of stations_gpd
    data: xr.DataArray
        Gridded data

    Returns
    -------
    xr.DataArray
    """
    # Get the path to the zarr store using the station ID
    station_row = stations_df.loc[stations_df["station"] == station_name]
    station_id = str(station_row["station id"].item())
    s3_path = "s3://cadcat/tmp/hadisd/HadISD_" + station_id + ".zarr"
    # Open the zarr store
    obs_da = xr.open_dataset(
        s3_path,
        engine="zarr",
        consolidated=False,
        backend_kwargs=dict(storage_options={"anon": True}),
    ).tas
    # Convert units to whatever the gridded data units are
    obs_da = _convert_obs_da_units(obs_da, da.units)
    # Rechunk data. Cannot be chunked along time dimension
    # Error raised by xclim: ValueError: Multiple chunks along the main adjustment dimension time is not supported.
    da = da.chunk(dict(time=-1))
    # Convert calendar to no leap year
    obs_da = convert_calendar(obs_da, "noleap")
    da = convert_calendar(da, "noleap")
    # Data at the desired time slice
    data_sliced = da.sel(time=slice(str(time_slice[0]), str(time_slice[1])))
    # Get QDS
    QDM = QuantileDeltaMapping.train(
        obs_da,
        # Input data, sliced to time period of observational data
        da.sel(
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
    da_adj.name = da.name  # Rename it to get back to original name
    return da_adj


def _convert_obs_da_units(obs_data, new_units):
    """Observational data is in degrees Celcius.
    Convert to correct units.
    Must be converted to Kelvin first to user _convert_units function
    """
    if obs_data.units != "degC":
        raise ValueError(
            "Expected observational data to have units of degC, but found other units. Unable to perform unit conversion. Consult code base and raw data."
        )
    if obs_data.units != new_units:
        obs_data = obs_data + 273.15  # Convert Celcius to Kelvin
        obs_data.attrs["units"] = "K"
        if new_units != "K":
            obs_data = _convert_units(obs_data, new_units)
    return obs_data


def _retrieve_and_format_closest_gridcell(stations_df, station_name, data):
    """Retrieve the closest gridcell to an input weather station
    Format the DataArray

    Parameters
    ----------
    stations_df: pd.DataFrame
        Weather station names and coordinates
    station_name: str
        Name of weather station to use.
        Must directly correspond to a value in the "station" column of stations_gpd
    data: xr.DataArray
        Gridded data

    Returns
    -------
    xr.DataArray
        Input DataArray subsetted to the closest gridcell to the station coordinates
        DataArray name is set to station_name

    See also
    --------
    climakitae.utils.get_closest_gridcell

    """

    # Get the closest grid cell
    station_row = stations_df.loc[stations_df["station"] == station_name]
    data_closest_gridcell = get_closest_gridcell(
        data, station_row.LAT_Y.item(), station_row.LON_X.item(), print_coords=False
    )

    # Clean and reformat the DataArray
    attrs_to_keep = {
        key: data_closest_gridcell.attrs[key]
        for key in data_closest_gridcell.attrs
        if key
        not in [
            "resolution",
            "location_subset",
            "grid_mapping",
            "source",
            "institution",
        ]
    }
    new_attrs = {
        # Presevere the gridded data's central coordinate and the name of the closest station
        "coordinates": (
            data_closest_gridcell.lat.values.item(),
            data_closest_gridcell.lon.values.item(),
        ),
        "variable": data_closest_gridcell.name,
    }
    data_closest_gridcell.attrs = attrs_to_keep | new_attrs  # Reassign attributes
    data_closest_gridcell.name = station_name  # Make the name the station name
    return data_closest_gridcell


# ============ Retrieve data from a csv input ===============


def _read_catalog_from_csv(selections, location, cat, csv, merge=True):
    """Retrieve data from csv input.

    Allows user to bypass app.select GUI and allows developers to
    pre-set inputs in a csv file for ease of use in a notebook.
    location: LocSelectorArea
        Location settings
    selections: DataSelector
        Data settings (variable, unit, timescale, etc)
    Parameters
    ----------
    selections: DataLoaders
        Data settings (variable, unit, timescale, etc).
    location: LocSelectorArea
        Location settings.
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
        location.area_subset = row.area_subset
        location.cached_area = row.cached_area

        # Retrieve data
        xr_da = _read_catalog_from_select(selections, location, cat)
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
