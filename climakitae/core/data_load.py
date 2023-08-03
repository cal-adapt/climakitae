"""Backend functions for retrieving and subsetting data from the AE catalog"""

import xarray as xr
import dask
import rioxarray
from rioxarray.exceptions import NoDataInBounds
import intake
import numpy as np
import pandas as pd
import pkg_resources
import psutil
import warnings
import fnmatch
from functools import partial
from ast import literal_eval
from shapely.geometry import box

from xclim.core.calendar import convert_calendar
from xclim.sdba import Grouper
from xclim.sdba.adjustment import QuantileDeltaMapping
from climakitae.core.catalog_convert import (
    downscaling_method_to_activity_id,
    resolution_to_gridlabel,
    timescale_to_table_id,
    scenario_to_experiment_id,
)
from climakitae.util.unit_conversions import convert_units
from climakitae.util.utils import _readable_bytes, get_closest_gridcell
from climakitae.tools.derive_variables import (
    _compute_relative_humidity,
    _compute_wind_mag,
    _compute_wind_dir,
    _compute_dewpointtemp,
    _compute_specific_humidity,
)

# Set options
xr.set_options(keep_attrs=True)
dask.config.set({"array.slicing.split_large_chunks": True})


# ============================ Read data into memory ================================


def load(xr_da):
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
            "Processing data to read {0} of data into memory... ".format(
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
        selections (DataParameters): Data settings (variable, unit, timescale, etc)

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


def _sim_index_item(ds_name, member_id):
    """Identify a simulation by its downscaling type, driving GCM, and member id.

    Args:
        one (str): dataset name from catalog
        member_id (xr.Dataset.attr): ensemble member id from dataset attributes

    Returns:
        str: joined by underscores
    """
    downscaling_type = ds_name.split(".")[0]
    gcm_name = ds_name.split(".")[2]
    ensemble_member = str(member_id.values)
    return "_".join([downscaling_type, gcm_name, ensemble_member])


def _scenarios_in_data_dict(keys):
    """Return unique list of ssp scenarios in dataset dictionary.

    Args:
        keys (list[str]): list of dataset names from catalog

    Returns:
        scenario_list: list[str]: unique scenarios

    """
    scenarios = set([one.split(".")[3] for one in keys if "ssp" in one])

    return list(scenarios)


# ============= Main functions used in data reading/processing =================


def _get_cat_subset(selections):
    """For an input set of data selections, get the catalog subset.

    Args:
        selections (DataParameters): object holding user's selections

    Returns:
        cat_subset (intake_esm.core.esm_datastore): catalog subset

    """

    scenario_selections = selections.scenario_ssp + selections.scenario_historical

    # Get catalog keys
    # Convert user-friendly names to catalog names (i.e. "45 km" to "d01")
    activity_id = [
        downscaling_method_to_activity_id(dm) for dm in selections.downscaling_method
    ]
    table_id = timescale_to_table_id(selections.timescale)
    grid_label = resolution_to_gridlabel(selections.resolution)
    experiment_id = [scenario_to_experiment_id(x) for x in scenario_selections]
    source_id = selections.simulation
    variable_id = selections.variable_id

    cat_subset = selections._data_catalog.search(
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
            for institution in selections._data_catalog.df.institution_id.unique()
            if institution != "UCSD"
        ]
        cat_subset = cat_subset.search(institution_id=wrf_on_native_grid)

    return cat_subset


def _time_slice(dset, selections):
    """Subset over time
    Args:
        dset (xr.Dataset): one dataset from the catalog
        selections (DataParameters): object holding user's selections

    Returns:
        xr.Dataset: time-slice of dset
    """

    window_start = str(selections.time_slice[0])
    window_end = str(selections.time_slice[1])

    return dset.sel(time=slice(window_start, window_end))


def _override_area_selections(selections):
    """Account for 'station' special-case
    You need to retrieve the entire domain because the shapefiles will cut out
    the ocean grid cells, but the some station's closest gridcells are the ocean!

    Args:
        selections (DataParameters): object holding user's selections

    Returns:
        area_subset (str):
        cached_area (str):
    """
    if selections.data_type == "Station":
        area_subset = "none"
        cached_area = "entire domain"
    else:
        area_subset = selections.area_subset
        cached_area = selections.cached_area

    return area_subset, cached_area


def _area_subset_geometry(selections):
    """Get geometry to perform area subsetting with.

    Args:
        selections (DataParameters): object holding user's selections

    Returns:
        ds_region (shapely.geometry): geometry to use for subsetting

    """
    area_subset, cached_area = _override_area_selections(selections)

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


def _clip_to_geometry(dset, ds_region):
    """Clip to geometry if large enough
    Args:
        dset (xr.Dataset): one dataset from the catalog
        ds_region (shapely.geometry.polygon.Polygon): area to clip to

    Returns:
        xr.Dataset: clipped area of dset
    """
    try:
        dset = dset.rio.clip(geometries=ds_region, crs=4326, drop=True)

    except NoDataInBounds as e:
        # Catch small geometry error
        print(e)
        print("Skipping spatial subsetting.")

    return dset


def _clip_to_geometry_loca(dset, ds_region):
    """Clip to geometry, adding missing grid info
        because crs and x, y are missing from LOCA datasets
        Otherwise rioxarray will raise this error:
        'MissingSpatialDimensionError: x dimension not found.'
    Args:
        dset (xr.Dataset): one dataset from the catalog
        ds_region (shapely.geometry.polygon.Polygon): area to clip to

    Returns:
        xr.Dataset: clipped area of dset
    """
    dset = dset.rename({"lon": "x", "lat": "y"})
    dset = dset.rio.write_crs("EPSG:4326")
    dset = _clip_to_geometry(dset, ds_region)
    dset = dset.rename({"x": "lon", "y": "lat"}).drop("spatial_ref")
    return dset


def _spatial_subset(dset, selections):
    """Subset over spatial area
    Args:
        dset (xr.Dataset): one dataset from the catalog
        selections (DataParameters): object holding user's selections

    Returns:
        xr.Dataset: subsetted area of dset
    """
    ds_region = _area_subset_geometry(selections)

    if ds_region is not None:  # Perform subsetting
        if selections.downscaling_method == ["Dynamical"]:
            dset = _clip_to_geometry(dset, ds_region)
        else:
            dset = _clip_to_geometry_loca(dset, ds_region)

    return dset


def _area_average(dset):
    """Weighted area-average

    Args:
        dset (xr.Dataset): one dataset from the catalog

    Returns:
        xr.Dataset: sub-setted output data

    """
    weights = np.cos(np.deg2rad(dset.lat))
    if set(["x", "y"]).issubset(set(dset.dims)):
        # WRF data has x,y
        dset = dset.weighted(weights).mean("x").mean("y")
    elif set(["lat", "lon"]).issubset(set(dset.dims)):
        # LOCA data has lat, lon
        dset = dset.weighted(weights).mean("lat").mean("lon")
    return dset


def _process_dset(ds_name, dset, selections):
    """Subset over time and space, as described in user selections;
       renaming to facilitate concatenation.

    Args:
        dset (xr.Dataset): one dataset from the catalog
        selections (DataParameters): object holding user's selections

    Returns:
        xr.Dataset: sub-setted output data

    """
    # Time slice
    dset = _time_slice(dset, selections)

    # Perform area subsetting
    dset = _spatial_subset(dset, selections)

    # Perform area averaging
    if selections.area_average == "Yes":
        dset = _area_average(dset)

    # Rename member_id value to include more info
    dset = dset.assign_coords(
        member_id=[_sim_index_item(ds_name, mem_id) for mem_id in dset.member_id]
    )
    # Rename variable to display name:
    dset = dset.rename({list(dset.data_vars)[0]: selections.variable})

    return dset


def _concat_sims(data_dict, hist_data, selections, scenario):
    """Combine datasets along expanded 'member_id' dimension, and append
        historical if relevant.

    Args:
        data_dict (dictionary): dictionary of zarrs from catalog, with each key
            being its name and each item the zarr store
        hist_data (xr.Dataset): subsetted historical data to append
        selections (DataParameters): class holding data selections
        scenario (str): short designation for one SSP

    Returns:
        one_scenario (xr.Dataset): combined data object
    """
    scen_name = scenario_to_experiment_id(scenario, reverse=True)

    # Merge along expanded 'member_id' dimension:
    one_scenario = xr.concat(
        [
            _process_dset(one, data_dict[one], selections)
            for one in data_dict.keys()
            if scenario in one
        ],
        dim="member_id",
    )

    # Append historical if relevant:
    if hist_data != None:
        scen_name = "Historical + " + scen_name
        one_scenario = xr.concat([hist_data, one_scenario], dim="time")

    # Set-up coordinate:
    one_scenario = one_scenario.assign_coords({"scenario": scen_name})

    return one_scenario


def _override_unit_defaults(da, var_id):
    """Override non-standard unit specifications in some dataset attributes

    Args:
        da (xr.DataArray): any xarray DataArray with a units attribute

    Returns:
        xr.DataArray: output data

    """
    if var_id == "huss":
        # Units for LOCA specific humidity are set to 1
        # Reset to kg/kg so they can be converted if neccessary to g/kg
        da.attrs["units"] = "kg/kg"
    elif var_id == "rsds":
        # rsds units are "W m-2"
        # rename them to W/m2 to match the lookup catalog, and the units for WRF radiation variables
        da.attrs["units"] = "W/m2"
    return da


def _merge_all(selections, data_dict, cat_subset):
    """Merge all datasets into one, subsetting each consistently;
       clean-up format, and convert units.

    Args:
        selections (DataParameters): object holding user's selections
        data_dict (dictionary): dictionary of zarrs from catalog, with each key
            being its name and each item the zarr store
        cat_subset (intake_esm.core.esm_datastore): catalog subset

    Returns:
        da (xr.DataArray): output data

    """

    # Get corresponding data for historical period to append:
    hist_keys = [one for one in data_dict.keys() if "historical" in one]
    if hist_keys:
        all_hist = xr.concat(
            [
                _process_dset(one, data_dict[one], selections)
                for one in data_dict.keys()
                if "historical" in one
            ],
            dim="member_id",
        )
    else:
        all_hist = None

    # Get (and double-check) list of SSP scenarios:
    _scenarios = _scenarios_in_data_dict(data_dict.keys())

    if _scenarios:
        # Merge along new 'scenario' dimension:
        all_ssps = xr.concat(
            [
                _concat_sims(data_dict, all_hist, selections, scenario)
                for scenario in _scenarios
            ],
            combine_attrs="drop_conflicts",
            dim="scenario",
        )
    else:
        all_ssps = all_hist
        all_ssps = all_ssps.assign_coords(
            {"scenario": selections.scenario_historical[0]}
        )
        all_ssps = all_ssps.expand_dims(dim={"scenario": 1})

    # Rename expanded dimension:
    all_ssps = all_ssps.rename({"member_id": "simulation"})

    # Convert to xr.DataArray:
    var_id = list(all_ssps.data_vars)[0]
    all_ssps = all_ssps[var_id]

    # Convert units:
    all_ssps = _override_unit_defaults(all_ssps, var_id)
    all_ssps = convert_units(da=all_ssps, selected_units=selections.units)

    return all_ssps


def _get_data_one_var(selections):
    """Get data for one variable
    Retrieves dataset dictionary from AWS, handles some special cases, merges
    datasets along new dimensions into one xr.DataArray, and adds metadata.

    Args:
        selections (DataParameters): object holding user's selections

    Returns:
        da (xr.DataArray): with datasets combined over new dimensions 'simulation' and 'scenario'
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Silence warning if empty dataset returned

        # Get catalog subset for a set of user selections
        cat_subset = _get_cat_subset(selections=selections)

        if len(cat_subset.df["institution_id"].unique()) == 1:
            _institution = cat_subset.df["institution_id"].unique()[0]
        else:
            _institution = "Multiple"

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
        cat_subset2 = selections._data_catalog.search(
            activity_id=[
                downscaling_method_to_activity_id(dm)
                for dm in selections.downscaling_method
            ],
            table_id=timescale_to_table_id(selections.timescale),
            grid_label=resolution_to_gridlabel(selections.resolution),
            variable_id=selections.variable_id,
            experiment_id=[
                scenario_to_experiment_id(x) for x in selections.scenario_ssp
            ],
            source_id=["CESM2"],
        )
        data_dict2 = cat_subset2.to_dataset_dict(
            zarr_kwargs={"consolidated": True},
            storage_options={"anon": True},
            progressbar=False,
        )
        data_dict = {**data_dict, **data_dict2}

    # Merge individual Datasets into one DataArray object.
    da = _merge_all(selections=selections, data_dict=data_dict, cat_subset=cat_subset)

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
        "institution": _institution,
    }
    if "grid_mapping" in da.attrs:
        da_new_attrs = da_new_attrs | {"grid_mapping": da.attrs["grid_mapping"]}
    da.attrs = da_new_attrs
    da.name = selections.variable
    return da


def read_catalog_from_select(selections):
    """The primary and first data loading method, called by
    core.Application.retrieve, it returns a DataArray (which can be quite large)
    containing everything requested by the user (which is stored in 'selections').

    Args:
        selections (DataParameters): object holding user's selections

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
            2014,
        )  # Bounds of the observational data used in bias-correction
        if original_time_slice[0] > obs_data_bounds[0]:
            selections.time_slice = (obs_data_bounds[0], original_time_slice[1])
        if original_time_slice[1] < obs_data_bounds[1]:
            selections.time_slice = (selections.time_slice[0], obs_data_bounds[1])

    # Deal with derived variables
    orig_var_id_selection = selections.variable_id[0]
    orig_unit_selection = selections.units
    orig_variable_selection = selections.variable
    if "_derived" in orig_var_id_selection:
        if orig_var_id_selection in ["wind_speed_derived", "wind_direction_derived"]:
            # Load u10 data
            selections.variable_id = ["u10"]
            selections.units = (
                "m s-1"  # Need to set units to required units for _compute_wind_mag
            )
            u10_da = _get_data_one_var(selections)

            # Load v10 data
            selections.variable_id = ["v10"]
            selections.units = "m s-1"
            v10_da = _get_data_one_var(selections)

            # Derive wind magnitude
            if orig_var_id_selection == "wind_speed_derived":
                da = _compute_wind_mag(u10=u10_da, v10=v10_da)  # m/s  # m/s
            # Or, derive wind speed
            elif orig_var_id_selection == "wind_direction_derived":
                da = _compute_wind_dir(u10=u10_da, v10=v10_da)

        elif orig_var_id_selection == "dew_point_derived":
            # Daily/monthly dew point inputs have different units
            # Hourly dew point temp derived differently because you also have to derive relative humidity

            # Load temperature data
            selections.variable_id = ["t2"]
            selections.units = (
                "K"  # Kelvin required for humidity and dew point computation
            )
            t2_da = _get_data_one_var(selections)

            selections.variable_id = ["rh"]
            selections.units = "[0 to 100]"
            rh_da = _get_data_one_var(selections)

            # Derive dew point temperature
            # Returned in units of Kelvin
            da = _compute_dewpointtemp(
                temperature=t2_da, rel_hum=rh_da  # Kelvin  # [0-100]
            )
        else:
            # Load temperature data
            selections.variable_id = ["t2"]
            selections.units = (
                "K"  # Kelvin required for humidity and dew point computation
            )
            t2_da = _get_data_one_var(selections)

            # Load mixing ratio data
            selections.variable_id = ["q2"]
            selections.units = "kg kg-1"
            q2_da = _get_data_one_var(selections)

            # Load pressure data
            selections.variable_id = ["psfc"]
            selections.units = "Pa"
            pressure_da = _get_data_one_var(selections)

            # Derive relative humidity
            # Returned in units of [0-100]
            rh_da = _compute_relative_humidity(
                pressure=pressure_da,  # Pa
                temperature=t2_da,  # Kelvin
                mixing_ratio=q2_da,  # kg/kg
            )

            if orig_var_id_selection == "rh_derived":
                da = rh_da

            else:
                # Derive dew point temperature
                # Returned in units of Kelvin
                dew_pnt_da = _compute_dewpointtemp(
                    temperature=t2_da, rel_hum=rh_da  # Kelvin  # [0-100]
                )

                if orig_var_id_selection == "dew_point_derived_hrly":
                    da = dew_pnt_da

                elif orig_var_id_selection == "q2_derived":
                    # Derive specific humidity
                    # Returned in units of g/kg
                    da = _compute_specific_humidity(
                        tdps=dew_pnt_da, pressure=pressure_da  # Kelvin  # Pa
                    )

                else:
                    raise ValueError(
                        "You've encountered a bug. No data available for selected derived variable."
                    )

        da = convert_units(da, selected_units=orig_unit_selection)
        da.attrs["variable_id"] = orig_var_id_selection  # Reset variable ID attribute
        da.attrs["units"] = orig_unit_selection
        da.name = orig_variable_selection  # Set name of DataArray

        # Reset selections to user's original selections
        selections.variable_id = [orig_var_id_selection]
        selections.units = orig_unit_selection

    else:
        da = _get_data_one_var(selections)

    if selections.data_type == "Station":
        da = _station_apply(selections, da, original_time_slice)
        # Reset original selections
        if "Historical Climate" not in original_scenario_historical:
            selections.scenario_historical.remove("Historical Climate")
            da["scenario"] = [x.split("Historical + ")[1] for x in da.scenario.values]
        selections.time_slice = original_time_slice

    return da


# USE XR APPLY TO GET BIAS CORRECTED DATA TO STATION


def _station_apply(selections, da, original_time_slice):
    # Grab zarr data
    station_subset = selections._stations_gdf.loc[
        selections._stations_gdf["station"].isin(selections.station)
    ]
    filepaths = [
        "s3://cadcat/hadisd/HadISD_{}.zarr".format(s_id)
        for s_id in station_subset["station id"]
    ]

    _partial_func = partial(_preprocess_hadisd, stations_gdf=selections._stations_gdf)

    station_ds = xr.open_mfdataset(
        filepaths,
        preprocess=_partial_func,
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
    obs_da,
    gridded_da,
    time_slice,
    window=90,
    nquantiles=20,
    group="time.dayofyear",
    kind="+",
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
    # Get group by window
    # Use 90 day window (+/- 45 days) to account for seasonality
    grouper = Grouper(group, window=window)

    # Convert units to whatever the gridded data units are
    obs_da = convert_units(obs_da, gridded_da.units)
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
        group=grouper,
        kind=kind,
    )
    # Bias correct the data
    da_adj = QDM.adjust(data_sliced)
    da_adj.name = gridded_da.name  # Rename it to get back to original name
    return da_adj


def _preprocess_hadisd(ds, stations_gdf):
    """
    Preprocess station data so that it can be more seamlessly integrated into the wrangling process
    Get name of station id and station name
    Rename data variable to the station name; this allows the return of a Dataset object, with each unique station as a data variable
    Convert celcius to kelvin
    Assign descriptive attributes
    Drop unneccessary coordinates that can cause issues when bias correcting with the model data

    Args:
        ds (xr.Dataset): data for a single HadISD station
        stations_gdf (pd.GeoDataFrame): station data frame

    Returns:
        xr.Dataset

    """
    # Get station ID from file name
    station_id = ds.encoding["source"].split("HadISD_")[1].split(".zarr")[0]
    # Get name of station from station_id
    station_name = stations_gdf.loc[stations_gdf["station id"] == int(station_id)][
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


def read_catalog_from_csv(selections, csv, merge=True):
    """Retrieve user data selections from csv input.

    Allows user to bypass app.select GUI and allows developers to
    pre-set inputs in a csv file for ease of use in a notebook.
    selections: DataSelector
        Data settings (variable, unit, timescale, etc)
    Parameters
    ----------
    selections: DataParameters
        Data settings (variable, unit, timescale, etc).
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
        xr_da = read_catalog_from_select(selections)
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
