import xarray as xr
import dask
import rioxarray
import intake
import numpy as np
import psutil
from shapely.geometry import box
from .catalog_convert import (
    _resolution_to_gridlabel,
    _timescale_to_table_id,
    _scenario_to_experiment_id,
)
from .unit_conversions import _convert_units
from .utils import _readable_bytes
from .derive_variables import (
    _compute_relative_humidity,
    _compute_wind_mag,
    _compute_dewpointtemp,
)

# Set options
xr.set_options(keep_attrs=True)
dask.config.set({"array.slicing.split_large_chunks": True})


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
    activity_id = selections.downscaling_method
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
                    if ( # Need to get CESM2 data if ensmean is selected for ssp2-4.5 or ssp5-8.5
                        (simulation == "ensmean") and 
                        (scenario in ["SSP 2-4.5 -- Middle of the Road","SSP 5-8.5 -- Burn it All"])
                    ):
                        ssp_filename = [
                            name
                            for name in dsets.keys()
                            if "CESM2." + _scenario_to_experiment_id(scenario)
                            in name
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

    # Get catalog subset for a set of user selections
    cat_subset = _get_cat_subset(selections=selections, cat=cat)

    # Read data from AWS.
    data_dict = cat_subset.to_dataset_dict(
        zarr_kwargs={"consolidated": True},
        storage_options={"anon": True},
        progressbar=False,
    )
    
    # If SSP 2-4.5 or SSP 5-8.5 are selected, along with ensmean as the simulation, 
    # We want to return the single available CESM2 model 
    if (
        ("ensmean" in selections.simulation) and 
        ({"SSP 2-4.5 -- Middle of the Road",
          "SSP 5-8.5 -- Burn it All"
         }.intersection(set(selections.scenario_ssp)))
    ): 
        cat_subset2 = cat.search(
            activity_id=selections.downscaling_method,
            table_id=_timescale_to_table_id(selections.timescale),
            grid_label=_resolution_to_gridlabel(selections.resolution),
            variable_id=selections.variable_id,
            experiment_id=[_scenario_to_experiment_id(x) for x in selections.scenario_ssp],
            source_id=["CESM2"]
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

        # Perform area subsetting and area averaging
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

    return da


def _read_from_catalog(selections, location, cat):
    """
    The primary and first data loading method, called by
    core.Application.retrieve, it returns a DataArray (which can be quite large)
    containing everything requested by the user (which is stored in 'selections'
    and 'location').

    Args:
        selections (DataLoaders): object holding user's selections
        cat (intake_esm.core.esm_datastore): catalog

    Returns:
        da (xr.DataArray): output data

    """
    scenario_selections = selections.scenario_ssp + selections.scenario_historical

    # Raise error if no scenarios are selected
    if scenario_selections == []:
        raise ValueError("Please select as least one dataset.")

    # Deal with derived variables
    orig_var_id_selection = selections.variable_id
    orig_variable_selection = selections.variable
    if orig_var_id_selection in [
        "wind_speed_derived",
        "rh_derived",
        "dew_point_derived",
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

            # Derive relative humidity
            rh_da = _compute_relative_humidity(
                pressure=pressure_da, temperature=t2_da, mixing_ratio=q2_da
            )
            if orig_var_id_selection == "dew_point_derived":
                da = _compute_dewpointtemp(temperature=t2_da, rel_hum=rh_da)
            elif orig_var_id_selection == "rh_derived":
                da = rh_da

        selections.variable_id = orig_var_id_selection
        da.attrs["variable_id"] = orig_var_id_selection  # Reset variable ID attribute
        da.name = orig_variable_selection  # Set name of DataArray

    else:
        da = _get_data_one_var(selections, location, cat)

    da = _convert_units(da=da, selected_units=selections.units)  # Convert units
    return da
