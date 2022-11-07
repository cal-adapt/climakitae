import xarray as xr
import dask 
import rioxarray
import intake
import numpy as np
from shapely.geometry import box
from .catalog_convert import (
    _resolution_to_gridlabel,
    _timescale_to_table_id,
    _scenario_to_experiment_id
)
from .unit_conversions import _convert_units

# Set options 
xr.set_options(keep_attrs = True)
dask.config.set({"array.slicing.split_large_chunks": True})

# ============================ Helper functions ================================

def add_sim_coord(ds):
    """Add simulation as Dataset coords and dimensions.
    Used when reading in data from catalog. 
    """
    ds = ds.assign_coords({"simulation": ds.attrs["source_id"]})
    return ds

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
        location.longitude[0], # minx
        location.latitude[0], # miny
        location.longitude[1], # maxx
        location.latitude[1] # maxy
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
    # Add back in Historical Climate if append_historical was selected
    if (selections.append_historical == True and
        "Historical Climate" not in scenario_selections):
        scenario_selections += ["Historical Climate"]

    # Get catalog keys
    # Convert user-friendly names to catalog names (i.e. "45km" to "d01")
    activity_id = selections.downscaling_method
    table_id = _timescale_to_table_id(selections.timescale)
    grid_label = _resolution_to_gridlabel(selections.resolution)
    experiment_id = [_scenario_to_experiment_id(x) for x in scenario_selections]
    variable_id = selections.variable_id

    # Get catalog subset
    cat_subset = cat.search(
        activity_id = activity_id,
        table_id = table_id,
        grid_label = grid_label,
        variable_id = variable_id,
        experiment_id = experiment_id
    )
    return cat_subset

def _get_data_dict_and_names(cat_subset):
    """For an input catalog subset, grab the data.

    Args:
        cat_subset (intake_esm.core.esm_datastore): catalog subset

    Returns:
        data_dict (dictionary): dictionary of zarrs from catalog, with each key
        being its name and each item the zarr store

    """
    data_dict = cat_subset.to_dataset_dict(
        zarr_kwargs = {'consolidated': True},
        storage_options = {'anon': True},
        preprocess = add_sim_coord,
        progressbar = False
    )
    return data_dict

def _get_area_subset(location): 
    """ Get geometry to perform area subsetting with.

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
            raise ValueError("Please go back to 'select' and choose"
                             + " a valid lat/lon range.")
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
        ds_region = [shape]
    else:
        ds_region = None
    return ds_region

def _process_and_concat(selections, dsets, cat_subset):
    """ Process data if append_historical was selected.
    Merge all datasets into one.

    Args:
        selections (DataLoaders): object holding user's selections
        dsets (dictionary): dictionary of zarrs from catalog, with each key
            being its name and each item the zarr store
        cat_subset (intake_esm.core.esm_datastore): catalog subset

    Returns:
        da (xr.DataArray): output data

    """
    da_list = []
    scenario_list = cat_subset.unique()["experiment_id"]["values"]
    
    # If append historical is true, we don't need to have an additional
    # Historical Climate scenario coordinate
    if ("historical" in scenario_list and
        selections.append_historical == True):
        scenario_list.remove("historical")

    for scenario in scenario_list:
        sim_list = []
        da_name = _scenario_to_experiment_id(scenario, reverse = True)
        for simulation in cat_subset.unique()["source_id"]["values"]:
            if selections.append_historical and "ssp" in scenario:

                # Reset name
                da_name = "Historical + " + _scenario_to_experiment_id(scenario, reverse = True)

                # Get filenames
                try:
                    historical_filename = [name for name in dsets.keys() if simulation + "." + "historical" in name][0]
                    ssp_filename = [name for name in dsets.keys() if simulation + "." + scenario in name][0]
                except: # Some simulation + ssp options are not available. Just continue with the loop if no filename is found
                    continue
                # Grab data
                historical_data = dsets[historical_filename][selections.variable_id]
                ssp_data = dsets[ssp_filename][selections.variable_id]

                # Concatenate data. Rename scenario attribute
                historical_appended = xr.concat(
                    [historical_data, ssp_data],
                    dim = "time",
                    coords = 'minimal',
                    compat = 'override',
                    join = 'inner'
                )
                sim_list.append(historical_appended)

            else:
                try:
                    filename = [name for name in dsets.keys() if simulation + "." + scenario in name][0]
                except:
                    continue
                sim_list.append(dsets[filename][selections.variable_id])

        # Concatenate along simulation dimension
        da = xr.concat(
            sim_list,
            dim = "simulation",
            coords = 'minimal',
            compat = 'override'
        )
        da_list.append(da.assign_coords({"scenario": da_name}))

    # Concatenate along scenario dimension
    da_final = xr.concat(
        da_list,
        dim = "scenario",
        coords = 'minimal',
        compat = 'override'
    )

    # Rename
    da_final.name = selections.variable

    # Add attributes
    orig_attrs = dsets[list(dsets.keys())[0]].attrs
    da_final.attrs = { # Add descriptive attributes to DataArray
        #"conventions": orig_attrs["conventions"],
        "institution": orig_attrs["institution"],
        "source": orig_attrs["source"],
        "resolution": selections.resolution,
        "frequency": selections.timescale,
        "grid_mapping": da_final.attrs["grid_mapping"],
        "variable_id": selections.variable_id,
        #"long_name": da_final.attrs["long_name"],
        "extended_description": selections.extended_description,
        "units": da_final.attrs["units"]
    }
    return da_final

# ============ Read from catalog function used by ck.Application ===============

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
    assert not scenario_selections == [], "Please select as least one dataset."
    
    # Raise error if no simulation is selected and append_historical == True
    if selections.append_historical:
        if not any(['SSP' in s for s in scenario_selections]):
            raise ValueError('Please also select at least one SSP to '
                     'which the historical simulation should be appended.')

    # Get catalog subset for a set of user selections
    cat_subset = _get_cat_subset(selections = selections, cat = cat)

    # Read data from AWS.
    data_dict = _get_data_dict_and_names(cat_subset = cat_subset)

    # Process data if append_historical was selected.
    # Merge individual Datasets into one DataArray object.
    da = _process_and_concat(
        selections = selections,
        dsets = data_dict,
        cat_subset = cat_subset
    )

    # Time slice
    da = da.sel(
        time = slice(
            str(selections.time_slice[0]),
            str(selections.time_slice[1]))
    )

    # Perform area subsetting and area averaging
    ds_region = _get_area_subset(location = location)
    if ds_region is not None: # Perform subsetting
        da = da.rio.clip(
            geometries = ds_region,
            crs = 4326,
            drop = True
        )

    # Perform area averaging
    if selections.area_average == True:
        weights = np.cos(np.deg2rad(da.lat))
        da = da.weighted(weights).mean("x").mean("y")

    # Convert units
    da = _convert_units(da = da, selected_units = selections.units)
    return da
