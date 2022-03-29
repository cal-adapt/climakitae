import xarray as xr
from shapely.geometry import box
import regionmask
import intake
import numpy as np

# support methods for core.Application.generate


def _get_file_list(selections, scenario):
    """
    Returns a list of simulation names for all of the simulations present in the catalog
    for a given scenario, contingent on other user-supplied constraints in 'selections'.
    """
    cat = selections._choices._cat
    lookup = {v: k for k, v in selections._choices._scenario_choices.items()}
    file_list = []
    for item in list(cat):
        if cat[item].metadata["nominal_resolution"] == selections.resolution:
            if cat[item].metadata["experiment_id"] == lookup[scenario]:
                file_list.append(cat[item].name)
    return file_list


def _open_and_concat(file_list, selections, ds_region):
    """
    Open multiple zarr files, and add them to one big xarray Dataset. Coarsens in time, and/or
    subsets in space if selections so-indicates. Won't work unless the file_list supplied
    contains files of only one nominal resolution (_get_file_list ensures this).
    """
    cat = selections._choices._cat
    all_files = xr.Dataset()
    for one_file in file_list:
        data = cat[one_file].to_dask()
        source_id = data.attrs["source_id"]
        if selections.variable not in ("precipitation (total)", "wind 10m magnitude"):
            data = data[selections.variable]
        elif selections.variable == "precipitation (total)":
            pass
        elif selections.variable == "wind 10m magnitude":
            pass

        # coarsen in time if 'selections' so-indicates:
        if selections.timescale == "daily":
            data = data.resample(time="1D").mean("time")
        elif selections.timescale == "monthly":
            data = data.resample(time="1MS").mean("time")
        if ds_region:
            # subset data spatially:
            mask = ds_region.mask(data.lon, data.lat, wrap_lon=False)
            assert (
                False in mask.isnull()
            ), "Insufficient gridcells are contained within the bounds."

            data = (
                data.where(np.isnan(mask) == False)
                .dropna("x", how="all")
                .dropna("y", how="all")
            )

        if selections.area_average:
            weights = np.cos(np.deg2rad(data.lat))
            data = data.weighted(weights).mean('x').mean('y')

        # add data to larger Dataset being built
        all_files[source_id] = data
        
    return all_files.to_array("simulation")


def _get_as_shapely(location):
    """
    Takes the location data in the 'location' parameter, and turns it into a shapely object.
    Just doing polygons for now. Later other point/station data will be available too.
    """
    # shapely.geometry.box(minx, miny, maxx, maxy):
    return box(
        location.longitude[0],
        location.latitude[0],
        location.longitude[1],
        location.latitude[1],
    )


def _read_from_catalog(selections, location):
    """
    The primary and first data loading method, called by core.Application.generate, it returns
    a dataset (which can be quite large) containing everything requested by the user (which is
    stored in 'selections' and 'location').
    """
    assert not selections.scenario is None, "Please select at least one scenario."

    if location.area_subset == "lat/lon":
        geom = _get_as_shapely(location)
        assert (
            geom.is_valid
        ), "Please go back to 'select' and choose a valid lat/lon range."
        ds_region = regionmask.Regions([geom], abbrevs=["lat/lon box"], name="box mask")
    elif location.area_subset == "states":
        shape_index = int(
            location._geography_choose[location.area_subset][location.cached_area]
        )
        ds_region = location._geographies._us_states[[shape_index]]
    elif location.area_subset != "none":
        shape_index = int(
            location._geography_choose[location.area_subset][location.cached_area]
        )
        if location.area_subset == "CA watersheds":
            shape = location._geographies._ca_watersheds
            shape = shape[shape["OBJECTID"] == shape_index].iloc[0].geometry
        elif location.area_subset == "CA counties":
            shape = location._geographies._ca_counties
            shape = shape[shape.index == shape_index].iloc[0].geometry
        ds_region = regionmask.Regions(
            [shape], abbrevs=["geographic area"], name="area mask"
        )
    else:
        ds_region = None

    if selections.append_historical:
        one_scenario = 'Historical Climate'
        files_by_scenario = _get_file_list(selections, one_scenario)
        historical = _open_and_concat(files_by_scenario, selections, ds_region)
        
    all_files = xr.Dataset()
    for one_scenario in selections.scenario:
        if selections.append_historical:
            if "SSP" in one_scenario:
                files_by_scenario = _get_file_list(selections, one_scenario)
                temp = _open_and_concat(files_by_scenario, selections, ds_region)
                all_files["Historical + "+one_scenario] = xr.concat([historical,temp],dim="time")
            elif one_scenario != "Historical Climate":
                files_by_scenario = _get_file_list(selections, one_scenario)
                temp = _open_and_concat(files_by_scenario, selections, ds_region)
                all_files[one_scenario] = temp
        else:
            files_by_scenario = _get_file_list(selections, one_scenario)
            temp = _open_and_concat(files_by_scenario, selections, ds_region)
            all_files[one_scenario] = temp
            
    all_files = all_files.to_array("scenario")
    all_files.name = selections.variable
    return all_files
