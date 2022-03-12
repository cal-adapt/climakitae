import xarray as xr
from shapely.geometry import box #, Point, Polygon
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
    file_list = []
    for item in list(cat):
        if cat[item].metadata["nominal_resolution"] == selections.resolution:
            if cat[item].metadata["experiment_id"] == scenario:
                file_list.append(cat[item].name)
    return file_list


def _open_and_concat(file_list, selections, geom):
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
        if geom:
            # subset data spatially:
            ds_region = regionmask.Regions(
                        [geom], abbrevs=["lat/lon box"], name="box mask"
            )
            mask = ds_region.mask(data.lon, data.lat, wrap_lon=False)
            assert False not in mask.isnull() #, "No grid cells are within the lat/lon bounds."
            data = (
                data.where(np.isnan(mask) == False)
                .dropna("x", how="all")
                .dropna("y", how="all")
            )
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
    if location.subset_by_lat_lon:
        geom = _get_as_shapely(location)
        assert geom.is_valid, "Please go back to 'select' and choose a valid lat/lon range."
    else:
        geom = False  # for now... later a cached polygon will be an elseif option too

    all_files = xr.Dataset()
    for one_scenario in selections.scenario:
        files_by_scenario = _get_file_list(selections, one_scenario)
        temp = _open_and_concat(files_by_scenario, selections, geom)
        all_files[one_scenario] = temp
        # if selections.append_historical:
        #    files_historical = get_file_list(selections,'historical')
        #    all_files = xr.concat([files_historical,all_files],dim='time')
    all_files = all_files.to_array('scenario')
    all_files.name = selections.variable
    return all_files
