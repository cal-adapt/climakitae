import xarray as xr
import cartopy.crs as ccrs
import pyproj
from shapely.geometry import box
import regionmask
import intake
import numpy as np
from copy import deepcopy
import metpy

# support methods for core.Application.generate
xr.set_options(keep_attrs=True)


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
        attributes = deepcopy(data.attrs)
        source_id = data.attrs["source_id"]
        if selections.variable not in ("Precipitation (total)", "Relative Humidity", "Wind Magnitude at 10 m", "Wind Direction at 10 m", "Daily Maximum Hourly Temperature"):
            data = data[selections.variable]
        elif selections.variable == "Precipitation (total)":
            data = data["RAINC"] + data["RAINNC"]
        elif selections.variable == "Relative Humidity":
            data = metpy.calc.relative_humidity_from_mixing_ratio(data["Q2"], data["PSFC"], data["T2"]) #technically using surface pressure, not full atm pressure
        elif selections.variable == "Wind Magnitude at 10 m":
            data = np.sqrt(np.square(data["U10"]) + np.square(data["V10"]))
        elif selections.variable == "Wind Direction at 10 m":
            data = metpy.calc.wind_direction(data["U10"], data["V10"], convention="from")
        elif selections.variable == "Daily Maximum Hourly Temperature":
            pass

        # coarsen in time if 'selections' so-indicates:
        if selections.timescale == "daily":
            data = data.resample(time="1D").mean("time")
            attributes["frequency"] = "1day"
        elif selections.timescale == "monthly":
            data = data.resample(time="1MS").mean("time")
            attributes["frequency"] = "1month"
        # time-slice:
        data = data.sel(
            time=slice(
                str(selections.time_slice[0]),
                str(selections.time_slice[1]),
            )
        )
        # subset data spatially:
        data_crs = ccrs.CRS(pyproj.CRS.from_cf(data['Lambert_Conformal'].attrs))

        if ds_region:
            output = data_crs.transform_points(ccrs.PlateCarree(),
                                                   x=ds_region.coords[0][:,0],
                                                   y=ds_region.coords[0][:,1])

            data = data.sel(x=slice(np.nanmin(output[:,0]), np.nanmax(output[:,0])),
                y=slice(np.nanmin(output[:,1]), np.nanmax(output[:,1])))

            mask = ds_region.mask(data.lon, data.lat, wrap_lon=False)
            assert (
                False in mask.isnull()
            ), "Insufficient gridcells are contained within the bounds."
            data = data.where(np.isnan(mask) == False)

        if selections.area_average:
            weights = np.cos(np.deg2rad(data.lat))
            data = data.weighted(weights).mean("x").mean("y")

        # add data to larger Dataset being built:
        attrs_var = data.attrs
        all_files[source_id] = data

    to_delete = [
        k for k in attributes if k.isupper()
    ]  # these are all of the WRF-config attributes
    [attributes.pop(k) for k in to_delete]
    del attributes["source_id"]  # This is now indicated by the 'simulation' dimension.
    del attributes["variant_label"]  # This will be handled in a future version.
    attributes.update(attrs_var)
    all_files = all_files.to_array("simulation")
    all_files.attrs = attributes

    return all_files


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
    if selections.scenario[0] is None:
        raise ValueError("Please select at least one scenario.")

    if location.area_subset == "lat/lon":
        geom = _get_as_shapely(location)
        if not geom.is_valid:
            raise ValueError("Please go back to 'select' and choose a valid lat/lon range.")
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
        if not any(['SSP' in s for s in selections.scenario]):
            raise ValueError('Please also select at least one SSP to '
                     'which the historical simulation should be appended.')
        one_scenario = "Historical Climate"
        files_by_scenario = _get_file_list(selections, one_scenario)
        historical = _open_and_concat(files_by_scenario, selections, ds_region)

    all_files_list = []
    for one_scenario in selections.scenario:
        if selections.append_historical:
            if "SSP" in one_scenario:
                files_by_scenario = _get_file_list(selections, one_scenario)
                temp = _open_and_concat(files_by_scenario, selections, ds_region)
                temp = xr.concat([historical, temp], dim="time")
                temp.name = "Historical + " + one_scenario
            elif one_scenario != "Historical Climate":
                files_by_scenario = _get_file_list(selections, one_scenario)
                temp = _open_and_concat(files_by_scenario, selections, ds_region)
                temp.name = one_scenario

        else:
            files_by_scenario = _get_file_list(selections, one_scenario)
            temp = _open_and_concat(files_by_scenario, selections, ds_region)
            temp.name = one_scenario

        all_files_list.append(temp)
    all_files = xr.merge(all_files_list)
    attributes = temp.attrs
    attributes.pop(
        "experiment_id"
    )  # This is now indicated by the 'scenario' variable name.
    all_files = all_files.to_array("scenario")
    all_files.name = selections.variable
    all_files.attrs = attributes
    if not all_files.time.size:
        raise ValueError("Dataset will be empty. Please adjust selections.")
    return all_files
