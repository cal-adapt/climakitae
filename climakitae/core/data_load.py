import warnings
from ast import literal_eval
from datetime import timedelta
from functools import partial

# Importing DataParameters causes ImportError due to circular import
# so only import for type checking and reference as str 'DataParameters'
from typing import TYPE_CHECKING

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import rioxarray
import shapely
import xarray as xr
from dask.diagnostics import ProgressBar
from geopandas import GeoDataFrame
from rioxarray.exceptions import NoDataInBounds
from shapely.geometry import box
from xclim.sdba import Grouper
from xclim.sdba.adjustment import QuantileDeltaMapping

from climakitae.core.boundaries import Boundaries
from climakitae.core.constants import WARMING_LEVELS
from climakitae.tools.derived_variables import (
    compute_dewpointtemp,
    compute_relative_humidity,
    compute_specific_humidity,
    compute_wind_dir,
    compute_wind_mag,
)
from climakitae.tools.indices import effective_temp, fosberg_fire_index, noaa_heat_index
from climakitae.util.unit_conversions import convert_units, get_unit_conversion_options
from climakitae.util.utils import (
    _get_cat_subset,
    _get_scenario_from_selections,
    area_average,
    downscaling_method_as_list,
    downscaling_method_to_activity_id,
    get_closest_gridcell,
    readable_bytes,
    resolution_to_gridlabel,
    scenario_to_experiment_id,
    timescale_to_table_id,
)
from climakitae.util.warming_levels import (
    calculate_warming_level,
    drop_invalid_sims,
    create_new_warming_level_table,
)

if TYPE_CHECKING:
    from climakitae.core.data_interface import DataParameters

# Set Xarray options
# keep array attributes after operations
xr.set_options(keep_attrs=True)
# slice array into smaller chunks to opitimize reading
dask.config.set({"array.slicing.split_large_chunks": True})

# Silence performance warnings
# This warning is triggered when large chunks are created
from dask.array.core import PerformanceWarning
from dask.diagnostics import ProgressBar

warnings.filterwarnings("ignore", category=PerformanceWarning)


def load(xr_da: xr.DataArray, progress_bar: bool = False) -> xr.DataArray:
    """Read lazily loaded dask array into memory for faster access

    Parameters
    ----------
    xr_da: xr.DataArray
    progress_bar: boolean

    Returns
    -------
    da_computed: xr.DataArray
    """

    # Check if data is already loaded into memory
    if xr_da.chunks is None:
        print("Your data is already loaded into memory")
        return xr_da

    # Get memory information
    avail_mem = psutil.virtual_memory().available  # Available system memory
    xr_data_nbytes = xr_da.nbytes  # Memory of data

    # If it will cause the system to have less than 256MB after loading the data, do not allow the compute to proceed.
    if avail_mem - xr_data_nbytes < 268435456:
        print("Available memory: {0}".format(readable_bytes(avail_mem)))
        print("Total memory of input data: {0}".format(readable_bytes(xr_data_nbytes)))
        raise MemoryError("Your input dataset is too large to read into memory!")
    else:
        print(
            "Processing data to read {0} of data into memory... ".format(
                readable_bytes(xr_data_nbytes)
            ),
            end="",
        )
        if progress_bar:
            with ProgressBar():
                print("\r")
                da_computed = xr_da.compute()
        else:
            da_computed = xr_da.compute()
        print("Complete!")
        return da_computed


def _scenarios_in_data_dict(keys: list[str]) -> list[str]:
    """Return unique list of ssp scenarios in dataset dictionary.

    Parameters
    ----------
    keys: list[str]
        list of dataset names from catalog

    Returns
    -------
    scenario_list: list[str]
        unique scenarios
    """
    scenarios = set([one.split(".")[3] for one in keys if "ssp" in one])

    return list(scenarios)


def _time_slice(dset: xr.Dataset, selections: "DataParameters") -> xr.Dataset:
    """Subset over time

    Parameters
    ----------
    dset: xr.Dataset
        one dataset from the catalog
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    xr.Dataset
        time-slice of dset
    """

    window_start = str(selections.time_slice[0])
    window_end = str(selections.time_slice[1])

    return dset.sel(time=slice(window_start, window_end))


def area_subset_geometry(
    selections: "DataParameters",
) -> list[shapely.geometry.polygon.Polygon] | None:
    """Get geometry to perform area subsetting with.

    Parameters
    ----------
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    ds_region: shapely.geometry
        geometry to use for subsetting
    """

    def _override_area_selections(selections: "DataParameters") -> tuple[str, str]:
        """Account for 'station' special-case
        You need to retrieve the entire domain because the shapefiles will cut out
        the ocean grid cells, but the some station's closest gridcells are the ocean!

        Parameters
        ----------
        selections: DataParameters
            object holding user's selections

        Returns
        -------
        area_subset: str
        cached_area: str
        """
        if selections.data_type == "Stations":
            area_subset = "none"
            cached_area = "entire domain"
        else:
            area_subset = selections.area_subset
            cached_area = selections.cached_area

        return area_subset, cached_area

    def _set_subarea(
        boundary_dataset: Boundaries, shape_indices: list[int]
    ) -> GeoDataFrame:
        return boundary_dataset.loc[shape_indices].geometry.union_all()

    def _get_as_shapely(selections: "DataParameters") -> shapely.geometry:
        """
        Takes the location data, and turns it into a
        shapely box object. Just doing polygons for now. Later other point/station data
        will be available too.

        Parameters
        ----------
        selections: DataParameters
            Data settings (variable, unit, timescale, etc)

        Returns
        -------
        shapely_geom: shapely.geometry

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

    area_subset, cached_area = _override_area_selections(selections)

    def _get_shape_indices(
        selections: "DataParameters", area_subset: str, cached_area: str
    ) -> list:
        """
        Gets the indices of the Boundary parquet file that match the area_subet and cached_area.

        Parameters
        ----------
        selections: DataParameters
            Data settings (variable, unit, timescale, etc)

        area_subset: str
            dataset to use from Boundaries for sub area selection

        cached_area: list of strs
            one or more features from area_subset datasets to use for selection

        Returns
        -------
        list

        """
        shape_indices = list(
            {
                key: selections._geography_choose[area_subset][key]
                for key in cached_area
            }.values()
        )
        return shape_indices

    match area_subset:
        case "lat/lon":
            geom = _get_as_shapely(selections)
            if not geom.is_valid:
                raise ValueError(
                    "Please go back to 'select' and choose" + " a valid lat/lon range."
                )
            ds_region = [geom]
        case "states":
            ds_region = [
                _set_subarea(
                    selections._geographies._us_states,
                    _get_shape_indices(selections, area_subset, cached_area),
                )
            ]
        case "CA counties":
            ds_region = [
                _set_subarea(
                    selections._geographies._ca_counties,
                    _get_shape_indices(selections, area_subset, cached_area),
                )
            ]
        case "CA watersheds":
            ds_region = [
                _set_subarea(
                    selections._geographies._ca_watersheds,
                    _get_shape_indices(selections, area_subset, cached_area),
                )
            ]
        case "CA Electric Load Serving Entities (IOU & POU)":
            ds_region = [
                _set_subarea(
                    selections._geographies._ca_utilities,
                    _get_shape_indices(selections, area_subset, cached_area),
                )
            ]
        case "CA Electricity Demand Forecast Zones":
            ds_region = [
                _set_subarea(
                    selections._geographies._ca_forecast_zones,
                    _get_shape_indices(selections, area_subset, cached_area),
                )
            ]
        case "CA Electric Balancing Authority Areas":
            ds_region = [
                _set_subarea(
                    selections._geographies._ca_electric_balancing_areas,
                    _get_shape_indices(selections, area_subset, cached_area),
                )
            ]
        case _:
            ds_region = None
    return ds_region


def _spatial_subset(dset: xr.Dataset, selections: "DataParameters") -> xr.Dataset:
    """Subset over spatial area

    Parameters
    ----------
    dset: xr.Dataset
        one dataset from the catalog
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    xr.Dataset
        subsetted area of dset
    """

    def _clip_to_geometry(
        dset: xr.Dataset, ds_region: shapely.geometry.polygon.Polygon, all_touched: bool
    ) -> xr.Dataset:
        """Clip to geometry if large enough

        Parameters
        ----------
        dset: xr.Dataset
            one dataset from the catalog
        ds_region: shapely.geometry.polygon.Polygon
            area to clip to
        all_touched: boolean
            select within or touching area

        Returns
        -------
        xr.Dataset
            clipped area of dset
        """
        try:
            dset = dset.rio.clip(
                geometries=ds_region, crs=4326, drop=True, all_touched=all_touched
            )

        except NoDataInBounds as e:
            # Catch small geometry error
            print(e)
            print("Skipping spatial subsetting.")

        return dset

    def _clip_to_geometry_loca(
        dset: xr.Dataset, ds_region: shapely.geometry.polygon.Polygon, all_touched: bool
    ) -> xr.Dataset:
        """Clip to geometry, adding missing grid info
            because crs and x, y are missing from LOCA datasets
            Otherwise rioxarray will raise this error:
            'MissingSpatialDimensionError: x dimension not found.'

        Parameters
        ----------
        dset: xr.Dataset
            one dataset from the catalog
        ds_region: shapely.geometry.polygon.Polygon
            area to clip to
        all_touched: boolean
            select within or touching area

        Returns
        -------
        xr.Dataset
            clipped area of dset
        """
        dset = dset.rename({"lon": "x", "lat": "y"})
        dset = dset.rio.write_crs("epsg:4326", inplace=True)
        dset = _clip_to_geometry(dset, ds_region, all_touched)
        dset = dset.rename({"x": "lon", "y": "lat"}).drop("spatial_ref")
        return dset

    ds_region = area_subset_geometry(selections)

    if ds_region is not None:  # Perform subsetting
        if selections.downscaling_method == "Dynamical":
            dset = _clip_to_geometry(dset, ds_region, selections.all_touched)
        else:
            dset = _clip_to_geometry_loca(dset, ds_region, selections.all_touched)

    return dset


def _process_dset(
    ds_name: str, dset: xr.Dataset, selections: "DataParameters"
) -> xr.Dataset:
    """Subset over time and space, as described in user selections;
       renaming to facilitate concatenation.

    Parameters
    ----------
    ds_name: str
        dataset name from catalog
    dset: xr.Dataset
        one dataset from the catalog
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    xr.Dataset
        sub-setted output data
    """
    # Time slice
    dset = _time_slice(dset, selections)

    # Trim WRF domain edges
    # Clip 10 grid cells around the entire grid
    if selections.downscaling_method == "Dynamical":
        dset = dset.isel(x=slice(10, -10), y=slice(10, -10))

    # Perform area subsetting
    dset = _spatial_subset(dset, selections)

    # Perform area averaging
    if selections.area_average == "Yes":
        dset = area_average(dset)

    def _sim_index_item(ds_name: str, member_id: dict[str, str]) -> str:
        """Identify a simulation by its downscaling type, driving GCM, and member id.

        Parameters
        ----------
        ds_name: str
            dataset name from catalog
        member_id: xr.Dataset.attr
            ensemble member id from dataset attributes

        Returns
        -------
        str: joined by underscores
        """
        downscaling_type = ds_name.split(".")[0]
        gcm_name = ds_name.split(".")[2]
        ensemble_member = str(member_id.values)
        if ensemble_member != "nan":
            return "_".join([downscaling_type, gcm_name, ensemble_member])
        else:
            return "_".join([downscaling_type, gcm_name])

    # Rename member_id value to include more info
    dset = dset.assign_coords(
        member_id=[_sim_index_item(ds_name, mem_id) for mem_id in dset.member_id]
    )
    # Rename variable to display name:
    dset = dset.rename({list(dset.data_vars)[0]: selections.variable})

    return dset


def _override_unit_defaults(da: xr.DataArray, var_id: str) -> xr.DataArray:
    """Override non-standard unit specifications in some dataset attributes

    Parameters
    ----------
    da: xr.DataArray
        any xarray DataArray with a units attribute
    var_id: str
        variable id

    Returns
    -------
    xr.DataArray
        output data
    """
    match var_id:
        case "huss":
            # Units for LOCA specific humidity are set to 1
            # Reset to kg/kg so they can be converted if neccessary to g/kg
            da.attrs["units"] = "kg/kg"
        case "rsds":
            # rsds units are "W m-2"
            # rename them to W/m2 to match the lookup catalog, and the units for WRF radiation variables
            da.attrs["units"] = "W/m2"
    return da


def _merge_all(
    selections: "DataParameters", data_dict: dict[str, xr.core.dataset.Dataset]
) -> xr.DataArray:
    """Merge all datasets into one, subsetting each consistently;
       clean-up format, and convert units.

    Parameters
    ----------
    selections: DataParameters
        object holding user's selections
    data_dict: dict
        dictionary of zarrs from catalog, with each key
        being its name and each item the zarr store

    Returns
    -------
    da: xr.DataArray
        output data
    """
    # Two LOCA2 simulations report a daily timestamp coordinate at 12am (midnight) when the rest of the simulations report at 12pm (noon)
    # Here we reindex the time dimension to shift it by 12HR for the two troublesome simulations
    # This avoids the issue where every other day is set to NaN when you concat the datasets!
    if (selections.downscaling_method in ["Statistical", "Dynamical+Statistical"]) and (
        selections.timescale == "daily"
    ):
        troublesome_sims = ["HadGEM3-GC31-LL", "KACE-1-0-G"]
        for sim in troublesome_sims:
            for dset_name in list(data_dict):
                if sim in dset_name:
                    data_dict[dset_name]["time"] = data_dict[dset_name].time.get_index(
                        "time"
                    ) + timedelta(hours=12)

    # Get corresponding data for historical period to append:
    reconstruction = [one for one in data_dict.keys() if "reanalysis" in one]
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

    def _concat_sims(
        data_dict: dict[str, xr.core.dataset.Dataset],
        hist_data: xr.Dataset,
        selections: "DataParameters",
        scenario: str,
    ) -> xr.Dataset:
        """Combine datasets along expanded 'member_id' dimension, and append
            historical if relevant.

        Parameters
        ----------
        data_dict: dict
            dictionary of zarrs from catalog, with each key
            being its name and each item the zarr store
        hist_data: xr.Dataset
            subsetted historical data to append
        selections: DataParameters
            class holding data selections
        scenario: str
            short designation for one SSP

        Returns
        -------
        one_scenario: xr.Dataset
            combined data object
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
            hist_data = hist_data.sel(member_id=one_scenario.member_id)
            scen_name = "Historical + " + scen_name
            one_scenario = xr.concat([hist_data, one_scenario], dim="time")

        # Set-up coordinate:
        one_scenario = one_scenario.assign_coords({"scenario": scen_name})

        return one_scenario

    def _add_scenario_dim(da: xr.DataArray, scen_name: str) -> xr.DataArray:
        """Add a singleton dimension for 'scenario' to the DataArray.

        Parameters
        ----------
        da: xr.DataArray
            consolidated data object missing a scenario dimension
        scen_name: str
            desired value for scenario along new dimension

        Returns
        -------
        da: xr.DataArray
            data object with singleton scenario dimension added.
        """
        da = da.assign_coords({"scenario": scen_name})
        da = da.expand_dims(dim={"scenario": 1})
        return da

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
        if all_hist:
            all_ssps = all_hist
            all_ssps = _add_scenario_dim(all_ssps, "Historical Climate")
            if reconstruction:
                one_key = reconstruction[0]
                era5_wrf = _process_dset(one_key, data_dict[one_key], selections)
                era5_wrf = _add_scenario_dim(era5_wrf, "Historical Reconstruction")
                all_ssps = xr.concat(
                    [all_ssps, era5_wrf],
                    dim="scenario",
                )
        elif reconstruction:
            one_key = reconstruction[0]
            all_ssps = _process_dset(one_key, data_dict[one_key], selections)
            all_ssps = _add_scenario_dim(all_ssps, "Historical Reconstruction")

    # Rename expanded dimension:
    all_ssps = all_ssps.rename({"member_id": "simulation"})

    # Convert to xr.DataArray:
    var_id = list(all_ssps.data_vars)[0]
    all_ssps = all_ssps[var_id]

    return all_ssps


def _get_data_one_var(selections: "DataParameters") -> xr.DataArray:
    """Get data for one variable
    Retrieves dataset dictionary from AWS, handles some special cases, merges
    datasets along new dimensions into one xr.DataArray, and adds metadata.

    Parameters
    ----------
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    da: xr.DataArray
        with datasets combined over new dimensions 'simulation' and 'scenario'
    """

    orig_units = selections.units

    scenario_ssp, scenario_historical = _get_scenario_from_selections(selections)

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
        {"SSP 2-4.5", "SSP 5-8.5"}.intersection(set(scenario_ssp))
    ):
        method_list = downscaling_method_as_list(selections.downscaling_method)

        cat_subset2 = selections._data_catalog.search(
            activity_id=[downscaling_method_to_activity_id(dm) for dm in method_list],
            table_id=timescale_to_table_id(selections.timescale),
            grid_label=resolution_to_gridlabel(selections.resolution),
            variable_id=selections.variable_id,
            experiment_id=[scenario_to_experiment_id(x) for x in scenario_ssp],
            source_id=["CESM2"],
        )
        data_dict2 = cat_subset2.to_dataset_dict(
            zarr_kwargs={"consolidated": True},
            storage_options={"anon": True},
            progressbar=False,
        )
        data_dict = {**data_dict, **data_dict2}

    # Merge individual Datasets into one DataArray object.
    da = _merge_all(selections=selections, data_dict=data_dict)

    # Set data attributes and name
    native_units = da.attrs["units"]
    data_attrs = _get_data_attributes(selections)
    data_attrs = data_attrs | {"institution": _institution}
    da.attrs = data_attrs
    da.name = selections.variable

    # Convert units
    da.attrs["units"] = native_units
    da = _override_unit_defaults(da, selections.variable_id)
    da = convert_units(da=da, selected_units=orig_units)

    return da


def _get_data_attributes(
    selections: "DataParameters",
) -> dict[str, str | float | int | list[str]]:
    """Return dictionary of xr.DataArray attributes based on selections

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    new_attrs: dict
        attributes
    """
    new_attrs = {  # Add descriptive attributes to DataArray
        "variable_id": ", ".join(
            selections.variable_id
        ),  # Convert list to comma separated string
        "extended_description": selections.extended_description,
        "units": selections.units,
        "data_type": selections.data_type,
        "resolution": selections.resolution,
        "frequency": selections.timescale,
        "location_subset": selections.cached_area,
        "approach": selections.approach,
        "downscaling_method": selections.downscaling_method,
    }

    if selections.approach == "Warming Level":
        new_attrs["warming_level_window"] = "+/- {0} years from centered year".format(
            selections.warming_level_window
        )
    if selections.warming_level_months != list(np.arange(1, 13)):
        # Only add this attribute if it's not a full year
        # Otherwise users can just assume its a full year
        # This option is hidden from the GUI so I think adding the attribute is confusing otherwise
        new_attrs["warming_level_months"] = ", ".join(
            [str(x) for x in selections.warming_level_months]
        )
    return new_attrs


def _check_valid_unit_selection(selections: "DataParameters") -> None:
    """Check that units weren't manually set in DataParameters to an invalid option.
    Raises ValueError if units are not set to a valid option.

    Parameters
    -----------
    selections: DataParameters

    Returns
    -------
    None

    """
    native_unit = selections.variable_options_df[
        selections.variable_options_df["variable_id"].isin(selections.variable_id)
    ].unit.item()

    try:
        # See if there are more than one unit option for this variable
        valid_units = get_unit_conversion_options()[native_unit]
    except:
        # If not, the only unit option is the native unit
        valid_units = [native_unit]

    if selections.units not in valid_units:
        print("Units selected: {}".format(selections.units))
        print("Valid units: " + ", ".join(valid_units))
        raise ValueError("Selected unit is not valid for the selected variable.")
    return None


def _get_Uearth(selections: "DataParameters") -> xr.DataArray:
    """Rotate winds from WRF grid --> spherical earth
    We need to rotate U and V to Earth-relative coordinates in order to properly derive wind direction and generally make comparisons against observations.
    Reference: https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
    Ideally, this should NOT be done programmatically... we should update the zarrs and remove this code

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """
    # Load v10 data
    selections.variable_id = ["v10"]
    v10_da = _get_data_one_var(selections)

    # Load u10 data
    selections.variable_id = ["u10"]
    u10_da = _get_data_one_var(selections)

    # Read in the appropriate file depending on the data resolution
    # This file contains sinalpha and cosalpha for the WRF grid
    gridlabel = resolution_to_gridlabel(selections.resolution)
    wrf_angles_ds = xr.open_zarr(
        "s3://cadcat/tmp/era/wrf/wrf_angles_{}.zarr/".format(gridlabel),
        storage_options={"anon": True},
    )
    wrf_angles_ds = _spatial_subset(
        wrf_angles_ds, selections
    )  # Clip to spatial subset of data
    sinalpha = wrf_angles_ds.SINALPHA
    cosalpha = wrf_angles_ds.COSALPHA

    # Compute Uearth
    Uearth = u10_da * cosalpha - v10_da * sinalpha

    # Add variable name
    Uearth.name = selections.variable

    return Uearth


def _get_Vearth(selections: "DataParameters") -> xr.DataArray:
    """Rotate winds from WRF grid --> spherical earth
    We need to rotate U and V to Earth-relative coordinates in order to properly derive wind direction and generally make comparisons against observations.
    Reference: https://www-k12.atmos.washington.edu/~ovens/wrfwinds.html
    Ideally, this should NOT be done programmatically... we should update the zarrs and remove this code

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """
    # Load u10 data
    selections.variable_id = ["u10"]
    u10_da = _get_data_one_var(selections)

    # Load v10 data
    selections.variable_id = ["v10"]
    v10_da = _get_data_one_var(selections)

    # Read in the appropriate file depending on the data resolution
    # This file contains sinalpha and cosalpha for the WRF grid
    gridlabel = resolution_to_gridlabel(selections.resolution)
    wrf_angles_ds = xr.open_zarr(
        "s3://cadcat/tmp/era/wrf/wrf_angles_{}.zarr/".format(gridlabel),
        storage_options={"anon": True},
    )
    wrf_angles_ds = _spatial_subset(
        wrf_angles_ds, selections
    )  # Clip to spatial subset of data
    sinalpha = wrf_angles_ds.SINALPHA
    cosalpha = wrf_angles_ds.COSALPHA

    # Compute Uearth
    Vearth = v10_da * cosalpha + u10_da * sinalpha

    # Add variable name
    Vearth.name = selections.variable

    return Vearth


def _get_wind_speed_derived(selections: "DataParameters") -> xr.DataArray:
    """Get input data and derive wind speed for hourly data

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """
    # Load u10 data
    selections.variable_id = ["u10"]
    selections.units = (
        "m s-1"  # Need to set units to required units for compute_wind_mag
    )
    # u10_da = _get_data_one_var(selections)
    u10_da = _get_Uearth(selections)

    # Load v10 data
    selections.variable_id = ["v10"]
    selections.units = "m s-1"
    # v10_da = _get_data_one_var(selections)
    v10_da = _get_Vearth(selections)

    # Derive the variable
    da = compute_wind_mag(u10=u10_da, v10=v10_da)  # m/s
    return da


def _get_wind_dir_derived(selections: "DataParameters") -> xr.DataArray:
    """Get input data and derive wind direction for hourly data

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """
    # Load u10 data
    selections.variable_id = ["u10"]
    selections.units = (
        "m s-1"  # Need to set units to required units for compute_wind_mag
    )
    # u10_da = _get_data_one_var(selections)
    u10_da = _get_Uearth(selections)

    # Load v10 data
    selections.variable_id = ["v10"]
    selections.units = "m s-1"
    # v10_da = _get_data_one_var(selections)
    v10_da = _get_Vearth(selections)

    # Derive the variable
    da = compute_wind_dir(u10=u10_da, v10=v10_da)
    return da


def _get_monthly_daily_dewpoint(selections: "DataParameters") -> xr.DataArray:
    """Derive dew point temp for monthly/daily data.

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """
    # Daily/monthly dew point inputs have different units
    # Hourly dew point temp derived differently because you also have to derive relative humidity

    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "K"  # Kelvin required for humidity and dew point computation
    t2_da = _get_data_one_var(selections)

    selections.variable_id = ["rh"]
    selections.units = "[0 to 100]"
    rh_da = _get_data_one_var(selections)

    # Derive dew point temperature
    # Returned in units of Kelvin
    da = compute_dewpointtemp(temperature=t2_da, rel_hum=rh_da)  # Kelvin  # [0-100]
    return da


def _get_hourly_dewpoint(selections: "DataParameters") -> xr.DataArray:
    """Derive dew point temp for hourly data.
    Requires first deriving relative humidity.

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """
    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "degC"  # Celsius required for humidity
    t2_da = _get_data_one_var(selections)

    # Load mixing ratio data
    selections.variable_id = ["q2"]
    selections.units = "g kg-1"
    q2_da = _get_data_one_var(selections)

    # Load pressure data
    selections.variable_id = ["psfc"]
    selections.units = "hPa"
    pressure_da = _get_data_one_var(selections)

    # Derive relative humidity
    # Returned in units of [0-100]
    rh_da = compute_relative_humidity(
        pressure=pressure_da,  # hPa
        temperature=t2_da,  # degC
        mixing_ratio=q2_da,  # g/kg
    )

    # Dew point temperature requires temperature in Kelvin
    t2_da = convert_units(t2_da, "K")

    # Derive dew point temperature
    # Returned in units of Kelvin
    da = compute_dewpointtemp(temperature=t2_da, rel_hum=rh_da)  # Kelvin  # [0-100]
    return da


def _get_hourly_rh(selections: "DataParameters") -> xr.DataArray:
    """Derive hourly relative humidity.

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """
    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "degC"  # Celsius required for humidity
    t2_da = _get_data_one_var(selections)

    # Load mixing ratio data
    selections.variable_id = ["q2"]
    selections.units = "g kg-1"
    q2_da = _get_data_one_var(selections)

    # Load pressure data
    selections.variable_id = ["psfc"]
    selections.units = "hPa"
    pressure_da = _get_data_one_var(selections)

    # Derive relative humidity
    # Returned in units of [0-100]
    da = compute_relative_humidity(
        pressure=pressure_da,  # hPa
        temperature=t2_da,  # degC
        mixing_ratio=q2_da,  # g/kg
    )
    return da


def _get_hourly_specific_humidity(selections: "DataParameters") -> xr.DataArray:
    """Derive hourly specific humidity.
    Requires first deriving relative humidity, then dew point temp.

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """
    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "degC"  # degC required for humidity
    t2_da = _get_data_one_var(selections)

    # Load mixing ratio data
    selections.variable_id = ["q2"]
    selections.units = "g kg-1"
    q2_da = _get_data_one_var(selections)

    # Load pressure data
    selections.variable_id = ["psfc"]
    selections.units = "hPa"
    pressure_da = _get_data_one_var(selections)

    # Derive relative humidity
    # Returned in units of [0-100]
    rh_da = compute_relative_humidity(
        pressure=pressure_da,  # hPa
        temperature=t2_da,  # degC
        mixing_ratio=q2_da,  # g/kg
    )

    # Dew point temperature requires temperature in Kelvin
    t2_da = convert_units(t2_da, "K")

    # Derive dew point temperature
    # Returned in units of Kelvin
    dew_pnt_da = compute_dewpointtemp(
        temperature=t2_da, rel_hum=rh_da  # Kelvin  # [0-100]
    )

    # Derive specific humidity
    # Returned in units of g/kg
    da = compute_specific_humidity(
        tdps=dew_pnt_da, pressure=pressure_da  # Kelvin  # Pa
    )
    return da


def _get_noaa_heat_index(selections: "DataParameters") -> xr.DataArray:
    """Derive NOAA heat index

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """

    # Load mixing ratio data
    selections.variable_id = ["q2"]
    selections.units = "kg kg-1"
    q2_da = _get_data_one_var(selections)

    # Load pressure data
    selections.variable_id = ["psfc"]
    selections.units = "Pa"
    pressure_da = _get_data_one_var(selections)

    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "K"  # Kelvin required for humidity and dew point computation
    t2_da_K = _get_data_one_var(selections)

    # Derive relative humidity
    # Returned in units of [0-100]
    rh_da = compute_relative_humidity(
        pressure=pressure_da,  # Pa
        temperature=t2_da_K,  # Kelvin
        mixing_ratio=q2_da,  # kg/kg
    )

    # Convert temperature to proper units for noaa heat index
    t2_da_F = convert_units(t2_da_K, "degF")

    # Derive index
    # Returned in units of F
    da = noaa_heat_index(T=t2_da_F, RH=rh_da)
    return da


def _get_eff_temp(selections: "DataParameters") -> xr.DataArray:
    """Derive the effective temperature

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """

    # Load temperature data
    selections.variable_id = ["t2"]
    t2_da = _get_data_one_var(selections)

    # Derive effective temp
    da = effective_temp(T=t2_da)
    return da


def _get_fosberg_fire_index(selections: "DataParameters") -> xr.DataArray:
    """Derive the fosberg fire index.

    Parameters
    ----------
    selections: DataParameters

    Returns
    -------
    da: xr.DataArray
    """

    # Hard set timescale to hourly
    orig_timescale = selections.timescale  # Preserve original user selection
    selections.timescale = "hourly"

    # Load temperature data
    selections.variable_id = ["t2"]
    selections.units = "degC"  # Kelvin required for humidity
    t2_da_C = _get_data_one_var(selections)

    # Load mixing ratio data
    selections.variable_id = ["q2"]
    selections.units = "g kg-1"
    q2_da = _get_data_one_var(selections)

    # Load pressure data
    selections.variable_id = ["psfc"]
    selections.units = "hPa"
    pressure_da = _get_data_one_var(selections)

    # Load u10 data
    selections.variable_id = ["u10"]
    selections.units = (
        "m s-1"  # Need to set units to required units for compute_wind_mag
    )
    u10_da = _get_data_one_var(selections)

    # Load v10 data
    selections.variable_id = ["v10"]
    selections.units = "m s-1"
    v10_da = _get_data_one_var(selections)

    # Derive relative humidity
    # Returned in units of [0-100]
    rh_da = compute_relative_humidity(
        pressure=pressure_da,  # hPa
        temperature=t2_da_C,  # degC
        mixing_ratio=q2_da,  # g/kg
    )

    # Derive windspeed
    # Returned in units of m/s
    windspeed_da_ms = compute_wind_mag(u10=u10_da, v10=v10_da)  # m/s

    # Convert units to proper units for fosberg index
    t2_da_F = convert_units(t2_da_C, "degF")
    windspeed_da_mph = convert_units(windspeed_da_ms, "mph")

    # Compute the index
    da = fosberg_fire_index(
        t2_F=t2_da_F, rh_percent=rh_da, windspeed_mph=windspeed_da_mph
    )

    return da


def read_catalog_from_select(selections: "DataParameters") -> xr.DataArray:
    """The primary and first data loading method, called by
    DataParameters.retrieve, it returns a DataArray (which can be quite large)
    containing everything requested by the user (which is stored in 'selections').

    Parameters
    ----------
    selections: DataParameters
        object holding user's selections

    Returns
    -------
    da: xr.DataArray
        output data
    """

    if selections.approach == "Warming Level":
        selections.time_slice = (1980, 2100)  # Retrieve entire time period

    # Raise appropriate errors for time-based retrieval
    if selections.approach == "Time":
        if (selections.scenario_ssp != []) and (
            "Historical Reconstruction" in selections.scenario_historical
        ):
            raise ValueError(
                "Historical Reconstruction data is not available with SSP data. Please modify your selections and try again."
            )

        # Validate unit selection
        # Returns None if units are valid, raises error if not
        _check_valid_unit_selection(selections)

        # Raise error if no scenarios are selected
        scenario_selections = selections.scenario_ssp + selections.scenario_historical
        if scenario_selections == []:
            raise ValueError("Please select as least one dataset.")

    # Raise error if station data selected, but no station is selected
    if (selections.data_type == "Stations") and (
        selections.stations in [[], ["No stations available at this location"]]
    ):
        raise ValueError(
            "Please select at least one weather station, or retrieve gridded data."
        )

    # For station data, need to expand time slice to ensure the historical period is included
    # At the end, the data will be cut back down to the user's original selection
    if selections.data_type == "Stations":
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

    ## ------ Deal with derived variables ------
    orig_var_id_selection = selections.variable_id[0]
    orig_unit_selection = selections.units
    orig_variable_selection = selections.variable

    # Get data attributes beforehand since selections is modified
    data_attrs = _get_data_attributes(selections)
    if "_derived" in orig_var_id_selection:
        match orig_var_id_selection:
            case "wind_speed_derived":  # Hourly
                da = _get_wind_speed_derived(selections)
            case "wind_direction_derived":  # Hourly
                da = _get_wind_dir_derived(selections)
            case "dew_point_derived":  # Monthly/daily
                da = _get_monthly_daily_dewpoint(selections)
            case "dew_point_derived_hrly":  # Hourly
                da = _get_hourly_dewpoint(selections)
            case "rh_derived":  # Hourly
                da = _get_hourly_rh(selections)
            case "q2_derived":  # Hourly
                da = _get_hourly_specific_humidity(selections)
            case "fosberg_index_derived":  # Hourly
                da = _get_fosberg_fire_index(selections)
            case "noaa_heat_index_derived":  # Hourly
                da = _get_noaa_heat_index(selections)
            case "effective_temp_index_derived":
                da = _get_eff_temp(selections)
            case _:  # none of the above
                raise ValueError(
                    "You've encountered a bug. No data available for selected derived variable."
                )

        # ------ Set attributes ------
        # Some of the derived variables may be constructed from data that comes from the same institution
        # The dev team hasn't looked into this yet -- opportunity for future improvement
        data_attrs = data_attrs | {"institution": "Multiple"}
        da.attrs = data_attrs

        # Convert units
        da = convert_units(da, selected_units=orig_unit_selection)
        da.name = orig_variable_selection  # Set name of DataArray

        # Reset selections to user's original selections
        selections.variable_id = [orig_var_id_selection]
        selections.units = orig_unit_selection

    # Rotate wind vectors
    elif (
        any(x in selections.variable_id for x in ["u10", "v10"])
        and selections.downscaling_method == "Dynamical"
    ):
        if "u10" in selections.variable_id:
            da = _get_Uearth(selections)
        elif "v10" in selections.variable_id:
            da = _get_Vearth(selections)

    # Any other variable... i.e. not an index, derived var, or a WRF wind vector
    else:
        da = _get_data_one_var(selections)

    # Assure that CRS and grid_mapping are in place for all data returned
    if (selections.downscaling_method == "Dynamical") and (
        "Lambert_Conformal" in da.coords
    ):
        da.attrs = da.attrs | {"grid_mapping": "Lambert_Conformal"}
    elif selections.downscaling_method in ["Statistical", "Dynamical+Statistical"]:
        da = da.rio.write_crs("epsg:4326", inplace=True)

    if selections.data_type == "Stations":
        # Bias-correct the station data
        da = _station_apply(selections, da, original_time_slice)

        # Reset original selections
        if "Historical Climate" not in original_scenario_historical:
            selections.scenario_historical.remove("Historical Climate")
            da["scenario"] = [x.split("Historical + ")[1] for x in da.scenario.values]
        selections.time_slice = original_time_slice

    if selections.approach == "Warming Level":
        # Process data object using warming levels approach
        # Dimensions and coordinates will change
        # See function documentation for more information
        da = _apply_warming_levels_approach(da, selections)

        # Reset original selections
        selections.scenario_ssp = ["n/a"]
        selections.scenario_historical = ["n/a"]

    return da


def _apply_warming_levels_approach(
    da: xr.DataArray, selections: "DataParameters"
) -> xr.DataArray:
    """
    Apply warming levels approach to data object.
    Internal function only-- many settings are set in the backend for this function to work appropriately.

    Parameters
    ----------
    da: xr.DataArray
        Object returned by _get_data_one_var for a time-based approach
        Needs to have simulation, scenario, and time dimension.
        Time needs to be from 1980-2100.
        Historical Climate must be appended.
    selections: DataParameters
        Data settings (variable, unit, timescale, etc).
        selections.approach must be "Warming Level".

    Returns
    -------
    warming_data: xr.DataArray
        Object with dimensions warming_level, time_delta, simulation, and spatial coordinates
        "simulation" dimension reflects the simulation+scenario combo from the time-based approach; i.e. it is the coordinate returned by stacking both simulation and scenario dimensions.
        "time_delta" dimensions reflects the hours/days/months (depends on user selections for timescale) from the central year
    """

    # Stack by simulation and scenario to combine the coordinates into a single dimension
    data_stacked = da.stack(all_sims=["simulation", "scenario"])
    # The xarray stacking function results in some non-existant scenario/simulation combos
    # We need to drop them here such that the global warming levels table can be adequately parsed by the calculate_warming_level function
    data_stacked = drop_invalid_sims(data_stacked, selections)

    da_list = []
    for level in selections.warming_level:
        gwl_table = selections._warming_level_times

        if level not in WARMING_LEVELS:
            gwl_table = create_new_warming_level_table(warming_level=level)

        da_by_wl = calculate_warming_level(
            data_stacked,
            gwl_times=gwl_table,
            level=level,
            months=selections.warming_level_months,
            window=selections.warming_level_window,
        )
        da_list.append(da_by_wl)

    # Concatenate along warming levels dimension
    warming_data = xr.concat(da_list, dim="warming_level")

    # Need to remove confusing MultiIndex dimension
    # And add back in the simulation + scenario information in a better format
    # Do we think these are the right names??
    sim_and_scen_str = [
        x[0]
        + "_historical+"
        + scenario_to_experiment_id(x[1].split("Historical + ")[1])
        for x in warming_data["all_sims"].values
    ]
    warming_data = warming_data.drop(["simulation", "scenario"])
    warming_data["all_sims"] = sim_and_scen_str

    # Add descriptive attributes to coordinates
    freq_strs = {"monthly": "months", "daily": "days", "hourly": "hours"}
    warming_data["time"].attrs = {
        "description": freq_strs[warming_data.frequency] + " from center year"
    }
    warming_data["centered_year"].attrs = {
        "description": "central year in +/-{0} year window".format(
            selections.warming_level_window
        )
    }
    warming_data["warming_level"].attrs = {
        "description": "degrees Celsius above the historical baseline"
    }
    warming_data["all_sims"].attrs = {"description": "combined simulation and scenario"}

    # Give a better name to the "all_sims" and "time" dimension to better reflect the warming levels approach
    warming_data = warming_data.rename({"all_sims": "simulation", "time": "time_delta"})

    return warming_data


def _station_apply(
    selections: "DataParameters", da: xr.DataArray, original_time_slice: tuple[int, int]
) -> xr.DataArray:
    """Use xr.apply to get bias corrected data to station

    Parameters
    ----------
    selections: DataParameters
        object holding user's selections
    da: xr.DataArray
    original_time_slice: tuple

    Returns
    -------
    apply_output: xr.DataArray
        output data
    """
    # Grab zarr data
    station_subset = selections._stations_gdf.loc[
        selections._stations_gdf["station"].isin(selections.stations)
    ]
    filepaths = [
        "s3://cadcat/hadisd/HadISD_{}.zarr".format(s_id)
        for s_id in station_subset["station id"]
    ]

    def _preprocess_hadisd(
        ds: xr.Dataset, stations_gdf: gpd.GeoDataFrame
    ) -> xr.Dataset:
        """
        Preprocess station data so that it can be more seamlessly integrated into the wrangling process
        Get name of station id and station name
        Rename data variable to the station name; this allows the return of a Dataset object, with each unique station as a data variable
        Convert Celsius to Kelvin
        Assign descriptive attributes
        Drop unneccessary coordinates that can cause issues when bias correcting with the model data

        Parameters
        ----------
        ds: xr.Dataset
            data for a single HadISD station
        stations_gdf: gpd.GeoDataFrame
            station data frame

        Returns
        -------
        ds: xr.Dataset

        """
        # Get station ID from file name
        station_id = ds.encoding["source"].split("HadISD_")[1].split(".zarr")[0]
        # Get name of station from station_id
        station_name = stations_gdf.loc[stations_gdf["station id"] == int(station_id)][
            "station"
        ].item()
        # Rename data variable to station name
        ds = ds.rename({"tas": station_name})
        # Convert Celsius to Kelvin
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
        ds = ds.drop_vars(["elevation", "latitude", "longitude"])
        return ds

    _partial_func = partial(_preprocess_hadisd, stations_gdf=selections._stations_gdf)

    station_ds = xr.open_mfdataset(
        filepaths,
        preprocess=_partial_func,
        engine="zarr",
        consolidated=False,
        parallel=True,
        backend_kwargs=dict(storage_options={"anon": True}),
    )

    def _get_bias_corrected_closest_gridcell(
        station_da: xr.DataArray, gridded_da: xr.DataArray, time_slice: tuple[int, int]
    ) -> xr.DataArray:
        """Get the closest gridcell to a weather station.
        Bias correct the data using historical station data

        Parameters
        ----------
        station_da: xr.DataArray
        gridded_da: xr.DataArray
        time_slice: tuple

        Returns
        -------
        bias_corrected: xr.DataArray
        """
        # Get the closest gridcell to the station
        station_lat, station_lon = station_da.attrs["coordinates"]
        gridded_da_closest_gridcell = get_closest_gridcell(
            gridded_da, station_lat, station_lon, print_coords=False
        )

        # Droop any coordinates in the output dataset that are not also dimensions
        # This makes merging all the stations together easier and drops superfluous coordinates
        gridded_da_closest_gridcell = gridded_da_closest_gridcell.drop_vars(
            [
                i
                for i in gridded_da_closest_gridcell.coords
                if i not in gridded_da_closest_gridcell.dims
            ]
        )

        def _bias_correct_model_data(
            obs_da: xr.DataArray,
            gridded_da: xr.DataArray,
            time_slice: tuple[int, int],
            window: int = 90,
            nquantiles: int = 20,
            group: str = "time.dayofyear",
            kind: str = "+",
        ) -> xr.DataArray:
            """Bias correct model data using observational station data
            Converts units of the station data to whatever the input model data's units are
            Converts calendars of both datasets to a no leap calendar
            Time slices the data
            Performs bias correction

            Parameters
            ----------
            obs_da: xr.DataArray
                station data, preprocessed with the function _preprocess_hadisd
            gridded_da: xr.DataArray
                input model data
            time_slice: tuple
                temporal slice to cut gridded_da to, after bias correction
            window: int
                window of days +/-
            nquantiles: int
                number of quantiles
            group: str
                time frequency to group data by
            kind: str
                the adjustment kind, either additive or multiplicative

            Returns
            -------
            da_adj: xr.DataArray
                output data

            """
            # Get group by window
            # Use 90 day window (+/- 45 days) to account for seasonality
            grouper = Grouper(group, window=window)

            # Convert units to whatever the gridded data units are
            obs_da = convert_units(obs_da, gridded_da.units)
            # Rechunk data. Cannot be chunked along time dimension
            # Error raised by xclim: ValueError: Multiple chunks along the main adjustment dimension time is not supported.
            gridded_da = gridded_da.chunk(chunks=dict(time=-1))
            obs_da = obs_da.sel(time=slice(obs_da.time.values[0], "2014-08-31"))
            obs_da = obs_da.chunk(chunks=dict(time=-1))
            # Convert calendar to no leap year
            obs_da = obs_da.convert_calendar("noleap")
            gridded_da = gridded_da.convert_calendar("noleap")
            # Data at the desired time slice
            data_sliced = gridded_da.sel(
                time=slice(str(time_slice[0]), str(time_slice[1]))
            )
            # Input data, sliced to time period of observational data
            gridded_da = gridded_da.sel(
                time=slice(str(obs_da.time.values[0]), str(obs_da.time.values[-1]))
            )
            # Observational data sliced to time period of input data
            obs_da = obs_da.sel(
                time=slice(
                    str(gridded_da.time.values[0]), str(gridded_da.time.values[-1])
                )
            )
            # Get QDS
            QDM = QuantileDeltaMapping.train(
                obs_da,
                gridded_da,
                nquantiles=nquantiles,
                group=grouper,
                kind=kind,
            )
            # Bias correct the data
            da_adj = QDM.adjust(data_sliced)
            da_adj.name = gridded_da.name  # Rename it to get back to original name
            da_adj["time"] = da_adj.indexes["time"].to_datetimeindex()

            return da_adj

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

    apply_output = station_ds.map(
        _get_bias_corrected_closest_gridcell,
        keep_attrs=False,
        gridded_da=da,
        time_slice=original_time_slice,
    )
    return apply_output
