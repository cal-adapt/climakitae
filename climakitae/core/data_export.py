"""Backend functions for exporting data."""

import os
import shutil
import warnings
import datetime
import xarray as xr
import pandas as pd
import rasterio
from importlib.metadata import version as _version

from climakitae.core.data_interface import DataInterface

xr.set_options(keep_attrs=True)


def _export_to_netcdf(data_to_export, save_name, **kwargs):
    """
    exports user-selected data to netCDF format.
    this function is called from the _export_to_user
    function if the user selected netCDF output.

    data_to_export: xarray dataset or array to export
    save_name: string corresponding to desired output file name + file extension
    kwargs: reserved for future use
    """
    print("Alright, exporting specified data to NetCDF.")
    comp = dict(_FillValue=None)
    encoding = {coord: comp for coord in data_to_export.coords}
    data_to_export.to_netcdf(save_name, encoding=encoding)


def _add_unit_to_header(df, variable, unit):
    """
    Add variable unit to data table header.

    Insert a 2nd row into the header of the DataFrame `df` to include the
    `unit` associated with the `variable` column.

    Parameters
    ----------
    df : pandas.DataFrame
        data table to update
    variable : string
        name of the variable column
    unit : string
        unit associated with the variable

    Returns
    -------
    pandas.DataFrame
        data table with the variable unit added to its header

    """
    df.columns = pd.MultiIndex.from_tuples(
        [(col, unit) if col == variable else (col, "") for col in df.columns],
        name=["variable", "unit"],
    )
    df.reset_index(inplace=True)  # simplifies header
    return df


def _export_to_csv(data_to_export, save_name, **kwargs):
    """
    Export user-selected data to CSV format.

    Export the xarray DataArray `data_to_export` to a CSV file named
    `save_name`. This function is called from the `_export_to_user`
    function if the user selected CSV output.

    Parameters
    ----------
    data_to_export : xarray.DataArray
        data to export to CSV format
    save_name : string
        desired output file name, including the file extension

    Returns
    -------
    None

    """
    if not data_to_export.name:
        # name it in order to call to_dataframe on it
        data_to_export.name = "data"

    # ease column access in R
    data_to_export.name = (
        data_to_export.name.replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("-", "_")
    )

    df = data_to_export.to_dataframe()

    if "units" in data_to_export.attrs and data_to_export.attrs["units"] is not None:
        unit = data_to_export.attrs["units"]
        variable = data_to_export.name
        df = _add_unit_to_header(df, variable, unit)

    excel_row_limit = 1048576
    excel_column_limit = 16384
    csv_nrows, csv_ncolumns = df.shape
    if csv_nrows > excel_row_limit or csv_ncolumns > excel_column_limit:
        warnings.warn(
            f"Dataset exceeds Excel limits of {excel_row_limit} rows "
            f"and {excel_column_limit} columns."
        )

    _metadata_to_file(data_to_export, save_name)
    df.to_csv(save_name, compression="gzip")


def _export_to_geotiff(data_to_export, save_name, **kwargs):
    """
    exports user-selected data to geoTIFF format.
    this function is called from the _export_to_user
    function if the user selected geoTIFF output.

    data_to_export: xarray dataset or array to export
    save_name: string corresponding to desired output file name + file extension
    kwargs: reserved for future use
    """
    ds_attrs = data_to_export.attrs

    # squeeze singleton dimensions as long as they are
    # simulation and/or scenario dimensions;
    # retain simulation and/or scenario metadata
    if "scenario" in data_to_export.coords and "scenario" not in data_to_export.dims:
        scen_attrs = {"scenario": str(data_to_export.coords["scenario"].values)}
        ds_attrs = dict(ds_attrs, **scen_attrs)
    if "scenario" in data_to_export.dims and len(data_to_export.scenario) == 1:
        scen_attr = {"scenario": str(data_to_export.scenario.values[0])}
        ds_attrs = dict(ds_attrs, **scen_attr)
        data_to_export = data_to_export.squeeze(dim="scenario")
    elif (
        "scenario" not in data_to_export.dims
        and "scenario" not in data_to_export.coords
    ):
        warnings.warn(
            (
                "'scenario' not in data array as"
                " dimension or coordinate; this information"
                " will be lost on export to raster."
                " Either provide a data array"
                " which contains a single scenario"
                " as a dimension and/or coordinate,"
                " or record the scenario sampled"
                " for your records."
            )
        )

    if (
        "simulation" in data_to_export.coords
        and "simulation" not in data_to_export.dims
    ):
        sim_attrs = {"simulation": str(data_to_export.coords["simulation"].values)}
        ds_attrs = dict(ds_attrs, **sim_attrs)
    if str("simulation") in data_to_export.dims and len(data_to_export.simulation) == 1:
        sim_attrs = {"simulation": str(data_to_export.simulation.values)}
        ds_attrs = dict(ds_attrs, **sim_attrs)
        data_to_export = data_to_export.squeeze(dim="simulation")
    elif (
        "simulation" not in data_to_export.dims
        and "simulation" not in data_to_export.coords
    ):
        warnings.warn(
            (
                "'simulation' not in data array as"
                " dimension or coordinate; this information"
                " will be lost on export to raster."
                " Either provide a data array"
                " which contains a single simulation"
                " as a dimension and/or coordinate,"
                " or record the simulation sampled"
                " for your records."
            )
        )

    ndim = len(data_to_export.dims)
    if ndim == 3:
        if "time" in data_to_export.dims:
            data_to_export = data_to_export.transpose("time", "y", "x")
            if len(data_to_export.time) > 1:
                print(
                    (
                        "Saving as multiband raster in which"
                        " each band corresponds to a time step."
                    )
                )
        elif "simulation" in data_to_export.dims:
            data_to_export = data_to_export.transpose("simulation", "y", "x")
            if len(data_to_export.simulation) > 1:
                print(
                    (
                        "Saving as multiband raster in which"
                        " each band corresponds to a simulation."
                    )
                )
        elif "scenario" in data_to_export.dims:
            data_to_export = data_to_export.transpose("scenario", "y", "x")
            if len(data_to_export.scenario) > 1:
                print(
                    (
                        "Saving as multiband raster in which"
                        " each band corresponds to a climate scenario."
                    )
                )

    print("Saving as GeoTIFF...")
    data_to_export.rio.to_raster(save_name)
    meta_data_dict = ds_attrs

    with rasterio.open(save_name, "r+") as raster:
        raster.update_tags(**meta_data_dict)
        raster.close()


def export_dataset(data_to_export, file_name, **kwargs):
    """
    The data export method, called by core.Application.export_dataset. Saves
    a dataset to the current working directory in the output
    format requested by the user (which is stored in 'user_export_format').

    user_export_format: pulled from dropdown called by app.export_as()
    data_to_export: xarray ds or da to export
    file_name: string corresponding to desired output file name
    kwargs: variable, scenario, and simulation (as needed)
    """
    ftype = type(data_to_export)

    if ftype not in [xr.core.dataset.Dataset, xr.core.dataarray.DataArray]:
        raise Exception(
            "Cannot export object of type "
            + str(ftype).strip("<class >")
            + ". Please pass an xarray dataset or data array."
        )
    ndims = len(data_to_export.dims)

    if type(file_name) is not str:
        raise Exception(
            (
                "Please pass a string"
                " (any characters surrounded by quotation marks)"
                " for your file name."
            )
        )
    file_name = file_name.split(".")[0]

    req_format = DataInterface().export_type

    if req_format is None:
        raise Exception("Please select a file format from the dropdown menu.")

    extension_dict = {"NetCDF": ".nc", "CSV": ".csv.gz", "GeoTIFF": ".tif"}

    save_name = "./" + file_name + extension_dict[req_format]

    if os.path.exists(save_name):
        raise Exception(
            "File "
            + save_name
            + (
                " exists, please either delete that file from the work"
                " space or specify a new file name here."
            )
        )

    ds_attrs = data_to_export.attrs
    ct = datetime.datetime.now()
    ct_str = ct.strftime("%d-%b-%Y (%H:%M)")

    ck_attrs = {
        "Data_exported_from": "Cal-Adapt Analytics Engine",
        "Data_export_timestamp": ct_str,
        "Analysis_package_name": "climakitae",
        "Version": _version,
        "Author": "Cal-Adapt Analytics Engine Team",
        "Author_email": "analytics@cal-adapt.org",
        "Home_page": "https://github.com/cal-adapt/climakitae",
        "License": "BSD 3-Clause License",
    }

    # metadata stuff
    ds_attrs.update(ck_attrs)
    data_to_export.attrs = ds_attrs

    # now check file size and avail workspace disk space
    # raise error for not enough space
    # and warning for large file
    file_size_threshold = 5  # in GB
    bytes_per_gigabyte = 1024 * 1024 * 1024
    disk_space = shutil.disk_usage("./")[2] / bytes_per_gigabyte
    data_size = data_to_export.nbytes / bytes_per_gigabyte

    if disk_space <= data_size:
        raise Exception(
            "Not enough disk space to export data! You need at least "
            + str(data_size)
            + (
                " GB free in the hub directory, which has 10 GB total space."
                " Try smaller subsets of space, time, scenario, and/or"
                " simulation; pick a coarser spatial or temporal scale;"
                " or clean any exported datasets which you have already"
                " downloaded or do not want."
            )
        )

    if data_size > file_size_threshold:
        print(
            "WARNING: xarray dataset size = "
            + str(data_size)
            + " GB. This might take a while!"
        )

    # now here is where exporting actually begins
    # we will have different functions for each file type
    # to keep things clean-ish
    if "NetCDF" in req_format:
        _export_to_netcdf(data_to_export, save_name, **kwargs)
    else:
        if ftype == xr.core.dataset.Dataset:
            dv_list = list(data_to_export.data_vars)
            if len(dv_list) > 1:
                raise Exception(
                    (
                        "We cannot convert multivariate datasets"
                        " to CSV or GeoTiff at this time. Please supply"
                        " a dataset or array with a single data variable."
                        " A single variable array can be extracted"
                        " from a multivariate dataset like so:"
                        " app.export_dataset(ds['var'],'filename')"
                        " where ds is the dataset or data array"
                        " you attempted to export, and 'var' is a data"
                        " variable (in single or double quotes)."
                    )
                )
            else:
                var_name = dv_list[0]
                data_to_export = data_to_export.to_array()
                data_to_export.name = var_name

        if "CSV" in req_format:
            _export_to_csv(data_to_export, save_name, **kwargs)

        elif "GeoTIFF" in req_format:
            # sometimes "variable" might be a singleton dimension:
            data_to_export = data_to_export.squeeze()

            # if x and/or y exist as coordinates
            # but have been squeezed out as dimensions
            # (eg we have point data), add them back in as dimensions.
            # rasters require both x and y dimensions
            if "x" not in data_to_export.dims:
                if "x" in data_to_export.coords:
                    data_to_export = data_to_export.expand_dims("x")
                else:
                    raise Exception(
                        (
                            "No x dimension or coordinate exists;"
                            " cannot export to GeoTIFF. Please provide"
                            " a data array with both x and y"
                            " spatial coordinates."
                        )
                    )
            if "y" not in data_to_export.dims:
                if "y" in data_to_export.coords:
                    data_to_export = data_to_export.expand_dims("y")
                else:
                    raise Exception(
                        (
                            "No y dimension or coordinate exists;"
                            " cannot export to GeoTIFF. Please provide"
                            " a data array with both x and y"
                            " spatial coordinates."
                        )
                    )

            dim_check = data_to_export.isel(x=0, y=0).squeeze().shape

            if sum([int(dim > 1) for dim in dim_check]) > 1:
                dim_list = data_to_export.dims
                shape_list = data_to_export.shape
                dim_shape = str(
                    [str(d) + ": " + str(s) for d, s in list(zip(dim_list, shape_list))]
                )
                raise Exception(
                    (
                        "Too many non-spatial dimensions"
                        " with length > 1 -- cannot convert"
                        " to GeoTIFF. Current dimensionality is "
                    )
                    + dim_shape
                    + ". Please subset your selection accordingly."
                )

            _export_to_geotiff(data_to_export, save_name, **kwargs)

    return print(
        (
            "Saved! You can find your file(s) in the panel to the left"
            " and download to your local machine from there."
        )
    )


def _metadata_to_file(ds, output_name):
    """
    Writes NetCDF metadata to a txt file so users can still access it
    after exporting to a CSV.
    """

    def _rchop(s, suffix):
        if suffix and s.endswith(suffix):
            return s[: -len(suffix)]
        return s

    output_name = _rchop(output_name, ".csv.gz")

    if os.path.exists(output_name + "_metadata.txt"):
        os.remove(output_name + "_metadata.txt")

    print(
        "NOTE: File metadata will be written in "
        + output_name
        + (
            "_metadata.txt. We recommend you download this along with "
            "the CSV for your records."
        )
    )

    with open(output_name + "_metadata.txt", "w") as f:
        f.write("======== Metadata for CSV file " + output_name + " ========")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("===== Global file attributes =====")
        f.write("\n")
        f.write("Name: " + ds.name)
        f.write("\n")
        for att_keys, att_values in list(zip(ds.attrs.keys(), ds.attrs.values())):
            f.write(str(att_keys) + " : " + str(att_values))
            f.write("\n")

        f.write("\n")
        f.write("\n")
        f.write("===== Coordinate descriptions =====")
        f.write("\n")
        f.write("Note: coordinate values are in the CSV")
        f.write("\n")

        for coord in ds.coords:
            f.write("\n")
            f.write("== " + str(coord) + " ==")
            f.write("\n")
            for att_keys, att_values in list(
                zip(ds[coord].attrs.keys(), ds[coord].attrs.values())
            ):
                f.write(str(att_keys) + " : " + str(att_values))
                f.write("\n")
