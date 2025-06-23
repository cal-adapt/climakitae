import datetime
import logging
import os
import shutil
import urllib
import warnings
from importlib.metadata import version as _version
from math import prod

import boto3
import botocore
import fsspec
import numpy as np
import pandas as pd
import pytz
import requests
import xarray as xr
from botocore.exceptions import ClientError
from timezonefinder import TimezoneFinder

from climakitae.core.paths import (
    export_s3_bucket,
    stations_csv_path,
    variable_descriptions_csv_path,
)
from climakitae.util.utils import read_csv_file

xr.set_options(keep_attrs=True)
bytes_per_gigabyte = 1024 * 1024 * 1024


def remove_zarr(filename: str):
    """Remove Zarr directory structure helper function. As Zarr format is a directory
    tree it is not easily removed using JupyterHUB GUI. This function simply deletes
    an entire directory tree.

    Parameters
    ----------
    filename : str
        Output Zarr file name (without file extension, i.e. "my_filename" instead
        of "my_filename.zarr").
    """
    if type(filename) is not str:
        raise Exception(
            (
                "Please pass a string"
                " (any characters surrounded by quotation marks)"
                " for your file name."
            )
        )
    filename = filename.split(".")[0]

    dir_path = filename + ".zarr"

    try:
        shutil.rmtree(dir_path)
        print(f"Zarr dataset '{dir_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"Zarr dataset '{dir_path}' not found.")
    except OSError as e:
        print(f"Error deleting Zarr dataset '{dir_path}': {e}")


def _add_metadata(data: xr.Dataset):
    """
    Add attributes to xarray dataset in-place.

    Parameters
    ----------
    data: xarray.Dataset
    """
    ds_attrs = data.attrs

    ct = datetime.datetime.now()
    ct_str = ct.strftime("%d-%b-%Y (%H:%M)")

    ck_attrs = {
        "Data_exported_from": "Cal-Adapt Analytics Engine",
        "Data_export_timestamp": ct_str,
        "Analysis_package_name": "climakitae",
        "Version": _version("climakitae"),
        "Author": "Cal-Adapt Analytics Engine Team",
        "Author_email": "analytics@cal-adapt.org",
        "Home_page": "https://github.com/cal-adapt/climakitae",
        "License": "BSD 3-Clause License",
    }

    ds_attrs.update(ck_attrs)
    data.attrs = ds_attrs


def _estimate_file_size(data: xr.DataArray | xr.Dataset, format: str) -> float:
    """
    Estimate uncompressed file size in gigabytes when exporting `data` in `format`.

    Parameters
    ----------
    data: xarray.DataArray or xarray.Dataset
        data to export to the specified `format`
    format: str
        file format ("Zarr", "NetCDF", "CSV")

    Returns
    -------
    float
        estimated file size in gigabytes
    """
    match format:
        case "NetCDF" | "Zarr":
            data_size = data.nbytes
            buffer_size = 100 * 1024 * 1024  # 100 MB for miscellaneous metadata
            est_file_size = data_size + buffer_size
        case "CSV":
            # Rough estimate of the number of chars per CSV line
            # Will overestimate uncompressed size by 10-20%
            chars_per_line = 150

            match data:
                case xr.DataArray():
                    est_file_size = data.size * chars_per_line
                case xr.Dataset():
                    est_file_size = prod(data.sizes.values()) * chars_per_line
        case _:
            raise Exception('format needs to be "NetCDF", "Zarr", "CSV"')

    return est_file_size / bytes_per_gigabyte


def _warn_large_export(file_size: float, file_size_threshold: float | int = 5):
    """Print warning message if predicted file size exceeds threshold.

    Parameters
    ----------
    file_size: float
        Predicted file size in GB.
    file_size_threshold: float or int
        Threshold size in GB for warning.
    """
    if file_size > file_size_threshold:
        print(
            "WARNING: Estimated file size is "
            + str(round(file_size, 2))
            + " GB. This might take a while!"
        )


def _update_encoding(data: xr.Dataset):
    """
    Update data encodings to prevent issues when exporting them to NetCDF.

    Drop `missing_value` encoding, if any, on `data` as well as its coordinates
    and data variables.

    Parameters
    ----------
    data: xarray.Dataset

    Returns
    -------
    None

    Notes
    -----
    These encoding updates resolve errors raised when writing NetCDF files to
    S3.
    """

    def _unencode_missing_value(d: xr.Dataset):
        """Drop `missing_value` encoding, if any, on data object `d`.

        Parameters
        ----------
        d: xarray.Dataset

        Returns
        -------
        None
        """
        try:
            del d.encoding["missing_value"]
        except:
            pass

    _unencode_missing_value(data)
    for coord in data.coords:
        _unencode_missing_value(data[coord])

    for data_var in data.data_vars:
        _unencode_missing_value(data[data_var])


def _fillvalue_encoding(data: xr.Dataset) -> dict[str, int | float | None]:
    """
    Creates FillValue encoding for each variable for export to NetCDF.

    Parameters
    ----------
    data: xarray.Dataset

    Returns
    -------
    encoding: dict
    """
    fill = dict(_FillValue=None)
    filldict = {coord: fill for coord in data.coords}
    return filldict


def _compression_encoding(data: xr.Dataset) -> dict[str, int | float | None]:
    """
    Creates compression encoding for each variable for export to NetCDF.

    Parameters
    ----------
    data: xarray.Dataset

    Returns
    -------
    encoding: dict
    """
    comp = dict(zlib=True, complevel=6)
    compdict = {var: comp for var in data.data_vars}
    return compdict


def _convert_da_to_ds(data: xr.DataArray | xr.Dataset) -> xr.Dataset:
    """Convert xarray data array to dataset.

    Parameters
    ----------
    data: xarray.DataArray or xarray.Dataset
    """
    match data:
        case xr.DataArray():
            if not data.name:
                # name it in order to call to_dataset on it
                data.name = "data"
            return data.to_dataset()
        case xr.Dataset():
            return data
        case _:
            raise Exception("Input must be either an Xarray DataArray or Dataset")


def _export_to_netcdf(data: xr.DataArray | xr.Dataset, save_name: str):
    """
    Export user-selected data to NetCDF format.

    Export the xarray DataArray or Dataset `data` to a NetCDF file `save_name`.
    If there is enough disk space, the function saves the file locally to the
    jupyter hub.


    Parameters
    ----------
    data: xarray.DataArray or xarray.Dataset
        data to export to NetCDF format
    save_name: string
        desired output file name, including the file extension

    Returns
    -------
    None
    """
    print("Exporting specified data to NetCDF...")

    # Convert xr.DataArray to xr.Dataset so that compression can be utilized
    _data = _convert_da_to_ds(data)

    est_file_size = _estimate_file_size(_data, "NetCDF")
    disk_space = shutil.disk_usage(os.path.expanduser("~"))[2] / bytes_per_gigabyte

    _warn_large_export(est_file_size)

    _add_metadata(_data)

    _update_encoding(_data)

    if disk_space <= est_file_size:
        raise Exception(
            "Data too large to save locally. Use the format='Zarr', mode='s3' options."
        )

    print("Saving file locally as NetCDF4...")
    path = os.path.join(os.getcwd(), save_name)

    if os.path.exists(path):
        raise Exception(
            (
                f"File {save_name} exists. "
                "Please either delete that file from the work space "
                "or specify a new file name here."
            )
        )
    encoding = _fillvalue_encoding(_data) | _compression_encoding(_data)
    _data.to_netcdf(path, format="NETCDF4", engine="netcdf4", encoding=encoding)
    print(
        (
            "Saved! You can find your file in the panel to the left"
            " and download to your local machine from there."
        )
    )


def _export_to_zarr(data: xr.DataArray | xr.Dataset, save_name: str, mode: str):
    """
    Export user-selected data to Zarr format.
    Export the xarray DataArray or Dataset `data` to a Zarr dataset `save_name`.
    If `local` mode used it is saved to the HUB user partition. If `s3` mode used
    it is saved to the AWS S3 bucket `cadcat-tmp` and provides a URL for download.

    Parameters
    ----------
    data: xarray.DataArray or xarray.Dataset
        data to export to Zarr format
    save_name: string
        desired output Zarr directory name
    mode: string
        location logic for storing export file (`local`, `s3`)

    Returns
    -------
    None
    """
    print("Exporting specified data to Zarr...")

    # Convert xr.DataArray to xr.Dataset so that compression can be utilized
    _data = _convert_da_to_ds(data)

    est_file_size = _estimate_file_size(_data, "Zarr")
    disk_space = shutil.disk_usage(os.path.expanduser("~"))[2] / bytes_per_gigabyte

    _warn_large_export(est_file_size)

    _add_metadata(_data)

    _update_encoding(_data)

    def _write_zarr(path: str, data: xr.Dataset):
        encoding = _fillvalue_encoding(data)
        chunks = {k: v[0] for k, v in data.chunks.items()}
        data = data.chunk(chunks)
        data.to_zarr(path, encoding=encoding)

    def _write_zarr_to_s3(
        display_path: str, path: str, save_name: str, data: xr.Dataset
    ):
        _write_zarr(path, data)

        print(
            (
                "Saved! To open the file in your local machine, "
                "open the following S3 URI using xarray:"
                "\n\n"
                f"{display_path}"
                "\n\n"
                "Example of opening and saving to netCDF:\n"
                "ds = xr.open_zarr('"
                + display_path
                + "', storage_options={'anon': True})\n"
                "comp = dict(zlib=True, complevel=6)\n"
                "compdict = {var: comp for var in ds.data_vars}\n"
                "ds.to_netcdf('"
                + save_name.rstrip(".zarr")
                + ".nc', encoding=compdict)\n"
                "\n\n"
                ""
                "Note: The URL will remain valid for 1 week."
            )
        )

    match mode:
        case "local":
            print("Saving file locally as Zarr...")
            if disk_space <= est_file_size:
                raise Exception(
                    "Data too large to save locally. Use the format='Zarr', mode='s3' options."
                )
            path = os.path.join(os.getcwd(), save_name)

            if os.path.exists(path):
                raise Exception(
                    (
                        f"File {save_name} exists. "
                        "Please either delete that file from the work space "
                        "or specify a new file name."
                    )
                )
            _write_zarr(path, _data)
        case "s3":
            print("Saving file to S3 scratch bucket as Zarr...")
            display_path = f"{os.environ['SCRATCH_BUCKET']}/{save_name}"
            path = "simplecache::" + display_path
            prefix = display_path.split(export_s3_bucket + "/")[-1]

            s3 = boto3.resource("s3")
            try:
                s3.Object(export_s3_bucket, prefix + "/.zattrs").load()
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    # The object does not exist so go ahead and write to S3
                    _write_zarr_to_s3(display_path, path, save_name, _data)
                else:
                    # Something else has gone wrong.
                    raise
            else:
                # The object does exist
                raise Exception(f"File {save_name} exists. Specify a new file name.")
        case _:
            raise Exception("Correct mode not specified. Use either 'local' or 's3'.")


def _get_unit(dataarray: xr.DataArray) -> str:
    """
    Return unit of data variable in `dataarray`, if any, or an empty string.

    Parameters
    ----------
    dataarray: xarray.DataArray

    Returns
    -------
    str
    """
    data_attrs = dataarray.attrs
    if "units" in data_attrs and data_attrs["units"] is not None:
        return data_attrs["units"]
    else:
        return ""


def _ease_access_in_R(column_name: str) -> str:
    """
    Return a copy of the input that can be used in R easily.

    Modify the `column_name` string so that when it is the name of an R data
    table column, the column can be accessed by $. The modified string contains
    no spaces or special characters, and starts with a letter or a dot.

    Parameters
    ----------
    column_name: str

    Returns
    -------
    str

    Notes
    -----
    The input is assumed to be a column name of a pandas DataFrame converted
    from an xarray DataArray or Dataset available on the Cal-Adapt Analytics
    Engine. The conversions are through the to_dataframe method.

    The function acts on one of the display names of the variables:
    https://github.com/cal-adapt/climakitae/blob/main/climakitae/data/variable_descriptions.csv
    or one of the station names:
    https://github.com/cal-adapt/climakitae/blob/main/climakitae/data/hadisd_stations.csv
    """
    return (
        column_name.replace("(", "")
        .replace(")", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _update_header(
    df: pd.DataFrame, variable_unit_map: list[tuple[str, str]]
) -> pd.DataFrame:
    """
    Update data table header to match the given variable names and units.

    Update the header of the DataFrame `df` so that name and unit of the data
    variable contained in each column are as specified in `variable_unit_map`.
    The resulting header starts with a row labeled "variable" holding variable
    names. A 2nd "unit" row include the units associated with the variables.


    Parameters
    ----------
    df: pandas.DataFrame
        data table to update
    variable_unit_map: list of tuple
        list of tuples where each tuple contains the name and unit of the data
        variable in a column of the input data table

    Returns
    -------
    pandas.DataFrame
        data table with updated header
    """
    df.columns = pd.MultiIndex.from_tuples(
        variable_unit_map,
        name=["variable", "unit"],
    )
    df.reset_index(inplace=True)  # simplifies header
    return df


def _dataarray_to_dataframe(dataarray: xr.DataArray) -> pd.DataFrame:
    """
    Prepare xarray DataArray for export as CSV file.

    Convert the xarray DataArray `dataarray` to a pandas DataFrame ready to be
    exported to CSV format. The DataArray is converted through its to_dataframe
    method. The DataFrame header is renamed as needed to ease the access of
    columns in R. It is also enriched with the unit associated with the data
    variable in the DataArray.

    Parameters
    ----------
    dataarray: xarray.DataArray
        data to be prepared for export

    Returns
    -------
    pandas.DataFrame
        data ready for export
    """
    if not dataarray.name:
        # name it in order to call to_dataframe on it
        dataarray.name = "data"

    df = dataarray.to_dataframe()

    variable = dataarray.name
    unit = _get_unit(dataarray)
    variable_unit_map = []
    for col in df.columns:
        if col == variable:
            variable_unit_map.append((_ease_access_in_R(col), unit))
        else:
            variable_unit_map.append((_ease_access_in_R(col), ""))

    df = _update_header(df, variable_unit_map)
    return df


def _dataset_to_dataframe(dataset: xr.Dataset) -> pd.DataFrame:
    """
    Prepare xarray Dataset for export as CSV file.

    Convert the xarray Dataset `dataset` to a pandas DataFrame ready to be
    exported to CSV format. The Dataset is converted through its to_dataframe
    method. The DataFrame header is renamed as needed to ease the access of
    columns in R. It is also enriched with the units associated with the data
    variables and other non-index variables in the Dataset. If the Dataset
    contains station data, the name of any climate variable associated with
    the station(s) is added to the header as well.

    Parameters
    ----------
    dataset: xarray.Dataset
        data to be prepared for export

    Returns
    -------
    pandas.DataFrame
        data ready for export
    """
    df = dataset.to_dataframe()

    variable_unit_map = [
        (var_name, _get_unit(dataset[var_name])) for var_name in df.columns
    ]
    df = _update_header(df, variable_unit_map)

    # Helpers for adding to header climate variable names associated w/ stations
    station_df = read_csv_file(stations_csv_path)
    station_lst = list(station_df.station)

    def _is_station(name):
        """Return True if `name` is an HadISD station name."""
        return name in station_lst

    variable_description_df = read_csv_file(variable_descriptions_csv_path)
    variable_ids = variable_description_df.variable_id.values

    def _variable_id_to_name(var_id: str) -> str:
        """Convert variable ID to variable name.

        Return the "display_name" associated with the "variable_id" in
        variable_descriptions.csv. If `var_id` is not a "variable_id" in the
        CSV file, return an empty string.

        Parameters
        ----------
        var_id: str

        Returns
        -------
        str
        """
        if var_id in variable_ids:
            var_name_series = variable_description_df.loc[
                variable_ids == var_id, "display_name"
            ]
            var_name = var_name_series.to_list()[0]
            return var_name
        else:
            return ""

    def _get_station_variable_name(dataset: xr.Dataset, station: str) -> str:
        """Get name of climate variable stored in `dataset` variable `station`.

        Return an empty string if that is not possible.

        Parameters
        ----------
        dataset: xr.Dataset
        station: str

        Returns
        -------
        var_name: str
        """
        try:
            station_da = dataset[station]  # DataArray
            data_attrs = station_da.attrs
            if "variable_id" in data_attrs and data_attrs["variable_id"] is not None:
                var_id = data_attrs["variable_id"]
                var_name = _variable_id_to_name(var_id)
                return var_name
            else:
                return ""
        except:
            return ""

    # Add to header: name of any climate variable associated with stations
    column_names = df.columns.get_level_values(0)
    climate_var_lst = []
    for name in column_names:
        if _is_station(name):
            climate_var = _get_station_variable_name(dataset, station=name)
        else:
            climate_var = ""
        climate_var_lst.append(climate_var)

    if set(climate_var_lst) != {""}:
        # Insert climate variable names to the 2nd row
        header_df = df.columns.to_frame()
        header_df.insert(1, "", climate_var_lst)
        # The 1st row was named "variable" by `_update_header`
        header_df.variable = header_df.variable.map(_ease_access_in_R)
        df.columns = pd.MultiIndex.from_frame(header_df)

    return df


def _export_to_csv(data: xr.DataArray | xr.Dataset, save_name: str):
    """
    Export user-selected data to CSV format.

    Export the xarray DataArray or Dataset `data` to a CSV file at
    `output_path`.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        data to export to CSV format
    save_name : string
        desired export file prefix

    Returns
    -------
    None
    """
    # Check file size and avail workspace disk space
    # raise error for not enough space
    # and warning for large file
    est_file_size = _estimate_file_size(data, "CSV")
    disk_space = shutil.disk_usage(os.path.expanduser("~"))[2] / bytes_per_gigabyte

    if disk_space <= est_file_size:
        raise Exception(
            "Not enough disk space to export data! You need at least "
            + str(round(est_file_size, 2))
            + (
                " GB free in the hub directory, which has 10 GB total space."
                " Try smaller subsets of space, time, scenario, and/or"
                " simulation; pick a coarser spatial or temporal scale;"
                " or delete any exported datasets which you have already"
                " downloaded or do not want."
            )
        )

    # Check if export file already exists and exit if so
    output_path = os.path.join(os.getcwd(), save_name)
    if os.path.exists(output_path):
        raise Exception(
            (
                f"File {save_name} exists. "
                "Please either delete that file from the work space "
                "or specify a new file name here."
            )
        )

    print("Exporting specified data to CSV...")
    _warn_large_export(est_file_size, 1.0)

    match data:
        case xr.DataArray():
            df = _dataarray_to_dataframe(data)
        case xr.Dataset():
            df = _dataset_to_dataframe(data)
        case _:
            raise Exception("Input data needs to be an Xarray DataArray or Dataset")

    # Warn about exceedance of Excel row or column limit
    excel_row_limit = 1048576
    excel_column_limit = 16384
    csv_nrows, csv_ncolumns = df.shape
    if csv_nrows > excel_row_limit or csv_ncolumns > excel_column_limit:
        warnings.warn(
            f"Dataset exceeds Excel limits of {excel_row_limit} rows "
            f"and {excel_column_limit} columns."
        )

    def _metadata_to_file(ds: xr.Dataset, output_name: str):
        """
        Write NetCDF metadata to a txt file so users can still access it
        after exporting to a CSV.

        Parameters
        ----------
        ds: xr.Dataset
        output_name: str

        Returns
        -------
        None
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
            if type(ds) == xr.core.dataarray.DataArray:
                f.write("\n")
                f.write("Name: " + ds.name)
            f.write("\n")
            for att_keys, att_values in ds.attrs.items():
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
                for att_keys, att_values in ds[coord].attrs.items():
                    f.write(str(att_keys) + " : " + str(att_values))
                    f.write("\n")

            if type(ds) == xr.core.dataset.Dataset:
                f.write("\n")
                f.write("\n")
                f.write("===== Variable descriptions =====")
                f.write("\n")

                for var in ds.data_vars:
                    f.write("\n")
                    f.write("== " + str(var) + " ==")
                    f.write("\n")
                    for att_keys, att_values in ds[var].attrs.items():
                        f.write(str(att_keys) + " : " + str(att_values))
                        f.write("\n")

    _metadata_to_file(data, output_path)
    df.to_csv(output_path, compression="gzip")
    print(
        (
            "Saved! You can find your file(s) in the panel to the left"
            " and download to your local machine from there."
        )
    )


def export(
    data: xr.DataArray | xr.Dataset,
    filename: str = "dataexport",
    format: str = "NetCDF",
    mode: str = "local",
):
    """Save xarray data as NetCDF, Zarr, or CSV in the current working directory, or if Zarr optionally
    stream the export file to an AWS S3 scratch bucket and give download URL. NetCDF can only be written
    to the HUB user partition if it will fit. Zarr can either be written to the HUB user partition or to
    S3 scratch bucket using the mode option.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data to export, as output by e.g. `DataParameters.retrieve()`.
    filename : str, optional
        Output file name (without file extension, i.e. "my_filename" instead
        of "my_filename.nc"). The default is "dataexport".
    format : str, optional
        File format ("Zarr", "NetCDF", "CSV"). The default is "NetCDF".
    mode : str, optional
        Save location logic for Zarr file ("local", "s3"). The default is "local"

    Returns
    -------
     None
    """
    ftype = type(data)

    if ftype not in [xr.core.dataset.Dataset, xr.core.dataarray.DataArray]:
        raise Exception(
            "Cannot export object of type "
            + str(ftype).strip("<class >")
            + ". Please pass an Xarray Dataset or DataArray."
        )

    if type(filename) is not str:
        raise Exception(
            (
                "Please pass a string"
                " (any characters surrounded by quotation marks)"
                " for your file name."
            )
        )
    filename = filename.split(".")[0]

    req_format = format.lower()

    if req_format not in ["zarr", "netcdf", "csv"]:
        raise Exception('Please select "Zarr", "NetCDF" or "CSV" as the file format.')

    extension_dict = {"zarr": ".zarr", "netcdf": ".nc", "csv": ".csv.gz"}

    save_name = filename + extension_dict[req_format]

    if (mode == "s3") and (req_format != "zarr"):
        raise Exception('To export to AWS S3 you must use the format="Zarr" option.')

    # now here is where exporting actually begins
    # we will have different functions for each file type
    # to keep things clean-ish
    match req_format:
        case "zarr":
            _export_to_zarr(data, save_name, mode)
        case "netcdf":
            _export_to_netcdf(data, save_name)
        case "csv":
            _export_to_csv(data, save_name)
        case _:
            raise Exception(
                'Please select "Zarr", "NetCDF" or "CSV" as the file format.'
            )


## TMY export functions
def _grab_dem_elev_m(lat: float, lon: float) -> float:
    """
    Pulls elevation value from the USGS Elevation Point Query Service,
    lat lon must be in decimal degrees (which it is after cleaning)
    Modified from:
    https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python

    Note: This is breaking at present (2/29/2024) -- setting to pulling station elevation from csv, 0 for custom
    """
    url = r"https://epqs.nationalmap.gov/v1/json?"

    # define rest query params
    params = {"output": "json", "x": lon, "y": lat, "units": "Meters"}

    # format query string and return value
    result = requests.get((url + urllib.parse.urlencode(params)))

    # error checking on api call
    if "value" not in result.json():
        print("Please re-run the current cell to re-try the API call")
    else:
        dem_elev_long = float(result.json()["value"])
        # make sure to round off lat-lon values so they are not improbably precise for our needs
        dem_elev_short = np.round(dem_elev_long, decimals=2)

    return dem_elev_short.astype("float")


def _epw_format_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs TMY output file in specific order and missing data codes
    Source: EnergyPlus Version 23.1.0 Documentation

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    df: pd.DataFrame
    """

    # set time col to datetime object for easy split
    df["time"] = pd.to_datetime(df["time"])
    df = df.assign(
        year=df["time"].dt.year,
        month=df["time"].dt.month,
        day=df["time"].dt.day,
        hour=df["time"].dt.hour + 1,  # 1-24, not 0-23
        minute=df["time"].dt.minute,
    )

    # set epw variable order, very specific -- manually set
    # Note: vars not provided by AE are noted as missing
    colnames = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "data_source",  # missing
        "Air Temperature at 2m",
        "Dew point temperature",
        "Relative humidity",
        "Surface Pressure",
        "exthorrad",  # missing - extraterrestrial horizontal radiation
        "extdirrad",  # missing - extraterrestrial direct normal radiation
        "extirsky",  # missing - horizontal IR radiation intensity from sky
        "Instantaneous downwelling shortwave flux at bottom",
        "Shortwave surface downward direct normal irradiance",
        "Shortwave surface downward diffuse irradiance",
        "glohorillum",  # missing - global horizontal illuminance (lx)
        "dirnorillum",  # missing - direct normal illuminance (lx)
        "difhorillum",  # missing - diffuse horizontal illuminance (lx)
        "zenlum",  # missing - zenith luminnace (lx)
        "Wind direction at 10m",
        "Wind speed at 10m",
        "totskycvr",  # missing - total sky cover (tenths)
        "opaqskycvr",  # missing - opaque sky cover (tenths)
        "visibility",  # missing - visibility (km)
        "ceiling_hgt",  # missing - ceiling height (m)
        "presweathobs",  # missing - present weather observation
        "presweathcodes",  # missing - present weather codes
        "precip_wtr",  # missing - precipitatble water (mm)
        "aerosol_opt_depth",  # missing - aerosol optical depth (thousandths)
        "snowdepth",  # missing - snow depth (cm)
        "days_last_snow",  # missing - days since last snow
        "albedo",  # missing - albedo
        "liq_precip_depth",  # missing - liquid precip depth (mm)
        "liq_precip_rate",  # missing - liquid precip rate (h)
    ]

    # set specific missing data flags per variable
    for var in [
        "exthorrad",
        "extdirrad",
        "extirsky",
        "dirnorrad",
        "zenlum",
        "visibility",
    ]:
        df[var] = 9999
    for var in ["glohorillum", "dirnorillum", "difhorillum"]:
        df[var] = 999900
    for var in ["days_last_snow", "liq_precip_rate"]:
        df[var] = 99
    for var in ["precip_wtr", "snowdepth", "albedo", "liq_precip_depth"]:
        df[var] = 999
    df["ceiling_hgt"] = 99999
    df["aerosol_opt_depth"] = 0.999
    df["presweathobs"] = 9
    df["presweathcodes"] = 999999999

    # Note: Setting cloud cover to 5, per stakeholder recommendation, indicates 5/10ths skycover = 50% cloudy
    # Note: At present AE has no plans to provide cloud coverage data
    for var in ["totskycvr", "opaqskycvr"]:
        df[var] = 5

    # lastly set data source / uncertainty flag (section 2.13 of doc)
    # on AE: ? = var does not fit source options
    # on AE: 9 = uncertainty unknown
    df["data_source"] = "?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9"

    # resets col order and drops any unnamed column from original df
    df = df.reindex(columns=colnames)

    return df


def _leap_day_fix(df: pd.DataFrame) -> pd.DataFrame:
    """Addresses leap day inclusion in TMY dataframe bug by removing the extra nan rows and resetting time index to Feb 28

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    df_leap: pd.DataFrame
    """
    df_leap = df.copy(deep=True)
    df_leap["time"] = pd.to_datetime(df["time"])  # set time to datetime
    df_leap = df_leap.dropna()  # drops extra rows

    # 3 models have leap days, 1 model does not -- handling for both
    # handling for TaiESM1 (no leap day natively)
    match df_leap.simulation.unique()[0]:
        case "WRF_TaiESM1_r1i1p1f1":
            df_leap["time"] = np.where(
                (df_leap.time.dt.month == 2) & (df_leap.time.dt.day == 29),
                df_leap.time - pd.DateOffset(days=1),
                df_leap.time,
            )  # reset remaining feb 29 hours to feb 28

        # handling for 3 models with native leap days
        case _:
            df_leap["time"] = pd.to_datetime(df["time"])  # set time to datetime
            df_leap = df_leap.loc[
                ~((df_leap.time.dt.month == 2) & (df_leap.time.dt.day == 29))
            ]

    return df_leap


def _find_missing_val_month(df: pd.DataFrame) -> int:
    """Finds month that does not match expected hours

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    int
    """
    hrs_per_month = {
        1: 744,
        2: 672,
        3: 744,
        4: 720,
        5: 744,
        6: 720,
        7: 744,
        8: 744,
        9: 720,
        10: 744,
        11: 720,
        12: 744,
    }
    for m in range(1, 13, 1):
        df_month = df.loc[df.time.dt.month == m]
        if len(df_month) != hrs_per_month[m]:
            return m


def _missing_hour_fix(df: pd.DataFrame) -> pd.DataFrame:
    """Addresses missing hour in TMY dataframe bug by adding the missing hour at the appropriate spot and duplicating the previous hour's values

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    df_fixed: pd.DataFrame

    Notes
    -----
    Only fixes missing hour if missing hour is not the first or last hour of the month.
    """
    df_missing = df.copy(deep=True)
    df_missing["time"] = pd.to_datetime(df["time"])  # set time to datetime

    # first identify where missing hour is
    miss_month = _find_missing_val_month(
        df_missing
    )  # typically march or april when DST "goes forward an hour"

    # brute force way - as it is not a continuous time index (months are spliced together)
    df_prior = df_missing.loc[
        df_missing.time.dt.month < miss_month
    ]  # data prior to miss_month
    df_post = df_missing.loc[
        df_missing.time.dt.month > miss_month
    ]  # data after miss_month

    # fix missing hour
    df_bad = df_missing.loc[
        df_missing.time.dt.month == miss_month
    ]  # pulls out just the month with the missing hour
    df_bad.index = pd.to_datetime(df_bad.time)

    # set up correct df of month with all hours
    df_full = pd.DataFrame(
        pd.date_range(start=df_bad.index.min(), end=df_bad.index.max(), freq="h")
    )
    missing_cols = [col for col in df_bad.columns]
    df_full[missing_cols] = np.nan
    df_full["time"] = df_full[0]
    df_full.index = pd.to_datetime(df_full.time)

    df_month_fixed = pd.concat([df_bad, df_full])
    df_month_fixed = df_month_fixed.drop_duplicates(subset=["time"], keep="first")
    df_month_fixed = df_month_fixed.drop(columns=["time", 0])
    df_month_fixed = df_month_fixed.sort_values(by="time", ascending=True)
    df_month_fixed = df_month_fixed.reset_index()
    df_month_fixed = df_month_fixed.ffill()  # fill from previous days values

    # concat dfs together
    df_fixed = pd.concat([df_prior, df_month_fixed, df_post])
    return df_fixed


def _tmy_8760_size_check(df: pd.DataFrame) -> pd.DataFrame:
    """Checks the size of the TMY dataframe for export to ensure that it is explicitly 8760 in size.
    There are several scenarios where the input TMY dataframe would not be 8760 in size:
    (1) Size 8761, additional single hour due to time change for local time. Fix removes the duplicate row (typically in Nov.)
    (2) Size 8759, missing a single hour due to time change for local time. Fix adds the missing row (typically in Mar/Apr) by filling in from the previous hour.
    (3) Size 8784, 24 extra hours due to inclusion of a leap year February and specific models that retain leap days. Fix removes the additional rows.
    (4) Size 8783, 24 extra hours due to inclusion of a leap year February and a missing hour due to time change. Fix adds missing row and removes additional leap day rows.
    (5) Size 8758, missing two single hours due to time change for local time. Fix adds the missing row by filling in from the previous hour. Run twice.
        e.g. March 2008 and April 2000 are both time change months. Source: https://en.wikipedia.org/wiki/History_of_time_in_the_United_States

    Note: This is a bug introduced by the time zone correction to local time and should be addressed in the future.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of TMY to export

    Returns
    -------
    df: pd.Dataframe
        Dataframe of TMY to export, explicitly 8760 in size
    """

    # first drop any duplicate time rows -- some df with 8760 are 8759 with duplicate rows, i.e., not a true 8760
    # this should handle cases of 8761 by reducing to 8760 or 8759
    df_to_check = df.copy(deep=True)
    df_to_check = df_to_check.drop_duplicates(subset=["time"], keep="first")

    # fix cases
    match len(df_to_check):
        case 8760:
            return df_to_check
        case 8759:  # Missing hour, add missing row
            df_to_check = _missing_hour_fix(df_to_check)
            return df_to_check
        case 8784:  # Leap day added, remove Feb 29
            df_to_check = _leap_day_fix(df_to_check)
            return df_to_check
        case 8783:  # Leap day added and missing hour
            # remove leap day
            df_to_check = _leap_day_fix(df_to_check)
            # add missing hour
            df_to_check = _missing_hour_fix(df_to_check)
            return df_to_check
        case 8758:  # double missing hour
            df_to_check = _missing_hour_fix(df_to_check)  # march fix
            df_to_check = _missing_hour_fix(df_to_check)  # april fix
            return df_to_check
        case 8782:  # Leap day and double missing hour
            # remove leap day
            df_to_check = _leap_day_fix(df_to_check)
            # add missing hours
            df_to_check = _missing_hour_fix(df_to_check)  # march fix
            df_to_check = _missing_hour_fix(df_to_check)  # april fix
            return df_to_check
        case _:  # none of the above
            print(
                "Error: The size of the input dataframe ({}) does not comform to standard 8760 size. Please confirm.".format(
                    len(df)
                )
            )
            return None


def write_tmy_file(
    filename_to_export: str,
    df: pd.DataFrame,
    location_name: str,
    station_code: int,
    stn_lat: float,
    stn_lon: float,
    stn_state: str,
    stn_elev: float = 0.0,
    file_ext: str = "tmy",
):
    """Exports TMY data either as .epw or .tmy file

    Parameters
    ----------
    filename_to_export: str
        Filename string, constructed with station name and simulation
    df: pd.DataFrame
        Dataframe of TMY data to export
    location_name: str
        Location name string, often station name
    station_code: int
        Station code
    stn_lat: float
        Station latitude
    stn_lon: float
        Station longitude
    stn_state: str
        State of station location
    stn_elev: float, optional
        Elevation of station, default is 0.0
    file_ext: str, optional
        File extension for export, default is .tmy, options are "tmy" and "epw"

    Returns
    -------
    None
    """

    station_df = read_csv_file(stations_csv_path)

    # check that data passed is a DataFrame object
    if type(df) != pd.DataFrame:
        raise ValueError(
            "The function requires a pandas DataFrame object as the data input"
        )

    # size check on TMY dataframe
    df = _tmy_8760_size_check(df)

    def _utc_offset_timezone(lat, lon):
        """
        Based on user input of lat lon, returns the UTC offset for that timezone

        Parameters
        ----------
        lat: float
        lon: float

        Returns
        -------
        str

        Modified from:
        https://stackoverflow.com/questions/5537876/get-utc-offset-from-time-zone-name-in-python
        """
        tf = TimezoneFinder()
        tzn = tf.timezone_at(lng=lon, lat=lat)

        time_now = datetime.datetime.now(pytz.timezone(tzn))
        tz_offset = time_now.utcoffset().total_seconds() / 60 / 60

        diff = "{:d}".format(int(tz_offset))

        return diff

    # custom location input handling
    match station_code:
        case str():  # custom code passed
            station_code = station_code
            state = stn_state
            timezone = _utc_offset_timezone(lon=stn_lon, lat=stn_lat)
            elevation = (
                stn_elev  # default of 0.0 on custom inputs if elevation is not provided
            )

        case int():  # hadisd station code passed
            # look up info
            if station_code in station_df["station id"].values:
                state = station_df.loc[station_df["station id"] == station_code][
                    "state"
                ].values[0]
                elevation = station_df.loc[station_df["station id"] == station_code][
                    "elevation"
                ].values[0]
                station_code = str(station_code)[:6]
                timezone = _utc_offset_timezone(lon=stn_lon, lat=stn_lat)
        case _:
            raise ValueError("station_code needs to be either str or int")

    def _tmy_header(
        location_name: str,
        station_code: int,
        stn_lat: float,
        stn_lon: float,
        state: str,
        timezone: str,
        elevation: float,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Constructs the header for the TMY output file in .tmy format

        Parameters
        ----------
        location_name: str
        station_code: int
        stn_lat: float
        stn_lon: float
        state: str
        timezone: str
        elevation: float
        df: pd.DataFrame

        Returns
        -------
        headers: list of strs

        Source: https://www.nrel.gov/docs/fy08osti/43156.pdf (pg. 3)
        """

        # line 1 - site information
        # line 1: USAF, station name quote delimited, state, time zone, lat, lon, elev (m)
        line_1 = "{0},'{1}',{2},{3},{4},{5},{6},{7}\n".format(
            station_code,
            location_name,
            state,
            timezone,
            stn_lat,
            stn_lon,
            elevation,
            df["simulation"].values[0],
        )

        # line 2 - data field name and units, manually setting to ensure matches TMY3 labeling
        line_2 = "Air Temperature at 2m (degC),Dew point temperature (degC),Relative humidity (%),Instantaneous downwelling shortwave flux at bottom (W m-2),Shortwave surface downward direct normal irradiance (W m-2),Shortwave surface downward diffuse irradiance (W m-2),Instantaneous downwelling longwave flux at bottom (W m-2),Wind speed at 10m (m s-1),Wind direction at 10m (deg),Surface Pressure (Pa)\n"

        headers = [line_1, line_2]

        return headers

    def _epw_header(
        location_name: str,
        station_code: int,
        stn_lat: float,
        stn_lon: float,
        state: str,
        timezone: str,
        elevation: float,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Constructs the header for the TMY output file in .epw format

        Parameters
        ----------
        location_name: str
        station_code: int
        stn_lat: float
        stn_lon: float
        state: str
        timezone: str
        elevation: float
        df: pd.DataFrame

        Returns
        -------
        headers: list of strs

        Source: EnergyPlus Version 23.1.0 Documentation
        """

        # line 1 - location, location name, state, country, WMO, lat, lon
        # line 1 - location, location name, state, country, weather station number (2 cols), lat, lon, time zone, elevation
        line_1 = "LOCATION,{0},{1},USA,{2},{3},{4},{5},{6},{7}\n".format(
            location_name.upper(),
            state,
            "Custom_{}".format(station_code),
            station_code,
            stn_lat,
            stn_lon,
            timezone,
            elevation,
        )

        # line 2 - design conditions, leave blank for now
        line_2 = "DESIGN CONDITIONS\n"

        # line 3 - typical/extreme periods, leave blank for now
        line_3 = "TYPICAL/EXTREME PERIODS\n"

        # line 4 - ground temperatures, leave blank for now
        line_4 = "GROUND TEMPERATURES\n"

        # line 5 - holidays/daylight savings, leap year (yes/no), daylight savings start, daylight savings end, num of holidays
        line_5 = "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n"

        # line 6 - comments 1, going to include simulation + scenario information here
        line_6 = "COMMENTS 1,TMY data produced on the Cal-Adapt: Analytics Engine, Scenario: {0}, Simulation: {1}\n".format(
            df["scenario"].values[0], df["simulation"].values[0]
        )

        # line 7 - comments 2, including date range here from which TMY calculated
        line_7 = "COMMENTS 2, TMY data produced using 1990-2020 climatological period\n"

        # line 8 - data periods, num data periods, num records per hour, data period name, data period start day of week, data period start (Jan 1), data period end (Dec 31)
        line_8 = "DATA PERIODS,1,1,Data,,1/ 1,12/31\n"

        headers = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8]

        return headers

    # typical meteorological year format
    match file_ext:
        case "tmy":
            path_to_file = filename_to_export + ".tmy"

            with open(path_to_file, "w") as f:
                f.writelines(
                    _tmy_header(
                        location_name,
                        station_code,
                        stn_lat,
                        stn_lon,
                        state,
                        timezone,
                        elevation,
                        df,
                    )
                )  # writes required header lines
                df = df.drop(
                    columns=["simulation", "lat", "lon", "scenario"]
                )  # drops header columns from df
                dfAsString = df.to_csv(sep=",", header=False, index=False)
                f.write(dfAsString)  # writes data in TMY format
            print(
                "TMY data exported to .tmy format with filename {}.tmy, with size {}".format(
                    filename_to_export, len(df)
                )
            )
        # energy plus weather format
        case "epw":
            path_to_file = filename_to_export + ".epw"
            with open(path_to_file, "w") as f:
                f.writelines(
                    _epw_header(
                        location_name,
                        station_code,
                        stn_lat,
                        stn_lon,
                        state,
                        timezone,
                        elevation,
                        df,
                    )
                )  # writes required header lines
                df_string = _epw_format_data(df).to_csv(
                    sep=",", header=False, index=False
                )
                f.write(df_string)  # writes data in EPW format
            print(
                "TMY data exported to .epw format with filename {}, with size {}.epw".format(
                    filename_to_export, len(df)
                )
            )
        case _:
            print('Please pass either "tmy" or "epw" as a file format for export.')
