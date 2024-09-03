import os
import boto3
import fsspec
import shutil
import logging
import warnings
import datetime
import xarray as xr
import pandas as pd
import numpy as np
import requests
import urllib
import pytz
from timezonefinder import TimezoneFinder
from importlib.metadata import version as _version
from botocore.exceptions import ClientError
from math import prod
from climakitae.util.utils import read_csv_file
from climakitae.core.paths import (
    variable_descriptions_csv_path,
    stations_csv_path,
    export_s3_bucket,
)

xr.set_options(keep_attrs=True)
bytes_per_gigabyte = 1024 * 1024 * 1024


def _estimate_file_size(data, format):
    """
    Estimate uncompressed file size in gigabytes when exporting `data` in `format`.

    Parameters
    ----------
    data: xarray.DataArray or xarray.Dataset
        data to export to the specified `format`
    format: str
        file format ("NetCDF" or "CSV")

    Returns
    -------
    float
        estimated file size in gigabytes
    """
    if format == "NetCDF":
        data_size = data.nbytes
        buffer_size = 100 * 1024 * 1024  # 100 MB for miscellaneous metadata
        est_file_size = data_size + buffer_size
    elif format == "CSV":
        # Rough estimate of the number of chars per CSV line
        # Will overestimate uncompressed size by 10-20%
        chars_per_line = 150

        if isinstance(data, xr.core.dataarray.DataArray):
            est_file_size = data.size * chars_per_line
        elif isinstance(data, xr.core.dataset.Dataset):
            est_file_size = prod(data.dims.values()) * chars_per_line
    return est_file_size / bytes_per_gigabyte


def _warn_large_export(file_size, file_size_threshold=5):
    if file_size > file_size_threshold:
        print(
            "WARNING: Estimated file size is "
            + str(round(file_size, 2))
            + " GB. This might take a while!"
        )


def _export_to_netcdf(data, save_name, mode):
    """
    Export user-selected data to NetCDF format.

    Export the xarray DataArray or Dataset `data` to a NetCDF file `save_name`.
    If there is enough disk space, the function saves the file locally to the
    jupyter hub; otherwise, it saves the file to the S3 bucket `cadcat-tmp`
    and provides a URL for download. The optional `mode` parameters allows user
    to override automatic behavior.


    Parameters
    ----------
    data: xarray.DataArray or xarray.Dataset
        data to export to NetCDF format
    save_name: string
        desired output file name, including the file extension
    mode: string
        location logic for storing export file.

    Returns
    -------
    None
    """
    print("Exporting specified data to NetCDF...")

    # Convert xr.DataArray to xr.Dataset so that compression can be utilized
    _data = data
    if isinstance(_data, xr.core.dataarray.DataArray):
        if not _data.name:
            # name it in order to call to_dataset on it
            _data.name = "data"
        _data = _data.to_dataset()

    est_file_size = _estimate_file_size(_data, "NetCDF")
    disk_space = shutil.disk_usage(os.path.expanduser("~"))[2] / bytes_per_gigabyte

    _warn_large_export(est_file_size)

    def _update_attributes(data):
        """
        Update data attributes to prevent issues when exporting them to NetCDF.

        Convert list and None attributes to strings. If `time` is a coordinate of
        `data`, remove any of its `units` attribute. Attributes include global data
        attributes as well as that of coordinates and data variables.

        Parameters
        ----------
        data: xarray.Dataset

        Returns
        -------
        None

        Notes
        -----
        These attribute updates resolve errors raised when using the scipy engine
        to write NetCDF files to S3.
        """

        def _list_n_none_to_string(dic):
            """Convert list and None to string.

            Parameters
            ----------
            dic: dict

            Returns
            -------
            dict
            """
            for k, v in dic.items():
                if isinstance(v, list):
                    dic[k] = str(v)
                if v is None:
                    dic[k] = ""
            return dic

        data.attrs = _list_n_none_to_string(data.attrs)
        for coord in data.coords:
            data[coord].attrs = _list_n_none_to_string(data[coord].attrs)
        if "time" in data.coords and "units" in data["time"].attrs:
            del data["time"].attrs["units"]

        for data_var in data.data_vars:
            data[data_var].attrs = _list_n_none_to_string(data[data_var].attrs)

    _update_attributes(_data)

    def _update_encoding(data):
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

        def _unencode_missing_value(d):
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

    _update_encoding(_data)

    def _fillvalue_encoding(data):
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

    def _compression_encoding(data):
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

    def _create_presigned_url(bucket_name, object_name, expiration=60 * 60 * 24 * 7):
        """
        Generate a presigned URL to share an S3 object.

        Parameters
        ----------
        bucket_name: str
        object_name: str
        expiration: int, optional
            Time in seconds for the presigned URL to remain valid. The default is
            one week.

        Returns
        -------
        str
            Presigned URL. If error, returns None.

        References
        ----------
        https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-presigned-urls.html#presigned-urls
        """
        s3_client = boto3.client("s3")
        try:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": object_name},
                ExpiresIn=expiration,
            )
        except ClientError as e:
            logging.error(e)
            return None

        return url

    file_location = "local"

    if mode == "local":
        if disk_space <= est_file_size:
            raise Exception("Data too large to save locally. Use the mode=s3 option.")
        file_location = "local"
    elif mode == "s3":
        file_location = "s3"
    elif mode == "auto":
        if disk_space > est_file_size:
            file_location = "local"
        else:
            file_location = "s3"
    else:
        raise Exception("Specified mode needs to one of (local, s3, auto)")

    if file_location == "local":
        print("Saving file locally with compression...")
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
        _data.to_netcdf(path, engine="h5netcdf", encoding=encoding)
        print(
            (
                "Saved! You can find your file in the panel to the left"
                " and download to your local machine from there."
            )
        )

    else:
        path = f"simplecache::{os.environ['SCRATCH_BUCKET']}/{save_name}"

        with fsspec.open(path, "wb") as fp:
            print("Saving file to S3 scratch bucket without compression...")
            encoding = _fillvalue_encoding(_data)
            _data.to_netcdf(fp, engine="h5netcdf", encoding=encoding)

            download_url = _create_presigned_url(
                bucket_name=export_s3_bucket,
                object_name=path.split(export_s3_bucket + "/")[-1],
            )
            print(
                (
                    "Saved! To download the file to your local machine, "
                    "open the following URL in a web browser:"
                    "\n\n"
                    f"{download_url}"
                    "\n\n"
                    "Note: The URL will remain valid for 1 week."
                )
            )


def _get_unit(dataarray):
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


def _ease_access_in_R(column_name):
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


def _update_header(df, variable_unit_map):
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


def _dataarray_to_dataframe(dataarray):
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


def _dataset_to_dataframe(dataset):
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

    def _variable_id_to_name(var_id):
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

    def _get_station_variable_name(dataset, station):
        """Get name of climate variable stored in `dataset` variable `station`.

        Return an empty string if that is not possible.

        Parameters
        ----------
        dataset: xr.Dataset
        station: str

        Returns
        -------
        str
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


def _export_to_csv(data, save_name):
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

    ftype = type(data)

    if ftype == xr.core.dataarray.DataArray:
        df = _dataarray_to_dataframe(data)

    elif ftype == xr.core.dataset.Dataset:
        df = _dataset_to_dataframe(data)

    # Warn about exceedance of Excel row or column limit
    excel_row_limit = 1048576
    excel_column_limit = 16384
    csv_nrows, csv_ncolumns = df.shape
    if csv_nrows > excel_row_limit or csv_ncolumns > excel_column_limit:
        warnings.warn(
            f"Dataset exceeds Excel limits of {excel_row_limit} rows "
            f"and {excel_column_limit} columns."
        )

    def _metadata_to_file(ds, output_name):
        """
        Write NetCDF metadata to a txt file so users can still access it
        after exporting to a CSV.

        Parameters
        ----------
        ds: xr.DataSet
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


def export(data, filename="dataexport", format="NetCDF", mode="auto"):
    """Save xarray data as either a NetCDF or CSV in the current working directory,
    or stream the export file to an AWS S3 scratch bucket and give download URL. Default
    behavior is for the code to automatically determine the output destination based on whether
    file is small enough to fit in HUB user partition, this can be overridden using the mode parameter.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data to export, as output by e.g. `climakitae.Select().retrieve()`.
    filename : str, optional
        Output file name (without file extension, i.e. "my_filename" instead
        of "my_filename.nc"). The default is "dataexport".
    format : str, optional
        File format ("NetCDF" or "CSV"). The default is "NetCDF".
    mode : str, optional
        Save location logic for NetCDF file ("auto", "local", "s3"). The default is "auto"
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

    if req_format not in ["netcdf", "csv"]:
        raise Exception('Please select "NetCDF" or "CSV" as the file format.')

    extension_dict = {"netcdf": ".nc", "csv": ".csv.gz"}

    save_name = filename + extension_dict[req_format]

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

    # now here is where exporting actually begins
    # we will have different functions for each file type
    # to keep things clean-ish
    if "netcdf" == req_format:
        _export_to_netcdf(data, save_name, mode)
    elif "csv" == req_format:
        _export_to_csv(data, save_name)


## TMY export functions
def _grab_dem_elev_m(lat, lon):
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


def _epw_format_data(df):
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


def _leap_day_fix(df):
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
    if df_leap.simulation.unique()[0] == "WRF_TaiESM1_r1i1p1f1":
        df_leap["time"] = np.where(
            (df_leap.time.dt.month == 2) & (df_leap.time.dt.day == 29),
            df_leap.time - pd.DateOffset(days=1),
            df_leap.time,
        )  # reset remaining feb 29 hours to feb 28

    # handling for 3 models with native leap days
    elif df_leap.simulation.unique()[0] != "WRF_TaiESM1_r1i1p1f1":
        df_leap["time"] = pd.to_datetime(df["time"])  # set time to datetime
        df_leap = df_leap.loc[
            ~((df_leap.time.dt.month == 2) & (df_leap.time.dt.day == 29))
        ]

    return df_leap


def _find_missing_val_month(df):
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


def _missing_hour_fix(df):
    """Addresses missing hour in TMY dataframe bug by adding the missing hour at the appropriate spot and duplicating the previous hour's values

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    df_fixed: pd.DataFrame
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
        pd.date_range(start=df_bad.index.min(), end=df_bad.index.max(), freq="H")
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
    df_month_fixed = df_month_fixed.fillna(
        method="ffill"
    )  # fill from previous days values

    # concat dfs together
    df_fixed = pd.concat([df_prior, df_month_fixed, df_post])
    return df_fixed


def _tmy_8760_size_check(df):
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
    if len(df_to_check) == 8760:
        return df_to_check
    elif len(df_to_check) != 8760:
        if len(df_to_check) == 8759:  # Missing hour, add missing row
            df_to_check = _missing_hour_fix(df_to_check)
            return df_to_check

        elif len(df_to_check) == 8784:  # Leap day added, remove Feb 29
            df_to_check = _leap_day_fix(df_to_check)
            return df_to_check

        elif len(df_to_check) == 8783:  # Leap day added and missing hour
            # remove leap day
            df_to_check = _leap_day_fix(df_to_check)
            # add missing hour
            df_to_check = _missing_hour_fix(df_to_check)
            return df_to_check

        elif len(df_to_check) == 8758:  # double missing hour
            df_to_check = _missing_hour_fix(df_to_check)  # march fix
            df_to_check = _missing_hour_fix(df_to_check)  # april fix
            return df_to_check

        elif len(df_to_check) == 8782:  # Leap day and double missing hour
            # remove leap day
            df_to_check = _leap_day_fix(df_to_check)
            # add missing hours
            df_to_check = _missing_hour_fix(df_to_check)  # march fix
            df_to_check = _missing_hour_fix(df_to_check)  # april fix
            return df_to_check

        else:
            print(
                "Error: The size of the input dataframe ({}) does not comform to standard 8760 size. Please confirm.".format(
                    len(df)
                )
            )
            return None


def write_tmy_file(
    filename_to_export,
    df,
    location_name,
    station_code,
    stn_lat,
    stn_lon,
    stn_state,
    stn_elev=0.0,
    file_ext="tmy",
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
    if type(station_code) == str:  # custom code passed
        station_code = station_code
        state = stn_state
        timezone = _utc_offset_timezone(lon=stn_lon, lat=stn_lat)
        elevation = (
            stn_elev  # default of 0.0 on custom inputs if elevation is not provided
        )

    elif type(station_code) == int:  # hadisd statio code passed
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

    def _tmy_header(
        location_name, station_code, stn_lat, stn_lon, state, timezone, elevation, df
    ):
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
        location_name, station_code, stn_lat, stn_lon, state, timezone, elevation, df
    ):
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
    if file_ext == "tmy":
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
    elif file_ext == "epw":
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
            df_string = _epw_format_data(df).to_csv(sep=",", header=False, index=False)
            f.write(df_string)  # writes data in EPW format
        print(
            "TMY data exported to .epw format with filename {}, with size {}.epw".format(
                filename_to_export, len(df)
            )
        )
    else:
        print('Please pass either "tmy" or "epw" as a file format for export.')
