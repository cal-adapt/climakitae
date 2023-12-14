"""Backend functions for exporting data."""

import os
import shutil
import warnings
import datetime
import xarray as xr
import pandas as pd
import numpy as np
import requests
import urllib
import pytz
from datetime import datetime, timezone
from timezonefinder import TimezoneFinder
from importlib.metadata import version as _version
from climakitae.util.utils import read_csv_file
from climakitae.core.paths import variable_descriptions_csv_path, stations_csv_path

xr.set_options(keep_attrs=True)


def _export_to_netcdf(data, save_name):
    """
    exports user-selected data to netCDF format.
    this function is called from the _export_to_user
    function if the user selected netCDF output.

    data: xarray dataset or array to export
    save_name: string corresponding to desired output file name + file extension
    """
    print("Alright, exporting specified data to NetCDF.")
    comp = dict(_FillValue=None)
    encoding = {coord: comp for coord in data.coords}
    data.to_netcdf(save_name, encoding=encoding)


def _get_unit(dataarray):
    """
    Return unit of data variable in `dataarray`, if any, or an empty string.

    Parameters
    ----------
    dataarray : xarray.DataArray

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
    column_name : str

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
    df : pandas.DataFrame
        data table to update
    variable_unit_map : list of tuple
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
    dataarray : xarray.DataArray
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
    dataset : xarray.Dataset
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

    Export the xarray DataArray or Dataset `data` to a CSV file named
    `save_name`.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        data to export to CSV format
    save_name : string
        desired output file name, including the file extension

    Returns
    -------
    None

    """
    print("Alright, exporting specified data to CSV.")

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

    _metadata_to_file(data, save_name)
    df.to_csv(save_name, compression="gzip")


def export(data, filename="dataexport", format="NetCDF"):
    """Save data as a file in the current working directory.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data to export, as output by e.g. `climakitae.Select().retrieve()`.
    filename : str, optional
        Output file name (without file extension, i.e. "my_filename" instead
        of "my_filename.nc"). The default is "dataexport".
    format : str, optional
        File format ("NetCDF" or "CSV"). The default is "NetCDF".

    """
    ftype = type(data)

    if ftype not in [xr.core.dataset.Dataset, xr.core.dataarray.DataArray]:
        raise Exception(
            "Cannot export object of type "
            + str(ftype).strip("<class >")
            + ". Please pass an xarray dataset or data array."
        )
    ndims = len(data.dims)

    if type(filename) is not str:
        raise Exception(
            (
                "Please pass a string"
                " (any characters surrounded by quotation marks)"
                " for your file name."
            )
        )
    filename = filename.split(".")[0]

    req_format = format

    if req_format is None:
        raise Exception("Please select a file format from the dropdown menu.")

    extension_dict = {"NetCDF": ".nc", "CSV": ".csv.gz"}

    save_name = "./" + filename + extension_dict[req_format]

    if os.path.exists(save_name):
        raise Exception(
            "File "
            + save_name
            + (
                " exists, please either delete that file from the work"
                " space or specify a new file name here."
            )
        )

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

    # metadata stuff
    ds_attrs.update(ck_attrs)
    data.attrs = ds_attrs

    # now check file size and avail workspace disk space
    # raise error for not enough space
    # and warning for large file
    file_size_threshold = 5  # in GB
    bytes_per_gigabyte = 1024 * 1024 * 1024
    disk_space = shutil.disk_usage("./")[2] / bytes_per_gigabyte
    data_size = data.nbytes / bytes_per_gigabyte

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
        _export_to_netcdf(data, save_name)
    elif "CSV" in req_format:
        _export_to_csv(data, save_name)

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


## TMY export functions
def _grab_dem_elev_m(lat, lon):
    """
    Pulls elevation value from the USGS Elevation Point Query Service, 
    lat lon must be in decimal degrees (which it is after cleaning)
    Modified from: 
    https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python
    """
    url = r'https://epqs.nationalmap.gov/v1/json?'
    
    # define rest query params
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }

    # format query string and return value
    result = requests.get((url + urllib.parse.urlencode(params)))
    dem_elev_long = float(result.json()['value'])
    # make sure to round off lat-lon values so they are not improbably precise for our needs
    dem_elev_short = np.round(dem_elev_long, decimals=2) 

    return dem_elev_short.astype("float")

def _utc_offset_timezone(lat, lon):
    '''
    Based on user input of lat lon, returns the UTC offset for that timezone
    Modified from: 
    https://www.reddit.com/r/learnpython/comments/zhatrd/how_to_get_time_offset_of_a_given_coordinates/
    '''
    
    tzn = tf.timezone_at(lng=lon, lat=lat)
    tz = pytz.timezone(tzn)
    dt = datetime.utcnow()

    offset_seconds = tz.utcoffset(dt).seconds
    offset_hours = offset_seconds / 3600.0
    diff = "{:+d}:{:02d}".format(int(offset_hours), int((offset_hours % 1) * 60))

    return diff


def _tmy_header(location_name, station_code, state, timezone, df):
    """
    Constructs the header for the TMY output file in .tmy format
    Source: https://www.nrel.gov/docs/fy08osti/43156.pdf (pg. 3)
    """

    # line 1 - site information
    # line 1: USAF, station name quote delimited, state, time zone, lat, lon, elev (m), simulation
    line_1 = "{0}, '{1}', {2}, {3}, {4}, {5}, {6}, {7}\n".format(
        station_code, 
        location_name,
        state,
        _utc_offset_timezone(lon=df["lon"].values[0], lat=df["lat"].values[0]),
        df["lat"].values[0],
        df["lon"].values[0],
        _grab_dem_elev_m(lat=df["lat"].values[0], lon=df["lon"].values[0]),
        df["simulation"].values[0]
    )

    # line 2 - data field name and units, manually setting to ensure matches TMY3 labeling
    line_2 = "Air Temperature at 2m (degC),Dew point temperature (degC),Relative humidity (%),Instantaneous downwelling shortwave flux at bottom (W m-2),Shortwave surface downward direct normal irradiance (W m-2),Shortwave surface downward diffuse irradiance (W m-2),Instantaneous downwelling longwave flux at bottom (W m-2),Wind speed at 10m (m s-1),Wind direction at 10m (deg),Surface Pressure (Pa)\n"

    headers = [line_1, line_2]

    return headers


def _epw_header(location_name, station_code, state, timezone, df):
    """
    Constructs the header for the TMY output file in .epw format
    Source: EnergyPlus Version 23.1.0 Documentation
    """

    # line 1 - location, location name, state, country, WMO, lat, lon
    # line 1 - location, location name, state, country, weather station number (2 cols), lat, lon, time zone, elevation
    line_1 = "LOCATION,{0},{1},USA,{2},{3},{4},{5},{6}\n".format(
        location_name.upper(),
        state,
        "Custom_{}".format(station_code),
        station_code,
        df["lat"].values[0],
        df["lon"].values[0],
        _utc_offset_timezone(lon=df["lon"].values[0], lat=df["lat"].values[0]),
        _grab_dem_elev_m(lat=df["lat"].values[0], lon=df["lon"].values[0])
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


def _epw_format_data(df):
    """
    Constructs TMY output file in specific order and missing data codes
    Source: EnergyPlus Version 23.1.0 Documentation
    """

    # set time col to datetime object for easy split
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M")
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


def write_tmy_file(filename_to_export, df, location_name="location", station_code="custom", file_ext="tmy"):
    """Exports TMY data either as .epw or .tmy file

    Parameters
    ---------
    filename_to_export (str): Filename string, constructed with station name and simulation
    df (pd.DataFrame): Dataframe of TMY data to export
    location_name (str, optional): Location name string, often station name
    file_ext (str, optional): File extension for export, default is .tmy, options are "tmy" and "epw"

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

    # custom station code handling
    if type(station_code) == str: # custom code passed
        station_code = station_code
        state = 'XX' # will need to look up via lat/lon
        timezone = '-8' # will need to look up via lat/lon
    elif type(station_code) == int: # hadisd statio code passed
        # look up info
        if station_code in station_df['station id'].values:
            state = station_df.loc[station_df['station id'] == station_code]['state'].values[0]
            station_code = str(station_code)[:6]
            timezone = '-8'
 
    # typical meteorological year format
    if file_ext == "tmy":
        path_to_file = filename_to_export + ".tmy"

        with open(path_to_file, "w") as f:
            f.writelines(_tmy_header(location_name, station_code, state, timezone, df))  # writes required header lines
            df = df.drop(
                columns=["simulation", "lat", "lon", "scenario"]
            )  # drops header columns from df
            dfAsString = df.to_csv(sep=",", header=False, index=False)
            f.write(dfAsString)  # writes data in TMY format
        print(
            "TMY data exported to .tmy format with filename {}.tmy".format(
                filename_to_export
            )
        )

    # energy plus weather format
    elif file_ext == "epw":
        path_to_file = filename_to_export + ".epw"
        with open(path_to_file, "w") as f:
            f.writelines(_epw_header(location_name, station_code, state, timezone, df))  # writes required header lines
            df_string = _epw_format_data(df).to_csv(sep=",", header=False, index=False)
            f.write(df_string)  # writes data in EPW format
        print(
            "TMY data exported to .epw format with filename {}.epw".format(
                filename_to_export
            )
        )

    else:
        print('Please pass either "tmy" or "epw" as a file format for export.')
