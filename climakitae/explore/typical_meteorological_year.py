"""
Functions for Typical Meteorological Year creation.

This code has been ported from the cae-notebooks typical_meteorological_year notebook.
It includes statistical code for creating cumulative distributions and the F-S statistic
along with a TMY class that organizes the workflow code.
"""

import numpy as np
import pandas as pd
import pkg_resources
import xarray as xr
from scipy import optimize
from tqdm.auto import tqdm  # Progress bar

from climakitae.core.constants import UNSET
from climakitae.core.data_export import write_tmy_file
from climakitae.core.data_interface import get_data
from climakitae.tools.derived_variables import compute_relative_humidity
from climakitae.util.utils import (
    convert_to_local_time,
    get_closest_gridcell,
    add_dummy_time_to_wl,
)

WEIGHTS_PER_VAR = {
    "Daily max air temperature": 1 / 20,
    "Daily min air temperature": 1 / 20,
    "Daily mean air temperature": 2 / 20,
    "Daily max dewpoint temperature": 1 / 20,
    "Daily min dewpoint temperature": 1 / 20,
    "Daily mean dewpoint temperature": 2 / 20,
    "Daily max wind speed": 1 / 20,
    "Daily mean wind speed": 1 / 20,
    "Global horizontal irradiance": 5 / 20,
    "Direct normal irradiance": 5 / 20,
}


def match_str_to_wl(warming_level: float | int) -> str:
    """Return warming level description string.

    Parameters
    ----------
    warming_level: float | int
        A standard warming level

    Returns
    -------
    str
        A string translating warming level to period in century
    """
    match warming_level:
        case _ if warming_level < 1.5:
            return "present-day"
        case 1.5:
            return "near-future"
        case 2.0:
            return "mid-century"
        case 2.5:
            return "mid-late-century"
        case 3.0:
            return "late-century"
        case _:
            return f"warming-level-{warming_level}"


def is_HadISD(station_name: str) -> bool:
    """Return true if station_name matches a HadISD station name.

    Parameters
    ----------
    station_name: str
        Name of station
    """
    stn_file = pkg_resources.resource_filename("climakitae", "data/hadisd_stations.csv")
    stn_file = pd.read_csv(stn_file, index_col=[0])
    if station_name in list(stn_file["station"]):
        return True
    return False


def _compute_cdf(da: xr.DataArray) -> xr.DataArray:
    """Compute the cumulative density function for an input DataArray.

    Parameters
    ----------
    da: xr.DataArray
        Single simulation and month

    Returns
    -------
    xr.DataArray
    """
    da_np = da.values  # Get numpy array of values
    num_samples = 1024  # Number of samples to generate
    count, bins_count = np.histogram(  # Create a numpy histogram of the values
        da_np,
        bins=np.linspace(
            da_np.min(),  # Start at the minimum value of the array
            da_np.max(),  # End at the maximum value of the array
            num_samples,
        ),
    )
    cdf_np = np.cumsum(count / sum(count))  # Compute the CDF

    # Turn the CDF array into xarray DataArray
    # New dimension is the bin values
    cdf_da = xr.DataArray(
        [bins_count[1:], cdf_np],
        dims=["data", "bin_number"],
        coords={
            "data": ["bins", "probability"],
        },
    )
    cdf_da.name = da.name
    return cdf_da


def _get_cdf_by_sim(da: xr.DataArray) -> xr.DataArray:
    """Function to help get_cdf.

    Parameters
    ----------
    da: xr.DataArray
        Hourly data

    Returns
    -------
    xr.DataArray
    """
    # Group the DataArray by simulation
    return da.groupby("simulation").map(_compute_cdf)


def _get_cdf_by_mon_and_sim(da: xr.DataArray) -> xr.DataArray:
    """Function to help get_cdf.

    Parameters
    ----------
    da: xr.DataArray
        Hourly data

    Returns
    -------
    xr.DataArray
    """
    # Group the DataArray by month in the year
    return da.groupby("time.month").map(_get_cdf_by_sim)


def get_cdf(ds: xr.DataArray) -> xr.Dataset:
    """Get the cumulative density function.

    Parameters
    -----------
    ds: xr.DataArray
        Input data for which to compute CDF

    Returns
    -------
    xr.Dataset
    """
    return ds.map(_get_cdf_by_mon_and_sim)


def get_cdf_monthly(ds: xr.DataArray) -> xr.Dataset:
    """Get the cumulative density function by unique mon-yr combos

    Parameters
    -----------
    ds: xr.DataArray
        Input data for which to compute CDF

    Returns
    -------
    xr.Dataset
    """

    def get_cdf_mon_yr(da):
        return da.groupby("time.year").map(_get_cdf_by_mon_and_sim)

    return ds.map(get_cdf_mon_yr)


def remove_pinatubo_years(ds: xr.Dataset) -> xr.Dataset:
    """Drop years after Pinatubo eruption from dataset.

    Parameters
    ----------
    ds: xr.Dataset

    Returns
    -------
    xr.Dataset
    """
    ds = ds.where((~ds.year.isin([1991, 1992, 1993, 1994])), np.nan, drop=True)
    return ds


def fs_statistic(cdf_climatology: xr.Dataset, cdf_monthly: xr.Dataset) -> xr.Dataset:
    """Calculates the Finkelstein-Schafer statistic.

    Absolute difference between long-term climatology and candidate CDF, divided by number of days in month.

    Parameters
    -----------
    cdf_climatology: xr.Dataset
       Climatological CDF dataset (get_cdf result)
    cdf_monthly: xr.Dataset
       Monthly CDF dataset (get_cdf_monthly result)

    Returns
    -------
    xr.Dataset
        F-S statistic
    """
    days_per_mon = xr.DataArray(
        data=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        coords={"month": np.arange(1, 13)},
    )
    fs_stat = abs(cdf_monthly - cdf_climatology).sel(data="probability") / days_per_mon
    return fs_stat


def compute_weighted_fs(da_fs: xr.Dataset) -> xr.Dataset:
    """Weights the Finkelstein-Schafer (F-S) statistic based on TMY3 methodology.

    Uses weights recommended by NREL for TMY files. See https://nsrdb.nrel.gov/data-sets/tmy.

    Parameters
    -----------
    da_fs: xr.Dataset
       F-S statistic

    Returns
    -------
    xr.Dataset
        Weighted F-S statistic

    """

    for var, weight in WEIGHTS_PER_VAR.items():
        # Multiply each variable by it's appropriate weight
        da_fs[var] = da_fs[var] * weight
    return da_fs


def compute_weighted_fs_sum(
    cdf_climatology: xr.Dataset, cdf_monthly: xr.Dataset
) -> xr.DataArray:
    """Sum f-s statistic over variable and bin number.

    Parameters
    ----------
    cdf_climatology: xr.Dataset
       Climatological CDF dataset (get_cdf result)
    cdf_monthly: xr.Dataset
       Monthly CDF dataset (get_cdf_monthly result)

    Returns
    -------
    xr.DataArray
    """
    all_vars_fs = fs_statistic(cdf_climatology, cdf_monthly)
    weighted_fs = compute_weighted_fs(all_vars_fs)

    # Sum
    return (
        weighted_fs.to_array().sum(dim=["variable", "bin_number"]).drop_vars(["data"])
    )


def get_top_months(da_fs: xr.DataArray, skip_last: bool = False) -> pd.DataFrame:
    """Return dataframe of top months by simulation.

    Parameters
    ----------
    da_fs: xr.Dataset
       Summed weighted f-s statistic
    skip_last: bool
        True to exclude the final month, e.g. if data missing after time conversion

    Returns
    -------
    pd.DataFrame

    """
    df_list = []
    num_values = (
        1  # Selecting the top value for now, persistence statistics calls for top 5
    )
    for sim in da_fs.simulation.values:
        for mon in da_fs.month.values:
            da_i = da_fs.sel(month=mon, simulation=sim)
            top_xr = da_i.sortby(da_i, ascending=True)[:num_values].expand_dims(
                ["month", "simulation"]
            )
            if num_values == 1 & skip_last:
                # Check that last year/month not chosen
                if top_xr.year == da_fs.year[-1]:
                    if top_xr.month == da_fs.month[-1]:
                        # If chosen, exclude it and pick the next match
                        # This logic can be folded into persistence statistics when those are developed
                        top_xr = da_i.sortby(da_i, ascending=True)[1:2].expand_dims(
                            ["month", "simulation"]
                        )
            top_df_i = top_xr.to_dataframe(name="top_values")
            df_list.append(top_df_i)

    # Concatenate list together for all months and simulations
    return pd.concat(df_list).drop(columns=["top_values"]).reset_index()


class TMY:
    """Encapsulate the code needed to generate Typical Meteorological Year (TMY) files.

    Uses WRF hourly data to produce TMYs. User provides the start and end years along
    with location to generate file.

    How to set location: The location can either be provided as latitude and
    longitude coordinates or as the name of a HadISD station in California. Do not set
    `latitude` and `longitude` if using a HadISD station. If `latitude` and `longitude`
    are set along with a custom value for `station_name` (NOT a HadISD station), the
    custom station name will be used in file headers where appropriate.

    How to set time period: The time period can either be set with a time approach using
    `start_year` and `end_year` or with a warming level approach using `warming_level`.
    If the warming level approach is used, a 30-year period is obtained centered around
    the given warming level and the start and end years are taken for that warming level.


    Parameters
    ----------
    start_year : str
        Initial year of TMY period (time approach)
    end_year : str
        Final year of TMY period (time approach)
    warming_level: float | int
        Desired warming level (warming level approach)
    station_name: str (optional)
        Long name of desired station
    latitude : float | int (optional)
        Latitude for TMY data if station_name not set
    longitude : float | int (optional)
        Longitude for TMY data if station_name not set
    verbose: bool
        True to increase verbosity

    Attributes
    ----------
    start_year: str
        Initial year of TMY period
    end_year: str
        Final year of TMY period
    warming_level: float | int
        Warming level value
    lat_range: tuple
        Pair of latitudes that bracket `latitude`
    lon_range: tuple
        Pair of longitudes that bracket `longitude`
    simulations: list[str]
        List of included simulations
    scenario: list[str]
        List of scenarios
    vars_and_units: dict[str,str]
        Dictionary of all required variables and units
    verbose: bool
        True to increase verbosity
    cdf_climatology: xr.Dataset
        CDF climatology data
    cdf_monthly: xr.Dataset
        CDF monthly data by model
    weighted_fs_sum: xr.Dataset
        Weighted F-S statistic results
    top_months: pd.DataFrame
        Table of top months by model
    all_vars: xr.Dataset
        All loaded variables for TMY
    tmy_data_to_export: dict[pd.Dataframe]
        Dictionary of TMY results by simulation
    _skip_last: bool
        Internal flag to track last year for warming level approach
    """

    def __init__(
        self,
        start_year: int = UNSET,
        end_year: int = UNSET,
        warming_level: float | int = UNSET,
        station_name: str = UNSET,
        latitude: float | int = UNSET,
        longitude: float | int = UNSET,
        verbose: bool = True,
    ):

        # Here we go through a few different ways to get the TMY location
        match latitude, longitude, station_name:
            # UNSET will match to object type
            # Case 1: All variables set
            case float() | int(), float() | int(), str():
                if is_HadISD(station_name):
                    raise ValueError(
                        "Do not set `latitude` and `longitude` when using a HadISD station for `station_name`. Change `station_name` value if using custom location."
                    )
                else:
                    print(
                        f"Initializing TMY object for custom location: {latitude} N, {longitude} E with name '{station_name}'."
                    )
                    self._set_loc_from_lat_lon(latitude, longitude)
                    self.stn_name = station_name
            # Case 2: lat/lon provided, no station_name string
            case float() | int(), float() | int(), object():
                print(
                    f"Initializing TMY object for custom location: {latitude} N, {longitude} E."
                )
                self._set_loc_from_lat_lon(latitude, longitude)
            # Case 3: station name provided, lat/lon not numeric
            case object(), object(), str():
                print(f"Initializing TMY object for {station_name}.")
                self._set_loc_from_stn_name(station_name)
            # Last case: something else provided
            case _:
                raise ValueError(
                    "No valid station name or latitude and longitude provided."
                )
        # Time variables
        match start_year, end_year, warming_level:
            # User set all options - bad
            case float() | int(), float() | int(), float() | int():
                raise ValueError(
                    "Variables `start_year` and `end_year` cannot be paired with `warming_level`. Set either `start_year` and `end_year` OR `warming_level."
                )
            # Some other combo - unset variable will be saved as UNSET
            case _:
                self.start_year = start_year
                self.end_year = end_year
                self.warming_level = warming_level
                if isinstance(self.warming_level, int):
                    self.warming_level = float(self.warming_level)
        # Whether to drop the last month as a possible match
        self._skip_last = False
        if self.warming_level:
            # True for warming levels because final hours get lost to UTC conversion
            self._skip_last = True
        # Ranges used in get_data to pull smaller dataset without warnings
        self.lat_range = (self.stn_lat - 0.1, self.stn_lat + 0.1)
        self.lon_range = (self.stn_lon - 0.1, self.stn_lon + 0.1)
        # These 4 simulations have the solar variables needed
        self.simulations = [
            "WRF_EC-Earth3_r1i1p1f1",
            "WRF_MPI-ESM1-2-HR_r3i1p1f1",
            "WRF_TaiESM1_r1i1p1f1",
            "WRF_MIROC6_r1i1p1f1",
        ]
        # Data only available for these scenarios
        self.scenario = ["Historical Climate", "SSP 3-7.0"]
        # These are the variables used in TMY
        self.vars_and_units = {
            "Air Temperature at 2m": "degC",
            "Dew point temperature": "degC",
            "Relative humidity": "[0 to 100]",
            "Instantaneous downwelling shortwave flux at bottom": "W/m2",
            "Shortwave surface downward direct normal irradiance": "W/m2",
            "Shortwave surface downward diffuse irradiance": "W/m2",
            "Instantaneous downwelling longwave flux at bottom": "W/m2",
            "Wind speed at 10m": "m s-1",
            "Wind direction at 10m": "degrees",
            "Surface Pressure": "Pa",
            "Water Vapor Mixing Ratio at 2m": "g kg-1",
        }
        self.verbose = verbose
        # These will get set later in analysis
        self.cdf_climatology = UNSET
        self.cdf_monthly = UNSET
        self.weighted_fs_sum = UNSET
        self.top_months = UNSET
        self.all_vars = UNSET
        self.tmy_data_to_export = UNSET

    def _set_loc_from_stn_name(self, station_name: str):
        """Get coordinates and other station metadata from station

        Parameters
        ----------
        station_name: str
           Name of HadISD station
        """
        # read in station file of CA HadISD stations
        stn_file = pkg_resources.resource_filename(
            "climakitae", "data/hadisd_stations.csv"
        )
        stn_file = pd.read_csv(stn_file, index_col=[0])
        # grab airport
        try:
            self.stn_name = station_name
            self.stn_code = stn_file.loc[stn_file["station"] == self.stn_name][
                "station id"
            ].item()
            one_station = stn_file.loc[stn_file["station"] == self.stn_name]
        except ValueError as e:
            raise ValueError(
                "Could  not find station in hadisd_stations.csv. Please provide valid station name."
            ) from e
        self.stn_lat = one_station.LAT_Y.item()
        self.stn_lon = one_station.LON_X.item()
        self.stn_state = one_station.state.item()

    def _set_loc_from_lat_lon(self, latitude: float | int, longitude: float | int):
        """Set class attributes based on latitude and longitude variables.

        Parameters:
        -----------
        latitude: float
           Location latitude in degrees
        longitude: float
           Location longitude in degrees
        """
        self.stn_lat = latitude
        self.stn_lon = longitude
        # Set station variables to string "None"
        # to be written to file with final data
        self.stn_name = "None"
        self.stn_code = "None"
        self.stn_state = "None"

    def _vprint(self, msg: str):
        """Checks verbosity and prints as allowed."""
        if self.verbose:
            print(msg)

    def _load_time_approach(self, varname: str, units: str) -> xr.DataArray:
        """Run get_data with the time level approach.

        Parameters
        ----------
        varname: str
           Name of desired catalog variable
        units: str
           Desired units

        Returns
        -------
        xr.DataArray
        """
        # Extra year in UTC time to get full period in local time.
        if self.end_year == 2100:
            print(
                "End year is 2100. The final day in timeseries may be incomplete after data is converted to local time."
            )
            new_end_year = self.end_year
        else:
            new_end_year = self.end_year + 1

        data = get_data(
            variable=varname,
            resolution="3 km",
            timescale="hourly",
            data_type="Gridded",
            units=units,
            latitude=self.lat_range,
            longitude=self.lon_range,
            area_average="No",
            scenario=self.scenario,
            time_slice=(self.start_year, new_end_year),
        )
        return data

    def _load_warming_level_approach(self, varname: str, units: str) -> xr.DataArray:
        """Run get_data with the warming level approach.

        Parameters
        ----------
        varname: str
           Name of desired catalog variable
        units: str
           Desired units

        Returns
        -------
        xr.DataArray
        """
        # Extra year in UTC time to get full period in local time.
        data = get_data(
            variable=varname,
            resolution="3 km",
            timescale="hourly",
            data_type="Gridded",
            units=units,
            latitude=self.lat_range,
            longitude=self.lon_range,
            area_average="No",
            approach="Warming Level",
            warming_level=[self.warming_level],
        )
        data = add_dummy_time_to_wl(data)
        # Set the start and end years based on the dummy time
        self.start_year = data.time[0].dt.year.item()
        self.end_year = data.time[-1].dt.year.item()

        return data

    def _load_single_variable(self, varname: str, units: str) -> xr.DataArray:
        """Fetch catalog data for one variable.

        Parameters
        ----------
        varname: str
           Name of desired catalog variable
        units: str
           Desired units

        Returns
        -------
        xr.DataArray
        """
        # Warming level approach
        if self.warming_level is not UNSET:
            data = self._load_warming_level_approach(varname, units)
            simulations = [x + "_historical+ssp370" for x in self.simulations]
        # Use Time approach
        else:
            data = self._load_time_approach(varname, units)
            simulations = self.simulations

        # Compute over single gridcell
        data = get_closest_gridcell(
            data, self.stn_lat, self.stn_lon, print_coords=False
        )
        # Work in local time
        data = convert_to_local_time(data)
        # Get desired time slice in local time
        data = data.sel(
            {"time": slice(f"{self.start_year}-01-01-00", f"{self.end_year}-12-31-23")}
        )
        # Only use preset models with solar variables
        data = data.sel(simulation=simulations)
        return data

    def _get_tmy_variable(self, varname: str, units: str, stats: list[str]) -> list:
        """Fetch a single variable, resample and reduce.

        Parameters
        ----------
        varname: str
           Variable to load.
        units: str
            Desired units.
        stats: list[str]
            Daily stats to compute ('max','min','mean', and/or 'sum')

        Returns
        -------
        xr.Dataset
        """

        data = self._load_single_variable(varname, units)
        returned_data = []
        stat_options = ["max", "min", "mean", "sum"]
        for stat in stat_options:
            if stat not in stats:
                continue
            stat_data = getattr(data.resample(time="1D"), stat)()
            stat_data.attrs["frequency"] = "daily"
            returned_data.append(stat_data)

        return returned_data

    @staticmethod
    def _smooth_month_transition_hours(df: pd.DataFrame) -> pd.DataFrame:
        """Following the NREL procedure, smooth the data in the transitions between months.

        As described in https://docs.nrel.gov/docs/fy08osti/43156.pdf, the hourly data is smoothed
        between months via a curve fit during a 12-hour window centered on day 1, hour 0.
        The radiation variables are not smoothed. Relative humidity during the 12-hour window
        is calculated from smoothed air temperature, surface pressure, and mixing ratio.

        Parameters
        ----------
        df: pd.DataFrame
            Data to smooth for a single simulation

        Returns
        -------
        df: pd.DataFrame
        """
        times = df["time"]
        # Find hour 0 of first day of each month
        times = pd.to_datetime(times)
        day1hour1 = times[(times.dt.day == 1) & (times.dt.hour == 0)][
            1:
        ]  # skip Jan 1 since no prior month
        day1ind = [int(np.where(times.to_numpy() == x)[0][0]) for x in day1hour1]
        # bracket around the 1st of the month
        start_times = [x - 6 for x in day1ind]
        end_times = [x + 6 for x in day1ind]

        # These are the variables getting smoothed. Plus RH, but
        # that gets calculated separately.
        variable_list = [
            "Air Temperature at 2m",
            "Dew point temperature",
            "Wind speed at 10m",
            "Wind direction at 10m",
            "Surface Pressure",
            "Water Vapor Mixing Ratio at 2m",
        ]

        # For each month, do a linear fit to smooth the data in the 12 hour window
        # around day 01 hour 00
        x = np.arange(0, 12)
        for ts, te in zip(start_times, end_times):
            row_ind = np.arange(ts, te)
            # Smooth the listed variables
            for variable in variable_list:
                tseries = df[variable].to_numpy()
                to_smooth = tseries[ts:te]
                # Assign higher certainty to the end points
                # to keep continuity with the surrounding data
                sigma = np.ones(len(to_smooth))
                sigma[[0, -1]] = 0.01

                # Second order polynomial fit
                def f(x, *p):
                    return np.poly1d(p)(x)

                fit, _ = optimize.curve_fit(f, x, to_smooth, (0, 0, 0), sigma=sigma)
                fitted_line = np.poly1d(fit)(x)
                df.loc[row_ind, variable] = np.float32(fitted_line)

            # Smoothed relative humidity has to be calculated from
            # fitted temperature, mixing ratio, and pressure
            pressure_da = xr.DataArray(
                df.loc[row_ind, "Surface Pressure"] / 100.0
            )  # to hPa
            t2_da = xr.DataArray(df.loc[row_ind, "Air Temperature at 2m"])  # C
            q2_da = xr.DataArray(
                df.loc[row_ind, "Water Vapor Mixing Ratio at 2m"]
            )  # g/kg
            rh_da = compute_relative_humidity(
                pressure=pressure_da,  # hPa
                temperature=t2_da,  # degC
                mixing_ratio=q2_da,  # g/kg
            ).data
            df.loc[row_ind, "Relative humidity"] = np.float32(rh_da)
        return df

    @staticmethod
    def _make_8760_tables(all_vars_ds: xr.Dataset, top_months: pd.DataFrame) -> dict:
        """Extract top months from loaded data and arrange in table.

        Pulled out of the run_tmy_analysis() code for easier testing.

        Parameters
        ----------
        all_vars_ds: xr.Dataset
           Timeseries of all loaded variables needed for TMY.
        top_months: pd.DataFrame
           Dataframe of top months by model.

        Returns
        -------
        pd.DataFrame
        """
        tmy_df_all = {}
        for sim in all_vars_ds.simulation.values:
            df_list = []
            print(f"Calculating TMY for simulation: {sim}")
            for mon in tqdm(np.arange(1, 13, 1)):
                # Get year corresponding to month and simulation combo
                year = top_months.loc[
                    (top_months["month"] == mon) & (top_months["simulation"] == sim)
                ].year.item()

                # Select data for unique month, year, and simulation
                data_at_stn_mon_sim_yr = all_vars_ds.sel(
                    simulation=sim, time=f"{mon}-{year}"
                ).expand_dims("simulation")

                # Reformat as dataframe
                df_by_mon_sim_yr = data_at_stn_mon_sim_yr.to_dataframe()
                df_by_mon_sim_yr = df_by_mon_sim_yr.reset_index()

                # Reformat time index to remove seconds
                df_by_mon_sim_yr["time"] = pd.to_datetime(
                    df_by_mon_sim_yr["time"].values
                ).strftime("%Y-%m-%d %H:%M")
                df_list.append(df_by_mon_sim_yr)

            # Concatenate all DataFrames together
            tmy_df_by_sim = pd.concat(df_list)

            tmy_df_all[sim] = tmy_df_by_sim
        return tmy_df_all

    def load_all_variables(self):
        """Load the datasets needed to create TMY."""
        print("Loading data from catalog. Expected runtime: 7 minutes")

        # Configuration for each variable group
        variable_configs = [
            {
                "name": "air temperature",
                "variable": "Air Temperature at 2m",
                "units": "degC",
                "stats": ["max", "min", "mean"],
                "output_names": [
                    "Daily max air temperature",
                    "Daily min air temperature",
                    "Daily mean air temperature",
                ],
            },
            {
                "name": "dew point temperature",
                "variable": "Dew point temperature",
                "units": "degC",
                "stats": ["max", "min", "mean"],
                "output_names": [
                    "Daily max dewpoint temperature",
                    "Daily min dewpoint temperature",
                    "Daily mean dewpoint temperature",
                ],
            },
            {
                "name": "wind speed",
                "variable": "Wind speed at 10m",
                "units": "m s-1",
                "stats": ["max", "mean"],
                "output_names": ["Daily max wind speed", "Daily mean wind speed"],
            },
            {
                "name": "global irradiance",
                "variable": "Instantaneous downwelling shortwave flux at bottom",
                "units": "W/m2",
                "stats": ["sum"],
                "output_names": ["Global horizontal irradiance"],
            },
            {
                "name": "direct normal irradiance",
                "variable": "Shortwave surface downward direct normal irradiance",
                "units": "W/m2",
                "stats": ["sum"],
                "output_names": ["Direct normal irradiance"],
            },
        ]

        # Load and process each variable group
        all_data_arrays = []
        for config in variable_configs:
            print(f"  Getting {config['name']}", end="... ")

            # Get the data using the refactored _get_tmy_variable method
            data_list = self._get_tmy_variable(
                config["variable"], config["units"], config["stats"]
            )

            # Rename each data array and add to collection
            for data_array, output_name in zip(data_list, config["output_names"]):
                data_array.name = output_name
                all_data_arrays.append(data_array.squeeze())

        self._vprint("  Loading all variables into memory.")
        all_vars = xr.merge(all_data_arrays)

        # load all indices in
        self.all_vars = all_vars.compute()
        self._vprint("  All TMY variables loaded.")

    def set_cdf_climatology(self):
        """Calculate the long-term climatology for each index for each month so
        we can establish the baseline pattern.
        """
        if self.all_vars is UNSET:
            self.load_all_variables()
        self._vprint("Calculating CDF climatology.")
        self.cdf_climatology = get_cdf(self.all_vars)

    def set_cdf_monthly(self):
        """Get CDF for each month and variable."""
        if self.all_vars is UNSET:
            self.load_all_variables()
        self._vprint("Calculating monthly CDF.")
        self.cdf_monthly = get_cdf_monthly(self.all_vars)
        # Remove the years for the Pinatubo eruption
        self.cdf_monthly = remove_pinatubo_years(self.cdf_monthly)

    def set_weighted_statistic(self):
        """Calculate the weighted F-S statistic."""
        if self.cdf_climatology is UNSET:
            self.set_cdf_climatology()
        if self.cdf_monthly is UNSET:
            self.set_cdf_monthly()
        self._vprint("Calculating weighted F-S statistic.")
        self.weighted_fs_sum = compute_weighted_fs_sum(
            self.cdf_climatology, self.cdf_monthly
        )

    def set_top_months(self):
        """Calculate top months dataframe."""
        # Pass the weighted F-S sum data for simplicity
        if self.weighted_fs_sum is UNSET:
            self.set_weighted_statistic()
        self._vprint("Finding top months (lowest F-S statistic)")
        self.top_months = get_top_months(
            self.weighted_fs_sum, skip_last=self._skip_last
        )

    def show_tmy_data_to_export(self, simulation: str):
        """Show line plots of TMY data for single model.

        Parameters
        ----------
        simulation: str
            Simulation to display.

        """
        if self.tmy_data_to_export is UNSET:
            print("No TMY data generated.")
            print("Please run TMY.generate_tmy() to create TMY data for viewing.")
            return

        # WVMR not in final TMY dataframe
        fig_y = list(self.vars_and_units.keys())
        fig_y.remove("Water Vapor Mixing Ratio at 2m")

        self.tmy_data_to_export[simulation].plot(
            x="time",
            y=fig_y,
            title=f"Typical Meteorological Year ({simulation})",
            subplots=True,
            figsize=(10, 8),
            legend=True,
        )

    def run_tmy_analysis(self):
        """Generate typical meteorological year data
        Output will be a list of dataframes per simulation.
        Print statements throughout the function indicate to the user the progress of the computatioconvert_to_local_time.

        Parameters
        -----------
        top_df: pd.DataFrame
            Table with column values month, simulation, and year
            Each month-sim-yr combo represents the top candidate that has the lowest weighted sum from the FS statistic

        Notes
        -----
        Results are saved to the class variable `tmy_data_to_export`.
        """
        print("Assembling TMY data to export. Expected runtime: 30 minutes")

        self._vprint("  STEP 1: Retrieving hourly data from catalog")
        # Loop through each variable and grab data from catalog
        all_vars_list = []

        for var, units in self.vars_and_units.items():
            print(f"  Getting {var}", end="... ")
            data_by_var = self._load_single_variable(var, units)

            # Drop unwanted coords
            data_by_var = data_by_var.squeeze().drop_vars(
                ["lakemask", "landmask", "x", "y", "Lambert_Conformal"]
            )

            all_vars_list.append(data_by_var)  # Append to list

        # Merge data from all variables into a single xr.Dataset object
        all_vars_ds = xr.merge(all_vars_list)

        # Construct TMY
        self._vprint(
            "\n  STEP 2: Calculating Typical Meteorological Year per model simulation\n  Progress bar shows code looping through each month in the year.\n"
        )

        tmy_data_to_export = self._make_8760_tables(
            all_vars_ds, self.top_months
        )  # Return dict of TMY by simulation

        self._vprint("  Smoothing data at transitions between months.")
        self._vprint("  Dropping water vapor mixing ratio.")
        # Smooth transition hours
        for sim in tmy_data_to_export:
            tmy_data_to_export[sim] = tmy_data_to_export[sim].reset_index()
            tmy_data_to_export[sim] = self._smooth_month_transition_hours(
                tmy_data_to_export[sim]
            )

            # Mixing ratio was only needed for smoothing relative humidity,
            # so it can be dropped now.
            tmy_data_to_export[sim] = tmy_data_to_export[sim].drop(
                columns="Water Vapor Mixing Ratio at 2m"
            )
        self.tmy_data_to_export = tmy_data_to_export
        self._vprint("TMY analysis complete.")

    def export_tmy_data(self, extension: str = "epw"):
        """Write TMY data to EPW file.

        Parameters
        ----------
        extension: str
            Desired file extension ('tmy','epw', or 'csv')

        """
        print("Exporting TMY to file.")
        for sim, _ in self.tmy_data_to_export.items():
            # Get right year range
            if self.warming_level is UNSET:
                years = (self.start_year, self.end_year)
                clean_sim = sim
            else:
                centered_year = self.all_vars.sel(simulation=sim).centered_year.data
                year1 = centered_year - 15
                year2 = centered_year + 14
                years = (year1, year2)
                # replace scenario with descriptive name if present for gwl case
                clean_sim = sim.replace(
                    "_historical+ssp370", f"_{match_str_to_wl(self.warming_level)}"
                )
            clean_stn_name = (
                self.stn_name.replace(" ", "_").replace("(", "").replace(")", "")
            )
            filename = f"TMY_{clean_stn_name}_{clean_sim}".lower()
            write_tmy_file(
                filename,
                self.tmy_data_to_export[sim],
                years,
                self.stn_name,
                self.stn_code,
                self.stn_lat,
                self.stn_lon,
                self.stn_state,
                file_ext=extension,
            )

    def get_candidate_months(self):
        """Run CDF functions to get top candidates.

        This function can be used to view the candidate months
        without running the entire TMY workflow.
        """
        self._vprint(
            "Getting top months for TMY. Expected runtime with loaded data: 1 min"
        )
        self.set_cdf_climatology()
        self.set_cdf_monthly()
        self.set_weighted_statistic()
        self.set_top_months()

    def generate_tmy(self):
        """Run the whole TMY workflow."""
        # This runs the whole workflow at once
        print("Running TMY workflow. Expected overall runtime: 40 minutes")
        self.load_all_variables()
        self.get_candidate_months()
        self.run_tmy_analysis()
        self.export_tmy_data()
