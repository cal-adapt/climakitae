"""
Functions for Typical Meteorological Year creation.

This code has been ported from the cae-notebooks typical_meteorological_year notebook.
It includes statistical code for creating cumulative distributions and the F-S statistic
along with a TMY class that organizes the workflow code.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import pkg_resources
import pytz
import xarray as xr
from scipy import optimize
from timezonefinder import TimezoneFinder
from tqdm.auto import tqdm  # Progress bar

from climakitae.core.constants import UNSET
from climakitae.core.data_export import write_tmy_file
from climakitae.new_core.user_interface import ClimateData
from climakitae.tools.derived_variables import (
    compute_dewpointtemp,
    compute_relative_humidity,
    compute_wind_dir,
    compute_wind_mag,
)

_hadisd_stations_cache = None


def _wait_with_progress(futures, label="data"):
    """Show a tqdm progress bar while dask distributed futures compute.

    Works with both distributed and local schedulers. Falls back to a
    simple .compute() if no distributed client is available.

    Parameters
    ----------
    futures : dask collection (Dataset/DataArray)
        A persisted dask collection whose futures to track.
    label : str
        Description shown in the progress bar.
    """
    try:
        from dask.distributed import futures_of, wait

        all_futures = futures_of(futures)
        if not all_futures:
            return
        total = len(all_futures)
        with tqdm(total=total, desc=label, unit="task") as pbar:
            done = 0
            while done < total:
                # wait for at least one new future to finish
                wait(all_futures, return_when="FIRST_COMPLETED")
                newly_done = sum(f.status == "finished" for f in all_futures)
                pbar.update(newly_done - done)
                done = newly_done
    except (ImportError, ValueError):
        # No distributed client — futures are already computed
        pass


def _get_hadisd_stations() -> pd.DataFrame:
    """Read and cache the HadISD stations CSV.

    Returns
    -------
    pd.DataFrame
    """
    global _hadisd_stations_cache
    if _hadisd_stations_cache is None:
        stn_file = pkg_resources.resource_filename(
            "climakitae", "data/hadisd_stations.csv"
        )
        _hadisd_stations_cache = pd.read_csv(stn_file, index_col=[0])
    return _hadisd_stations_cache


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
    stn_df = _get_hadisd_stations()
    return station_name in stn_df["station"].values


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

    Computes CDF for each variable, month, and simulation using vectorized
    numpy operations instead of nested xarray groupby.map calls.

    Parameters
    -----------
    ds: xr.DataArray
        Input data for which to compute CDF

    Returns
    -------
    xr.Dataset
    """
    sims = ds.simulation.values
    n_sims = len(sims)
    num_bins = 1023
    time_months = ds.time.dt.month.values

    result_vars = {}
    n_vars = len(ds.data_vars)
    for i, var_name in enumerate(ds.data_vars, 1):
        print(f"  CDF climatology [{i}/{n_vars}]: {var_name}")
        da = ds[var_name].transpose("time", "simulation", ...)
        data = da.values

        combined = np.full((2, 12, n_sims, num_bins), np.nan)
        for m_idx in range(12):
            month_mask = time_months == (m_idx + 1)
            for s_idx in range(n_sims):
                values = data[month_mask, s_idx]
                valid = values[~np.isnan(values)]
                if len(valid) < 2:
                    continue
                bin_edges = np.linspace(valid.min(), valid.max(), num_bins + 1)
                count, _ = np.histogram(valid, bins=bin_edges)
                total = count.sum()
                cdf = np.cumsum(count / total) if total > 0 else np.zeros(num_bins)
                combined[0, m_idx, s_idx] = bin_edges[1:]
                combined[1, m_idx, s_idx] = cdf

        result_vars[var_name] = xr.DataArray(
            combined,
            dims=["data", "month", "simulation", "bin_number"],
            coords={
                "data": ["bins", "probability"],
                "month": np.arange(1, 13),
                "simulation": sims,
            },
        )

    return xr.Dataset(result_vars)


def get_cdf_monthly(ds: xr.DataArray) -> xr.Dataset:
    """Get the cumulative density function by unique mon-yr combos.

    Computes CDF for each variable, year, month, and simulation using vectorized
    numpy operations instead of nested xarray groupby.map calls.

    Parameters
    -----------
    ds: xr.DataArray
        Input data for which to compute CDF

    Returns
    -------
    xr.Dataset
    """
    sims = ds.simulation.values
    n_sims = len(sims)
    num_bins = 1023
    time_months = ds.time.dt.month.values
    time_years = ds.time.dt.year.values
    unique_years = np.unique(time_years)

    result_vars = {}
    n_vars = len(ds.data_vars)
    for i, var_name in enumerate(ds.data_vars, 1):
        print(f"  CDF monthly [{i}/{n_vars}]: {var_name}")
        da = ds[var_name].transpose("time", "simulation", ...)
        data = da.values

        combined = np.full((2, len(unique_years), 12, n_sims, num_bins), np.nan)
        for y_idx, year in enumerate(unique_years):
            year_mask = time_years == year
            for m_idx in range(12):
                ym_mask = year_mask & (time_months == (m_idx + 1))
                for s_idx in range(n_sims):
                    values = data[ym_mask, s_idx]
                    valid = values[~np.isnan(values)]
                    if len(valid) < 2:
                        continue
                    bin_edges = np.linspace(valid.min(), valid.max(), num_bins + 1)
                    count, _ = np.histogram(valid, bins=bin_edges)
                    total = count.sum()
                    cdf = np.cumsum(count / total) if total > 0 else np.zeros(num_bins)
                    combined[0, y_idx, m_idx, s_idx] = bin_edges[1:]
                    combined[1, y_idx, m_idx, s_idx] = cdf

        result_vars[var_name] = xr.DataArray(
            combined,
            dims=["data", "year", "month", "simulation", "bin_number"],
            coords={
                "data": ["bins", "probability"],
                "year": unique_years,
                "month": np.arange(1, 13),
                "simulation": sims,
            },
        )

    return xr.Dataset(result_vars)


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
    if skip_last:
        last_year = int(da_fs.year.values[-1])
        last_month = int(da_fs.month.values[-1])
        # Mask the last year of the last month so it won't be selected
        da_work = da_fs.copy(deep=True)
        da_work.loc[dict(year=last_year, month=last_month)] = np.inf
    else:
        da_work = da_fs

    # For each (simulation, month), find the year with the minimum F-S value
    best_years = da_work.idxmin(dim="year")
    result = best_years.to_dataframe(name="year").reset_index()
    return result[["month", "simulation", "year"]]


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
        # Raw catalog variables to fetch (variable_id → display name)
        # These are fetched directly from the catalog in their native units.
        self._raw_vars = {
            "t2": "Air Temperature at 2m",
            "q2": "Water Vapor Mixing Ratio at 2m",
            "psfc": "Surface Pressure",
            "u10": "u10",
            "v10": "v10",
            "swdnb": "Instantaneous downwelling shortwave flux at bottom",
            "swddni": "Shortwave surface downward direct normal irradiance",
            "swddif": "Shortwave surface downward diffuse irradiance",
            "lwdnb": "Instantaneous downwelling longwave flux at bottom",
        }
        # Full set of TMY variables (including derived) with desired units.
        # Used for display name references throughout the rest of the TMY code.
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
        stn_file = _get_hadisd_stations()
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

    def _get_utc_offset_hours(self) -> int:
        """Compute and cache the UTC offset for this location.

        Returns
        -------
        int
            Hours offset from UTC (negative for west of prime meridian).
        """
        if hasattr(self, "_utc_offset_hours"):
            return self._utc_offset_hours

        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lng=self.stn_lon, lat=self.stn_lat)
        if tz_name is None:
            raise ValueError(
                f"Could not determine timezone for coordinates "
                f"({self.stn_lat}, {self.stn_lon})."
            )
        tz = pytz.timezone(tz_name)
        # Use a fixed reference date for consistent standard-time offset
        offset = tz.utcoffset(datetime(2020, 1, 1))
        self._utc_offset_hours = int(offset.total_seconds() / 3600)
        self._timezone_name = tz_name
        return self._utc_offset_hours

    def _fetch_raw_variable(
        self, variable_id: str, table_id: str = "1hr"
    ) -> xr.DataArray:
        """Fetch a single raw catalog variable via ClimateData.

        Uses the new core ClimateData interface with clip processor
        for point selection and time_slice or warming_level as appropriate.

        Parameters
        ----------
        variable_id : str
            Catalog variable_id (e.g., 't2', 'q2', 'psfc').
        table_id : str
            Temporal resolution: '1hr' or 'day'. Default '1hr'.

        Returns
        -------
        xr.DataArray
        """
        cd = ClimateData(verbosity=-1)
        query = (
            cd.catalog("cadcat")
            .activity_id("WRF")
            .institution_id("UCLA")
            .table_id(table_id)
            .grid_label("d03")
            .variable(variable_id)
        )

        processes = {
            "clip": (self.stn_lat, self.stn_lon),
        }

        if self.warming_level is not UNSET:
            processes["warming_level"] = {
                "warming_levels": [self.warming_level],
                "add_dummy_time": True,
            }
        else:
            # Extra year in UTC time to get full period in local time.
            if self.end_year == 2100:
                print(
                    "End year is 2100. The final day in timeseries may be "
                    "incomplete after data is converted to local time."
                )
                new_end_year = self.end_year
            else:
                new_end_year = self.end_year + 1
            query = query.experiment_id(["historical", "ssp370"])
            processes["time_slice"] = (self.start_year, new_end_year)

        data = query.processes(processes).get()

        if data is None:
            raise RuntimeError(
                f"ClimateData returned no data for variable '{variable_id}'."
            )

        # Extract the variable as a DataArray
        if isinstance(data, xr.Dataset):
            data = data[variable_id]

        # ClimateData uses "sim" dimension; rename to "simulation" for TMY pipeline
        if "sim" in data.dims:
            data = data.rename({"sim": "simulation"})

        # Drop warming_level dimension (always length 1 for TMY)
        if "warming_level" in data.dims:
            data = data.squeeze("warming_level", drop=True)

        # For warming level: set start/end year from dummy time
        if self.warming_level is not UNSET:
            self.start_year = data.time[0].dt.year.item()
            self.end_year = data.time[-1].dt.year.item()

        # Filter to the 4 TMY simulations by matching source_id+member_id
        # ClimateData sim values: "wrf_ucla_ec-earth3_historical+ssp370_r1i1p1f1"
        # self.simulations values: "WRF_EC-Earth3_r1i1p1f1"
        all_sims = list(data.simulation.values)
        sim_mapping = {}  # maps ClimateData sim name → legacy sim name
        for legacy_sim in self.simulations:
            # Extract source_id and member_id from legacy name (e.g. "EC-Earth3", "r1i1p1f1")
            parts = legacy_sim.split("_")
            source_id = parts[1].lower()
            member_id = parts[2].lower()
            for cd_sim in all_sims:
                cd_lower = (
                    cd_sim.lower() if isinstance(cd_sim, str) else str(cd_sim).lower()
                )
                if source_id in cd_lower and member_id in cd_lower:
                    sim_mapping[cd_sim] = legacy_sim
                    break

        # Select and rename to legacy simulation names
        matched_cd_sims = list(sim_mapping.keys())
        data = data.sel(simulation=matched_cd_sims)
        data["simulation"] = [sim_mapping[s] for s in matched_cd_sims]

        # Work in local time (cached offset avoids repeated CSV reads)
        offset_hours = self._get_utc_offset_hours()
        data["time"] = data.time + pd.Timedelta(hours=offset_hours)
        data.attrs["timezone"] = self._timezone_name

        # Get desired time slice in local time
        data = data.sel(
            {"time": slice(f"{self.start_year}-01-01-00", f"{self.end_year}-12-31-23")}
        )
        return data

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
        def _poly2(x, *p):
            """Second order polynomial for curve fitting."""
            return np.poly1d(p)(x)

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

                fit, _ = optimize.curve_fit(
                    _poly2, x, to_smooth, (0, 0, 0), sigma=sigma
                )
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
                year = int(
                    top_months.loc[
                        (top_months["month"] == mon) & (top_months["simulation"] == sim)
                    ].year.item()
                )

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
        """Load hourly and daily TMY variables via ClimateData.

        Fetches hourly raw variables for the 8760 profile assembly,
        and daily variables directly from the catalog for CDF/F-S analysis.
        Derived variables (dew point, radiation sums) are computed from
        the fetched data where no direct catalog variable exists.
        """
        print("Loading data from catalog via ClimateData.")

        def _fetch_and_clean(variable_id, table_id="1hr"):
            """Fetch a single variable and clean coordinate cruft."""
            self._vprint(f"  Fetching {variable_id} ({table_id})...")
            da = self._fetch_raw_variable(variable_id, table_id=table_id)
            return da.squeeze().drop_vars(
                [
                    "lakemask",
                    "landmask",
                    "x",
                    "y",
                    "Lambert_Conformal",
                    "centered_year",
                ],
                errors="ignore",
            )

        # --- Hourly variables (needed for 8760 profile assembly) ---
        # First hourly fetch must be synchronous for warming level (sets year range)
        hourly_var_ids = list(self._raw_vars.keys())
        if self.warming_level is not UNSET:
            self._vprint(f"  Getting {hourly_var_ids[0]} (sets year range)...")
            # Fetch raw first to capture centered_year before cleaning drops it
            raw_first = self._fetch_raw_variable(hourly_var_ids[0], table_id="1hr")
            if "centered_year" in raw_first.coords:
                self._sim_centered_years = dict(
                    zip(
                        raw_first.simulation.values,
                        raw_first.centered_year.values.ravel(),
                    )
                )
            first_var = raw_first.squeeze().drop_vars(
                ["lakemask", "landmask", "x", "y", "Lambert_Conformal", "centered_year"],
                errors="ignore",
            )
            first_var.name = self._raw_vars[hourly_var_ids[0]]
            remaining_hourly = hourly_var_ids[1:]
        else:
            first_var = None
            remaining_hourly = hourly_var_ids

        # --- Daily variables (needed for CDF/F-S analysis) ---
        # Map: catalog variable_id → TMY display name
        daily_catalog_vars = {
            "t2max": "Daily max air temperature",
            "t2min": "Daily min air temperature",
            "t2": "Daily mean air temperature",
            "wspd10max": "Daily max wind speed",
            "wspd10mean": "Daily mean wind speed",
            "rh": "Daily mean relative humidity",  # for dew point derivation
            "sw_dwn": "Global horizontal irradiance",
        }

        # Fetch remaining hourly + all daily in parallel
        self._vprint("  Loading hourly and daily variables in parallel...")

        def _fetch_hourly(vid):
            da = _fetch_and_clean(vid, "1hr")
            da.name = self._raw_vars[vid]
            return ("hourly", vid, da)

        def _fetch_daily(vid):
            da = _fetch_and_clean(vid, "day")
            da.name = daily_catalog_vars[vid]
            da.attrs["frequency"] = "daily"
            return ("daily", vid, da)

        with ThreadPoolExecutor(max_workers=4) as executor:
            hourly_futures = [
                executor.submit(_fetch_hourly, vid) for vid in remaining_hourly
            ]
            daily_futures = [
                executor.submit(_fetch_daily, vid) for vid in daily_catalog_vars
            ]
            all_results = [f.result() for f in hourly_futures + daily_futures]

        # Separate hourly and daily results
        hourly_list = [r[2] for r in all_results if r[0] == "hourly"]
        if first_var is not None:
            hourly_list = [first_var] + hourly_list
        daily_das = {r[1]: r[2] for r in all_results if r[0] == "daily"}

        # --- Build hourly dataset for 8760 profile ---
        self._vprint("  Merging raw hourly data.")
        raw_ds = xr.merge(hourly_list)

        # Compute hourly derived variables
        self._vprint("  Computing hourly derived variables.")
        t2_degc = raw_ds["Air Temperature at 2m"] - 273.15
        t2_degc.attrs["units"] = "degC"

        q2_gkg = raw_ds["Water Vapor Mixing Ratio at 2m"] * 1000
        q2_gkg.attrs["units"] = "g kg-1"

        psfc_hpa = raw_ds["Surface Pressure"] / 100.0
        psfc_hpa.attrs["units"] = "hPa"

        rh = compute_relative_humidity(
            pressure=psfc_hpa,
            temperature=t2_degc,
            mixing_ratio=q2_gkg,
        )
        rh.name = "Relative humidity"
        rh.attrs["units"] = "[0 to 100]"

        dew_point_k = compute_dewpointtemp(
            temperature=raw_ds["Air Temperature at 2m"],
            rel_hum=rh,
        )
        dew_point = dew_point_k - 273.15
        dew_point.name = "Dew point temperature"
        dew_point.attrs["units"] = "degC"

        wind_speed = compute_wind_mag(u10=raw_ds["u10"], v10=raw_ds["v10"])
        wind_speed.name = "Wind speed at 10m"

        wind_dir = compute_wind_dir(u10=raw_ds["u10"], v10=raw_ds["v10"])
        wind_dir.name = "Wind direction at 10m"

        t2_out = t2_degc.copy()
        t2_out.name = "Air Temperature at 2m"
        t2_out.attrs["units"] = "degC"

        q2_out = q2_gkg.copy()
        q2_out.name = "Water Vapor Mixing Ratio at 2m"

        derived_list = [t2_out, dew_point, rh, wind_speed, wind_dir, q2_out]
        keep_vars = [
            "Instantaneous downwelling shortwave flux at bottom",
            "Shortwave surface downward direct normal irradiance",
            "Shortwave surface downward diffuse irradiance",
            "Instantaneous downwelling longwave flux at bottom",
            "Surface Pressure",
        ]
        kept_from_raw = [raw_ds[v] for v in keep_vars]
        hourly_ds = xr.merge(derived_list + kept_from_raw)
        self._hourly_data = hourly_ds

        # --- Build daily dataset for CDF/F-S analysis ---
        self._vprint("  Building daily dataset for CDF analysis.")

        # Convert daily temperature units: K → degC
        for vid in ("t2max", "t2min", "t2"):
            daily_das[vid] = daily_das[vid] - 273.15
            daily_das[vid].attrs["units"] = "degC"

        # Compute daily dew point from daily t2 (now degC) and daily rh
        daily_t2_k = daily_das["t2"] + 273.15  # back to K for dewpoint formula
        daily_dp_k = compute_dewpointtemp(
            temperature=daily_t2_k,
            rel_hum=daily_das["rh"],
        )
        daily_dp = daily_dp_k - 273.15
        daily_dp.attrs["units"] = "degC"

        # Create max/min/mean dew point approximations from daily mean
        daily_dp_max = daily_dp.copy()
        daily_dp_max.name = "Daily max dewpoint temperature"
        daily_dp_min = daily_dp.copy()
        daily_dp_min.name = "Daily min dewpoint temperature"
        daily_dp_mean = daily_dp.copy()
        daily_dp_mean.name = "Daily mean dewpoint temperature"

        # Radiation: derive daily sums from hourly (no daily sum in catalog).
        # Compute eagerly — the resample graphs reference all hourly chunks
        # and are too large for Dask's graph optimizer when merged with the
        # simpler daily catalog arrays.
        self._vprint("  Computing daily radiation sums from hourly data...")
        with ProgressBar():
            ghi_sum = (
                hourly_ds["Instantaneous downwelling shortwave flux at bottom"]
                .resample(time="1D")
                .sum()
                .compute()
            )
            ghi_sum.name = "Global horizontal irradiance"
            ghi_sum.attrs["frequency"] = "daily"

            dni_sum = (
                hourly_ds["Shortwave surface downward direct normal irradiance"]
                .resample(time="1D")
                .sum()
                .compute()
            )
            dni_sum.name = "Direct normal irradiance"
            dni_sum.attrs["frequency"] = "daily"

        # Drop the catalog-fetched sw_dwn (daily mean, not sum) and rh helper
        daily_arrays = [
            daily_das["t2max"],
            daily_das["t2min"],
            daily_das["t2"],
            daily_dp_max,
            daily_dp_min,
            daily_dp_mean,
            daily_das["wspd10max"],
            daily_das["wspd10mean"],
            ghi_sum,
            dni_sum,
        ]

        self._vprint("  Computing daily statistics...")
        with ProgressBar():
            self.all_vars = xr.merge(daily_arrays).compute()
        self._vprint("  Daily statistics ready.")

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
        """Generate typical meteorological year data.

        Output will be a list of dataframes per simulation.
        Print statements throughout the function indicate progress.

        Parameters
        -----------
        top_df: pd.DataFrame
            Table with column values month, simulation, and year
            Each month-sim-yr combo represents the top candidate that has the lowest weighted sum from the FS statistic

        Notes
        -----
        Results are saved to the class variable `tmy_data_to_export`.
        """
        print("Assembling TMY data to export.")

        self._vprint("  STEP 1: Computing hourly data")

        # Use cached hourly data instead of re-downloading
        if not hasattr(self, "_hourly_data") or self._hourly_data is None:
            # Fallback: load from catalog if run_tmy_analysis called standalone
            self.load_all_variables()
        with ProgressBar():
            all_vars_ds = self._hourly_data.compute()

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
                centered_year = int(self._sim_centered_years[sim])
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
        self._vprint("Getting top months for TMY.")
        self.set_cdf_climatology()
        self.set_cdf_monthly()
        self.set_weighted_statistic()
        self.set_top_months()

    def generate_tmy(self):
        """Run the whole TMY workflow."""
        # This runs the whole workflow at once
        print("Running TMY workflow.")
        self.load_all_variables()
        self.get_candidate_months()
        self.run_tmy_analysis()
        self.export_tmy_data()
