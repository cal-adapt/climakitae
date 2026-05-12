"""
Functions for Shock Extreme Meteorological Year creation.

This code has been ported from the cae-notebooks shock_extreme_meteorological_year notebook.
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

# from climakitae.core.data_export import write_tmy_file
#! until updated tmy export function is added, use this
from climakitae.core.dev_data_export import write_tmy_file
from climakitae.new_core.user_interface import ClimateData
from climakitae.tools.derived_variables import (
    compute_dewpointtemp,
    compute_relative_humidity,
    compute_wind_dir,
    compute_wind_mag,
)

from climakitae.explore.typical_meteorological_year import (
    is_HadISD,
    get_cdf,
    remove_pinatubo_years,
    match_str_to_wl,
    get_cdf_monthly,
    _get_hadisd_stations,
)


def find_hot_cold_extreme_from_median(
    sub_month: xr.DataArray,
    sub_clim: xr.DataArray,
    target: float = 0.5,
    extreme: str = "cold",  # "cold" or "hot"
):
    """
    Identifies hottest or coldest year based on deviation from the median (CDF=0.5)
    temperature value across years.

    Parameters
    ----------
    sub_month : xr.DataArray
        dims: (year, data, bin_number)


    sub_clim : xr.DataArray
        climatology with same structure

    target : float
        CDF level (default 0.5)

    extreme : str
        "cold" -> pick minimum deviation
        "hot"  -> pick maximum deviation

    Returns
    -------
    results : list
        median values per year

    worst_year : scalar
        identified extreme year
    """

    # -----------------------------
    # Step 1: climatology median (p50 reference)
    # -----------------------------
    clim_prob = sub_clim.sel(data="probability")
    clim_bins = sub_clim.sel(data="bins")

    clim_idx = np.abs(clim_prob - target).argmin(dim="bin_number")
    clim_05 = clim_bins.isel(bin_number=clim_idx).values

    # -----------------------------
    # Step 2: per-year median extraction
    # -----------------------------
    yr_prob = sub_month.sel(data="probability")
    yr_bins = sub_month.sel(data="bins")

    idx_list = np.abs(yr_prob - target).argmin(dim="bin_number").values.tolist()

    results = []

    years = sub_month["year"].values

    for i, yr in enumerate(years):
        idx = idx_list[i]

        val = yr_bins.sel(year=yr).isel(bin_number=idx).values
        results.append(val)

    results = np.array(results)

    # -----------------------------
    # Step 3: compute anomalies vs climatology
    # -----------------------------
    anomaly = results - clim_05

    # -----------------------------
    # Step 4: pick extreme year
    # -----------------------------
    if extreme == "cold":
        worst_idx = np.argmin(anomaly)  # most negative deviation
    elif extreme == "hot":
        worst_idx = np.argmax(anomaly)  # most positive deviation
    else:
        raise ValueError(f"Extreme must be 'cold' or 'hot', Received {extreme}.")

    worst_year = years[worst_idx]

    return results, anomaly, worst_year


def generate_candidate_months(
    cdf_monthly: xr.DataArray,
    cdf_climatology: xr.DataArray,
    extreme: str = "cold",  # "cold" or "hot"
):
    """
    Run find_hot_cold_extreme_from_median() over entire input dataset.
    Generate a dataframe of selected years per month and simulation
        for input to XMY generation function.

    Parameters
    ----------
    cdf_monthly : xr.DataArray
        Monthly CDF to compare against climatological CDF

    cdf_climatology : xr.DataArray
        CDF representing climatology

    extreme : str
        The type of shock XMY.
        "cold" -> pick minimum deviation
        "hot"  -> pick maximum deviation

    Returns
    -------
    top_df: dataframe of selected years per simulation and month
    """

    # select priority variable based on shock type
    if extreme == "cold":
        var = "Daily min air temperature"
    elif extreme == "hot":
        var = "Daily max air temperature"

    # subset CDFs for priority variable
    subset_clim = cdf_climatology[var]
    subset_month = cdf_monthly[var]

    results = []

    # iterate over all simulations and months
    for sim in cdf_monthly.simulation.values:
        clim_sim = subset_clim.sel(simulation=sim)
        month_sim = subset_month.sel(simulation=sim)
        for mon in cdf_monthly.month.values:
            clim_mon = clim_sim.sel(month=mon)
            month_mon = month_sim.sel(month=mon)
            _, _, worst_year = find_hot_cold_extreme_from_median(
                month_mon, clim_mon, extreme=extreme
            )
            row = {"month": mon, "simulation": sim, "year": worst_year}
            results.append(row)
    top_df = pd.DataFrame(results)

    return top_df


class shock_XMY:
    """Encapsulate the code needed to generate Typical Meteorological Year (shock XMY) files.

    Uses WRF hourly data to produce shock XMYs. User provides the start and end years along
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

    How to set extreme: The extreme can either be set as 'hot' or 'cold'. The 'hot' extreme
    corresponds to extreme events like heat waves, while the 'cold' extreme corresponds to
    extreme evens like cold snaps.

    Parameters
    ----------
    extreme: str, 'hot' or 'cold'
        Type of shock extreme
    start_year : str
        Initial year of XMY period (time approach)
    end_year : str
        Final year of XMY period (time approach)
    warming_level: float | int
        Desired warming level (warming level approach)
    station_name: str (optional)
        Long name of desired station
    latitude : float | int (optional)
        Latitude for shock XMY data if station_name not set
    longitude : float | int (optional)
        Longitude for shock XMY data if station_name not set
    verbose: bool
        True to increase verbosity

    Attributes
    ----------
    extreme: str
        extreme type, either "hot" or "cold"
    start_year: str
        Initial year of shock XMY period
    end_year: str
        Final year of shock XMY period
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
    top_months: pd.DataFrame
        Table of top months by model
    all_vars: xr.Dataset
        All loaded variables for shock XMY
    air_temp_vars: xr.Dataset
        Air temperature variables, for use in finding candidate months
    xmy_data_to_export: dict[pd.Dataframe]
        Dictionary of shock XMY results by simulation
    _skip_last: bool
        Internal flag to track last year for warming level approach
    """

    def __init__(
        self,
        extreme: str = UNSET,
        start_year: int = UNSET,
        end_year: int = UNSET,
        warming_level: float | int = UNSET,
        station_name: str = UNSET,
        latitude: float | int = UNSET,
        longitude: float | int = UNSET,
        verbose: bool = True,
    ):

        # Here we go through a few different ways to get the shock XMY location
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
                        f"Initializing shock XMY object for custom location: {latitude} N, {longitude} W with name '{station_name}'."
                    )
                    self._set_loc_from_lat_lon(latitude, longitude)
                    self.stn_name = station_name
            # Case 2: lat/lon provided, no station_name string
            case float() | int(), float() | int(), object():
                print(
                    f"Initializing shock XMY object for custom location: {latitude} N, {longitude} W."
                )
                self._set_loc_from_lat_lon(latitude, longitude)
            # Case 3: station name provided, lat/lon not numeric
            case object(), object(), str():
                print(f"Initializing shock XMY object for {station_name}.")
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
        # extreme type
        if extreme in ["hot", "cold"]:
            self.extreme = extreme
            print(f"Generating {extreme} shock XMY.")
        else:
            raise TypeError("Variable `extreme` must be either 'hot' or 'cold'.")
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
        # Full set of shock XMY variables (including derived) with desired units.
        # Used for display name references throughout the rest of the shock XMY code.
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
        self.top_months = UNSET
        self.all_vars = UNSET
        self.air_temp_vars = UNSET
        self.xmy_data_to_export = UNSET

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

        # ClimateData uses "sim" dimension; rename to "simulation" for shock XMY pipeline
        if "sim" in data.dims:
            data = data.rename({"sim": "simulation"})

        # Drop warming_level dimension (always length 1 for shock XMY)
        if "warming_level" in data.dims:
            data = data.squeeze("warming_level", drop=True)

        # For warming level: set start/end year from dummy time
        if self.warming_level is not UNSET:
            self.start_year = data.time[0].dt.year.item()
            self.end_year = data.time[-1].dt.year.item()

        # Filter to the 4 shock XMY simulations by matching source_id+member_id
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

        Pulled out of the run_xmy_analysis() code for easier testing.

        Parameters
        ----------
        all_vars_ds: xr.Dataset
           Timeseries of all loaded variables needed for shock XMY.
        top_months: pd.DataFrame
           Dataframe of top months by model.

        Returns
        -------
        pd.DataFrame
        """
        xmy_df_all = {}
        for sim in all_vars_ds.simulation.values:
            df_list = []
            print(f"Calculating shock XMY for simulation: {sim}")
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
            xmy_df_by_sim = pd.concat(df_list)

            xmy_df_all[sim] = xmy_df_by_sim
        return xmy_df_all

    def load_all_variables(self):
        """Load hourly shock XMY variables and derive daily statistics for CDF/F-S.

        Fetches hourly raw variables via ClimateData for the 8760 profile
        assembly, then derives ALL daily statistics from the hourly data
        in local time.  This matches the original shock XMY code's approach and
        avoids two problems with fetching daily catalog variables directly:

        1. **UTC vs local time**: Catalog daily variables are pre-aggregated
           over UTC day boundaries, which differ from local-time days by the
           station's UTC offset (e.g., 8 hours for California).
        2. **Non-determinism**: With a Dask distributed client active, lazy
           reductions (``.resample().sum()``) can produce slightly different
           floating-point results on each run due to non-deterministic task
           ordering.  Computing hourly data eagerly to numpy before
           resampling guarantees deterministic daily statistics.
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
            first_var.name = self._raw_vars[hourly_var_ids[0]]
            remaining_hourly = hourly_var_ids[1:]
        else:
            first_var = None
            remaining_hourly = hourly_var_ids

        # --- Daily variables (needed for CDF/F-S analysis) ---
        # Derive daily stats from hourly data in local time.
        # This matches the original code's behavior (hourly → local time →
        # daily resample) and avoids using catalog daily variables which are
        # pre-aggregated over UTC day boundaries (different 24-hour window).
        # We also compute hourly data eagerly here so all downstream
        # operations use deterministic numpy math, avoiding non-deterministic
        # floating-point reduction ordering from the dask distributed scheduler.

        # Fetch remaining hourly variables in parallel
        self._vprint("  Loading hourly variables in parallel...")

        def _fetch_hourly(vid):
            da = _fetch_and_clean(vid, "1hr")
            da.name = self._raw_vars[vid]
            return ("hourly", vid, da)

        with ThreadPoolExecutor(max_workers=4) as executor:
            hourly_futures = [
                executor.submit(_fetch_hourly, vid) for vid in remaining_hourly
            ]
            all_results = [f.result() for f in hourly_futures]

        # Separate hourly results
        hourly_list = [r[2] for r in all_results]
        if first_var is not None:
            hourly_list = [first_var] + hourly_list

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
        # Compute hourly data eagerly so all daily resampling uses
        # deterministic numpy math (not affected by dask scheduler ordering).
        # For a single grid cell this is small (~30yr × 8760hr × 4sims).
        self._vprint("  Computing hourly data for daily resampling...")
        with ProgressBar():
            hourly_computed = hourly_ds.compute()

        self._vprint("  Resampling hourly data to daily statistics...")
        # Temperature: daily max, min, mean (in degC, already converted above)
        daily_tmax = hourly_computed["Air Temperature at 2m"].resample(time="1D").max()
        daily_tmax.name = "Daily max air temperature"
        daily_tmax.attrs["frequency"] = "daily"

        daily_tmin = hourly_computed["Air Temperature at 2m"].resample(time="1D").min()
        daily_tmin.name = "Daily min air temperature"
        daily_tmin.attrs["frequency"] = "daily"

        daily_tmean = (
            hourly_computed["Air Temperature at 2m"].resample(time="1D").mean()
        )
        daily_tmean.name = "Daily mean air temperature"
        daily_tmean.attrs["frequency"] = "daily"

        # Dew point: daily max, min, mean
        daily_dp_max = (
            hourly_computed["Dew point temperature"].resample(time="1D").max()
        )
        daily_dp_max.name = "Daily max dewpoint temperature"
        daily_dp_max.attrs["frequency"] = "daily"

        daily_dp_min = (
            hourly_computed["Dew point temperature"].resample(time="1D").min()
        )
        daily_dp_min.name = "Daily min dewpoint temperature"
        daily_dp_min.attrs["frequency"] = "daily"

        daily_dp_mean = (
            hourly_computed["Dew point temperature"].resample(time="1D").mean()
        )
        daily_dp_mean.name = "Daily mean dewpoint temperature"
        daily_dp_mean.attrs["frequency"] = "daily"

        # Wind speed: daily max, mean
        daily_ws_max = hourly_computed["Wind speed at 10m"].resample(time="1D").max()
        daily_ws_max.name = "Daily max wind speed"
        daily_ws_max.attrs["frequency"] = "daily"

        daily_ws_mean = hourly_computed["Wind speed at 10m"].resample(time="1D").mean()
        daily_ws_mean.name = "Daily mean wind speed"
        daily_ws_mean.attrs["frequency"] = "daily"

        # Radiation: daily sums
        ghi_sum = (
            hourly_computed["Instantaneous downwelling shortwave flux at bottom"]
            .resample(time="1D")
            .sum()
        )
        ghi_sum.name = "Global horizontal irradiance"
        ghi_sum.attrs["frequency"] = "daily"

        dni_sum = (
            hourly_computed["Shortwave surface downward direct normal irradiance"]
            .resample(time="1D")
            .sum()
        )
        dni_sum.name = "Direct normal irradiance"
        dni_sum.attrs["frequency"] = "daily"

        daily_arrays = [
            daily_tmax,
            daily_tmin,
            daily_tmean,
            daily_dp_max,
            daily_dp_min,
            daily_dp_mean,
            daily_ws_max,
            daily_ws_mean,
            ghi_sum,
            dni_sum,
        ]

        air_temp_var_arrays = [daily_tmax, daily_tmin]

        self.all_vars = xr.merge(daily_arrays)
        self._vprint("  Daily statistics ready.")
        # subet all loaded variables to only air temperature, which is the only variable used to determine the candidate month
        self.air_temp_vars = xr.merge(air_temp_var_arrays)
        self._vprint(
            "   Air temperature variables stored for use in finding candidate months."
        )

    def set_cdf_climatology(self):
        """Calculate the long-term climatology for each index for each month so
        we can establish the baseline pattern.
        """
        if self.air_temp_vars is UNSET:
            self.load_all_variables()
        self._vprint("Calculating CDF climatology using only air temp variables.")
        self.cdf_climatology = get_cdf(self.air_temp_vars)

    def set_cdf_monthly(self):
        """Get CDF for each month and variable."""
        if self.air_temp_vars is UNSET:
            self.load_all_variables()
        self._vprint("Calculating monthly CDF using only air temp variables.")
        self.cdf_monthly = get_cdf_monthly(self.air_temp_vars)
        # Remove the years for the Pinatubo eruption
        self.cdf_monthly = remove_pinatubo_years(self.cdf_monthly)

    def set_top_months(self):
        """Calculate top months dataframe."""
        # Pass the weighted F-S sum data for simplicity

        if self.cdf_climatology is UNSET:
            self.set_cdf_climatology()

        if self.cdf_monthly is UNSET:
            self.set_cdf_monthly()

        self._vprint(
            "Finding top months (greatest deviation from climatological CDF median)."
        )
        self.top_months = generate_candidate_months(
            self.cdf_monthly, self.cdf_climatology, self.extreme
        )

    def show_xmy_data_to_export(self, simulation: str):
        """Show line plots of shock XMY data for single model.

        Parameters
        ----------
        simulation: str
            Simulation to display.

        """
        if self.xmy_data_to_export is UNSET:
            print("No XMY data generated.")
            print("Please run xmy.generate_xmy() to create XMY data for viewing.")
            return

        # WVMR not in final XMY dataframe
        fig_y = list(self.vars_and_units.keys())
        fig_y.remove("Water Vapor Mixing Ratio at 2m")

        self.xmy_data_to_export[simulation].plot(
            x="time",
            y=fig_y,
            title=f"Shock Extreme Meteorological Year ({simulation})",
            subplots=True,
            figsize=(10, 8),
            legend=True,
        )

    def run_xmy_analysis(self):
        """Generate shock extreme meteorological year data.

        Output will be a list of dataframes per simulation.
        Print statements throughout the function indicate progress.

        Parameters
        -----------
        top_df: pd.DataFrame
            Table with column values month, simulation, and year
            Each month-sim-yr combo represents the top candidate that has the lowest weighted sum from the FS statistic

        Notes
        -----
        Results are saved to the class variable `xmy_data_to_export`.
        """
        print("Assembling XMY data to export.")

        self._vprint("  STEP 1: Computing hourly data")

        # Use cached hourly data instead of re-downloading
        if not hasattr(self, "_hourly_data") or self._hourly_data is None:
            # Fallback: load from catalog if run_xmy_analysis called standalone
            self.load_all_variables()
        with ProgressBar():
            all_vars_ds = self._hourly_data.compute()

        # Construct XMY
        self._vprint(
            f"\n  STEP 2: Calculating {self.extreme} Shock Extreme Meteorological Year per model simulation\n  Progress bar shows code looping through each month in the year.\n"
        )

        xmy_data_to_export = self._make_8760_tables(
            all_vars_ds, self.top_months
        )  # Return dict of shock XMY by simulation

        self._vprint("  Smoothing data at transitions between months.")
        self._vprint("  Dropping water vapor mixing ratio.")
        # Smooth transition hours
        for sim in xmy_data_to_export:
            xmy_data_to_export[sim] = xmy_data_to_export[sim].reset_index()
            xmy_data_to_export[sim] = self._smooth_month_transition_hours(
                xmy_data_to_export[sim]
            )

            # Mixing ratio was only needed for smoothing relative humidity,
            # so it can be dropped now.
            xmy_data_to_export[sim] = xmy_data_to_export[sim].drop(
                columns="Water Vapor Mixing Ratio at 2m"
            )

            # Add metadata columns needed by EPW header writer.
            # The new-core pipeline squeezes these dimensions, so they must be
            # re-attached before export.
            if self.warming_level is not UNSET:
                xmy_data_to_export[sim]["warming_level"] = self.warming_level
            else:
                xmy_data_to_export[sim]["scenario"] = "historical+ssp370"

        self.xmy_data_to_export = xmy_data_to_export
        self._vprint("shock XMY analysis complete.")

    def export_xmy_data(self, extension: str = "epw"):
        """Write XMY data to EPW file.

        Parameters
        ----------
        extension: str
            Desired file extension ('xmy','epw', or 'csv')

        """
        print("Exporting shock XMY to file.")
        for sim, _ in self.xmy_data_to_export.items():
            # Get right year range
            if self.warming_level is UNSET:
                years = (self.start_year, self.end_year)
                clean_sim = sim
            else:
                centered_year = int(self._sim_centered_years[sim])
                year1 = centered_year - 15
                year2 = centered_year + 14
                years = (year1, year2)
                # Append warming level descriptor to simulation name.
                # Legacy sim names no longer contain "_historical+ssp370"
                # after the new-core migration, so we append directly.
                wl_label = match_str_to_wl(self.warming_level)
                clean_sim = f"{sim}_{wl_label}"
            # Attach centered_year so CSV export can include it
            if self.warming_level is not UNSET:
                self.xmy_data_to_export[sim]["centered_year"] = centered_year
            clean_stn_name = (
                self.stn_name.replace(" ", "_").replace("(", "").replace(")", "")
            )
            filename = f"{self.extreme}_shock_xmy_{clean_stn_name}_{clean_sim}".lower()
            write_tmy_file(
                filename,
                self.xmy_data_to_export[sim],
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
        without running the entire shock XMY workflow.
        """
        self._vprint(f"Getting top months for {self.extreme} shock XMY.")
        self.set_cdf_climatology()
        self.set_cdf_monthly()
        self.set_top_months()

    def generate_xmy(self):
        """Run the whole XMY workflow."""
        # This runs the whole workflow at once
        print("Running shock XMY workflow.")
        self.load_all_variables()
        self.get_candidate_months()
        self.run_xmy_analysis()
        self.export_xmy_data()
