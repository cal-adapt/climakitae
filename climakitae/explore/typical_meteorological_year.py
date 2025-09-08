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
from tqdm.auto import tqdm  # Progress bar

from climakitae.core.constants import UNSET
from climakitae.core.data_export import write_tmy_file
from climakitae.core.data_interface import get_data
from climakitae.util.utils import convert_to_local_time, get_closest_gridcell


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


def fs_statistic(cdf_climatology: xr.Dataset, cdf_monthly: xr.DataArray) -> xr.Dataset:
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
    weights_per_var = {
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

    for var, weight in weights_per_var.items():
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


def get_top_months(da_fs):
    """Return dataframe of top months by simulation.

    Parameters
    ----------
    da_fs: xr.Dataset
       Summed weighted f-s statistic

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
            top_df_i = top_xr.to_dataframe(name="top_values")
            df_list.append(top_df_i)

    # Concatenate list together for all months and simulations
    return pd.concat(df_list).drop(columns=["top_values"]).reset_index()


class TMY:
    """Encapsulate the code needed to generate Typical Meteorological Year (TMY) files.

    Uses WRF hourly data to produce TMYs. User provides the start and end years along
    with station location to generate file.


    Parameters
    ----------
    start_year : str
        Initial year of TMY period
    end_year : str
        Final year of TMY period
    station_name: str (optional)
        Long name of desired station
    latitude : float (optional)
        Latitude for TMY data if station_name not set
    longitude : float (optional)
        Longitude for TMY data if station_name not set
    verbose: bool
        True to increase verbosity

    Attributes
    ----------
    start_year: str
        Initial year of TMY period
    end_year: str
        Final year of TMY period
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

    """

    def __init__(
        self,
        start_year: int,
        end_year: int,
        station_name: str = UNSET,
        latitude: float = UNSET,
        longitude: float = UNSET,
        verbose: bool = True,
    ):
        # Set variables
        if station_name is not UNSET:
            print(f"Initializing TMY object for {station_name}")
            self._set_loc_from_stn_name(station_name)
        elif (latitude is not UNSET) and (longitude is not UNSET):
            print(f"Initializing TMY object for {latitude},{longitude}")
            self._set_loc_from_lat_lon(latitude, longitude)
        else:
            raise ValueError(
                "No valid station name or latitude and longitude provided."
            )
        self.start_year = start_year
        self.end_year = end_year
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
        self._set_lat_lon()
        return

    def _set_loc_from_lat_lon(self, latitude: float, longitude: float):
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
        self._set_lat_lon()
        # Set station variables to string "None"
        # to be written to file with final data
        self.stn_name = "None"
        self.stn_code = "None"
        self.stn_state = "None"
        return

    def _set_lat_lon(self):
        """Set the lat/lon ranges for selecting data around the grid point
        in call to get_data()."""
        self.lat_range = (self.stn_lat - 0.1, self.stn_lat + 0.1)
        self.lon_range = (self.stn_lon - 0.1, self.stn_lon + 0.1)
        return

    def _vprint(self, msg: str):
        """Checks verbosity and prints as allowed."""
        if self.verbose:
            print(msg)
        return

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
        data = data.sel(simulation=self.simulations)
        return data

    def _get_tmy_variable(
        self, varname: str, units: str, stats: list[str]
    ) -> xr.Dataset:
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

        if "max" in stats:
            # max air temp
            max_data = data.resample(time="1D").max()
            max_data.attrs["frequency"] = "daily"
            returned_data.append(max_data)

        if "min" in stats:
            # min air temp
            min_data = data.resample(time="1D").min()
            min_data.attrs["frequency"] = "daily"
            returned_data.append(min_data)

        if "mean" in stats:
            # mean air temp
            mean_data = data.resample(time="1D").mean()
            mean_data.attrs["frequency"] = "daily"
            returned_data.append(mean_data)

        if "sum" in stats:
            sum_data = data.resample(time="1D").sum()
            sum_data.attrs["frequency"] = "daily"
            returned_data.append(sum_data)

        return returned_data

    @staticmethod
    def _make_8760_tables(
        all_vars_ds: xr.Dataset, top_months: pd.DataFrame
    ) -> pd.DataFrame:
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

        print("  Getting air temperature", end="... ")
        airtemp_data = self._get_tmy_variable(
            "Air Temperature at 2m", "degC", ["max", "min", "mean"]
        )

        # unpack and rename
        max_airtemp_data = airtemp_data[0]
        max_airtemp_data.name = "Daily max air temperature"
        min_airtemp_data = airtemp_data[1]
        min_airtemp_data.name = "Daily min air temperature"
        mean_airtemp_data = airtemp_data[2]
        mean_airtemp_data.name = "Daily mean air temperature"

        print("  Getting dew point temperature", end="... ")
        # dew point temperature
        dewpt_data = self._get_tmy_variable(
            "Dew point temperature", "degC", ["max", "min", "mean"]
        )
        # unpack and rename
        max_dewpt_data = dewpt_data[0]
        max_dewpt_data.name = "Daily max dewpoint temperature"
        min_dewpt_data = dewpt_data[1]
        min_dewpt_data.name = "Daily min dewpoint temperature"
        mean_dewpt_data = dewpt_data[2]
        mean_dewpt_data.name = "Daily mean dewpoint temperature"

        # wind speed
        print("  Getting wind speed", end="... ")
        wndspd_data = self._get_tmy_variable(
            "Wind speed at 10m", "m s-1", ["max", "mean"]
        )
        # unpack and rename
        max_windspd_data = wndspd_data[0]
        max_windspd_data.name = "Daily max wind speed"
        mean_windspd_data = wndspd_data[1]
        mean_windspd_data.name = "Daily mean wind speed"

        # global irradiance
        print("  Getting global irradiance", end="... ")
        total_ghi_data = self._get_tmy_variable(
            "Instantaneous downwelling shortwave flux at bottom", "W/m2", ["sum"]
        )
        total_ghi_data = total_ghi_data[0]
        total_ghi_data.name = "Global horizontal irradiance"

        # direct normal irradiance
        print("  Getting direct normal irradiance", end="... ")
        total_dni_data = self._get_tmy_variable(
            "Shortwave surface downward direct normal irradiance", "W/m2", ["sum"]
        )
        total_dni_data = total_dni_data[0]
        total_dni_data.name = "Direct normal irradiance"

        self._vprint("  Loading all variables into memory.")
        all_vars = xr.merge(
            [
                max_airtemp_data.squeeze(),
                min_airtemp_data.squeeze(),
                mean_airtemp_data.squeeze(),
                max_dewpt_data.squeeze(),
                min_dewpt_data.squeeze(),
                mean_dewpt_data.squeeze(),
                max_windspd_data.squeeze(),
                mean_windspd_data.squeeze(),
                total_ghi_data.squeeze(),
                total_dni_data.squeeze(),
            ]
        )

        # load all indices in
        self.all_vars = all_vars.compute()
        self._vprint("  All TMY variables loaded.")
        return

    def set_cdf_climatology(self):
        """Calculate the long-term climatology for each index for each month so
        we can establish the baseline pattern.
        """
        if self.all_vars is UNSET:
            self.load_all_variables()
        self._vprint("Calculating CDF climatology.")
        self.cdf_climatology = get_cdf(self.all_vars)
        return

    def set_cdf_monthly(self):
        """Get CDF for each month and variable."""
        if self.all_vars is UNSET:
            self.load_all_variables()
        self._vprint("Calculating monthly CDF.")
        self.cdf_monthly = get_cdf_monthly(self.all_vars)
        # Remove the years for the Pinatubo eruption
        self.cdf_monthly = remove_pinatubo_years(self.cdf_monthly)
        return

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
        return

    def set_top_months(self):
        """Calculate top months dataframe."""
        # Pass the weighted F-S sum data for simplicity
        if self.weighted_fs_sum is UNSET:
            self.set_weighted_statistic()
        self._vprint("Finding top months (lowest F-S statistic)")
        self.top_months = get_top_months(self.weighted_fs_sum)
        return

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

        self.tmy_data_to_export[simulation].plot(
            x="time",
            y=list(self.vars_and_units.keys()),
            title=f"Typical Meteorological Year ({simulation})",
            subplots=True,
            figsize=(10, 8),
            legend=True,
        )
        return

    def run_tmy_analysis(self):
        """Generate typical meteorological year data
        Output will be a list of dataframes per simulation.
        Print statements throughout the function indicate to the user the progress of the computatioconvert_to_local_time

        Parameters
        -----------
        top_df: pd.DataFrame
            Table with column values month, simulation, and year
            Each month-sim-yr combo represents the top candidate that has the lowest weighted sum from the FS statistic

        Returns
        --------
        dict of str: pd.DataFrame
            Dictionary in the format of {simulation:TMY corresponding to that simulation}

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
            "\n  STEP 2: Calculating Typical Meteorological Year per model simulation\nProgress bar shows code looping through each month in the year.\n"
        )

        self.tmy_data_to_export = self._make_8760_tables(
            all_vars_ds, self.top_months
        )  # Return dict of TMY by simulation
        self._vprint("TMY analysis complete")
        return

    def export_tmy_data(self, extension: str = "epw"):
        """Write TMY data to EPW file.

        Parameters
        ----------
        extension: str
            Desired file extension ('tmy' or 'epw')

        """
        print("Exporting TMY to file.")
        for sim, tmy in self.tmy_data_to_export.items():
            filename = "TMY_{0}_{1}".format(
                self.stn_name.replace(" ", "_").replace("(", "").replace(")", ""), sim
            ).lower()
            write_tmy_file(
                filename,
                self.tmy_data_to_export[sim],
                (self.start_year, self.end_year),
                self.stn_name,
                self.stn_code,
                self.stn_lat,
                self.stn_lon,
                self.stn_state,
                file_ext=extension,
            )
        return

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
        return

    def generate_tmy(self):
        """Run the whole TMY workflow."""
        # This runs the whole workflow at once
        print("Running TMY workflow. Expected overall runtime: 40 minutes")
        self.load_all_variables()
        self.get_candidate_months()
        self.run_tmy_analysis()
        self.export_tmy_data()
        return
