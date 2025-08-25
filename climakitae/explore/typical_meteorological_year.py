"""Working version of TMY class."""

from climakitae.core.constants import UNSET
from climakitae.util.utils import (
    convert_to_local_time,
    get_closest_gridcell,
)
from climakitae.core.data_export import write_tmy_file
from climakitae.core.data_interface import get_data
import climakitaegui as ckg  # Need for hvplot to work

import panel

import pandas as pd
import xarray as xr
import numpy as np
import pkg_resources
from tqdm.auto import tqdm  # Progress bar


def compute_cdf(da: xr.DataArray) -> xr.DataArray:
    """Compute the cumulative density function for an input DataArray."""
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


def get_cdf_by_sim(da: xr.DataArray) -> xr.DataArray:
    """Function to help get_cdf."""
    # Group the DataArray by simulation
    return da.groupby("simulation").apply(compute_cdf)


def get_cdf_by_mon_and_sim(da: xr.DataArray) -> xr.DataArray:
    """Function to help get_cdf."""
    # Group the DataArray by month in the year
    return da.groupby("time.month").apply(get_cdf_by_sim)


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
    return ds.apply(get_cdf_by_mon_and_sim)


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
        return da.groupby("time.year").apply(get_cdf_by_mon_and_sim)

    return ds.apply(get_cdf_mon_yr)


def fs_statistic(cdf_climatology: xr.Dataset, cdf_monthly: xr.DataArray) -> xr.Dataset:
    """
    Calculates the Finkelstein-Schafer statistic:
    Absolute difference between long-term climatology and candidate CDF, divided by number of days in month

    Parameters
    -----------
    cdf_climatology: xr.Dataset
        Climatological CDF
    cdf_monthly: xr.Dataset
        Monthly CDF

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


def plot_one_var_cdf(cdf_da: xr.Dataset, var: str) -> panel.layout.base.Column:
    """Plot CDF for a single variable
    Written to function for the unique configuration of the CDF DataArray object
    Silences an annoying hvplot warning
    Will show every simulation together on the plot

    Parameters
    -----------
    cdf: xr.DataArray
       Cumulative density function for a single variable

    Returns
    -------
    panel.layout.base.Column
        Hvplot lineplot

    """
    cdf_da = cdf_da[var]
    prob_da = cdf_da.sel(data="probability", drop=True).rename(
        "probability"
    )  # Grab only probability da
    bins_da = cdf_da.sel(data="bins", drop=True).rename("bins")  # Grab just bin values
    ds = xr.merge([prob_da, bins_da])  # Merge the two to form a single Dataset object
    cdf_pl = ds.hvplot(
        "bins",
        "probability",
        by="simulation",  # Simulations should all be displayed together
        widget_location="bottom",
        grid=True,
        xlabel="{0} ({1})".format(var, cdf_da.attrs["units"]),
        xlim=(
            bins_da.min().item(),
            bins_da.max().item(),
        ),  # Fix the x-limits for all months
        ylabel="Probability (0-1)",
    )
    return cdf_pl


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
    data_models: list[str]
        List of included simulations
    scenario: list[str]
        List of scenarios
    verbose: bool
        True to increase verbosity
    cdf_climatology: xr.Dataset
        CDF climatology data
    cdf_monthly: xr.Dataset
        CDF monthly data by model
    weighted_fs_sum: xr.Dataset
        Weighted F-S statistic results
    top_df: pd.DataFrame
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
        verbose: bool = False,
    ):
        # Set variables
        if station_name is not UNSET:
            print(f"Initializing TMY object for {station_name}")
            self._set_loc_from_stn_name(station_name)
        elif (latitude is not UNSET) and (longitude is not UNSET):
            print(f"Initializing TMY object for {latitude},{longitude}")
            self._set_loc_from_lat_lon(latitude, longitude)
        else:
            print("Please provide valid station name or latitude and longitude")
            # TODO: raise error for missing input
        self.start_year = start_year
        self.end_year = end_year
        # These 4 models have the solar variables needed
        self.data_models = [
            "WRF_EC-Earth3_r1i1p1f1",
            "WRF_MPI-ESM1-2-HR_r3i1p1f1",
            "WRF_TaiESM1_r1i1p1f1",
            "WRF_MIROC6_r1i1p1f1",
        ]
        self.scenario = ["Historical Climate", "SSP 3-7.0"]
        self.verbose = verbose
        self.cdf_climatology = UNSET
        self.cdf_monthly = UNSET
        self.weighted_fs_sum = UNSET
        self.top_df = UNSET
        self.all_vars = UNSET
        self.tmy_data_to_export = UNSET

    def _set_loc_from_stn_name(self, station_name: str):
        """Get coordinates and other station metadata from station"""
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
        except Exception as e:
            # TODO: raise error correctly
            print("Could  not find station in hadisd_stations.csv")
            print("Please provide valid station name")
            raise (e)
        self.stn_lat = one_station.LAT_Y.item()
        self.stn_lon = one_station.LON_X.item()
        self.stn_state = one_station.state.item()
        self._set_lat_lon()
        return

    def _set_loc_from_lat_lon(self, latitude, longitude):
        self.stn_lat = latitude
        self.stn_lon = longitude
        self._set_lat_lon()
        # TODO: set stn_name, stn_code, stn_state
        self.stn_name = "None"
        self.stn_code = "None"
        self.stn_state = "None"

    def _set_lat_lon(self):
        # TODO: check if this buffer is generalizable
        self.latitude = (self.stn_lat - 0.05, self.stn_lat + 0.05)
        self.longitude = (self.stn_lon - 0.06, self.stn_lon + 0.06)

    def _vprint(self, msg):
        """Checks verbosity and prints as allowed."""
        if self.verbose:
            print(msg)

    def generate_tmy(self):
        """Run the whole TMY workflow."""
        # This runs the whole workflow at once
        print("Running TMY workflow. Expected overall runtime: 40 minutes")
        self.load_all_variables()
        self.get_candidate_months()
        self.run_tmy_analysis()
        self.export_tmy_data_epw()
        return

    def get_tmy_variable(self, varname, units, stats):
        """Fetch a single variable, resample and reduce."""
        if self.end_year == 2100:
            print(
                "End year is 2100. The final day in timeseries may be incomplete after data is converted to local time."
            )
            new_end_year = self.end_year
        else:
            new_end_year = self.end_year + 1

        data = get_data(
            variable=varname,
            resolution="9 km",
            timescale="hourly",
            data_type="Gridded",
            units=units,
            latitude=self.latitude,
            longitude=self.longitude,
            area_average="Yes",
            scenario=self.scenario,
            time_slice=(self.start_year, new_end_year),
        )

        data = convert_to_local_time(
            data, self.stn_lon, self.stn_lat
        )  # convert to local timezone, provide lon/lat because area average data lacks coordinates
        data = data.sel(
            {"time": slice(f"{self.start_year}-01-01", f"{self.end_year}-12-31")}
        )
        data = data.sel(simulation=self.data_models)

        returned_data = []

        if "max" in stats:
            # max air temp
            max_data = data.resample(time="1D").max()
            returned_data.append(max_data)

        if "min" in stats:
            # min air temp
            min_data = data.resample(time="1D").min()
            returned_data.append(min_data)

        if "mean" in stats:
            # mean air temp
            mean_data = data.resample(time="1D").mean()
            returned_data.append(mean_data)

        if "sum" in stats:
            sum_data = data.resample(time="1D").sum()
            returned_data.append(sum_data)

        return returned_data

    def load_all_variables(self):
        """Load the datasets needed to create TMY."""
        self._vprint("Loading variables. Expected runtime: 7 minutes")

        print("Getting air temperature", end="... ")
        airtemp_data = self.get_tmy_variable(
            "Air Temperature at 2m", "degC", ["max", "min", "mean"]
        )
        # unpack and rename
        max_airtemp_data = airtemp_data[0]
        max_airtemp_data.name = "Daily max air temperature"
        min_airtemp_data = airtemp_data[1]
        min_airtemp_data.name = "Daily min air temperature"
        mean_airtemp_data = airtemp_data[2]
        mean_airtemp_data.name = "Daily mean air temperature"

        print("Getting dew point temperature", end="... ")
        # dew point temperature
        dewpt_data = self.get_tmy_variable(
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
        print("Getting wind speed", end="... ")
        wndspd_data = self.get_tmy_variable(
            "Wind speed at 10m", "m s-1", ["max", "mean"]
        )
        # unpack and rename
        max_windspd_data = wndspd_data[0]
        max_windspd_data.name = "Daily max wind speed"
        mean_windspd_data = wndspd_data[1]
        mean_windspd_data.name = "Daily mean wind speed"

        # global irradiance
        print("Getting global irradiance", end="... ")
        total_ghi_data = self.get_tmy_variable(
            "Instantaneous downwelling shortwave flux at bottom", "W/m2", ["sum"]
        )
        total_ghi_data = total_ghi_data[0]
        total_ghi_data.name = "Global horizontal irradiance"

        # direct normal irradiance
        print("Getting direct normal irradiance", end="... ")
        total_dni_data = self.get_tmy_variable(
            "Shortwave surface downward direct normal irradiance", "W/m2", ["sum"]
        )
        total_dni_data = total_dni_data[0]
        total_dni_data.name = "Direct normal irradiance"

        print("Loading all variables into memory.")
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
        print("All TMY variables loaded.")

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
        self.cdf_monthly = self.cdf_monthly.where(
            (~self.cdf_monthly.year.isin([1991, 1992, 1993, 1994])), np.nan, drop=True
        )
        return

    def set_weighted_statistic(self):
        """Calculate the weighted f-s statistic."""
        if self.cdf_climatology is UNSET:
            self.set_cdf_climatology()
        if self.cdf_monthly is UNSET:
            self.set_cdf_monthly()
        self._vprint("Calculating weighted FS statistic.")
        all_vars_fs = fs_statistic(self.cdf_climatology, self.cdf_monthly)
        weighted_fs = compute_weighted_fs(all_vars_fs)

        # Sum
        self.weighted_fs_sum = (
            weighted_fs.to_array().sum(dim=["variable", "bin_number"]).drop(["data"])
        )
        return

    def set_top_df(self):
        """Calculate top months dataframe."""
        # Pass the weighted F-S sum data for simplicity
        if self.weighted_fs_sum is UNSET:
            self.set_weighted_statistic()
        ds = self.weighted_fs_sum

        df_list = []
        num_values = (
            1  # Selecting the top value for now, persistence statistics calls for top 5
        )
        for sim in ds.simulation.values:
            for mon in ds.month.values:
                da_i = ds.sel(month=mon, simulation=sim)
                top_xr = da_i.sortby(da_i, ascending=True)[:num_values].expand_dims(
                    ["month", "simulation"]
                )
                top_df_i = top_xr.to_dataframe(name="top_values")
                df_list.append(top_df_i)

        # Concatenate list together for all months and simulations
        self.top_df = pd.concat(df_list).drop(columns=["top_values"]).reset_index()
        self._vprint("Top months:")
        self._vprint(self.top_df)
        return

    def get_candidate_months(self):
        """Run CDF functions to get top candidates."""
        self._vprint(
            "Getting top months for TMY. Expected runtime with loaded data: 1 min"
        )
        self.set_cdf_climatology()
        self.set_cdf_monthly()
        self.set_weighted_statistic()
        self.set_top_df()
        print("Done.")
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
            y=[
                "Air Temperature at 2m",
                "Dew point temperature",
                "Relative humidity",
                "Instantaneous downwelling shortwave flux at bottom",
                "Shortwave surface downward direct normal irradiance",
                "Shortwave surface downward diffuse irradiance",
                "Instantaneous downwelling longwave flux at bottom",
                "Wind speed at 10m",
                "Wind direction at 10m",
                "Surface Pressure",
            ],
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
        self._vprint("Generating TMY data to export. Expected runtime: 30 minutes")

        ## ================== GET DATA FROM CATALOG ==================
        vars_and_units = {
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

        if self.end_year == 2100:
            new_end_year = 2100
        else:
            new_end_year = self.end_year + 1

        # Loop through each variable and grab data from catalog
        all_vars_list = []
        print("STEP 1: RETRIEVING HOURLY DATA FROM CATALOG\n")
        for var, units in vars_and_units.items():
            print(f"Retrieving data for {var}", end="... ")
            data_by_var = get_data(
                variable=var,
                resolution="9 km",
                timescale="hourly",
                data_type="Gridded",
                units=units,
                latitude=self.latitude,
                longitude=self.longitude,
                area_average="No",
                scenario=self.scenario,
                time_slice=(self.start_year, self.end_year + 1),
            )
            data_by_var = convert_to_local_time(
                data_by_var
            )  # convert to local timezone.
            data_by_var = data_by_var.sel(
                {"time": slice(f"{self.start_year}-01-01", f"{new_end_year}-12-31")}
            )  # get desired time slice in local time
            data_by_var = get_closest_gridcell(
                data_by_var, self.stn_lat, self.stn_lon, print_coords=False
            )  # retrieve only closest gridcell
            data_by_var = data_by_var.sel(
                simulation=self.data_models
            )  # Subset for only the models that have solar variables

            # Drop unwanted coords
            data_by_var = data_by_var.squeeze().drop(
                ["lakemask", "landmask", "x", "y", "Lambert_Conformal"]
            )

            all_vars_list.append(data_by_var)  # Append to list
            print("complete!")

        # Merge data from all variables into a single xr.Dataset object
        all_vars_ds = xr.merge(all_vars_list)

        ## ================== CONSTRUCT TMY ==================
        print(
            "\nSTEP 2: CALCULATING TYPICAL METEOROLOGICAL YEAR PER MODEL SIMULATION\nProgress bar shows code looping through each month in the year.\n"
        )
        tmy_df_all = {}
        for sim in all_vars_ds.simulation.values:
            df_list = []
            print(f"Calculating TMY for simulation: {sim}")
            for mon in tqdm(np.arange(1, 13, 1)):
                # Get year corresponding to month and simulation combo
                year = self.top_df.loc[
                    (self.top_df["month"] == mon) & (self.top_df["simulation"] == sim)
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

        self.tmy_data_to_export = tmy_df_all  # Return dict of TMY by simulation

    def export_tmy_data(self, extension: str = "tmy"):
        """Write TMY data to EPW file.

        Parameters
        ----------
        extension: str
            Desired file extension ('tmy' or 'epw')

        """
        self._vprint("Exporting TMY to file.")
        for sim, tmy in self.tmy_data_to_export.items():
            filename = "TMY_{0}_{1}".format(
                stn_name.replace(" ", "_").replace("(", "").replace(")", ""), sim
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
            if self.verbose:
                print("  Wrote", filename)
        return
