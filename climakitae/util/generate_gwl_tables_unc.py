"""
Util for generating warming level reference data file
"gwl_1981-2010ref_EC-Earth3_ssp370.csv" in ../data

The CSV file is generated for use in ../explore/uncertainty.py. It contains,
for each ensemble member of EC-Earth3, the times when five major warming levels
are reached under SSP3-7.0.

To run, type: <<python generate_gwl_tables_unc.py>> in the command line and wait
for printed model outputs showing progress.
"""

import concurrent.futures

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from climakitae.util.utils import write_csv_file

global test
test = False


def make_weighted_timeseries(temp: xr.DataArray) -> xr.DataArray:
    """
    Creates a spatially-weighted single-dimension time series of global temperature.

    The function weights the latitude grids by size and averages across all longitudes,
    resulting in a single time series object.

    Parameters:
    ----------
    temp : xarray.DataArray
        An xarray DataArray of global temperature with latitude and longitude coordinates.

    Returns:
    -------
    xarray.DataArray
        A time series of global temperature that is spatially weighted across latitudes and averaged
        across all longitudes.

    Raises:
    -------
    ValueError
        If the DataArray doesn't contain recognizable latitude/longitude coordinates.
    """
    # Find variable names for latitude and longitude to make code more readable
    lat_candidates = ["lat", "latitude"]
    lon_candidates = ["lon", "longitude"]

    # Try to find latitude coordinate
    lat = None
    for lat_name in lat_candidates:
        if lat_name in temp.coords:
            lat = lat_name
            break

    # Try to find longitude coordinate
    lon = None
    for lon_name in lon_candidates:
        if lon_name in temp.coords:
            lon = lon_name
            break

    # Check if both coordinates were found
    if lat is None or lon is None:
        raise ValueError(
            "Input DataArray must have latitude and longitude coordinates."
        )

    # Weight latitude grids by size, then average across all longitudes to create single time-series object
    weightlat = np.sqrt(np.cos(np.deg2rad(temp[lat])))
    weightlat = weightlat / np.sum(weightlat)
    timeseries = (temp * weightlat).sum(lat).mean(lon)
    return timeseries


class GWLGenerator:
    """
    Class for generating Global Warming Level (GWL) reference data.
    Encapsulates the parameters and methods needed for GWL calculations.

    Attributes
    ----------
    df : pandas.DataFrame
        DataFrame containing metadata for CMIP6 simulations
    sims_on_aws : pandas.DataFrame
        DataFrame listing available simulations on AWS
    fs : s3fs.S3FileSystem
        S3 file system object for accessing AWS data

    Methods
    -------
    get_sims_on_aws() -> pandas.DataFrame
        Generates a DataFrame listing all relevant CMIP6 simulations available on AWS.
    build_timeseries(model_config: dict) -> xarray.Dataset
        Builds an xarray Dataset with a time dimension, containing the concatenated historical
        and SSP time series for all specified scenarios of a given model and ensemble member.
    get_gwl(smoothed: pandas.DataFrame, degree: float) -> pandas.DataFrame
        Computes the timestamp when a given GWL is first reached.
    get_gwl_table_one(model_config: dict, reference_period: dict) -> tuple[pandas.DataFrame, pandas.DataFrame]
        Generates a GWL table for a single model and ensemble member.
    get_gwl_table(model_config: dict, reference_period: dict) -> tuple[pandas.DataFrame, pandas.DataFrame]
        Generates GWL tables for a model across all its ensemble members.
    generate_gwl_file(models: list[str], scenarios: list[str], reference_periods: list[dict])
        Generates global warming level (GWL) reference files for specified models.

    Examples
    --------
    >>> df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")
    >>> gwl_generator = GWLGenerator(df)
    >>> models = ["EC-Earth3"]
    >>> scenarios = ["ssp370"]
    >>> reference_periods = [{"start_year": "19810101", "end_year": "20101231"}]
    >>> gwl_generator.generate_gwl_file(models, scenarios, reference_periods)
    """

    def __init__(self, df, sims_on_aws=None):
        """
        Initialize the GWLGenerator with the CMIP6 data catalog.

        Parameters:
        ----------
        df : pandas.DataFrame
            DataFrame containing metadata for CMIP6 simulations
        sims_on_aws : pandas.DataFrame, optional
            DataFrame listing available simulations on AWS. If None, it will be generated.
        """
        self.df = df
        self.sims_on_aws = (
            sims_on_aws if sims_on_aws is not None else self.get_sims_on_aws()
        )
        self.fs = s3fs.S3FileSystem(anon=True)

    def get_sims_on_aws(self) -> pd.DataFrame:
        """
        Generates a pandas DataFrame listing all relevant CMIP6 simulations available on AWS.

        Returns:
        -------
        pandas.DataFrame
            DataFrame indexed by model names with columns for different scenarios.
        """
        df_subset = self.df[
            (self.df.table_id == "Amon")
            & (self.df.variable_id == "tas")
            & (self.df.experiment_id == "historical")
        ]
        models = list(set(df_subset.source_id))
        models.sort()

        # First cut through the catalog
        scenarios = ["historical", "ssp585", "ssp370", "ssp245", "ssp126"]
        sims_on_aws = pd.DataFrame(index=models, columns=scenarios)

        # Fix 1: Replace chained assignment with loc indexer
        for model in models:
            for scenario in scenarios:
                df_scenario = self.df[
                    (self.df.table_id == "Amon")
                    & (self.df.variable_id == "tas")
                    & (self.df.experiment_id == scenario)
                    & (self.df.source_id == model)
                ]
                ensMembers = list(set(df_scenario.member_id))
                # Use .loc instead of chained indexing
                sims_on_aws.loc[model, scenario] = ensMembers

        # cut the table to those GCMs that have a historical + at least one SSP ensemble member
        models_to_drop = []
        for i, item in enumerate(sims_on_aws.T.columns):
            no_ssp = True
            for ssp in ["ssp585", "ssp370", "ssp245", "ssp126"]:
                if len(sims_on_aws.loc[item, ssp]) > 0:
                    no_ssp = False
            if (len(sims_on_aws.loc[item, "historical"]) < 1) or (no_ssp):
                models_to_drop.append(item)

        sims_on_aws = sims_on_aws.drop(index=models_to_drop)

        # Find the historical ensemble members for each model
        # and remove them from the SSPs
        for i, item in enumerate(sims_on_aws.T.columns):
            variants_to_keep = []
            for variant in sims_on_aws.loc[item, "historical"]:
                for ssp in ["ssp585", "ssp370", "ssp245", "ssp126"]:
                    if str(variant) in sims_on_aws.loc[item, ssp]:
                        variants_to_keep.append(variant)
            sims_on_aws.loc[item, "historical"] = list(set(variants_to_keep))

        sims_on_aws.index.name = "source_id"

        return sims_on_aws

    def build_timeseries(self, model_config: dict) -> xr.Dataset:
        """
        Builds an xarray Dataset with a time dimension, containing the concatenated historical
        and SSP time series for all specified scenarios of a given model and ensemble member.

        Parameters:
        ----------
        model_config : dict
            Dictionary containing 'variable', 'model', 'ens_mem', and 'scenarios' keys

        Returns:
        -------
        xarray.Dataset
            Dataset with time as the dimension, containing the appended historical and SSP time series.
        """
        variable = model_config["variable"]
        model = model_config["model"]
        ens_mem = model_config["ens_mem"]
        scenarios = model_config["scenarios"]

        # Get historical data first
        scenario = "historical"
        data_historical = xr.Dataset()
        df_scenario = self.df[
            (self.df.table_id == "Amon")
            & (self.df.variable_id == variable)
            & (self.df.experiment_id == scenario)
            & (self.df.source_id == model)
            & (self.df.member_id == ens_mem)
        ]
        if not df_scenario.empty:
            try:
                store_url = df_scenario.zstore.values[0]
                temp_hist = xr.open_zarr(
                    store_url, consolidated=True, storage_options={"anon": True}
                )
                # Create global weighted time-series of variable
                data_historical = make_weighted_timeseries(temp_hist[variable])
                data_historical = data_historical.sortby(
                    "time"
                )  # needed for MPI-ESM-2-HR
            except Exception as e:
                print(f"Error loading historical data: {e}")
                return xr.Dataset()

        # Now process each scenario
        data_one_model = xr.Dataset()
        for scenario in scenarios:
            # Check if the ensemble member exists for this scenario
            if (
                scenario in self.sims_on_aws.T[model]
                and ens_mem in self.sims_on_aws.T[model][scenario]
            ):
                df_scenario = self.df[
                    (self.df.table_id == "Amon")
                    & (self.df.variable_id == variable)
                    & (self.df.experiment_id == scenario)
                    & (self.df.source_id == model)
                    & (self.df.member_id == ens_mem)
                ]
                if not df_scenario.empty:
                    try:
                        store_url = df_scenario.zstore.values[0]
                        temp_hist = xr.open_zarr(
                            store_url, consolidated=True, storage_options={"anon": True}
                        )
                        temp_hist = temp_hist.isel(time=slice(0, 1032))
                        temp_hist = xr.decode_cf(temp_hist)

                        # Create global weighted time-series of variable
                        timeseries = make_weighted_timeseries(temp_hist[variable])

                        # Clean data and append to `data_one_model`
                        timeseries = timeseries.sortby(
                            "time"
                        )  # needed for MPI-ESM1-2-LR
                        data_one_model[scenario] = xr.concat(
                            [data_historical, timeseries], dim="time"
                        )
                    except Exception as e:
                        print(f"Error loading scenario {scenario} data: {e}")
        return data_one_model

    @staticmethod
    def get_gwl(smoothed: pd.DataFrame, degree: float) -> pd.DataFrame:
        """
        Computes the timestamp when a given GWL is first reached.

        Parameters:
        ----------
        smoothed : pandas.DataFrame
            DataFrame containing global mean temperature time series for multiple scenarios
        degree : float
            The global warming level to detect

        Returns:
        -------
        pandas.DataFrame
            Table with timestamps for when each scenario first crosses the specified warming level
        """

        def get_wl_timestamp(scenario: str, degree: float) -> pd.Timestamp:
            """
            Find the timestamp that first crosses the given degree.
            Return np.nan if none of the timestamps pass this degree level.
            """
            if any(scenario >= degree):
                wl_ts = scenario[scenario >= degree].index[0]
                return wl_ts
            else:
                return np.nan

        gwl = smoothed.apply(lambda scenario: get_wl_timestamp(scenario, degree))
        return gwl

    def get_gwl_table_for_single_model_and_ensemble(
        self, model_config: dict, reference_period: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates a GWL table for a single model and ensemble member.

        Parameters:
        ----------
        model_config : dict
            Dictionary containing 'variable', 'model', 'ens_mem', and 'scenarios' keys
        reference_period : dict
            Dictionary containing 'start_year' and 'end_year' keys

        Returns:
        -------
        tuple
            DataFrame containing warming levels and DataFrame with global mean temperature time series
        """
        model = model_config["model"]
        ens_mem = model_config["ens_mem"]
        start_year = reference_period["start_year"]
        end_year = reference_period["end_year"]

        data_one_model = self.build_timeseries(model_config)

        # Add an outer try-except block to catch all anomaly calculation errors
        try:
            try:
                anom = data_one_model - data_one_model.sel(
                    time=slice(start_year, end_year)
                ).mean("time")
            except:
                # some calendars won't allow a 31st of the month end date
                end_year = str(
                    (pd.to_datetime(end_year) - pd.DateOffset(days=1)).date()
                ).replace("-", "")
                anom = data_one_model - data_one_model.sel(
                    time=slice(start_year, end_year)
                ).mean("time")

            smoothed = anom.rolling(time=20 * 12, center=True).mean("time")

            ### one_model is a dataframe of times and warming levels by scenario
            one_model = (
                smoothed.to_array(dim="scenario", name=model)
                .dropna("time", how="all")
                .to_pandas()  # Dropping time slices that are NaN across all SSPs
            )
            gwlevels = pd.DataFrame()
            try:
                for level in [1.5, 2, 2.5, 3, 4]:
                    gwlevels[level] = self.get_gwl(one_model.T, level)
            except Exception as e:
                print(
                    model, ens_mem, " problems"
                )  # helps EC-Earth3 not be skipped altogether
                print(e)

            # Modifying and returning one_model to be seen as a WL table
            final_model = one_model.T
            final_model.columns = ens_mem + "_" + final_model.columns
            return gwlevels, final_model

        # Handle all anomaly calculation failures with a proper error message and return empty DataFrames
        except Exception as e:
            print(f"Error calculating anomalies for {model}, {ens_mem}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_gwl_table(
        self, model_config: dict, reference_period: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates GWL tables for a model across all its ensemble members.

        Parameters:
        ----------
        model_config : dict
            Dictionary containing 'variable', 'model', and 'scenarios' keys
        reference_period : dict
            Dictionary containing 'start_year' and 'end_year' keys

        Returns:
        -------
        tuple
            DataFrame containing warming levels and DataFrame with global mean temperature time series
        """
        global test
        model = model_config["model"]
        scenarios = model_config["scenarios"]

        ens_mem_list = (
            self.sims_on_aws.T[model]["ssp370"].copy() if "ssp370" in scenarios else []
        )
        # "historical" gives same list as "ssp370"
        print(f"Number of ensemble members: {len(ens_mem_list)}")
        if test:
            ens_mem_list = ens_mem_list[:10]

        # Combining all the ensemble members for a given model
        gwlevels_tbl, wl_data_tbls = [], []
        num_ens_mems = len(ens_mem_list)

        def _process_ensemble_member(i_ens_mem: tuple[int, str]) -> tuple:
            i, ens_mem = i_ens_mem
            print(
                f"Getting gwl table for ensemble member {ens_mem} ({i+1}/{num_ens_mems})..."
            )
            try:
                member_config = model_config.copy()
                member_config["ens_mem"] = ens_mem

                gwlevels, wl_data_tbl = (
                    self.get_gwl_table_for_single_model_and_ensemble(
                        member_config, reference_period
                    )
                )
                return (i, ens_mem, gwlevels, wl_data_tbl, None)
            except Exception as e:
                error_msg = f"Cannot get gwl table for {i}th ensemble member {ens_mem}: {str(e)}"
                return (i, ens_mem, None, None, error_msg)

        # Create enumerated list for processing
        enum_ens_mem_list = list(enumerate(ens_mem_list))

        results = [
            _process_ensemble_member(i_ens_mem) for i_ens_mem in enum_ens_mem_list
        ]

        # Process results
        successful_ens_mems = []  # Track successful ensemble members
        for i, ens_mem, gwlevels, wl_data_tbl, error_msg in results:
            print(i, ens_mem, gwlevels, wl_data_tbl)
            if error_msg:
                print(error_msg)
            else:
                gwlevels_tbl.append(gwlevels)
                wl_data_tbls.append(wl_data_tbl)
                successful_ens_mems.append(ens_mem)  # Append only if successful

        if gwlevels_tbl and wl_data_tbls:
            # Renaming columns of all ensemble members within model
            try:
                # Align indexes before concatenation
                wl_data_tbls = [
                    df.reindex(wl_data_tbls[0].index) for df in wl_data_tbls
                ]
                wl_data_tbl_sim = pd.concat(wl_data_tbls, axis=1)
            except Exception as e:
                print(f"Error concatenating timeseries results for model {model}: {e}")
                return pd.DataFrame(), pd.DataFrame()

            print(model, wl_data_tbl_sim.columns)
            wl_data_tbl_sim.columns = model + "_" + wl_data_tbl_sim.columns

            # Use the filtered list for concatenation
            try:
                gwlevels_tbl = [
                    df.reindex(gwlevels_tbl[0].index) for df in gwlevels_tbl
                ]
                return (
                    pd.concat(gwlevels_tbl, keys=successful_ens_mems),
                    wl_data_tbl_sim,
                )
            except Exception as e:
                print(
                    f"Error concatenating warming level results for model {model}: {e}"
                )
                return pd.DataFrame(), pd.DataFrame()
        else:
            print(f"No valid ensemble members for model {model}")
            return pd.DataFrame(), pd.DataFrame()

    def generate_gwl_file(
        self, models: list[str], scenarios: list[str], reference_periods: list[dict]
    ):
        """
        Generates global warming level (GWL) reference files for specified models.

        Parameters:
        ----------
        models : list
            List of model names to process
        scenarios : list
            List of scenario names to include
        reference_periods : list
            List of dictionaries with 'start_year' and 'end_year' keys
        """
        variable = "tas"

        for period in reference_periods:
            start_year = period["start_year"]
            end_year = period["end_year"]

            print("Generate all WL table {}-{}".format(start_year[:4], end_year[:4]))
            all_wl_data_tbls = pd.DataFrame()
            all_gw_tbls, all_gw_data_tbls = [], []

            # Extracts GWL information for each model
            for i, model in enumerate(models):
                print(f"\n...Model {i} {model}...\n")

                model_config = {
                    "variable": variable,
                    "model": model,
                    "scenarios": scenarios,
                }

                gw_tbl, wl_data_tbl_sim = self.get_gwl_table(model_config, period)

                if not gw_tbl.empty:
                    all_gw_tbls.append(gw_tbl)
                    all_gw_data_tbls.append(wl_data_tbl_sim)

                    try:
                        all_wl_data_tbls = pd.concat(
                            [all_wl_data_tbls, wl_data_tbl_sim], axis=1
                        )
                    except ValueError as e:  # Change Exception to ValueError
                        print(
                            f"\n Model {model} is skipped. Its table cannot be concatenated as its datetime indices are different: \n"
                        )
                        print(e)

            # Creating WL lookup table if results were found
            if all_gw_tbls:
                [print(x.head()) for x in all_gw_tbls]
                all_gw_levels = pd.concat(all_gw_tbls, keys=models)
                all_gw_levels.index = pd.MultiIndex.from_tuples(
                    all_gw_levels.index, names=["GCM", "run", "scenario"]
                )
                try:
                    print(all_gw_levels.head())
                    success = write_csv_file(
                        all_gw_levels,
                        "data/gwl_{}-{}ref_EC-Earth3_ssp370.csv".format(
                            start_year[:4], end_year[:4]
                        ),
                    )
                    print(
                        f"Successfully wrote warming level file for {start_year[:4]}-{end_year[:4]}"
                    )
                except Exception as e:
                    print(
                        f"Error writing GWL file for {start_year[:4]}-{end_year[:4]}: {e}"
                    )

            else:
                print(
                    f"No warming level data was generated for {start_year[:4]}-{end_year[:4]}"
                )


def main(_kTest=False):
    """
    Main function to run the GWL generator for EC-Earth3 model.
    """
    global test
    test = _kTest
    try:
        print("Loading CMIP6 catalog...")
        df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")

        print("Initializing GWL generator...")
        try:
            gwl_generator = GWLGenerator(df)

            # Pre-defined configuration
            models = ["EC-Earth3"]  # Just one model
            scenarios = ["ssp370"]  # Just one scenario
            reference_periods = [{"start_year": "19810101", "end_year": "20101231"}]

            print(f"Generating GWL file for {models}, {scenarios}...")
            try:
                gwl_generator.generate_gwl_file(models, scenarios, reference_periods)
                print("GWL file generation complete.")
            except Exception as e:
                print(f"Error generating GWL file: {e}")
        except Exception as e:
            print(f"Error initializing GWL generator: {e}")
    except Exception as e:
        print(f"Error loading CMIP6 catalog: {e}")


if __name__ == "__main__":
    import sys

    # Check if --test passed
    if "--test" in sys.argv:
        print("Running in test mode...")
        main(_kTest=True)
    else:
        main()
#
