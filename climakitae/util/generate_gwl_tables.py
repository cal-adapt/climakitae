"""Util for generating warming level reference data in ../data/ ###

To run, type: <<python generate_gwl_tables.py>> in the command line and wait for printed model outputs showing progress.

"""

from typing import Any

import cftime
import intake
import intake_esm
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from climakitae.core.constants import WARMING_LEVELS
from climakitae.util.utils import write_csv_file

global test
test = False


def make_weighted_timeseries(temp: xr.DataArray) -> xr.DataArray:
    """Creates a spatially-weighted single-dimension time series of global temperature.

    The function weights the latitude grids by size and averages across all longitudes,
    resulting in a single time series object.

    Parameters
    ----------
    temp : xarray.DataArray
        An xarray DataArray of global temperature with latitude and longitude coordinates.

    Returns
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
    """Class for generating Global Warming Level (GWL) reference data.
    Encapsulates the parameters and methods needed for GWL calculations.

    Attributes
    ----------
    df : pandas.DataFrame
        DataFrame containing metadata for CMIP6 simulations
    sims_on_aws : pandas.DataFrame
        DataFrame listing available simulations on AWS
    fs : s3fs.S3FileSystem
        S3 file system object for accessing AWS data
    ens_mem_cesm : dict
        List of realizations for CESM2
    cesm2_lens : xr.Dataset
        CESM2 LENS data loaded from catalog

    Methods
    -------
    set_cesm2_lens() -> xr.Dataset()
        Pull subset of CESM2 model data.
    get_sims_on_aws() -> pandas.DataFrame
        Generates a DataFrame listing all relevant CMIP6 simulations available on AWS.
    build_timeseries(model_config: dict) -> xarray.Dataset
        Builds an xarray Dataset with a time dimension, containing the concatenated historical
        and SSP time series for all specified scenarios of a given model and ensemble member.
    buildDFtimeSeries_cesm2(model_config: dict) -> xarray.Dataset
    get_gwl(smoothed: pandas.DataFrame, degree: float) -> pandas.DataFrame
        Computes the timestamp when a given GWL is first reached.
    get_gwl_table_for_single_model_and_ensemble(model_config: dict, reference_period: dict) -> tuple[pandas.DataFrame, pandas.DataFrame]
        Generates a GWL table for a single model and ensemble member.
    get_gwl_table(model_config: dict, reference_period: dict) -> tuple[pandas.DataFrame, pandas.DataFrame]
        Generates GWL tables for a model across all its ensemble members.
    get_table_one_cesm2(model_config: dict, reference_period: dict) : Generates a GWL table for one member of CESM2.
    get_table_cesm2(model_config: dict, reference_period: dict) : Generates a GWL table for the CESM2 model.

    Examples
    --------
    >>> df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")
    >>> catalog_cesm = intake.open_esm_datastore(
            "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json"
        )
    >>> gwl_generator = GWLGenerator(df, catalog_cesm)
    >>> models = ["EC-Earth3"]
    >>> reference_periods = [{"start_year": "19810101", "end_year": "20101231"}]
    >>> gwl_generator.generate_gwl_file(models, reference_periods)

    """

    def __init__(
        self,
        df: pd.DataFrame,
        catalog_cesm: intake_esm.core.esm_datastore,
        sims_on_aws: dict = None,
    ):
        """Initialize the GWLGenerator with the CMIP6 data catalog.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing metadata for CMIP6 simulations
        catalog_cesm : intake_esm.esm_datastore
            Intake ESM catalog pointing to CESM2 data
        sims_on_aws : pandas.DataFrame, optional
            DataFrame listing available simulations on AWS. If None, it will be generated.

        """
        self.df = df
        self.sims_on_aws = (
            sims_on_aws if sims_on_aws is not None else self.get_sims_on_aws()
        )
        self.fs = s3fs.S3FileSystem(anon=True)

        # Settings specific to CESM2 LENS model
        # the LOCA-downscaled ensemble members are these, naming as described
        # in https://ncar.github.io/cesm2-le-aws/model_documentation.html) :
        self.ens_mems_cesm = {
            "r10i1p1f1": "r10i1181p1f1",
            "r1i1p1f1": "r1i1001p1f1",
            "r2i1p1f1": "r2i1021p1f1",
            "r3i1p1f1": "r3i1041p1f1",
            "r4i1p1f1": "r4i1061p1f1",
            "r5i1p1f1": "r5i1081p1f1",
            "r6i1p1f1": "r6i1101p1f1",
            "r7i1p1f1": "r7i1121p1f1",
            "r8i1p1f1": "r8i1141p1f1",
            "r9i1p1f1": "r9i1161p1f1",
        }
        self.cesm2_lens = self._set_cesm2_lens(catalog_cesm)

    def _set_cesm2_lens(
        self, catalog_cesm: intake_esm.core.esm_datastore
    ) -> xr.Dataset:
        """Pull CESM2 LENS dataset subset from Intake catalog and reformat datasets.

        Parameters
        ----------
        catalog_cesm : intake_esm.esm_datastore
            Intake ESM catalog pointing to CESM2 data

        Returns
        -------
        xr.Dataset
            CESM2 data for historical and ssp370

        """
        catalog_cesm_subset = catalog_cesm.search(
            variable="TREFHT", frequency="monthly", forcing_variant="cmip6"
        )

        dsets_cesm = catalog_cesm_subset.to_dataset_dict(storage_options={"anon": True})
        for ds in dsets_cesm:
            dsets_cesm[ds] = dsets_cesm[ds].sel(
                member_id=[v for k, v in self.ens_mems_cesm.items()]
            )
        historical_cmip6 = dsets_cesm["atm.historical.monthly.cmip6"]
        future_cmip6 = dsets_cesm["atm.ssp370.monthly.cmip6"]
        cesm2_lens = xr.concat(
            [historical_cmip6["TREFHT"], future_cmip6["TREFHT"]], dim="time"
        )
        return cesm2_lens

    def get_sims_on_aws(self) -> pd.DataFrame:
        """Generates a pandas DataFrame listing all relevant CMIP6 simulations available on AWS.

        This function filters the input DataFrame `df` and identifies and lists CMIP6 model simulations
        for historical and various SSP (Shared Socioeconomic Pathway) scenarios. It only includes
        models that have both historical and at least one SSP ensemble member. Additionally, it ensures
        that only historical ensemble members with variants in at least one SSP are kept.

        Returns
        -------
        pandas.DataFrame
            A DataFrame indexed by model names (source_id) and columns corresponding to scenarios
            ('historical', 'ssp585', 'ssp370', 'ssp245', 'ssp126'). Each cell contains a list of
            ensemble member IDs available on AWS for that model and scenario.

        """
        # Get model list
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

        for model in models:
            for scenario in scenarios:
                df_scenario = self.df[
                    (self.df.table_id == "Amon")
                    & (self.df.variable_id == "tas")
                    & (self.df.experiment_id == scenario)
                    & (self.df.source_id == model)
                ]
                ensMembers = list(set(df_scenario.member_id))
                sims_on_aws.loc[model, scenario] = ensMembers

        # cut the table to those GCMs that have a historical + at least one SSP ensemble member
        for i, item in enumerate(sims_on_aws.T.columns):
            no_ssp = True
            for ssp in ["ssp585", "ssp370", "ssp245", "ssp126"]:
                if len(sims_on_aws.loc[item][ssp]) > 0:
                    no_ssp = False
            if (len(sims_on_aws.loc[item]["historical"]) < 1) or (no_ssp):
                sims_on_aws = sims_on_aws.drop(index=item)

        # next also drop any historical ensemble members that don't have a variant in an SSP
        for i, item in enumerate(sims_on_aws.T.columns):
            variants_to_keep = []
            for variant in sims_on_aws.loc[item]["historical"]:
                for ssp in ["ssp585", "ssp370", "ssp245", "ssp126"]:
                    if str(variant) in sims_on_aws.loc[item][ssp]:
                        variants_to_keep.append(variant)
            sims_on_aws.loc[item, "historical"] = list(set(variants_to_keep))

        return sims_on_aws

    def build_timeseries(self, model_config: dict[str, Any]) -> xr.Dataset:
        """Builds an xarray Dataset with a time dimension, containing the concatenated historical
        and SSP time series for all specified scenarios of a given model and ensemble member.
        Works for all of the models(/GCMs) in the list `models`, which appear in the current
        data catalog of WRF downscaling.

        Parameters
        ----------
        model_config : dict
            Dictionary containing 'variable', 'model', 'ens_mem', and 'scenarios' keys

        Returns
        -------
        xarray.Dataset
            A dataset with time as the dimension, containing the appended historical and SSP time series.

        """
        variable = model_config["variable"]
        model = model_config["model"]
        ens_mem = model_config["ens_mem"]
        scenarios = model_config["scenarios"]

        df_subset = self.df[
            (self.df.table_id == "Amon")
            & (self.df.variable_id == "tas")
            & (self.df.experiment_id == "historical")
        ]
        scenario = "historical"
        data_historical = xr.Dataset()
        df_scenario = df_subset[
            (df_subset.source_id == model) & (df_subset.member_id == ens_mem)
        ]
        if not df_scenario.empty:
            try:
                store_url = df_scenario.zstore.values[0]
                temp_hist = xr.open_zarr(
                    store_url, consolidated=True, storage_options={"anon": True}
                )
                # Create global weighted time-series of variable
                data_historical = make_weighted_timeseries(temp_hist[variable])

                if model == "FGOALS-g3" or (
                    model == "EC-Earth3-Veg" and ens_mem == "r5i1p1f1"
                ):
                    data_historical = data_historical.isel(time=slice(0, -12 * 2))
                data_historical = data_historical.sortby(
                    "time"
                )  # needed for MPI-ESM-2-HR
            except Exception as e:
                print(f"Error loading historical data: {e}")
                return xr.Dataset()

        data_one_model = xr.Dataset()
        for scenario in scenarios:
            if ens_mem in self.sims_on_aws.T[model][scenario]:
                df_scenario = self.df[
                    (self.df.table_id == "Amon")
                    & (self.df.variable_id == variable)
                    & (self.df.experiment_id == scenario)
                    & (self.df.source_id == model)
                    & (self.df.member_id == ens_mem)
                ]
                if not df_scenario.empty:
                    try:
                        # BUG: Some scenarios not returning a full predictive period of values (i.e. not returning time period of 2015-2100 of data)
                        store_url = df_scenario.zstore.values[0]
                        temp_hist = xr.open_zarr(
                            store_url,
                            decode_times=False,
                            consolidated=True,
                            storage_options={"anon": True},
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

    def buildDFtimeSeries_cesm2(self, model_config: dict[str, Any]) -> xr.Dataset:
        """Builds a global temperature time series by weighting latitudes and averaging longitudes
        for the CESM2 model across specified scenarios from 1980 to 2100.

        Parameters
        ----------
        model_config : dict
            Dictionary containing 'variable', 'model', 'ens_mem', and 'scenarios' keys

        Returns
        -------
        xarray.Dataset
            A dataset containing the global temperature time series for each scenario.

        """
        ens_mem = model_config["ens_mem"]
        scenarios = model_config["scenarios"]

        temp = self.cesm2_lens.sel(member_id=ens_mem)
        data_one_model = xr.Dataset()
        for scenario in scenarios:
            # Create global weighted time-series of variable
            timeseries = make_weighted_timeseries(temp)
            timeseries = timeseries.sortby("time")

            data_one_model[scenario] = timeseries
        return data_one_model

    @staticmethod
    def get_gwl(smoothed: pd.DataFrame, degree: float) -> pd.DataFrame:
        """Computes the timestamp when a given GWL is first reached.
        Takes a smoothed time series of global mean temperature of different scenarios for a model
        and returns a table indicating the timestamp at which the specified warming level is reached.

        Parameters
        ----------
        smoothed : pandas.DataFrame
            A DataFrame containing a global mean temperature time series for a model for multiple scenarios.
        degree : float
            The global warming level to detect, e.g., 1.5, 2, etc.

        Returns
        -------
        pandas.DataFrame
            A table containing timestamps for when each scenario first crosses the specified warming level.

        """

        def get_wl_timestamp(
            scenario: pd.Series, degree: float
        ) -> cftime.DatetimeNoLeap | float:
            """Given a scenario of wl's and timestamps, find the timestamp that first crosses the degree passed in.
            Return np.NaN if none of the timestamps pass this degree level.

            """
            if any(scenario >= degree):
                wl_ts = scenario[scenario >= degree].index[0]
                return wl_ts
            else:
                return np.nan

        gwl = smoothed.apply(lambda scenario: get_wl_timestamp(scenario, degree))
        return gwl

    def get_table_one_cesm2(
        self, model_config: dict, reference_period: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generates a GWL lookup table for one ensemble member of CESM2.

        Parameters
        ----------
        model_config : dict
            Dictionary containing 'variable', 'model', 'ens_mem', and 'scenarios' keys
        reference_period : dict
            Dictionary containing 'start_year' and 'end_year' keys

        Returns
        -------
        tuple
            A DataFrame of warming levels and a DataFrame of global mean temperature time series.

        """
        model = model_config["model"]
        ens_mem = model_config["ens_mem"]
        start_year = reference_period["start_year"]
        end_year = reference_period["end_year"]

        data_one_model = self.buildDFtimeSeries_cesm2(model_config)
        anom = data_one_model - data_one_model.sel(
            time=slice(start_year, end_year)
        ).mean(
            "time"
        )  #'18500101','19000101'
        smoothed = anom.rolling(time=20 * 12, center=True).mean("time")
        oneModel = (
            smoothed.to_array(dim="scenario", name=model).dropna("time").to_pandas()
        )
        gwlevels = pd.DataFrame()
        try:
            for level in WARMING_LEVELS:
                gwlevels[level] = self.get_gwl(oneModel.T, level)
        except Exception as e:
            print(
                model, ens_mem, " problems"
            )  # helps EC-Earth3 not be skipped altogether
            print(e)

        # Modifying and returning oneModel to be seen as a WL lookup table with timestamp as index, to get the average WL across all simulations.
        final_model = oneModel.T
        final_model.columns = ens_mem + "_" + final_model.columns
        return gwlevels, final_model

    def get_table_cesm2(
        self, model_config: dict[str, Any], reference_period: dict[str, str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generates a GWL table for the CESM2 model.

        Parameters
        ----------
        model_config : dict
            Dictionary containing 'variable', 'model', 'ens_mem', and 'scenarios' keys
        reference_period : dict
            Dictionary containing 'start_year' and 'end_year' keys

        Returns
        -------
        tuple
            A DataFrame of warming levels and a DataFrame of global mean temperature time series for the CESM2 model.

        """
        global test
        model = model_config["model"]

        ens_mem_cesm_rev = dict([(v, k) for k, v in self.ens_mems_cesm.items()])
        ens_mem_list = [v for k, v in self.ens_mems_cesm.items()]
        if test:
            ens_mem_list = ens_mem_list[:2]
        gwlevels_tbl, wl_data_tbls = [], []
        for ens_mem in ens_mem_list:
            model_config["ens_mem"] = ens_mem
            gwlevels, wl_data_tbl = self.get_table_one_cesm2(
                model_config, reference_period
            )
            gwlevels_tbl.append(gwlevels)
            wl_data_tbls.append(wl_data_tbl)

        # Renaming columns of all ensemble members within model
        wl_data_tbl_sim = pd.concat(wl_data_tbls, axis=1)
        wl_data_tbl_sim.columns = model + "_" + wl_data_tbl_sim.columns
        return (
            pd.concat(
                gwlevels_tbl,
                keys=[("CESM2-LENS", ens_mem_cesm_rev[one]) for one in ens_mem_list],
            ),
            wl_data_tbl_sim,
        )

    def get_gwl_table_for_single_model_and_ensemble(
        self, model_config: dict[str, Any], reference_period: dict[str, str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generates a GWL table for a single model and ensemble member.

        Loops through various global warming levels from `climakitae.core.constants`
        for the requested model/variant and scenarios.

        Parameters
        ----------
        model_config : dict
            Dictionary containing 'variable', 'model', 'ens_mem', and 'scenarios' keys
        reference_period : dict
            Dictionary containing 'start_year' and 'end_year' keys

        Returns
        -------
        tuple
            A DataFrame containing warming levels and a DataFrame with global mean temperature time series.

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

            ### one_model is a dataframe of times and warming levels by scenario (scenarios as columns) WITHIN a specific model and ensemble member
            one_model = (
                smoothed.to_array(dim="scenario", name=model)
                .dropna("time", how="all")
                .to_pandas()  # Dropping time slices that are NaN across all SSPs
            )
            gwlevels = pd.DataFrame()
            try:
                for level in WARMING_LEVELS:
                    gwlevels[level] = self.get_gwl(one_model.T, level)
            except Exception as e:
                print(
                    model, ens_mem, " problems"
                )  # helps EC-Earth3 not be skipped altogether
                print(e)

            # Modifying and returning one_model to be seen as a WL table to index by timestamp to get average WL across all simulations.one_model.
            final_model = one_model.T
            final_model.columns = ens_mem + "_" + final_model.columns
            return gwlevels, final_model

        # Handle all anomaly calculation failures with a proper error message and return empty DataFrames
        except Exception as e:
            print(f"Error calculating anomalies for {model}, {ens_mem}: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_gwl_table(
        self, model_config: dict[str, Any], reference_period: dict[str, str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generates a GWL table for a given model.

        Parameters
        ----------
        model_config : dict
            Dictionary containing 'variable', 'model', 'ens_mem', and 'scenarios' keys
        reference_period : dict
            Dictionary containing 'start_year' and 'end_year' keys

        Returns
        -------
        tuple
            A DataFrame containing warming levels and a DataFrame with global mean temperature time series.
            To be exported into `gwl_[time period]ref.csv` and `gwl_[time period]ref_timeidx.csv`.

        """
        global test
        model = model_config["model"]

        ens_mem_list = self.sims_on_aws.T[model]["historical"].copy()
        if test:
            ens_mem_list = ens_mem_list[:2]
        if (model == "EC-Earth3") or (model == "EC-Earth3-Veg"):
            for ens_mem in ens_mem_list[:]:
                if int(ens_mem.split("r")[1].split("i")[0]) > 100:
                    # These ones were branched off another at 1970
                    ens_mem_list.remove(ens_mem)

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
                wl_data_tbl_sim = pd.concat(wl_data_tbls, axis=1)
            except Exception as e:
                print(f"Error concatenating timeseries results for model {model}: {e}")
                return pd.DataFrame(), pd.DataFrame()

            print(model, wl_data_tbl_sim.columns)
            wl_data_tbl_sim.columns = model + "_" + wl_data_tbl_sim.columns

            # Use the filtered list for concatenation
            try:
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

    def generate_gwl_file(self, models: list[str], reference_periods: list[dict]):
        """Generates global warming level (GWL) reference files for specified models.

        Parameters
        ----------
        models : list
            List of model names to processThe keys in the returned dictionary
        reference_periods : list
            List of dictionaries with 'start_year' and 'end_year' keys

        """

        for period in reference_periods:

            # Setting variables
            variable = "tas"
            start_year = period["start_year"]
            end_year = period["end_year"]

            # Writing out CESM2-LENS data
            model_config = {
                "variable": variable,
                "model": "CESM2-LENS",
                "scenarios": ["ssp370"],
            }
            print("Generate cesm2 table {}-{}".format(start_year[:4], end_year[:4]))
            cesm2_table, wl_data_tbl_cesm2 = self.get_table_cesm2(model_config, period)

            ## Generating GWL information for rest of models
            scenarios = ["ssp585", "ssp370", "ssp245"]
            print("Generate all WL table {}-{}".format(start_year[:4], end_year[:4]))
            all_wl_data_tbls = pd.DataFrame()
            all_gw_tbls, all_gw_data_tbls = [], []
            model_config = {
                "variable": variable,
                "model": "",
                "scenarios": scenarios,
            }

            # Extracts GWL information for each model
            for i, model in enumerate(models):

                print(f"\n...Model {i} {model}...\n")
                model_config["model"] = model
                gw_tbl, wl_data_tbl_sim = self.get_gwl_table(model_config, period)
                all_gw_tbls.append(gw_tbl)
                all_gw_data_tbls.append(wl_data_tbl_sim)
                try:
                    all_wl_data_tbls = pd.concat(
                        [all_wl_data_tbls, wl_data_tbl_sim], axis=1
                    )
                except Exception as e:
                    print(
                        f"\n Model {model} is skipped. Its table cannot be concatenated as its datetime indices are different: \n"
                    )
                    print(e)

            # Combining dataframes and resetting time index due to conflicting datetime object types
            try:
                wl_timeidx = pd.concat([all_wl_data_tbls, wl_data_tbl_cesm2], axis=1)
                wl_timeidx.index = wl_timeidx.index.map(
                    lambda time: "-".join(map(str, [time.year, time.month]))
                )  # resetting index
                wl_timeidx = wl_timeidx.groupby(
                    level=0
                ).mean()  # grouping times and removing NaNs via mean()
                write_csv_file(
                    wl_timeidx,
                    "data/gwl_{}-{}ref_timeidx.csv".format(
                        start_year[:4], end_year[:4]
                    ),
                )
            except Exception as e:
                print(
                    f"Error writing GWL index file for {start_year[:4]}-{end_year[:4]}: {e}"
                )

            # Create WL lookup table for reference period
            if all_gw_data_tbls:
                [print(x.head()) for x in all_gw_tbls]
                all_gw_levels = pd.concat(all_gw_tbls, keys=models)
                all_gw_levels = pd.concat([all_gw_levels, cesm2_table])
                all_gw_levels.index = pd.MultiIndex.from_tuples(
                    all_gw_levels.index, names=["GCM", "run", "scenario"]
                )
                try:
                    success = write_csv_file(
                        all_gw_levels,
                        "data/gwl_{}-{}ref.csv".format(start_year[:4], end_year[:4]),
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
    """Generates global warming level (GWL) reference files for all available CMIP6 GCMs and CESM2-LENS.

    This includes:
    - Connecting to AWS S3 storage to access CMIP6 and CESM2-LENS data.
    - Filtering and processing data to create global temperature time series.
    - Generating and saving warming level tables in CSV format for different reference periods.

    """
    # Connect to AWS S3 storage

    global test
    test = _kTest
    try:
        print("Loading CMIP6 catalog...")
        df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")

        # CESM2-LENS is in a separate catalog:
        print("Loading CESM catalog...")
        catalog_cesm = intake.open_esm_datastore(
            "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json"
        )

        print("Initializing GWL generator...")
        try:
            gwl_generator = GWLGenerator(df, catalog_cesm)
            sims_on_aws = gwl_generator.get_sims_on_aws()
            models = list(sims_on_aws.T.columns)

            if test:
                models = ["ACCESS-CM2"]

            # Pre-defined configuration
            reference_periods = [
                {"start_year": "18500101", "end_year": "19000101"},
                {"start_year": "19810101", "end_year": "20101231"},
            ]

            print(f"Generating GWL file for {models}...")
            try:
                gwl_generator.generate_gwl_file(models, reference_periods)
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
