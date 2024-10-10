""" 
Util for generating warming level reference data in ../data/ ###

To run, type: <<python generate_gwl_tables.py>> in the command line and wait for printed model outputs showing progress.
Generation takes ~1.5 hours for generating all 4 csv's.
"""

import s3fs
import intake
import pandas as pd
import xarray as xr
import numpy as np
from climakitae.util.utils import write_csv_file


def main():
    """
    Generates global warming level (GWL) reference files for all available CMIP6 GCMs and CESM2-LENS.

    This includes:
    - Connecting to AWS S3 storage to access CMIP6 and CESM2-LENS data.
    - Filtering and processing data to create global temperature time series.
    - Generating and saving warming level tables in CSV format for different reference periods.
    """
    # Connect to AWS S3 storage
    fs = s3fs.S3FileSystem(anon=True)

    df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")
    df_subset = df[
        (df.table_id == "Amon")
        & (df.variable_id == "tas")
        & (df.experiment_id == "historical")
    ]
    sims_on_aws = get_sims_on_aws(df)
    models = ['EC-Earth3']


    def make_weighted_timeseries(temp):
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
        """
        # Find variable names for latitude and longitude to make code more readable
        lat, lon = "lat", "lon"
        if "lat" not in temp.coords and "lon" not in temp.coords:
            lat, lon = "latitude", "longitude"

        # Weight latitude grids by size, then average across all longitudes to create single time-series object
        weightlat = np.sqrt(np.cos(np.deg2rad(temp[lat])))
        weightlat = weightlat / np.sum(weightlat)
        timeseries = (temp * weightlat).sum(lat).mean(lon)
        return timeseries


    def build_timeseries(variable, model, ens_mem, scenarios):
        """
        Builds an xarray Dataset with a time dimension, containing the concatenated historical
        and SSP time series for all specified scenarios of a given model and ensemble member.
        Works for all of the models(/GCMs) in the list `models`, which appear in the current
        data catalog of WRF downscaling.

        Parameters:
        ----------
        variable : str
            The variable to process, e.g., `tas`. `tas` is the only variable used in this file currently.
        model : str
            The model name.
        ens_mem : str
            The ensemble member ID.
        scenarios : list
            A list of scenario names to include, e.g., ['historical', 'ssp585', 'ssp370'].

        Returns:
        -------
        xarray.Dataset
            A dataset with time as the dimension, containing the appended historical and SSP time series.
        """
        scenario = "historical"
        data_historical = xr.Dataset()
        df_scenario = df_subset[
            (df_subset.source_id == model) & (df_subset.member_id == ens_mem)
        ]
        with xr.open_zarr(fs.get_mapper(df_scenario.zstore.values[0])) as temp:

            # Create global weighted time-series of variable
            data_historical = make_weighted_timeseries(temp[variable])
            data_historical = data_historical.sortby("time")  # needed for MPI-ESM-2-HR

        data_one_model = xr.Dataset()
        for scenario in scenarios:
            if ens_mem in sims_on_aws.T[model][scenario]:
                df_scenario = df[
                    (df.table_id == "Amon")
                    & (df.variable_id == variable)
                    & (df.experiment_id == scenario)
                    & (df.source_id == model)
                    & (df.member_id == ens_mem)
                ]
                if not df_scenario.empty:
                    with xr.open_zarr(
                        fs.get_mapper(df_scenario.zstore.values[0]), decode_times=False
                    ) as temp:  # BUG: Some scenarios not returning a full predictive period of values (i.e. not returning time period of 2015-2100 of data)
                        temp = temp.isel(time=slice(0, 1032))
                        temp = xr.decode_cf(temp)

                        # Create global weighted time-series of variable
                        timeseries = make_weighted_timeseries(temp[variable])

                        # Clean data and append to `data_one_model`
                        timeseries = timeseries.sortby(
                            "time"
                        )  # needed for MPI-ESM1-2-LR
                        data_one_model[scenario] = xr.concat(
                            [data_historical, timeseries], dim="time"
                        )  # .to_pandas())
        return data_one_model

    def get_gwl(smoothed, degree):
        """
        Computes the timestamp when a given GWL is first reached.
        Takes a smoothed time series of global mean temperature of different scenarios for a model
        and returns a table indicating the timestamp at which the specified warming level is reached.

        Parameters:
        ----------
        smoothed : pandas.DataFrame
            A DataFrame containing a global mean temperature time series for a model for multiple scenarios.
        degree : float
            The global warming level to detect, e.g., 1.5, 2, etc.

        Returns:
        -------
        pandas.DataFrame
            A table containing timestamps for when each scenario first crosses the specified warming level.
        """

        def get_wl_timestamp(scenario, degree):
            """
            Given a scenario of wl's and timestamps, find the timestamp that first crosses the degree passed in.
            Return np.NaN if none of the timestamps pass this degree level.
            """
            if any(scenario >= degree):
                wl_ts = scenario[scenario >= degree].index[0]
                return wl_ts
            else:
                return np.NaN

        gwl = smoothed.apply(lambda scenario: get_wl_timestamp(scenario, degree))
        return gwl


    def get_gwl_table_one(
        variable, model, ens_mem, scenarios, start_year="18500101", end_year="19000101"
    ):
        """
        Generates a GWL table for a single model and ensemble member.

        Loops through various global warming levels (1.5, 2, 2.5, 3, and 4 degrees)
        for the requested model/variant and scenarios.

        Parameters:
        ----------
        variable : str
            The variable to process, e.g., 'tas'.
        model : str
            The model name.
        ens_mem : str
            The ensemble member ID.
        scenarios : list
            A list of scenario names to include, e.g., ['historical', 'ssp585', 'ssp370'].
        start_year : str, optional
            The start year for the reference period in the format 'YYYYMMDD'.
        end_year : str, optional
            The end year for the reference period in the format 'YYYYMMDD'.

        Returns:
        -------
        tuple
            A DataFrame containing warming levels and a DataFrame with global mean temperature time series.
        """
        data_one_model = build_timeseries(variable, model, ens_mem, scenarios)
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
            for level in [1.5, 2, 2.5, 3, 4]:
                gwlevels[level] = get_gwl(one_model.T, level)
        except Exception as e:
            print(
                model, ens_mem, " problems"
            )  # helps EC-Earth3 not be skipped altogether
            print(e)

        # Modifying and returning one_model to be seen as a WL table to index by timestamp to get average WL across all simulations.one_model.
        final_model = one_model.T
        final_model.columns = ens_mem + "_" + final_model.columns
        return gwlevels, final_model

    def get_gwl_table(
        variable, model, scenarios, start_year="18500101", end_year="19000101"
    ):
        """
        Generates a GWL table for a given model and scenarios.

        Parameters:
        ----------
        variable : str
            The variable to process, e.g., 'tas'.
        model : str
            The model name.
        scenarios : list
            A list of scenario names to include, e.g., ['ssp585', 'ssp370', 'ssp245'].
        start_year : str, optional
            The start year for the reference period in the format 'YYYYMMDD'.
        end_year : str, optional
            The end year for the reference period in the format 'YYYYMMDD'.

        Returns:
        -------
        tuple
            A DataFrame containing warming levels and a DataFrame with global mean temperature time series.
            To be exported into `gwl_[time period]ref.csv` and `gwl_[time period]ref_timeidx.csv`.
        """
        ens_mem_list = sims_on_aws.T[model]["ssp370"].copy()
        print(f'Number of ensemble members: {len(ens_mem_list)}')
        try:
            # Combining all the ensemble members for a given model
            gwlevels_tbl, wl_data_tbls = [], []
            for i, ens_mem in enumerate(ens_mem_list):
                print(f'Getting gwl table for ensemble member {ens_mem}...')
                gwlevels, wl_data_tbl = get_gwl_table_one(
                    variable, model, ens_mem, scenarios, start_year, end_year
                )
                gwlevels_tbl.append(gwlevels)
                wl_data_tbls.append(wl_data_tbl)

            # Renaming columns of all ensemble members within model
            wl_data_tbl_sim = pd.concat(wl_data_tbls, axis=1)
            wl_data_tbl_sim.columns = model + "_" + wl_data_tbl_sim.columns
            return pd.concat(gwlevels_tbl, keys=ens_mem_list), wl_data_tbl_sim

        except:
            print('Cannot get gwl table')
            print(f'{i}th ensemble member {ens_mem}')

    ##### Generating and writing GWL data tables for all GCMS #####

    ### Generating WL CSVs for two reference periods: pre-industrial and secondary reference period overlapping with downscaled data availability:
    time_periods = [
        {"start_year": "19810101", "end_year": "20101231"}
    ]

    for period in time_periods:

        # Setting variables
        variable = "tas"
        start_year = period["start_year"]
        end_year = period["end_year"]

        ## Generating GWL information for rest of models
        scenarios = ["ssp370"]
        print("Generate all WL table {}-{}".format(start_year[:4], end_year[:4]))
        all_wl_data_tbls = pd.DataFrame()
        all_gw_tbls, all_gw_data_tbls = [], []

        # Extracts GWL information for each model
        for i, model in enumerate(models):
            print(f"\n...Model {i} {model}...\n")
            gw_tbl, wl_data_tbl_sim = get_gwl_table(
                variable, model, scenarios, start_year, end_year
            )
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


        # Creating WL lookup table with 1850-1900 reference period
        all_gw_levels = pd.concat(all_gw_tbls, keys=models)
        all_gw_levels.index = pd.MultiIndex.from_tuples(
            all_gw_levels.index, names=["GCM", "run", "scenario"]
        )
        write_csv_file(
            all_gw_levels, "data/gwl_{}-{}ref_EC-Earth3.csv".format(start_year[:4], end_year[:4])
        )


def get_sims_on_aws(df):
    """
    Generates a pandas DataFrame listing all relevant CMIP6 simulations available on AWS.

    This function filters the input DataFrame `df` and identifies and lists CMIP6 model simulations
    for historical and various SSP (Shared Socioeconomic Pathway) scenarios. It only includes
    models that have both historical and at least one SSP ensemble member. Additionally, it ensures
    that only historical ensemble members with variants in at least one SSP are kept.

    Parameters:
    ----------
    df : pandas.DataFrame
        A DataFrame containing metadata for CMIP6 simulations.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame indexed by model names (source_id) and columns corresponding to scenarios
        ('historical', 'ssp585', 'ssp370', 'ssp245', 'ssp126'). Each cell contains a list of
        ensemble member IDs available on AWS for that model and scenario.
    """
    df_subset = df[
        (df.table_id == "Amon")
        & (df.variable_id == "tas")
        & (df.experiment_id == "historical")
    ]
    models = list(set(df_subset.source_id))
    models.sort()

    # First cut through the catalog
    scenarios = ["historical", "ssp585", "ssp370", "ssp245", "ssp126"]
    sims_on_aws = pd.DataFrame(index=models, columns=scenarios)

    for model in models:
        for scenario in scenarios:
            df_scenario = df[
                (df.table_id == "Amon")
                & (df.variable_id == "tas")
                & (df.experiment_id == scenario)
                & (df.source_id == model)
            ]
            ensMembers = list(set(df_scenario.member_id))
            sims_on_aws[scenario][model] = ensMembers

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
        sims_on_aws.loc[item]["historical"] = list(set(variants_to_keep))

    return sims_on_aws


if __name__ == "__main__":
    main()
