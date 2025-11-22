"""Station Bias Correction Processor for ClimakitAE.

This module provides the StationBiasCorrection processor for bias-correcting
gridded climate model data to weather station locations using Quantile Delta
Mapping (QDM) with historical observational data from the HadISD dataset.

The processor performs the following operations:
1. Loads HadISD weather station observations from S3 zarr stores
2. Finds the closest gridcell in the climate model data to each station
3. Applies quantile delta mapping bias correction using historical overlap period
4. Returns bias-corrected data at station locations with station metadata

Classes
-------
StationBiasCorrection : DataProcessor
    Main processor for station-based bias correction using QDM method.

Examples
--------
>>> # Create processor for single station
>>> processor = StationBiasCorrection(
...     stations=["Sacramento (KSAC)"],
...     historical_slice=(1980, 2014)
... )
>>> result = processor.execute(gridded_data, context)

>>> # Multiple stations with custom bias correction parameters
>>> processor = StationBiasCorrection(
...     stations=["Sacramento (KSAC)", "San Francisco (KSFO)"],
...     historical_slice=(1980, 2014),
...     window=60,  # 60-day window instead of default 90
...     nquantiles=30  # 30 quantiles instead of default 20
... )
>>> result = processor.execute(gridded_data, context)

Notes
-----
- Requires gridded data to include historical period (1980-2014) for bias correction
- Station observational data is available through 2014-08-31
- Uses xclim's QuantileDeltaMapping for bias correction
- Converts all data to noleap calendar for consistency
- Final output is time-sliced to user's requested period after bias correction
"""

import logging
import re
from functools import partial
from typing import Any, Dict, Iterable, Optional, Union

import geopandas as gpd
import pandas as pd
import xarray as xr
from xsdba import Grouper
from xsdba.adjustment import QuantileDeltaMapping

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.util.unit_conversions import convert_units
from climakitae.util.utils import get_closest_gridcell

# Module logger
logger = logging.getLogger(__name__)


@register_processor("bias_adjust_model_to_station", priority=60)
class BiasCorrectStationData(DataProcessor):
    """Bias-correct gridded climate data to weather station locations using QDM.

    This processor applies Quantile Delta Mapping (QDM) bias correction to gridded
    climate model data using historical observations from HadISD weather stations.
    The method corrects for systematic biases in climate model output by matching
    the statistical distribution of model data to observed data during a historical
    training period, then applying these corrections to future projections.

    The processor handles:
    - Loading HadISD station observations from S3 zarr stores
    - Preprocessing station data (unit conversions, calendar conversions)
    - Finding closest gridcells to station locations
    - Training QDM bias correction on historical overlap period (1980-2014)
    - Applying corrections to user-specified time period
    - Preserving station metadata in output

    **Important**: This processor returns lazy dask arrays to maintain memory
    efficiency and pipeline composability. Users should call `.compute()` on the
    result when ready to load data into memory. This matches the behavior of the
    legacy interface and allows for efficient chaining of operations.

    Parameters
    ----------
    stations : list[str]
        List of station names to process (e.g., ["Sacramento (KSAC)", "San Francisco (KSFO)"])
    historical_slice : tuple[int, int], optional
        Start and end years for historical training period (default: (1980, 2014))
    window : int, optional
        Window size in days for seasonal grouping (default: 90, representing +/- 45 days)
    nquantiles : int, optional
        Number of quantiles for QDM training (default: 20)
    group : str, optional
        Temporal grouping strategy for bias correction (default: "time.dayofyear")
    kind : str, optional
        Adjustment kind: "+" for additive (temperature) or "*" for multiplicative
        (precipitation) (default: "+")

    Attributes
    ----------
    stations : list[str]
        Station names to process
    historical_slice : tuple[int, int]
        Historical training period (default: 1980-2014)
    window : int
        Seasonal grouping window in days
    nquantiles : int
        Number of quantiles for QDM
    group : str
        Temporal grouping strategy
    kind : str
        Adjustment kind (additive or multiplicative)
    name : str
        Processor name for context tracking
    catalog : DataCatalog
        Data catalog for accessing station metadata

    Methods
    -------
    execute(result, context)
        Apply station bias correction to gridded data
    update_context(context)
        Update context with bias correction operation metadata
    set_data_accessor(catalog)
        Set data catalog accessor (not used in this processor)

    See Also
    --------
    climakitae.util.utils.get_closest_gridcell : Find closest gridcell to point
    xsdba.adjustment.QuantileDeltaMapping : QDM bias correction implementation

    References
    ----------
    .. [1] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of
       GCM precipitation by quantile mapping: How well do methods preserve changes
       in quantiles and extremes? Journal of Climate, 28(17), 6938-6959.
    """

    def __init__(self, value: Dict[str, Any]):
        """Initialize the station bias correction processor.

        Parameters
        ----------
        value : Dict[str, Any]
            Configuration dictionary containing:
            - stations : list[str]
                List of station names to process
            - historical_slice : tuple[int, int], optional
                Start and end years for historical training period (default: (1980, 2014))
            - window : int, optional
                Window size in days for seasonal grouping (default: 90)
            - nquantiles : int, optional
                Number of quantiles for QDM (default: 20)
            - group : str, optional
                Temporal grouping strategy (default: "time.dayofyear")
            - kind : str, optional
                Adjustment kind: "+" or "*" (default: "+")
        """
        # Validate input
        if not isinstance(value, dict):
            raise TypeError(
                "Expected dictionary for station bias correction configuration"
            )

        # Extract configuration parameters with defaults
        self.stations = value.get("stations", [])
        self.historical_slice = value.get("historical_slice", (1980, 2014))
        self.window = value.get("window", 90)
        self.nquantiles = value.get("nquantiles", 20)
        self.group = value.get("group", "time.dayofyear")
        self.kind = value.get("kind", "+")
        self.name = "bias_adjust_model_to_station"
        self.catalog: Union[DataCatalog, object] = UNSET
        self.needs_catalog = True

    def _preprocess_hadisd(
        self, ds: xr.Dataset, stations_gdf: gpd.GeoDataFrame
    ) -> xr.Dataset:
        """Preprocess HadISD station data for bias correction.

        This method prepares raw station data by:
        - Extracting station ID from file path
        - Looking up station name from metadata
        - Renaming data variable to station name
        - Converting temperature from Celsius to Kelvin
        - Adding descriptive attributes (coordinates, elevation)
        - Dropping non-time coordinates

        Parameters
        ----------
        ds : xr.Dataset
            Raw HadISD station dataset with 'tas' variable
        stations_gdf : gpd.GeoDataFrame
            Station metadata GeoDataFrame

        Returns
        -------
        xr.Dataset
            Preprocessed station dataset with station name as variable
        """
        # Get station ID from file name
        station_id = ds.encoding["source"].split("HadISD_")[1].split(".zarr")[0]

        # Get name of station from station_id
        station_name = stations_gdf.loc[stations_gdf["station id"] == int(station_id)][
            "station"
        ].item()

        # Rename data variable to station name
        ds = ds.rename({"tas": station_name})

        # Convert Celsius to Kelvin
        ds[station_name] = ds[station_name] + 273.15

        # Assign descriptive attributes to the data variable
        ds[station_name] = ds[station_name].assign_attrs(
            {
                "coordinates": (
                    ds.latitude.values.item(),
                    ds.longitude.values.item(),
                ),
                "elevation": "{0} {1}".format(
                    ds.elevation.item(), ds.elevation.attrs["units"]
                ),
                "units": "K",
            }
        )

        # Drop all coordinates except time
        ds = ds.drop_vars(["elevation", "latitude", "longitude"])

        return ds

    def _load_station_data(self) -> xr.Dataset:
        """Load HadISD station data from S3 zarr stores.

        Constructs file paths for selected stations and loads them using
        xarray's open_mfdataset with preprocessing for seamless integration.
        Uses get_station_coordinates for robust station validation.

        Returns
        -------
        xr.Dataset
            Combined dataset with each station as a separate data variable

        Raises
        ------
        RuntimeError
            If station data cannot be loaded or catalog is not available.
        ValueError
            If any station identifier is invalid or not found.
        """
        from climakitae.new_core.processors.processor_utils import (
            convert_stations_to_points,
        )

        # Validate all stations and get their metadata using the shared utility
        # This will raise ValueError with suggestions if any station is invalid
        _, metadata_list = convert_stations_to_points(self.stations, self.catalog)

        # Get full station metadata DataFrame for preprocessing
        station_metadata = self.catalog["stations"]

        # Extract numeric station IDs from validated metadata for HadISD file paths
        # The numeric ID is required for the HadISD zarr file naming convention
        station_ids = [meta["station_id_numeric"] for meta in metadata_list]

        # Construct S3 zarr paths for each station using numeric IDs
        filepaths = [
            f"s3://cadcat/hadisd/HadISD_{station_id}.zarr" for station_id in station_ids
        ]

        # Create informative log message with station codes and names
        station_info = [
            f"{meta['station_id']} ({meta['station_name']})" for meta in metadata_list
        ]
        logger.info(
            "Loading station data for %d validated station(s): %s",
            len(station_ids),
            ", ".join(station_info),
        )

        # Create partial function for preprocessing with station metadata
        preprocess_func = partial(
            self._preprocess_hadisd, stations_gdf=station_metadata
        )

        # Load all station data with preprocessing
        station_ds = xr.open_mfdataset(
            filepaths,
            preprocess=preprocess_func,
            engine="zarr",
            consolidated=False,
            parallel=True,
            backend_kwargs=dict(storage_options={"anon": True}),
        )

        return station_ds

    def _bias_correct_model_data(
        self,
        obs_da: xr.DataArray,
        gridded_da: xr.DataArray,
        historical_da: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """Apply Quantile Delta Mapping bias correction to model data.

        This method performs the core bias correction by:
        1. Converting units to match gridded data
        2. Rechunking data (QDM requires unchunked time dimension)
        3. Training QDM on historical overlap period (1980-2014)
        4. Applying correction to the input data

        Note: Data must be in noleap calendar before calling this method.
        Calendar conversion is handled by _process_single_dataset.

        Parameters
        ----------
        obs_da : xr.DataArray
            Observational station data (preprocessed, noleap calendar)
        gridded_da : xr.DataArray
            Climate model gridded data (noleap calendar)
        historical_da : xr.DataArray, optional
            Historical climate model data for training QDM (noleap calendar).
            If None, assumes gridded_da contains the historical period (legacy behavior).

        Returns
        -------
        xr.DataArray
            Bias-corrected data (noleap calendar)
        """
        logger.debug("=== Starting bias correction for station: %s ===", obs_da.name)
        # Avoid accessing .values for logging as it triggers computation
        logger.debug(
            "Input gridded_da time size: %s",
            gridded_da.sizes.get("time"),
        )
        if historical_da is not None:
            logger.debug(
                "Input historical_da time size: %s",
                historical_da.sizes.get("time"),
            )
        logger.debug(
            "Input obs_da time range: %s to %s",
            obs_da.time.values[0],
            obs_da.time.values[-1],
        )

        # Create grouper for seasonal window
        # Use 90 day window (+/- 45 days) to account for seasonality
        grouper = Grouper(self.group, window=self.window)
        logger.debug("Created grouper: group=%s, window=%s", self.group, self.window)

        # Convert units to match gridded data
        logger.debug(
            "Converting obs units from %s to %s", obs_da.units, gridded_da.units
        )
        obs_da = convert_units(obs_da, gridded_da.units)

        # Slice observational data to available period (through 2014-08-31)
        obs_da = obs_da.sel(time=slice(obs_da.time.values[0], "2014-08-31"))
        logger.debug(
            "Sliced obs_da to available period: %s to %s",
            obs_da.time.values[0],
            obs_da.time.values[-1],
        )

        # Rechunk data - cannot be chunked along time dimension
        # Error raised by xclim: ValueError: Multiple chunks along the main
        # adjustment dimension time is not supported.
        logger.debug("Rechunking data along time dimension")
        gridded_da = gridded_da.chunk(chunks=dict(time=-1))
        obs_da = obs_da.chunk(chunks=dict(time=-1))
        if historical_da is not None:
            historical_da = historical_da.chunk(chunks=dict(time=-1))

        if historical_da is not None:
            # Use provided historical data
            # Slice to match obs data period
            gridded_da_historical = historical_da.sel(
                time=slice(str(obs_da.time.values[0]), str(obs_da.time.values[-1]))
            )
        else:
            # Slice gridded data to match obs data period (legacy approach)
            # This ensures we only use the overlapping historical period
            gridded_da_historical = gridded_da.sel(
                time=slice(str(obs_da.time.values[0]), str(obs_da.time.values[-1]))
            )

        # Avoid accessing .values for logging
        logger.debug(
            "Gridded historical shape: %s",
            gridded_da_historical.shape,
        )

        # Now slice obs data to match the gridded historical data exactly
        # This handles any edge cases where times don't align perfectly
        # We need to access values here for slicing, but we can do it efficiently
        # obs_da is in memory (loaded from zarr), so accessing values is fast
        # gridded_da_historical might be lazy.
        # However, we need the time bounds.
        # If gridded_da_historical is lazy, accessing time.values triggers computation.
        # But we need it for the slice.
        # Let's try to use the time coordinate directly if possible, or accept the cost here.
        # But we can avoid printing it in the log if we already accessed it.

        # Optimization: Use min/max of time coordinate if available without loading all values?
        # For now, we assume we need the start/end.
        # But we can avoid the logger call accessing it AGAIN.

        t_start = str(gridded_da_historical.time.values[0])
        t_end = str(gridded_da_historical.time.values[-1])

        obs_da = obs_da.sel(time=slice(t_start, t_end))
        logger.debug(
            "Final obs period shape: %s",
            obs_da.shape,
        )

        # Check if data has a 'sim' dimension from concatenation
        # QDM must be trained and applied separately for each simulation
        if "sim" in gridded_da.dims:
            logger.debug("Data has 'sim' dimension, using broadcasted QDM")

            # Rename 'sim' to 'simulation' to avoid conflict with xsdba internal naming
            # xsdba uses 'sim' internally for the simulated dataset
            data_sliced_renamed = gridded_da.rename({"sim": "simulation"})
            hist_renamed = gridded_da_historical.rename({"sim": "simulation"})

            logger.debug(
                "Training QDM with nquantiles=%s, kind=%s (broadcasted)",
                self.nquantiles,
                self.kind,
            )

            # Train QDM with broadcasting
            # obs_da is (time), hist_renamed is (simulation, time)
            # QDM will learn distributions for each simulation
            QDM = QuantileDeltaMapping.train(
                obs_da,
                hist_renamed,
                nquantiles=self.nquantiles,
                group=grouper,
                kind=self.kind,
            )

            # Apply QDM
            logger.debug("Applying QDM adjustment (broadcasted)")
            da_adj = QDM.adjust(data_sliced_renamed)

            # Rename 'simulation' back to 'sim'
            da_adj = da_adj.rename({"simulation": "sim"})
            da_adj.name = gridded_da_historical.name

        else:
            # No sim dimension - train and apply QDM once
            logger.debug(
                "Training QDM with nquantiles=%s, kind=%s", self.nquantiles, self.kind
            )
            QDM = QuantileDeltaMapping.train(
                obs_da,
                gridded_da_historical,
                nquantiles=self.nquantiles,
                group=grouper,
                kind=self.kind,
            )
            logger.debug("QDM training complete")
            # No sim dimension, process as single array
            logger.debug("Applying QDM adjustment to data_sliced")
            logger.debug("data_sliced time dtype before QDM: %s", gridded_da.time.dtype)
            logger.debug(
                "data_sliced is dask array: %s", hasattr(gridded_da.data, "dask")
            )

            da_adj = QDM.adjust(gridded_da)
            da_adj.name = gridded_da_historical.name

            logger.debug("QDM adjustment complete (lazy)")

        # Convert time back to datetime64 (from CFTime noleap)
        # This matches legacy behavior and ensures compatibility with downstream operations
        logger.debug("Converting time coordinate back to datetime64")
        try:
            time_index = da_adj.indexes["time"]
            if hasattr(time_index, "to_datetimeindex"):
                da_adj["time"] = time_index.to_datetimeindex()
            else:
                # Fallback if to_datetimeindex not available
                da_adj = da_adj.convert_calendar("standard", use_cftime=False)
        except Exception as e:
            logger.warning(
                "Could not convert time back to datetime64: %s. Leaving as-is.", e
            )

        # Rechunk to convert back to dask array for downstream processing
        # This maintains lazy evaluation for subsequent operations
        logger.debug("=== Bias correction complete for %s ===", da_adj.name)

        return da_adj  # type: ignore[return-value]

    def _process_single_dataset(
        self,
        result: Union[xr.Dataset, xr.DataArray],
        station_ds: xr.Dataset,
        context: Dict[str, Any],
        historical_da: Optional[xr.DataArray] = None,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Process a single dataset/dataarray.

        Parameters
        ----------
        result : xr.Dataset or xr.DataArray
            Input data to bias correct
        station_ds : xr.Dataset
            Loaded station data
        context : Dict[str, Any]
            Processing context
        historical_da : xr.DataArray, optional
            Historical data for training QDM

        Returns
        -------
        xr.Dataset or xr.DataArray
            Bias corrected data
        """
        # Convert Dataset to DataArray if needed
        result_da: xr.DataArray
        if isinstance(result, xr.Dataset):
            # Get the data variable (should only be one for climate data)
            data_vars = list(result.data_vars)
            if len(data_vars) == 0:
                raise ValueError("Input Dataset has no data variables")
            if len(data_vars) > 1:
                logger.warning(
                    "Input Dataset has multiple data variables: %s. Using first: %s",
                    data_vars,
                    data_vars[0],
                )
            result_da = result[data_vars[0]]
            # Copy important attributes from Dataset to DataArray
            # These are needed by utility functions like get_closest_gridcell
            # We copy all attributes to be safe, as resolution and others are needed
            result_da.attrs.update(result.attrs)
            logger.info("Converted Dataset to DataArray: %s", result_da.name)
        elif isinstance(result, xr.DataArray):
            result_da = result
        else:
            raise TypeError(
                f"StationBiasCorrection requires xr.DataArray or xr.Dataset input, "
                f"got {type(result)}"
            )

        # Ensure resolution attribute is present for get_closest_gridcell
        if "resolution" not in result_da.attrs:
            # Try to get from grid_label in attrs
            grid_label = result_da.attrs.get("grid_label")

            # If not in attrs, try context
            if not grid_label and context:
                if "query" in context:
                    grid_label = context["query"].get("grid_label")

                # If still not found, try context directly (flat structure)
                if not grid_label:
                    grid_label = context.get("grid_label")

            if grid_label:
                grid_mapping = {
                    "d01": "45 km",
                    "d02": "9 km",
                    "d03": "3 km",
                }
                if grid_label in grid_mapping:
                    result_da.attrs["resolution"] = grid_mapping[grid_label]
                    logger.debug(
                        "Inferred resolution %s from grid_label %s",
                        result_da.attrs["resolution"],
                        grid_label,
                    )
                else:
                    logger.warning(
                        "Could not infer resolution from grid_label: %s", grid_label
                    )
            else:
                logger.warning(
                    "Resolution attribute missing and no grid_label found to infer it."
                )

        # OPTIMIZATION: Load coordinates into memory to speed up get_closest_gridcell
        # This prevents triggering dask computations for every station search
        if "lat" in result_da.coords:
            result_da.coords["lat"].load()
        if "lon" in result_da.coords:
            result_da.coords["lon"].load()

        if historical_da is not None:
            if "lat" in historical_da.coords:
                historical_da.coords["lat"].load()
            if "lon" in historical_da.coords:
                historical_da.coords["lon"].load()

        logger.info(
            "Applying QDM bias adjustment on models for %d station(s)",
            len(self.stations),
        )
        logger.debug("Input result_da name: %s", result_da.name)
        logger.debug("Input result_da shape: %s", result_da.shape)
        logger.debug("Input result_da dims: %s", result_da.dims)
        logger.debug("Input result_da time dtype: %s", result_da.time.dtype)
        logger.debug("Input result_da is dask: %s", hasattr(result_da.data, "dask"))

        # Vectorized processing: Collect data for all stations first
        station_names = list(station_ds.data_vars.keys())
        gridded_list = []
        historical_list = []

        # Pre-calculate station metadata to restore later
        station_metadata = {}

        for station_name in station_names:
            logger.debug("Extracting grid cell for station: %s", station_name)
            station_da = station_ds[station_name]
            station_lat, station_lon = station_da.attrs["coordinates"]
            station_metadata[station_name] = {
                "coordinates": (station_lat, station_lon),
                "elevation": station_da.attrs.get("elevation", "N/A"),
            }

            # Extract model data
            gridded_da_closest = get_closest_gridcell(
                result_da, station_lat, station_lon, print_coords=False
            )

            if gridded_da_closest is None:
                raise ValueError(
                    f"Could not find closest gridcell at ({station_lat}, {station_lon})"
                )

            # Drop extra coords
            gridded_da_closest = gridded_da_closest.drop_vars(
                [
                    c
                    for c in gridded_da_closest.coords
                    if c not in gridded_da_closest.dims
                ]
            )
            gridded_list.append(gridded_da_closest)

            # Extract historical data if present
            if historical_da is not None:
                historical_da_closest = get_closest_gridcell(
                    historical_da, station_lat, station_lon, print_coords=False
                )
                if historical_da_closest is None:
                    raise ValueError(
                        f"Could not find historical gridcell at ({station_lat}, {station_lon})"
                    )
                historical_da_closest = historical_da_closest.drop_vars(
                    [
                        c
                        for c in historical_da_closest.coords
                        if c not in historical_da_closest.dims
                    ]
                )
                historical_list.append(historical_da_closest)

        # Stack data along 'station' dimension
        station_index = pd.Index(station_names, name="station")

        # Stack gridded data
        gridded_stacked = xr.concat(gridded_list, dim=station_index)

        # Stack historical data
        historical_stacked = None
        if historical_list:
            historical_stacked = xr.concat(historical_list, dim=station_index)

        # Stack station data
        # Use to_array to convert Dataset to DataArray with 'station' dim
        station_stacked = station_ds.to_array(dim="station", name="station_data")

        # Restore units (assuming all stations have same units, which they do: K)
        if station_names:
            first_station = station_ds[station_names[0]]
            if "units" in first_station.attrs:
                station_stacked.attrs["units"] = first_station.attrs["units"]

        # Convert calendars to noleap (vectorized)
        gridded_stacked = gridded_stacked.convert_calendar("noleap")
        if historical_stacked is not None:
            historical_stacked = historical_stacked.convert_calendar("noleap")
        station_stacked = station_stacked.convert_calendar("noleap")

        # Apply Bias Correction (Vectorized)
        # This applies QDM once across all stations (broadcasting over 'station' dim)
        bias_corrected_stacked = self._bias_correct_model_data(
            station_stacked,
            gridded_stacked,
            historical_da=historical_stacked,
        )

        # Unstack to Dataset
        apply_output = bias_corrected_stacked.to_dataset(dim="station")

        # Restore attributes
        for station_name in station_names:
            if station_name in apply_output:
                apply_output[station_name].attrs["station_coordinates"] = (
                    station_metadata[station_name]["coordinates"]
                )
                apply_output[station_name].attrs["station_elevation"] = (
                    station_metadata[station_name]["elevation"]
                )

        logger.info(
            "Station bias correction complete. Output shape: %s", apply_output.dims
        )
        logger.debug("Output variables: %s", list(apply_output.data_vars))

        return apply_output

    def _execute_dict(
        self,
        result: Dict[str, Union[xr.Dataset, xr.DataArray]],
        context: Dict[str, Any],
    ) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """Execute bias correction on a dictionary of datasets.

        This method handles the pre-concatenation case where we have separate
        historical and SSP datasets. It pairs them up and applies bias correction
        using the historical data for training.

        Parameters
        ----------
        result : Dict[str, Union[xr.Dataset, xr.DataArray]]
            Dictionary of datasets/dataarrays
        context : Dict[str, Any]
            Processing context

        Returns
        -------
        Dict[str, Union[xr.Dataset, xr.DataArray]]
            Dictionary of bias-corrected datasets
        """
        logger.debug("Loading station data from HadISD...")
        station_ds = self._load_station_data()
        logger.debug("Station data loaded. Variables: %s", list(station_ds.data_vars))

        ret = {}
        for key, data in result.items():
            historical_da = None

            if "ssp" in key:
                # Find corresponding historical data
                hist_key = re.sub(r"ssp.{3}", "historical", key)
                if hist_key in result:
                    historical_da = result[hist_key]
                    # Convert to DataArray if needed
                    if isinstance(historical_da, xr.Dataset):
                        # Preserve attributes from Dataset when extracting DataArray
                        historical_da_ds = historical_da
                        historical_da = historical_da_ds[
                            list(historical_da_ds.data_vars)[0]
                        ]
                        historical_da.attrs.update(historical_da_ds.attrs)
                else:
                    logger.warning(
                        f"No historical data found for {key} (expected {hist_key}). "
                        "Using SSP data itself for training (suboptimal)."
                    )
            elif "historical" in key:
                # Use itself as historical training data
                historical_da = data
                if isinstance(historical_da, xr.Dataset):
                    # Preserve attributes from Dataset when extracting DataArray
                    historical_da_ds = historical_da
                    historical_da = historical_da_ds[
                        list(historical_da_ds.data_vars)[0]
                    ]
                    historical_da.attrs.update(historical_da_ds.attrs)

            # Process
            ret[key] = self._process_single_dataset(
                data, station_ds, context, historical_da
            )

        return ret

    def execute(
        self,
        result: Union[
            xr.Dataset,
            xr.DataArray,
            Iterable[Union[xr.Dataset, xr.DataArray]],
            Dict[str, Union[xr.Dataset, xr.DataArray]],
        ],
        context: Dict[str, Any],
    ) -> Union[
        xr.Dataset,
        xr.DataArray,
        Iterable[Union[xr.Dataset, xr.DataArray]],
        Dict[str, Union[xr.Dataset, xr.DataArray]],
    ]:
        """Apply station bias correction to gridded climate data.

        This method orchestrates the complete bias correction workflow:
        1. Validates input data type (must be DataArray, Dataset, or Dict)
        2. Loads HadISD station observational data
        3. Applies bias correction to each station using xarray.map
        4. Returns dataset with bias-corrected data at station locations

        The output will have stations as data variables in the returned dataset,
        with each variable containing bias-corrected time series for that location.

        Parameters
        ----------
        result : xr.Dataset or xr.DataArray or Dict or Iterable
            Gridded climate model data to be bias-corrected. Can be:
            - xr.DataArray with the climate variable
            - xr.Dataset (will extract first data variable)
            - Dict of Datasets/DataArrays (pre-concatenation)
            Must have time dimension covering at least the historical period (1980-2014)
            for training the bias correction.
        context : dict
            Processing context dictionary. Updated with information about the
            bias correction operation.

        Returns
        -------
        xr.Dataset or Dict
            Bias-corrected data at station locations.

        Raises
        ------
        TypeError
            If input data is not an xr.DataArray, xr.Dataset, or Dict
        ValueError
            If input Dataset has no data variables or doesn't contain required time dimension

        Notes
        -----
        - Input data must include historical period (1980-2014) for bias correction training
        - Station observational data is available through 2014-08-31
        - All data is converted to noleap calendar for consistency
        - Final output is time-sliced to the user's requested period
        """
        if isinstance(result, dict):
            return self._execute_dict(result, context)

        # Load station observational data from HadISD
        logger.debug("Loading station data from HadISD...")
        station_ds = self._load_station_data()
        logger.debug("Station data loaded. Variables: %s", list(station_ds.data_vars))

        if isinstance(result, (xr.Dataset, xr.DataArray)):
            return self._process_single_dataset(result, station_ds, context)
        else:
            raise TypeError(
                f"StationBiasCorrection requires xr.DataArray, xr.Dataset, or Dict input, "
                f"got {type(result)}"
            )

    def update_context(self, context: Dict[str, Any]):
        """Update the context with information about the bias correction operation.

        This method adds metadata about the bias correction to the processing context,
        documenting the stations processed, historical training period, and QDM parameters used.

        Parameters
        ----------
        context : dict[str, Any]
            Processing context dictionary. Updated in place with bias correction metadata.

        Returns
        -------
        None
        """
        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        # Build informative context message
        station_list = ", ".join(self.stations)
        context[_NEW_ATTRS_KEY][self.name] = (
            f"Station bias correction applied using Quantile Delta Mapping (QDM). "
            f"Stations: {station_list}. "
            f"Historical training period: {self.historical_slice[0]}-{self.historical_slice[1]}. "
            f"QDM parameters: window={self.window} days, "
            f"nquantiles={self.nquantiles}, group='{self.group}', kind='{self.kind}'. "
            f"Observational data from HadISD weather stations."
        )

    def set_data_accessor(self, catalog: DataCatalog):
        """Set the data catalog accessor for the processor.

        The processor requires access to station metadata through the DataCatalog.
        Station observational data is loaded directly from S3 zarr stores.

        Parameters
        ----------
        catalog : DataCatalog
            Data catalog for accessing station metadata.

        Returns
        -------
        None
        """
        self.catalog = catalog
