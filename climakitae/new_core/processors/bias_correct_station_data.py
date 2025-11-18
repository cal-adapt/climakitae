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
from functools import partial
from typing import Any, Dict, Iterable, Union

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


@register_processor("bias_correct_station_data", priority=150)
class StationBiasCorrection(DataProcessor):
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
        self.name = "bias_correct_station_data"
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
        output_slice: tuple[int, int],
    ) -> xr.DataArray:
        """Apply Quantile Delta Mapping bias correction to model data.

        This method performs the core bias correction by:
        1. Converting units to match gridded data
        2. Rechunking data (QDM requires unchunked time dimension)
        3. Converting calendars to noleap for consistency
        4. Training QDM on historical overlap period (1980-2014)
        5. Applying correction to requested output slice

        Parameters
        ----------
        obs_da : xr.DataArray
            Observational station data (preprocessed)
        gridded_da : xr.DataArray
            Climate model gridded data
        output_slice : tuple[int, int]
            Start and end years for output (extracted from input data time range)

        Returns
        -------
        xr.DataArray
            Bias-corrected data for the requested output slice
        """
        logger.debug("=== Starting bias correction for station: %s ===", obs_da.name)
        logger.debug(
            "Input gridded_da time range: %s to %s",
            gridded_da.time.values[0],
            gridded_da.time.values[-1],
        )
        logger.debug(
            "Input obs_da time range: %s to %s",
            obs_da.time.values[0],
            obs_da.time.values[-1],
        )
        logger.debug(
            "Output slice requested: %s to %s", output_slice[0], output_slice[1]
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

        # Convert calendar to no leap year
        logger.debug("Converting calendars to noleap")
        logger.debug("Before conversion - obs_da time dtype: %s", obs_da.time.dtype)
        logger.debug(
            "Before conversion - gridded_da time dtype: %s", gridded_da.time.dtype
        )
        obs_da = obs_da.convert_calendar("noleap")
        gridded_da = gridded_da.convert_calendar("noleap")
        logger.debug("After conversion - obs_da time dtype: %s", obs_da.time.dtype)
        logger.debug(
            "After conversion - gridded_da time dtype: %s", gridded_da.time.dtype
        )

        # Data at the desired output slice (final output period)
        data_sliced = gridded_da.sel(
            time=slice(str(output_slice[0]), str(output_slice[1]))
        )
        logger.debug(
            "Data sliced for output: %s to %s (shape: %s)",
            data_sliced.time.values[0],
            data_sliced.time.values[-1],
            data_sliced.shape,
        )

        # Slice gridded data to match obs data period (legacy approach)
        # This ensures we only use the overlapping historical period
        gridded_da_historical = gridded_da.sel(
            time=slice(str(obs_da.time.values[0]), str(obs_da.time.values[-1]))
        )
        logger.debug(
            "Gridded historical period: %s to %s (shape: %s)",
            gridded_da_historical.time.values[0],
            gridded_da_historical.time.values[-1],
            gridded_da_historical.shape,
        )

        # Now slice obs data to match the gridded historical data exactly
        # This handles any edge cases where times don't align perfectly
        obs_da = obs_da.sel(
            time=slice(
                str(gridded_da_historical.time.values[0]),
                str(gridded_da_historical.time.values[-1]),
            )
        )
        logger.debug(
            "Final obs period after alignment: %s to %s (shape: %s)",
            obs_da.time.values[0],
            obs_da.time.values[-1],
            obs_da.shape,
        )

        # Check if data has a 'sim' dimension from concatenation
        # QDM must be trained and applied separately for each simulation
        if "sim" in data_sliced.dims:
            logger.debug(
                "Data has 'sim' dimension, training and applying QDM for each simulation separately"
            )
            sim_results = []

            for sim_name in data_sliced.sim.values:
                logger.debug(f"Processing simulation: {sim_name}")

                # Select and DROP the sim coordinate to get pure time-series
                sim_data_full = data_sliced.sel(sim=sim_name, drop=True)
                sim_hist = gridded_da_historical.sel(sim=sim_name, drop=True)

                logger.debug(
                    "Selected sim=%s, full_shape=%s, hist_shape=%s",
                    sim_name,
                    sim_data_full.shape,
                    sim_hist.shape,
                )

                # Train QDM for this simulation
                logger.debug(
                    "Training QDM for %s with nquantiles=%s, kind=%s",
                    sim_name,
                    self.nquantiles,
                    self.kind,
                )
                sim_QDM = QuantileDeltaMapping.train(
                    obs_da,
                    sim_hist,
                    nquantiles=self.nquantiles,
                    group=grouper,
                    kind=self.kind,
                )

                # Apply QDM to this simulation
                logger.debug("Applying QDM adjustment to %s", sim_name)
                sim_adj = sim_QDM.adjust(sim_data_full)

                # Compute immediately with cftime coordinates
                logger.debug("Computing QDM result for %s", sim_name)
                sim_adj = sim_adj.compute()

                # Convert calendar to standard datetime64
                logger.debug("Converting calendar for %s", sim_name)
                sim_adj = sim_adj.convert_calendar("standard", use_cftime=False)

                # Expand dims to add sim coordinate back
                sim_adj = sim_adj.expand_dims(sim=[sim_name])
                sim_results.append(sim_adj)
                logger.debug("Completed bias correction for %s", sim_name)

            # Concatenate all simulations back together
            logger.debug("Concatenating %d simulations", len(sim_results))
            da_adj = xr.concat(sim_results, dim="sim")
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
            logger.debug(
                "data_sliced time dtype before QDM: %s", data_sliced.time.dtype
            )
            logger.debug(
                "data_sliced is dask array: %s", hasattr(data_sliced.data, "dask")
            )

            da_adj = QDM.adjust(data_sliced)
            da_adj.name = gridded_da_historical.name

            logger.debug("QDM adjustment complete (lazy)")
            logger.debug("Computing QDM result with cftime coordinates...")
            da_adj = da_adj.compute()

            logger.debug("Compute complete")
            logger.debug("Converting calendar from noleap to standard (datetime64)")
            da_adj = da_adj.convert_calendar("standard", use_cftime=False)

        logger.debug("Calendar conversion complete")
        logger.debug("da_adj time dtype after conversion: %s", da_adj.time.dtype)
        logger.debug("da_adj time index type: %s", type(da_adj.indexes["time"]))
        logger.debug("First few time values: %s", da_adj.time.values[:3])

        # Ensure time coordinate is datetime64[ns] for plotting compatibility
        # Check if time is not already datetime64 (check dtype string representation)
        time_dtype_str = str(da_adj.time.dtype)
        if 'datetime64' not in time_dtype_str:
            logger.debug("Converting time from %s to datetime64[ns]", da_adj.time.dtype)
            try:
                # Convert using pandas DatetimeIndex
                da_adj["time"] = pd.DatetimeIndex(da_adj.time.values)
                logger.debug("Time conversion complete, new dtype: %s", da_adj.time.dtype)
            except Exception as e:
                logger.error("Failed to convert time coordinate: %s", e)
                raise

        # Rechunk to convert back to dask array for downstream processing
        # This maintains lazy evaluation for subsequent operations
        logger.debug("Rechunking to create dask array for downstream processing")
        da_adj = da_adj.chunk({"time": "auto"})
        logger.debug("da_adj is dask array: %s", hasattr(da_adj.data, "dask"))
        logger.debug("=== Bias correction complete for %s ===", da_adj.name)

        return da_adj  # type: ignore[return-value]

    def _get_bias_corrected_closest_gridcell(
        self,
        station_da: xr.DataArray,
        gridded_da: xr.DataArray,
        output_slice: tuple[int, int],
    ) -> xr.DataArray:
        """Get closest gridcell to station and apply bias correction.

        This method:
        1. Extracts station coordinates from attributes
        2. Finds closest gridcell in climate model data
        3. Drops unnecessary coordinates for cleaner merging
        4. Applies bias correction
        5. Adds station metadata to output

        Parameters
        ----------
        station_da : xr.DataArray
            Observational station data
        gridded_da : xr.DataArray
            Climate model gridded data
        output_slice : tuple[int, int]
            Start and end years for output (extracted from input data time range)

        Returns
        -------
        xr.DataArray
            Bias-corrected data with station metadata
        """
        # Get the closest gridcell to the station
        station_lat, station_lon = station_da.attrs["coordinates"]
        gridded_da_closest_gridcell = get_closest_gridcell(
            gridded_da, station_lat, station_lon, print_coords=False
        )

        # Validate we got a result
        if gridded_da_closest_gridcell is None:
            raise ValueError(
                f"Could not find closest gridcell for station at "
                f"({station_lat}, {station_lon})"
            )

        # Drop any coordinates in the output dataset that are not also dimensions
        # This makes merging all the stations together easier and drops superfluous coordinates
        gridded_da_closest_gridcell = gridded_da_closest_gridcell.drop_vars(  # type: ignore[union-attr]
            [
                i
                for i in gridded_da_closest_gridcell.coords  # type: ignore[union-attr]
                if i not in gridded_da_closest_gridcell.dims  # type: ignore[union-attr]
            ]
        )

        # Bias correct the model data using the station data
        # Output data will be cut to the requested output slice
        bias_corrected = self._bias_correct_model_data(
            station_da, gridded_da_closest_gridcell, output_slice
        )

        # Add descriptive coordinates to the bias corrected data
        bias_corrected.attrs["station_coordinates"] = station_da.attrs[
            "coordinates"
        ]  # Coordinates of station
        bias_corrected.attrs["station_elevation"] = station_da.attrs[
            "elevation"
        ]  # Elevation of station

        return bias_corrected

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """Apply station bias correction to gridded climate data.

        This method orchestrates the complete bias correction workflow:
        1. Validates input data type (must be DataArray)
        2. Loads HadISD station observational data
        3. Applies bias correction to each station using xarray.map
        4. Returns dataset with bias-corrected data at station locations

        The output will have stations as data variables in the returned dataset,
        with each variable containing bias-corrected time series for that location.

        Parameters
        ----------
        result : xr.Dataset or xr.DataArray or Iterable of these
            Gridded climate model data to be bias-corrected. Can be either:
            - xr.DataArray with the climate variable
            - xr.Dataset (will extract first data variable)
            Must have time dimension covering at least the historical period (1980-2014)
            for training the bias correction.
        context : dict
            Processing context dictionary. Updated with information about the
            bias correction operation.

        Returns
        -------
        xr.Dataset
            Bias-corrected data at station locations. The returned Dataset will
            have stations as data variables, with station metadata preserved in
            attributes.

        Raises
        ------
        TypeError
            If input data is not an xr.DataArray or xr.Dataset
        ValueError
            If input Dataset has no data variables or doesn't contain required time dimension

        Notes
        -----
        - Input data must include historical period (1980-2014) for bias correction training
        - Station observational data is available through 2014-08-31
        - All data is converted to noleap calendar for consistency
        - Final output is time-sliced to the user's requested period
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
            if "resolution" in result.attrs:
                result_da.attrs["resolution"] = result.attrs["resolution"]
            logger.info("Converted Dataset to DataArray: %s", result_da.name)
        elif isinstance(result, xr.DataArray):
            result_da = result
        else:
            raise TypeError(
                f"StationBiasCorrection requires xr.DataArray or xr.Dataset input, "
                f"got {type(result)}"
            )

        logger.info(
            "Applying station bias correction for %d station(s)", len(self.stations)
        )
        logger.debug("Input result_da name: %s", result_da.name)
        logger.debug("Input result_da shape: %s", result_da.shape)
        logger.debug("Input result_da dims: %s", result_da.dims)
        logger.debug("Input result_da time dtype: %s", result_da.time.dtype)
        logger.debug("Input result_da is dask: %s", hasattr(result_da.data, "dask"))

        # Load station observational data from HadISD
        logger.debug("Loading station data from HadISD...")
        station_ds = self._load_station_data()
        logger.debug("Station data loaded. Variables: %s", list(station_ds.data_vars))

        # Extract time range from input data for output
        # The TimeSlice processor (if used) will have already sliced the data
        time_values = result_da.time.values
        output_start_year = int(str(time_values[0])[:4])
        output_end_year = int(str(time_values[-1])[:4])
        logger.debug("Output time range: %s to %s", output_start_year, output_end_year)

        # Apply bias correction to each station using xarray map
        # This processes all stations and combines results
        logger.debug("Applying bias correction via xarray.map...")
        apply_output = station_ds.map(
            self._get_bias_corrected_closest_gridcell,
            keep_attrs=False,
            gridded_da=result_da,
            output_slice=(output_start_year, output_end_year),
        )

        logger.info(
            "Station bias correction complete. Output shape: %s", apply_output.dims
        )
        logger.debug("Output variables: %s", list(apply_output.data_vars))
        logger.debug("Output time dtype: %s", apply_output.time.dtype)
        logger.debug(
            "Output is dask: %s",
            any(
                hasattr(apply_output[var].data, "dask")
                for var in apply_output.data_vars
            ),
        )

        # Final check: Ensure time coordinate is datetime64[ns] for plotting
        time_dtype_str = str(apply_output.time.dtype)
        if 'datetime64' not in time_dtype_str:
            logger.debug("Final conversion: time from %s to datetime64[ns]", apply_output.time.dtype)
            apply_output["time"] = pd.DatetimeIndex(apply_output.time.values)
            logger.debug("Final time dtype: %s", apply_output.time.dtype)

        return apply_output

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
