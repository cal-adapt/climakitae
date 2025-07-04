"""
Localize gridded climate data to weather station locations.

This module provides the Localize processor that extracts climate data at specific
weather station locations from gridded datasets. The processor supports both
simple nearest-neighbor localization and advanced bias correction techniques
using historical observational data.

The module implements quantile delta mapping (QDM) bias correction using xclim
to adjust model data based on historical station observations. Station metadata
is loaded from CSV files and observational data is accessed from S3 storage.

Functions
---------
This module defines the Localize class which is registered as a data processor
with the new core architecture.

Classes
-------
Localize : DataProcessor
    A processor that localizes gridded climate data to weather station points
    with optional bias correction capabilities.

Notes
-----
This processor requires access to station metadata CSV files and S3-hosted
observational data for bias correction functionality. The bias correction
implementation uses the xclim library for quantile delta mapping.
"""

import re
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import xarray as xr
from xclim.sdba import Grouper, QuantileDeltaMapping

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.new_core.processors.processor_utils import extend_time_domain
from climakitae.util.utils import get_closest_gridcell


@register_processor("localize", priority=155)
class Localize(DataProcessor):
    """
    Localize gridded climate data to historical station data.

    This processor extracts climate data from gridded datasets at specific weather
    station locations using nearest-neighbor spatial interpolation. It supports
    optional bias correction using quantile delta mapping with historical
    observational data from weather stations.

    Parameters
    ----------
    value : str or list of str or dict
        Station specification. Can be:
        - Single station name (str)
        - List of station names (list of str)
        - Dictionary with configuration options including:
            - 'stations': list of station names
            - 'bias_correction': bool, whether to apply bias correction
            - 'method': str, bias correction method ('quantile_delta_mapping')
            - 'window': int, window size for seasonal grouping (default 30)
            - 'nquantiles': int, number of quantiles for QDM (default 100)

    Attributes
    ----------
    name : str
        Processor name identifier ('localize')
    stations : list of str
        List of station names to process
    bias_correction : bool
        Whether bias correction is enabled
    method : str
        Bias correction method to use
    window : int
        Window size in days for seasonal grouping in bias correction
    nquantiles : int
        Number of quantiles for quantile delta mapping
    station_metadata : list or None
        Collected metadata for processed stations

    Raises
    ------
    ValueError
        If the value parameter is not a valid station specification format
        If specified stations are not found in the metadata
        If bias correction is requested but station data is unavailable

    Notes
    -----
    The processor uses HadISD (Hadley Integrated Surface Database) observational
    data for bias correction when enabled. Station metadata is loaded from CSV
    files containing station coordinates and identifiers.

    The bias correction uses xclim's QuantileDeltaMapping implementation with
    seasonal grouping to adjust model data based on the relationship between
    historical model and observational data.

    When bias correction is applied, the output contains the full time range with:
    - Historical period (up to 2014-08-31): Original uncorrected data
    - Future period (from 2014-09-01): Bias-corrected data

    This ensures that downstream time slicing operations work correctly with
    the complete time series while applying bias correction only where appropriate.
    """

    def __init__(
        self,
        value: str | List[str] | Dict[str, Any],
    ):
        """
        Initialize the Localize processor.

        Parameters
        ----------
        value : str or list of str or dict
            Station specification. Can be:
            - Single station name (str)
            - List of station names (list of str)
            - Dictionary with configuration options including:
                - 'stations': list of station names
                - 'bias_correction': bool, whether to apply bias correction
                - 'method': str, bias correction method ('quantile_delta_mapping')
                - 'window': int, window size for seasonal grouping (default 30)
                - 'nquantiles': int, number of quantiles for QDM (default 100)

        Raises
        ------
        ValueError
            If value is not a string, list, or dictionary with valid configuration
        """
        self.name = "localize"
        match value:
            case str() | list():
                self.stations = [value] if isinstance(value, str) else value
                self.bias_correction = True
                self.method = "quantile_delta_mapping"
                self.window = 90
                self.nquantiles = 20
            case dict():
                self.stations = value.get("stations", [])
                self.bias_correction = value.get("bias_correction", True)
                self.method = value.get("method", "quantile_delta_mapping")
                self.window = value.get("window", 90)
                self.nquantiles = value.get("nquantiles", 20)
            case _:
                raise ValueError(
                    "Invalid value for Localize processor. Expected a station name, list of names, or a dictionary with parameters."
                )

        self._stations_gdf = self._load_station_metadata()
        self.station_metadata = None

    def _load_station_metadata(self) -> gpd.GeoDataFrame:
        """
        Load station metadata and create GeoDataFrame with spatial information.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing station metadata with geometry points
            created from latitude and longitude coordinates. Uses EPSG:4326
            coordinate reference system.

        Notes
        -----
        Reads station metadata from the CSV file specified in STATIONS_CSV_PATH
        and creates Point geometries from LON_X and LAT_Y columns.
        """
        return DataCatalog()["stations"]

    def _load_all_station_data(
        self, station_subset: gpd.GeoDataFrame
    ) -> Dict[str, xr.DataArray]:
        """
        OPTIMIZATION: Load all station data at once using bulk operations.

        This replaces individual S3 calls with a single bulk operation,
        similar to the original implementation's approach.

        Parameters
        ----------
        station_subset : gpd.GeoDataFrame
            Subset of stations to load data for

        Returns
        -------
        Dict[str, xr.DataArray]
            Dictionary mapping station names to their observational data
        """
        # Get all file paths for the stations
        filepaths = [
            f"s3://cadcat/hadisd/HadISD_{station_id}.zarr"
            for station_id in station_subset["station id"]
        ]

        # Create preprocessing function
        def _preprocess_hadisd_bulk(
            ds: xr.Dataset, station_subset: gpd.GeoDataFrame
        ) -> xr.Dataset:
            """Preprocess station data for bulk loading"""
            # Extract station ID from file path
            station_id = int(
                ds.encoding["source"].split("HadISD_")[1].split(".zarr")[0]
            )

            # Get station info
            station_info = station_subset[
                station_subset["station id"] == station_id
            ].iloc[0]
            station_name = station_info["station"]

            # Convert and rename
            ds = ds.rename({"tas": station_name})
            ds[station_name] = ds[station_name] + 273.15  # Convert C to K

            # Add metadata
            ds[station_name] = ds[station_name].assign_attrs(
                {
                    "coordinates": (
                        ds.latitude.values.item(),
                        ds.longitude.values.item(),
                    ),
                    "elevation": f"{ds.elevation.item()} {ds.elevation.attrs['units']}",
                    "units": "K",
                }
            )

            # Clean up coordinates
            ds = ds.drop_vars(["elevation", "latitude", "longitude"], errors="ignore")
            return ds

        try:
            # Use bulk loading with parallel processing like the original implementation
            _partial_func = partial(
                _preprocess_hadisd_bulk, station_subset=station_subset
            )

            station_ds = xr.open_mfdataset(
                filepaths,
                preprocess=_partial_func,
                engine="zarr",
                consolidated=False,
                parallel=True,
                backend_kwargs=dict(storage_options={"anon": True}),
            )

            # Convert to dictionary for easier access
            station_data_dict = {}
            for var_name in station_ds.data_vars:
                station_data_dict[var_name] = station_ds[var_name]

            return station_data_dict

        except Exception as e:
            print(f"Warning: Bulk station data loading failed: {e}")
            print("Falling back to individual loading...")
            return {}

    def _process_all_stations_optimized(
        self,
        gridded_data: Union[xr.DataArray, xr.Dataset],
        station_subset: gpd.GeoDataFrame,
        context: Dict[str, Any],
        station_data_dict: Optional[Dict[str, xr.DataArray]] = None,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        OPTIMIZATION: Process all stations with vectorized operations where possible.

        Parameters
        ----------
        gridded_data : xr.DataArray or xr.Dataset
            Input gridded data
        station_subset : gpd.GeoDataFrame
            Stations to process
        context : Dict[str, Any]
            Processing context
        station_data_dict : Dict[str, xr.DataArray], optional
            Pre-loaded station data for bias correction

        Returns
        -------
        Union[xr.DataArray, xr.Dataset]
            Combined results for all stations
        """
        station_results = []

        for _, station_row in station_subset.iterrows():
            station_name = station_row["station"]
            station_lat = station_row["LAT_Y"]
            station_lon = station_row["LON_X"]

            try:
                # Get closest grid cell (this is the main bottleneck that can't be easily vectorized)
                closest_data = get_closest_gridcell(
                    gridded_data, station_lat, station_lon, print_coords=False
                )

                if closest_data is None:
                    continue

                # Ensure closest_data is a DataArray for bias correction
                if isinstance(closest_data, xr.Dataset):
                    # Get the first data variable from the Dataset
                    data_vars = list(closest_data.data_vars.keys())
                    if data_vars:
                        closest_data = closest_data[data_vars[0]]
                    else:
                        print(
                            f"Warning: No data variables found in Dataset for station {station_name}"
                        )
                        continue

                # Drop superfluous coordinates
                closest_data = closest_data.drop_vars(
                    [
                        coord
                        for coord in closest_data.coords
                        if coord not in closest_data.dims
                    ],
                    errors="ignore",
                )

                # Apply bias correction if available
                if (
                    self.bias_correction
                    and station_data_dict
                    and station_name in station_data_dict
                ):
                    closest_data = self._bias_correct_model_data_optimized(
                        station_data_dict[station_name], closest_data, context
                    )
                elif self.bias_correction:
                    # Fallback to individual loading if bulk loading failed
                    station_data = self._load_station_data(
                        station_row["station id"], station_row
                    )
                    if station_data is not None:
                        closest_data = self._bias_correct_model_data_optimized(
                            station_data, closest_data, context
                        )

                # Ensure calendar compatibility for downstream processing
                if hasattr(closest_data, "time") and hasattr(closest_data.time, "dt"):
                    calendar_type = getattr(
                        closest_data.time.dt, "calendar", "standard"
                    )
                    if calendar_type == "noleap":
                        try:
                            closest_data = closest_data.convert_calendar(
                                "standard", use_cftime=False
                            )
                        except Exception as e:
                            print(
                                f"Warning: Could not convert calendar for station {station_name}: {e}"
                            )

                # Add metadata
                closest_data.attrs["station_coordinates"] = (station_lat, station_lon)
                closest_data.attrs["station_elevation"] = station_row.get(
                    "elevation", "unknown"
                )
                closest_data.attrs["name"] = station_name

                # Store metadata
                if self.station_metadata is None:
                    self.station_metadata = []
                self.station_metadata.append(self._get_station_metadata(station_row))

                station_results.append(closest_data)

            except Exception as e:
                print(f"Warning: Failed to process station {station_name}: {e}")
                continue

        if not station_results:
            raise ValueError("No valid station data could be processed")

        return self._combine_station_results(station_results)

    def _validate_and_convert_stations(self):
        """
        Validate that all specified stations exist and convert airport codes to station names.

        Raises
        ------
        ValueError
            If specified stations are not found in metadata
            If airport codes are ambiguous or not found
        """
        available_stations = self._stations_gdf["station"].tolist()
        invalid_stations = []

        for i, s in enumerate(self.stations):
            if len(s) == 4:
                # This is a 4 letter airport code, convert to station name
                mask = self._stations_gdf["station id"].str.contains(s, na=False)
                if not mask.any():
                    raise ValueError(f"Station ID '{s}' not found in metadata")
                elif sum(mask) > 1:
                    raise ValueError(f"Multiple stations found for ID '{s}'")
                else:
                    self.stations[i] = self._stations_gdf[mask]["station"].values[0]
            else:
                if s not in available_stations:
                    invalid_stations.append(s)

        if invalid_stations:
            raise ValueError(f"Invalid stations: {invalid_stations}")

    def _validate_inputs_dict(self, data: Dict[str, Union[xr.DataArray, xr.Dataset]]):
        """
        Validate dictionary inputs for station processing.

        Parameters
        ----------
        data : Dict[str, Union[xr.DataArray, xr.Dataset]]
            Dictionary of data to validate

        Raises
        ------
        ValueError
            If any data lacks required 'lat' and 'lon' coordinates
            If any data is not a DataArray or Dataset
        """
        print(f"Localizing data to stations {self.stations}...")
        for key, value in data.items():
            if not isinstance(value, (xr.DataArray, xr.Dataset)):
                raise ValueError(
                    f"Invalid data type for key '{key}': expected DataArray or Dataset"
                )
            if not hasattr(value, "lat") or not hasattr(value, "lon"):
                raise ValueError(
                    f"Data for key '{key}' must have 'lat' and 'lon' coordinates"
                )

    def execute(
        self,
        result: Union[
            xr.DataArray, xr.Dataset, Dict[str, Union[xr.DataArray, xr.Dataset]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.DataArray, xr.Dataset, Dict[str, Union[xr.DataArray, xr.Dataset]]]:
        """
        Execute station localization on the input gridded data.

        This method processes gridded climate data to extract values at specified
        weather station locations. It performs validation, spatial extraction,
        optional bias correction, and metadata management.

        Parameters
        ----------
        result : xr.DataArray or xr.Dataset
            Gridded climate data to localize. Must contain 'lat' and 'lon'
            coordinates for spatial processing.
        context : dict of str to Any
            Processing context dictionary that stores metadata and configuration
            information throughout the processing pipeline.

        Returns
        -------
        xr.DataArray or xr.Dataset
            Station-localized data as a Dataset with each station as a separate
            data variable. Contains station coordinates, elevation, and processing
            metadata as attributes.

        Raises
        ------
        ValueError
            If input data lacks required 'lat' and 'lon' coordinates
            If no valid station data could be processed
            If specified stations are not found in metadata

        Notes
        -----
        The method extracts the original time slice information for bias correction
        and processes each station individually before combining results into a
        unified Dataset structure.
        """

        ret = {}
        print(result)

        match result:
            case xr.DataArray() | xr.Dataset():
                # Get station subset
                station_subset = self._get_station_subset()

                # Store original time slice for bias correction if needed
                if hasattr(result, "time") and len(result.time) > 0:
                    time_values = result.time.values
                    original_time_slice = (
                        int(str(time_values[0])[:4]),  # First year
                        int(str(time_values[-1])[:4]),  # Last year
                    )
                    context["original_time_slice"] = original_time_slice

                # OPTIMIZATION: Load all station data at once if bias correction is enabled
                station_data_dict = None
                if self.bias_correction:
                    station_data_dict = self._load_all_station_data(station_subset)

                # Process all stations efficiently
                combined = self._process_all_stations_optimized(
                    result, station_subset, context, station_data_dict
                )

                return combined

            case dict():
                self._validate_inputs_dict(result)
                result = extend_time_domain(result)
                for key, item in result.items():
                    if "ssp" not in str(key):
                        continue  # Skip non-SSP data
                    ret[key] = self.execute(item, context)
                    ret[key].attrs = item.attrs

        self.update_context(context)
        return ret

    def _get_station_subset(self) -> gpd.GeoDataFrame:
        """
        Get subset of station metadata for requested stations.

        Filters the complete station metadata GeoDataFrame to include only
        the stations specified in self.stations.

        Returns
        -------
        gpd.GeoDataFrame
            Filtered GeoDataFrame containing metadata only for the requested
            stations, with the same structure as the complete metadata.

        Notes
        -----
        Uses pandas DataFrame filtering to select rows where the 'station'
        column matches any of the station names in self.stations.
        """
        return self._stations_gdf[self._stations_gdf["station"].isin(self.stations)]

    def set_data_accessor(self, catalog: DataCatalog):
        """
        Set the data accessor for the processor.

        Parameters
        ----------
        catalog : DataCatalog
            Data catalog instance for accessing datasets.

        Notes
        -----
        This is a placeholder method for compatibility with the DataProcessor
        interface. Currently not implemented for the Localize processor.
        """

    def _load_station_data(
        self, station_id: int, station_row: pd.Series
    ) -> Optional[xr.DataArray]:
        """
        Load and preprocess observational station data for bias correction.

        Loads HadISD (Hadley Integrated Surface Database) station data from
        S3 storage and preprocesses it for use in bias correction.

        Parameters
        ----------
        station_id : int
            Numerical station identifier used to construct the data file path
        station_row : pd.Series
            Station metadata row containing station information

        Returns
        -------
        xr.DataArray or None
            Preprocessed station data ready for bias correction, or None if
            loading fails due to missing files or other errors

        Notes
        -----
        Station data is accessed from S3 using anonymous access. The data is
        stored in Zarr format at paths following the pattern:
        's3://cadcat/hadisd/HadISD_{station_id}.zarr'

        If data loading fails, a warning is printed and None is returned,
        allowing the processing to continue without bias correction.
        """

        filepath = f"s3://cadcat/hadisd/HadISD_{station_id}.zarr"

        try:
            # Load station data
            station_ds = xr.open_zarr(
                filepath, storage_options={"anon": True}, consolidated=False
            )

            # Preprocess station data
            station_data = self._preprocess_hadisd(station_ds, station_row)

            return station_data

        except (FileNotFoundError, OSError, ValueError, KeyError) as e:
            print(
                f"Warning: Failed to load station data for station {station_row['station']}: {e}"
            )
            return None

    def _preprocess_hadisd(
        self, ds: xr.Dataset, station_row: pd.Series
    ) -> xr.DataArray:
        """
        Preprocess HadISD station data for bias correction.

        Converts temperature from Celsius to Kelvin, assigns station metadata
        as attributes, and cleans up unnecessary coordinate variables.

        Parameters
        ----------
        ds : xr.Dataset
            Raw HadISD dataset loaded from storage containing temperature
            data and metadata
        station_row : pd.Series
            Station metadata row containing station name and other information

        Returns
        -------
        xr.DataArray
            Preprocessed station temperature data with:
            - Temperature converted from Celsius to Kelvin
            - Station coordinates and elevation as attributes
            - Unnecessary coordinate variables removed
            - Station name assigned as the DataArray name

        Notes
        -----
        The preprocessing assumes the temperature variable is named 'tas' in
        the input dataset. The conversion adds 273.15 to convert from Celsius
        to Kelvin to match typical model data units.
        """

        station_name = station_row["station"]

        # Rename data variable to station name and convert C to K
        station_data = ds["tas"] + 273.15
        station_data.name = station_name

        # Assign descriptive attributes
        station_data = station_data.assign_attrs(
            {
                "coordinates": (
                    ds.latitude.values.item(),
                    ds.longitude.values.item(),
                ),
                "elevation": f"{ds.elevation.item()} {ds.elevation.attrs['units']}",
                "units": "K",
            }
        )

        # Drop unnecessary coordinates except time
        station_data = station_data.drop_vars(
            ["elevation", "latitude", "longitude"], errors="ignore"
        )

        return station_data

    def _bias_correct_model_data_optimized(
        self,
        obs_da: xr.DataArray,
        gridded_da: xr.DataArray,
        context: Dict[str, Any],
    ) -> xr.DataArray:
        """
        OPTIMIZATION: Simplified bias correction with reduced overhead.

        This version eliminates redundant calendar conversions and unit handling
        that were causing performance issues in the original implementation.

        Parameters
        ----------
        obs_da : xr.DataArray
            Station observational data (already preprocessed)
        gridded_da : xr.DataArray
            Model gridded data
        context : Dict[str, Any]
            Processing context containing time slice information

        Returns
        -------
        xr.DataArray
            Bias corrected data
        """
        print("INFO: running optimized bias correction")
        try:
            # Get grouper
            grouper = Grouper("time.dayofyear", window=self.window)

            # Ensure units match (obs_da should already be in Kelvin from preprocessing)
            target_units = gridded_da.attrs.get("units", "K")
            if obs_da.attrs.get("units") != target_units:
                obs_da = obs_da.copy()
                obs_da.attrs["units"] = target_units

            # Rechunk for xclim compatibility
            gridded_da = gridded_da.chunk(chunks=dict(time=-1))
            obs_da = obs_da.chunk(chunks=dict(time=-1))

            # Limit obs data to available period
            obs_da = obs_da.sel(time=slice(obs_da.time.values[0], "2014-08-31"))

            # Check if we need calendar conversion based on the gridded data's calendar
            gridded_calendar = getattr(gridded_da.time.dt, "calendar", "standard")
            obs_calendar = getattr(obs_da.time.dt, "calendar", "standard")

            # Convert both to the same calendar if they differ
            if gridded_calendar != obs_calendar:
                if gridded_calendar == "noleap":
                    obs_da = obs_da.convert_calendar("noleap")
                else:
                    # Convert gridded data to standard calendar for compatibility
                    gridded_da = gridded_da.convert_calendar(
                        "standard", use_cftime=False
                    )

            # Split into historical and future
            historical_end = "2014-08-31"
            gridded_historical = gridded_da.sel(time=slice(None, historical_end))
            gridded_future = gridded_da.sel(time=slice("2014-09-01", None))

            if len(gridded_future.time) == 0:
                print("WARNING: No future period found for bias correction")
                return gridded_da

            # Find training overlap
            obs_start = obs_da.time.values[0]
            obs_end = obs_da.time.values[-1]
            hist_start = (
                gridded_historical.time.values[0]
                if len(gridded_historical.time) > 0
                else obs_end
            )
            hist_end = (
                gridded_historical.time.values[-1]
                if len(gridded_historical.time) > 0
                else obs_start
            )

            # Check for overlap
            if (
                obs_end < hist_start
                or obs_start > hist_end
                or len(gridded_historical.time) == 0
            ):
                print("WARNING: No temporal overlap for bias correction")
                return gridded_da

            # Extract training data
            train_start = max(obs_start, hist_start)
            train_end = min(obs_end, hist_end)

            gridded_train = gridded_historical.sel(
                time=slice(str(train_start), str(train_end))
            )
            obs_train = obs_da.sel(time=slice(str(train_start), str(train_end)))

            # Train and apply QDM
            qdm = QuantileDeltaMapping.train(
                obs_train,
                gridded_train,
                nquantiles=self.nquantiles,
                group=grouper,
                kind="+",
            )
            da_adj_future = qdm.adjust(gridded_future)
            da_adj_future.name = gridded_da.name

            # Combine with historical data
            da_combined = xr.concat([gridded_historical, da_adj_future], dim="time")  # type: ignore
            da_combined.attrs = gridded_da.attrs.copy()
            da_combined.attrs["bias_correction_applied"] = "true"

            # Ensure output calendar is compatible with downstream processing
            # Convert back to standard calendar if the original had a standard calendar
            original_calendar = getattr(
                context.get("original_data", gridded_da).time.dt, "calendar", "standard"
            )
            current_calendar = getattr(da_combined.time.dt, "calendar", "standard")

            if original_calendar == "standard" and current_calendar != "standard":
                da_combined = da_combined.convert_calendar("standard", use_cftime=False)

            return da_combined

        except (ValueError, KeyError, AttributeError, TypeError) as e:
            print(f"WARNING: Optimized bias correction failed: {e}")
            return gridded_da

    def update_context(self, context: Dict[str, Any]):
        """
        Update the processing context with localization metadata.

        Adds information about the stations processed, bias correction settings,
        and processing methods to the context for downstream processors and
        final output metadata.

        Parameters
        ----------
        context : dict of str to Any
            Processing context dictionary that accumulates metadata throughout
            the processing pipeline. Modified in place.

        Notes
        -----
        The context is updated with processor-specific metadata including:
        - List of stations processed
        - Bias correction settings and methods
        - Localization methodology
        - Station-specific metadata if available

        This method does not return anything as the context is modified in place.
        """

        processor_metadata = {
            "message": f"Process '{self.name}' applied to the data.",
            "stations_processed": self.stations,
            "bias_correction_enabled": self.bias_correction,
            "bias_correction_method": (self.method if self.bias_correction else "none"),
            "localization_method": "nearest_neighbor_grid_cell",
            "processing_note": "Gridded data localized to weather station points",
        }

        # Add station-specific metadata if available
        if self.station_metadata:
            for station, metadata in self.station_metadata:
                processor_metadata[station] = metadata

        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        context[_NEW_ATTRS_KEY][self.name] = processor_metadata

    def _get_station_metadata(
        self, station_row: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Extract station-specific metadata for inclusion in processed data.

        Creates a tuple containing the station name and a dictionary of metadata
        that will be associated with the processed station data.

        Parameters
        ----------
        station_row : pd.Series
            Row from station metadata DataFrame containing station information
            including coordinates, elevation, and identifiers

        Returns
        -------
        tuple of (str, dict)
            Tuple containing:
            - Station name (str)
            - Dictionary of station metadata including:
              - station_id: Station identifier
              - station_coordinates: (latitude, longitude) tuple
              - station_elevation: Elevation information
              - processing_method: Method used for processing
              - spatial_method: Spatial interpolation approach used

        Notes
        -----
        The spatial_method varies based on whether bias correction is enabled,
        indicating either 'nearest_neighbor' or 'bias_corrected_nearest_neighbor'.
        """

        return (
            station_row["station"],
            {
                "station_id": station_row["station id"],
                "station_coordinates": (station_row["LAT_Y"], station_row["LON_X"]),
                "station_elevation": station_row.get("elevation", "unknown"),
                "processing_method": "station_localization",
                "spatial_method": (
                    "nearest_neighbor"
                    if not self.bias_correction
                    else "bias_corrected_nearest_neighbor"
                ),
            },
        )

    def _extract_station_id(self, station_name: str) -> str:
        """
        Extract 4-letter station ID from station name string.

        Station names typically end with the format "(KXXX)" where KXXX is the 4-letter ID.

        Parameters
        ----------
        station_name : str
            Full station name like "Oxnard Ventura County Airport (KOXR)"

        Returns
        -------
        str
            4-letter station ID like "KOXR", or empty string if not found
        """

        # Look for pattern like "(KXXX)" at the end of the string
        match = re.search(r"\(([A-Z]{4})\)$", station_name)
        if match:
            return match.group(1)
        return ""

    def _combine_station_results(
        self, station_results: List[Union[xr.DataArray, xr.Dataset]]
    ) -> xr.Dataset:
        """
        Combine individual station results into a unified Dataset.

        Takes a list of DataArrays or Datasets representing processed data for individual
        stations and combines them into a single Dataset with each station
        as a separate data variable. Variable names preserve the original variable name
        and append the station ID (e.g., "t2@KOXR").

        Parameters
        ----------
        station_results : list of xr.DataArray or xr.Dataset
            List of processed DataArrays or Datasets, one for each station. Each should
            have a name attribute corresponding to the station name.

        Returns
        -------
        xr.Dataset
            Dataset containing all station data as separate variables, with
            variable names in format "original_variable@STATION_ID"

        Notes
        -----
        The station ID is extracted from the station name (text in parentheses).
        If no station ID is found, falls back to using cleaned station name.
        """

        # Create dataset with each station as a data variable
        station_dict = {}
        for station_data in station_results:
            station_name = station_data.attrs.get("name", "unknown_station")
            station_id = self._extract_station_id(station_name)

            # Handle both DataArray and Dataset cases
            if isinstance(station_data, xr.DataArray):
                # Single variable case - use original variable name with station ID
                original_var_name = getattr(station_data, "name", "data")
                if station_id:
                    var_key = f"{original_var_name}@{station_id}"
                else:
                    # Fallback to cleaned station name if no ID found
                    clean_name = (
                        station_name.replace(" ", "_").replace("(", "").replace(")", "")
                    )
                    var_key = clean_name

                station_dict[var_key] = station_data

            elif isinstance(station_data, xr.Dataset):
                # Multiple variables case - append station ID to each variable
                for var_name in station_data.data_vars:
                    if station_id:
                        var_key = f"{var_name}@{station_id}"
                    else:
                        # Fallback to station name prefix
                        clean_name = (
                            station_name.replace(" ", "_")
                            .replace("(", "")
                            .replace(")", "")
                        )
                        var_key = f"{clean_name}_{var_name}"

                    data_array = station_data[var_name]
                    # Preserve station attributes
                    data_array.attrs.update(station_data.attrs)
                    data_array.attrs["name"] = station_name
                    data_array = data_array.squeeze("member_id", drop=True)
                    station_dict[var_key] = data_array

        combined = xr.Dataset(station_dict)
        return combined

    def _simple_convert_temperature(
        self, da: xr.DataArray, target_units: str
    ) -> xr.DataArray:
        """
        Simple temperature conversion that doesn't rely on time coordinate .dt accessor.

        Parameters
        ----------
        da : xr.DataArray
            Data array to convert
        target_units : str
            Target units for conversion

        Returns
        -------
        xr.DataArray
            Converted data array
        """
        current_units = da.attrs.get("units", "K")

        # If same units, no conversion needed
        if current_units == target_units:
            return da

        # Convert temperature units - assuming we're dealing with temperature data
        match (current_units, target_units):
            case ("K", "degC"):
                da = da - 273.15
            case ("degC", "K"):
                da = da + 273.15
            case ("K", "degF"):
                da = (1.8 * (da - 273.15)) + 32
            case ("degC", "degF"):
                da = (1.8 * da) + 32
            case ("degF", "K"):
                da = ((da - 32) / 1.8) + 273.15
            case ("degF", "degC"):
                da = (da - 32) / 1.8
            case _:
                raise ValueError(
                    f"Unsupported temperature conversion from {current_units} to {target_units}"
                )

        # Update units attribute
        da.attrs["units"] = target_units

        return da
