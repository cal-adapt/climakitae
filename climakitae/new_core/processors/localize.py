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
from climakitae.util.unit_conversions import convert_units
from climakitae.util.utils import get_closest_gridcell


@register_processor("localize", priority=1)
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

    def execute(
        self,
        result: Union[
            xr.DataArray, xr.Dataset, Dict[str, Union[xr.DataArray, xr.Dataset]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.DataArray, xr.Dataset]:
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

                # Process each station
                station_results = []
                for _, station_row in station_subset.iterrows():
                    station_data = self._process_single_station(
                        result, station_row, context
                    )
                    if station_data is not None:
                        station_results.append(station_data)

                # Combine station results
                if station_results:
                    combined = self._combine_station_results(station_results)
                else:
                    raise ValueError("No valid station data could be processed")

                return combined

            case dict():
                self._validate_inputs(result)
                for key, item in result.items():
                    if "ssp" not in key:
                        continue  # Skip non-SSP data
                    ret[key] = self.execute(item, context)
                    ret[key].attrs = item.attrs

        self.update_context(context)
        return ret

    def _validate_inputs(self, data: Union[xr.DataArray, xr.Dataset]):
        """
        Validate that inputs are suitable for station processing.

        Checks that the input data contains required spatial coordinates and
        validates that all specified stations exist in the metadata. Converts
        4-letter airport codes to full station names when applicable.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset or dict or list/tuple
            Input data to validate. Must contain 'lat' and 'lon' coordinates
            for spatial processing.

        Raises
        ------
        ValueError
            If data lacks required 'lat' and 'lon' coordinates
            If data type is not supported
            If specified stations are not found in metadata
            If airport codes are ambiguous or not found

        Notes
        -----
        This method also handles conversion of 4-letter airport codes to full
        station names by looking up the codes in the station metadata.
        """
        # Check that we have gridded data
        match data:
            case xr.DataArray() | xr.Dataset():
                if not hasattr(data, "lat") or not hasattr(data, "lon"):
                    raise ValueError(
                        "Input data must have 'lat' and 'lon' coordinates for station localization"
                    )
            case dict():
                for key, value in data.items():
                    if not isinstance(value, (xr.DataArray, xr.Dataset)):
                        raise ValueError(
                            f"Invalid data type for key '{key}': expected DataArray or Dataset"
                        )
                    if not hasattr(value, "lat") or not hasattr(value, "lon"):
                        raise ValueError(
                            f"Data for key '{key}' must have 'lat' and 'lon' coordinates"
                        )
            case list() | tuple():
                for item in data:
                    if not isinstance(item, (xr.DataArray, xr.Dataset)):
                        raise ValueError(
                            f"Invalid item in list/tuple: expected DataArray or Dataset, got {type(item)}"
                        )
                    if not hasattr(item, "lat") or not hasattr(item, "lon"):
                        raise ValueError(
                            "All items in list/tuple must have 'lat' and 'lon' coordinates"
                        )
            case _:
                raise ValueError(
                    "Input data must be a DataArray, Dataset, dict of DataArrays/Datasets, or list/tuple of DataArrays/Datasets"
                )

        # Validate stations exist
        available_stations = self._stations_gdf["station"].tolist()
        invalid_stations = []
        for i, s in enumerate(self.stations):
            if len(s) == 4:
                # this is a 4 letter airport code, convert to station name
                mask = self._stations_gdf["station id"].contains(s)
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

    def _process_single_station(
        self,
        gridded_data: Union[xr.DataArray, xr.Dataset],
        station_row: pd.Series,
        context: Dict[str, Any],
    ) -> Optional[Union[xr.DataArray, xr.Dataset]]:
        """
        Process gridded data for a single weather station.

        Extracts data from the nearest grid cell to the station location,
        applies optional bias correction, and adds station metadata as attributes.

        Parameters
        ----------
        gridded_data : xr.DataArray or xr.Dataset
            Input gridded climate data to extract from
        station_row : pd.Series
            Row from station metadata containing station information including
            coordinates, name, and identifiers
        context : dict of str to Any
            Processing context passed through the pipeline

        Returns
        -------
        xr.DataArray or xr.Dataset or None
            Extracted and processed data for the station, or None if processing
            failed. Contains station coordinates and elevation as attributes.
            Returns DataArray for single-variable input, Dataset for multi-variable input.

        Notes
        -----
        This method uses nearest-neighbor spatial extraction and removes
        superfluous coordinates that are not dimensions to facilitate merging
        multiple stations. Station metadata is collected for later use.
        """

        # Extract station info
        station_name = station_row["station"]
        station_lat = station_row["LAT_Y"]
        station_lon = station_row["LON_X"]

        try:
            # Get closest grid cell
            closest_data = get_closest_gridcell(
                gridded_data, station_lat, station_lon, print_coords=False
            )

            # Drop any coordinates that are not also dimensions
            # This makes merging stations easier and drops superfluous coordinates
            closest_data = closest_data.drop_vars(
                [
                    coord
                    for coord in closest_data.coords
                    if coord not in closest_data.dims
                ],
                errors="ignore",
            )

            # Apply bias correction if requested
            if self.bias_correction:
                closest_data = self._apply_bias_correction(
                    closest_data, station_row, context
                )

            # Store station metadata
            if self.station_metadata is None:
                self.station_metadata = []
            self.station_metadata.append(self._get_station_metadata(station_row))

            # Add station coordinates and elevation as attributes
            closest_data.attrs["station_coordinates"] = (station_lat, station_lon)
            closest_data.attrs["station_elevation"] = station_row.get(
                "elevation", "unknown"
            )

            # Rename to station name for easy identification
            closest_data.attrs["name"] = station_name

            return closest_data

        except (ValueError, KeyError, IndexError, AttributeError) as e:
            print(f"Warning: Failed to process station {station_name}: {e}")
            return None

    def _apply_bias_correction(
        self,
        model_data: Union[xr.DataArray, xr.Dataset],
        station_row: pd.Series,
        context: Dict[str, Any],
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Apply bias correction using observational data from the weather station.

        Dispatches to the appropriate bias correction method based on the
        configured method. Currently supports quantile delta mapping.

        Parameters
        ----------
        model_data : xr.DataArray or xr.Dataset
            Model gridded data to be bias corrected
        station_row : pd.Series
            Station metadata row containing station information
        context : dict of str to Any
            Processing context with configuration information

        Returns
        -------
        xr.DataArray or xr.Dataset
            Bias corrected model data

        Raises
        ------
        ValueError
            If the specified bias correction method is not supported

        Notes
        -----
        This method serves as a dispatcher to specific bias correction
        implementations. Currently only 'quantile_delta_mapping' is supported.
        Handles both DataArray and Dataset inputs.
        """

        if self.method == "quantile_delta_mapping":
            return self._apply_qdm_correction(model_data, station_row, context)
        else:
            raise ValueError(f"Unsupported bias correction method: {self.method}")

    def _apply_qdm_correction(
        self,
        model_data: Union[xr.DataArray, xr.Dataset],
        station_row: pd.Series,
        context: Dict[str, Any],
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Apply Quantile Delta Mapping (QDM) bias correction.

        Loads observational station data and applies quantile delta mapping
        bias correction to adjust model data based on historical relationships
        between model and observational data.

        Parameters
        ----------
        model_data : xr.DataArray or xr.Dataset
            Model data to be bias corrected
        station_row : pd.Series
            Station metadata containing station ID and other information
        context : dict of str to Any
            Processing context with time slice and other configuration

        Returns
        -------
        xr.DataArray or xr.Dataset
            Bias corrected model data, or original data if correction fails

        Notes
        -----
        If station data cannot be loaded, a warning is printed and the original
        model data is returned without correction. The method handles the
        complete QDM workflow including data loading, preprocessing, and
        bias correction application. Handles both DataArray and Dataset inputs.
        """

        # Get station data
        station_id = station_row["station id"]
        station_data = self._load_station_data(station_id, station_row)

        if station_data is None:
            print(
                f"\n\nWARNING: No station data available for station {station_row['station']}"
            )
            return model_data

        # Apply bias correction
        try:
            # Handle Dataset vs DataArray
            if isinstance(model_data, xr.Dataset):
                # For Dataset, apply correction to each data variable
                corrected_vars = {}
                for var_name, data_array in model_data.data_vars.items():
                    corrected_vars[var_name] = self._bias_correct_model_data(
                        station_data, data_array, context
                    )
                # Reconstruct Dataset with corrected variables
                corrected_data = xr.Dataset(corrected_vars)
                # Preserve original attributes and coordinates
                corrected_data.attrs = model_data.attrs
                for coord_name, coord_data in model_data.coords.items():
                    if coord_name not in corrected_data.coords:
                        corrected_data = corrected_data.assign_coords(
                            {coord_name: coord_data}
                        )
                return corrected_data
            else:
                # For DataArray, apply correction directly
                corrected_data = self._bias_correct_model_data(
                    station_data, model_data, context
                )
                return corrected_data
        except Exception as e:
            print(
                f"Warning: Bias correction failed for station {station_row['station']}: {e}"
            )
            return model_data

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

    def _bias_correct_model_data(
        self,
        obs_da: xr.DataArray,
        gridded_da: xr.DataArray,
        context: Dict[str, Any],
        window: int = 90,
        group: str = "time.dayofyear",
        kind: str = "+",
    ) -> xr.DataArray:
        """
        Bias correct model data using observational station data.

        Parameters
        ----------
        obs_da : xr.DataArray
            Station observational data
        gridded_da : xr.DataArray
            Model gridded data
        context : Dict[str, Any]
            Processing context containing time slice information
        window : int, optional
            Window of days +/- for grouping (default: 90)
        group : str, optional
            Time frequency to group data by (default: "time.dayofyear")
        kind : str, optional
            Adjustment kind, additive "+" or multiplicative "*" (default: "+")

        Returns
        -------
        xr.DataArray
            Bias corrected data
        """

        # Get grouper with window for seasonality
        grouper = Grouper(group, window=window)

        # Convert units to match gridded data
        # Use K (Kelvin) as default if units attribute is missing
        target_units = gridded_da.attrs.get("units", "K")

        # Ensure gridded data has units attribute
        if "units" not in gridded_da.attrs:
            gridded_da.attrs["units"] = "K"

        # Check if obs_da has units attribute, if not assume it's in Kelvin
        if "units" not in obs_da.attrs:
            obs_da.attrs["units"] = "K"

        try:
            obs_da = convert_units(obs_da, target_units)
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not convert units, using original data: {e}")
            # Ensure both have the same units for comparison
            obs_da.attrs["units"] = target_units

        # Rechunk data - cannot be chunked along time dimension for xclim
        gridded_da = gridded_da.chunk(chunks=dict(time=-1))

        # Limit observational data to available period
        obs_da = obs_da.sel(time=slice(obs_da.time.values[0], "2014-08-31"))
        obs_da = obs_da.chunk(chunks=dict(time=-1))

        # Convert calendars to no leap year for consistency
        obs_da = obs_da.convert_calendar("noleap")
        gridded_da = gridded_da.convert_calendar("noleap")

        # Get original time slice from context
        original_time_slice = context.get("original_time_slice", (2015, 2100))
        data_sliced = gridded_da.sel(
            time=slice(str(original_time_slice[0]), str(original_time_slice[1]))
        )

        # Align training periods - use overlapping time period
        train_start = max(obs_da.time.values[0], gridded_da.time.values[0])
        train_end = min(obs_da.time.values[-1], gridded_da.time.values[-1])

        gridded_train = gridded_da.sel(time=slice(str(train_start), str(train_end)))
        obs_train = obs_da.sel(time=slice(str(train_start), str(train_end)))

        # Train quantile delta mapping
        quant_delt_map = QuantileDeltaMapping.train(
            obs_train,
            gridded_train,
            nquantiles=self.nquantiles,
            group=grouper,
            kind=kind,
        )

        # Apply bias correction to future/target data
        da_adj = quant_delt_map.adjust(data_sliced)
        da_adj.name = gridded_da.name  # Preserve original name

        # Convert time index back to datetime if needed
        try:
            time_index = da_adj.indexes.get("time")
            if time_index is not None and hasattr(time_index, "to_datetimeindex"):
                da_adj = da_adj.assign_coords(time=time_index.to_datetimeindex())
        except (AttributeError, KeyError, TypeError):
            # Time index conversion not needed or not available
            pass

        return da_adj

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
