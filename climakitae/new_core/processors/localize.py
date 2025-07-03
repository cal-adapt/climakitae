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
import traceback
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
                result = extend_time_domain(result)
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
            print(
                f"DEBUG: Starting bias correction for station {station_row['station']}"
            )
            print(f"DEBUG: model_data type: {type(model_data)}")
            if hasattr(model_data, "time"):
                print(f"DEBUG: model_data.time type: {type(model_data.time)}")
                print(
                    f"DEBUG: model_data.time has dt: {hasattr(model_data.time, 'dt')}"
                )
            if hasattr(station_data, "time"):
                print(f"DEBUG: station_data.time type: {type(station_data.time)}")
                print(
                    f"DEBUG: station_data.time has dt: {hasattr(station_data.time, 'dt')}"
                )

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
            Bias corrected data spanning the full time range:
            - Historical period (up to 2014-08-31): Original uncorrected data
            - Future period (from 2014-09-01): Bias-corrected data
            If bias correction fails, returns the original data unchanged.
        """

        print("DEBUG: _bias_correct_model_data called")
        print(f"DEBUG: obs_da type: {type(obs_da)}")
        print(f"DEBUG: gridded_da type: {type(gridded_da)}")
        if hasattr(obs_da, "time"):
            print(f"DEBUG: obs_da.time type: {type(obs_da.time)}")
            print(f"DEBUG: obs_da.time has dt: {hasattr(obs_da.time, 'dt')}")
        if hasattr(gridded_da, "time"):
            print(f"DEBUG: gridded_da.time type: {type(gridded_da.time)}")
            print(f"DEBUG: gridded_da.time has dt: {hasattr(gridded_da.time, 'dt')}")

        # Get grouper with window for seasonality
        grouper = Grouper(group, window=window)

        # Convert units to match gridded data
        # Use K (Kelvin) as default if units attribute is missing
        target_units = gridded_da.attrs.get("units", "K")

        # Ensure gridded data has units attribute
        if "units" not in gridded_da.attrs:
            gridded_da.attrs["units"] = "K"

        # Ensure frequency attribute is set correctly for unit conversion
        if "frequency" not in gridded_da.attrs:
            # must be hourly
            gridded_da.attrs["frequency"] = "hourly"

        # Ensure time coordinate is properly formatted
        if "time" in gridded_da.coords:
            print(f"DEBUG: Gridded data time coordinate type: {type(gridded_da.time)}")
            print(
                f"DEBUG: Gridded data time values type: {type(gridded_da.time.values)}"
            )
            print(
                f"DEBUG: Gridded data time first value type: {type(gridded_da.time.values[0]) if len(gridded_da.time.values) > 0 else 'empty'}"
            )
            print(
                f"DEBUG: Gridded data has dt attribute: {hasattr(gridded_da.time, 'dt')}"
            )
            print(
                f"DEBUG: Gridded data time index type: {type(gridded_da.time.to_index())}"
            )
            try:
                # Ensure time coordinate has datetime index
                if not hasattr(gridded_da.time, "dt"):
                    print("DEBUG: Converting gridded data time coordinate to datetime")
                    gridded_da = gridded_da.assign_coords(
                        time=pd.to_datetime(gridded_da.time)
                    )
                    print(
                        f"DEBUG: After conversion, gridded data has dt attribute: {hasattr(gridded_da.time, 'dt')}"
                    )
                    print(
                        f"DEBUG: After conversion, gridded data time type: {type(gridded_da.time)}"
                    )
            except (ValueError, TypeError) as e:
                print(f"DEBUG: Time conversion failed for gridded data: {e}")
                # If time conversion fails, continue without it

        # Check if obs_da has units attribute, if not assume it's in Kelvin
        if "units" not in obs_da.attrs:
            obs_da.attrs["units"] = "K"

        # Ensure frequency attribute is set correctly for unit conversion
        if "frequency" not in obs_da.attrs:
            # Try to infer frequency from time coordinate
            if "time" in obs_da.coords and len(obs_da.time) > 1:
                try:
                    time_diff = obs_da.time[1] - obs_da.time[0]
                    if hasattr(time_diff, "days"):
                        if time_diff.days >= 28:  # Monthly data
                            obs_da.attrs["frequency"] = "monthly"
                        elif time_diff.days >= 1:  # Daily data
                            obs_da.attrs["frequency"] = "daily"
                        else:  # Sub-daily data (hourly)
                            obs_da.attrs["frequency"] = "hourly"
                    elif hasattr(time_diff, "seconds"):
                        if time_diff.seconds <= 3600:  # 1 hour or less
                            obs_da.attrs["frequency"] = "hourly"
                        elif time_diff.seconds <= 86400:  # 1 day or less
                            obs_da.attrs["frequency"] = "daily"
                        else:
                            obs_da.attrs["frequency"] = "monthly"
                    else:
                        obs_da.attrs["frequency"] = "daily"
                except (ValueError, TypeError, AttributeError):
                    obs_da.attrs["frequency"] = "daily"
            else:
                obs_da.attrs["frequency"] = "daily"

        # Ensure time coordinate is properly formatted
        if "time" in obs_da.coords:
            print(f"DEBUG: Obs data time coordinate type: {type(obs_da.time)}")
            print(f"DEBUG: Obs data time values type: {type(obs_da.time.values)}")
            print(
                f"DEBUG: Obs data time first value type: {type(obs_da.time.values[0]) if len(obs_da.time.values) > 0 else 'empty'}"
            )
            print(f"DEBUG: Obs data has dt attribute: {hasattr(obs_da.time, 'dt')}")
            print(f"DEBUG: Obs data time index type: {type(obs_da.time.to_index())}")
            try:
                # Ensure time coordinate has datetime index
                if not hasattr(obs_da.time, "dt"):
                    print("DEBUG: Converting obs data time coordinate to datetime")
                    obs_da = obs_da.assign_coords(time=pd.to_datetime(obs_da.time))
                    print(
                        f"DEBUG: After conversion, obs data has dt attribute: {hasattr(obs_da.time, 'dt')}"
                    )
                    print(
                        f"DEBUG: After conversion, obs data time type: {type(obs_da.time)}"
                    )
            except (ValueError, TypeError) as e:
                print(f"DEBUG: Time conversion failed for obs data: {e}")
                # If time conversion fails, continue without it

        try:
            print(
                f"DEBUG: About to call simple_convert_temperature on obs_da with units: {obs_da.attrs.get('units', 'None')}"
            )
            print(
                f"DEBUG: obs_da time before conversion has dt: {hasattr(obs_da.time, 'dt') if 'time' in obs_da.coords else 'No time coord'}"
            )
            obs_da = self._simple_convert_temperature(obs_da, target_units)
            print("DEBUG: Successfully converted obs_da units")
        except (ValueError, KeyError, AttributeError) as e:
            print(f"Warning: Could not convert units, using original data: {e}")
            import traceback

            print("DEBUG: Full traceback for obs unit conversion error:")
            traceback.print_exc()
            # Ensure both have the same units for comparison
            obs_da.attrs["units"] = target_units

        # Rechunk data - cannot be chunked along time dimension for xclim
        print("DEBUG: About to rechunk gridded_da")
        print(
            f"DEBUG: gridded_da.time has dt before rechunk: {hasattr(gridded_da.time, 'dt')}"
        )
        gridded_da = gridded_da.chunk(chunks=dict(time=-1))
        print(
            f"DEBUG: gridded_da.time has dt after rechunk: {hasattr(gridded_da.time, 'dt')}"
        )

        # Limit observational data to available period
        print("DEBUG: About to slice obs_da time")
        print(f"DEBUG: obs_da.time has dt before slice: {hasattr(obs_da.time, 'dt')}")
        obs_da = obs_da.sel(time=slice(obs_da.time.values[0], "2014-08-31"))
        print(f"DEBUG: obs_da.time has dt after slice: {hasattr(obs_da.time, 'dt')}")

        print("DEBUG: About to rechunk obs_da")
        obs_da = obs_da.chunk(chunks=dict(time=-1))
        print(f"DEBUG: obs_da.time has dt after rechunk: {hasattr(obs_da.time, 'dt')}")

        # NOTE: We'll convert calendars later, just before training QDM
        # Converting too early creates cftime objects that are incompatible with xclim

        # Get original time slice from context
        original_time_slice = context.get("original_time_slice", (2015, 2100))
        print(f"DEBUG: About to slice data for time range: {original_time_slice}")
        print(
            f"DEBUG: gridded_da.time has dt before data slice: {hasattr(gridded_da.time, 'dt')}"
        )
        data_sliced = gridded_da.sel(
            time=slice(str(original_time_slice[0]), str(original_time_slice[1]))
        )
        print(
            f"DEBUG: data_sliced.time has dt after slice: {hasattr(data_sliced.time, 'dt')}"
        )

        # Fix time coordinate if .dt attribute is lost
        if not hasattr(data_sliced.time, "dt"):
            print("DEBUG: Fixing data_sliced time coordinate")
            data_sliced = data_sliced.assign_coords(
                time=pd.to_datetime(data_sliced.time)
            )
            print(
                f"DEBUG: data_sliced.time has dt after fix: {hasattr(data_sliced.time, 'dt')}"
            )

        # Manual separation of historical and future periods from combined array
        print("DEBUG: About to manually separate historical and future periods")

        # WORKFLOW OVERVIEW:
        # 1. Input: Combined array with historical (1980-2014) + future (2015-2100) data
        # 2. Separate into historical period (for training) and future period (for correction)
        # 3. Find overlap between historical gridded data and observational data
        # 4. Train QDM on overlapping historical period only
        # 5. Apply bias correction to future period only
        # 6. Return bias-corrected future data

        # Define the split date between historical and future periods
        # Historical: 1980-2014-08-31, Future: 2014-09-01-2100-09-01
        historical_end_date = "2014-08-31"
        future_start_date = "2014-09-01"

        print(
            f"DEBUG: Combined gridded data period: {gridded_da.time.values[0]} to {gridded_da.time.values[-1]}"
        )
        print(
            f"DEBUG: Separating at: historical <= {historical_end_date}, future >= {future_start_date}"
        )

        # Extract historical period from combined gridded data (for training with obs)
        print(
            f"DEBUG: gridded_da.time has dt before historical slice: {hasattr(gridded_da.time, 'dt')}"
        )
        gridded_historical = gridded_da.sel(time=slice(None, historical_end_date))
        print(
            f"DEBUG: gridded_historical.time has dt after slice: {hasattr(gridded_historical.time, 'dt')}"
        )
        print(
            f"DEBUG: Historical gridded period: {gridded_historical.time.values[0] if len(gridded_historical.time) > 0 else 'empty'} to {gridded_historical.time.values[-1] if len(gridded_historical.time) > 0 else 'empty'}"
        )

        # Fix time coordinate if .dt attribute is lost
        if not hasattr(gridded_historical.time, "dt"):
            print("DEBUG: Fixing gridded_historical time coordinate")
            gridded_historical = gridded_historical.assign_coords(
                time=pd.to_datetime(gridded_historical.time)
            )
            print(
                f"DEBUG: gridded_historical.time has dt after fix: {hasattr(gridded_historical.time, 'dt')}"
            )

        # Extract future period from combined gridded data (target for bias correction)
        print(
            f"DEBUG: gridded_da.time has dt before future slice: {hasattr(gridded_da.time, 'dt')}"
        )
        print(f"DEBUG: About to slice future period from {future_start_date} to end")
        print(
            f"DEBUG: Sample gridded_da time values around split: {gridded_da.time.sel(time=slice('2014-08-01', '2014-10-01')).values if len(gridded_da.time) > 0 else 'no data'}"
        )

        try:
            # Try different date formats for slicing
            gridded_future = gridded_da.sel(time=slice(future_start_date, None))
            if len(gridded_future.time) == 0:
                print(
                    "DEBUG: First slice attempt returned empty, trying alternative date format"
                )
                # Try with explicit datetime conversion
                future_start_dt = pd.to_datetime(future_start_date)
                gridded_future = gridded_da.where(
                    gridded_da.time >= future_start_dt, drop=True
                )
        except Exception as e:
            print(f"DEBUG: Future slicing failed: {e}")
            # Fallback: use index-based slicing
            time_index = gridded_da.time.to_index()
            future_mask = time_index >= pd.to_datetime(future_start_date)
            if future_mask.any():
                gridded_future = gridded_da.isel(time=future_mask)
            else:
                print("ERROR: No future period found in data")
                return gridded_da

        print(
            f"DEBUG: gridded_future.time has dt after slice: {hasattr(gridded_future.time, 'dt')}"
        )
        print(
            f"DEBUG: Future gridded period: {gridded_future.time.values[0] if len(gridded_future.time) > 0 else 'empty'} to {gridded_future.time.values[-1] if len(gridded_future.time) > 0 else 'empty'}"
        )
        print(f"DEBUG: Future period shape: {gridded_future.shape}")
        print(f"DEBUG: Future period time dimension size: {len(gridded_future.time)}")

        # Fix time coordinate if .dt attribute is lost
        if not hasattr(gridded_future.time, "dt"):
            print("DEBUG: Fixing gridded_future time coordinate")
            gridded_future = gridded_future.assign_coords(
                time=pd.to_datetime(gridded_future.time)
            )
            print(
                f"DEBUG: gridded_future.time has dt after fix: {hasattr(gridded_future.time, 'dt')}"
            )

        # Debug: Check if we have any data in the future period
        print(f"DEBUG: Future period length: {len(gridded_future.time)}")
        if len(gridded_future.time) == 0:
            print(
                "ERROR: Future period is empty! This will cause issues with bias correction."
            )
            print(
                f"DEBUG: Original combined data period: {gridded_da.time.values[0]} to {gridded_da.time.values[-1]}"
            )
            print(f"DEBUG: Future start date used for slicing: {future_start_date}")
            print("DEBUG: Returning uncorrected original combined data")
            return gridded_da  # Return the original gridded data without correction

        # Determine training period from overlap between obs and historical gridded data
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

        print(f"DEBUG: Obs data period: {obs_start} to {obs_end}")
        print(f"DEBUG: Historical gridded period: {hist_start} to {hist_end}")

        # Find overlap between observations and historical gridded data for training
        if (
            obs_end < hist_start
            or obs_start > hist_end
            or len(gridded_historical.time) == 0
        ):
            print(
                "WARNING: No temporal overlap between observational and historical gridded data"
            )
            print(
                "WARNING: Cannot perform bias correction without overlapping training period"
            )
            print("WARNING: Returning original combined data without bias correction")
            # Return the full original data instead of just future
            return gridded_da

        # Calculate training period as the overlap between obs and historical gridded data
        train_start = max(obs_start, hist_start)
        train_end = min(obs_end, hist_end)
        print(
            f"DEBUG: Training period (historical overlap): {train_start} to {train_end}"
        )

        # Check if we have a meaningful training period (at least some data)
        if train_start >= train_end:
            print("WARNING: No meaningful overlapping training period")
            print("WARNING: Returning original combined data without bias correction")
            # Return the full original data instead of just future
            return gridded_da

        # Additional check: ensure training period has sufficient data (at least 1 year)
        train_duration_days = (
            pd.to_datetime(train_end) - pd.to_datetime(train_start)
        ).days
        if train_duration_days < 365:
            print(
                f"WARNING: Training period too short ({train_duration_days} days < 365 days)"
            )
            print(
                "WARNING: Bias correction may be unreliable with < 1 year of training data"
            )
            print("WARNING: Continuing with available training data...")

        # Extract training data from historical period
        print(
            f"DEBUG: gridded_historical.time has dt before train slice: {hasattr(gridded_historical.time, 'dt')}"
        )
        gridded_train = gridded_historical.sel(
            time=slice(str(train_start), str(train_end))
        )
        print(
            f"DEBUG: gridded_train.time has dt after slice: {hasattr(gridded_train.time, 'dt')}"
        )

        # Fix time coordinate if .dt attribute is lost
        if not hasattr(gridded_train.time, "dt"):
            print("DEBUG: Fixing gridded_train time coordinate")
            gridded_train = gridded_train.assign_coords(
                time=pd.to_datetime(gridded_train.time)
            )
            print(
                f"DEBUG: gridded_train.time has dt after fix: {hasattr(gridded_train.time, 'dt')}"
            )

        # Extract corresponding observational training data
        print(
            f"DEBUG: obs_da.time has dt before train slice: {hasattr(obs_da.time, 'dt')}"
        )
        obs_train = obs_da.sel(time=slice(str(train_start), str(train_end)))
        print(
            f"DEBUG: obs_train.time has dt after slice: {hasattr(obs_train.time, 'dt')}"
        )

        # Fix time coordinate if .dt attribute is lost
        if not hasattr(obs_train.time, "dt"):
            print("DEBUG: Fixing obs_train time coordinate")
            obs_train = obs_train.assign_coords(time=pd.to_datetime(obs_train.time))
            print(
                f"DEBUG: obs_train.time has dt after fix: {hasattr(obs_train.time, 'dt')}"
            )

        # The target for bias correction is the future period
        print("DEBUG: Setting target for bias correction to future period")

        # Convert calendars to no leap year for consistency just before QDM training
        # This minimizes issues with cftime objects in earlier processing steps
        print("DEBUG: About to convert calendars for QDM training")
        try:
            print("DEBUG: Converting obs_train calendar")
            obs_train = obs_train.convert_calendar("noleap")
            print(
                f"DEBUG: obs_train calendar converted, time type: {type(obs_train.time.values[0]) if len(obs_train.time) > 0 else 'empty'}"
            )

            print("DEBUG: Converting gridded_train calendar")
            gridded_train = gridded_train.convert_calendar("noleap")
            print(
                f"DEBUG: gridded_train calendar converted, time type: {type(gridded_train.time.values[0]) if len(gridded_train.time) > 0 else 'empty'}"
            )

            print("DEBUG: Converting gridded_future calendar")
            gridded_future = gridded_future.convert_calendar("noleap")
            print(
                f"DEBUG: gridded_future calendar converted, time type: {type(gridded_future.time.values[0]) if len(gridded_future.time) > 0 else 'empty'}"
            )

        except Exception as e:
            print(f"DEBUG: Calendar conversion failed: {e}")
            print("DEBUG: Continuing without calendar conversion")

        # Convert cftime objects back to standard datetime for xclim compatibility
        print("DEBUG: Converting cftime objects to standard datetime for xclim")
        print(
            f"DEBUG: gridded_train time values type: {type(gridded_train.time.values[0]) if len(gridded_train.time) > 0 else 'empty'}"
        )
        print(
            f"DEBUG: obs_train time values type: {type(obs_train.time.values[0]) if len(obs_train.time) > 0 else 'empty'}"
        )
        print(
            f"DEBUG: gridded_future time values type: {type(gridded_future.time.values[0]) if len(gridded_future.time) > 0 else 'empty'}"
        )

        try:
            # Check if we have cftime objects
            if hasattr(gridded_train.time.values[0], "strftime") and "cftime" in str(
                type(gridded_train.time.values[0])
            ):
                # Convert cftime to standard datetime using strftime/strptime
                print("DEBUG: Converting gridded_train time from cftime to datetime")
                time_strings = [
                    t.strftime("%Y-%m-%d %H:%M:%S") for t in gridded_train.time.values
                ]
                datetime_index = pd.to_datetime(time_strings)
                gridded_train = gridded_train.assign_coords(time=datetime_index)
                print(
                    f"DEBUG: gridded_train time converted, has dt: {hasattr(gridded_train.time, 'dt')}"
                )
                print(
                    f"DEBUG: gridded_train new time type: {type(gridded_train.time.values[0])}"
                )

            if hasattr(obs_train.time.values[0], "strftime") and "cftime" in str(
                type(obs_train.time.values[0])
            ):
                # Convert cftime to standard datetime using strftime/strptime
                print("DEBUG: Converting obs_train time from cftime to datetime")
                time_strings = [
                    t.strftime("%Y-%m-%d %H:%M:%S") for t in obs_train.time.values
                ]
                datetime_index = pd.to_datetime(time_strings)
                obs_train = obs_train.assign_coords(time=datetime_index)
                print(
                    f"DEBUG: obs_train time converted, has dt: {hasattr(obs_train.time, 'dt')}"
                )
                print(
                    f"DEBUG: obs_train new time type: {type(obs_train.time.values[0])}"
                )

            if hasattr(gridded_future.time.values[0], "strftime") and "cftime" in str(
                type(gridded_future.time.values[0])
            ):
                # Convert cftime to standard datetime using strftime/strptime
                print("DEBUG: Converting gridded_future time from cftime to datetime")
                time_strings = [
                    t.strftime("%Y-%m-%d %H:%M:%S") for t in gridded_future.time.values
                ]
                datetime_index = pd.to_datetime(time_strings)
                gridded_future = gridded_future.assign_coords(time=datetime_index)
                print(
                    f"DEBUG: gridded_future time converted, has dt: {hasattr(gridded_future.time, 'dt')}"
                )
                print(
                    f"DEBUG: gridded_future new time type: {type(gridded_future.time.values[0])}"
                )

        except (AttributeError, TypeError, ValueError, IndexError) as e:
            print(
                f"DEBUG: Time conversion from cftime failed, continuing with existing time coords: {e}"
            )
            # Try alternative approach - use xarray's built-in conversion
            try:
                print("DEBUG: Trying alternative time conversion approach")
                if len(gridded_train.time) > 0:
                    gridded_train = gridded_train.convert_calendar(
                        "standard", use_cftime=False
                    )
                    print(f"DEBUG: gridded_train converted to standard calendar")
                if len(obs_train.time) > 0:
                    obs_train = obs_train.convert_calendar("standard", use_cftime=False)
                    print(f"DEBUG: obs_train converted to standard calendar")
                if len(gridded_future.time) > 0:
                    gridded_future = gridded_future.convert_calendar(
                        "standard", use_cftime=False
                    )
                    print(f"DEBUG: gridded_future converted to standard calendar")
            except Exception as e2:
                print(f"DEBUG: Alternative time conversion also failed: {e2}")
                print(
                    "DEBUG: Proceeding with original time coordinates - bias correction may fail"
                )

        # Train quantile delta mapping
        print("DEBUG: About to train QuantileDeltaMapping")
        print(
            f"DEBUG: obs_train.time has dt before QDM train: {hasattr(obs_train.time, 'dt')}"
        )
        print(
            f"DEBUG: gridded_train.time has dt before QDM train: {hasattr(gridded_train.time, 'dt')}"
        )
        print(
            f"DEBUG: Training data shapes - obs_train: {obs_train.shape}, gridded_train: {gridded_train.shape}"
        )
        quant_delt_map = QuantileDeltaMapping.train(
            obs_train,
            gridded_train,
            nquantiles=self.nquantiles,
            group=grouper,
            kind=kind,
        )
        print("DEBUG: Successfully trained QuantileDeltaMapping")

        # Apply bias correction to future/target data
        print("DEBUG: About to apply bias correction")
        print(
            f"DEBUG: gridded_future.time has dt before QDM adjust: {hasattr(gridded_future.time, 'dt')}"
        )
        print(
            f"DEBUG: Future data shape before bias correction: {gridded_future.shape}"
        )
        da_adj_future = quant_delt_map.adjust(gridded_future)
        print("DEBUG: Successfully applied bias correction")
        print(f"DEBUG: Bias corrected future data shape: {da_adj_future.shape}")
        da_adj_future.name = gridded_da.name  # Preserve original name

        # Convert time index back to datetime if needed
        try:
            time_index = da_adj_future.indexes.get("time")
            if time_index is not None and hasattr(time_index, "to_datetimeindex"):
                da_adj_future = da_adj_future.assign_coords(
                    time=time_index.to_datetimeindex()
                )
        except (AttributeError, KeyError, TypeError):
            # Time index conversion not needed or not available
            pass

        # Combine uncorrected historical data with bias-corrected future data
        print(
            "DEBUG: Combining uncorrected historical data with bias-corrected future data"
        )

        # Ensure historical data has the same calendar and time format as bias-corrected future
        try:
            # Convert historical data to match the future data's calendar and time format
            gridded_historical_for_concat = gridded_historical.copy()

            # Apply same calendar conversion as used for bias correction
            if len(gridded_historical_for_concat.time) > 0:
                try:
                    gridded_historical_for_concat = (
                        gridded_historical_for_concat.convert_calendar("noleap")
                    )
                    print(
                        "DEBUG: Converted historical data calendar to noleap for concatenation"
                    )
                except Exception as e:
                    print(f"DEBUG: Historical calendar conversion failed: {e}")

                # Convert cftime to standard datetime if needed (same as done for future data)
                if (
                    len(gridded_historical_for_concat.time) > 0
                    and hasattr(
                        gridded_historical_for_concat.time.values[0], "strftime"
                    )
                    and "cftime"
                    in str(type(gridded_historical_for_concat.time.values[0]))
                ):
                    print(
                        "DEBUG: Converting historical time from cftime to datetime for concatenation"
                    )
                    time_strings = [
                        t.strftime("%Y-%m-%d %H:%M:%S")
                        for t in gridded_historical_for_concat.time.values
                    ]
                    datetime_index = pd.to_datetime(time_strings)
                    gridded_historical_for_concat = (
                        gridded_historical_for_concat.assign_coords(time=datetime_index)
                    )

            # Concatenate historical and bias-corrected future data
            print(
                f"DEBUG: Historical period for concat: {gridded_historical_for_concat.time.values[0] if len(gridded_historical_for_concat.time) > 0 else 'empty'} to {gridded_historical_for_concat.time.values[-1] if len(gridded_historical_for_concat.time) > 0 else 'empty'}"
            )
            print(
                f"DEBUG: Future period for concat: {da_adj_future.time.values[0] if len(da_adj_future.time) > 0 else 'empty'} to {da_adj_future.time.values[-1] if len(da_adj_future.time) > 0 else 'empty'}"
            )

            # Combine the data along time dimension
            # Type ignore needed due to mypy not understanding that both objects are DataArrays
            da_combined = xr.concat([gridded_historical_for_concat, da_adj_future], dim="time")  # type: ignore
            print(f"DEBUG: Combined data shape: {da_combined.shape}")
            print(
                f"DEBUG: Combined time range: {da_combined.time.values[0]} to {da_combined.time.values[-1]}"
            )

            # Preserve original attributes and name
            da_combined.attrs = gridded_da.attrs.copy()
            da_combined.name = gridded_da.name

            # Add bias correction metadata to attributes
            da_combined.attrs["bias_correction_applied"] = "true"
            da_combined.attrs["bias_correction_method"] = "quantile_delta_mapping"
            da_combined.attrs["bias_correction_note"] = (
                f"Historical period (up to {historical_end_date}) uncorrected, future period (from {future_start_date}) bias-corrected"
            )
            da_combined.attrs["historical_future_split_date"] = historical_end_date

            print(
                "DEBUG: Successfully combined historical (uncorrected) and future (bias-corrected) data"
            )
            return da_combined

        except Exception as e:
            print(f"WARNING: Failed to combine historical and future data: {e}")
            print("WARNING: Returning only bias-corrected future data")
            print("WARNING: This may cause issues with downstream time slicing")
            return da_adj_future  # type: ignore

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
        if current_units == "K" and target_units == "degC":
            da = da - 273.15
        elif current_units == "degC" and target_units == "K":
            da = da + 273.15
        elif current_units == "K" and target_units == "degF":
            da = (1.8 * (da - 273.15)) + 32
        elif current_units == "degC" and target_units == "degF":
            da = (1.8 * da) + 32
        elif current_units == "degF" and target_units == "K":
            da = ((da - 32) / 1.8) + 273.15
        elif current_units == "degF" and target_units == "degC":
            da = (da - 32) / 1.8
        # Add more conversions as needed

        # Update units attribute
        da.attrs["units"] = target_units

        return da
