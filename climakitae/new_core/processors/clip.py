"""
Data processor to clip data based on spatial boundaries.

This module provides the `Clip` processor class for spatial data clipping operations
in the climakitae data processing pipeline. It supports multiple clipping modes
including boundary-based clipping, coordinate-based clipping, and point-based
gridcell extraction.

Classes
-------
Clip : DataProcessor
    Main processor class for clipping operations with support for:
    - Single and multiple boundary clipping using predefined boundaries
    - Coordinate bounding box clipping
    - Single and multiple point closest-gridcell extraction
    - Custom geometry clipping from shapefiles

Clipping Modes
--------------
1. **Boundary Clipping**: Clip data using predefined administrative or utility boundaries
   - Single boundary: `Clip("CA")` or `Clip("Los Angeles County")`
   - Multiple boundaries: `Clip(["CA", "OR", "WA"])` (combined using union)
   - Supports states, counties, watersheds, utilities, and forecast zones

2. **Station Clipping**: Clip data to weather station locations
   - Single station by code: `Clip("KSAC")` - Sacramento station
   - Single station by name: `Clip("Sacramento (KSAC)")`
   - Multiple stations: `Clip(["KSAC", "KBFL", "KSFO"])` - returns closest gridcells

3. **Coordinate Clipping**: Clip data using lat/lon coordinate bounds
   - Bounding box: `Clip(((lat_min, lat_max), (lon_min, lon_max)))`
   - Single point: `Clip((lat, lon))` - returns closest gridcell
   - Multiple points: `Clip([(lat1, lon1), (lat2, lon2)])` - returns closest gridcells

4. **Custom Geometry**: Clip using custom shapefiles
   - Shapefile path: `Clip("/path/to/shapefile.shp")`

Key Features
------------
- **Smart Point Handling**: For point-based clipping, automatically searches for
  the nearest gridcell with valid (non-NaN) data within expanding search radii
- **Efficient Multi-Point Processing**: Uses vectorized operations for multiple
  points to minimize computation time
- **Duplicate Filtering**: Automatically removes duplicate gridcells when multiple
  points map to the same location
- **Comprehensive Error Handling**: Provides detailed error messages and suggestions
  for invalid boundary keys
- **Context Tracking**: Records clipping operations in dataset attributes for
  reproducibility and provenance

Data Types Supported
-------------------
- xarray.Dataset
- xarray.DataArray
- Dictionary of datasets/arrays
- Lists/tuples of datasets/arrays

Dependencies
------------
- geopandas: For geometry operations and spatial data handling
- shapely: For geometric operations and coordinate transformations
- xarray: For multi-dimensional data array operations
- rioxarray: For rasterio-based clipping operations
- pyproj: For coordinate reference system transformations
- geopy: For geodesic distance calculations

Examples
--------
>>> # Single state boundary
>>> processor = Clip("CA")
>>> clipped_data = processor.execute(dataset, context)

>>> # Multiple state boundaries (union)
>>> processor = Clip(["CA", "OR", "WA"])
>>> clipped_data = processor.execute(dataset, context)

>>> # Single station by code
>>> processor = Clip("KSAC")  # Sacramento station
>>> clipped_data = processor.execute(dataset, context)

>>> # Multiple stations
>>> processor = Clip(["KSAC", "KBFL", "KSFO"])  # Sacramento, Bakersfield, San Francisco
>>> clipped_data = processor.execute(dataset, context)

>>> # Coordinate bounding box
>>> processor = Clip(((32.0, 42.0), (-125.0, -114.0)))
>>> clipped_data = processor.execute(dataset, context)

>>> # Single point (closest gridcell)
>>> processor = Clip((37.7749, -122.4194))  # San Francisco
>>> clipped_data = processor.execute(dataset, context)

>>> # Multiple points (closest gridcells)
>>> points = [(37.7749, -122.4194), (34.0522, -118.2437)]  # SF and LA
>>> processor = Clip(points)
>>> clipped_data = processor.execute(dataset, context)

>>> # Custom shapefile
>>> processor = Clip("/path/to/custom_boundary.shp")
>>> clipped_data = processor.execute(dataset, context)

Notes
-----
- The processor requires a DataCatalog to access predefined boundary data
- Point-based clipping includes intelligent search for valid data within expanding
  radii (0.01°, 0.05°, 0.1°, 0.2°, 0.5°) if the closest gridcell contains NaN values
- Multi-point operations are optimized using vectorized calculations where possible
- All clipping operations preserve the original data's coordinate reference system
- Results include metadata about the clipping operation in the dataset attributes
"""

import logging
import os
from typing import Any, Dict, Iterable, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

# geodesic distance not used in immediate-neighborhood search
from shapely.geometry import box, mapping

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.new_core.processors.processor_utils import is_station_identifier
from climakitae.util.utils import get_closest_gridcell, get_closest_gridcells

# Module logger
logger = logging.getLogger(__name__)


@register_processor("clip", priority=200)
class Clip(DataProcessor):
    """
    Clip data based on spatial boundaries.

    This processor supports single and multiple boundary clipping operations.
    By default, multiple boundaries are combined using union operations.
    Use the ``separated`` option to keep boundaries as separate dimensions.

    Parameters
    ----------
    value : str | list | tuple | dict
        The value(s) to clip the data by. Can be:
        - str: Single boundary key, file path, or coordinate specification
        - list: Multiple boundary keys of the same category OR list of (lat, lon) tuples
        - tuple: Coordinate bounds ((lat_min, lat_max), (lon_min, lon_max)) or single (lat, lon) point
        - dict: Configuration with ``boundaries`` or ``points`` key and optional flags

    Examples
    --------
    Single boundary:
    >>> clip = Clip("CA")  # Single state
    >>> clip = Clip("Los Angeles County")  # Single county

    Multiple boundaries (union, default):
    >>> clip = Clip(["CA", "OR", "WA"])  # Multiple states combined
    >>> clip = Clip(["Los Angeles County", "Orange County"])  # Multiple counties combined

    Multiple boundaries (separated):
    >>> clip = Clip({"boundaries": ["CA", "OR", "WA"], "separated": True})
    >>> # Returns data with a 'state' dimension containing each boundary

    Coordinate bounds:
    >>> clip = Clip(((32.0, 42.0), (-125.0, -114.0)))  # lat/lon bounds

    Single point (closest gridcell):
    >>> clip = Clip((37.7749, -122.4194))  # Single lat, lon point

    Multiple points as mask (default behavior):
    >>> clip = Clip([(37.7749, -122.4194), (34.0522, -118.2437)])
    >>> # Returns gridded data with NaN everywhere except selected points
    >>> # Points without data are filled with 3x3 neighborhood average

    Multiple points extracted to dimension:
    >>> clip = Clip({"points": [(37.7749, -122.4194), (34.0522, -118.2437)], "separated": True})
    >>> # Returns data with 'points' dimension containing each location's values
    """

    def __init__(self, value, persist: bool = False):
        """
        Initialize the Clip processor.

        Parameters
        ----------
        value : str | list | tuple | dict
            The value(s) to clip the data by. Can be:
            - str: Single boundary key, file path, station code/name, or coordinate specification
            - list: Multiple boundary keys, station codes/names, or (lat, lon) tuples for multiple points
            - tuple: Coordinate bounds ((lat_min, lat_max), (lon_min, lon_max)) or single (lat, lon) point
            - dict: Configuration dict with options:
              - ``boundaries`` (list): Boundary names for boundary clipping
              - ``points`` (list): List of (lat, lon) tuples for multi-point clipping
              - ``separated`` (bool): For boundaries, keep as separate dimensions.
                For points, extract along 'points' dimension instead of returning masked grid.
              - ``persist`` (bool): Compute data to memory after clipping (recommended for
                multi-point clipping with downstream computations like 1-in-X analysis)
        persist : bool, optional
            If True, compute the clipped data to memory after clipping. This collapses
            the Dask task graph, which is critical for efficient downstream operations
            like 1-in-X analysis. For multi-point clipping with many points, the task
            graph can become very large (millions of tasks), causing OOM errors during
            subsequent computations. Default is False for backward compatibility.
        """
        # Handle dict input: extract boundaries, separated flag, and persist option
        self.separated = False
        self.boundary_names: list[str] = []
        self.dimension_name: str | None = None
        self.persist = persist

        # Flag to extract individual points along a dimension (like old multi-point behavior)
        self.extract_points = False

        if isinstance(value, dict):
            # Check for persist in dict config
            if "persist" in value:
                self.persist = value.get("persist", False)

            if "boundaries" in value:
                self.separated = value.get("separated", False)
                self.boundary_names = list(value["boundaries"])
                # Store the boundaries list as the value for processing
                value = value["boundaries"]
            elif "points" in value:
                # Multi-point clipping with dict config
                # Check for separated flag - if True, extract points along a dimension
                self.extract_points = value.get("separated", False)
                value = value["points"]
            else:
                # No recognized key - raise error
                raise ValueError(
                    "Dict input to Clip must contain 'boundaries' or 'points' key. "
                    "Example: {'points': [(lat1, lon1), (lat2, lon2)], 'persist': True}"
                )

        self.value = value
        self.name = "clip"
        self.catalog: Union[DataCatalog, object] = UNSET
        self.needs_catalog = True

        # Station-related attributes
        self.is_station = False
        self.is_multi_station = False
        self.station_info = None  # Will store station metadata

        # Check if this is a list of lat/lon tuples
        self.is_multi_point = (
            isinstance(value, list)
            and len(value) > 0
            and all(
                isinstance(item, tuple)
                and len(item) == 2
                and all(isinstance(coord, (int, float)) for coord in item)
                for item in value
            )
        )

        if self.is_multi_point:
            self.point_list = [(float(lat), float(lon)) for lat, lon in value]
            self.multi_mode = False
            self.operation = None
        else:
            # Determine if this is a multi-boundary operation
            self.multi_mode = isinstance(value, list) and len(value) > 1
            # Only use union if not separated mode
            self.operation = (
                "union" if (self.multi_mode and not self.separated) else None
            )

        # Check if this is a single point operation and store coordinates
        self.is_single_point = (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(coord, (int, float)) for coord in value)
        )
        if self.is_single_point:
            self.lat, self.lon = float(value[0]), float(value[1])
        # log initialization
        logger.debug(
            "Clip processor initialized value=%s multi_mode=%s is_multi_point=%s is_single_point=%s separated=%s",
            self.value,
            self.multi_mode,
            self.is_multi_point,
            getattr(self, "is_single_point", False),
            self.separated,
        )

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for data subsetting logic
        geom = UNSET
        match self.value:
            case str():
                # Check if this is a station identifier
                if is_station_identifier(self.value):
                    try:
                        self.lat, self.lon, self.station_info = (
                            self._get_station_coordinates(self.value)
                        )
                        self.is_station = True
                        self.is_single_point = True
                        geom = None  # Will be handled as single point
                    except (ValueError, RuntimeError) as e:
                        raise ValueError(f"Station clipping failed: {str(e)}")
                # check if string is path-like
                elif os.path.exists(self.value):
                    geom = gpd.read_file(self.value)
                else:
                    # try to find corresponding boundary key
                    geom = self._get_boundary_geometry(self.value)
            case list():
                # Check if this is a list of station identifiers
                if all(
                    isinstance(item, str) and is_station_identifier(item)
                    for item in self.value
                ):
                    try:
                        self.point_list, station_metadata_list = (
                            self._convert_stations_to_points(self.value)
                        )
                        self.is_multi_station = True
                        self.is_multi_point = True
                        self.station_info = station_metadata_list
                        geom = None  # Will be handled as multi-point
                    except (ValueError, RuntimeError) as e:
                        raise ValueError(f"Multi-station clipping failed: {str(e)}")
                # Handle multiple boundary keys (Phase 1) OR multiple lat/lon points
                elif self.is_multi_point:
                    geom = None  # Will be handled differently for multi-point
                elif self.separated:
                    # Separated mode: don't create union geometry, will clip each boundary separately
                    geom = None  # Will be handled in separated clipping
                else:
                    geom = self._get_multi_boundary_geometry(self.value)
            case tuple():
                # Check if this is a single (lat, lon) point or bounds ((lat_min, lat_max), (lon_min, lon_max))
                if len(self.value) == 2 and all(
                    isinstance(coord, (int, float)) for coord in self.value
                ):
                    # Single (lat, lon) point - use closest gridcell approach
                    geom = None  # Will be handled differently in clipping
                else:
                    # Coordinate bounds ((lat_min, lat_max), (lon_min, lon_max))
                    geom = gpd.GeoDataFrame(
                        geometry=[
                            box(
                                minx=self.value[1][0],
                                miny=self.value[0][0],
                                maxx=self.value[1][1],
                                maxy=self.value[0][1],
                            )
                        ],
                        crs=pyproj.CRS.from_epsg(4326),
                        # 4326 is WGS84 i.e. lat/lon
                    )
            case _:
                raise ValueError(
                    f"Invalid value type for clipping. Expected str, list, or tuple but got {type(self.value)}."
                )

        if (
            geom is None
            and not self.is_single_point
            and not self.is_multi_point
            and not self.separated
        ):
            raise ValueError("Failed to create geometry for clipping operation.")

        ret = None
        logger.debug(
            "Clip.execute called with value=%s result_type=%s",
            self.value,
            type(result).__name__,
        )
        match result:
            case dict():
                # Clip each dataset in the dictionary
                if self.is_single_point:
                    ret = {
                        key: self._clip_data_to_point(value, self.lat, self.lon)
                        for key, value in result.items()
                    }
                elif self.is_multi_point:
                    ret = {
                        key: self._clip_data_to_points_as_mask(
                            value, self.point_list, self.extract_points
                        )
                        for key, value in result.items()
                    }
                elif self.separated:
                    ret = {
                        key: self._clip_data_separated(value, self.value)
                        for key, value in result.items()
                    }
                else:
                    ret = {
                        key: self._clip_data_with_geom(value, geom)
                        for key, value in result.items()
                        if geom is not None
                    }
            case xr.Dataset() | xr.DataArray():
                # Clip the single dataset
                if self.is_single_point:
                    ret = self._clip_data_to_point(result, self.lat, self.lon)
                elif self.is_multi_point:
                    ret = self._clip_data_to_points_as_mask(
                        result, self.point_list, self.extract_points
                    )
                elif self.separated:
                    ret = self._clip_data_separated(result, self.value)
                elif geom is not None:
                    ret = self._clip_data_with_geom(result, geom)
            case list() | tuple():
                # return as passed type
                if self.is_single_point:
                    clipped_data = [
                        self._clip_data_to_point(data, self.lat, self.lon)
                        for data in result
                    ]
                    # Filter out None results
                    valid_data = [data for data in clipped_data if data is not None]
                    ret = type(result)(valid_data) if valid_data else None
                elif self.is_multi_point:
                    clipped_data = [
                        self._clip_data_to_points_as_mask(
                            data, self.point_list, self.extract_points
                        )
                        for data in result
                    ]
                    # Filter out None results
                    valid_data = [data for data in clipped_data if data is not None]
                    ret = type(result)(valid_data) if valid_data else None
                elif self.separated:
                    ret = type(result)(
                        [self._clip_data_separated(data, self.value) for data in result]
                    )
                elif geom is not None:
                    ret = type(result)(
                        [self._clip_data_with_geom(data, geom) for data in result]
                    )

            case _:
                raise ValueError(
                    f"Invalid result type for clipping. Expected dict, xr.Dataset, xr.DataArray, or Iterable but got {type(result)}."
                )

        if ret is None:
            raise ValueError("Clipping operation failed to produce valid results.")

        # Persist to memory if requested - this collapses the Dask task graph
        # Critical for multi-point clipping where task graph can be very large
        if self.persist:
            logger.info(
                "Persisting clipped data to memory to collapse Dask task graph..."
            )
            if isinstance(ret, dict):
                ret = {
                    k: v.compute() if hasattr(v, "compute") else v
                    for k, v in ret.items()
                }
            elif hasattr(ret, "compute"):
                ret = ret.compute()
            logger.info("Clipped data persisted to memory successfully")

        self.update_context(context)
        return ret

    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with information about the clipping operation, to be stored
        in the "new_attrs" attribute.

        Parameters
        ----------
        context : dict[str, Any]
            Parameters for processing the data.

        Note
        ----
        The context is updated in place. This method does not return anything.
        """
        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        # Add clipping information to the context
        if self.is_station and self.station_info:
            station_desc = f"{self.station_info['station_id']} - {self.station_info['station_name']}, {self.station_info['city']}, {self.station_info['state']}"
            context[_NEW_ATTRS_KEY][
                self.name
            ] = f"""Process '{self.name}' applied to the data. Station clipping was done using closest gridcell to station: {station_desc} at coordinates: ({self.lat:.4f}, {self.lon:.4f})."""
        elif self.is_multi_station and self.station_info:
            station_list = [
                f"{s['station_id']} - {s['station_name']}" for s in self.station_info
            ]
            context[_NEW_ATTRS_KEY][
                self.name
            ] = f"""Process '{self.name}' applied to the data. Multi-station clipping was done using closest gridcells to {len(self.station_info)} stations: {station_list}."""
        elif self.is_single_point:
            context[_NEW_ATTRS_KEY][
                self.name
            ] = f"""Process '{self.name}' applied to the data. Single point clipping was done using closest gridcell to coordinates: ({self.lat}, {self.lon})."""
        elif self.is_multi_point:
            if self.extract_points:
                context[_NEW_ATTRS_KEY][
                    self.name
                ] = f"""Process '{self.name}' applied to the data. Multi-point mask clipping was done: {len(self.point_list)} coordinate pairs were indexed to the grid, points without data were filled with 3x3 neighborhood average, then extracted along 'points' dimension: {self.point_list}."""
            else:
                context[_NEW_ATTRS_KEY][
                    self.name
                ] = f"""Process '{self.name}' applied to the data. Multi-point mask clipping was done: {len(self.point_list)} coordinate pairs were indexed to the grid, points without data were filled with 3x3 neighborhood average, non-selected grid cells were set to NaN. Coordinates: {self.point_list}."""
        elif self.separated:
            # Separated mode: boundaries are clipped individually along a dimension
            dim_name = self.dimension_name or "region"
            context[_NEW_ATTRS_KEY][
                self.name
            ] = f"""Process '{self.name}' applied to the data. Separated boundary clipping was done using {len(self.boundary_names)} boundaries along '{dim_name}' dimension: {self.boundary_names}."""
        elif self.multi_mode:
            context[_NEW_ATTRS_KEY][
                self.name
            ] = f"""Process '{self.name}' applied to the data. Multi-boundary clipping was done using {len(self.value)} boundaries with {self.operation} operation: {self.value}."""
        else:
            context[_NEW_ATTRS_KEY][
                self.name
            ] = f"""Process '{self.name}' applied to the data. Clipping was done using the following value: {self.value}."""

    def set_data_accessor(self, catalog: DataCatalog):
        """Set the data catalog for accessing boundary data."""
        self.catalog = catalog

    def _get_station_coordinates(
        self, station_identifier: str
    ) -> tuple[float, float, dict]:
        """
        Get lat/lon coordinates for a station identifier.

        This is a wrapper around the shared utility function in processor_utils.

        Parameters
        ----------
        station_identifier : str
            Station code (e.g., "KSAC") or full station name

        Returns
        -------
        tuple[float, float, dict]
            Latitude, longitude, and station metadata dictionary

        Raises
        ------
        ValueError
            If station is not found or catalog is not available
        """
        from climakitae.new_core.processors.processor_utils import (
            get_station_coordinates,
        )

        return get_station_coordinates(station_identifier, self.catalog)

    def _convert_stations_to_points(
        self, station_identifiers: list[str]
    ) -> tuple[list[tuple[float, float]], list[dict]]:
        """
        Convert a list of station identifiers to lat/lon coordinates.

        This is a wrapper around the shared utility function in processor_utils.

        Parameters
        ----------
        station_identifiers : list[str]
            List of station codes or names

        Returns
        -------
        tuple[list[tuple[float, float]], list[dict]]
            List of (lat, lon) tuples and list of metadata dictionaries
        """
        from climakitae.new_core.processors.processor_utils import (
            convert_stations_to_points,
        )

        return convert_stations_to_points(station_identifiers, self.catalog)

    @staticmethod
    def _clip_data_with_geom(data: xr.DataArray | xr.Dataset, gdf: gpd.GeoDataFrame):
        """
        Clip the data using a bounding box.

        Parameters
        ----------
        data : xr.Dataset | Iterable[xr.Dataset]
            The data to be clipped.
        gdf : gpd.GeoDataFrame
            The GeoDataFrame containing the geometry

        Returns
        -------
        xr.Dataset | Iterable[xr.Dataset]
            The clipped data.
        """
        # Ensure data has CRS set
        if data.rio.crs is None:
            # Check if this is WRF data with Lambert Conformal projection
            if "Lambert_Conformal" in data.coords:
                # WRF data: try spatial_ref attribute first, then build from CF attrs
                spatial_ref = data["Lambert_Conformal"].attrs.get("spatial_ref")
                if spatial_ref:
                    data = data.rio.write_crs(spatial_ref, inplace=True)
                else:
                    # Build CRS from CF convention attributes
                    attrs = data["Lambert_Conformal"].attrs
                    try:
                        crs = pyproj.CRS.from_cf(
                            {
                                "grid_mapping_name": attrs["grid_mapping_name"],
                                "latitude_of_projection_origin": attrs[
                                    "latitude_of_projection_origin"
                                ],
                                "longitude_of_central_meridian": attrs[
                                    "longitude_of_central_meridian"
                                ],
                                "standard_parallel": attrs["standard_parallel"],
                                "earth_radius": attrs["earth_radius"],
                            }
                        )
                        data = data.rio.write_crs(crs, inplace=True)
                    except KeyError as e:
                        raise ValueError(
                            f"Lambert_Conformal coordinate found but missing required "
                            f"CF convention attribute: {e}"
                        )
            else:
                # LOCA2 or other lat/lon data: use WGS84/EPSG:4326
                data = data.rio.write_crs("epsg:4326", inplace=True)

        # Ensure GeoDataFrame has CRS set (boundaries are always in EPSG:4326)
        if gdf.crs is None:
            logger.warning(
                "GeoDataFrame does not have a CRS set. Assuming EPSG:4326 (WGS84)."
            )
            gdf.set_crs("epsg:4326", inplace=True)

        # If GeoDataFrame CRS differs from data CRS, reproject it to match data
        elif gdf.crs != data.rio.crs:
            gdf = gdf.to_crs(data.rio.crs)

        return data.rio.clip(
            gdf.geometry.apply(mapping),
            gdf.crs,
            drop=True,
            all_touched=True,
        )

    @staticmethod
    def _clip_data_to_point(dataset: xr.DataArray | xr.Dataset, lat: float, lon: float):
        """
        Clip data to the closest gridcell with valid data for a single point.

        This method will search for the nearest gridcell that contains
        non-NaN values rather than just the geographically closest point.

        Parameters
        ----------
        dataset : xr.Dataset or xr.DataArray
            The dataset to clip
        lat : float
            Target latitude
        lon : float
            Target longitude

        Returns
        -------
        xr.Dataset or xr.DataArray or None
            Clipped dataset containing only the closest valid gridcell,
            or None if no valid gridcell found within search radius
        """

        # First try the original method
        closest_gridcell = get_closest_gridcell(dataset, lat, lon, print_coords=False)

        if closest_gridcell is not None:
            # Check if this gridcell has valid data (check first variable, first time step)
            first_var = next(iter(closest_gridcell.data_vars))
            test_data = closest_gridcell[first_var]

            # Identify spatial dimensions
            spatial_dims = [d for d in test_data.dims if d in ["x", "y", "lat", "lon"]]

            # Get a sample data point (first time step, first sim if available)
            # Reduce all non-spatial dimensions to a single point
            for dim in test_data.dims:
                if dim not in spatial_dims:
                    test_data = test_data.isel({dim: 0})

            if not test_data.isnull().all():
                # Log coordinates safely — not all selections will retain spatial dims
                try:
                    if len(spatial_dims) >= 2:
                        coord0 = float(
                            closest_gridcell[spatial_dims[0]].values
                            if hasattr(closest_gridcell[spatial_dims[0]], "values")
                            else closest_gridcell[spatial_dims[0]]
                        )
                        coord1 = float(
                            closest_gridcell[spatial_dims[1]].values
                            if hasattr(closest_gridcell[spatial_dims[1]], "values")
                            else closest_gridcell[spatial_dims[1]]
                        )
                        logger.info(
                            "Found valid data at closest gridcell: %s=%0.4f, %s=%0.4f",
                            spatial_dims[0],
                            coord0,
                            spatial_dims[1],
                            coord1,
                        )
                    elif (
                        "lat" in closest_gridcell.coords
                        and "lon" in closest_gridcell.coords
                    ):
                        lat_val = float(
                            closest_gridcell.coords["lat"].to_numpy().item()
                        )
                        lon_val = float(
                            closest_gridcell.coords["lon"].to_numpy().item()
                        )
                        logger.info(
                            "Found valid data at closest gridcell: lat=%0.4f, lon=%0.4f",
                            lat_val,
                            lon_val,
                        )
                    else:
                        logger.info(
                            "Found valid data at closest gridcell (coords unavailable)"
                        )
                except Exception:
                    logger.info("Found valid data at closest gridcell")
                return closest_gridcell

        # If no valid data found at closest point, search neighboring grid cells
        # using index-based nearest-dimension ids (faster and more robust than
        # repeated lat/lon slicing). This approach finds the closest grid index
        # (using x/y or lat/lon coordinates) and expands in cell-space until a
        # valid (non-NaN) cell is found within a reasonable radius.
        logger.info("Closest gridcell contains NaN values, searching nearby indices...")

        # Determine spatial dimension names
        if "x" in dataset.dims and "y" in dataset.dims:
            dim1_name, dim2_name = "x", "y"

        elif "lat" in dataset.dims and "lon" in dataset.dims:
            dim1_name, dim2_name = "lat", "lon"
        else:
            # Unknown grid layout;
            logger.warning("Unknown spatial dims, cannot search for nearest gridcell")
            return None

        # Find nearest index along each spatial dim
        try:
            coord1_vals = dataset[dim1_name].to_index()
            coord2_vals = dataset[dim2_name].to_index()
            if dim1_name in ["lat", "y"]:
                coord1_target = lat
                coord2_target = lon
            else:
                # For x/y dims transform lat/lon -> x/y coordinates
                try:
                    fwd_transformer = pyproj.Transformer.from_crs(
                        "epsg:4326", dataset.rio.crs, always_xy=True
                    )
                    coord2_target, coord1_target = fwd_transformer.transform(lon, lat)
                except Exception:
                    # Fall back to lat/lon if transform fails
                    coord1_target = lat
                    coord2_target = lon

            idx1 = coord1_vals.get_indexer([coord1_target], method="nearest")[0]
            idx2 = coord2_vals.get_indexer([coord2_target], method="nearest")[0]
        except Exception as e:
            logger.error(
                "Failed to determine nearest grid indices: %s", e, exc_info=True
            )
            return None

        # Only search the immediate 3x3 neighborhood around the nearest index
        max_i = dataset.sizes[dim1_name]
        max_j = dataset.sizes[dim2_name]

        center_i = int(idx1)
        center_j = int(idx2)
        logger.debug(
            "Searching immediate 3x3 neighborhood around indices (%d, %d)",
            center_i,
            center_j,
        )

        neighbor_cells = []

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni = center_i + di
                nj = center_j + dj
                # Skip out-of-bounds indices
                if ni < 0 or nj < 0 or ni >= max_i or nj >= max_j:
                    continue
                try:
                    cell = dataset.isel({dim1_name: ni, dim2_name: nj})

                    # Determine a DataArray to test for NaNs
                    if isinstance(cell, xr.Dataset):
                        sample_var = next(iter(cell.data_vars))
                        sample_da = cell[sample_var]
                    else:
                        sample_da = cell

                    if sample_da.isnull().all():
                        logger.debug(
                            "Skipping all-NaN cell at indices (%d, %d)", ni, nj
                        )
                        continue

                    neighbor_cells.append(cell)
                    logger.debug("Including cell at indices (%d, %d)", ni, nj)
                except Exception:
                    # Skip any candidate that raises (robust to missing coords)
                    continue

        logger.debug("Found %d valid neighbor cells", len(neighbor_cells))
        if not neighbor_cells:
            logger.warning("No valid gridcells found in immediate 3x3 neighborhood")
            return None

        try:
            logger.debug("Averaging all valid neighbor cells")
            concat = xr.concat(neighbor_cells, dim="nearest_cell")
            averaged = concat.mean(dim="nearest_cell")

            # Try to set lat/lon to mean of selected cells if available
            try:
                if "lat" in concat.coords and "lon" in concat.coords:
                    averaged = averaged.assign_coords(
                        lat=float(np.nanmean(concat["lat"].to_numpy())),
                        lon=float(np.nanmean(concat["lon"].to_numpy())),
                    )
            except Exception:
                pass

            return averaged
        except Exception:
            # Fallback to returning the center cell if averaging fails
            logger.warning("Averaging failed", exc_info=True)
            try:
                return dataset.isel({dim1_name: center_i, dim2_name: center_j})
            except Exception:
                logger.warning("Returning center cell using isel failed", exc_info=True)
                try:
                    logger.debug("Returning center cell using sel with lat/lon")
                    return dataset.sel(
                        {
                            "lat": dataset["lat"][center_i],
                            "lon": dataset["lon"][center_j],
                        }
                    )
                except Exception:
                    logger.warning(
                        "No valid gridcells found in neighborhood", exc_info=True
                    )
                    return None

    @staticmethod
    def _clip_data_to_multiple_points(
        dataset: xr.DataArray | xr.Dataset, point_list: list[tuple[float, float]]
    ):
        """
        Clip data to multiple closest gridcells using efficient vectorized operations.

        This method uses the efficient `get_closest_gridcells` function to find all
        closest gridcells at once, then renames the dimension to 'closest_cell' for
        consistency with the previous API.

        Parameters
        ----------
        dataset : xr.Dataset or xr.DataArray
            The dataset to clip
        point_list : list[tuple[float, float]]
            List of (lat, lon) tuples to clip to

        Returns
        -------
        xr.Dataset or xr.DataArray or None
            Dataset with a 'closest_cell' dimension containing closest gridcells for all points,
            or None if no valid gridcells found for any points
        """
        logger.info(
            "Processing %d points ...",
            len(point_list),
        )

        # Extract lat/lon arrays from point_list
        lats = [lat for lat, lon in point_list]
        lons = [lon for lat, lon in point_list]

        # Use the efficient vectorized approach
        try:
            # Convert DataArray to Dataset if needed since get_closest_gridcells expects Dataset
            if isinstance(dataset, xr.DataArray):
                dataset_for_clipping = dataset.to_dataset()
            else:
                dataset_for_clipping = dataset

            closest_gridcells = get_closest_gridcells(dataset_for_clipping, lats, lons)

            if closest_gridcells is None:
                logger.warning(
                    "No valid gridcells found for any of the provided points"
                )
                return None

            # Rename the 'points' dimension to 'closest_cell' for API consistency
            closest_gridcells = closest_gridcells.rename({"points": "closest_cell"})

            # Add coordinate information for the target points
            closest_gridcells = closest_gridcells.assign_coords(
                target_lats=("closest_cell", lats),
                target_lons=("closest_cell", lons),
                point_index=("closest_cell", list(range(len(point_list)))),
            )

            # Convert back to DataArray if original input was DataArray
            if isinstance(dataset, xr.DataArray):
                # Get the original variable name
                if hasattr(dataset, "name") and dataset.name:
                    closest_gridcells = closest_gridcells[dataset.name]
                else:
                    # If no name, take the first data variable
                    var_name = list(closest_gridcells.data_vars)[0]
                    closest_gridcells = closest_gridcells[var_name]

            logger.info(
                "Successfully found closest gridcells for %d points", len(point_list)
            )
            return closest_gridcells

        except Exception as e:
            logger.error(
                "Error in vectorized multi-point clipping: %s", e, exc_info=True
            )
            logger.info("Falling back to individual point processing...")

            # Fallback to the original approach if vectorized fails
            return Clip._clip_data_to_multiple_points_fallback(dataset, point_list)

    @staticmethod
    def _clip_data_to_multiple_points_fallback(
        dataset: xr.DataArray | xr.Dataset, point_list: list[tuple[float, float]]
    ):
        """
        Fallback method for multiple point clipping using individual point processing.

        This is the original implementation kept as a fallback in case the vectorized
        approach fails for any reason.
        """
        clipped_results = []
        valid_indices = []

        for i, (lat, lon) in enumerate(point_list):
            logger.debug(
                "Processing point %d/%d: (%0.4f, %0.4f)",
                i + 1,
                len(point_list),
                lat,
                lon,
            )

            # Clip to single point
            clipped_data = Clip._clip_data_to_point(dataset, lat, lon)

            if clipped_data is not None:
                # Add point information as coordinates
                clipped_data = clipped_data.assign_coords(
                    target_lat=lat, target_lon=lon, point_index=i
                )
                clipped_results.append(clipped_data)
                valid_indices.append(i)
            else:
                logger.warning("No valid data found for point (%0.4f, %0.4f)", lat, lon)

        if not clipped_results:
            logger.warning("No valid gridcells found for any of the provided points")
            return None

        # Filter by unique grid cells before concatenating
        unique_results = []
        unique_indices = []
        seen_gridcells = set()

        for i, (clipped_data, orig_idx) in enumerate(
            zip(clipped_results, valid_indices)
        ):
            # Get the actual grid cell coordinates (rounded to avoid floating point precision issues)
            actual_lat = float(clipped_data.lat.values)
            actual_lon = float(clipped_data.lon.values)
            gridcell_key = (round(actual_lat, 6), round(actual_lon, 6))

            if gridcell_key not in seen_gridcells:
                seen_gridcells.add(gridcell_key)
                unique_results.append(clipped_data)
                unique_indices.append(orig_idx)
            else:
                target_lat = point_list[orig_idx][0]
                target_lon = point_list[orig_idx][1]
                logger.debug(
                    "Skipping duplicate grid cell at (%0.4f, %0.4f) for target point (%0.4f, %0.4f)",
                    actual_lat,
                    actual_lon,
                    target_lat,
                    target_lon,
                )

        if not unique_results:
            logger.warning("No unique gridcells found after filtering")
            return None

        try:
            # Concatenate all unique results along a new dimension
            concatenated = xr.concat(unique_results, dim="closest_cell")

            # Add coordinate information
            concatenated = concatenated.assign_coords(
                closest_cell=unique_indices,
                target_lats=(
                    "closest_cell",
                    [point_list[i][0] for i in unique_indices],
                ),
                target_lons=(
                    "closest_cell",
                    [point_list[i][1] for i in unique_indices],
                ),
            )

            logger.info(
                "Successfully concatenated %d unique closest gridcells",
                len(unique_results),
            )
            return concatenated

        except Exception as e:
            logger.error("Error concatenating results: %s", e, exc_info=True)
            return None

    @staticmethod
    def _clip_data_to_points_as_mask(
        dataset: xr.DataArray | xr.Dataset,
        point_list: list[tuple[float, float]],
        extract_points: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply a point-based mask to gridded data, preserving the grid structure.

        This method treats the list of lat/lon points as a spatial mask:
        1. Index each lat/lon to the closest grid cell
        2. For grid cells without valid data, fill with 3x3 neighborhood average
        3. Set all non-selected grid cells to NaN
        4. Return the masked gridded dataset

        If extract_points=True, additionally collapses the spatial dimensions to
        a "points" dimension, returning the selected points as a 1D series.

        Parameters
        ----------
        dataset : xr.Dataset or xr.DataArray
            The dataset to mask
        point_list : list[tuple[float, float]]
            List of (lat, lon) tuples defining the mask
        extract_points : bool, optional
            If True, collapse spatial dimensions to a "points" dimension.
            Default is False (return masked gridded data).

        Returns
        -------
        xr.Dataset or xr.DataArray
            If extract_points=False: Masked gridded dataset with NaN for non-selected cells
            If extract_points=True: Dataset with 'points' dimension containing selected values
        """
        logger.info(
            "Applying point mask to data with %d points (extract_points=%s)",
            len(point_list),
            extract_points,
        )

        # Identify spatial dimensions
        if "x" in dataset.dims and "y" in dataset.dims:
            lat_dim, lon_dim = "y", "x"
        elif "lat" in dataset.dims and "lon" in dataset.dims:
            lat_dim, lon_dim = "lat", "lon"
        else:
            raise ValueError(
                f"Cannot identify spatial dimensions. Dataset has dims: {dataset.dims}"
            )

        # Get coordinate arrays
        lat_coords = dataset[lat_dim].values
        lon_coords = dataset[lon_dim].values

        # Determine if we need to transform coordinates (for projected grids)
        needs_transform = lat_dim in ["y", "x"]
        if needs_transform:
            try:
                fwd_transformer = pyproj.Transformer.from_crs(
                    "epsg:4326", dataset.rio.crs, always_xy=True
                )
            except Exception as e:
                logger.warning(
                    "Could not create coordinate transformer: %s. Using lat/lon directly.",
                    e,
                )
                needs_transform = False

        # Convert point coordinates and find nearest grid indices
        lat_indices = []
        lon_indices = []
        valid_points = []

        for lat, lon in point_list:
            # Transform coordinates if needed
            if needs_transform:
                try:
                    x_target, y_target = fwd_transformer.transform(lon, lat)
                    lat_target, lon_target = y_target, x_target
                except Exception:
                    lat_target, lon_target = lat, lon
            else:
                lat_target, lon_target = lat, lon

            # Find nearest indices
            lat_idx = np.abs(lat_coords - lat_target).argmin()
            lon_idx = np.abs(lon_coords - lon_target).argmin()

            lat_indices.append(lat_idx)
            lon_indices.append(lon_idx)
            valid_points.append((lat, lon))

        # Create unique set of (lat_idx, lon_idx) pairs to avoid duplicates
        unique_indices = list(set(zip(lat_indices, lon_indices)))
        logger.debug(
            "Found %d unique grid cells from %d input points",
            len(unique_indices),
            len(point_list),
        )

        # Get a sample variable to check for NaN values and create mask
        if isinstance(dataset, xr.Dataset):
            first_var = next(iter(dataset.data_vars))
            sample_data = dataset[first_var]
        else:
            sample_data = dataset

        # Reduce to 2D for mask creation (take first time step, first simulation, etc.)
        sample_2d = sample_data
        for dim in list(sample_2d.dims):
            if dim not in [lat_dim, lon_dim]:
                sample_2d = sample_2d.isel({dim: 0})

        # Compute if dask-backed (this is just 2D, so should be cheap)
        if hasattr(sample_2d.data, "compute"):
            sample_2d = sample_2d.compute()

        # Create the point mask (True where we want data)
        mask = np.zeros((len(lat_coords), len(lon_coords)), dtype=bool)
        point_values_valid = np.zeros((len(lat_coords), len(lon_coords)), dtype=bool)

        # Track which points need 3x3 averaging due to NaN at exact location
        needs_averaging = []

        for lat_idx, lon_idx in unique_indices:
            # Check if this grid cell has valid data
            cell_value = sample_2d.values[lat_idx, lon_idx]
            if np.isnan(cell_value):
                needs_averaging.append((lat_idx, lon_idx))
            else:
                point_values_valid[lat_idx, lon_idx] = True
            mask[lat_idx, lon_idx] = True

        logger.debug(
            "%d points have valid data, %d need 3x3 averaging",
            np.sum(point_values_valid),
            len(needs_averaging),
        )

        # For points needing averaging, compute 3x3 neighborhood average
        # We'll handle this by creating a filled version of the data
        lat_size = len(lat_coords)
        lon_size = len(lon_coords)

        # Create xarray mask for broadcasting
        mask_da = xr.DataArray(
            mask,
            dims=[lat_dim, lon_dim],
            coords={lat_dim: lat_coords, lon_dim: lon_coords},
        )

        # Apply the mask - set non-selected cells to NaN
        masked_data = dataset.where(mask_da)

        # Handle points that need 3x3 averaging
        if needs_averaging:
            logger.info(
                "Computing 3x3 neighborhood averages for %d points",
                len(needs_averaging),
            )

            for lat_idx, lon_idx in needs_averaging:
                # Define 3x3 neighborhood bounds
                lat_min = max(0, lat_idx - 1)
                lat_max = min(lat_size, lat_idx + 2)
                lon_min = max(0, lon_idx - 1)
                lon_max = min(lon_size, lon_idx + 2)

                # Extract neighborhood
                neighborhood = dataset.isel(
                    {lat_dim: slice(lat_min, lat_max), lon_dim: slice(lon_min, lon_max)}
                )

                # Compute mean of valid cells in neighborhood
                neighborhood_mean = neighborhood.mean(dim=[lat_dim, lon_dim])

                # Insert the averaged value at the target location
                # We need to do this for each variable if Dataset
                if isinstance(masked_data, xr.Dataset):
                    for var in masked_data.data_vars:
                        # Create indexer for this specific cell
                        masked_data[var].loc[
                            {lat_dim: lat_coords[lat_idx], lon_dim: lon_coords[lon_idx]}
                        ] = neighborhood_mean[var]
                else:
                    masked_data.loc[
                        {lat_dim: lat_coords[lat_idx], lon_dim: lon_coords[lon_idx]}
                    ] = neighborhood_mean

        # Clip to bounding box of the selected points (with padding for context)
        # This makes plotting and analysis more manageable
        selected_lat_indices = [idx[0] for idx in unique_indices]
        selected_lon_indices = [idx[1] for idx in unique_indices]

        # Add 1-cell padding around bounding box for visualization context
        bbox_padding = 1
        lat_min_idx = max(0, min(selected_lat_indices) - bbox_padding)
        lat_max_idx = min(lat_size - 1, max(selected_lat_indices) + bbox_padding)
        lon_min_idx = max(0, min(selected_lon_indices) - bbox_padding)
        lon_max_idx = min(lon_size - 1, max(selected_lon_indices) + bbox_padding)

        # Clip to bounding box
        masked_data = masked_data.isel(
            {
                lat_dim: slice(lat_min_idx, lat_max_idx + 1),
                lon_dim: slice(lon_min_idx, lon_max_idx + 1),
            }
        )
        logger.debug(
            "Clipped to bounding box: %s[%d:%d], %s[%d:%d]",
            lat_dim,
            lat_min_idx,
            lat_max_idx + 1,
            lon_dim,
            lon_min_idx,
            lon_max_idx + 1,
        )

        # If extract_points is True, collapse spatial dims to points dimension
        if extract_points:
            logger.info(
                "Extracting %d unique points along 'points' dimension",
                len(unique_indices),
            )

            # Extract values at each unique point from the ORIGINAL masked data
            # (before bbox clip) to get correct indices
            point_datasets = []
            point_coords_lat = []
            point_coords_lon = []

            # Use coordinate-based selection since we've clipped the data
            clipped_lat_coords = masked_data[lat_dim].values
            clipped_lon_coords = masked_data[lon_dim].values

            # Create a mapping from (lat_idx, lon_idx) to the original point_list lat/lon
            idx_to_latlon = {}
            for lat, lon in point_list:
                # Transform coordinates if needed (same logic as above)
                if needs_transform:
                    try:
                        x_target, y_target = fwd_transformer.transform(lon, lat)
                        lat_target, lon_target = y_target, x_target
                    except Exception:
                        lat_target, lon_target = lat, lon
                else:
                    lat_target, lon_target = lat, lon

                # Find nearest indices
                lat_idx = np.abs(lat_coords - lat_target).argmin()
                lon_idx = np.abs(lon_coords - lon_target).argmin()
                idx_to_latlon[(lat_idx, lon_idx)] = (lat, lon)

            for lat_idx, lon_idx in unique_indices:
                target_grid_lat = lat_coords[lat_idx]
                target_grid_lon = lon_coords[lon_idx]
                point_data = masked_data.sel(
                    {lat_dim: target_grid_lat, lon_dim: target_grid_lon},
                    method="nearest",
                )
                point_datasets.append(point_data)

                # Use the original geographic lat/lon from point_list, not grid coordinates
                original_lat, original_lon = idx_to_latlon.get(
                    (lat_idx, lon_idx), (target_grid_lat, target_grid_lon)
                )
                point_coords_lat.append(float(original_lat))
                point_coords_lon.append(float(original_lon))

            # Concatenate along new 'points' dimension
            result = xr.concat(point_datasets, dim="points")

            # Add coordinate information
            result = result.assign_coords(
                point_lat=("points", point_coords_lat),
                point_lon=("points", point_coords_lon),
                point_index=("points", list(range(len(unique_indices)))),
            )

            logger.info("Successfully extracted %d points", len(unique_indices))
            return result

        logger.info(
            "Successfully created masked grid with %d selected cells (bbox clipped)",
            len(unique_indices),
        )
        return masked_data

    def _clip_data_separated(
        self,
        data: xr.Dataset | xr.DataArray,
        boundary_keys: list[str],
    ) -> xr.Dataset | xr.DataArray:
        """
        Clip data to multiple boundaries, keeping each boundary as a separate dimension.

        Instead of merging all boundaries into one geometry (union), this method
        clips the data to each boundary individually and concatenates the results
        along a new dimension with a name inferred from the boundary category.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to clip
        boundary_keys : list[str]
            List of boundary keys to clip to (e.g., ["CA", "OR", "WA"])

        Returns
        -------
        xr.Dataset | xr.DataArray
            Data with a new dimension for each boundary, containing the clipped data

        Raises
        ------
        ValueError
            If no valid clipped data is produced for any boundary
        """
        logger.info(
            "Clipping data to %d boundaries with separated mode", len(boundary_keys)
        )

        # Infer the dimension name from the first boundary
        if not self.dimension_name:
            self.dimension_name = self._infer_dimension_name(boundary_keys[0])
        dim_name = self.dimension_name

        logger.debug("Using dimension name: %s", dim_name)

        # --- Batch geometry retrieval optimization ---
        # Get boundary_dict once (cached) and determine category from first key
        boundary_dict = self.catalog.boundaries.boundary_dict()  # type: ignore
        category = self._get_boundary_category(boundary_keys[0])

        if category is None:
            raise ValueError(
                f"Could not determine category for boundary key: {boundary_keys[0]}"
            )

        # Get the lookup dict for this category and the full DataFrame
        lookup = boundary_dict.get(category, {})
        boundaries = self.catalog.boundaries  # type: ignore
        category_df_map = {
            "states": boundaries._us_states,
            "CA counties": boundaries._ca_counties,
            "CA watersheds": boundaries._ca_watersheds,
            "CA Electric Load Serving Entities (IOU & POU)": boundaries._ca_utilities,
            "CA Electricity Demand Forecast Zones": boundaries._ca_forecast_zones,
            "CA Electric Balancing Authority Areas": boundaries._ca_electric_balancing_areas,
            "CA Census Tracts": boundaries._ca_census_tracts,
        }
        category_df = category_df_map.get(category)

        if category_df is None:
            raise ValueError(f"Unknown boundary category: {category}")

        # Validate all keys and get their indices in one pass
        valid_keys = []
        indices = []
        for key in boundary_keys:
            if key in lookup:
                valid_keys.append(key)
                indices.append(lookup[key])
            else:
                logger.warning(
                    "Boundary key '%s' not found in %s, skipping", key, category
                )

        if not indices:
            raise ValueError(
                f"No valid boundary keys found in category '{category}': {boundary_keys}"
            )

        # Batch retrieve all geometries at once
        all_geometries = category_df.loc[indices]
        if not isinstance(all_geometries, gpd.GeoDataFrame):
            all_geometries = gpd.GeoDataFrame(all_geometries)
        if all_geometries.crs is None:
            all_geometries = all_geometries.set_crs(epsg=4326)

        logger.debug(
            "Batch retrieved %d geometries from %s", len(all_geometries), category
        )

        # --- End batch optimization ---

        clipped_results = []
        valid_boundary_names = []

        for boundary_key, (idx, geom_row) in zip(valid_keys, all_geometries.iterrows()):
            try:
                # Create single-row GeoDataFrame for clipping
                single_geom = gpd.GeoDataFrame([geom_row], crs=all_geometries.crs)

                # Clip the data
                clipped = self._clip_data_with_geom(data, single_geom)

                if clipped is not None:
                    clipped_results.append(clipped)
                    valid_boundary_names.append(boundary_key)
                    logger.debug(
                        "Successfully clipped data for boundary: %s", boundary_key
                    )
                else:
                    logger.warning(
                        "No valid data after clipping for boundary: %s", boundary_key
                    )

            except Exception as e:
                logger.error("Error clipping boundary '%s': %s", boundary_key, e)
                # Continue with other boundaries
                continue

        if not clipped_results:
            raise ValueError(
                f"No valid clipped data produced for any of the boundaries: {boundary_keys}"
            )

        # Concatenate all results along the new dimension
        try:
            concatenated = xr.concat(clipped_results, dim=dim_name)

            # Add the boundary names as coordinates
            concatenated = concatenated.assign_coords({dim_name: valid_boundary_names})

            logger.info(
                "Successfully created separated clip with %d boundaries along '%s' dimension",
                len(valid_boundary_names),
                dim_name,
            )

            # Store boundary names for context update
            self.boundary_names = valid_boundary_names

            return concatenated

        except Exception as e:
            raise ValueError(
                f"Failed to concatenate separated clip results: {e}"
            ) from e

    def _get_boundary_geometry(self, boundary_key: str) -> gpd.GeoDataFrame:
        """
        Get geometry data for a boundary key from the boundaries catalog.

        Parameters
        ----------
        boundary_key : str
            The boundary key to look up (e.g., "CA", "Los Angeles County", "PG&E")

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the geometry for the specified boundary

        Raises
        ------
        ValueError
            If the boundary key is not found in any boundary category
        RuntimeError
            If the catalog is not set or boundaries are not available
        """
        if self.catalog is UNSET or not isinstance(self.catalog, DataCatalog):
            raise RuntimeError(
                "DataCatalog is not set. Cannot access boundary data for clipping."
            )

        # Get all available boundary categories and their lookup dictionaries
        try:
            boundary_dict = self.catalog.boundaries.boundary_dict()  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to access boundary data: {e}") from e

        # Search for the boundary key in all categories
        found_category = None
        found_index = None

        for category, lookups in boundary_dict.items():
            # Skip special categories that don't have actual boundary data
            if category in ["none", "lat/lon"]:
                continue

            # Check if the boundary key exists in this category's lookup
            if boundary_key in lookups:
                found_category = category
                found_index = lookups[boundary_key]
                break

        if found_category is None or found_index is None:
            # Provide helpful error message with available options
            available_keys = []
            for category, lookups in boundary_dict.items():
                if category not in ["none", "lat/lon"]:
                    available_keys.append(f"{category}: {list(lookups.keys())}")

            raise ValueError(
                f"Boundary key '{boundary_key}' not found in any boundary category. "
                f"Available boundary keys:\n" + "\n".join(available_keys)
            )

        # Get the appropriate DataFrame and extract the geometry
        return self._extract_geometry_from_category(found_category, found_index)

    def _extract_geometry_from_category(
        self, category: str, index: int
    ) -> gpd.GeoDataFrame:
        """
        Extract geometry from a specific boundary category at the given index.

        Parameters
        ----------
        category : str
            The boundary category name
        index : int
            The index of the boundary in the category's DataFrame

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the geometry for the specified boundary
        """
        boundaries = self.catalog.boundaries  # type: ignore

        # Map category names to the appropriate DataFrame properties
        category_map = {
            "states": boundaries._us_states,
            "CA counties": boundaries._ca_counties,
            "CA watersheds": boundaries._ca_watersheds,
            "CA Electric Load Serving Entities (IOU & POU)": boundaries._ca_utilities,
            "CA Electricity Demand Forecast Zones": boundaries._ca_forecast_zones,
            "CA Electric Balancing Authority Areas": boundaries._ca_electric_balancing_areas,
            "CA Census Tracts": boundaries._ca_census_tracts,
        }

        if category not in category_map:
            raise ValueError(f"Unknown boundary category: {category}")

        # Get the DataFrame for this category
        df = category_map[category]

        # Extract the specific row as a GeoDataFrame
        if index not in df.index:
            raise ValueError(f"Index {index} not found in {category} data")

        geometry_row = df.loc[[index]]

        # Ensure it's a GeoDataFrame with proper CRS
        if not isinstance(geometry_row, gpd.GeoDataFrame):
            geometry_row = gpd.GeoDataFrame(geometry_row)

        # Set CRS if not already set (most boundary data should be in WGS84)
        if geometry_row.crs is None:
            geometry_row.set_crs(epsg=4326, inplace=True)

        return geometry_row

    def _infer_dimension_name(self, boundary_key: str) -> str:
        """
        Infer an appropriate dimension name based on the boundary category.

        This method determines the dimension name for separated clipping based
        on the category of the boundary key (e.g., "state" for US states,
        "county" for CA counties).

        Parameters
        ----------
        boundary_key : str
            A boundary key to determine the category from

        Returns
        -------
        str
            The inferred dimension name (e.g., "state", "county", "watershed")
        """
        # Map category names to dimension names
        category_to_dimension = {
            "states": "state",
            "CA counties": "county",
            "CA watersheds": "watershed",
            "CA Electric Load Serving Entities (IOU & POU)": "utility",
            "CA Electricity Demand Forecast Zones": "forecast_zone",
            "CA Electric Balancing Authority Areas": "balancing_area",
            "CA Census Tracts": "census_tract",
        }

        # Get the category for this boundary key
        validation = self.validate_boundary_key(boundary_key)
        if validation.get("valid") and "category" in validation:
            category = validation["category"]
            return category_to_dimension.get(category, "region")

        # Default fallback
        return "region"

    def _get_boundary_category(self, boundary_key: str) -> str | None:
        """
        Get the category for a boundary key.

        Parameters
        ----------
        boundary_key : str
            The boundary key to look up

        Returns
        -------
        str | None
            The category name, or None if not found
        """
        validation = self.validate_boundary_key(boundary_key)
        if validation.get("valid"):
            return validation.get("category")
        return None

    def validate_boundary_key(self, boundary_key: str) -> Dict[str, Any]:
        """
        Validate if a boundary key exists and return information about it.

        Parameters
        ----------
        boundary_key : str
            The boundary key to validate

        Returns
        -------
        Dict[str, Any]
            Dictionary containing validation results:
            - 'valid': bool, whether the key is valid
            - 'category': str, the category if found
            - 'suggestions': list, similar keys if not found
        """
        if self.catalog is UNSET:
            return {
                "valid": False,
                "error": "DataCatalog is not set",
                "suggestions": [],
            }

        try:
            boundary_dict = self.catalog.boundaries.boundary_dict()  # type: ignore
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to access boundary data: {e}",
                "suggestions": [],
            }

        # Check if key exists in any category
        for category, lookups in boundary_dict.items():
            if category not in ["none", "lat/lon"] and boundary_key in lookups:
                return {"valid": True, "category": category, "suggestions": []}

        # If not found, provide suggestions based on partial matches
        suggestions = []
        for category, lookups in boundary_dict.items():
            if category not in ["none", "lat/lon"]:
                for key, value in lookups.items():
                    # Check if search term matches key or value (case-insensitive)
                    key_match = (
                        boundary_key.lower() in key.lower()
                        or key.lower() in boundary_key.lower()
                    )
                    value_match = False
                    if isinstance(value, str):
                        value_match = (
                            boundary_key.lower() in value.lower()
                            or value.lower() in boundary_key.lower()
                        )

                    if key_match or value_match:
                        # Include both key and value in suggestion for clarity
                        if isinstance(value, str) and value != key:
                            suggestions.append(f"{category}: {key} ({value})")
                        else:
                            suggestions.append(f"{category}: {key}")

        return {
            "valid": False,
            "error": f"Boundary key '{boundary_key}' not found",
            "suggestions": suggestions[:10],  # Limit to top 10 suggestions
        }

    def _get_multi_boundary_geometry(self, boundary_keys: list) -> gpd.GeoDataFrame:
        """
        Get combined geometry for multiple boundary keys.

        This method handles multiple boundaries of the same category by:
        1. Validating each boundary key exists
        2. Retrieving individual geometries
        3. Combining them using union operation (default)

        Parameters
        ----------
        boundary_keys : list
            List of boundary keys to combine (e.g., ["CA", "OR", "WA"])

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing the combined geometry

        Raises
        ------
        ValueError
            If any boundary key is invalid
        RuntimeError
            If catalog is not available
        """
        if not boundary_keys:
            raise ValueError("Empty list provided for multi-boundary clipping")

        if len(boundary_keys) == 1:
            # If only one boundary, use single boundary method
            return self._get_boundary_geometry(boundary_keys[0])

        # Validate all boundary keys first
        invalid_keys = []
        valid_geometries = []

        for key in boundary_keys:
            validation = self.validate_boundary_key(key)
            if not validation["valid"]:
                invalid_keys.append(key)
            else:
                # Get the geometry for this valid key
                try:
                    geom = self._get_boundary_geometry(key)
                    valid_geometries.append(geom)
                except Exception:
                    invalid_keys.append(key)

        # Report any invalid keys
        if invalid_keys:
            suggestions = []
            for key in invalid_keys:
                validation = self.validate_boundary_key(key)
                suggestions.extend(validation.get("suggestions", []))

            error_msg = f"Invalid boundary keys: {invalid_keys}"
            if suggestions:
                error_msg += f"\nSuggestions: {suggestions[:5]}"  # Limit suggestions
            error_msg += "\nTo see all available boundaries, use: clip_processor.print_available_boundaries()"
            raise ValueError(error_msg)

        if not valid_geometries:
            raise ValueError("No valid geometries found for the provided boundary keys")

        # Combine all geometries using union operation
        return self._combine_geometries(valid_geometries, operation="union")

    def _combine_geometries(
        self, geometries: list, operation: str = "union"
    ) -> gpd.GeoDataFrame:
        """
        Combine multiple geometries using the specified operation.

        Parameters
        ----------
        geometries : list
            List of GeoDataFrames to combine
        operation : str, default "union"
            Operation to use for combining geometries ("union" supported in Phase 1)

        Returns
        -------
        gpd.GeoDataFrame
            Combined geometry as a GeoDataFrame

        Raises
        ------
        ValueError
            If operation is not supported or geometries list is empty
        """
        if not geometries:
            raise ValueError("No geometries provided for combination")

        if len(geometries) == 1:
            return geometries[0]

        if operation != "union":
            raise ValueError(
                f"Operation '{operation}' not supported in Phase 1. Only 'union' is supported."
            )

        # Ensure consistent CRS - use the CRS from the first geometry
        # Convert all geometries to reference CRS BEFORE concatenation
        reference_crs = geometries[0].crs
        if reference_crs is not None:
            normalized_geometries = []
            for geom in geometries:
                if geom.crs != reference_crs:
                    normalized_geometries.append(geom.to_crs(reference_crs))
                else:
                    normalized_geometries.append(geom)
        else:
            normalized_geometries = geometries

        # Concatenate all geometries
        try:
            combined_df = gpd.GeoDataFrame(
                pd.concat(normalized_geometries, ignore_index=True)
            )
        except Exception as e:
            raise ValueError(f"Failed to concatenate geometries: {e}")

        # Perform union operation
        try:
            if len(combined_df) > 1:
                # Use unary_union to combine all geometries
                union_geom = combined_df.unary_union
                result = gpd.GeoDataFrame(geometry=[union_geom], crs=combined_df.crs)
            else:
                result = combined_df
        except Exception as e:
            raise ValueError(f"Failed to perform union operation on geometries: {e}")

        return result
