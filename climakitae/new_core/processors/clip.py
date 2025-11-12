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

import os
import warnings
from typing import Any, Dict, Iterable, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from geopy.distance import geodesic
from shapely.geometry import box, mapping

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.new_core.processors.processor_utils import (
    find_station_match,
    is_station_identifier,
)
from climakitae.util.utils import get_closest_gridcell, get_closest_gridcells


@register_processor("clip", priority=200)
class Clip(DataProcessor):
    """
    Clip data based on spatial boundaries.

    This processor supports single and multiple boundary clipping operations.
    In Phase 1, it supports multiple boundaries of the same category using
    union operations to combine geometries.

    Parameters
    ----------
    value : str | list | tuple
        The value(s) to clip the data by. Can be:
        - str: Single boundary key, file path, or coordinate specification
        - list: Multiple boundary keys of the same category (Phase 1) OR list of (lat, lon) tuples for multiple points
        - tuple: Coordinate bounds ((lat_min, lat_max), (lon_min, lon_max)) or a single (lat, lon) point


    Examples
    --------
    Single boundary:
    >>> clip = Clip("CA")  # Single state
    >>> clip = Clip("Los Angeles County")  # Single county

    Multiple boundaries (Phase 1):
    >>> clip = Clip(["CA", "OR", "WA"])  # Multiple states
    >>> clip = Clip(["Los Angeles County", "Orange County"])  # Multiple counties

    Coordinate bounds:
    >>> clip = Clip(((32.0, 42.0), (-125.0, -114.0)))  # lat/lon bounds

    Single point (closest gridcell):
    >>> clip = Clip((37.7749, -122.4194))  # Single lat, lon point

    Multiple points (closest gridcells):
    >>> clip = Clip([(37.7749, -122.4194), (34.0522, -118.2437)])  # Multiple lat, lon points
    """

    def __init__(self, value):
        """
        Initialize the Clip processor.

        Parameters
        ----------
        value : str | list | tuple
            The value(s) to clip the data by. Can be:
            - str: Single boundary key, file path, station code/name, or coordinate specification
            - list: Multiple boundary keys, station codes/names, or (lat, lon) tuples for multiple points
            - tuple: Coordinate bounds ((lat_min, lat_max), (lon_min, lon_max)) or single (lat, lon) point
        """
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
            self.operation = "union" if self.multi_mode else None

        # Check if this is a single point operation and store coordinates
        self.is_single_point = (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(coord, (int, float)) for coord in value)
        )
        if self.is_single_point:
            self.lat, self.lon = float(value[0]), float(value[1])

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

        if geom is None and not self.is_single_point and not self.is_multi_point:
            raise ValueError("Failed to create geometry for clipping operation.")

        ret = None
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
                        key: self._clip_data_to_multiple_points(value, self.point_list)
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
                    ret = self._clip_data_to_multiple_points(result, self.point_list)
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
                        self._clip_data_to_multiple_points(data, self.point_list)
                        for data in result
                    ]
                    # Filter out None results
                    valid_data = [data for data in clipped_data if data is not None]
                    ret = type(result)(valid_data) if valid_data else None
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
            context[_NEW_ATTRS_KEY][
                self.name
            ] = f"""Process '{self.name}' applied to the data. Multi-point clipping was done using closest gridcells to {len(self.point_list)} coordinate pairs, filtered for unique grid cells, and concatenated along 'closest_cell' dimension: {self.point_list}."""
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
        if self.catalog is UNSET or not isinstance(self.catalog, DataCatalog):
            raise RuntimeError(
                "DataCatalog is not set. Cannot access station data for clipping."
            )

        stations_df = self.catalog["stations"]

        if stations_df is None or len(stations_df) == 0:
            raise RuntimeError("Station data is not available in the catalog.")

        # Use the generalized matching logic
        match = find_station_match(station_identifier, stations_df)

        if len(match) == 0:
            # Station not found - provide suggestions
            all_stations = stations_df["ID"].tolist() + stations_df["station"].tolist()
            from climakitae.new_core.param_validation.param_validation_tools import (
                _get_closest_options,
            )

            suggestions = _get_closest_options(
                station_identifier, all_stations, cutoff=0.5
            )

            error_msg = f"Station '{station_identifier}' not found in station database."
            if suggestions:
                error_msg += "\n\nDid you mean one of these?\n  - " + "\n  - ".join(
                    suggestions[:5]
                )
            error_msg += (
                "\n\nTo see all available stations, use: cd.show_station_options()"
            )

            raise ValueError(error_msg)

        if len(match) > 1:
            # Multiple matches found - show options
            station_list = match[["ID", "station", "city", "state"]].to_string(
                index=False
            )
            raise ValueError(
                f"Multiple stations match '{station_identifier}':\n{station_list}\n\n"
                f"Please use a more specific identifier."
            )

        # Extract coordinates and metadata
        station_row = match.iloc[0]
        lat = float(station_row["LAT_Y"])
        lon = float(station_row["LON_X"])

        metadata = {
            "station_id": station_row["ID"],
            "station_name": station_row["station"],
            "city": station_row["city"],
            "state": station_row["state"],
            "elevation": station_row.get("elevation", None),
        }

        return lat, lon, metadata

    def _convert_stations_to_points(
        self, station_identifiers: list[str]
    ) -> tuple[list[tuple[float, float]], list[dict]]:
        """
        Convert a list of station identifiers to lat/lon coordinates.

        Parameters
        ----------
        station_identifiers : list[str]
            List of station codes or names

        Returns
        -------
        tuple[list[tuple[float, float]], list[dict]]
            List of (lat, lon) tuples and list of metadata dictionaries
        """
        points = []
        metadata_list = []

        for station_id in station_identifiers:
            try:
                lat, lon, metadata = self._get_station_coordinates(station_id)
                points.append((lat, lon))
                metadata_list.append(metadata)
            except ValueError as e:
                # Re-raise with context about which station failed
                raise ValueError(f"Error processing station '{station_id}': {str(e)}")

        return points, metadata_list

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
                # WRF data: use Lambert Conformal projection from spatial_ref attribute
                spatial_ref = data["Lambert_Conformal"].attrs.get("spatial_ref")
                if spatial_ref:
                    data = data.rio.write_crs(spatial_ref, inplace=True)
                else:
                    raise ValueError(
                        "Lambert_Conformal coordinate found but missing spatial_ref attribute"
                    )
            else:
                # LOCA2 or other lat/lon data: use WGS84/EPSG:4326
                data = data.rio.write_crs("epsg:4326", inplace=True)

        # Ensure GeoDataFrame has CRS set (boundaries are always in EPSG:4326)
        if gdf.crs is None:
            warnings.warn(
                "GeoDataFrame does not have a CRS set. Assuming EPSG:4326 (WGS84).",
                UserWarning,
                stacklevel=999,
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
                print(
                    f"Found valid data at closest gridcell: lat={closest_gridcell.lat.values:.4f}, lon={closest_gridcell.lon.values:.4f}"
                )
                return closest_gridcell

        # If no valid data found at closest point, search in expanding radius
        print(
            f"Closest gridcell contains NaN values, searching for nearest valid gridcell..."
        )

        # Define search radii (in degrees)
        search_radii = [0.01, 0.05, 0.1, 0.2, 0.5]

        for radius in search_radii:
            print(f"Searching within {radius}° radius...")

            # Create a larger bounding box
            lat_min, lat_max = lat - radius, lat + radius
            lon_min, lon_max = lon - radius, lon + radius

            # Clip to bounding box
            try:
                larger_region = dataset.sel(
                    lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
                )

                # Determine spatial dimensions dynamically
                spatial_dims = []
                for dim in larger_region.dims:
                    if dim in ["x", "y", "lat", "lon"]:
                        spatial_dims.append(dim)

                # Check if we have any spatial data
                if len(spatial_dims) == 0 or any(
                    larger_region.sizes.get(dim, 0) == 0 for dim in spatial_dims
                ):
                    continue

                # Find valid gridcells in this region
                first_var = next(iter(larger_region.data_vars))
                test_data = larger_region[first_var]

                # Reduce to spatial dimensions only by taking first element of other dimensions
                for dim in test_data.dims:
                    if dim not in spatial_dims:
                        test_data = test_data.isel({dim: 0})

                # Find coordinates of valid (non-NaN) points
                valid_mask = ~test_data.isnull()

                if valid_mask.any():
                    # Get lat/lon coordinates of valid points
                    valid_lats = larger_region.lat.where(valid_mask)
                    valid_lons = larger_region.lon.where(valid_mask)

                    # Calculate distances to all valid points using geodesic
                    distances = []
                    valid_coords = []

                    # Iterate over all spatial dimensions
                    spatial_indices = {}
                    for dim in spatial_dims:
                        spatial_indices[dim] = range(valid_lats.sizes[dim])

                    # Use the first two spatial dimensions for iteration
                    dim_names = list(spatial_indices.keys())
                    if len(dim_names) >= 2:
                        dim1, dim2 = dim_names[0], dim_names[1]
                        for i in spatial_indices[dim1]:
                            for j in spatial_indices[dim2]:
                                isel_dict = {dim1: i, dim2: j}
                                if valid_mask.isel(**isel_dict):
                                    point_lat = valid_lats.isel(**isel_dict).values
                                    point_lon = valid_lons.isel(**isel_dict).values

                                if not (np.isnan(point_lat) or np.isnan(point_lon)):
                                    dist = geodesic(
                                        (lat, lon), (point_lat, point_lon)
                                    ).kilometers

                                    distances.append(dist)
                                    valid_coords.append(
                                        (i, j, point_lat, point_lon, dim1, dim2)
                                    )

                    if valid_coords:
                        # Find closest valid point
                        min_idx = np.argmin(distances)
                        closest_i, closest_j, closest_lat, closest_lon, dim1, dim2 = (
                            valid_coords[min_idx]
                        )

                        print(
                            f"Found valid gridcell at distance {distances[min_idx]:.2f} km"
                        )
                        print(
                            f"Valid gridcell coordinates: lat={closest_lat:.4f}, lon={closest_lon:.4f}"
                        )

                        # Return the closest valid gridcell from the larger region
                        return larger_region.isel({dim1: closest_i, dim2: closest_j})

            except Exception as e:
                print(f"Error searching radius {radius}: {e}")
                continue

        print("No valid gridcells found within search radius")
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
        print(
            f"Processing {len(point_list)} points using efficient vectorized approach..."
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
                print("No valid gridcells found for any of the provided points")
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

            print(f"Successfully found closest gridcells for {len(point_list)} points")
            return closest_gridcells

        except Exception as e:
            print(f"Error in vectorized multi-point clipping: {e}")
            print("Falling back to individual point processing...")

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
            print(f"Processing point {i+1}/{len(point_list)}: ({lat:.4f}, {lon:.4f})")

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
                print(f"Warning: No valid data found for point ({lat:.4f}, {lon:.4f})")

        if not clipped_results:
            print("No valid gridcells found for any of the provided points")
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
                print(
                    f"Skipping duplicate grid cell at ({actual_lat:.4f}, {actual_lon:.4f}) "
                    f"for target point ({target_lat:.4f}, {target_lon:.4f})"
                )

        if not unique_results:
            print("No unique gridcells found after filtering")
            return None

        print(
            f"Filtered from {len(clipped_results)} to {len(unique_results)} unique gridcells"
        )

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

            print(
                f"Successfully concatenated {len(unique_results)} unique closest gridcells"
            )
            return concatenated

        except Exception as e:
            print(f"Error concatenating results: {e}")
            return None

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
