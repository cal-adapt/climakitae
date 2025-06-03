"""
Dafrom typing import Any, Dict, Iterable, Union

import geopandas as gpd
import pyproj
import xarray as xr
from shapely.geometry import box, mapping

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)or clipping data based on spatial boundaries.
"""

import os
import warnings
from typing import Any, Dict, Iterable, Union

import geopandas as gpd
import pandas as pd
import pyproj
import xarray as xr
from shapely.geometry import box, mapping

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)


@register_processor("clip", priority=50)
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
        - list: Multiple boundary keys of the same category (Phase 1)
        - tuple: Coordinate bounds ((lat_min, lat_max), (lon_min, lon_max))

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
    """

    def __init__(self, value):
        """
        Initialize the Clip processor.

        Parameters
        ----------
        value : str | list | tuple
            The value(s) to clip the data by. Can be:
            - str: Single boundary key, file path, or coordinate specification
            - list: Multiple boundary keys of the same category (Phase 1)
            - tuple: Coordinate bounds ((lat_min, lat_max), (lon_min, lon_max))
        """
        self.value = value
        self.name = "clip"
        self.catalog: Union[DataCatalog, object] = UNSET
        self.needs_catalog = True

        # Determine if this is a multi-boundary operation
        self.multi_mode = isinstance(value, list) and len(value) > 1
        self.operation = "union" if self.multi_mode else None

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
                # check if string is path-like
                if os.path.exists(self.value):
                    geom = gpd.read_file(self.value)
                else:
                    # try to find corresponding boundary key
                    geom = self._get_boundary_geometry(self.value)
            case list():
                # Handle multiple boundary keys (Phase 1)
                geom = self._get_multi_boundary_geometry(self.value)
            case tuple():
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

        ret = None
        match result:
            case dict():
                # Clip each dataset in the dictionary
                ret = {
                    key: self._clip_data_with_geom(value, geom)
                    for key, value in result.items()
                }
            case xr.Dataset() | xr.DataArray():
                # Clip the single dataset
                ret = self._clip_data_with_geom(result, geom)
            case list() | tuple():
                # return as passed type
                ret = type(result)(
                    [self._clip_data_with_geom(data, geom) for data in result]
                )

            case _:
                raise ValueError(
                    f"Invalid result type for clipping. Expected dict, xr.Dataset, xr.DataArray, or Iterable but got {type(result)}."
                )
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
        if self.multi_mode:
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
        # check crs of geom and data
        if gdf.crs is None and data.rio.crs is not None:
            warnings.warn(
                "The GeoDataFrame does not have a CRS set. Setting the CRS to the data's CRS."
            )
            gdf.set_crs(data.rio.crs, inplace=True)

        return data.rio.clip(
            gdf.geometry.apply(mapping),
            gdf.crs,
            drop=True,
            all_touched=True,
        )

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
                for key in lookups.keys():
                    if (
                        boundary_key.lower() in key.lower()
                        or key.lower() in boundary_key.lower()
                    ):
                        suggestions.append(f"{category}: {key}")

        return {
            "valid": False,
            "error": f"Boundary key '{boundary_key}' not found",
            "suggestions": suggestions[:10],  # Limit to top 10 suggestions
        }

    def list_available_boundaries(self) -> Dict[str, list]:
        """
        List all available boundary options that can be used for clipping.

        Returns
        -------
        Dict[str, list]
            Dictionary with boundary categories as keys and lists of available
            boundary names as values

        Raises
        ------
        RuntimeError
            If the catalog is not set or boundaries are not available

        Examples
        --------
        >>> clip_processor = Clip("some_value")
        >>> clip_processor.catalog = DataCatalog()
        >>> boundaries = clip_processor.list_available_boundaries()
        >>> print(boundaries["states"])
        ['CA', 'OR', 'WA', 'NV', 'AZ', 'UT', 'ID', 'MT', 'WY', 'CO', 'NM']
        """
        if self.catalog is UNSET or not isinstance(self.catalog, DataCatalog):
            raise RuntimeError(
                "DataCatalog is not set. Cannot access boundary data. "
                "Please ensure the catalog is properly initialized."
            )

        try:
            boundary_dict = self.catalog.boundaries.boundary_dict()  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to access boundary data: {e}") from e

        # Create a clean dictionary with boundary categories and their available options
        available_boundaries = {}

        for category, lookups in boundary_dict.items():
            # Skip special categories that don't represent actual boundary data
            if category in ["none", "lat/lon"]:
                continue

            # Convert keys to a sorted list for better presentation
            boundary_keys = sorted(list(lookups.keys()))
            available_boundaries[category] = boundary_keys

        return available_boundaries

    def print_available_boundaries(self) -> None:
        """
        Print all available boundary options in a user-friendly format.

        This method provides a nicely formatted output showing all boundary
        categories and their available options for clipping operations.

        Raises
        ------
        RuntimeError
            If the catalog is not set or boundaries are not available

        Examples
        --------
        >>> clip_processor = Clip("some_value")
        >>> clip_processor.catalog = DataCatalog()
        >>> clip_processor.print_available_boundaries()
        Available Boundary Options for Clipping:
        ========================================

        states:
          - CA, OR, WA, NV, AZ, UT, ID, MT, WY, CO, NM

        CA counties:
          - Alameda County, Alpine County, Amador County, ...
        """
        try:
            boundaries = self.list_available_boundaries()
        except RuntimeError as e:
            print(f"Error: {e}")
            return

        print("Available Boundary Options for Clipping:")
        print("=" * 40)
        print()

        for category, boundary_list in boundaries.items():
            print(f"{category}:")

            # Format the list nicely - wrap long lists
            if len(boundary_list) <= 5:
                # For short lists, show all on one line
                print(f"  - {', '.join(boundary_list)}")
            else:
                # For longer lists, show first few and count
                displayed = boundary_list[:5]
                remaining = len(boundary_list) - 5
                print(f"  - {', '.join(displayed)}")
                if remaining > 0:
                    print(f"    ... and {remaining} more options")
            print()

    @classmethod
    def get_boundary_examples(cls) -> Dict[str, str]:
        """
        Get example boundary keys for each category.

        Returns
        -------
        Dict[str, str]
            Dictionary with boundary categories as keys and example boundary
            names as values

        Examples
        --------
        >>> examples = Clip.get_boundary_examples()
        >>> print(examples["states"])
        'CA'
        >>> print(examples["CA counties"])
        'Los Angeles County'
        """
        return {
            "states": "CA",
            "CA counties": "Los Angeles County",
            "CA watersheds": "Russian River",
            "CA Electric Load Serving Entities (IOU & POU)": "PG&E",
            "CA Electricity Demand Forecast Zones": "PG&E Bay Area",
            "CA Electric Balancing Authority Areas": "CALISO",
        }

    @staticmethod
    def get_supported_boundary_categories() -> Dict[str, str]:
        """
        Get information about supported boundary categories.

        Returns
        -------
        Dict[str, str]
            Dictionary with boundary categories as keys and descriptions as values

        Examples
        --------
        >>> categories = Clip.get_supported_boundary_categories()
        >>> for category, description in categories.items():
        ...     print(f"{category}: {description}")
        states: Western US states (CA, OR, WA, etc.)
        CA counties: California counties
        """
        return {
            "states": "Western US states (CA, OR, WA, NV, AZ, UT, ID, MT, WY, CO, NM)",
            "CA counties": "California counties (e.g., Los Angeles County, San Francisco County)",
            "CA watersheds": "California HUC-8 watersheds (e.g., Russian River, Sacramento River)",
            "CA Electric Load Serving Entities (IOU & POU)": "California electric utilities (e.g., PG&E, SCE, SDG&E)",
            "CA Electricity Demand Forecast Zones": "California electricity demand forecast zones",
            "CA Electric Balancing Authority Areas": "California electric balancing authority areas (e.g., CALISO)",
        }

    @staticmethod
    def print_boundary_usage_examples() -> None:
        """
        Print usage examples for boundary clipping.

        This provides users with concrete examples of how to use different
        boundary types for clipping operations.
        """
        print("Boundary Clipping Usage Examples:")
        print("=" * 35)
        print()

        examples = [
            ("File path", 'Clip("/path/to/shapefile.shp")'),
            ("State", 'Clip("CA")'),
            ("County", 'Clip("Los Angeles County")'),
            ("Watershed", 'Clip("Russian River")'),
            ("Utility", 'Clip("PG&E")'),
            ("Forecast Zone", 'Clip("PG&E Bay Area")'),
            ("Balancing Area", 'Clip("CALISO")'),
            ("Lat/Lon box", "Clip(((lat_min, lat_max), (lon_min, lon_max)))"),
        ]

        for description, example in examples:
            print(f"{description}:")
            print(f"  {example}")
            print()

    def _get_multi_boundary_geometry(self, boundary_keys: list) -> gpd.GeoDataFrame:
        """
        Get combined geometry for multiple boundary keys (Phase 1 implementation).

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

        # Concatenate all geometries
        try:
            combined_df = gpd.GeoDataFrame(pd.concat(geometries, ignore_index=True))
        except Exception as e:
            raise ValueError(f"Failed to concatenate geometries: {e}")

        # Ensure consistent CRS - use the CRS from the first geometry
        reference_crs = geometries[0].crs
        if reference_crs is not None:
            combined_df = combined_df.to_crs(reference_crs)

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
