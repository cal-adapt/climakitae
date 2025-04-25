from typing import List, Tuple, Union

from climakitae.core.constants import UNSET
from climakitae.core.data_access import S3DataAccess, StationDataAccess
from climakitae.core.data_interface import DataInterface
from climakitae.core.data_processor import (
    TimeClimateDataProcessor,
    TimeStationDataProcessor,
    WarmingLevelDataProcessor,
    WarmingLevelStationProcessor,
)
from climakitae.core.dataset import ClimateDataset, StationDataset
from climakitae.core.param_validation import (
    StationDataValidator,
    TimeClimateValidator,
    WarmingLevelClimateValidator,
)


class ClimateData:
    """
    An interface for accessing climate data.

    This class provides a fluent interface for setting parameters and retrieving
    climate data. It uses a factory pattern to create datasets and validators
    based on the specified parameters. The class is designed to be chainable,
    allowing users to set multiple parameters in a single line of code.

    Parameters
    ----------
    variable : str
        The climate variable to retrieve (e.g., "Air Temperature at 2m", "Precipitation").
    resolution : str
        The resolution of the data (e.g., "3 km", "9 km", "45 km").
    timescale : str
        The timescale of the data (e.g., "hourly", "daily", "monthly").
    downscaling_method : str
        The downscaling method to use (e.g., "Dynamical", "Statistical").
    data_type : str
        The type of data to retrieve (e.g., "Gridded", "Stations").
    approach : str
        The approach to use for data retrieval (e.g., "Time", "Warming Level").
    scenario : str or list of str
        The scenario(s) to retrieve data for (e.g., "SSP 2-4.5", "SSP 3-7.0").
    units : str
        The units for the variable (e.g., "Celsius", "Fahrenheit").
    warming_level : float or list of float
        The warming level(s) to retrieve data for (e.g., 0.8, 1.0, 1.5).
    area_subset : str
        The area subset to retrieve data for (e.g., "none", "region").
    latitude : tuple of float
        The latitude bounds for the data (e.g., (min_lat, max_lat)).
    longitude : tuple of float
        The longitude bounds for the data (e.g., (min_lon, max_lon)).
    cached_area : str or list of str
        The cached area to retrieve data for (e.g., "entire domain", "region").
    area_average : str
        Whether to average over the spatial domain (e.g., "yes", "no").
    time_slice : tuple of int
        The time range for the data (e.g., (start_year, end_year)).
    stations : str or list of str
        The station(s) to retrieve data for (e.g., "station1", "station2").
    warming_level_window : int
        The warming level window for the data (e.g., 10).
    warming_level_months : int or list of int
        The warming level months for the data (e.g., 1, 2, 3).

    Methods
    -------
    variable(value: str)
        Set the climate variable.
    resolution(value: str)
        Set the resolution (e.g., "3 km", "9 km", "45 km").
    timescale(value: str)
        Set the timescale (e.g., "hourly", "daily", "monthly").
    downscaling_method(value: str)
        Set the downscaling method.
    data_type(value: str)
        Set the data type (e.g., "Gridded", "Stations").
    approach(value: str)
        Set the approach (e.g., "Time", "Warming Level").
    scenario(value: Union[str, List[str]])
        Set the scenario(s).
    units(value: str)
        Set the units for the variable.
    warming_level(value: Union[float, List[float]])
        Set the warming level(s).
    area_subset(value: str)
        Set the area subset category.
    latitude(value: Tuple[float, float])
        Set the latitude bounds.
    longitude(value: Tuple[float, float])
        Set the longitude bounds.
    cached_area(value: Union[str, List[str]])
        Set the cached area.
    area_average(value: str)
        Set whether to average over spatial domain.
    time_slice(value: Tuple[int, int])
        Set the time range.
    stations(value: Union[str, List[str]])
        Set the station(s) to retrieve data for.
    warming_level_window(value: int)
        Set the warming level window.
    warming_level_months(value: Union[int, List[int]])
        Set the warming level months.
    get()
        Retrieve climate data based on the configured parameters.

    Returns
    -------
    xr.DataArray
        The retrieved lazy-loaded climate data.

    Raises
    ------
    ValueError
        If any required parameters are missing or invalid.
    Exception
        If there is an error during data retrieval or processing.


    Example
    -------
    >>> climate_data = ClimateData()
    >>> data = (
    ...     climate_data
    ...     .variable("Air Temperature at 2m")
    ...     .resolution("3 km")
    ...     .timescale("hourly")
    ...     .downscaling_method("Dynamical")
    ...     .data_type("Gridded")
    ...     .approach("Time")
    ...     .scenario("SSP 2-4.5")
    ...     .units("Celsius")
    ...     .warming_level(1.0)
    ...     .area_subset("region")
    ...     .latitude((30.0, 50.0))
    ...     .longitude((-120.0, -80.0))
    ...     .cached_area("entire domain")
    ...     .area_average("yes")
    ...     .time_slice((2020, 2050))
    ...     .stations(["station1", "station2"])
    ...     .warming_level_window(10)
    ...     .warming_level_months([1, 2, 3])
    ...     .get()
    ... )
    """

    def __init__(self):
        """
        Initialize the ClimateData facade.
        Data sources are managed internally by the DataSourceManager.
        """
        self._factory = DatasetFactory()
        self._data_interface = DataInterface()
        self._reset_query()

    def _reset_query(self):
        """Reset the query parameters to defaults."""
        self._query = {
            "variable": UNSET,
            "resolution": UNSET,
            "timescale": UNSET,
            "downscaling_method": "Dynamical",
            "data_type": "Gridded",
            "approach": "Time",
            "scenario": UNSET,
            "units": UNSET,
            "warming_level": UNSET,
            "area_subset": UNSET,
            "latitude": UNSET,
            "longitude": UNSET,
            "cached_area": ["entire domain"],
            "area_average": UNSET,
            "time_slice": UNSET,
            "stations": UNSET,
            "warming_level_window": UNSET,
            "warming_level_months": UNSET,
        }
        return self

    # Parameter setter methods (chainable)
    def variable(self, value: str):
        """Set the climate variable."""
        self._query["variable"] = value
        return self

    def resolution(self, value: str):
        """Set the resolution (e.g., "3 km", "9 km", "45 km")."""
        self._query["resolution"] = value
        return self

    def timescale(self, value: str):
        """Set the timescale (e.g., "hourly", "daily", "monthly")."""
        self._query["timescale"] = value
        return self

    def downscaling_method(self, value: str = "Dynamical"):
        """Set the downscaling method."""
        self._query["downscaling_method"] = value
        return self

    def data_type(self, value: str = "Gridded"):
        """Set the data type (e.g., "Gridded", "Stations")."""
        self._query["data_type"] = value
        return self

    def approach(self, value: str = "Time"):
        """Set the approach (e.g., "Time", "Warming Level")."""
        self._query["approach"] = value
        return self

    def scenario(self, value: Union[str, List[str]]):
        """Set the scenario(s)."""
        self._query["scenario"] = value if isinstance(value, list) else [value]
        return self

    def units(self, value: str):
        """Set the units for the variable."""
        self._query["units"] = value
        return self

    def warming_level(self, value: Union[float, List[float]]):
        """Set the warming level(s)."""
        self._query["warming_level"] = value if isinstance(value, list) else [value]
        return self

    def area_subset(self, value: str = "none"):
        """Set the area subset category."""
        self._query["area_subset"] = value
        return self

    def latitude(self, value: Tuple[float, float]):
        """Set the latitude bounds."""
        self._query["latitude"] = value
        return self

    def longitude(self, value: Tuple[float, float]):
        """Set the longitude bounds."""
        self._query["longitude"] = value
        return self

    def cached_area(self, value: Union[str, List[str]] = ["entire domain"]):
        """Set the cached area."""
        self._query["cached_area"] = value if isinstance(value, list) else [value]
        return self

    def area_average(self, value: str):
        """Set whether to average over spatial domain."""
        self._query["area_average"] = value
        return self

    def time_slice(self, value: Tuple[int, int]):
        """Set the time range."""
        self._query["time_slice"] = value
        return self

    def stations(self, value: Union[str, List[str]]):
        """Set the station(s) to retrieve data for."""
        self._query["stations"] = value if isinstance(value, list) else [value]
        return self

    def warming_level_window(self, value: int):
        """Set the warming level window."""
        self._query["warming_level_window"] = value
        return self

    def warming_level_months(self, value: Union[int, List[int]]):
        """Set the warming level months."""
        self._query["warming_level_months"] = (
            value if isinstance(value, list) else [value]
        )
        return self

    def get(self):
        """
        Retrieve climate data based on the configured parameters.

        Returns
        -------
        xr.DataArray
            The retrieved climate data
        """
        # Check required parameters
        if not self._validate_required_parameters():
            self._reset_query()
            return None

        try:
            # Create appropriate validator
            validator = self._factory.create_validator(
                self._query["data_type"], self._query["approach"]
            )

            # Validate parameters
            validated_params = validator.validate(self._query)

            # Create dataset
            dataset = self._factory.create_dataset(
                validated_params["data_type"], validated_params["approach"]
            )

            # Retrieve data
            data = dataset.retrieve(validated_params)
            self._reset_query()
            return data

        except Exception as e:
            print(f"Error: {str(e)}")
            self._reset_query()
            return None

    def _validate_required_parameters(self) -> bool:
        """Check if all required parameters are set."""
        required_params = ["variable", "resolution", "timescale"]
        for param in required_params:
            if self._query[param] is None:
                print(f"ERROR: {param} is a required parameter")
                return False
        return True


class DatasetFactory:
    """
    Factory for creating datasets and associated components.
    Centralizes creation of datasets, validators, processors, and data access objects.
    """

    def __init__(self):
        """Initialize the factory with a data interface."""
        self._data_interface = DataInterface()

    def create_validator(self, data_type, approach):
        """Create appropriate validator based on data type and approach."""

        match (data_type, approach):
            case ("Gridded", "Time"):
                return TimeClimateValidator()
            case ("Gridded", "Warming Level"):
                return WarmingLevelClimateValidator()
            case ("Gridded", _):
                raise ValueError(f"Unknown approach for Gridded data: {approach}")
            case ("Stations", _):  # Station validator doesn't depend on approach
                return StationDataValidator(self._data_interface.stations_gdf)
            case _:
                raise ValueError(f"Unknown data type: {data_type}")

    @staticmethod
    def create_processor(data_type, approach):
        """Create appropriate processor based on data type and approach."""

        match (data_type, approach):
            case ("Gridded", "Time"):
                return TimeClimateDataProcessor()
            case ("Gridded", "Warming Level"):
                return WarmingLevelDataProcessor()
            case ("Stations", "Time"):
                return TimeStationDataProcessor()
            case ("Stations", "Warming Level"):
                return WarmingLevelStationProcessor()
            case _:
                raise ValueError(
                    f"Unknown data type or approach: {data_type}, {approach}"
                )

    def create_data_access(self, data_type):
        """Create appropriate data access based on data type."""

        match data_type:
            case "Gridded":
                return S3DataAccess(self._data_interface.data_catalog)
            case "Stations":
                return StationDataAccess(
                    self._data_interface.data_catalog, self._data_interface.stations_gdf
                )
            case _:
                raise ValueError(f"Unknown data type: {data_type}")

    @staticmethod
    def create_dataset(data_type, approach):
        """Create a complete dataset with associated components."""
        data_access = DatasetFactory.create_data_access(data_type)
        validator = DatasetFactory.create_validator(data_type, approach)
        processor = DatasetFactory.create_processor(data_type, approach)

        if data_type == "Gridded":
            return ClimateDataset(data_access, validator, processor)
        elif data_type == "Stations":
            return StationDataset(data_access, validator, processor)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
