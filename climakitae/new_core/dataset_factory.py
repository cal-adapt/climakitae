"""
DatasetFactory Module

This module provides a factory class for creating climate data processing components
and complete datasets. It serves as a central point for constructing validators,
processors, and data access objects appropriate for different data types and
analytical approaches.

The factory pattern implemented here simplifies the instantiation of the correct
combination of components based on whether the data is gridded climate data or
station-based observations, and whether the analysis follows a time-based or
warming-level approach.

Classes:
    DatasetFactory: Factory for creating datasets and associated components.

Dependencies:
    - climakitae.core.constants
    - climakitae.core.data_interface
    - climakitae.new_core.data_access
    - climakitae.new_core.data_processor
    - climakitae.new_core.dataset
    - climakitae.new_core.param_validation
"""

from __future__ import annotations

from climakitae.core.data_interface import DataInterface
from climakitae.new_core.data_access import (
    DataAccessor,
    S3DataAccessor,
    StationDataAccessor,
)
from climakitae.new_core.data_processor import (
    DataProcessor,
    TimeClimateDataProcessor,
    TimeStationDataProcessor,
    WarmingLevelDataProcessor,
    WarmingLevelStationProcessor,
)
from climakitae.new_core.dataset import ClimateDataset, StationDataset
from climakitae.new_core.param_validation import (
    ParameterValidator,
    StationDataValidator,
    TimeClimateValidator,
    WarmingLevelClimateValidator,
)


class DatasetFactory:
    """
    Factory for creating datasets and associated components.
    Centralizes creation of datasets, validators, processors, and data access objects.
    """

    def __init__(self):
        """Initialize the factory with a data interface."""
        self._data_interface = DataInterface()

    def create_validator(self, data_type: str, approach: str) -> ParameterValidator:
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

    def create_processor(self, data_type: str, approach: str) -> DataProcessor:
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

    def create_data_access(self, data_type: str) -> DataAccessor:
        """Create appropriate data access based on data type."""

        match data_type:
            case "Gridded":
                return S3DataAccessor(self._data_interface.data_catalog)
            case "Stations":
                return StationDataAccessor(
                    self._data_interface.data_catalog, self._data_interface.stations_gdf
                )
            case _:
                raise ValueError(f"Unknown data type: {data_type}")

    def create_dataset(self, data_type, approach):
        """Create a complete dataset with associated components."""
        data_access = self.create_data_access(data_type)
        validator = self.create_validator(data_type, approach)
        processor = self.create_processor(data_type, approach)

        match (data_type, approach):
            case ("Gridded", "Time"):
                return ClimateDataset(data_access, validator, processor)
            case ("Gridded", "Warming Level"):
                return ClimateDataset(data_access, validator, processor)
            case ("Stations", _):
                return StationDataset(data_access, validator, processor)
            case (_, _):
                raise ValueError(
                    f"Unknown data type or approach: {data_type}, {approach}"
                )
