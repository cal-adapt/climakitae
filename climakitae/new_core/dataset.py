from abc import ABC, abstractmethod

from climakitae.core.constants import UNSET


class Dataset(ABC):
    """Base class for all dataset types providing a common interface."""

    def __init__(self, data_access, parameter_validator, data_processor):
        self.data_access = data_access
        self.parameter_validator = parameter_validator
        self.data_processor = data_processor

    def retrieve(self, parameters):
        """Retrieve data based on parameters."""
        # Validate parameters
        if self.parameter_validator:
            parameters = self.parameter_validator.validate(parameters)

        # Retrieve raw data
        raw_data = self.data_access.get_data(parameters)

        # Process the raw data
        if self.data_processor:
            return self.data_processor.process(raw_data, parameters)

        return raw_data


class ClimateDataset(Dataset):
    """Dataset specifically for climate data."""

    def __init__(self, data_access, parameter_validator, data_processor):
        super().__init__(data_access, parameter_validator, data_processor)

    def get_variable_info(self, variable_id):
        """Get information about a specific variable."""
        # Implementation specific to climate data
        pass


class StationDataset(Dataset):
    """Dataset specifically for station data."""

    def __init__(self, data_access, parameter_validator, data_processor):
        super().__init__(data_access, parameter_validator, data_processor)

    def get_nearby_stations(self, lat, lon, radius=50):
        """Find stations within a radius of a point."""
        # Implementation specific to station data
        pass
