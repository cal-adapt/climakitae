from abc import ABC, abstractmethod


class DataAccess(ABC):
    """Abstract base class for data access."""

    @abstractmethod
    def get_data(self, parameters):
        """Get data from the source."""
        pass


class S3DataAccess(DataAccess):
    """Handles access to data stored in S3."""

    def __init__(self, data_catalog):
        """Initialize with data catalog."""
        self.data_catalog = data_catalog

    def get_data(self, parameters):
        """Get data from S3 based on parameters."""
        # Get catalog subset based on parameters
        cat_subset = self._get_catalog_subset(parameters)

        # Read data from S3
        data_dict = cat_subset.to_dataset_dict(
            zarr_kwargs={"consolidated": True},
            storage_options={"anon": True},
            progressbar=False,
        )

        return data_dict

    def _get_catalog_subset(self, parameters):
        """Get subset of catalog based on parameters."""
        # Implementation of catalog subsetting logic
        pass


class StationDataAccess(DataAccess):
    """Handles access to station data."""

    def __init__(self, data_catalog, stations_gdf):
        """Initialize with data catalog and stations geodataframe."""
        self.data_catalog = data_catalog
        self.stations_gdf = stations_gdf

    def get_data(self, parameters):
        """Get station data based on parameters."""
        # Get catalog subset for stations
        cat_subset = self._get_station_catalog_subset(parameters)

        # Filter by requested stations
        if parameters.get("stations"):
            # Filter catalog by stations
            pass

        # Read data from S3
        data_dict = cat_subset.to_dataset_dict(
            zarr_kwargs={"consolidated": True},
            storage_options={"anon": True},
            progressbar=False,
        )

        return data_dict

    def _get_station_catalog_subset(self, parameters):
        """Get subset of station catalog based on parameters."""
        # Implementation of station catalog subsetting logic
        pass
