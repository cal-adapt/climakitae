"""
Climate Data Interface Module for Accessing Climate Data.

This module provides a high-level interface for accessing climate data through
the ClimateData class. It implements a fluent interface pattern that allows users
to chain method calls to configure data queries.

The module facilitates retrieving climate data with various parameters such as
variables, resolutions, timescales, and spatial boundaries. It implements a factory
pattern for creating appropriate datasets and validators based on specified parameters.
"""

import traceback
from typing import Iterable

from climakitae.core.constants import UNSET
from climakitae.new_core.dataset_factory import DatasetFactory


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
    """

    def __init__(self):
        """
        Initialize the ClimateData facade.
        Data sources are managed internally by the DataSourceManager.
        """
        print("Initializing ClimateData...")
        self._factory = DatasetFactory()
        self._reset_query()

    def _reset_query(self):
        """Reset the query parameters to defaults."""
        self._query = {
            "catalog": UNSET,  # catalog name, e.g. "renewables"
            "installation": UNSET,  # renewables only
            "activity_id": UNSET,  # downscaling method
            "institution_id": UNSET,  # renewables only
            "source_id": UNSET,  # renewables only
            "experiment_id": UNSET,  # renewables only
            "table_id": UNSET,  # timescale, e.g., "hourly", "daily", "monthly"
            "grid_label": UNSET,  # resolution, e.g., "3 km", "9 km", "45 km"
            "variable_id": UNSET,  # variable name, e.g., "Air Temperature at 2m"
            "processes": UNSET,  # dictionary of processes to apply
        }
        return self

    def catalog(self, catalog: str) -> "ClimateData":
        """
        Set the catalog for the data source.

        Parameters
        ----------
        catalog : str
            The name of the catalog to use.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["catalog"] = catalog
        return self

    def installation(self, installation: str) -> "ClimateData":
        """
        Set the installation for the data source.

        Parameters
        ----------
        installation : str
            The name of the installation to use.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["installation"] = installation
        return self

    def activity_id(self, activity_id: str) -> "ClimateData":
        """
        Set the activity ID for the data source.

        Parameters
        ----------
        activity_id : str
            The activity ID to use.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["activity_id"] = activity_id
        return self

    def institution_id(self, institution_id: str) -> "ClimateData":
        """
        Set the institution ID for the data source.

        Parameters
        ----------
        institution_id : str
            The institution ID to use.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["institution_id"] = institution_id
        return self

    def source_id(self, source_id: str) -> "ClimateData":
        """
        Set the source ID for the data source.

        Parameters
        ----------
        source_id : str
            The source ID to use.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["source_id"] = source_id
        return self

    def experiment_id(self, experiment_id: str) -> "ClimateData":
        """
        Set the experiment ID for the data source.

        Parameters
        ----------
        experiment_id : str
            The experiment ID to use.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["experiment_id"] = experiment_id
        return self

    def table_id(self, table_id: str) -> "ClimateData":
        """
        Set the table ID for the data source.

        Parameters
        ----------
        table_id : str
            The table ID to use.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["table_id"] = table_id
        return self

    def grid_label(self, grid_label: str) -> "ClimateData":
        """
        Set the grid label for the data source.

        Parameters
        ----------
        grid_label : str
            The grid label to use.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["grid_label"] = grid_label
        return self

    def variable(self, variable: str) -> "ClimateData":
        """
        Set the variable for the data source.

        Parameters
        ----------
        variable : str
            The variable to retrieve.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["variable_id"] = variable
        return self

    def processes(self, processes: dict[str, str | Iterable]) -> "ClimateData":
        """
        Set the processes to apply to the data.

        Parameters
        ----------
        processes : dict
            A dictionary of processes to apply.

        Returns
        -------
        ClimateData
            The current instance of ClimateData allowing method chaining.
        """
        self._query["processes"] = processes
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
        data = None
        if not self._validate_required_parameters():
            self._reset_query()
            return data

        try:
            # Create dataset directly from the query
            dataset = self._factory.create_dataset(self._query)
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error during dataset creation:\n{str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return None

        try:
            # Execute dataset with query parameters
            data = dataset.execute(self._query)
        except (ValueError, KeyError, IOError, RuntimeError) as e:
            print(f"Error: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")

        # Reset the query after retrieval
        self._reset_query()
        return data

    def _validate_required_parameters(self) -> bool:
        """Check if all required parameters are set."""
        required_params = ["variable_id", "grid_label", "table_id", "catalog"]
        for param in required_params:
            if self._query[param] is None:
                print(f"ERROR: {param} is a required parameter")
                return False
        return True

    def show_query(self):
        """Print the current query parameters."""
        print("Current query parameters:")
        for key, value in self._query.items():
            print(f"{key}: {value if value is not UNSET else 'UNSET'}")

    def show_catalog_options(self):
        """Print the available catalogs."""
        print("Available catalog keys:")
        for x in self._factory.get_catalog_options("catalog"):
            print(f"{x}")

    def show_installation_options(self):
        """Print the available installations."""
        print("Available installation keys:")
        for x in self._factory.get_catalog_options("installation"):
            print(f"{x}")

    def show_activity_id_options(self):
        """Print the available activity IDs."""
        print("Available activity IDs:")
        for x in self._factory.get_catalog_options("activity_id"):
            print(f"{x}")

    def show_institution_id_options(self):
        """Print the available institution IDs."""
        print("Available institution IDs:")
        for x in self._factory.get_catalog_options("institution_id"):
            print(f"{x}")

    def show_source_id_options(self):
        """Print the available source IDs."""
        print("Available source IDs:")
        for x in self._factory.get_catalog_options("source_id"):
            print(f"{x}")

    def show_experiment_id_options(self):
        """Print the available experiment IDs."""
        print("Available experiment IDs:")
        for x in self._factory.get_catalog_options("experiment_id"):
            print(f"{x}")

    def show_table_id_options(self):
        """Print the available table IDs."""
        print("Available table IDs:")
        for x in self._factory.get_catalog_options("table_id"):
            print(f"{x}")

    def show_grid_label_options(self):
        """Print the available grid labels."""
        print("Available grid labels:")
        for x in self._factory.get_catalog_options("grid_label"):
            print(f"{x}")

    def show_variable_options(self):
        """Print the available variables."""
        print("WARNING: not all variables are available in all datasets")
        print("Available variables:")

        for x in self._factory.get_catalog_options("variable_id"):
            print(f"{x}")

    def show_validators(self):
        """Print the available validators."""
        print("Available validators:")
        for key in self._factory.get_validators():
            print(f"{key}")

    def show_processors(self):
        """Print the available processors."""
        print("Available processors:")
        for key in self._factory.get_processors():
            print(f"{key}")

    def show_all_options(self):
        """Print all available options."""
        # loop over methods starting with "show_" and call them
        for m in self.__class__.__dict__:
            # don't call this method
            if m.startswith("show_") and m != "show_all_options":
                getattr(self, m)()
                print("=" * 40)
