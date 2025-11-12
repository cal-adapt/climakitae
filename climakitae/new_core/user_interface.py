"""Climate Data Interface Module for Accessing Climate Data.

This module provides a high-level interface for accessing climate data through
the ClimateData class. It implements a fluent interface pattern that allows users
to chain method calls to configure data queries.

The module facilitates retrieving climate data with various parameters such as
catalogs, installations, activities, institutions, sources, experiments, variables,
and processing options. It implements a factory pattern for creating appropriate
datasets and validators based on specified parameters.

Example Usage:
    >>> data = ClimateData()
    >>> result = (data.catalog("renewables")
    ...               .installation("pv_utility")
    ...               .activity_id("CMIP6")
    ...               .variable("tasmax")
    ...               .table_id("day")
    ...               .grid_label("d03")
    ...               .get())

"""

import traceback
from typing import Any, Dict, Iterable, Optional, Union

from climakitae.core.constants import UNSET
from climakitae.core.data_export import export
from climakitae.core.paths import VARIABLE_DESCRIPTIONS_CSV_PATH
from climakitae.new_core.dataset_factory import DatasetFactory
from climakitae.util.utils import read_csv_file


class ClimateData:
    """A fluent interface for accessing climate data.

    This class provides a chainable interface for setting parameters and retrieving
    climate data. It uses a factory pattern to create datasets and validators
    based on the specified parameters. The class is designed to be chainable,
    allowing users to set multiple parameters in a single expression.

    The interface supports various climate data sources and allows for flexible
    querying with different combinations of parameters. All methods return the
    instance itself to enable method chaining.

    Parameters supported in queries:
    - catalog: The data catalog to use (e.g., "renewable energy generation", "cadcat")
    - installation: The installation type (e.g., "pv_utility", "wind_offshore")
    - activity_id: The activity identifier (e.g., "WRF", "LOCA2")
    - institution_id: The institution identifier (e.g., "CNRM", "DWD")
    - source_id: The source identifier (e.g., "GCM", "RCM", "Station")
    - experiment_id: The experiment identifier (e.g., "historical", "ssp245")
    - table_id: The temporal resolution (e.g., "1hr", "day", "mon")
    - grid_label: The spatial resolution (e.g., "d01", "d02", "d03")
    - variable_id: The climate variable (e.g., "tasmax", "pr", "cf")
    - processes: Dictionary of data processing operations to apply

    Methods
    -------
    catalog(catalog: str) -> ClimateData
        Set the data catalog to use.
    installation(installation: str) -> ClimateData
        Set the installation type.
    activity_id(activity_id: str) -> ClimateData
        Set the activity identifier.
    institution_id(institution_id: str) -> ClimateData
        Set the institution identifier.
    source_id(source_id: str) -> ClimateData
        Set the source identifier.
    experiment_id(experiment_id: str | list[str]) -> ClimateData
        Set the experiment identifier(s).
    table_id(table_id: str) -> ClimateData
        Set the temporal resolution.
    grid_label(grid_label: str) -> ClimateData
        Set the spatial resolution.
    variable(variable: str) -> ClimateData
        Set the climate variable to retrieve.
    processes(processes: Dict[str, Union[str, Iterable]]) -> ClimateData
        Set processing operations to apply to the data.
    get() -> Optional[xr.DataArray]
        Execute the query and retrieve the climate data.

    Utility methods for exploring available options:
    show_*_options() methods display available values for each parameter.
    show_query() displays the current query configuration.
    show_all_options() displays all available options for exploration.

    Returns
    -------
    xr.DataArray or None
        The retrieved climate data as a lazy-loaded xarray DataArray,
        or None if the query fails or required parameters are missing.

    Raises
    ------
    ValueError
        If required parameters are missing or invalid during validation.
    Exception
        If there is an error during data retrieval or processing.

    Examples
    --------
    Basic usage with method chaining:

    >>> cd = ClimateData()
    >>> data = (cd
    ...     .catalog("cadcat")
    ...     .activity_id("WRF")
    ...     .experiment_id("historical")
    ...     .table_id("1hr")
    ...     .grid_label("d02")
    ...     .variable("prec")
    ...     .get()
    ...    )

    Exploring available options:

    >>> cd = ClimateData()
    >>> cd.show_catalog_options()
    >>> cd.catalog("cadcat").show_variable_options()

    Using with processing:

    >>> processes = {"spatial_avg": "region", "temporal_avg": "monthly"}
    >>> data = (ClimateData()
    ...         .catalog("climate")
    ...         .variable("pr")
    ...         .processes(processes)
    ...         .get())

    """

    def __init__(self):
        """Initialize the ClimateData interface.

        Sets up the factory for dataset creation and initializes
        query parameters to their default (UNSET) state.

        """
        try:
            self._factory = DatasetFactory()
            self._reset_query()
            self.var_desc = read_csv_file(VARIABLE_DESCRIPTIONS_CSV_PATH)
            print("✅ Ready to query! ")
        except Exception as e:
            print(f"❌ Setup failed: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")
            return

    def _reset_query(self) -> "ClimateData":
        """Reset all query parameters to their default UNSET state.

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        self._query = {
            "catalog": UNSET,
            "installation": UNSET,
            "activity_id": UNSET,
            "institution_id": UNSET,
            "source_id": UNSET,
            "experiment_id": UNSET,
            "table_id": UNSET,
            "grid_label": UNSET,
            "variable_id": UNSET,
            "processes": UNSET,
        }
        self._factory.reset()
        return self

    # Core parameter setting methods
    def catalog(self, catalog: str) -> "ClimateData":
        """Set the data catalog to use for the query.

        Parameters
        ----------
        catalog : str
            The name of the catalog (e.g., "renewables", "climate").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        if not isinstance(catalog, str) or not catalog.strip():
            raise ValueError("Catalog must be a non-empty string")
        self._query["catalog"] = catalog.strip()
        return self

    def installation(self, installation: str) -> "ClimateData":
        """Set the installation type for the query.

        Parameters
        ----------
        installation : str
            The installation type (e.g., "pv_utility", "wind_offshore").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        if not isinstance(installation, str) or not installation.strip():
            raise ValueError("Installation must be a non-empty string")
        self._query["installation"] = installation.strip()
        return self

    def activity_id(self, activity_id: str) -> "ClimateData":
        """Set the activity identifier for the query.

        Parameters
        ----------
        activity_id : str
            The activity ID (e.g., "CMIP6", "CORDEX").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        if not isinstance(activity_id, str) or not activity_id.strip():
            raise ValueError("Activity ID must be a non-empty string")
        self._query["activity_id"] = activity_id.strip()
        return self

    def institution_id(self, institution_id: str) -> "ClimateData":
        """Set the institution identifier for the query.

        Parameters
        ----------
        institution_id : str
            The institution ID (e.g., "CNRM", "DWD").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        if not isinstance(institution_id, str) or not institution_id.strip():
            raise ValueError("Institution ID must be a non-empty string")
        self._query["institution_id"] = institution_id.strip()
        return self

    def source_id(self, source_id: str) -> "ClimateData":
        """Set the source identifier for the query.

        Parameters
        ----------
        source_id : str
            The source ID (e.g., "GCM", "RCM", "Station").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Source ID must be a non-empty string")
        self._query["source_id"] = source_id.strip()
        return self

    def experiment_id(self, experiment_id: str | list[str]) -> "ClimateData":
        """Set the experiment identifier for the query.

        Parameters
        ----------
        experiment_id : str
            The experiment ID (e.g., "historical", "ssp245").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        exp = []
        if not isinstance(experiment_id, (str, list)):
            raise ValueError(
                "Experiment ID must be a non-empty string or list of strings"
            )
        if isinstance(experiment_id, str):
            if not experiment_id.strip():
                raise ValueError("Experiment ID must be a non-empty string")
            exp.append(experiment_id.strip())
        else:
            for exp_id in experiment_id:
                if not isinstance(exp_id, str) or not exp_id.strip():
                    raise ValueError("Each experiment ID must be a non-empty string")
                exp.append(exp_id.strip())
        self._query["experiment_id"] = exp
        return self

    def table_id(self, table_id: str) -> "ClimateData":
        """Set the temporal resolution identifier for the query.

        Parameters
        ----------
        table_id : str
            The temporal resolution (e.g., "1hr", "day", "mon").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        if not isinstance(table_id, str) or not table_id.strip():
            raise ValueError("Table ID must be a non-empty string")
        self._query["table_id"] = table_id.strip()
        return self

    def grid_label(self, grid_label: str) -> "ClimateData":
        """Set the spatial resolution identifier for the query.

        Parameters
        ----------
        grid_label : str
            The spatial resolution (e.g., "d01", "d02", "d03").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        if not isinstance(grid_label, str) or not grid_label.strip():
            raise ValueError("Grid label must be a non-empty string")
        self._query["grid_label"] = grid_label.strip()
        return self

    def variable(self, variable: str) -> "ClimateData":
        """Set the climate variable to retrieve.

        Parameters
        ----------
        variable : str
            The variable identifier (e.g., "tasmax", "pr", "cf").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        if not isinstance(variable, str) or not variable.strip():
            raise ValueError("Variable must be a non-empty string")
        self._query["variable_id"] = variable.strip()
        return self

    def processes(self, processes: Dict[str, Union[str, Iterable]]) -> "ClimateData":
        """Set processing operations to apply to the retrieved data.

        Parameters
        ----------
        processes : Dict[str, Union[str, Iterable]]
            A dictionary of processing operations and their parameters.

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        if not isinstance(processes, dict):
            raise ValueError("Processes must be a dictionary")
        self._query["processes"] = processes.copy()
        return self

    # Main execution method
    def get(self) -> Optional[Any]:
        """Execute the configured query and retrieve climate data.

        Validates required parameters, creates the appropriate dataset using
        the factory pattern, executes the query, and resets the query state
        for the next use.

        Returns
        -------
        Optional[xr.DataArray]
            The retrieved climate data as a lazy-loaded xarray DataArray,
            or None if the query fails or validation errors occur.

        Raises
        ------
        ValueError
            If required parameters are missing during validation.
        Exception
            If there are errors during dataset creation or execution.

        """
        data = None

        # Validate required parameters
        if not self._validate_required_parameters():
            self._reset_query()

        try:
            # Create dataset using factory
            dataset = self._factory.create_dataset(self._query)
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error during dataset creation: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            self._reset_query()
            return None

        try:
            # Execute the query
            data = dataset.execute(self._query)
            # check if empty dataset
            # Check if data is empty/null
            if (
                data is None
                or (hasattr(data, "nbytes") and data.nbytes == 0)
                or (isinstance(data, dict) and not data)
            ):
                print("⚠️ Warning: Retrieved dataset is empty.")
            else:
                print("✅ Data retrieval successful!")
        except (ValueError, KeyError, IOError, RuntimeError) as e:
            print(f"Error during data retrieval: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            print("❌ Data retrieval failed. Please check your query parameters.")

        # Always reset query after execution
        self._reset_query()
        return data

    def _validate_required_parameters(self) -> bool:
        """Validate that all required parameters are set.

        Returns
        -------
        bool
            True if all required parameters are present, False otherwise.

        """
        required_params = ["variable_id", "grid_label", "table_id", "catalog"]
        missing_params = []

        for param in required_params:
            if self._query[param] is UNSET:
                missing_params.append(param)

        if missing_params:
            print(f"ERROR: Missing required parameters: {', '.join(missing_params)}")
            print("Use the show_*_options() methods to see available values")
            return False

        return True

    # Query inspection methods
    def show_query(self) -> None:
        """Display the current query configuration."""
        msg = "Current Query:"
        print(msg)
        print("-" * len(msg))
        for key, value in self._query.items():
            display_value = value if value is not UNSET else "UNSET"
            print(f"{key}: {display_value}")

    # Option exploration methods
    def show_catalog_options(self) -> None:
        """Display available catalog options."""
        self._show_options("catalog", "catalog options (Cloud data collections)")

    def show_installation_options(self) -> None:
        """Display available installation options."""
        self._show_options(
            "installation", "installation options (Renewable energy generation types)"
        )

    def show_activity_id_options(self) -> None:
        """Display available activity ID options."""
        self._show_options("activity_id", "activity_id options (Downscaling methods)")

    def show_institution_id_options(self) -> None:
        """Display available institution ID options."""
        self._show_options("institution_id", "institution_id options (Data producers)")

    def show_source_id_options(self) -> None:
        """Display available source ID options."""
        self._show_options("source_id", "source_id options (Climate model simulations)")

    def show_experiment_id_options(self) -> None:
        """Display available experiment ID options."""
        self._show_options("experiment_id", "experiment_id options (Simulation runs)")

    def show_table_id_options(self) -> None:
        """Display available table ID options (Temporal resolutions)."""
        self._show_options("table_id", "table_id options (Temporal resolutions)")

    def show_grid_label_options(self) -> None:
        """Display available grid label options (Spatial resolutions)."""
        self._show_options("grid_label", "grid_label options (Spatial resolutions)")

    def show_variable_options(self) -> None:
        """Display available variable options."""
        current_query = {k: v for k, v in self._query.items() if v is not UNSET}
        msg = ""
        if current_query:
            msg = "Variables (constrained by current query):"
        else:
            msg = "Variables"

        self._show_options("variable_id", msg)

    def show_processors(self) -> None:
        """Display available data processors."""
        msg = "Processors (Methods for transforming raw catalog data):"
        print(msg)
        print("-" * len(msg))
        try:
            for processor in self._factory.get_processors():
                print(f"{processor}")
            print("\n")
        except Exception as e:
            print(f"Error retrieving processors: {e}")

    def show_station_options(self) -> None:
        """Display available station options for data retrieval."""
        msg = "Stations (Available weather stations for localization):"
        print(msg)
        print("-" * len(msg))
        try:
            stations = self._factory.get_stations()
            if not stations:
                print("No stations available with current parameters")
            else:
                for station in sorted(stations):
                    print(f"{station}")
                print("\n")
        except Exception as e:
            print(f"Error retrieving stations: {e}")

    def show_boundary_options(self, type=UNSET) -> None:
        """Display available boundaries for spatial queries."""
        if type is UNSET:
            msg = "Boundary Types (call again with option type='...' to see options for any type):"
        else:
            msg = f"Avaliable '{" ".join([x.capitalize() for x in type.split("_")])}' Boundaries:"
        print(msg)
        print("-" * len(msg))
        try:
            boundaries = self._factory.get_boundaries(type)
            if not boundaries:
                print("No boundaries available with current parameters")
            else:
                for boundary in sorted(boundaries):
                    print(f"{boundary}")
                print("\n")
        except Exception as e:
            print(f"Error retrieving boundaries: {e}")

    def show_all_options(self) -> None:
        """Display all available options for exploration."""
        data_title = "CAL ADAPT DATA -- ALL AVAILABLE OPTIONS USING CLIMAKITAE"
        print("=" * len(data_title))
        print(data_title)
        print("=" * len(data_title))

        option_methods = [
            ("show_catalog_options", "Catalogs"),
            ("show_activity_id_options", "Activity IDs"),
            ("show_institution_id_options", "Institution IDs"),
            ("show_source_id_options", "Source IDs"),
            ("show_experiment_id_options", "Experiment IDs"),
            ("show_table_id_options", "Table IDs (Temporal Resolution)"),
            ("show_grid_label_options", "Grid Labels (Spatial Resolution)"),
            ("show_variable_options", "Variables"),
            ("show_installation_options", "Installations"),
            ("show_processors", "Processors"),
            ("show_station_options", "Stations"),
        ]

        for method_name, section_title in option_methods:
            try:
                getattr(self, method_name)()
            except Exception as e:
                print(f"Error displaying {section_title.lower()}: {e}")

        print("\n" + "=" * 60)
        print("Current Query Status:")
        print("=" * 60)
        self.show_query()

    def _show_options(self, option_type: str, title: str) -> None:
        """Helper method to display options with consistent formatting.

        Parameters
        ----------
        option_type : str
            The type of option to display.
        title : str
            The title for the options display.

        """
        print(f"{title}:")
        print("-" * (len(title) + 1))
        try:
            current_query = {k: v for k, v in self._query.items() if v is not UNSET}
            options = self._factory.get_catalog_options(option_type, current_query)
            if not options:
                print("No options available with current parameters")
            else:
                max_len = max(len(option) for option in options)
                for option in sorted(options):
                    print(
                        f"{self._format_option(option, option_type, spacing=4 + max_len - len(option))}"
                    )
                print("\n")
        except Exception as e:
            print(f"Error retrieving options: {e}")

    # Convenience methods for common workflows
    def reset(self) -> "ClimateData":
        """Manually reset the query parameters.

        Returns
        -------
        ClimateData
            The current instance with reset parameters.

        """
        return self._reset_query()

    def copy_query(self) -> Dict[str, Any]:
        """Get a copy of the current query parameters.

        Returns
        -------
        Dict[str, Any]
            A copy of the current query parameters.

        """
        return {k: v for k, v in self._query.items() if v is not UNSET}

    def load_query(self, query_params: Dict[str, Any]) -> "ClimateData":
        """Load query parameters from a dictionary.

        Parameters
        ----------
        query_params : Dict[str, Any]
            Dictionary of query parameters to load.

        Returns
        -------
        ClimateData
            The current instance with loaded parameters.

        """
        for key, value in query_params.items():
            if key in self._query:
                self._query[key] = value
        return self

    def _format_option(self, option: str, option_type: str, spacing: int = 0) -> str:
        """Format an option string for display.

        Parameters
        ----------
        option : str
            The option string to format.
        option_type : str
            The type of option being formatted.

        Returns
        -------
        str
            The formatted option string.

        """
        match option_type:
            case "grid_label":
                conversion = {
                    "d01": "45 km",
                    "d02": " 9 km",
                    "d03": " 3 km",
                }
                return f"{option} ({conversion.get(option, 'Unknown')})"
            case "variable_id":
                # look up variable description
                # grab row dataframe
                spaces = " " * spacing
                row = self.var_desc[self.var_desc["variable_id"] == option]
                if not row.empty:
                    desc = row["display_name"].values[0]
                    return f"{option}:{spaces}{desc}"
                else:
                    return f"{option}:{spaces}No description available"
            case _:
                # Default case for other option types
                return option
