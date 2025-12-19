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

import copy
import logging
import sys
import traceback
from typing import Any, Dict, Iterable, Optional, Union

from climakitae.core.constants import UNSET
from climakitae.core.paths import VARIABLE_DESCRIPTIONS_CSV_PATH
from climakitae.new_core.dataset_factory import DatasetFactory
from climakitae.util.utils import read_csv_file

# Module logger
logger = logging.getLogger(__name__)


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
    - verbosity: The logging verbosity level (e.g., -2, -1, 0, 1)
    - log_file: Path to log file (if None, logs to stdout)
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
    verbosity(level: int) -> ClimateData
        Set the logging verbosity level.
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
    station_id(station_id: str) --> ClimateData
        Set the station identifier
    network_id(network_id: str) --> ClimateData
        Set the network identifier
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

    def __init__(self, log_file: Optional[str] = None, verbosity: int = 0):
        """Initialize the ClimateData interface.

        Sets up the factory for dataset creation and initializes
        query parameters to their default (UNSET) state. Optionally
        configures logging to file or stdout.

        Parameters
        ----------
        log_file : str, optional
            Path to log file. If None, logs to stdout. Default is None.
        verbosity : int, optional
            Logging verbosity level:
            - 0: No logging (default)
            - 1: WARNING level
            - 2: INFO level
            - 3: DEBUG level
            Default is 0.

        """
        # Configure logging
        self._log_file = log_file
        self._verbosity = verbosity
        self._configure_logging()

        try:
            logger.info("Initializing ClimateData interface")
            self._factory = DatasetFactory()
            self._reset_query()
            self.var_desc = read_csv_file(VARIABLE_DESCRIPTIONS_CSV_PATH)
            logger.info("ClimateData initialization successful")
            logger.info("✅ Ready to query!")
        except Exception as e:
            logger.error("Setup failed: %s", str(e), exc_info=True)
            logger.error("❌ Setup failed: %s", str(e))
            # Emit the traceback at ERROR level so user-facing callers and
            # test harnesses that bridge logging->warnings/print capture it
            # as part of the failure output.
            logger.error("Error details: %s", traceback.format_exc())
            return

    def _reset_query(self) -> "ClimateData":
        """Reset all query parameters to their default UNSET state.

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        Note
        ----
        This method only resets the query parameters on this ClimateData instance.
        It does not reset global state on the DataCatalog singleton, making it
        safe to call in multi-threaded scenarios.

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
            "station_id": UNSET,
            "network_id": UNSET,
            "processes": UNSET,
        }
        # Note: We intentionally do NOT call self._factory.reset() here
        # The factory's registry state should persist, and the DataCatalog
        # singleton should not have its state modified per-query for thread safety
        return self

    def _configure_logging(self) -> None:
        """Configure logging based on verbosity level and log file.

        Sets up the logger with appropriate handler (file or stdout)
        and log level based on verbosity setting.

        """
        # Map verbosity to logging levels:
        # <= -2 : effectively silent (no handlers; set to CRITICAL+1)
        # -1    : WARNING
        #  0    : INFO (default)
        # >0    : DEBUG (user must specify >0 to get debug)
        if not isinstance(self._verbosity, int):
            # Fallback to INFO for unexpected types
            log_level = logging.INFO
        elif self._verbosity <= -2:
            log_level = logging.CRITICAL + 1
        elif self._verbosity == -1:
            log_level = logging.WARNING
        elif self._verbosity == 0:
            log_level = logging.INFO
        else:
            # verbosity > 0
            log_level = logging.DEBUG

        # Configure a package-level logger so all child modules under
        # 'climakitae' inherit the same handler and level. This ensures
        # debug messages from processors (e.g. concatenate) are visible
        # when verbosity is set to DEBUG.
        pkg_logger = logging.getLogger("climakitae")
        pkg_logger.setLevel(log_level)

        # Remove existing handlers on the package logger to avoid dupes
        for h in list(pkg_logger.handlers):
            pkg_logger.removeHandler(h)

        # Also clear any handlers that might have been added directly to
        # this module's logger in earlier runs to keep behavior deterministic.
        for h in list(logger.handlers):
            logger.removeHandler(h)

        # Add handler only when logging is not intentionally silenced
        if log_level != logging.CRITICAL + 1:
            # Create handler (file or stdout)
            if self._log_file:
                handler = logging.FileHandler(self._log_file, mode="a")
            else:
                handler = logging.StreamHandler(sys.stdout)

            # Set formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            handler.setLevel(log_level)

            # Add handler to package logger so all climakitae.* loggers
            # will propagate to it and be printed according to the
            # configured verbosity.
            pkg_logger.addHandler(handler)
            pkg_logger.propagate = False  # Don't propagate to root logger

            # Ensure this module-level logger at least has the same level
            # so its own messages are emitted consistently.
            logger.setLevel(log_level)

            # Route Python warnings (logger.warning) into the logging
            # system so they obey the same handler/level configuration.
            logging.captureWarnings(True)
        else:
            # Intentionally silent: disable package logger handlers and
            # stop capturing warnings
            pkg_logger.setLevel(logging.CRITICAL + 1)
            logging.captureWarnings(False)

        # Suppress noisy third-party libraries
        # These libraries can be very verbose at DEBUG level, so we force them
        # to WARNING level to keep the output clean.
        noisy_libs = [
            "botocore",
            "boto3",
            "s3fs",
            "fsspec",
            "asyncio",
            "urllib3",
            "numcodecs",
            "zarr",
            "aiobotocore",
            "distributed",
            "dask",
        ]
        for lib in noisy_libs:
            logging.getLogger(lib).setLevel(logging.WARNING)

    def verbosity(self, level: int) -> "ClimateData":
        """Set the logging verbosity level.

        This method allows dynamic adjustment of logging verbosity
        and supports method chaining.

        Parameters
        ----------
        level : int
            Logging verbosity mapping:
            - <= -2: effectively silent (no logs)
            - -1: WARNING level
            - 0: INFO level (default)
            - >0: DEBUG level (user must specify >0 to get debug)

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        Examples
        --------
        >>> cd = ClimateData()
        >>> cd.verbosity(-1)  # warnings only
        >>> cd.verbosity(0)   # info (default)
        >>> cd.verbosity(1)   # debug

        """
        if not isinstance(level, int):
            raise ValueError("Verbosity level must be an integer")

        logger.debug("Setting verbosity level to %d", level)
        self._verbosity = level
        self._configure_logging()
        logger.info("Verbosity level set to %d", level)
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
        logger.debug("Setting catalog to: %s", catalog)
        if not isinstance(catalog, str) or not catalog.strip():
            logger.error("Invalid catalog parameter: must be non-empty string")
            raise ValueError("Catalog must be a non-empty string")
        self._query["catalog"] = catalog.strip()
        logger.info("Catalog set to: %s", catalog.strip())
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
        logger.debug("Setting installation to: %s", installation)
        if not isinstance(installation, str) or not installation.strip():
            logger.error("Invalid installation parameter: must be non-empty string")
            raise ValueError("Installation must be a non-empty string")
        self._query["installation"] = installation.strip()
        logger.info("Installation set to: %s", installation.strip())
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
        logger.debug("Setting activity_id to: %s", activity_id)
        if not isinstance(activity_id, str) or not activity_id.strip():
            logger.error("Invalid activity_id parameter: must be non-empty string")
            raise ValueError("Activity ID must be a non-empty string")
        self._query["activity_id"] = activity_id.strip()
        logger.info("Activity ID set to: %s", activity_id.strip())
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
        logger.debug("Setting institution_id to: %s", institution_id)
        if not isinstance(institution_id, str) or not institution_id.strip():
            logger.error("Invalid institution_id parameter: must be non-empty string")
            raise ValueError("Institution ID must be a non-empty string")
        self._query["institution_id"] = institution_id.strip()
        logger.info("Institution ID set to: %s", institution_id.strip())
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
        logger.debug("Setting source_id to: %s", source_id)
        if not isinstance(source_id, str) or not source_id.strip():
            logger.error("Invalid source_id parameter: must be non-empty string")
            raise ValueError("Source ID must be a non-empty string")
        self._query["source_id"] = source_id.strip()
        logger.info("Source ID set to: %s", source_id.strip())
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
        logger.debug("Setting experiment_id to: %s", experiment_id)
        exp = []
        if not isinstance(experiment_id, (str, list)):
            logger.error(
                "Invalid experiment_id parameter: must be string or list of strings"
            )
            raise ValueError(
                "Experiment ID must be a non-empty string or list of strings"
            )
        if isinstance(experiment_id, str):
            if not experiment_id.strip():
                logger.error("Invalid experiment_id parameter: empty string")
                raise ValueError("Experiment ID must be a non-empty string")
            exp.append(experiment_id.strip())
        else:
            for exp_id in experiment_id:
                if not isinstance(exp_id, str) or not exp_id.strip():
                    logger.error(
                        "Invalid experiment_id in list: must be non-empty strings"
                    )
                    raise ValueError("Each experiment ID must be a non-empty string")
                exp.append(exp_id.strip())
        self._query["experiment_id"] = exp
        logger.info("Experiment ID(s) set to: %s", exp)
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
        logger.debug("Setting table_id to: %s", table_id)
        if not isinstance(table_id, str) or not table_id.strip():
            logger.error("Invalid table_id parameter: must be non-empty string")
            raise ValueError("Table ID must be a non-empty string")
        self._query["table_id"] = table_id.strip()
        logger.info("Table ID set to: %s", table_id.strip())
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
        logger.debug("Setting grid_label to: %s", grid_label)
        if not isinstance(grid_label, str) or not grid_label.strip():
            logger.error("Invalid grid_label parameter: must be non-empty string")
            raise ValueError("Grid label must be a non-empty string")
        self._query["grid_label"] = grid_label.strip()
        logger.info("Grid label set to: %s", grid_label.strip())
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
        logger.debug("Setting variable to: %s", variable)
        if not isinstance(variable, str) or not variable.strip():
            logger.error("Invalid variable parameter: must be non-empty string")
            raise ValueError("Variable must be a non-empty string")
        self._query["variable_id"] = variable.strip()
        logger.info("Variable set to: %s", variable.strip())
        return self

    def station_id(self, station_id: str | list[str]) -> "ClimateData":
        """Set the station identifier for the query.

        Parameters
        ----------
        station_id : str
            The station ID (e.g., "ASOSAWOS_72019300117", "ASOSAWOS_72020200118").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        logger.debug("Setting station_id to: %s", station_id)
        stn = []
        if not isinstance(station_id, (str, list)):
            logger.error(
                "Invalid experiment_id parameter: must be string or list of strings"
            )
            raise ValueError("Station ID must be a non-empty string or list of strings")
        if isinstance(station_id, str):
            if not station_id.strip():
                logger.error("Invalid station_id parameter: empty string")
                raise ValueError("Station ID must be a non-empty string")
            stn.append(station_id.strip())
        else:
            for id in station_id:
                if not isinstance(id, str) or not id.strip():
                    logger.error(
                        "Invalid station_id in list: must be non-empty strings"
                    )
                    raise ValueError("Each station ID must be a non-empty string")
                stn.append(id.strip())
        self._query["station_id"] = stn
        logger.info("Station ID(s) set to: %s", stn)
        return self

    def network_id(self, network_id: str | list[str]) -> "ClimateData":
        """Set the network identifier for the query.

        Parameters
        ----------
        network_id : str | list[str]
            The network ID (e.g., "ASOSAWOS", "CWOP").

        Returns
        -------
        ClimateData
            The current instance for method chaining.

        """
        logger.debug("Setting network_id to: %s", network_id)
        net = []
        if not isinstance(network_id, (str, list)):
            logger.error(
                "Invalid network_id parameter: must be string or list of strings"
            )
            raise ValueError("Network ID must be a non-empty string or list of strings")
        if isinstance(network_id, str):
            if not network_id.strip():
                logger.error("Invalid network_id parameter: empty string")
                raise ValueError("Network ID must be a non-empty string")
            net.append(network_id.strip())
        else:
            for id in network_id:
                if not isinstance(id, str) or not id.strip():
                    logger.error(
                        "Invalid network_id in list: must be non-empty strings"
                    )
                    raise ValueError("Each network ID must be a non-empty string")
                net.append(id.strip())
        self._query["network_id"] = net
        logger.info("Network ID(s) set to: %s", net)
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
        logger.debug("Setting processes to: %s", processes)
        if not isinstance(processes, dict):
            logger.error("Invalid processes parameter: must be a dictionary")
            raise ValueError("Processes must be a dictionary")
        self._query["processes"] = processes.copy()
        logger.info("Processes set: %d operations configured", len(processes))
        return self

    # Main execution method
    def get(self) -> Optional[Any]:
        """Execute the configured query and retrieve climate data.

        Validates required parameters, creates the appropriate dataset using
        the factory pattern, executes the query, and resets the query state
        for the next use.

        Thread Safety
        -------------
        This method takes a snapshot of the query at the start of execution,
        making it safe to call from multiple threads on the same ClimateData
        instance. However, for maximum clarity and safety, it is recommended
        to use separate ClimateData instances in multi-threaded scenarios.

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
        logger.info("Starting data retrieval with query: %s", self._query)
        data = None

        # Take a snapshot of the query for thread-safety
        # This allows concurrent calls to get() without corrupting each other
        query_snapshot = copy.deepcopy(self._query)

        # Validate required parameters
        logger.debug("Validating required parameters")
        if not self._validate_required_parameters():
            logger.warning("Required parameter validation failed")
            self._reset_query()
            return None

        try:
            # Create dataset using factory with the snapshot
            logger.debug("Creating dataset using factory")
            dataset = self._factory.create_dataset(query_snapshot)
            logger.info("Dataset created successfully")
        except (ValueError, KeyError, TypeError) as e:
            logger.error("Error during dataset creation: %s", str(e))
            logger.debug("Traceback:", exc_info=True)
            self._reset_query()
            return None

        try:
            # Execute the query with the snapshot
            logger.debug("Executing query")
            data = dataset.execute(query_snapshot)
            # check if empty dataset
            # Check if data is empty/null
            if (
                data is None
                or (hasattr(data, "nbytes") and data.nbytes == 0)
                or (isinstance(data, dict) and not data)
            ):
                logger.warning("⚠️ Warning: Retrieved dataset is empty.")

            else:
                logger.info("✅ Data retrieval successful!")

        except (ValueError, KeyError, IOError, RuntimeError) as e:
            logger.error("❌ Data retrieval failed: %s", str(e))
            logger.debug("Traceback:", exc_info=True)

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
        # Always require catalog
        required_params = ["catalog"]

        # Only require these params for specific catalogs
        catalog = self._query.get("catalog", UNSET)
        if catalog in ["renewable energy generation", "cadcat"]:
            required_params.extend(["variable_id", "grid_label", "table_id"])
        elif catalog == "hdp":
            required_params.extend(["network_id"])

        missing_params = []

        for param in required_params:
            if self._query[param] is UNSET:
                missing_params.append(param)

        if missing_params:
            logger.error("Missing required parameters: %s", ", ".join(missing_params))
            logger.info("Use the show_*_options() methods to see available values")
            # Maintain backward-compatible printed error for tests and
            # user-facing scripts that expect a printed ERROR: prefix.
            try:
                print(
                    f"ERROR: Missing required parameters: {', '.join(missing_params)}"
                )
            except Exception:
                pass
            return False

        return True

    # Query inspection methods
    def show_query(self) -> None:
        """Display the current query configuration."""
        msg = "Current Query:"
        logger.info(msg)
        logger.info("%s", "-" * len(msg))
        try:
            print(msg)
            print("%s" % ("-" * len(msg)))
        except Exception:
            pass
        for key, value in self._query.items():
            display_value = value if value is not UNSET else "UNSET"
            logger.info("%s: %s", key, display_value)
            try:
                print(f"{key}: {display_value}")
            except Exception:
                pass

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

    def show_station_id_options(self) -> None:
        """Display available station ID options."""
        self._show_options("station_id", "station_id options (Weather station names)")

    def show_network_id_options(self) -> None:
        """Display available network ID options."""
        self._show_options("network_id", "network_id options (Weather network names)")

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
        logger.info(msg)
        logger.info("%s", "-" * len(msg))
        try:
            print(msg)
            print("%s" % ("-" * len(msg)))
        except Exception:
            pass
        try:
            for processor in self._factory.get_processors():
                logger.info("%s", processor)

            logger.info("\n")
        except Exception as e:
            logger.error("Error retrieving processors: %s", e, exc_info=True)

    def show_station_options(self) -> None:
        """Display available station options for data retrieval."""
        msg = "Stations (Available weather stations for localization):"
        logger.info(msg)
        logger.info("%s", "-" * len(msg))
        try:
            print(msg)
            print("%s" % ("-" * len(msg)))
        except Exception:
            pass
        try:
            stations = self._factory.get_stations()
            if not stations:
                logger.info("No stations available with current parameters")

            else:
                for station in sorted(stations):
                    logger.info("%s", station)

                logger.info("\n")
        except Exception as e:
            logger.error("Error retrieving stations: %s", e, exc_info=True)

    def show_boundary_options(self, type=UNSET) -> None:
        """Display available boundaries for spatial queries."""
        if type is UNSET:
            msg = "Boundary Types (call again with option type='...' to see options for any type):"
        else:
            msg = "Available {} Boundaries:".format(
                " ".join([x.capitalize() for x in type.split("_")])
            )
        logger.info(msg)
        logger.info("%s", "-" * len(msg))

        try:
            boundaries = self._factory.get_boundaries(type)
            if not boundaries:
                logger.info("No boundaries available with current parameters")

            else:
                for boundary in sorted(boundaries):
                    logger.info("%s", boundary)

                logger.info("\n")
        except Exception as e:
            logger.error("Error retrieving boundaries: %s", e, exc_info=True)

    def show_all_options(self) -> None:
        """Display all available options for exploration."""
        data_title = "CAL ADAPT DATA -- ALL AVAILABLE OPTIONS USING CLIMAKITAE"
        logger.info("%s", "=" * len(data_title))
        logger.info(data_title)
        logger.info("%s", "=" * len(data_title))

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
            ("show_station_id_options", "Station IDs"),
            ("show_network_id_options", "Network IDs"),
            ("show_processors", "Processors"),
            ("show_station_options", "Stations"),
        ]

        for method_name, section_title in option_methods:
            try:
                getattr(self, method_name)()
            except Exception as e:
                logger.error(
                    "Error displaying %s: %s", section_title.lower(), e, exc_info=True
                )

        logger.info("%s", "\n" + "=" * 60)
        logger.info("Current Query Status:")
        logger.info("%s", "=" * 60)
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
        logger.info("%s:", title)
        logger.info("%s", "-" * (len(title) + 1))
        try:
            # Print header for backward compatibility/tests
            try:
                print(f"{title}:")
                print("%s" % ("-" * (len(title) + 1)))
            except Exception:
                pass
            current_query = {k: v for k, v in self._query.items() if v is not UNSET}
            options = self._factory.get_catalog_options(option_type, current_query)
            if not options:
                logger.info("No options available with current parameters")

            else:
                max_len = max(len(option) for option in options)
                for option in sorted(options):
                    formatted = self._format_option(
                        option, option_type, spacing=4 + max_len - len(option)
                    )
                    logger.info("%s", formatted)

                logger.info("\n")
        except Exception as e:
            logger.error("Error retrieving options: %s", e, exc_info=True)

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
