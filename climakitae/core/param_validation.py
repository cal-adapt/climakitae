from abc import ABC, abstractmethod
from typing import Any, Dict, List

from climakitae.core.constants import UNSET


class ParameterValidator(ABC):
    """Abstract base class for parameter validation."""

    @abstractmethod
    def validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters and return processed parameters.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Parameters to validate

        Returns
        -------
        Dict[str, Any]
            Processed parameters

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        pass


class ClimateDataValidator(ParameterValidator):
    """Validator for climate data parameters."""

    def validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate climate data parameters."""
        processed_params = parameters.copy()

        # Process scenario into scenario_ssp and scenario_historical
        if processed_params["approach"] == "Warming Level":
            processed_params["scenario_ssp"] = ["n/a"]
            processed_params["scenario_historical"] = ["n/a"]
        elif processed_params["approach"] == "Time":
            scenario = processed_params.get("scenario", [])
            if not scenario:
                processed_params["scenario_ssp"] = []
                processed_params["scenario_historical"] = []
            elif "Historical Reconstruction" in scenario:
                processed_params["scenario_historical"] = [
                    x for x in scenario if "Historical" in x
                ]
                processed_params["scenario_ssp"] = []
            else:
                processed_params["scenario_ssp"] = [
                    x for x in scenario if "Historical" not in x
                ]
                processed_params["scenario_historical"] = (
                    ["Historical Climate"] if "Historical Climate" in scenario else []
                )

        return processed_params


class TimeClimateValidator(ClimateDataValidator):
    """Validator for time-based climate data parameters."""

    def validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate time-based climate data parameters."""
        processed_params = super().validate(parameters)

        # Additional validation specific to time-based approach
        if "Historical Reconstruction" in processed_params.get(
            "scenario_historical", []
        ) and processed_params.get("scenario_ssp", []):
            raise ValueError(
                "Historical Reconstruction data is not available with SSP data."
            )

        # Validate unit selection
        self._check_valid_unit_selection(processed_params)

        # Check if scenarios are selected
        scenario_selections = processed_params.get(
            "scenario_ssp", []
        ) + processed_params.get("scenario_historical", [])
        if not scenario_selections:
            raise ValueError("Please select at least one dataset.")

        return processed_params

    def _check_valid_unit_selection(self, parameters: Dict[str, Any]) -> None:
        """Check for valid unit selection."""
        # Implementation of unit validation logic
        pass


class WarmingLevelClimateValidator(ClimateDataValidator):
    """Validator for warming level climate data parameters."""

    def validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate warming level climate data parameters."""
        processed_params = super().validate(parameters)

        # Force time_slice to cover entire period for warming level approach
        processed_params["time_slice"] = (1980, 2100)

        # Additional warming level specific validation
        if not processed_params.get("warming_level", []) or processed_params[
            "warming_level"
        ] == ["n/a"]:
            raise ValueError("Please select at least one warming level.")

        if processed_params.get("warming_level_window") is None:
            processed_params["warming_level_window"] = 15  # Default

        return processed_params


class StationDataValidator(ParameterValidator):
    """Validator for station data parameters."""

    def __init__(self, stations_gdf):
        """Initialize with stations geodataframe."""
        self.stations_gdf = stations_gdf

    def validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate station data parameters."""
        processed_params = parameters.copy()

        # Ensure station data requirements
        processed_params["variable"] = "Air Temperature at 2m"
        processed_params["timescale"] = "hourly"
        processed_params["downscaling_method"] = "Dynamical"

        # Process scenario info
        if processed_params["approach"] == "Warming Level":
            processed_params["scenario_ssp"] = ["n/a"]
            processed_params["scenario_historical"] = ["n/a"]
        else:  # "Time"
            if processed_params["scenario"] is None:
                if processed_params["time_slice"] is None:
                    processed_params["scenario"] = ["Historical Climate"]
                else:
                    processed_params["scenario"] = []

            if processed_params["time_slice"] is not None:
                if (
                    any(value < 2015 for value in processed_params["time_slice"])
                    and "Historical Climate" not in processed_params["scenario"]
                ):
                    processed_params["scenario"].append("Historical Climate")
                if any(
                    value >= 2015 for value in processed_params["time_slice"]
                ) and not any("SSP" in item for item in processed_params["scenario"]):
                    processed_params["scenario"].append("SSP 3-7.0")

            # Split scenario
            if "Historical Reconstruction" in processed_params["scenario"]:
                processed_params["scenario_historical"] = [
                    x for x in processed_params["scenario"] if "Historical" in x
                ]
                processed_params["scenario_ssp"] = []
            else:
                processed_params["scenario_ssp"] = [
                    x for x in processed_params["scenario"] if "Historical" not in x
                ]
                processed_params["scenario_historical"] = (
                    ["Historical Climate"]
                    if "Historical Climate" in processed_params["scenario"]
                    else []
                )

        # Validate stations if provided
        if processed_params.get("stations"):
            processed_params["stations"] = self._validate_stations(
                processed_params["stations"]
            )

        return processed_params

    def _validate_stations(self, stations: List[str]) -> List[str]:
        """Validate station names against available stations."""
        # Station validation logic
        valid_stations = sorted(self.stations_gdf.station.values)
        validated_stations = []

        for station in stations:
            if station in valid_stations:
                validated_stations.append(station)
            else:
                # Find closest match
                import difflib

                matches = difflib.get_close_matches(
                    station, valid_stations, n=1, cutoff=0.6
                )
                if matches:
                    print(
                        f"Station '{station}' not found. Using closest match: '{matches[0]}'"
                    )
                    validated_stations.append(matches[0])
                else:
                    raise ValueError(f"Invalid station name: {station}")

        return validated_stations


# Abstract Data Processor Classes
class DataProcessor(ABC):
    """Abstract base class for data processing."""

    @abstractmethod
    def process(self, data, parameters):
        """Process raw data into the required format."""
        pass


class ClimateDataProcessor(DataProcessor):
    """Base processor for climate data."""

    def process(self, data_dict, parameters):
        """Process climate data dictionary into a DataArray."""
        # Merge data along dimensions
        da = self._merge_data(data_dict, parameters)

        # Handle unit conversions
        da = self._process_units(da, parameters)

        return da

    def _merge_data(self, data_dict, parameters):
        """Merge data from multiple sources."""
        # Implementation of data merging logic
        pass

    def _process_units(self, da, parameters):
        """Process units for the DataArray."""
        # Implementation of unit conversion logic
        pass


class TimeClimateDataProcessor(ClimateDataProcessor):
    """Processor for time-based climate data."""

    def process(self, data_dict, parameters):
        """Process time-based climate data."""
        # Get base processing
        da = super().process(data_dict, parameters)

        # Time-specific processing
        # ...

        return da


class WarmingLevelDataProcessor(ClimateDataProcessor):
    """Processor for warming level climate data."""

    def process(self, data_dict, parameters):
        """Process warming level climate data."""
        # Get base processing
        da = super().process(data_dict, parameters)

        # Apply warming levels approach
        da = self._apply_warming_levels_approach(da, parameters)

        return da

    def _apply_warming_levels_approach(self, da, parameters):
        """Apply warming levels approach to data."""
        # Implementation of warming levels approach
        pass


class StationDataProcessor(DataProcessor):
    """Base processor for station data."""

    def process(self, data_dict, parameters):
        """Process station data."""
        # Base station data processing
        da = self._process_station_data(data_dict, parameters)

        return da

    def _process_station_data(self, data_dict, parameters):
        """Process station data."""
        # Implementation of station data processing
        pass


class TimeStationDataProcessor(StationDataProcessor):
    """Processor for time-based station data."""

    def process(self, data_dict, parameters):
        """Process time-based station data."""
        # Get base processing
        da = super().process(data_dict, parameters)

        # Time-specific station processing
        # ...

        return da


class WarmingLevelStationProcessor(StationDataProcessor):
    """Processor for warming level station data."""

    def process(self, data_dict, parameters):
        """Process warming level station data."""
        # Get base processing
        da = super().process(data_dict, parameters)

        # Apply warming levels approach to station data
        da = self._apply_warming_levels_approach(da, parameters)

        return da

    def _apply_warming_levels_approach(self, da, parameters):
        """Apply warming levels approach to station data."""
        # Implementation of warming levels approach for stations
        pass
