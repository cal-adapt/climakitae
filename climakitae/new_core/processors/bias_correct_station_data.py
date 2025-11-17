"""Station Bias Correction Processor for ClimakitAE.

This module provides the StationBiasCorrection processor for bias-correcting
gridded climate model data to weather station locations using Quantile Delta
Mapping (QDM) with historical observational data from the HadISD dataset.

The processor performs the following operations:
1. Loads HadISD weather station observations from S3 zarr stores
2. Finds the closest gridcell in the climate model data to each station
3. Applies quantile delta mapping bias correction using historical overlap period
4. Returns bias-corrected data at station locations with station metadata

Classes
-------
StationBiasCorrection : DataProcessor
    Main processor for station-based bias correction using QDM method.

Examples
--------
>>> # Create processor for single station
>>> processor = StationBiasCorrection(
...     stations=["Sacramento (KSAC)"],
...     station_metadata=stations_gdf,
...     time_slice=(2030, 2060)
... )
>>> result = processor.execute(gridded_data, context)

>>> # Multiple stations with custom bias correction parameters
>>> processor = StationBiasCorrection(
...     stations=["Sacramento (KSAC)", "San Francisco (KSFO)"],
...     station_metadata=stations_gdf,
...     time_slice=(2030, 2060),
...     window=60,  # 60-day window instead of default 90
...     nquantiles=30  # 30 quantiles instead of default 20
... )
>>> result = processor.execute(gridded_data, context)

Notes
-----
- Requires gridded data to include historical period (1980-2014) for bias correction
- Station observational data is available through 2014-08-31
- Uses xclim's QuantileDeltaMapping for bias correction
- Converts all data to noleap calendar for consistency
- Final output is time-sliced to user's requested period after bias correction
"""

import logging
from functools import partial
from typing import Any, Dict, Iterable, Union

import geopandas as gpd
import xarray as xr
from xsdba import Grouper
from xsdba.adjustment import QuantileDeltaMapping

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)
from climakitae.util.unit_conversions import convert_units
from climakitae.util.utils import get_closest_gridcell

# Module logger
logger = logging.getLogger(__name__)


@register_processor("bias_correct_station_data", priority=150)
class StationBiasCorrection(DataProcessor):
    """Bias-correct gridded climate data to weather station locations using QDM.

    This processor applies Quantile Delta Mapping (QDM) bias correction to gridded
    climate model data using historical observations from HadISD weather stations.
    The method corrects for systematic biases in climate model output by matching
    the statistical distribution of model data to observed data during a historical
    training period, then applying these corrections to future projections.

    The processor handles:
    - Loading HadISD station observations from S3 zarr stores
    - Preprocessing station data (unit conversions, calendar conversions)
    - Finding closest gridcells to station locations
    - Training QDM bias correction on historical overlap period (1980-2014)
    - Applying corrections to user-specified time period
    - Preserving station metadata in output

    Parameters
    ----------
    stations : list[str]
        List of station names to process (e.g., ["Sacramento (KSAC)", "San Francisco (KSFO)"])
    station_metadata : gpd.GeoDataFrame
        GeoDataFrame with station information including 'station', 'station id',
        'latitude', 'longitude', and 'elevation' columns
    time_slice : tuple[int, int]
        Start and end years for final output time slice (e.g., (2030, 2060))
    window : int, optional
        Window size in days for seasonal grouping (default: 90, representing +/- 45 days)
    nquantiles : int, optional
        Number of quantiles for QDM training (default: 20)
    group : str, optional
        Temporal grouping strategy for bias correction (default: "time.dayofyear")
    kind : str, optional
        Adjustment kind: "+" for additive (temperature) or "*" for multiplicative
        (precipitation) (default: "+")

    Attributes
    ----------
    stations : list[str]
        Station names to process
    station_metadata : gpd.GeoDataFrame
        Station metadata including coordinates and IDs
    time_slice : tuple[int, int]
        Final output time slice
    window : int
        Seasonal grouping window in days
    nquantiles : int
        Number of quantiles for QDM
    group : str
        Temporal grouping strategy
    kind : str
        Adjustment kind (additive or multiplicative)
    name : str
        Processor name for context tracking

    Methods
    -------
    execute(result, context)
        Apply station bias correction to gridded data
    update_context(context)
        Update context with bias correction operation metadata
    set_data_accessor(catalog)
        Set data catalog accessor (not used in this processor)

    See Also
    --------
    climakitae.util.utils.get_closest_gridcell : Find closest gridcell to point
    xsdba.adjustment.QuantileDeltaMapping : QDM bias correction implementation

    References
    ----------
    .. [1] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015). Bias correction of
       GCM precipitation by quantile mapping: How well do methods preserve changes
       in quantiles and extremes? Journal of Climate, 28(17), 6938-6959.
    """

    def __init__(
        self,
        stations: list[str],
        station_metadata: gpd.GeoDataFrame,
        time_slice: tuple[int, int],
        window: int = 90,
        nquantiles: int = 20,
        group: str = "time.dayofyear",
        kind: str = "+",
    ):
        """Initialize the station bias correction processor.

        Parameters
        ----------
        stations : list[str]
            List of station names to process
        station_metadata : gpd.GeoDataFrame
            GeoDataFrame containing station information
        time_slice : tuple[int, int]
            Start and end years for final output
        window : int, optional
            Window size in days for seasonal grouping (default: 90)
        nquantiles : int, optional
            Number of quantiles for QDM (default: 20)
        group : str, optional
            Temporal grouping strategy (default: "time.dayofyear")
        kind : str, optional
            Adjustment kind: "+" or "*" (default: "+")
        """
        self.stations = stations
        self.station_metadata = station_metadata
        self.time_slice = time_slice
        self.window = window
        self.nquantiles = nquantiles
        self.group = group
        self.kind = kind
        self.name = "station_bias_correction"

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """Run the processor on the provided data.

        Parameters
        ----------
        result : xr.Dataset or xr.DataArray or Iterable of these
            The data to be processed or sliced.
        context : dict
            The context for the processor. This is not used in this implementation but is included for consistency with the DataProcessor interface.

        Returns
        -------
        xr.Dataset, xr.DataArray, or Iterable of these
            The processed or sliced data. This can be a single Dataset/DataArray or an iterable of them.

        """

    def update_context(self, context: Dict[str, Any]):
        """Update the context with information about the transformation.

        Parameters
        ----------
        context : dict[str, Any]
            Parameters for processing the data. The context is updated in place.

        Returns
        -------
        None

        """

        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        context[_NEW_ATTRS_KEY][
            self.name
        ] = f"""Process '{self.name}' applied to the data. Transformation was done using the following value: {self.value}."""

    def set_data_accessor(self, catalog: DataCatalog):
        """Set the data accessor for the processor.

        Parameters
        ----------
        catalog : DataCatalog
            Data catalog for accessing datasets.

        Returns
        -------
        None

        """
        # Placeholder for setting data accessor
        pass
