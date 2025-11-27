"""
Units tests for the `metric_calc` processor.

This processor calculates various statistical metrics (e.g., mean, median, percentiles)
over specified dimensions of the input data.

The tests validate the correct functionality of the processor, including handling of
different metrics, percentiles, dimensions, and edge cases.
"""

import numpy as np
import pytest
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access.data_catalog import DataCatalog
from climakitae.new_core.processors.metric_calc import MetricCalc
