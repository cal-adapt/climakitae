import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.explore import threshold_tools
from climakitae.explore.threshold_tools import _calc_average_ess_gridded_data


class TestThresholdTools:

    def test_calculate_ess(self):
        # Not validating the result, just checking this runs
        test = xr.DataArray(data=np.arange(0, 100, 1))
        result = threshold_tools.calculate_ess(test)
        assert isinstance(result, xr.DataArray)
        assert not np.isnan(result.data)
        assert result.name == "ess"

        result = threshold_tools.calculate_ess(test, 3)
        assert isinstance(result, xr.DataArray)
        assert not np.isnan(result.data)
        assert result.name == "ess"

    def test__calc_average_ess_gridded_data(self):
        test_data = test_data = np.random.rand(10, 10, 365 * 3) * 100
        test = xr.DataArray(
            data=test_data,
            coords={
                "x": np.arange(10, 20, 1),
                "y": np.arange(10, 20, 1),
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = _calc_average_ess_gridded_data(test, 1)
        assert not np.isnan(result)
        assert isinstance(result, float)
