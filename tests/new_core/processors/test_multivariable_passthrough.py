"""
Shared pass-through tests for processors applied to multi-variable Datasets.

Each processor here operates on the whole Dataset without per-variable
logic. The single parametrized test checks that every variable survives,
keeps its own ``units`` attribute, and stays on a shared time coordinate.
"""

from typing import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.new_core.processors.convert_to_local_time import ConvertToLocalTime
from climakitae.new_core.processors.drop_leap_days import DropLeapDays
from climakitae.new_core.processors.filter_unadjusted_models import (
    FilterUnAdjustedModels,
)
from climakitae.new_core.processors.time_slice import TimeSlice
from climakitae.new_core.processors.update_attributes import UpdateAttributes


@pytest.fixture
def two_variable_dataset() -> xr.Dataset:
    """Build an hourly t2max (K) / prec (mm/day) Dataset spanning a leap day.

    Returns
    -------
    xr.Dataset
        Dataset with ``t2max`` and ``prec`` on shared ``sim``, ``time``,
        ``lat`` and ``lon`` coordinates.
    """
    time = pd.date_range("2000-02-28 00", "2000-03-01 23", freq="1h")
    shape = (1, time.size, 1, 1)
    dims = ("sim", "time", "lat", "lon")
    dataset = xr.Dataset(
        {
            "t2max": (dims, np.full(shape, 290.0)),
            "prec": (dims, np.full(shape, 2.0)),
        },
        coords={
            "sim": ["WRF_EC-Earth3_r1i1p1f1"],
            "time": time,
            "lat": [35.0],
            "lon": [-119.0],
        },
        attrs={
            "intake_esm_attrs:activity_id": "WRF",
            "intake_esm_attrs:source_id": "EC-Earth3",
            "intake_esm_attrs:member_id": "r1i1p1f1",
        },
    )
    dataset["t2max"].attrs["units"] = "K"
    dataset["prec"].attrs["units"] = "mm/day"
    return dataset


@pytest.mark.parametrize(
    ("processor_factory", "context"),
    [
        (lambda: TimeSlice(value=("2000-02-28", "2000-02-29")), {}),
        (lambda: DropLeapDays(value="yes"), {}),
        (
            lambda: ConvertToLocalTime(
                value={"convert": "yes", "reindex_time_axis": "no"}
            ),
            {},
        ),
        (lambda: FilterUnAdjustedModels(value="yes"), {}),
        (lambda: UpdateAttributes(), {_NEW_ATTRS_KEY: {"time_slice": "2000"}}),
    ],
    ids=[
        "time_slice",
        "drop_leap_days",
        "convert_to_local_time",
        "filter_unadjusted_models",
        "update_attributes",
    ],
)
def test_processor_preserves_multivariable_fields(
    processor_factory: Callable,
    context: dict,
    two_variable_dataset: xr.Dataset,
) -> None:
    """Check a processor keeps both variables, their units, and shared time.

    Parameters
    ----------
    processor_factory : Callable
        Zero-argument callable returning the processor under test.
    context : dict
        Processing context passed to ``execute``.
    two_variable_dataset : xr.Dataset
        Shared two-variable input fixture.
    """
    result = processor_factory().execute(two_variable_dataset, context=context)

    assert set(result.data_vars) == {"t2max", "prec"}
    assert result["t2max"].attrs["units"] == "K"
    assert result["prec"].attrs["units"] == "mm/day"
    assert result.time.size > 0
    assert result["t2max"].time.identical(result["prec"].time)
