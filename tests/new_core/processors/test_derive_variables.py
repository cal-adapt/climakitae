"""Tests for the internal DeriveVariables processor."""

import numpy as np
import pytest
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.derived_variables import registry
from climakitae.new_core.param_validation.cadcat_param_validator import DataValidator
from climakitae.new_core.param_validation.derive_variables_param_validator import (
    validate_derive_variables_param,
)
from climakitae.new_core.processors.derive_variables import DeriveVariables


@pytest.fixture
def wind_dataset():
    """Return a small dataset with wind components and precipitation."""
    coords = {"time": [0, 1], "y": [0.0, 1.0], "x": [0.0, 1.0]}
    shape = (2, 2, 2)
    return xr.Dataset(
        {
            "u10": (("time", "y", "x"), np.full(shape, 3.0)),
            "v10": (("time", "y", "x"), np.full(shape, 4.0)),
            "pr": (("time", "y", "x"), np.ones(shape)),
        },
        coords=coords,
    )


@pytest.fixture
def keep_inputs_variable(monkeypatch):
    """Register a derived variable that keeps its dependency."""

    def add_one(dataset):
        dataset["tasmax_plus_one"] = dataset["tasmax"] + 1
        return dataset

    monkeypatch.setitem(
        registry._DERIVED_METADATA,
        "tasmax_plus_one",
        registry.DerivedVariableInfo(
            name="tasmax_plus_one",
            depends_on=["tasmax"],
            description="Test derived field",
            units="K",
            func=add_one,
            source="user",
            drop_dependencies=False,
        ),
    )


def test_native_request_returns_data_unchanged(wind_dataset):
    """Do nothing when no requested variable is derived."""
    context = {"variable_id": ["u10", "pr"]}

    result = DeriveVariables().execute(wind_dataset, context)

    assert result is wind_dataset
    assert _NEW_ATTRS_KEY not in context


@pytest.mark.parametrize("container", [dict, list, tuple])
def test_single_derived_request_returns_only_requested_output(wind_dataset, container):
    """Calculate a derived variable and drop its dependencies."""
    data = {"key": wind_dataset} if container is dict else container([wind_dataset])
    context = {"variable_id": "wind_speed_10m"}

    result = DeriveVariables().execute(data, context)

    dataset = result["key"] if container is dict else result[0]
    assert type(result) is container
    assert list(dataset.data_vars) == ["wind_speed_10m"]
    np.testing.assert_allclose(dataset["wind_speed_10m"].values, 5.0)
    assert "wind_speed_10m" in context[_NEW_ATTRS_KEY]["derive_variables"]


def test_mixed_request_keeps_request_order_and_shared_dependencies(wind_dataset):
    """Keep a requested dependency when another output also uses it."""
    context = {"variable_id": ["wind_direction_10m", "u10", "wind_speed_10m"]}

    result = DeriveVariables().execute(wind_dataset, context)

    assert list(result.data_vars) == ["wind_direction_10m", "u10", "wind_speed_10m"]
    np.testing.assert_allclose(result["u10"].values, 3.0)


def test_drop_dependencies_false_keeps_inputs_after_requested(keep_inputs_variable):
    """Keep dependencies of functions registered with drop_dependencies=False."""
    dataset = xr.Dataset(
        {"tasmax": ("time", [1.0, 2.0]), "pr": ("time", [0.0, 0.0])},
        coords={"time": [0, 1]},
    )

    result = DeriveVariables().execute(
        dataset, {"variable_id": ["pr", "tasmax_plus_one"]}
    )

    assert list(result.data_vars) == ["pr", "tasmax_plus_one", "tasmax"]
    np.testing.assert_allclose(result["tasmax_plus_one"].values, [2.0, 3.0])


def test_missing_dependency_raises(wind_dataset):
    """Raise a clear error when a dependency is not in the data."""
    with pytest.raises(ValueError, match=r"missing dependencies \['v10'\]"):
        DeriveVariables().execute(
            {"key": wind_dataset.drop_vars("v10")},
            {"variable_id": "wind_speed_10m"},
        )


@pytest.mark.parametrize(
    "variable_id, expected",
    [
        ("wind_speed_10m", True),
        (["pr", "wind_speed_10m"], True),
        ("pr", False),
        (UNSET, False),
    ],
)
def test_default_processors_add_derive_variables_for_derived_requests(
    variable_id, expected
):
    """Add the internal processor only when a derived variable is requested."""
    validator = DataValidator.__new__(DataValidator)

    defaults = validator.get_default_processors({"variable_id": variable_id})

    assert ("derive_variables" in defaults) is expected
    assert defaults.get("derive_variables", UNSET) is UNSET


def test_derive_variables_validator_rejects_user_values():
    """Accept only the internal UNSET value."""
    assert validate_derive_variables_param(UNSET)
    assert not validate_derive_variables_param("yes")
