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


def _ordered_steps(processes, variable_id):
    """Build the processor list with the real registry and default processors."""
    from unittest.mock import MagicMock, patch

    from climakitae.core.constants import PROC_KEY
    from climakitae.new_core.dataset_factory import DatasetFactory

    with patch("climakitae.new_core.dataset_factory.DataCatalog") as mock_catalog:
        mock_catalog.return_value = MagicMock()
        factory = DatasetFactory()
    query = {
        PROC_KEY: dict(processes),
        "variable_id": variable_id,
        "experiment_id": "historical",
        "_catalog_key": "cadcat",
        _NEW_ATTRS_KEY: {},
    }
    with patch.object(
        factory,
        "create_validator",
        return_value=DataValidator.__new__(DataValidator),
    ):
        return factory._get_list_of_processing_steps(query)


def _grid_dataset(u_value=3.0):
    """Return a 5x5 lat/lon dataset with wind components."""
    shape = (2, 5, 5)
    return xr.Dataset(
        {
            "u10": (("time", "lat", "lon"), np.full(shape, u_value)),
            "v10": (("time", "lat", "lon"), np.full(shape, 4.0)),
        },
        coords={
            "time": [0, 1],
            "lat": np.linspace(33.0, 37.0, 5),
            "lon": np.linspace(-121.0, -117.0, 5),
        },
    )


def test_pipeline_orders_clip_then_derive_then_concat():
    """Run clip first, then the derivation, before concat and warming levels."""
    steps = _ordered_steps({"clip": (35.0, -119.0)}, "wind_speed_10m")

    names = [type(step).__name__ for step in steps]

    assert names.index("Clip") < names.index("DeriveVariables")
    assert names.index("DeriveVariables") < names.index("Concat")
    assert "DeriveVariables" not in [
        type(step).__name__ for step in _ordered_steps({}, "u10")
    ]


def test_derived_function_receives_clipped_data(monkeypatch):
    """Calculate derived variables on the clipped point, not the full grid."""
    seen_sizes = []

    def spy(dataset):
        seen_sizes.append(dict(dataset.sizes))
        dataset["spy_speed"] = np.hypot(dataset["u10"], dataset["v10"])
        return dataset

    monkeypatch.setitem(
        registry._DERIVED_METADATA,
        "spy_speed",
        registry.DerivedVariableInfo(
            name="spy_speed",
            depends_on=["u10", "v10"],
            description="Spy derived field",
            units="m/s",
            func=spy,
            source="user",
            drop_dependencies=True,
        ),
    )
    steps = [
        step
        for step in _ordered_steps({"clip": (35.0, -119.0)}, "spy_speed")
        if type(step).__name__ in ("Clip", "DeriveVariables")
    ]
    data = {"a": _grid_dataset(), "b": _grid_dataset()}
    context = {"variable_id": "spy_speed"}

    for step in steps:
        data = step.execute(data, context)

    assert seen_sizes == [{"time": 2}, {"time": 2}]
    for dataset in data.values():
        assert list(dataset.data_vars) == ["spy_speed"]
        np.testing.assert_allclose(dataset["spy_speed"].values, 5.0)


def test_derived_inputs_use_nearest_cell_without_nan_search():
    """Keep the nearest cell for derived inputs, also when it has NaN values."""
    from climakitae.new_core.processors.clip import Clip

    dataset = _grid_dataset()
    dataset["u10"][:, 2, 2] = np.nan
    context = {"variable_id": "wind_speed_10m"}

    clipped = Clip((35.0, -119.0)).execute({"a": dataset}, context)
    result = DeriveVariables().execute(clipped, context)

    assert float(result["a"]["lat"]) == 35.0
    assert float(result["a"]["lon"]) == -119.0
    assert np.isnan(result["a"]["wind_speed_10m"].values).all()
