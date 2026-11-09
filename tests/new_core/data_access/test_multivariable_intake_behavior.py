"""Characterize pinned intake-esm aggregation for multi-variable queries."""

import numpy as np
import pandas as pd
import xarray as xr
import intake
import pytest


def _make_local_catalog(
    tmp_path, catalog_kind="cadcat", registry=None, spatial_grid=False
):
    """Create a local multi-variable catalog with incomplete source coverage."""
    catalog_profiles = {
        "cadcat": {
            "groupby_attrs": [
                "activity_id",
                "institution_id",
                "source_id",
                "experiment_id",
                "table_id",
                "grid_label",
            ],
            "aggregations": [
                {"type": "join_new", "attribute_name": "member_id"},
                {"type": "union", "attribute_name": "variable_id"},
            ],
        },
        "renewables": {
            "groupby_attrs": [
                "installation",
                "activity_id",
                "institution_id",
                "source_id",
                "experiment_id",
                "table_id",
                "grid_label",
            ],
            "aggregations": [
                {"type": "union", "attribute_name": "variable_id"}
            ],
        },
        "sup3rcc": {
            "groupby_attrs": [
                "activity_id",
                "institution_id",
                "source_id",
                "experiment_id",
                "table_id",
                "grid_label",
            ],
            "aggregations": [
                {"type": "union", "attribute_name": "variable_id"}
            ],
        },
    }
    profile = catalog_profiles[catalog_kind]
    assets = {
        ("r1i1p1f1", "tasmax"): (
            ["2000-01-01", "2000-01-02", "2000-01-03"],
            [10.0, np.nan, 12.0],
        ),
        ("r1i1p1f1", "pr"): (
            ["2000-01-02", "2000-01-03"],
            [2.0, 3.0],
        ),
        ("r2i1p1f1", "tasmax"): (
            ["2000-01-01", "2000-01-02"],
            [20.0, 21.0],
        ),
    }
    if catalog_kind != "cadcat":
        assets = {
            key: value for key, value in assets.items() if key[0] == "r1i1p1f1"
        }
    records = []
    for (member_id, variable_id), (times, values) in assets.items():
        path = tmp_path / f"{member_id}-{variable_id}.nc"
        if spatial_grid:
            latitudes = [34.0, 35.0] if variable_id == "tasmax" else [34.0, 36.0]
            measurements = np.broadcast_to(
                np.asarray(values)[:, None, None], (len(times), 2, 2)
            )
            dataset = xr.Dataset(
                {variable_id: (("time", "lat", "lon"), measurements)},
                coords={
                    "time": pd.to_datetime(times),
                    "lat": latitudes,
                    "lon": [-120.0, -119.0],
                },
            )
        else:
            dataset = xr.Dataset(
                {variable_id: ("time", values)},
                coords={"time": pd.to_datetime(times)},
            )
        dataset.to_netcdf(path)
        record = {
            "activity_id": "WRF",
            "institution_id": "UCLA",
            "source_id": "WRF-test",
            "experiment_id": "historical",
            "table_id": "day",
            "grid_label": "d03",
            "variable_id": variable_id,
            "path": str(path),
            "format": "netcdf",
        }
        if catalog_kind == "cadcat":
            record["member_id"] = member_id
        elif catalog_kind == "renewables":
            record["installation"] = "wind"
        records.append(record)

    esmcat = {
        "esmcat_version": "0.0.1",
        "id": f"local-{catalog_kind}",
        "description": "Local catalog for testing intake-esm aggregation",
        "attributes": [
            {"column_name": name, "vocabulary": ""}
            for name in records[0]
            if name not in ("path", "format")
        ],
        "assets": {"column_name": "path", "format": "netcdf"},
        "aggregation_control": {
            "variable_column_name": "variable_id",
            **profile,
        },
    }
    return intake.open_esm_datastore(
        {"esmcat": esmcat, "df": pd.DataFrame(records)},
        progressbar=False,
        registry=registry,
    )


def test_aggregation_hides_source_coverage_but_unaggregated_assets_preserve_it(
    tmp_path,
):
    """Test merged NaNs are ambiguous while unaggregated assets retain provenance."""
    catalog = _make_local_catalog(tmp_path).search(variable_id=["tasmax", "pr"])

    aggregated = next(
        iter(catalog.to_dataset_dict(progressbar=False).values())
    )
    np.testing.assert_allclose(
        aggregated["tasmax"].sel(member_id="r1i1p1f1").values,
        [10.0, np.nan, 12.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        aggregated["pr"].sel(member_id="r1i1p1f1").values,
        [np.nan, 2.0, 3.0],
        equal_nan=True,
    )
    assert np.isnan(aggregated["pr"].sel(member_id="r2i1p1f1")).all()

    assets = catalog.to_dataset_dict(progressbar=False, aggregate=False)
    assets_by_member_variable = {
        (
            dataset.attrs["intake_esm_attrs:member_id"],
            dataset.attrs["intake_esm_attrs:variable_id"],
        ): dataset
        for dataset in assets.values()
    }
    assert set(assets_by_member_variable) == {
        ("r1i1p1f1", "tasmax"),
        ("r1i1p1f1", "pr"),
        ("r2i1p1f1", "tasmax"),
    }
    assert "pr" not in assets_by_member_variable[("r2i1p1f1", "tasmax")].data_vars
    np.testing.assert_array_equal(
        assets_by_member_variable[("r1i1p1f1", "tasmax")]["tasmax"]
        .isel(member_id=0)
        .values,
        [10.0, np.nan, 12.0],
    )
    assert assets_by_member_variable[("r1i1p1f1", "pr")].time.size == 2


@pytest.mark.parametrize("catalog_kind", ["renewables", "sup3rcc"])
def test_variable_union_catalogs_return_requested_fields(tmp_path, catalog_kind):
    """Test renewable and Sup3rCC-style unions combine distinct variable assets."""
    catalog = _make_local_catalog(tmp_path, catalog_kind=catalog_kind).search(
        variable_id=["tasmax", "pr"]
    )

    datasets = catalog.to_dataset_dict(progressbar=False)

    assert len(datasets) == 1
    dataset = next(iter(datasets.values()))
    assert set(dataset.data_vars) == {"tasmax", "pr"}
    np.testing.assert_allclose(dataset["tasmax"].values, [10.0, np.nan, 12.0])
    np.testing.assert_allclose(dataset["pr"].values, [np.nan, 2.0, 3.0])


def test_shared_derived_dependencies_are_separate_before_assembly(tmp_path):
    """Test local intake keeps shared derived inputs as separate raw assets."""
    from intake_esm import DerivedVariableRegistry

    registry = DerivedVariableRegistry()

    @registry.register(
        variable="temp_plus_precip",
        query={"variable_id": ["tasmax", "pr"]},
    )
    def add_fields(dataset):
        """Add the local temperature and precipitation test fields."""
        return dataset.assign(temp_plus_precip=dataset["tasmax"] + dataset["pr"])

    @registry.register(
        variable="temp_minus_precip",
        query={"variable_id": ["tasmax", "pr"]},
    )
    def subtract_fields(dataset):
        """Subtract the local precipitation test field from temperature."""
        return dataset.assign(temp_minus_precip=dataset["tasmax"] - dataset["pr"])

    query = _make_local_catalog(tmp_path, registry=registry).search(
        variable_id=["temp_plus_precip", "temp_minus_precip"]
    )
    assert set(query.df["variable_id"]) == {"tasmax", "pr"}

    raw_assets = query.to_dataset_dict(progressbar=False, aggregate=False)
    assert len(raw_assets) == 3
    assert all(
        set(dataset.data_vars).issubset({"tasmax", "pr"})
        for dataset in raw_assets.values()
    )


def test_aggregation_outer_aligns_mismatched_spatial_coordinates(tmp_path):
    """Test different grids are unioned and padded instead of rejected."""
    catalog = _make_local_catalog(tmp_path, spatial_grid=True).search(
        variable_id=["tasmax", "pr"]
    )

    dataset = next(
        iter(catalog.to_dataset_dict(progressbar=False).values())
    )

    np.testing.assert_array_equal(dataset["lat"].values, [34.0, 35.0, 36.0])
    assert dataset["tasmax"].sel(lat=36.0).isnull().all()
    assert dataset["pr"].sel(lat=35.0).isnull().all()


def test_aggregated_measurement_arrays_remain_lazy(tmp_path):
    """Test intake aggregation can retain Dask-backed measurement arrays."""
    catalog = _make_local_catalog(tmp_path).search(variable_id=["tasmax", "pr"])

    dataset = next(
        iter(
            catalog.to_dataset_dict(
                progressbar=False,
                xarray_open_kwargs={"chunks": {}},
            ).values()
        )
    )

    assert all(
        hasattr(dataset[name].data, "__dask_graph__")
        for name in ("tasmax", "pr")
    )
