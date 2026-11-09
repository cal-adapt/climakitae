"""Characterize pinned intake-esm aggregation for multi-variable queries."""

import numpy as np
import pandas as pd
import xarray as xr
import intake


def _make_local_catalog(tmp_path):
    """Create a small CADCAT-shaped catalog with incomplete member coverage."""
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
    records = []
    for (member_id, variable_id), (times, values) in assets.items():
        path = tmp_path / f"{member_id}-{variable_id}.nc"
        xr.Dataset(
            {variable_id: ("time", values)},
            coords={"time": pd.to_datetime(times)},
        ).to_netcdf(path)
        records.append(
            {
                "activity_id": "WRF",
                "experiment_id": "historical",
                "member_id": member_id,
                "variable_id": variable_id,
                "path": str(path),
                "format": "netcdf",
            }
        )

    esmcat = {
        "esmcat_version": "0.0.1",
        "id": "local-cadcat",
        "description": "Local catalog for testing intake-esm aggregation",
        "attributes": [
            {"column_name": name, "vocabulary": ""}
            for name in (
                "activity_id",
                "experiment_id",
                "member_id",
                "variable_id",
            )
        ],
        "assets": {"column_name": "path", "format": "netcdf"},
        "aggregation_control": {
            "variable_column_name": "variable_id",
            "groupby_attrs": ["activity_id", "experiment_id"],
            "aggregations": [
                {"type": "join_new", "attribute_name": "member_id"},
                {"type": "union", "attribute_name": "variable_id"},
            ],
        },
    }
    return intake.open_esm_datastore(
        {"esmcat": esmcat, "df": pd.DataFrame(records)},
        progressbar=False,
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
