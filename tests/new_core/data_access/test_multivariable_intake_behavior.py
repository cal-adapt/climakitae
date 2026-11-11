"""Characterize pinned intake-esm aggregation for multi-variable queries."""

import json
from unittest.mock import MagicMock

import dask
import dask.array as da
import intake
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import (
    CATALOG_CADCAT,
    CATALOG_REN_ENERGY_GEN,
    CATALOG_SUP3RCC,
)


@pytest.fixture(autouse=True)
def _synchronous_dask():
    """Open local assets synchronously; threaded netCDF reads segfault locally."""
    with dask.config.set(scheduler="synchronous"):
        yield


def _make_local_catalog(
    tmp_path,
    catalog_kind="cadcat",
    registry=None,
    spatial_grid=False,
    mismatched_spatial_grid=False,
    mismatched_calendar=False,
    include_ssp=False,
    no_time_overlap=False,
    only_incomplete_members=False,
    include_wind_components=False,
    include_spatial_metadata=False,
    asset_format="zarr",
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
            "aggregations": [{"type": "union", "attribute_name": "variable_id"}],
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
            "aggregations": [{"type": "union", "attribute_name": "variable_id"}],
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
    if no_time_overlap:
        assets[("r1i1p1f1", "pr")] = (
            ["2000-01-04", "2000-01-05"],
            [2.0, 3.0],
        )
        assets[("r2i1p1f1", "pr")] = (
            ["2000-01-02", "2000-01-03"],
            [4.0, 5.0],
        )
    if only_incomplete_members:
        assets.pop(("r1i1p1f1", "pr"))
    if include_wind_components:
        assets[("r1i1p1f1", "u10")] = (
            ["2000-01-01", "2000-01-02", "2000-01-03"],
            [3.0, 0.0, 4.0],
        )
        assets[("r1i1p1f1", "v10")] = (
            ["2000-01-01", "2000-01-02", "2000-01-03"],
            [4.0, 5.0, 3.0],
        )
    if catalog_kind != "cadcat":
        assets = {key: value for key, value in assets.items() if key[0] == "r1i1p1f1"}
    asset_rows = [
        ("historical", member_id, variable_id, times, values)
        for (member_id, variable_id), (times, values) in assets.items()
    ]
    if include_ssp:
        asset_rows.extend(
            (
                "ssp245",
                member_id,
                variable_id,
                [time.replace("2000", "2040") for time in times],
                values,
            )
            for (member_id, variable_id), (times, values) in assets.items()
            if member_id == "r1i1p1f1"
        )
    records = []
    for experiment_id, member_id, variable_id, times, values in asset_rows:
        suffix = "zarr" if asset_format == "zarr" else "nc"
        path = tmp_path / f"{experiment_id}-{member_id}-{variable_id}.{suffix}"
        if spatial_grid:
            latitudes = (
                [34.0, 36.0]
                if mismatched_spatial_grid and variable_id == "pr"
                else [34.0, 35.0]
            )
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
        if include_spatial_metadata:
            units = {
                "tasmax": "degC",
                "pr": "mm day-1",
                "u10": "m s-1",
                "v10": "m s-1",
            }
            dataset[variable_id].attrs.update(
                {"units": units[variable_id], "grid_mapping": "crs"}
            )
            dataset = dataset.assign_coords(
                crs=xr.DataArray(0, attrs={"crs_wkt": "EPSG:4326"})
            )
        if mismatched_calendar and variable_id == "pr":
            dataset["time"].encoding["calendar"] = "noleap"
        if asset_format == "zarr":
            dataset.to_zarr(path)
            # Zarr only accepts storage_options for remote-style fsspec URIs.
            path = f"local://{path}"
        else:
            dataset.to_netcdf(path, engine="scipy")
        record = {
            "activity_id": "WRF",
            "institution_id": "UCLA",
            "source_id": "WRF-test",
            "experiment_id": experiment_id,
            "table_id": "day",
            "grid_label": "d03",
            "variable_id": variable_id,
            "path": str(path),
            "format": asset_format,
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
        "assets": {"column_name": "path", "format": asset_format},
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


def _make_local_hdp_catalog(tmp_path):
    """Create HDP-shaped station assets with different observation fields."""
    stations = {
        "station-1": {
            "tas": [280.0, np.nan],
            "pr": [0.0, 1.0],
        },
        "station-2": {
            "tas": [282.0, 283.0],
        },
    }
    records = []
    for station_id, fields in stations.items():
        path = tmp_path / f"{station_id}.nc"
        xr.Dataset(
            {name: (("station", "time"), [values]) for name, values in fields.items()},
            coords={
                "station": [station_id],
                "time": pd.to_datetime(["2000-01-01", "2000-01-02"]),
            },
        ).to_netcdf(path, engine="scipy")
        records.append(
            {
                "network_id": "ASOSAWOS",
                "station_id": station_id,
                "path": str(path),
                "format": "netcdf",
            }
        )

    esmcat = {
        "esmcat_version": "0.0.1",
        "id": "local-hdp",
        "description": "Local catalog for testing HDP station assets",
        "attributes": [
            {"column_name": name, "vocabulary": ""}
            for name in ("network_id", "station_id")
        ],
        "assets": {"column_name": "path", "format": "netcdf"},
        "aggregation_control": {
            "variable_column_name": "station_id",
            "groupby_attrs": ["network_id", "station_id"],
            "aggregations": [{"type": "union", "attribute_name": "station_id"}],
        },
    }
    return intake.open_esm_datastore(
        {"esmcat": esmcat, "df": pd.DataFrame(records)},
        progressbar=False,
    )


def test_hdp_observation_fields_are_discovered_from_station_assets(tmp_path):
    """Test HDP option discovery returns time-varying fields from station assets."""
    from climakitae.new_core.data_access.data_access import CATALOG_HDP, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_HDP] = _make_local_hdp_catalog(tmp_path)

    fields = data_catalog.get_hdp_observation_fields({"network_id": "ASOSAWOS"})

    assert fields == ["pr", "tas"]


def test_hdp_query_without_variable_selection_keeps_existing_fields(tmp_path):
    """Test HDP queries without variable selection retain all station fields."""
    from climakitae.new_core.data_access.data_access import CATALOG_HDP, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    catalog = MagicMock()
    search_result = MagicMock()
    search_result.to_dataset_dict.return_value = {
        "station-1": xr.Dataset(
            {
                "tas": (("station", "time"), [[280.0, np.nan]]),
                "pr": (("station", "time"), [[0.0, 1.0]]),
            },
            coords={
                "station": ["station-1"],
                "time": pd.to_datetime(["2000-01-01", "2000-01-02"]),
            },
        )
    }
    catalog.search.return_value = search_result
    data_catalog[CATALOG_HDP] = catalog

    result = data_catalog.get_data({"network_id": "ASOSAWOS"}, catalog_key=CATALOG_HDP)

    search_result.to_dataset_dict.assert_called_once()
    assert search_result.to_dataset_dict.call_args.kwargs["aggregate"] is True
    dataset = next(iter(result.values()))
    assert set(dataset.data_vars) == {"tas", "pr"}
    assert dataset.station_id.item() == "station-1"


def test_hdp_field_selection_excludes_incomplete_stations_with_diagnostics(tmp_path):
    """Test HDP returns complete station fields and records excluded stations."""
    from climakitae.new_core.data_access.data_access import CATALOG_HDP, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_HDP] = _make_local_hdp_catalog(tmp_path)
    query = {
        "network_id": "ASOSAWOS",
        "variable_id": ["tas", "pr"],
        "_hdp_variable_query": True,
        "_hdp_requested_fields": ["tas", "pr"],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_HDP)

    assert len(result) == 1
    dataset = next(iter(result.values()))
    assert list(dataset.data_vars) == ["tas", "pr"]
    assert dataset.station_id.values.tolist() == ["station-1"]
    assert np.isnan(dataset["tas"].values).any()
    diagnostics = json.loads(dataset.attrs["multi_variable_diagnostics"])
    assert diagnostics["stations"]["station-2"] == {
        "missing_fields": ["pr"],
        "status": "excluded",
    }


def test_hdp_validator_plan_reaches_station_field_retrieval(tmp_path):
    """Test HDP validation metadata drives station-field retrieval end to end."""
    from climakitae.new_core.data_access.data_access import CATALOG_HDP, DataCatalog
    from climakitae.new_core.param_validation.hdp_param_validator import HDPValidator

    hdp_catalog = _make_local_hdp_catalog(tmp_path)
    validator = HDPValidator(MagicMock(hdp=hdp_catalog))
    query = {"network_id": "ASOSAWOS", "variable_id": ["tas", "pr"]}

    validated_query = validator.is_valid_query(query)
    assert validated_query is not None
    assert validated_query["_hdp_requested_fields"] == ["tas", "pr"]

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_HDP] = hdp_catalog
    result = data_catalog.get_data(validated_query, catalog_key=CATALOG_HDP)

    assert len(result) == 1
    assert list(next(iter(result.values())).data_vars) == ["tas", "pr"]


def test_hdp_selection_fails_when_no_station_has_requested_fields(tmp_path):
    """Test HDP selection fails with diagnostics when all stations are incomplete."""
    from climakitae.new_core.data_access.data_access import CATALOG_HDP, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_HDP] = _make_local_hdp_catalog(tmp_path)
    query = {
        "network_id": "ASOSAWOS",
        "variable_id": "missing_field",
        "_hdp_variable_query": True,
        "_hdp_requested_fields": ["missing_field"],
    }

    with pytest.raises(ValueError, match="No HDP stations contain all requested"):
        data_catalog.get_data(query, catalog_key=CATALOG_HDP)


def test_hdp_single_field_selection_preserves_station_compatibility(tmp_path):
    """Test explicit single-field HDP queries still return per-station datasets."""
    from climakitae.new_core.data_access.data_access import CATALOG_HDP, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_HDP] = _make_local_hdp_catalog(tmp_path)
    query = {
        "network_id": "ASOSAWOS",
        "variable_id": "tas",
        "_hdp_variable_query": True,
        "_hdp_requested_fields": ["tas"],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_HDP)

    assert len(result) == 2
    assert all(list(dataset.data_vars) == ["tas"] for dataset in result.values())
    assert {dataset.station_id.item() for dataset in result.values()} == {
        "station-1",
        "station-2",
    }


def test_hdp_station_without_shared_timestamps_fails_entire_query(tmp_path):
    """Test a disjoint complete station fails despite another aligned station."""
    from climakitae.new_core.data_access.data_access import CATALOG_HDP, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    early_times = pd.to_datetime(["2000-01-01", "2000-01-02"])
    late_times = pd.to_datetime(["2000-01-03", "2000-01-04"])

    def station_field(station_id, field, times, values):
        return xr.Dataset(
            {field: (("station", "time"), [values])},
            coords={"station": [station_id], "time": times},
            attrs={"intake_esm_attrs:station_id": station_id},
        )

    assets = {
        "station-1-tas": station_field("station-1", "tas", early_times, [280.0, 281.0]),
        "station-1-pr": station_field("station-1", "pr", late_times, [0.0, 1.0]),
        "station-2-tas": station_field("station-2", "tas", early_times, [282.0, 283.0]),
        "station-2-pr": station_field("station-2", "pr", early_times, [2.0, 3.0]),
    }
    query = {
        "network_id": "ASOSAWOS",
        "variable_id": ["tas", "pr"],
        "_hdp_variable_query": True,
        "_hdp_requested_fields": ["tas", "pr"],
    }

    with pytest.raises(ValueError, match="no timestamps shared"):
        data_catalog._assemble_hdp_station_fields(assets, query)


def test_hdp_fields_are_trimmed_to_shared_timestamps_with_diagnostics():
    """Test HDP fields use the station-level time intersection."""
    from climakitae.new_core.data_access.data_access import DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    station_id = "station-1"
    assets = {
        "station-1-tas": xr.Dataset(
            {"tas": (("station", "time"), [[280.0, 281.0]])},
            coords={
                "station": [station_id],
                "time": pd.to_datetime(["2000-01-01", "2000-01-02"]),
            },
            attrs={"intake_esm_attrs:station_id": station_id},
        ),
        "station-1-pr": xr.Dataset(
            {"pr": (("station", "time"), [[2.0, 3.0]])},
            coords={
                "station": [station_id],
                "time": pd.to_datetime(["2000-01-02", "2000-01-03"]),
            },
            attrs={"intake_esm_attrs:station_id": station_id},
        ),
    }

    result = data_catalog._assemble_hdp_station_fields(
        assets, {"_hdp_requested_fields": ["tas", "pr"]}
    )

    dataset = next(iter(result.values()))
    assert list(dataset.data_vars) == ["tas", "pr"]
    np.testing.assert_array_equal(dataset.time.values, [np.datetime64("2000-01-02")])
    diagnostics = json.loads(dataset.attrs["multi_variable_diagnostics"])
    assert diagnostics["stations"][station_id]["trimmed_timestamp_counts"] == {
        "pr": 1,
        "tas": 1,
    }


def test_data_catalog_excludes_members_missing_requested_variables(tmp_path, caplog):
    """Test retrieval keeps only complete members on shared, unpadded timestamps."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    caplog.set_level("INFO")

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(tmp_path, spatial_grid=True)
    query = {
        "variable_id": ["tasmax", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax", "pr"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_derived_variables": [],
        "_source_variables": [],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    assert len(result) == 1
    dataset = next(iter(result.values()))
    assert set(dataset.data_vars) == {"tasmax", "pr"}
    assert dataset.member_id.values.tolist() == ["r1i1p1f1"]
    np.testing.assert_array_equal(
        dataset.time.values,
        [
            np.datetime64("2000-01-02"),
            np.datetime64("2000-01-03"),
        ],
    )
    assert np.isnan(dataset["tasmax"].sel(member_id="r1i1p1f1").isel(time=0)).all()
    np.testing.assert_array_equal(
        dataset["pr"].sel(member_id="r1i1p1f1").isel(lat=0, lon=0).values,
        [2.0, 3.0],
    )
    assert not dataset["pr"].isnull().any()
    assert "'member_id': 'r2i1p1f1'" in caplog.text
    assert "missing requested catalog variables ['pr']" in caplog.text
    assert "shortened" in caplog.text and "'time': '3 -> 2'" in caplog.text


def test_data_catalog_single_variable_query_keeps_aggregated_path(tmp_path):
    """Test single-variable retrieval keeps every member on the outer-joined time."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(tmp_path, spatial_grid=True)

    result = data_catalog.get_data({"variable_id": "tasmax"}, CATALOG_CADCAT)

    dataset = next(iter(result.values()))
    assert list(dataset.data_vars) == ["tasmax"]
    assert dataset.member_id.values.tolist() == ["r1i1p1f1", "r2i1p1f1"]
    assert dataset.sizes["time"] == 3


def test_data_catalog_rejects_members_without_shared_timestamps(tmp_path):
    """Test complete members whose timestamps never all overlap fail the query."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(
        tmp_path, spatial_grid=True, no_time_overlap=True
    )
    query = {
        "variable_id": ["tasmax", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax", "pr"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_derived_variables": [],
        "_source_variables": [],
    }

    with pytest.raises(ValueError, match="share no common"):
        data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)


def test_data_catalog_intersects_mismatched_spatial_coordinates_with_warning(
    tmp_path, caplog
):
    """Test different spatial grids are intersected, not padded, and warned about."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(
        tmp_path, spatial_grid=True, mismatched_spatial_grid=True
    )
    query = {
        "variable_id": ["tasmax", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax", "pr"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_derived_variables": [],
        "_source_variables": [],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    dataset = next(iter(result.values()))
    np.testing.assert_array_equal(dataset["lat"].values, [34.0])
    assert not dataset["pr"].isnull().any()
    assert "'lat': '2 -> 1'" in caplog.text


def test_data_catalog_rejects_incompatible_calendars(tmp_path):
    """Test fields on different CF calendars share no timestamps and fail."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(
        tmp_path, spatial_grid=True, mismatched_calendar=True
    )
    query = {
        "variable_id": ["tasmax", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax", "pr"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_derived_variables": [],
        "_source_variables": [],
    }

    with pytest.raises(ValueError, match="share no common"):
        data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)


def test_data_catalog_preserves_lazy_multi_variable_assets(tmp_path):
    """Test retrieval leaves measurements lazy and executes no Dask tasks."""
    from dask.callbacks import Callback

    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(tmp_path, spatial_grid=True)
    query = {
        "variable_id": ["tasmax", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax", "pr"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_derived_variables": [],
        "_source_variables": [],
    }

    executed_tasks = []
    with Callback(posttask=lambda key, *args: executed_tasks.append(key)):
        result = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    dataset = next(iter(result.values()))
    assert isinstance(dataset["tasmax"].data, da.Array)
    assert isinstance(dataset["pr"].data, da.Array)
    assert all(str(task).startswith("_open_dataset-") for task in executed_tasks)


@pytest.mark.parametrize(
    "catalog_kind,catalog_key",
    [
        ("cadcat", CATALOG_CADCAT),
        ("renewables", CATALOG_REN_ENERGY_GEN),
        ("sup3rcc", CATALOG_SUP3RCC),
    ],
)
def test_data_catalog_assembles_each_supported_variable_catalog(
    tmp_path, catalog_kind, catalog_key
):
    """Test all supported catalogs return jointly covered requested fields."""
    from climakitae.new_core.data_access.data_access import DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    catalog_path = tmp_path / catalog_kind
    catalog_path.mkdir()
    data_catalog[catalog_key] = _make_local_catalog(
        catalog_path, catalog_kind=catalog_kind, spatial_grid=True
    )
    query = {
        "variable_id": ["tasmax", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax", "pr"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_derived_variables": [],
        "_source_variables": [],
    }

    result = data_catalog.get_data(query, catalog_key=catalog_key)

    assert len(result) == 1
    dataset = next(iter(result.values()))
    assert set(dataset.data_vars) == {"tasmax", "pr"}
    np.testing.assert_array_equal(
        dataset.time.values,
        [
            np.datetime64("2000-01-02"),
            np.datetime64("2000-01-03"),
        ],
    )


def test_data_catalog_keeps_historical_and_ssp_multi_variable_groups_separate(
    tmp_path,
):
    """Test distinct experiments remain separate during source assembly."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(
        tmp_path, spatial_grid=True, include_ssp=True
    )
    query = {
        "variable_id": ["tasmax", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax", "pr"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_derived_variables": [],
        "_source_variables": [],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    datasets = {
        dataset.attrs["intake_esm_attrs:experiment_id"]: dataset
        for dataset in result.values()
    }
    assert set(datasets) == {"historical", "ssp245"}
    assert str(datasets["historical"].time.values[0]).startswith("2000")
    assert str(datasets["ssp245"].time.values[0]).startswith("2040")


def test_data_catalog_fails_when_no_member_has_complete_variable_coverage(
    tmp_path, caplog
):
    """Test no complete member fails and logs each excluded member."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(
        tmp_path, only_incomplete_members=True
    )
    query = {
        "variable_id": ["tasmax", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax", "pr"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_derived_variables": [],
        "_source_variables": [],
    }

    caplog.set_level("INFO")

    with pytest.raises(ValueError, match="No simulation/member contains all requested"):
        data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    assert "missing requested catalog variables ['pr']" in caplog.text


def test_data_catalog_computes_derived_and_native_variables_in_request_order(
    tmp_path,
):
    """Test mixed native and derived fields expose only requested outputs."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(
        tmp_path, include_wind_components=True
    )
    query = {
        "variable_id": ["wind_speed_10m", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["wind_speed_10m", "pr"],
        "_derived_variables": ["wind_speed_10m"],
        "_catalog_variable_ids": ["u10", "v10", "pr"],
        "_source_variables": ["u10", "v10"],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    dataset = next(iter(result.values()))
    assert list(dataset.data_vars) == ["wind_speed_10m", "pr"]
    assert isinstance(dataset["wind_speed_10m"].data, da.Array)
    assert dataset["wind_speed_10m"].attrs["units"] == "m/s"
    np.testing.assert_allclose(
        dataset["wind_speed_10m"].isel(member_id=0).values, [5.0, 5.0]
    )
    np.testing.assert_allclose(dataset["pr"].isel(member_id=0).values, [2.0, 3.0])


def test_data_catalog_preserves_independent_units_and_crs_for_outputs(tmp_path):
    """Keep each output's units and CRS metadata through mixed assembly."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(
        tmp_path,
        spatial_grid=True,
        include_wind_components=True,
        include_spatial_metadata=True,
    )
    query = {
        "variable_id": ["wind_speed_10m", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["wind_speed_10m", "pr"],
        "_derived_variables": ["wind_speed_10m"],
        "_catalog_variable_ids": ["u10", "v10", "pr"],
        "_source_variables": ["u10", "v10"],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    dataset = next(iter(result.values()))
    assert dataset["wind_speed_10m"].attrs["units"] == "m/s"
    assert dataset["pr"].attrs["units"] == "mm day-1"
    assert dataset["wind_speed_10m"].attrs["grid_mapping"] == "crs"
    assert dataset["pr"].attrs["grid_mapping"] == "crs"
    assert dataset.coords["crs"].attrs["crs_wkt"] == "EPSG:4326"


def test_data_catalog_computes_shared_dependencies_without_dropping_requested_sources(
    tmp_path,
):
    """Test shared derived inputs and explicitly requested sources are retained."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(
        tmp_path, include_wind_components=True
    )
    query = {
        "variable_id": [
            "wind_direction_10m",
            "u10",
            "wind_speed_10m",
            "pr",
        ],
        "_multi_variable_query": True,
        "_requested_variables": [
            "wind_direction_10m",
            "u10",
            "wind_speed_10m",
            "pr",
        ],
        "_derived_variables": ["wind_direction_10m", "wind_speed_10m"],
        "_catalog_variable_ids": ["u10", "v10", "pr"],
        "_source_variables": ["u10", "v10"],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    dataset = next(iter(result.values()))
    assert list(dataset.data_vars) == [
        "wind_direction_10m",
        "u10",
        "wind_speed_10m",
        "pr",
    ]
    np.testing.assert_allclose(
        dataset["wind_direction_10m"].isel(member_id=0).values,
        [180.0, 233.13010235],
    )
    np.testing.assert_allclose(dataset["u10"].isel(member_id=0).values, [0.0, 4.0])
    np.testing.assert_allclose(
        dataset["wind_speed_10m"].isel(member_id=0).values, [5.0, 5.0]
    )


def test_data_catalog_preserves_dependencies_when_derived_registration_keeps_them(
    tmp_path, monkeypatch
):
    """Test drop_dependencies=False derived registrations retain their inputs."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog
    from climakitae.new_core.derived_variables import registry

    def add_one_to_tasmax(dataset):
        result = dataset.copy()
        result["tasmax_plus_one"] = result["tasmax"] + 1
        return result

    monkeypatch.setitem(
        registry._DERIVED_METADATA,
        "tasmax_plus_one",
        registry.DerivedVariableInfo(
            name="tasmax_plus_one",
            depends_on=["tasmax"],
            description="Test derived field",
            units="K",
            func=add_one_to_tasmax,
            source="user",
            drop_dependencies=False,
        ),
    )
    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(tmp_path, spatial_grid=True)
    query = {
        "variable_id": ["tasmax_plus_one", "tasmax"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax_plus_one", "tasmax"],
        "_derived_variables": ["tasmax_plus_one"],
        "_catalog_variable_ids": ["tasmax"],
        "_source_variables": ["tasmax"],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    dataset = next(iter(result.values()))
    assert list(dataset.data_vars) == ["tasmax_plus_one", "tasmax"]
    np.testing.assert_allclose(
        dataset["tasmax_plus_one"].isel(member_id=0, lat=0, lon=0).values,
        [11.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        dataset["tasmax"].isel(member_id=0, lat=0, lon=0).values,
        [10.0, np.nan],
        equal_nan=True,
    )


def test_data_catalog_rejects_custom_derived_alias_in_multi_variable_query(
    tmp_path, monkeypatch
):
    """Test custom functions must return the registered output field name."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog
    from climakitae.new_core.derived_variables import registry

    def create_alias(dataset):
        result = dataset.copy()
        result["alias"] = result["tasmax"] + 1
        return result

    monkeypatch.setitem(
        registry._DERIVED_METADATA,
        "requested_derived",
        registry.DerivedVariableInfo(
            name="requested_derived",
            depends_on=["tasmax"],
            description="Test derived field",
            units="K",
            func=create_alias,
            source="user",
            drop_dependencies=False,
        ),
    )
    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(tmp_path, spatial_grid=True)
    query = {
        "variable_id": ["requested_derived", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["requested_derived", "pr"],
        "_derived_variables": ["requested_derived"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_source_variables": ["tasmax"],
    }

    with pytest.raises(
        ValueError,
        match=r"missing requested output variables \['requested_derived'\]",
    ):
        data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)


def test_scalar_custom_derived_query_remains_compatible(tmp_path):
    """Keep scalar custom derived retrieval on the existing intake path."""
    from intake_esm import DerivedVariableRegistry

    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog

    registry = DerivedVariableRegistry()

    @registry.register(
        variable="tasmax_plus_one",
        query={"variable_id": ["tasmax"]},
    )
    def add_one(dataset):
        """Create a test derived field from the native temperature field."""
        return dataset.assign(tasmax_plus_one=dataset["tasmax"] + 1)

    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    # Scalar retrieval aggregates Zarr assets with consolidated=None, as in production.
    data_catalog[CATALOG_CADCAT] = _make_local_catalog(
        tmp_path, registry=registry, asset_format="zarr"
    )
    query = {
        "variable_id": "tasmax_plus_one",
        "_derived_variable": "tasmax_plus_one",
        "_source_variables": ["tasmax"],
    }

    result = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    dataset = next(iter(result.values()))
    assert "tasmax_plus_one" in dataset.data_vars
    assert "_multi_variable_query" not in query


def test_warming_level_keeps_both_variables_across_historical_and_ssp(tmp_path, caplog):
    """Test warming levels over multi-variable data keep both fields aligned."""
    from climakitae.new_core.data_access.data_access import CATALOG_CADCAT, DataCatalog
    from climakitae.new_core.processors.warming_level import WarmingLevel

    periods = {"historical": (1981, 2014), "ssp245": (2015, 2099)}
    records = []
    for experiment_id, (start, end) in periods.items():
        times = pd.date_range(f"{start}-01-01", f"{end}-01-01", freq="YS")
        for member_id in ("r1i1p1f1", "r2i1p1f1"):
            for variable_id, units in (("tasmax", "K"), ("pr", "mm/day")):
                if (experiment_id, member_id, variable_id) == (
                    "historical",
                    "r2i1p1f1",
                    "pr",
                ):
                    continue
                path = tmp_path / f"{experiment_id}-{member_id}-{variable_id}.zarr"
                dataset = xr.Dataset(
                    {variable_id: ("time", np.arange(len(times), dtype=float))},
                    coords={"time": times},
                )
                dataset[variable_id].attrs["units"] = units
                dataset.to_zarr(path)
                records.append(
                    {
                        "activity_id": "WRF",
                        "institution_id": "UCLA",
                        "source_id": "WRF-test",
                        "experiment_id": experiment_id,
                        "table_id": "mon",
                        "grid_label": "d03",
                        "member_id": member_id,
                        "variable_id": variable_id,
                        "path": f"local://{path}",
                        "format": "zarr",
                    }
                )
    esmcat = {
        "esmcat_version": "0.0.1",
        "id": "local-wl",
        "description": "Local multi-variable catalog for warming levels",
        "attributes": [
            {"column_name": name, "vocabulary": ""}
            for name in records[0]
            if name not in ("path", "format")
        ],
        "assets": {"column_name": "path", "format": "zarr"},
        "aggregation_control": {
            "variable_column_name": "variable_id",
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
    }
    data_catalog = dict.__new__(DataCatalog)
    dict.__init__(data_catalog)
    data_catalog[CATALOG_CADCAT] = intake.open_esm_datastore(
        {"esmcat": esmcat, "df": pd.DataFrame(records)}, progressbar=False
    )
    query = {
        "variable_id": ["tasmax", "pr"],
        "_multi_variable_query": True,
        "_requested_variables": ["tasmax", "pr"],
        "_catalog_variable_ids": ["tasmax", "pr"],
        "_derived_variables": [],
        "_source_variables": [],
    }
    caplog.set_level("INFO")
    retrieved = data_catalog.get_data(query, catalog_key=CATALOG_CADCAT)

    processor = WarmingLevel(value={"warming_levels": [2.0]})
    processor.warming_level_times = pd.DataFrame(
        [
            ["WRF-test", member_id, "ssp245", "2020-01-01"]
            for member_id in ("r1i1p1f1", "r2i1p1f1")
        ],
        columns=["GCM", "run", "scenario", "2.0"],
    ).set_index(["GCM", "run", "scenario"])
    warmed = processor.execute(retrieved, {"activity_id": "WRF"})

    assert "'member_id': 'r2i1p1f1'" in caplog.text
    assert set(warmed) == {
        "WRF.UCLA.WRF-test.ssp245.mon.d03.r1i1p1f1",
        "WRF.UCLA.WRF-test.ssp245.mon.d03.r2i1p1f1",
    }
    complete = warmed["WRF.UCLA.WRF-test.ssp245.mon.d03.r1i1p1f1"]
    assert list(complete.data_vars) == ["tasmax", "pr"]
    assert complete["tasmax"].attrs["units"] == "K"
    assert complete["pr"].attrs["units"] == "mm/day"
    assert complete.sizes["time_delta"] == 30
    for dataset in warmed.values():
        assert dataset["tasmax"].dims == dataset["pr"].dims
        assert not dataset["pr"].isnull().any()
