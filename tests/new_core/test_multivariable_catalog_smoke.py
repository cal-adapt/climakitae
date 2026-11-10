"""Network-backed smoke tests for multi-variable catalog queries."""

import xarray as xr
import pytest

from climakitae.new_core.user_interface import ClimateData


@pytest.mark.advanced
@pytest.mark.parametrize(
    "selection,variables,time_slice",
    [
        pytest.param(
            {
                "catalog": "cadcat",
                "activity_id": "WRF",
                "institution_id": "UCLA",
                "source_id": "EC-Earth3",
                "experiment_id": "historical",
                "table_id": "day",
                "grid_label": "d03",
            },
            ["t2max", "t2min"],
            ("2000-01-01", "2000-01-02"),
            id="cadcat-wrf",
        ),
        pytest.param(
            {
                "catalog": "cadcat",
                "activity_id": "LOCA2",
                "institution_id": "UCSD",
                "source_id": "ACCESS-CM2",
                "experiment_id": "historical",
                "table_id": "day",
                "grid_label": "d03",
            },
            ["tasmax", "pr"],
            ("2000-01-01", "2000-01-02"),
            id="cadcat-loca2",
        ),
        pytest.param(
            {
                "catalog": "renewable energy generation",
                "installation": "pv_utility",
                "activity_id": "WRF",
                "institution_id": "ERA",
                "source_id": "EC-Earth3",
                "experiment_id": "historical",
                "table_id": "1hr",
                "grid_label": "d02",
            },
            ["cf", "gen"],
            ("2000-01-01", "2000-01-02"),
            id="renewable-generation",
        ),
        pytest.param(
            {
                "catalog": "sup3rcc",
                "activity_id": "Sup3rCC",
                "source_id": "EC-Earth3-CC",
                "experiment_id": "historical",
                "table_id": "1hr",
                "grid_label": "conus4km",
            },
            ["t2", "ws10"],
            ("2000-01-01", "2000-01-02"),
            id="sup3rcc",
        ),
    ],
)
def test_gridded_catalog_multi_variable_query(selection, variables, time_slice):
    """Test public queries return lazy data for selected catalog variables."""
    climate_data = ClimateData(verbosity=-1)
    for parameter, value in selection.items():
        getattr(climate_data, parameter)(value)
    climate_data.variable_id(variables).processes({"time_slice": time_slice})

    result = climate_data.get()

    assert isinstance(result, xr.Dataset)
    assert set(variables).issubset(result.data_vars)
    assert all(
        hasattr(result[variable].data, "__dask_graph__") for variable in variables
    )
