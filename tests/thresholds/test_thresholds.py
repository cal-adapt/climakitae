import pytest
import xarray as xr

from climakitae.core.data_interface import DataParameters
from climakitae.explore import thresholds


@pytest.mark.advanced
class TestThreshold:
    @pytest.fixture
    def selections(self) -> DataParameters:
        # Create a DataParameters object
        test_selections = DataParameters()
        test_selections.append_historical = False
        test_selections.area_average = "Yes"
        test_selections.resolution = "45 km"
        test_selections.scenario_ssp = ["SSP 3-7.0"]
        test_selections.scenario_historical = []
        test_selections.time_slice = (2020, 2021)
        test_selections.timescale = "monthly"
        test_selections.variable = "Air Temperature at 2m"
        test_selections.approach = "Time"
        test_selections.simulation = ["TaiESM1"]

        # Location defaults
        test_selections.area_subset = "CA counties"
        test_selections.cached_area = ["Los Angeles County"]

        return test_selections

    @pytest.fixture
    def my_test_data(self, selections: DataParameters) -> xr.DataArray:
        data = thresholds.get_threshold_data(selections)
        return data

    def test_data_attributes(self, my_test_data: xr.DataArray):
        expected = [
            "t2",
            "K",
            "Gridded",
            "45 km",
            "monthly",
            ["Los Angeles County"],
            "Time",
            "Dynamical",
        ]
        actual = [
            my_test_data.variable_id,
            my_test_data.units,
            my_test_data.data_type,
            my_test_data.resolution,
            my_test_data.frequency,
            my_test_data.location_subset,
            my_test_data.approach,
            my_test_data.downscaling_method,
        ]
        assert actual == expected
        assert isinstance(my_test_data, xr.core.dataarray.DataArray)
        assert my_test_data.shape == (1, 1, 24)
