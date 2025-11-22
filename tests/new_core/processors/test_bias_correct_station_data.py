"""
Unit tests for climakitae/new_core/processors/bias_adjust_model_to_station.py

This module contains comprehensive unit tests for the BiasCorrectStationData
processor that performs bias correction of climate model data to weather station
locations using Quantile Delta Mapping (QDM).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import xarray as xr
import pytest
import numpy as np

from climakitae.new_core.processors.bias_adjust_model_to_station import (
    BiasCorrectStationData,
)


class TestBiasCorrectStationDataInit:
    """Tests for BiasCorrectStationData initialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ProcClass = BiasCorrectStationData

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        cfg = {"stations": ["Sacramento (KSAC)"]}
        proc = self.ProcClass(cfg)
        assert proc.stations == ["Sacramento (KSAC)"]
        # defaults
        assert proc.historical_slice == (1980, 2014)
        assert proc.window == 90
        assert proc.nquantiles == 20
        assert proc.group == "time.dayofyear"
        assert proc.kind == "+"
        assert proc.name == "bias_adjust_model_to_station"
        # Processor declares it needs a catalog to run
        assert getattr(proc, "needs_catalog", False) is True


class TestBiasCorrectStationDataPreprocessing:
    """Tests for HadISD preprocessing (_preprocess_hadisd)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ProcClass = BiasCorrectStationData

    def test_preprocess_hadisd_successful(self):
        """Test HadISD preprocessing with valid input."""
        import geopandas as gpd
        from shapely.geometry import Point

        proc = self.ProcClass({"stations": ["KSAC"]})

        # Build a minimal HadISD-like dataset
        times = pd.date_range("2010-01-01", periods=2)
        ds = xr.Dataset(
            {
                "tas": ("time", [10.0, 11.0]),
                "latitude": ([], 38.5),
                "longitude": ([], -121.5),
                "elevation": ([], 25.0),
            },
            coords={"time": times},
        )
        # Add expected encoding to extract station id
        ds.encoding["source"] = "s3://somepath/HadISD_1234.zarr"
        ds.elevation.attrs["units"] = "m"

        # Create station metadata GeoDataFrame
        stations_gdf = gpd.GeoDataFrame(
            {
                "station id": [1234],
                "station": ["KSAC"],
                "geometry": [Point(-121.5, 38.5)],
            }
        )

        out = proc._preprocess_hadisd(ds, stations_gdf)

        # After preprocessing, station variable name should be present
        assert "KSAC" in out.data_vars
        # Units should be converted to Kelvin
        assert out["KSAC"].attrs.get("units") == "K"
        # Coordinates and elevation attributes set
        assert out["KSAC"].attrs.get("coordinates") == (38.5, -121.5)
        assert "m" in str(out["KSAC"].attrs.get("elevation", ""))
        # Latitude/longitude/elevation variables dropped
        assert "latitude" not in out.variables
        assert "longitude" not in out.variables
        assert "elevation" not in out.variables


@patch("climakitae.new_core.processors.bias_adjust_model_to_station.xr.open_mfdataset")
@patch("climakitae.new_core.processors.processor_utils.convert_stations_to_points")
class TestBiasCorrectStationDataLoading:
    """Tests for loading station data (_load_station_data)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ProcClass = BiasCorrectStationData

    def test_load_station_data_single_station(self, mock_convert, mock_open_mf):
        """Test loading single station data."""
        proc = self.ProcClass({"stations": ["KSAC"]})

        # Provide a minimal catalog with stations table
        proc.catalog = {
            "stations": pd.DataFrame({"station id": [1234], "station": ["KSAC"]})
        }

        # Mock convert_stations_to_points
        mock_convert.return_value = (
            None,
            [
                {
                    "station_id_numeric": 1234,
                    "station_id": "1234",
                    "station_name": "KSAC",
                }
            ],
        )

        # Mock open_mfdataset to return a simple dataset
        times = pd.date_range("2010-01-01", periods=2)
        mock_open_mf.return_value = xr.Dataset(
            {"KSAC": ("time", [1.0, 2.0])}, coords={"time": times}
        )

        station_ds = proc._load_station_data()

        assert isinstance(station_ds, xr.Dataset)
        assert "KSAC" in station_ds.data_vars


class TestBiasCorrectStationDataBiasCorrection:
    """Tests for bias correction logic (_bias_correct_model_data)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ProcClass = BiasCorrectStationData

    @pytest.mark.advanced
    def test_bias_correct_model_data_successful(self):
        """Test _bias_correct_model_data method with observational and gridded data.

        This test uses realistic multi-year daily data to validate the QDM
        bias correction workflow with proper dayofyear grouping.
        """
        proc = self.ProcClass({"stations": ["KSAC"]})

        # Create realistic observational data (5 years, daily frequency)
        # QDM requires full year coverage for dayofyear grouping
        obs_times = pd.date_range("1980-01-01", "1984-12-31", freq="D")

        # Generate realistic temperature data with seasonal cycle
        # Base temp ~15°C with ±10°C seasonal variation
        dayofyear = obs_times.dayofyear
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (dayofyear - 80) / 365)
        # Add small random noise
        np.random.seed(42)
        obs_values = seasonal_temp + np.random.randn(len(obs_times)) * 2

        obs_da = xr.DataArray(obs_values, dims=("time",), coords={"time": obs_times})
        obs_da.name = "obs"
        obs_da.attrs["units"] = "K"

        # Create gridded data spanning historical + future (1980-2014)
        # This matches the typical historical training period used in the processor
        gr_times = pd.date_range("1980-01-01", "2014-12-31", freq="D")

        # Generate gridded data with similar pattern but slightly warmer (bias)
        gr_dayofyear = gr_times.dayofyear
        gr_seasonal_temp = 17 + 10 * np.sin(2 * np.pi * (gr_dayofyear - 80) / 365)
        np.random.seed(43)
        gr_values = gr_seasonal_temp + np.random.randn(len(gr_times)) * 2

        gr_da = xr.DataArray(gr_values, dims=("time",), coords={"time": gr_times})
        gr_da.name = "tas"
        gr_da.attrs["units"] = "K"

        # Test bias correction
        out = proc._bias_correct_model_data(obs_da, gr_da)

        # Verify output structure
        assert isinstance(out, xr.DataArray)
        assert "time" in out.dims
        assert out.name == "tas"

        # Verify output time range using pandas conversion
        start_time = pd.Timestamp(out.time.values[0])
        end_time = pd.Timestamp(out.time.values[-1])
        assert start_time.year == 1980
        assert end_time.year == 2014

        # Verify output has reasonable values (no NaN, finite values)
        # Note: .compute() is needed because out.isnull().any() returns a dask array
        assert not out.isnull().any().compute()
        assert np.isfinite(out.values).all()

        # Verify output length is reasonable for 35 years (1980-2014)
        # Should be ~12775 days (35 years × 365 days, after noleap calendar conversion)
        assert (
            12700 <= len(out.time) <= 12800
        )  # Allow some flexibility for calendar conversion

    @pytest.mark.advanced
    def test_bias_correct_model_data_with_sim_dimension(self):
        """Test _bias_correct_model_data with multiple simulations (sim dimension).

        This test covers the code path where data has a 'sim' dimension,
        requiring QDM to be trained and applied separately for each simulation.
        """
        proc = self.ProcClass({"stations": ["KSAC"]})

        # Create observational data (5 years, daily frequency)
        obs_times = pd.date_range("1980-01-01", "1984-12-31", freq="D")
        dayofyear = obs_times.dayofyear
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (dayofyear - 80) / 365)
        np.random.seed(42)
        obs_values = seasonal_temp + np.random.randn(len(obs_times)) * 2

        obs_da = xr.DataArray(obs_values, dims=("time",), coords={"time": obs_times})
        obs_da.name = "obs"
        obs_da.attrs["units"] = "K"

        # Create gridded data with 'sim' dimension (multiple simulations)
        gr_times = pd.date_range("1980-01-01", "2014-12-31", freq="D")
        n_times = len(gr_times)

        # Generate data for each simulation with different biases
        gr_dayofyear = gr_times.dayofyear

        # Simulation 1: warmer bias
        sim1_temp = 17 + 10 * np.sin(2 * np.pi * (gr_dayofyear - 80) / 365)
        np.random.seed(43)
        sim1_values = sim1_temp + np.random.randn(n_times) * 2

        # Simulation 2: even warmer bias
        sim2_temp = 18 + 10 * np.sin(2 * np.pi * (gr_dayofyear - 80) / 365)
        np.random.seed(44)
        sim2_values = sim2_temp + np.random.randn(n_times) * 2

        # Stack simulations into array with sim dimension
        gr_values = np.stack([sim1_values, sim2_values], axis=0)

        gr_da = xr.DataArray(
            gr_values,
            dims=("sim", "time"),
            coords={"sim": ["sim1", "sim2"], "time": gr_times},
        )
        gr_da.name = "tas"
        gr_da.attrs["units"] = "K"

        # Test bias correction
        out = proc._bias_correct_model_data(obs_da, gr_da)

        # Verify output structure includes sim dimension
        assert isinstance(out, xr.DataArray)
        assert "sim" in out.dims
        assert "time" in out.dims
        assert out.name == "tas"

        # Verify we have both simulations in output
        assert len(out.sim) == 2
        assert "sim1" in out.sim.values
        assert "sim2" in out.sim.values

        # Verify output time range
        start_time = pd.Timestamp(out.time.values[0])
        end_time = pd.Timestamp(out.time.values[-1])
        assert start_time.year == 1980
        assert end_time.year == 2014

        # Verify output has reasonable values (no NaN, finite values)
        # Note: .compute() is needed because out.isnull().any() returns a dask array
        assert not out.isnull().any().compute()
        assert np.isfinite(out.values).all()

        # Verify each simulation was processed independently
        # Both should have similar length (35 years × 365 days)
        assert (
            12700 <= len(out.time) <= 12800
        )  # Allow flexibility for calendar conversion


@patch(
    "climakitae.new_core.processors.bias_adjust_model_to_station.get_closest_gridcell"
)
class TestBiasCorrectStationDataClosestGridcell:
    """Tests for closest gridcell selection and bias correction wiring.

    Verifies that `_get_bias_corrected_closest_gridcell` calls
    `get_closest_gridcell`, removes non-dimension coords from the
    returned gridcell, and attaches station metadata to the bias-corrected
    output.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.ProcClass = BiasCorrectStationData

    def test_get_bias_corrected_closest_gridcell_successful(self, mock_get_closest):
        """Test getting bias-corrected closest gridcell."""
        proc = self.ProcClass({"stations": ["KSAC"]})

        # Minimal station observational dataarray (time series) with attrs
        times = pd.date_range("2000-01-01", periods=3)
        station_da = xr.DataArray(
            [0.1, 0.2, 0.3], dims=("time",), coords={"time": times}
        )
        station_da.attrs["coordinates"] = (38.5, -121.5)
        station_da.attrs["elevation"] = "10 m"

        # Placeholder gridded data
        gridded_da = xr.DataArray(
            [0.0, 0.0, 0.0], dims=("time",), coords={"time": times}
        )

        # Build a fake closest-gridcell dataset with extra non-dimension coords
        ds_times = pd.date_range("2000-01-01", periods=3)
        ds_closest = xr.Dataset(
            {"tas": ("time", [1.0, 2.0, 3.0])},
            coords={
                "time": ds_times,
                "latitude": 38.5,
                "longitude": -121.5,
                "elevation": 10.0,
            },
        )

        mock_get_closest.return_value = ds_closest

        captured = {}

        # Patch the bias-correction call to capture the gridded input
        def fake_bias(obs_da, gridded_da_arg, historical_da=None):
            captured["gridded_after_drop"] = gridded_da_arg
            return xr.DataArray(
                [10, 20, 30], dims=("time",), coords={"time": ds_times}, name="tas"
            )

        proc._bias_correct_model_data = fake_bias

        out = proc._get_bias_corrected_closest_gridcell(station_da, gridded_da)

        # Basic return type checks and metadata propagation
        assert isinstance(out, xr.DataArray)
        assert out.attrs.get("station_coordinates") == station_da.attrs["coordinates"]
        assert out.attrs.get("station_elevation") == station_da.attrs["elevation"]

        # Ensure get_closest_gridcell was called with correct coordinates
        mock_get_closest.assert_called_once()
        call_args = mock_get_closest.call_args
        assert call_args[0][1] == 38.5  # lat
        assert call_args[0][2] == -121.5  # lon

        # Confirm the gridded argument passed into bias-correction had non-dimension
        # coords (latitude/longitude/elevation) removed
        gr_after = captured.get("gridded_after_drop")
        assert "latitude" not in gr_after.coords
        assert "longitude" not in gr_after.coords
        assert "elevation" not in gr_after.coords


class TestBiasCorrectStationDataExecution:
    """Tests for main execute method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ProcClass = BiasCorrectStationData

    @patch(
        "climakitae.new_core.processors.bias_adjust_model_to_station.get_closest_gridcell"
    )
    @patch.object(BiasCorrectStationData, "_load_station_data")
    def test_execute_with_dataarray_input(self, mock_load, mock_get_closest):
        """Test execute with xr.DataArray input."""
        proc = self.ProcClass({"stations": ["KSAC"]})

        # Create a minimal input DataArray with time dimension
        times = pd.date_range("2000-01-01", periods=5)
        input_da = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0], dims=("time",), coords={"time": times}
        )
        input_da.name = "tas"

        # Mock _load_station_data to return a simple Dataset
        station_da = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims="time",
            coords={"time": times[:3]},
            attrs={"coordinates": (38.5, -121.5), "elevation": "10 m", "units": "K"},
        )
        mock_load.return_value = xr.Dataset({"KSAC": station_da})

        # Mock get_closest_gridcell to return input_da
        mock_get_closest.return_value = input_da

        # Mock _bias_correct_model_data to avoid QDM complexity
        with patch.object(proc, "_bias_correct_model_data") as mock_bias_correct:
            mock_bias_correct.return_value = xr.Dataset({"KSAC": station_da}).to_array(
                dim="station"
            )

            context = {}
            result = proc.execute(input_da, context)

        # Verify result is a Dataset
        assert isinstance(result, xr.Dataset)
        assert "KSAC" in result.data_vars

    @patch.object(BiasCorrectStationData, "_load_station_data")
    @patch.object(BiasCorrectStationData, "_process_single_dataset")
    def test_execute_with_dict_input(self, mock_process, mock_load):
        """Test execute with dictionary input (pre-concatenation)."""
        proc = self.ProcClass({"stations": ["KSAC"]})

        # Mock inputs
        ssp_da = MagicMock(spec=xr.DataArray)
        hist_da = MagicMock(spec=xr.DataArray)

        input_dict = {"ssp245": ssp_da, "historical": hist_da}

        # Mock return values
        mock_load.return_value = MagicMock(spec=xr.Dataset)
        mock_process.return_value = MagicMock(spec=xr.Dataset)

        context = {}
        result = proc.execute(input_dict, context)

        assert isinstance(result, dict)
        assert "ssp245" in result
        assert "historical" in result

        # Verify calls
        # Should call process for ssp245 with historical data
        # Should call process for historical with itself

        assert mock_process.call_count == 2

        # Check args for ssp245 call
        # We don't know order of iteration, so check call_args_list

        calls = mock_process.call_args_list

        # Find call for ssp245
        ssp_call = None
        for call in calls:
            if call[0][0] == ssp_da:
                ssp_call = call
                break

        assert ssp_call is not None
        assert ssp_call[0][2] == context  # context arg
        assert ssp_call[0][3] == hist_da  # historical_da arg

        # Find call for historical
        hist_call = None
        for call in calls:
            if call[0][0] == hist_da:
                hist_call = call
                break

        assert hist_call is not None
        assert hist_call[0][2] == context  # context arg
        assert hist_call[0][3] == hist_da  # historical_da arg


class TestBiasCorrectStationDataContext:
    """Test class for update_context method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiasCorrectStationData({})

    def test_update_context_creates_new_attrs_key(self):
        """Test that update_context creates 'new_attrs' key in context.

        The processor should add a 'new_attrs' key to context with attributes
        to be added to the final dataset.
        """
        context = {"catalog": "hadisd"}

        self.processor.update_context(context)

        # Verify 'new_attrs' key was created
        assert "new_attrs" in context
        # Verify it's a dictionary
        assert isinstance(context["new_attrs"], dict)


class TestBiasCorrectStationDataCatalogSetting:
    """Test class for set_data_accessor method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiasCorrectStationData({})

    def test_set_data_accessor_successful(self):
        """Test that set_data_accessor stores the data accessor.

        The processor should store a reference to the data accessor for
        loading additional data during processing.
        """
        mock_accessor = MagicMock()

        self.processor.set_data_accessor(mock_accessor)

        # Verify the accessor was stored as 'catalog' attribute
        assert self.processor.catalog is mock_accessor


class TestBiasCorrectStationDataEdgeCases:
    """Test class for edge cases and invalid inputs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BiasCorrectStationData({})

    @patch(
        "climakitae.new_core.processors.bias_adjust_model_to_station.get_closest_gridcell"
    )
    @patch.object(BiasCorrectStationData, "_load_station_data")
    def test_execute_with_dataset_input(self, mock_load, mock_get_closest):
        """Test execute method when input is a Dataset instead of DataArray.

        The processor should handle Dataset inputs by mapping over data
        variables. This verifies robustness to different input types.
        """
        # Create a simple Dataset with a data variable and time coordinate
        time_values = pd.date_range("2020-01-01", periods=3)
        ds = xr.Dataset(
            {
                "tas": xr.DataArray(
                    [1.0, 2.0, 3.0], dims=["time"], coords={"time": time_values}
                )
            }
        )

        # Mock _load_station_data to return a simple station Dataset
        station_time = pd.date_range("2020-01-01", periods=3)
        station_da = xr.DataArray(
            [10.0, 11.0, 12.0],
            dims=["time"],
            coords={"time": station_time},
            attrs={"coordinates": (38.5, -121.5), "elevation": "10 m", "units": "K"},
        )
        mock_load.return_value = xr.Dataset({"KSAC": station_da})

        # Mock get_closest_gridcell
        mock_get_closest.return_value = ds["tas"]

        # Mock _bias_correct_model_data
        with patch.object(
            self.processor, "_bias_correct_model_data"
        ) as mock_bias_correct:
            mock_bias_correct.return_value = xr.Dataset({"KSAC": station_da}).to_array(
                dim="station"
            )

            context = {}
            result = self.processor.execute(ds, context)

        # Should return a Dataset
        assert isinstance(result, xr.Dataset)
        # Should have station data
        assert "KSAC" in result.data_vars


class TestBiasCorrectConcatIntegration:
    """Test integration with concatenated data (sim dimension)."""

    def setup_method(self):
        self.processor = BiasCorrectStationData(
            {
                "stations": ["KSAC"],
                "historical_slice": (2000, 2001),  # Short period for test
            }
        )

    @patch(
        "climakitae.new_core.processors.bias_adjust_model_to_station.get_closest_gridcell"
    )
    @patch.object(BiasCorrectStationData, "_load_station_data")
    def test_execute_with_concatenated_input(self, mock_load, mock_get_closest):
        """Test execute with a single DataArray containing 'sim' dimension."""

        # Create concatenated input data (2 simulations, historical + future)
        # Time range: 2000-2003 (2000-2001 historical, 2002-2003 future)
        times = pd.date_range("2000-01-01", "2003-12-31", freq="D")
        # Filter to match simple calendar if needed, but standard is fine for mock

        # Create DataArray with sim, time, y, x
        da = xr.DataArray(
            np.random.rand(2, len(times), 5, 5),
            dims=["sim", "time", "y", "x"],
            coords={
                "sim": ["model1", "model2"],
                "time": times,
                "y": np.arange(5),
                "x": np.arange(5),
            },
            name="t2",
            attrs={"resolution": "9 km", "units": "K", "grid_label": "d02"},
        )

        # Mock station data
        # Station data should cover historical period
        station_times = pd.date_range("2000-01-01", "2001-12-31", freq="D")
        station_da = xr.DataArray(
            np.random.rand(len(station_times)) + 273.15,
            dims=["time"],
            coords={"time": station_times},
            name="KSAC",
            attrs={"units": "K", "coordinates": (38.5, -121.5), "elevation": "10 m"},
        )
        station_ds = xr.Dataset({"KSAC": station_da})
        mock_load.return_value = station_ds

        # Mock get_closest_gridcell to return a slice of the input
        # It needs to return something that looks like the input but spatially subsetted
        def side_effect(data, lat, lon, print_coords=False):
            # Return data at index 0,0 spatially, preserving sim and time
            if "y" in data.dims and "x" in data.dims:
                return data.isel(y=0, x=0)
            return data

        mock_get_closest.side_effect = side_effect

        # Context
        context = {"query": {"grid_label": "d02"}}

        # Execute
        # We need to mock QuantileDeltaMapping because it does complex stats
        with patch(
            "climakitae.new_core.processors.bias_adjust_model_to_station.QuantileDeltaMapping"
        ) as mock_qdm:
            # Mock QDM.train and .adjust
            mock_qdm_instance = MagicMock()
            mock_qdm.train.return_value = mock_qdm_instance

            # adjust returns the adjusted data
            # It should have same shape as input (or sliced input)
            # The processor slices output to input time range (or user requested?)
            # The processor extracts output slice from input data time range

            def adjust_side_effect(data):
                # Return data with same dims/coords
                return data

            mock_qdm_instance.adjust.side_effect = adjust_side_effect

            result = self.processor.execute(da, context)

            # Verification
            assert isinstance(result, xr.Dataset)
            assert "KSAC" in result.data_vars

            # Check that QDM.train was called
            assert mock_qdm.train.called

            # Check arguments to QDM.train
            # args[0] is obs (station data)
            # args[1] is hist (historical model data)
            call_args = mock_qdm.train.call_args
            obs_arg = call_args[0][0]
            hist_arg = call_args[0][1]

            # Obs should be 2D (station, time)
            assert obs_arg.dims == ("station", "time")

            # Hist should have station, simulation, time
            assert "station" in hist_arg.dims
            assert "simulation" in hist_arg.dims
            assert "time" in hist_arg.dims

            # Check that historical data was correctly sliced from input
            # Should cover 2000-2001
            assert hist_arg.time.dt.year.min() == 2000
            assert hist_arg.time.dt.year.max() == 2001

            # Check that result has 'sim' dimension
            assert "sim" in result["KSAC"].dims
            assert len(result["KSAC"].sim) == 2
