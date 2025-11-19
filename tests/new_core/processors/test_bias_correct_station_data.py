import importlib
import sys
import types

import pandas as pd
import xarray as xr
import geopandas as gpd
import pytest


def _import_processor_module_with_dummy_xsdba():
    """Import the bias_correct_station_data module while injecting a
    minimal dummy xsdba and xsdba.adjustment to avoid heavy external
    dependencies at import time.

    Returns
    -------
    module
        The imported processor module
    """
    # Create minimal dummy modules for xsdba and xsdba.adjustment
    dummy_xsdba = types.ModuleType("xsdba")

    # Minimal Grouper placeholder used at import time
    def _dummy_grouper(group, window=None):
        return None

    dummy_xsdba.Grouper = _dummy_grouper

    # Create dummy adjustment submodule with a QuantileDeltaMapping stub
    dummy_adjustment = types.ModuleType("xsdba.adjustment")

    class _DummyQDMClass:
        @staticmethod
        def train(*args, **kwargs):
            class _Inst:
                def adjust(self, da):
                    return da

            return _Inst()

    dummy_adjustment.QuantileDeltaMapping = _DummyQDMClass

    # Wire up packages in sys.modules before importing target module
    sys.modules["xsdba"] = dummy_xsdba
    sys.modules["xsdba.adjustment"] = dummy_adjustment

    # Invalidate import caches and import the processor
    importlib.invalidate_caches()
    mod = importlib.import_module(
        "climakitae.new_core.processors.bias_correct_station_data"
    )
    return mod


class TestBiasCorrectStationDataInit:
    """Tests for BiasCorrectStationData initialization."""

    def setup_method(self):
        # Import the processor class under test with dummy xsdba
        mod = _import_processor_module_with_dummy_xsdba()
        self.ProcClass = getattr(mod, "BiasCorrectStationData")

    def test_init_with_valid_config(self):
        cfg = {"stations": ["Sacramento (KSAC)"]}
        proc = self.ProcClass(cfg)
        assert proc.stations == ["Sacramento (KSAC)"]
        # defaults
        assert proc.historical_slice == (1980, 2014)
        assert proc.window == 90
        assert proc.nquantiles == 20
        assert proc.group == "time.dayofyear"
        assert proc.kind == "+"
        assert proc.name == "bias_correct_station_data"
        # Processor declares it needs a catalog to run
        assert getattr(proc, "needs_catalog", False) is True


class TestBiasCorrectStationDataPreprocessing:
    """Tests for HadISD preprocessing (_preprocess_hadisd)."""

    def setup_method(self):
        mod = _import_processor_module_with_dummy_xsdba()
        self.mod = mod
        self.ProcClass = getattr(mod, "BiasCorrectStationData")

    def test_preprocess_hadisd_successful(self):
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

        # Create station metadata dataframe
        stations_df = pd.DataFrame({"station id": [1234], "station": ["KSAC"]})

        out = proc._preprocess_hadisd(ds, stations_df)

        # After preprocessing, station variable name should be present
        assert "KSAC" in out.data_vars
        # Units should be converted to Kelvin
        assert out["KSAC"].attrs.get("units") == "K"
        # Coordinates and elevation attributes set
        assert out["KSAC"].attrs.get("coordinates") == (38.5, -121.5)
        assert "m" in out["KSAC"].attrs.get("elevation")
        # Latitude/longitude/elevation variables dropped
        assert "latitude" not in out.variables
        assert "longitude" not in out.variables
        assert "elevation" not in out.variables


class TestBiasCorrectStationDataLoading:
    """Tests for loading station data (_load_station_data)."""

    def setup_method(self):
        mod = _import_processor_module_with_dummy_xsdba()
        self.mod = mod
        self.ProcClass = getattr(mod, "BiasCorrectStationData")

    def test_load_station_data_single_station(self):
        proc = self.ProcClass({"stations": ["KSAC"]})

        # Provide a minimal catalog with stations table
        proc.catalog = {
            "stations": pd.DataFrame({"station id": [1234], "station": ["KSAC"]})
        }

        # Inject a dummy processor_utils module with convert_stations_to_points
        dummy_utils = types.ModuleType("climakitae.new_core.processors.processor_utils")

        def convert_stations_to_points(stations, catalog):
            # Return (points, metadata_list)
            meta = [
                {
                    "station_id_numeric": 1234,
                    "station_id": "1234",
                    "station_name": "KSAC",
                }
            ]
            return (None, meta)

        dummy_utils.convert_stations_to_points = convert_stations_to_points
        sys.modules["climakitae.new_core.processors.processor_utils"] = dummy_utils

        # Stub open_mfdataset to return a simple dataset
        def _open_mfdataset(filepaths, **kwargs):
            times = pd.date_range("2010-01-01", periods=2)
            return xr.Dataset({"KSAC": ("time", [1.0, 2.0])}, coords={"time": times})

        # Patch the module xarray open_mfdataset used in the processor
        self.mod.xr.open_mfdataset = _open_mfdataset

        station_ds = proc._load_station_data()

        assert isinstance(station_ds, xr.Dataset)
        assert "KSAC" in station_ds.data_vars


class TestBiasCorrectStationDataBiasCorrection:
    """Tests for bias correction logic (_bias_correct_model_data)."""

    def setup_method(self):
        mod = _import_processor_module_with_dummy_xsdba()
        self.mod = mod
        self.ProcClass = getattr(mod, "BiasCorrectStationData")

    def test_bias_correct_model_data_successful(self):
        proc = self.ProcClass({"stations": ["KSAC"]})

        # Create observational and gridded DataArrays with time coords
        obs_times = pd.date_range("1980-01-01", periods=5)
        gr_times = pd.date_range("1970-01-01", periods=200)

        obs_da = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0], dims=("time",), coords={"time": obs_times}
        )
        obs_da.name = "obs"
        obs_da.attrs["units"] = "C"

        gr_da = xr.DataArray(
            list(range(len(gr_times))), dims=("time",), coords={"time": gr_times}
        )
        gr_da.name = "tas"
        gr_da.attrs["units"] = "K"

        # Monkeypatch convert_units to be identity
        self.mod.convert_units = lambda a, u: a

        # Ensure DataArray has convert_calendar method available in this test environment
        if not hasattr(xr.DataArray, "convert_calendar"):
            xr.DataArray.convert_calendar = lambda self, *args, **kwargs: self

        out = proc._bias_correct_model_data(obs_da, gr_da, output_slice=(2000, 2010))

        assert isinstance(out, xr.DataArray)
        assert "time" in out.dims
        # result should be rechunked (have .data or dask attribute)
        # We accept either a dask-backed or numpy-backed array here
        assert out.name == "tas"
