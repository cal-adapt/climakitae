import importlib
import sys
import types
from types import SimpleNamespace

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
