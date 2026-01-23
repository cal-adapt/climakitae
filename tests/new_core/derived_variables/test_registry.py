import types

import xarray as xr
import numpy as np

from climakitae.new_core.derived_variables import registry as reg


def setup_function():
    # Reset module state between tests
    reg._DERIVED_METADATA.clear()
    reg._DERIVED_REGISTRY = None


def test_register_user_function_validation():
    def dummy(ds):
        return ds

    try:
        # empty name
        try:
            reg.register_user_function("", ["t2"], dummy)
            assert False, "Expected ValueError for empty name"
        except ValueError:
            pass

        # empty depends_on
        try:
            reg.register_user_function("myvar", [], dummy)
            assert False, "Expected ValueError for empty depends_on"
        except ValueError:
            pass

        # non-callable func
        try:
            reg.register_user_function("myvar", ["t2"], 123)
            assert False, "Expected TypeError for non-callable func"
        except TypeError:
            pass
    finally:
        reg._DERIVED_METADATA.clear()


def test_register_user_function_success_and_drop_flag():
    # simple function that creates a derived var from t2
    def make_myvar(ds):
        ds = ds.copy()
        ds["myvar"] = ds["t2"] + 2
        ds["myvar"].attrs = {"units": "K"}
        return ds

    # ensure clean state
    reg._DERIVED_METADATA.clear()
    reg._DERIVED_REGISTRY = None

    reg.register_user_function("myvar", ["t2"], make_myvar, description="d", units="K", drop_dependencies=False)

    assert reg.is_derived_variable("myvar")
    info = reg.get_derived_variable_info("myvar")
    assert info is not None
    assert info.source == "user"
    assert info.units == "K"

    # Call the stored wrapped function and ensure source variable remains when drop_dependencies=False
    ds = xr.Dataset({"t2": ("time", np.array([290.0]))}, coords={"time": [0]})
    out = info.func(ds.copy())
    assert "myvar" in out.data_vars
    assert "t2" in out.data_vars


def test_register_derived_decorator_and_wrapping():
    # clean
    reg._DERIVED_METADATA.clear()
    reg._DERIVED_REGISTRY = None

    @reg.register_derived(variable="tmp_range", query={"variable_id": ["t2"]}, description="range", units="K", drop_dependencies=False)
    def tmp_range(ds):
        ds = ds.copy()
        ds["tmp_range"] = ds["t2"] + 1
        return ds

    # decorator should have registered metadata
    assert reg.is_derived_variable("tmp_range")
    info = reg.get_derived_variable_info("tmp_range")
    assert info is not None
    assert info.source == "builtin"

    # call wrapped function via metadata and ensure it preserves source when drop_dependencies=False
    ds = xr.Dataset({"t2": ("time", np.array([291.0])),}, coords={"time": [0]})
    out = info.func(ds.copy())
    assert "tmp_range" in out.data_vars
    assert "t2" in out.data_vars


def test_preserve_spatial_metadata_copies_coords_and_attrs():
    # build dataset with source and derived variables
    times = [0]
    src = xr.DataArray(np.array([1.0]), dims=("time",), coords={"time": times})
    derived = xr.DataArray(np.array([0.0]), dims=("time",), coords={"time": times})

    ds = xr.Dataset({"source": src, "derived": derived}, coords={"time": times})
    # add CRS-like coordinate and grid_mapping attr
    ds = ds.assign_coords({"Lambert_Conformal": ("time", np.array([999]))})
    ds.coords["gridmap"] = ("time", np.array([1]))
    ds["source"].attrs["grid_mapping"] = "gridmap"

    # run preserve
    reg.preserve_spatial_metadata(ds, "derived", "source")

    # derived should now have Lambert_Conformal coord and grid_mapping attr
    assert "Lambert_Conformal" in ds["derived"].coords
    assert ds["derived"].attrs.get("grid_mapping") == "gridmap"

def test_wrap_with_ds_and_result_having_data_vars():
    # Wrapper should detect added vars when both ds and result have data_vars
    def fn(ds):
        ds = ds.copy()
        ds["newvar"] = ds["t2"] * 2
        return ds

    wrapped = reg._wrap_with_metadata_preservation(fn, "newvar", ["t2"], drop_dependencies=True)
    ds = xr.Dataset({"t2": ("time", np.array([290.0]))}, coords={"time": [0]})
    result = wrapped(ds)
    assert "newvar" in result.data_vars
    # t2 should be dropped
    assert "t2" not in result.data_vars


def test_wrap_source_var_lookup_from_original_ds():
    # Wrapper should find source_var in original ds when available
    def fn(ds):
        ds = ds.copy()
        ds["derived"] = ds["t2"] + 1
        return ds

    wrapped = reg._wrap_with_metadata_preservation(fn, "derived", ["t2"], drop_dependencies=False)
    ds = xr.Dataset({"t2": ("time", np.array([290.0]))}, coords={"time": [0]})
    ds["t2"].attrs["grid_mapping"] = "crs"
    result = wrapped(ds)
    # Should preserve grid_mapping
    assert "grid_mapping" in result["derived"].attrs


def test_wrap_multiple_added_vars_iteration():
    # Function adds multiple new variables, wrapper should iterate through all targets
    def fn(ds):
        ds = ds.copy()
        ds["var1"] = ds["t2"] + 1
        ds["var2"] = ds["t2"] + 2
        return ds

    wrapped = reg._wrap_with_metadata_preservation(fn, "var1", ["t2"], drop_dependencies=True)
    ds = xr.Dataset({"t2": ("time", np.array([290.0]))}, coords={"time": [0]})
    ds["t2"].attrs["grid_mapping"] = "test_grid"
    result = wrapped(ds)
    # Both vars should be in result
    assert "var1" in result.data_vars
    assert "var2" in result.data_vars


def test_preserve_spatial_metadata_missing_vars():
    # Test early return when derived or source not in dataset
    ds = xr.Dataset({"t2": ("time", np.array([290.0]))}, coords={"time": [0]})
    # Should handle gracefully when vars missing
    reg.preserve_spatial_metadata(ds, "missing_derived", "t2")
    reg.preserve_spatial_metadata(ds, "t2", "missing_source")
    # No error should occur


def test_preserve_spatial_metadata_lambert_conformal_coord():
    # Test copying Lambert_Conformal coordinate specifically
    times = [0]
    src = xr.DataArray(np.array([1.0]), dims=("time",), coords={"time": times})
    derived = xr.DataArray(np.array([0.0]), dims=("time",), coords={"time": times})

    ds = xr.Dataset({"source": src, "derived": derived}, coords={"time": times})
    # Add Lambert_Conformal coordinate (WRF-specific)
    ds = ds.assign_coords({"Lambert_Conformal": ("time", np.array([42]))})
    ds["source"].attrs["grid_mapping"] = "Lambert_Conformal"

    reg.preserve_spatial_metadata(ds, "derived", "source")

    # derived should now have Lambert_Conformal coord
    assert "Lambert_Conformal" in ds["derived"].coords
    assert ds["derived"].attrs.get("grid_mapping") == "Lambert_Conformal"


def test_preserve_spatial_metadata_rioxarray_grid_mapping_crs():
    """Test preserve_spatial_metadata with rioxarray when derived has grid_mapping.
    
    This test covers lines 274-291 in registry.py, where the derived variable
    has a grid_mapping attribute and we try to parse CRS from it, both when
    it succeeds and when it raises an exception.
    """
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        # Skip test if rioxarray not available
        return

    # Test 1: Successful CRS parsing from grid_mapping
    times = [0]
    src = xr.DataArray(np.array([1.0]), dims=("time",), coords={"time": times})
    derived = xr.DataArray(np.array([0.0]), dims=("time",), coords={"time": times})

    ds = xr.Dataset({"source": src, "derived": derived}, coords={"time": times})
    
    # Add a grid_mapping coordinate with CRS-parseable content
    ds.coords["Lambert_Conformal"] = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": [30.0, 60.0],
            "longitude_of_central_meridian": -97.0,
            "latitude_of_projection_origin": 40.0,
            "false_easting": 0.0,
            "false_northing": 0.0,
        }
    )
    
    # Set grid_mapping on derived variable (not source, to test the elif branch)
    ds["derived"].attrs["grid_mapping"] = "Lambert_Conformal"

    # This exercises lines 274-286 (successful CRS parsing)
    reg.preserve_spatial_metadata(ds, "derived", "source")
    
    # If successful, derived should still have the grid_mapping
    assert ds["derived"].attrs.get("grid_mapping") == "Lambert_Conformal"
    
    # Test 2: Exception during CRS access (lines 286-291)
    # Use mock to make .rio.crs raise an exception
    from unittest.mock import PropertyMock, patch
    
    src2 = xr.DataArray(np.array([1.0]), dims=("time",), coords={"time": times})
    derived2 = xr.DataArray(np.array([0.0]), dims=("time",), coords={"time": times})
    ds2 = xr.Dataset({"source": src2, "derived": derived2}, coords={"time": times})
    
    ds2.coords["crs_coord"] = xr.DataArray(0)
    ds2["derived"].attrs["grid_mapping"] = "crs_coord"
    
    # Mock .rio.crs to raise an exception when accessed
    with patch.object(type(ds2["derived"].rio), "crs", new_callable=PropertyMock) as mock_crs:
        mock_crs.side_effect = RuntimeError("Simulated CRS parsing error")
        
        # This should exercise lines 286-291 (exception handler)
        # The function should handle the exception gracefully and not crash
        reg.preserve_spatial_metadata(ds2, "derived", "source")
    
    # Function should complete without error despite CRS parsing failure
    assert ds2["derived"].attrs.get("grid_mapping") == "crs_coord"