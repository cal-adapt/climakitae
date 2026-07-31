# Legacy Data Export

The **`climakitae.core.data_export` module** provides the legacy multi-format
export helpers used to persist retrieved data to disk or cloud storage.

!!! warning
    This page documents a legacy support module. It is kept for backward
    compatibility. New code should prefer the
    [`export` processor](../climate-data-interface/processors/export.md) in the
    modern interface.

## What this module does

The legacy export layer writes an in-memory or lazily loaded xarray object to
one of several formats:

| Format | Typical use |
|--------|-------------|
| NetCDF | Standard climate-data format with full metadata (CF conventions). |
| CSV | Tabular export for time series or single-location data. |
| Zarr | Cloud-optimized chunked storage for large datasets. |
| GeoTIFF | Geographic raster for a single time slice, for use in GIS tools. |

---

## Public API

The single public entry point is `export()`. TMY/EPW export is reached through
`export()` rather than a separate function.

::: climakitae.core.data_export.export
    options:
      docstring_style: numpy
      show_source: true

---

## Related legacy modules

- [Legacy API Overview](index.md)
- [Data Interface](data-interface.md)
- [Data Loading](data-load.md)
- [Legacy → ClimateData migration guide](../migration/legacy-to-climate-data.md)
