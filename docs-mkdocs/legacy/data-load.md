# Legacy Data Loading

The **`climakitae.core.data_load` module** holds the internal helpers that
assemble and load legacy datasets behind `get_data()` and
`DataParameters.retrieve()`.

!!! warning
    This page documents a legacy support module. It is kept for backward
    compatibility. New code should use
    [`climakitae.new_core.user_interface.ClimateData`](../climate-data-interface/index.md).

## What this module does

- Builds the intake-esm query from a populated `DataParameters` object.
- Streams the matching datasets from S3 as lazily loaded xarray objects.
- Applies the unit, area-average, and concatenation steps the legacy pipeline
  expects before returning data to the caller.

These functions are rarely called directly; they are the machinery behind the
legacy [Data Interface](data-interface.md).

---

## Public API

::: climakitae.core.data_load
    options:
      docstring_style: numpy
      show_source: true

---

## Related legacy modules

- [Legacy API Overview](index.md)
- [Data Interface](data-interface.md)
- [Data Export](data-export.md)
- [Legacy → ClimateData migration guide](../migration/legacy-to-climate-data.md)
