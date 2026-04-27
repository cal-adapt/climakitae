# Migration: Legacy to New Core

This guide will provide side-by-side replacements for legacy `core/` usage.

## Migration policy

- New work should use `new_core`.
- Legacy APIs remain available during transition.
- Each legacy docs page should include a migration callout.

## Planned mapping sections

- `DataParameters` + `get_data(...)` -> `ClimateData()` fluent queries
- Legacy export patterns -> processor-driven export workflows
- Legacy options discovery -> `show_*_options()` in new core

Detailed examples will be added in `climakitae-6c9.7`.
