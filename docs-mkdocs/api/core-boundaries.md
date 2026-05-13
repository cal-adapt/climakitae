# Core Boundaries (Detailed)

Geographic boundary management for climate data clipping.

## Overview

`climakitae.core.boundaries` provides the `Boundaries` class for managing geographic regions used to subset climate data. It handles:  
- Loading predefined boundaries (counties, watersheds, utilities)
- Lazy loading to minimize memory usage
- Caching of boundary geometries
- Geographic operations (clipping, spatial queries)

!!! note
    Both legacy (`climakitae.core.boundaries`) and the ClimateData interface (`climakitae.new_core.data_access.boundaries`) implement similar functionality. New code should use the ClimateData version.

## Boundaries Class

::: climakitae.core.boundaries.Boundaries
    options:
      docstring_style: numpy
      show_source: true
      members_order: source
      merge_init_into_class: true

## Available Boundary Types

The Boundaries class provides access to several predefined boundary catalogs:

- **US States** — Western US states
- **CA Counties** — All 58 California counties
- **CA Watersheds** — HUC8-level watersheds in California
- **CA Utilities** — Investor-owned utilities (IOUs) and publicly-owned utilities (POUs)
- **CA Electric Zones** — Electricity demand forecast zones
- **CA Balancing Authorities** — Electric balancing authority areas
- **CA Census Tracts** — Census tract boundaries

## Usage Example

```python
from climakitae.core.boundaries import Boundaries

# Initialize boundary loader
boundaries = Boundaries()

# Get available counties
counties = boundaries.available_counties()

# Get geometry for specific county
la_geom = boundaries.get_geometry("Los Angeles")

# Multiple regions (union)
multi_geom = boundaries.get_geometry(["Alameda", "Contra Costa"])
```
