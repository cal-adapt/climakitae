# Access Boundary Geometries Directly

Retrieve GeoDataFrames for specific geographic regions using the named accessor
methods on the `Boundaries` class. This is useful when you need the geometry
itself — for plotting, spatial joins, or custom clipping logic — rather than
passing a name to the `clip` processor.

## Setup

```python
from climakitae.new_core import CATALOG

boundaries = CATALOG.boundaries
```

Data is loaded from S3 lazily — nothing is fetched until you call a method.

## Get All Boundaries of a Type

Call any accessor with no argument to return the full GeoDataFrame:

```python
all_counties   = boundaries.get_counties()
all_watersheds = boundaries.get_watersheds()
all_states     = boundaries.get_states()
all_utilities  = boundaries.get_utilities()
all_fz         = boundaries.get_forecast_zones()
all_eba        = boundaries.get_electric_balancing_areas()
all_tracts     = boundaries.get_census_tracts()
```

## Get a Single Boundary by Name

Pass a name to retrieve a single-row GeoDataFrame:

```python
# Counties — "County" suffix is optional
alameda     = boundaries.get_counties("Alameda")
alameda     = boundaries.get_counties("Alameda County")  # same result

# Watersheds
feather     = boundaries.get_watersheds("Feather")

# Western US states (use abbreviation)
california  = boundaries.get_states("CA")

# Electric utilities
pge         = boundaries.get_utilities("Pacific Gas & Electric Company")

# Electricity demand forecast zones
scge_fz     = boundaries.get_forecast_zones("SCE")

# Electric balancing authority areas
caliso      = boundaries.get_electric_balancing_areas("CALISO")

# Census tracts (use GEOID, not tract name — names are not unique across counties)
tract       = boundaries.get_census_tracts("06001400100")
```

An invalid name raises a `ValueError` listing all valid options:

```python
boundaries.get_counties("Fake County")
# ValueError: County 'Fake County' not found. Available: ['Alameda County', ...]
```

## Discover Available Names

To see what names are valid before making a call:

```python
# Returns a dict of {category: [name, ...]} for every boundary type
available = CATALOG.list_clip_boundaries()

print(available["CA counties"][:5])
# ['Alameda County', 'Alpine County', 'Amador County', 'Butte County', 'Calaveras County']

print(available["states"])
# ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
```

## Plot a Boundary

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 6))
boundaries.get_counties("Alameda County").plot(ax=ax, edgecolor="black")
ax.set_title("Alameda County")
plt.show()
```

## Use a Geometry for Custom Clipping

```python
import xarray as xr

geom = boundaries.get_counties("Santa Clara County").geometry.iloc[0]

ds = xr.open_dataset("my_climate_data.nc")
ds_clipped = ds.rio.clip([geom], crs="EPSG:4326")
```

---
