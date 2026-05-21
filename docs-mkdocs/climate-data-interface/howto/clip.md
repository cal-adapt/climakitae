# Clip Data to a Spatial Region

Subset your data to a specific geographic area using boundaries, points, or bounding boxes.

## By Named Region (County, Watershed, etc.)

```python
from climakitae.new_core.user_interface import ClimateData

cd = ClimateData()

# Single county
data = (cd
    .catalog("cadcat")
    .activity_id("LOCA2")
    .variable("tasmax")
    .table_id("mon")
    .grid_label("d03")
    .processes({"clip": "Los Angeles County"})
    .get())

# Multiple counties (union)
data = (cd
    .catalog("cadcat")
    .activity_id("LOCA2")
    .variable("tasmax")
    .table_id("mon")
    .grid_label("d03")
    .processes({"clip": ["Alameda County", "Contra Costa County", "Santa Clara County"]})
    .get())

# Discover available regions
cd.show_boundary_options()  # All boundary types
cd.show_boundary_options("ca_counties")  # All California counties
```

**Available boundary types:**

| Boundary Type | Description |
|---------------|-------------|
| `ca_counties` | California counties (58 total) |
| `ca_watersheds` | Hydrologic units (HUC8) |
| `ca_census_tracts` | Census geography |
| `us_states` | Western US states (11 states) |
| `ious_pous` | California investor-owned and public utilities |
| `forecast_zones` | NOAA forecast zones |
| `electric_balancing_areas` | Electric grid authorities |

## By Single Point (Lat/Lon)

```python
# Query closest grid cell to a point
data = (cd
    .catalog("cadcat")
    .activity_id("LOCA2")
    .variable("tasmax")
    .table_id("day")
    .grid_label("d03")
    .processes({"clip": (37.7749, -122.4194)})  # San Francisco
    .get())

# Inspect the closest location
print(f"Lat: {data.lat.values}, Lon: {data.lon.values}")
```

## By Multiple Points

```python
# Query nearest grid cell for each point
locations = [
    (34.05, -118.25),   # Los Angeles
    (37.77, -122.42),   # San Francisco
    (32.72, -117.16),   # San Diego
]

data = (cd
    .catalog("cadcat")
    .activity_id("LOCA2")
    .variable("tasmax")
    .table_id("day")
    .grid_label("d03")
    .processes({"clip": locations})
    .get())

# Data has 'closest_cell' dimension
print(f"Shape: {data['tasmax'].shape}")  # (time, closest_cell, ...)
la_data = data.isel(closest_cell=0)
sf_data = data.isel(closest_cell=1)
sd_data = data.isel(closest_cell=2)
```

## By Bounding Box

```python
# Rectangular region: (lat_range, lon_range)
data = (cd
    .catalog("cadcat")
    .activity_id("LOCA2")
    .variable("tasmax")
    .table_id("mon")
    .grid_label("d03")
    .processes({
        "clip": ((34.0, 36.0), (-121.0, -119.0))  # LA area
    })
    .get())

# Extract subset
data_subset = data.sel(lat=slice(34.5, 35.5), lon=slice(-120.5, -119.5))
```

## Best Practice: Combine with Time Slice

```python
# Always clip FIRST, then aggregate
# ✅ EFFICIENT
data = (cd
    .catalog("cadcat")
    .activity_id("LOCA2")
    .variable("tasmax")
    .table_id("mon")
    .grid_label("d03")
    .processes({
        "clip": "Los Angeles County",
        "time_slice": ("2015-01-01", "2015-12-31")
    })
    .get())

spatial_mean = data["tasmax"].mean(dim=["lat", "lon"]).compute()

# ❌ INEFFICIENT (loads all data first)
data_full = (cd
    .catalog("cadcat")
    .activity_id("LOCA2")
    .variable("tasmax")
    .table_id("mon")
    .grid_label("d03")
    .get())
la_data = data_full.sel(lat=slice(33.5, 35.5), lon=slice(-119, -117))
spatial_mean = la_data["tasmax"].mean().compute()
```

---
