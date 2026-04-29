# How-To Guides

**You have a goal — here's the recipe.** Each section below answers a concrete
analysis question with a runnable `ClimateData` chain.

For a parameter-by-parameter reference of any single processor, see
[Processor Reference](processors/index.md).

| Goal | Section |
|---|---|
| Clip to a county, watershed, point, or bbox | [Clip data to a spatial region](#clip-data) |
| Save to NetCDF / Zarr / CSV / GeoTIFF | [Export data to files](#export-data) |
| Query at 1.5 / 2 / 3 °C global warming | [Warming-level analysis](#warming-level-analysis) |
| Localize WRF temperature to a station | [Bias correction](#bias-correction-station-localization) |
| Process many points or scenarios | [Batch processing](#batch-processing-multiple-points-or-scenarios) |
| Build a multi-model / multi-region pipeline | [Combining techniques](#multi-model-ensemble) |
| Compute HDD / CDD / heat index / etc. | [Derived variables & climate indices](#derived-variables) |
| Subset by calendar dates | [Time-based queries](#time-based-queries) |

---

## Clip Data to a Spatial Region {#clip-data}

Subset your data to a specific geographic area using boundaries, points, or bounding boxes.

### By Named Region (County, Watershed, etc.)

```python
from climakitae.new_core.user_interface import ClimateData

cd = ClimateData()

# Single county
data = (cd
    .variable("tasmax")
    .table_id("mon")
    .grid_label("d03")
    .processes({"clip": "Los Angeles"})  # "Los Angeles" = Los Angeles County
    .get())

# Multiple counties (union)
data = (cd
    .variable("tasmax")
    .processes({"clip": ["Alameda", "Contra Costa", "San Francisco Bay"]})
    .get())

# Discover available regions
cd.show_boundary_options()  # All boundary types
cd.show_boundary_options("ca_counties")  # All California counties
```

**Available boundary types:**
- `ca_counties`: California counties (58 total)
- `ca_watersheds`: Hydrologic units (HUC8)
- `ca_census_tracts`: Census geography
- `us_states`: Western US states (11 states)
- `forecast_zones`: NOAA forecast zones
- `electric_balancing_areas`: Electric grid authorities

### By Single Point (Lat/Lon)

```python
# Query closest grid cell to a point
data = (cd
    .variable("tasmax")
    .processes({"clip": (37.7749, -122.4194)})  # San Francisco
    .get())

# Inspect the closest location
print(f"Lat: {data.lat.values}, Lon: {data.lon.values}")
```

### By Multiple Points

```python
# Query nearest grid cell for each point
locations = [
    (34.05, -118.25),   # Los Angeles
    (37.77, -122.42),   # San Francisco
    (32.72, -117.16),   # San Diego
]

data = (cd
    .variable("tasmax")
    .processes({"clip": locations})
    .get())

# Data has 'closest_cell' dimension
print(f"Shape: {data['tasmax'].shape}")  # (time, closest_cell, ...)
la_data = data.isel(closest_cell=0)
sf_data = data.isel(closest_cell=1)
sd_data = data.isel(closest_cell=2)
```

### By Bounding Box

```python
# Rectangular region: (lat_range, lon_range)
data = (cd
    .variable("tasmax")
    .processes({
        "clip": ((34.0, 36.0), (-121.0, -119.0))  # LA area
    })
    .get())

# Extract subset
data_subset = data.sel(lat=slice(34.5, 35.5), lon=slice(-120.5, -119.5))
```

### Best Practice: Combine with Time Slice

```python
# Always clip FIRST, then aggregate
# ✅ EFFICIENT
data = (cd
    .variable("tasmax")
    .processes({
        "clip": "Los Angeles",
        "time_slice": ("2015-01-01", "2015-12-31")
    })
    .get())

spatial_mean = data["tasmax"].mean(dim=["lat", "lon"]).compute()

# ❌ INEFFICIENT (loads all data first)
data_full = cd.variable("tasmax").get()
la_data = data_full.sel(lat=slice(33.5, 35.5), lon=slice(-119, -117))
spatial_mean = la_data["tasmax"].mean().compute()
```

---

## Export Data to Files {#export-data}

Save your climate data in multiple formats for external analysis, GIS, or archival.

### Export to NetCDF (Default)

```python
# Simple export during query
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "export": {
            "filename": "la_temperature_2015",
            "file_format": "NetCDF"
        }
    })
    .get())

# File saved as: la_temperature_2015.nc
# Retrieve as: data["tasmax"].compute()
```

### Export to Zarr (Cloud-Optimized, Scalable)

```python
# Zarr format: excellent for large datasets, cloud-native
data = (cd
    .variable("t2max")
    .activity_id("WRF")
    .institution_id("UCLA")  # Specify WRF producer
    .processes({
        "time_slice": ("2015-01-01", "2050-12-31"),
        "export": {
            "filename": "long_term_temps",
            "file_format": "Zarr"
        }
    })
    .get())

# File saved as: long_term_temps/
# Useful for: large multi-model ensembles, cloud storage (S3)
```

### Export to CSV (Tabular Data)

```python
# CSV export: for time series or point data
data = (cd
    .variable("tasmax")
    .processes({
        "clip": [(34.05, -118.25), (37.77, -122.42)],  # Two points
        "time_slice": ("2015-01-01", "2015-12-31"),
        "export": {
            "filename": "point_timeseries",
            "file_format": "CSV",
            "separated": True,  # Export each point separately
            "location_based_naming": True
        }
    })
    .get())

# Files saved as: point_timeseries_34.05_-118.25.csv, etc.
```

### Export to GeoTIFF (Raster, GIS-compatible)

```python
# GeoTIFF: for single time slice or aggregated spatial rasters
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2015-01-01", "2015-01-01"),  # Single day
        "export": {
            "filename": "temp_snapshot",
            "file_format": "GeoTIFF"
        }
    })
    .get())

# File saved as: temp_snapshot.tif
# Compatible with: QGIS, ArcGIS, GDAL tools
```

### Export with Compression and Checkpointing

```python
# Skip re-processing if file exists
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2015-01-01", "2050-12-31"),
        "export": {
            "filename": "long_term",
            "file_format": "Zarr",
            "export_method": "skip_existing"  # Don't re-process if exists
        }
    })
    .get())
```

### Best Practice: Chain Multiple Exports

```python
# Export intermediate results for different use cases
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2015-01-01", "2050-12-31"),
        "clip": "Los Angeles",
        "export": {  # First export: full temporal data for analysis
            "filename": "la_full_timeseries",
            "file_format": "NetCDF"
        }
    })
    .get())

# Then compute aggregates for sharing
annual_mean = data["tasmax"].resample(time="YS").mean()
annual_mean.to_netcdf("la_annual_means.nc")  # Annual averages
```

---

## Warming Level-Based Analysis {#warming-level-analysis}

Query climate data relative to global warming thresholds instead of calendar years.

### Query Around Multiple Warming Levels

```python
# Get data around 1.5°C, 2°C, and 3°C warming
data = (cd
    .variable("tasmax")
    .experiment_id("ssp245")
    .processes({
        "warming_level": {
            "warming_levels": [1.5, 2.0, 3.0],
            "warming_level_window": 15  # ±15 years around target (default: 30 years)
        }
    })
    .get())

# Data contains: time_period centered on each warming level crossing
print(f"Time range: {data['time'].min().values} to {data['time'].max().values}")
```

### Compare Multiple Models at Same Warming Level

```python
# All models show different years for 2°C warming
# But with GWL, we can compare apples-to-apples

# UCLA WRF source models (verify with cd.show_source_id_options())
models = ["CESM2", "EC-Earth3", "MPI-ESM1-2-HR"]
results = {}

for model in models:
    data = (cd
        .activity_id("WRF")
        .institution_id("UCLA")
        .source_id(model)
        .variable("prec")  # WRF precipitation (LOCA2 uses 'pr')
        .processes({
            "warming_level": {
                "warming_levels": [2.0],
                "warming_level_window": 10
            },
            "clip": "California"
        })
        .get())
    
    results[model] = data["prec"].mean(dim=["lat", "lon", "time"]).compute()

# Now all models are at exactly 2°C warming
# Direct comparison: model_A vs model_B at same climate state
for model, precip in results.items():
    print(f"{model}: {precip.values:.1f} mm/day")
```

### Handle Models Without Target Warming Level

```python
# Some models don't reach certain warming levels in low-emission scenarios
data = (cd
    .activity_id("WRF")
    .institution_id("UCLA")
    .experiment_id("ssp245")  # Moderate emissions
    .variable("t2max")
    .processes({
        "warming_level": {"warming_levels": [4.0]}  # Very high warming
    })
    .get())

# Check if data exists
if data is None or data["t2max"].isnull().all():
    print("Model doesn't reach 4°C in SSP2-4.5 scenario")
else:
    result = data["t2max"].mean().compute()
```

### Warming Level Reference Periods

```python
# GWL measured relative to 1850-1900 (pre-industrial)
# This is the standard for climate policy (Paris Agreement)

# 1.5°C, 2°C targets → reference to 1850-1900
data = (cd
    .activity_id("WRF")
    .institution_id("UCLA")
    .variable("t2max")
    .processes({
        "warming_level": {"warming_levels": [1.5, 2.0]}
    })
    .get())

# For regional impact analysis, you can compute anomalies
# relative to 1981-2010 locally (from separate baseline data)
```

---

## Bias Correction: Localize WRF to Weather Stations {#bias-correction-station-localization}

Use historical weather station observations to correct WRF model bias locally.

### Basic Localization

```python
# ⚠️  Currently WRF + hourly temperature only
data = (cd
    .activity_id("WRF")
    .institution_id("UCLA")      # Specify WRF producer
    .variable("t2")              # Hourly 2m temperature
    .table_id("1hr")             # Must be hourly
    .processes({
        "bias_adjust_model_to_station": {
            "stations": ["KSAC", "KSFO", "KLAX"]
        }
    })
    .get())

# Data now bias-corrected to observations
```

### Available Weather Stations

```python
# List all available weather stations
cd.show_station_options()  # Returns station codes (ICAO format)

# Use with clip to find nearby station
data = (cd
    .processes({
        "bias_adjust_model_to_station": {
            "stations": ["KSFO"]  # San Francisco airport
        }
    })
    .get())
```

### How Bias Correction Works

- **Training**: Uses historical station observations (1981-2010 baseline)
- **Method**: Quantile delta mapping (preserves model trends while matching observations)
- **Result**: WRF temperature distribution matches local observations
- **Benefit**: Reduces systematic bias for climate projections

### Limitations

**Currently available for:**
- ✅ WRF data only (not LOCA2 statistical downscaling)
- ✅ Hourly temperature (t2) only  
- ✅ HadISD weather stations (~600 globally, ~200 in western US)

**Why these limitations?**

Bias correction requires:
- **High-frequency observations** (hourly) to capture temperature variability that drives quantile mapping
- **WRF hourly data** because WRF's fast-varying dynamics need point-wise calibration
- **LOCA2 is already bias-corrected** by design using quantile mapping to observations during downscaling (no bias correction needed)
- **Weather station coverage** — only HadISD provides consistent historical hourly data

**For other scenarios:**
- Use direct model output (LOCA2 is already bias-corrected)
- Implement alternative bias correction method for daily/monthly aggregates
- Contact support for custom approaches

---

## Batch Processing: Multiple Points or Scenarios

Efficiently process many geographic locations or climate scenarios.

### Process Multiple Points

```python
from climakitae.tools import batch_select
import pandas as pd

# Define locations of interest
locations = pd.DataFrame({
    "name": ["LA", "SF", "SD", "Sacramento"],
    "latitude": [34.05, 37.77, 32.72, 38.58],
    "longitude": [-118.25, -122.42, -117.16, -121.49]
})

# Batch process
results = batch_select(
    cd,
    locations,
    variable="t2max",
    activity_id="WRF",
    institution_id="UCLA",  # Specify WRF producer
    table_id="mon"
)

# Results: dict[location_name] = xr.Dataset
for name, data in results.items():
    print(f"{name}: {data['t2max'].mean().compute():.1f} K")
```

### Process Multiple Scenarios

```python
# Compare warming levels across emissions scenarios
warming_levels = [1.5, 2.0, 3.0]
scenarios = ["ssp245", "ssp370", "ssp585"]

results = {}

for scenario in scenarios:
    scenario_data = {}
    for gwl in warming_levels:
        data = (cd
            .activity_id("WRF")
            .institution_id("UCLA")
            .experiment_id(scenario)
            .variable("tasmax")
            .processes({
                "warming_level": {"warming_levels": [gwl]},
                "clip": "Los Angeles"
            })
            .get())
        
        scenario_data[gwl] = data["tasmax"].mean(dim=["lat", "lon"]).compute()
    
    results[scenario] = scenario_data

# Analyze: how much warmer at higher emission levels?
# results["ssp585"][2.0] vs results["ssp245"][2.0]
```

### Large Batch with Progress Tracking

```python
from tqdm import tqdm

counties = ["Los Angeles", "San Francisco", "Sacramento", "Fresno", "San Diego"]
results = {}

for county in tqdm(counties, desc="Processing counties"):
    data = (cd
        .variable("tasmax")
        .processes({
            "time_slice": ("2030-01-01", "2060-12-31"),
            "clip": county,
            "warming_level": {"warming_levels": [2.0]}
        })
        .get())
    
    results[county] = data["tasmax"].mean(dim=["lat", "lon"]).compute()

print("✅ Batch processing complete")
```

### Parallel Batch Processing

```python
from multiprocessing import Pool
import functools

def query_county(county_name):
    """Query temperature for one county"""
    cd = ClimateData(verbosity=-1)  # Quiet mode
    data = (cd
        .variable("tasmax")
        .processes({"clip": county_name})
        .get())
    return county_name, data["tasmax"].mean().compute()

counties = ["Los Angeles", "San Francisco", "Sacramento", "Fresno"]

with Pool(processes=4) as pool:
    results = dict(pool.map(query_county, counties))

# Results computed in parallel on 4 cores
```

### Best Practice: Cache Intermediate Results

```python
import os
from pathlib import Path

# Store results to avoid re-querying
cache_dir = Path("climate_data_cache")
cache_dir.mkdir(exist_ok=True)

counties = ["Los Angeles", "San Francisco"]

for county in counties:
    cache_file = cache_dir / f"{county.lower()}_2030_2060.nc"
    
    if cache_file.exists():
        # Load from cache
        import xarray as xr
        data = xr.open_dataset(cache_file)
    else:
        # Query and cache
        data = (cd
            .activity_id("WRF")
            .institution_id("UCLA")
            .variable("tasmax")
            .processes({
                "time_slice": ("2030-01-01", "2060-12-31"),
                "clip": county
            })
            .get())
        data.to_netcdf(cache_file)
    
    print(f"{county}: mean={data['tasmax'].mean().values:.1f}K")
```

---

## Combining Multiple Techniques {#multi-model-ensemble}

Here's a complete workflow using multiple concepts:

```python
from climakitae.new_core.user_interface import ClimateData
import matplotlib.pyplot as plt

# Workflow: Analyze temperature extremes at warming levels across California
cd = ClimateData(verbosity=-1)

# Step 1: Define regions of interest
regions = {
    "Bay Area": ("San Francisco", "Alameda"),
    "Central Valley": ("Fresno", "Sacramento"),
    "Southern CA": ("Los Angeles", "San Diego")
}

# Step 2: Query for multiple regions at 2°C warming
results = {}

for region_name, counties in regions.items():
    region_data = {}
    
    for county in counties:
        data = (cd
            .catalog("cadcat")
            .activity_id("WRF")
            .institution_id("UCLA")  # UCLA WRF model: recommended for California
            .experiment_id("ssp245")
            .variable("tasmax")
            .table_id("day")
            .grid_label("d03")
            .processes({
                "warming_level": {
                    "warming_levels": [2.0],
                    "warming_level_window": 10
                },
                "clip": county
            })
            .get())
        
        region_data[county] = data
    
    # Combine counties in region
    results[region_name] = region_data

# Step 3: Analyze and visualize
for region_name, region_data in results.items():
    county_means = []
    
    for county, data in region_data.items():
        mean_temp = data["tasmax"].mean(dim=["lat", "lon", "time"]).compute()
        county_means.append(mean_temp.values)
    
    regional_mean = sum(county_means) / len(county_means)
    print(f"{region_name}: {regional_mean:.1f} K")

# Step 4: Export summary
# (See export section for file writing)
```

---

## Derived Variables & Climate Indices {#derived-variables}

Compute derived climate metrics from primary variables using the climakitae.tools module.

### Common Derived Variables

```python
from climakitae.tools.derived_variables import compute_hdd_cdd
from climakitae.tools.indices import effective_temp, noaa_heat_index

# Fetch base temperature data
# Note: convert_units processor ensures correct units for derived variable functions
data = (cd
    .variable("tasmax")
    .table_id("day")
    .grid_label("d03")
    .processes({
        "time_slice": ("2030-01-01", "2060-12-31"),
        "clip": "Los Angeles",
        "convert_units": "degC"  # Derived functions expect Celsius
    })
    .get())

# Compute heating/cooling degree days
# Thresholds are in °C for converted data
hdd, cdd = compute_hdd_cdd(
    data["tasmax"],
    hdd_threshold=18.3,  # °C (standard: ~65°F)
    cdd_threshold=18.3   # °C (standard: ~65°F)
)

# Compute effective temperature (energy demand)
eff_temp = effective_temp(data["tasmax"])
```

### Available Functions

- `compute_hdd_cdd()`: Heating/cooling degree days for building energy modeling
- `effective_temp()`: Exponentially smoothed temperature for demand forecasting
- `noaa_heat_index()`: Heat stress indicator combining temperature and humidity

For complete list, see [Tools → Derived Variables](../api/derived-variables.md)

---

## Time-Based Queries {#time-based-queries}

Query data using traditional calendar date ranges (alternative to warming level analysis).

### Date Range Subsetting

```python
from climakitae.new_core.user_interface import ClimateData

cd = ClimateData()

# Specify exact date range
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2030-01-01", "2060-12-31")  # 30-year period
    })
    .get())

# Query by years only
data = (cd
    .variable("pr")
    .processes({
        "time_slice": (2050, 2100)  # 2050-2100
    })
    .get())

# Single time point
data = (cd
    .variable("tasmax")
    .processes({
        "time_slice": ("2050-07-15", "2050-07-15")  # One day
    })
    .get())
```

### When to Use Time-Based vs. Warming Level

**Time-Based**: Planning for specific calendar years, historical analysis  
**Warming Level**: Climate impact assessment, multi-model consistency, policy targets

For comparison and advanced usage, see [Warming Level Analysis](#warming-level-analysis).

---

## Next Steps

- Review [Core Concepts](concepts.md) for deeper understanding of hierarchy and lazy evaluation
- Check [API Reference](../api/new-core.md) for complete parameter documentation
- See [Migration Guide](../migration/legacy-to-new-core.md) if upgrading from legacy API
