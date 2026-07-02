# Batch Processing: Multiple Points or Scenarios

Efficiently process many geographic locations or climate scenarios.

## Process Multiple Points

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

## Process Multiple Scenarios

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

## Large Batch with Progress Tracking

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

## Parallel Batch Processing

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

## Distributed Computation with Coiled (AWS)

climakitae returns lazy [Dask](https://docs.dask.org)-backed xarray objects, so any Dask scheduler — including a [Coiled](https://coiled.io) cluster — takes over automatically when `.compute()` is called. No climakitae-specific integration is needed.

```python
import coiled
from climakitae.new_core.user_interface import ClimateData

# Spin up a Coiled cluster in us-west-2 (same region as Cal-Adapt S3 data)
cluster = coiled.Cluster(
    n_workers=10,
    region="us-west-2",
    name="climakitae-batch",
)
client = cluster.get_client()  # Registers cluster as default Dask scheduler

# From here, all .compute() calls run on the cluster
cd = ClimateData(verbosity=-1)

results = {}
for scenario in ["historical", "ssp245", "ssp370", "ssp585"]:
    data = (cd
        .catalog("cadcat")
        .activity_id("LOCA2")
        .experiment_id(scenario)
        .variable("tasmax")
        .table_id("mon")
        .grid_label("d03")
        .processes({
            "time_slice": ("2020-01-01", "2060-12-31"),
            "clip": "Los Angeles",
        })
        .get())
    # Computation is dispatched to Coiled workers
    results[scenario] = data["tasmax"].mean(dim=["lat", "lon"]).compute()

cluster.shutdown()
```

!!! tip "Region matters"
    Place your Coiled cluster in `us-west-2`. Cal-Adapt data lives in S3 `us-west-2`
    buckets, so workers co-located in the same region avoid egress costs and
    significantly reduce transfer latency.

## Best Practice: Cache Intermediate Results

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
