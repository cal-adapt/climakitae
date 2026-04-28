# Core Concepts

Understanding these three foundational concepts will help you use climakitae effectively.

## Catalog Hierarchy

Climate data is organized hierarchically by **increasing specificity**. Each level narrows down the dataset:

```
Catalog
  ↓ (which data collection?)
Activity ID
  ↓ (what downscaling method?)
Institution ID
  ↓ (which organization produced it?)
Source ID
  ↓ (which climate model?)
Experiment ID
  ↓ (which scenario/emissions path?)
Table ID
  ↓ (what time resolution?)
Grid Label
  ↓ (what spatial resolution?)
Variable ID
  ↓ (what climate quantity?)
```

### Example: Navigating the Hierarchy

You want monthly maximum temperature from WRF downscaling. Here's how it maps to the hierarchy:

```python
from climakitae.new_core.user_interface import ClimateData

cd = ClimateData()

# Step 1: Choose catalog (data collection)
cd.catalog("cadcat")           # Cal-Adapt's main collection

# Step 2: Choose downscaling method
cd.activity_id("WRF")          # Dynamical downscaling (vs. "LOCA2" for statistical)

# Step 3: Narrow by institution (optional, WRF specific)
cd.institution_id("UCLA")      # UCLA produced these WRF runs

# Step 4: Choose emissions scenario
cd.experiment_id("ssp245")     # SSP 2-4.5 moderate emissions scenario

# Step 5: Choose time resolution
cd.table_id("mon")             # Monthly (vs. "day", "1hr")

# Step 6: Choose spatial resolution
cd.grid_label("d03")           # 3km fine-scale (vs. "d02" 9km, "d01" 45km)

# Step 7: Choose variable
cd.variable("t2max")           # Daily maximum temperature

# Execute the query
data = cd.get()
```

### Key Hierarchy Levels Explained

| Level | Purpose | Examples | Impact |
|-------|---------|----------|--------|
| **Catalog** | Data source collection | `"cadcat"`, `"renewable energy generation"` | Determines what datasets exist |
| **Activity ID** | Downscaling method | `"WRF"` (dynamical), `"LOCA2"` (statistical) | Different variable names, coverage, bias |
| **Institution** | Data producer | `"UCLA"`, `"CNRM"`, `"DWD"` | Different model implementations, bias characteristics |
| **Source ID** | Climate model | `"CNRM-CM6-1"`, `"ACCESS-CM2"` | Different model physics, skill varies |
| **Experiment** | Emissions scenario | `"historical"`, `"ssp245"`, `"ssp585"` | Different futures; historical is observation-based |
| **Table ID** | Time resolution | `"1hr"`, `"day"`, `"mon"` | Memory/processing trade-off |
| **Grid Label** | Spatial resolution | `"d01"` (45km), `"d02"` (9km), `"d03"` (3km) | Local detail vs. state-wide coverage |
| **Variable** | Climate quantity | `"tasmax"`, `"pr"`, `"huss"` | What you're analyzing |

### Why Hierarchy Matters

1. **Type safety**: Not all combinations are valid (e.g., can't mix WRF variables with LOCA2 experiment_id)
2. **Discovery**: Use `show_*_options()` to see what's available at each level
3. **Query building**: Chain selections to narrow the search space
4. **Error prevention**: The validator catches invalid combinations early

### Discovering Available Boundaries {#available-boundaries}

ClimateData provides multiple boundary types for spatial subsetting:

**California Divisions:**
- `ca_counties`: Administrative counties (58 total)
- `ca_watersheds`: Hydrologic units at HUC8 level
- `ca_census_tracts`: Census block groups and tracts

**Regional/National:**
- `us_states`: Western US states (11 states)

**Infrastructure:**
- `ious_pous`: California investor-owned and publicly-owned utilities
- `electric_balancing_areas`: Electric grid authority boundaries
- `forecast_zones`: NOAA NWS forecast zones

```python
# Discover available boundaries
cd.show_boundary_options()                  # All types
cd.show_boundary_options("ca_counties")    # All CA counties
```

For spatial subsetting examples, see [How-To → Clip Data](howto.md#clip-data).

---

## Lazy Evaluation with Dask and xarray

ClimateData returns **lazy-loaded datasets** that don't consume memory or download data until you explicitly compute results.

### What Lazy Evaluation Means

```python
# This line does NOT download or load data into memory
data = (cd
    .variable("tasmax")
    .processes({"time_slice": ("2015-01-01", "2050-12-31")})
    .get())

print(f"Dataset size: {data.nbytes / 1e9:.2f} GB")  # Shows theoretical size without loading

# Data is still on disk or in S3 as "tasks" (a computation graph)
print(type(data["tasmax"].data))  # <class 'dask.array.core.Array'>
```

### Triggering Computation

Only these operations actually compute and load data:

```python
# Compute explicitly
result = data["tasmax"].mean(dim=["lat", "lon"]).compute()

# Or implicitly (common operations trigger compute)
plot = data["tasmax"].isel(time=0).plot()  # plotting triggers load
df = data.to_pandas()                       # conversion triggers load
```

### Why Lazy Evaluation Matters

| Benefit | Impact |
|---------|--------|
| **Memory efficiency** | Process 100GB datasets on a laptop |
| **I/O optimization** | Only download/load what you use |
| **Parallel computing** | Dask distributes work across cores/clusters |
| **Exploration** | Quick schema inspection before heavy computation |

### Lazy Evaluation Best Practices

```python
# ✅ GOOD: Subset spatially BEFORE statistics
data_clipped = data.sel(lat=slice(34, 36), lon=slice(-120, -118))
temp_mean = data_clipped["tasmax"].mean().compute()

# ❌ INEFFICIENT: Compute full dataset then subset
data_full = data["tasmax"].compute()
temp_mean = data_full.sel(lat=slice(34, 36), lon=slice(-120, -118)).mean()

# ✅ GOOD: Use processors to subset in the query
data = (cd.variable("tasmax")
    .processes({"clip": ((34, 36), (-120, -118))})
    .get())
temp_mean = data["tasmax"].mean().compute()
```

### dask and xarray Integration

ClimateData uses two complementary libraries:

- **xarray**: Multi-dimensional labeled arrays (the data structure)
  - Provides dimensions like `time`, `lat`, `lon`, `sim`
  - Preserves metadata (attributes, coordinates)
  - Works with both NumPy arrays and dask arrays transparently

- **dask**: Parallel computing (the execution engine)
  - Builds a task graph of pending operations
  - Distributes computation across CPU cores
  - Handles larger-than-memory datasets via chunking
  - Can be deployed to clusters (Kubernetes, HPC) for distributed computing

```python
# Both use the same API
data = cd.variable("tasmax").get()

# Inspect the task graph (shows pending operations)
print(data["tasmax"].dask)

# Control parallelism
import dask
dask.config.set(scheduler='threads', num_workers=4)  # 4-thread computation
data["tasmax"].mean().compute()

# Or use persistent clusters for long-running work
from dask.distributed import Client
client = Client(n_workers=8, threads_per_worker=2)
result = data["tasmax"].mean().compute()
```

---

## Global Warming Levels (GWL) {#global-warming-levels}

Global warming levels represent **future climate states defined by degrees of global warming**, rather than specific calendar years. This is more scientifically meaningful than time-based analysis.

### Why GWL Matters

Climate impacts scale with global temperature increase, not calendar years:

- **Time-based**: "In 2050, temperature will increase by X°C"
  - Problem: Different models warm at different rates; 2050 is very different across models
  
- **GWL-based**: "When Earth warms 2°C above pre-industrial, temperature will increase by Y°C"
  - Advantage: Consistent across all models; directly tied to climate impacts

### GWL Reference Periods

Warming is measured relative to a baseline (pre-industrial):

- **1850-1900 reference**: Pre-industrial baseline (most common for 1.5/2°C targets)
- **1981-2010 reference**: Recent observational baseline (useful for impact quantification)

```python
# GWL defined relative to 1850-1900
warming_1850_ref = 0.8  # Current warming vs. 1850-1900
warming_1850_ref = 2.0  # 2°C warming target (Paris Agreement)
```

### Using GWL in Queries

```python
from climakitae.new_core.user_interface import ClimateData

cd = ClimateData()

# Query data around 2°C global warming
data = (cd
    .variable("tasmax")
    .processes({
        "warming_level": {
            "warming_levels": [1.5, 2.0, 3.0],  # Multiple levels
            "warming_level_window": 15  # ±15 years around crossing
        }
    })
    .get())

# Result has data centered on when each model crosses that warming level
print(data["tasmax"].shape)  # Time dimension spans the warming window
```

### GWL Advantages

| Aspect | Time-based | GWL-based |
|--------|-----------|-----------|
| **Alignment** | Each model uses different years | All models use same warming threshold |
| **Comparability** | Difficult to compare models | Direct model comparison |
| **Impact relevance** | Arbitrary calendar dates | Tied to climate sensitivity |
| **Policy alignment** | Less relevant to climate agreements | Direct link to Paris targets (1.5°C, 2°C) |

### GWL Availability

Not all model-scenario combinations reach all warming levels:

```python
# Some models never reach 4°C warming in SSP2-4.5
data = (cd
    .experiment_id("ssp245")
    .processes({"warming_level": {"warming_levels": [4.0]}})
    .get())

# Result may be NaN if model doesn't reach 4°C
if data.isnull().all():
    print("This model does not warm to 4°C in SSP2-4.5")
```

### GWL Window Interpretation

The `warming_level_window` parameter defines a time window around the warming level crossing:

```python
"warming_level": {
    "warming_levels": [2.0],
    "warming_level_window": 15  # ±15 years
}
```

- Retrieves data for years when global warming = 2.0 ± 0.5°C
- Spans roughly 30-year window centered on crossing
- Larger window = more temporal data at target warming level
- Useful for capturing climate variability around the threshold

---

## Putting It Together: A Complete Example

Here's how all three concepts work in a real query:

```python
from climakitae.new_core.user_interface import ClimateData
import matplotlib.pyplot as plt

# 1. Use hierarchy to specify the exact dataset
cd = ClimateData()
data = (cd
    .catalog("cadcat")           # Cal-Adapt data
    .activity_id("WRF")          # Dynamical downscaling
    .institution_id("UCLA")      # UCLA WRF runs
    .experiment_id("ssp370")     # High emissions scenario
    .table_id("day")             # Daily data
    .grid_label("d03")           # 3km resolution
    .variable("tasmax")          # Max temperature
    
    # 2. Use processors to subset (maintains lazy eval)
    .processes({
        "clip": "Los Angeles",   # Spatial subset
        "warming_level": {       # Use GWL instead of calendar years
            "warming_levels": [2.0],
            "warming_level_window": 10
        }
    })
    .get())  # Still lazy at this point

# 3. Now compute what you need (triggers lazy evaluation)
# Calculate mean at 2°C warming
mean_temp = data["tasmax"].mean(dim=["lat", "lon", "time"]).compute()

# Plot one snapshot
data["tasmax"].isel(sim=0, time=0).plot(cmap="RdYlBu_r")
plt.title("Temperature at 2°C Global Warming - LA")
plt.show()
```

Key takeaways:
- **Hierarchy**: Precise dataset selection → reduces ambiguity
- **Lazy evaluation**: Efficient computation → explores 100GB datasets easily
- **GWL**: Climate-science-meaningful analysis → policy-aligned results

---

## Next Steps

- See [Getting Started](../getting-started.md) for a quick tutorial
- Review [Migration Guide](../migration/legacy-to-new-core.md) if upgrading from legacy API
- Check [API Reference](../api/new-core.md) for detailed parameter documentation
