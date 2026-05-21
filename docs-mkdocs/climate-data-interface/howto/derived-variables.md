# Derived Variables & Climate Indices

Compute derived climate metrics from primary variables using the climakitae.tools module.

## Common Derived Variables

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

## Available Functions

- `compute_hdd_cdd()`: Heating/cooling degree days for building energy modeling
- `effective_temp()`: Exponentially smoothed temperature for demand forecasting
- `noaa_heat_index()`: Heat stress indicator combining temperature and humidity

For complete list, see [Tools → Derived Variables](../../api/derived-variables.md)

---
