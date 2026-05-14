# Time-Based Queries

Query data using traditional calendar date ranges (alternative to warming level analysis).

## Date Range Subsetting

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

## When to Use Time-Based vs. Warming Level

**Time-Based**: Planning for specific calendar years, historical analysis  
**Warming Level**: Climate impact assessment, multi-model consistency, policy targets

For comparison and advanced usage, see [Warming Level Analysis](warming-levels.md).

---
