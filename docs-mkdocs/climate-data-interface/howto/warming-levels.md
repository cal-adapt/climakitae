# Warming Level-Based Analysis

Query climate data relative to global warming thresholds instead of calendar years.

## Query Around Multiple Warming Levels

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

## Compare Multiple Models at Same Warming Level

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

## Handle Models Without Target Warming Level

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

## Warming Level Reference Periods

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
