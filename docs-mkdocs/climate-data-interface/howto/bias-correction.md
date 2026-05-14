# Bias Correction: Localize WRF to Weather Stations

Use historical weather station observations to correct WRF model bias locally.

## Basic Localization

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

## Available Weather Stations

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

## How Bias Correction Works

- **Training**: Uses historical station observations (1981-2010 baseline)
- **Method**: Quantile delta mapping (preserves model trends while matching observations)
- **Result**: WRF temperature distribution matches local observations
- **Benefit**: Reduces systematic bias for climate projections

## Limitations

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
