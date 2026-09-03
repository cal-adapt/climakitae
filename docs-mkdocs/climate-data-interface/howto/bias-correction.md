# Bias Correction: Localize WRF to Weather Stations

Use historical weather station observations to correct WRF model bias locally.

## Basic Localization

```python
# ⚠️  Currently WRF + hourly temperature only
data = (cd
    .activity_id("WRF")
    .institution_id("UCLA")      # Specify WRF producer
    .variable_id("tas")             # Hourly temperature
    .table_id("1hr")             # Must be hourly
    .processes({
        "bias_adjust_model_to_station": {
            "stations": ["ASOSAWOS_69007093217"]
        }
    })
    .get())

# Data now bias-corrected to observations
```

## Available Weather Stations

```python
# Browse the HDP catalog for available network_id/station_id combinations
from climakitae.new_core.data_access.data_access import DataCatalog
DataCatalog().hdp.df[["network_id", "station_id"]]

# All stations passed to bias_adjust_model_to_station in one call must
# belong to the same network_id.
data = (cd
    .processes({
        "bias_adjust_model_to_station": {
            "stations": ["ASOSAWOS_69007093217"]
        }
    })
    .get())
```

## How Bias Correction Works

- **Training**: Uses each station's actual historical observational overlap with the model's historical period (coverage varies per HDP station)
- **Method**: Quantile delta mapping (preserves model trends while matching observations)
- **Result**: WRF temperature distribution matches local observations
- **Benefit**: Reduces systematic bias for climate projections

## Limitations

**Currently available for:**  
- ✅ WRF data only (not LOCA2 statistical downscaling)  
- ✅ Hourly temperature (`tas`) only  
- ✅ HDP weather stations (single network per call; not every network provides temperature)  

**Why these limitations?**

Bias correction requires:  
- **High-frequency observations** (hourly) to capture temperature variability that drives quantile mapping  
- **WRF hourly data** because WRF's fast-varying dynamics need point-wise calibration  
- **LOCA2 is already bias-corrected** by design using quantile mapping to observations during downscaling (no bias correction needed)  
- **Weather station coverage** — stations are sourced from the HDP catalog, whose per-station time coverage and variable availability varies by network  

**For other scenarios:**  
- Use direct model output (LOCA2 is already bias-corrected)  
- Implement alternative bias correction method for daily/monthly aggregates  
- Contact support for custom approaches  

---
