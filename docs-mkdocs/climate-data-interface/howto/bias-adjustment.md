# Bias Adjustment: Localize WRF to Weather Stations

Use historical weather station observations to correct WRF model bias locally. Historical weather station data has gone through quality assurance (QA) and quality control (QC) protocols. More details about the data itself can be found at the [**Historical Data Platform (HDP)**](https://github.com/Eagle-Rock-Analytics/historical-obs-platform) repository.


## Basic Localization

```python
# ⚠️  Currently WRF + hourly temperature or dewpoint only
data = (cd
    .activity_id("WRF")
    .institution_id("UCLA")      # Specify WRF producer
    .variable_id("t2")             # Hourly temperature
    .table_id("1hr")             # Must be hourly
    .processes({
        "bias_adjust_model_to_station": {
            "stations": ["KSAC"]
        }
    })
    .get())

# Data now bias-adjusted to observations
```

## Available Weather Stations

```python
# Browse the HDP catalog for available network_id/station_id combinations
from climakitae.new_core.data_access.data_access import DataCatalog
DataCatalog().hdp.df[["network_id", "station_id"]]

# Stations passed to bias_adjust_model_to_station in one call may span
# multiple network_ids.
data = (cd
    .processes({
        "bias_adjust_model_to_station": {
            "stations": ["KSAC", "ASOSAWOS_69007093217"]
        }
    })
    .get()
)
```

## Year-Level Data Quality Checks

If the query pins a specific year range — directly via `time_slice`, or indirectly via `warming_level` — each requested station/variable/year is checked against a QA sheet before anything runs. Overall-coverage problems are hard errors; per-month/diurnal coverage gaps are warnings that don't block the query.

```python
# Error: KSAN's dew_point record has an overall coverage gap in this range,
# so the query is rejected before any data is fetched.
data = (cd
    .activity_id("WRF")
    .institution_id("UCLA")
    .variable_id("dew_point")
    .table_id("1hr")
    .processes({
        "time_slice": (2000, 2003),
        "bias_adjust_model_to_station": {"stations": ["KSAN"]},
    })
    .get())
# -> None, with a logged error naming the station/variable/year and the
#    nearest usable year(s) (or that none were found nearby).

# Warning: KSAN's tas record is missing some summer months in this range,
# but the query still runs.
data = (cd
    .activity_id("WRF")
    .institution_id("UCLA")
    .variable_id("t2")
    .table_id("1hr")
    .processes({
        "time_slice": (2000, 2003),
        "bias_adjust_model_to_station": {"stations": ["KSAN"]},
    })
    .get())
# -> data is returned, with a logged warning naming the affected month(s)
#    and noting the result may be biased.
```

## How Bias Adjustment Works

- **Training**: Uses each station's actual historical observational overlap with the model's historical period (coverage varies per HDP station)
- **Method**: Quantile delta mapping (preserves model trends while matching observations)
- **Result**: WRF temperature distribution matches local observations
- **Benefit**: Reduces systematic bias for climate projections

## Limitations

**Currently available for:**  
- ✅ WRF data only (not LOCA2 statistical downscaling)  
- ✅ Hourly temperature (t2) or dewpoint (dew_point)  
- ✅ HDP weather stations (not every network provides temperature)  

**Why these limitations?**

Bias adjustment requires:  
- **High-frequency observations** (hourly) to capture temperature variability that drives quantile mapping  
- **WRF hourly data** because WRF's fast-varying dynamics need point-wise calibration  
- **LOCA2 is already bias-adjusted** by design using quantile mapping to observations during downscaling (no bias adjustment needed)  
- **Weather station coverage** — stations are sourced from the HDP catalog, whose per-station time coverage and variable availability varies by network  

**For other scenarios:**  
- Use direct model output (LOCA2 is already bias-corrected)  
- Implement alternative bias adjustment method for daily/monthly aggregates  
- Contact support for custom approaches  

---
