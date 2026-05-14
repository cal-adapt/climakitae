# Combining Multiple Techniques

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
