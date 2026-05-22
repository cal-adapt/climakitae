# Notebook Gallery

Curated examples from the [cae-notebooks](https://github.com/cal-adapt/cae-notebooks) repository demonstrating climate data analysis workflows with climakitae.

Each notebook is interactive and can be run live on Binder or downloaded to your local environment. These notebooks show best practices for data access, climate analysis, and visualization.

---

## Data Access & Setup

### Basic Climate Data Access

**Access and subset climate data from the Cal-Adapt Analytics Engine catalog.**

- **Level**: Beginner
- **Duration**: 10-15 minutes
- **Key Topics**: Data selection • Spatial subsetting • Temporal subsetting • Export formats
- **What You'll Learn**:  
  - How to query the Cal-Adapt data catalog
  - Selecting variables, downscaling methods, and time periods
  - Clipping data to regions of interest
  - Exporting to NetCDF, CSV, and other formats

**Links**: 

- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/basic_data_access.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=basic_data_access.ipynb) Launch on Binder

---

### Renewables Data Access

**Retrieve capacity factor and generation data for solar PV and wind power installations across California.**

- **Level**: Beginner – Intermediate
- **Key Topics**: Renewable energy data • Solar PV • Wind power • Capacity factors
- **What You'll Learn**:
  - How to access renewable energy generation data via the climakitae API
  - Querying solar and wind capacity factor datasets
  - Analyzing generation potential across California regions

**Links**:

- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/renewables_data_access.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=renewables_data_access.ipynb) Launch on Binder

---

### Weather Station Data Access

**Access quality-controlled historical weather observations from the Historical Observations Data Platform.**

- **Level**: Intermediate
- **Key Topics**: Historical observations • HadISD • Weather stations • `intake` catalog
- **What You'll Learn**:
  - How to open and query the historical weather station catalog
  - Filtering stations by location and variable
  - Working with observational data alongside gridded climate projections

**Links**:

- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/weather_station_data_access.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=weather_station_data_access.ipynb) Launch on Binder

---

## Analysis & Climate Science

### Global Warming Levels: Methods & Applications

**Explore global warming levels (GWLs) as an alternative to time-based climate projections.**

- **Level**: Intermediate
- **Duration**: 20-30 minutes
- **Key Topics**: Global warming levels • Warming level trajectories • Cross-model comparison • Climate scenarios
- **What You'll Learn**:
  - Why global warming levels are scientifically meaningful
  - How to query data by warming level instead of calendar year
  - Comparing impacts across different climate scenarios
  - Handling models that don't reach specific warming levels

**Links**:  
- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/warming_level_methods.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=warming_level_methods.ipynb) Launch on Binder

---

### Threshold Exceedance & Extreme Events

**Analyze frequency and intensity of extreme weather events using threshold-based methods.**

- **Level**: Intermediate
- **Duration**: 25-35 minutes
- **Key Topics**: Threshold definition • Event frequency • Return periods • Compound events
- **What You'll Learn**:
  - How to define and detect threshold exceedance events
  - Counting consecutive days above/below a threshold
  - Analyzing how event frequency changes under warming
  - Visualizing compound conditions (e.g., heat + humidity)

**Links**:  
- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/threshold_tools.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=threshold_tools.ipynb) Launch on Binder

---

### Custom Derived Variables

**Define and use custom derived metrics within climakitae using the `register_user_function` tool.**

- **Level**: Intermediate – Advanced
- **Key Topics**: Derived variables • `register_user_function` • Custom metrics • Multi-variable computation
- **What You'll Learn**:
  - How to register a custom metric function from existing catalog variables
  - Using the `ClimateData` interface to compute derived quantities
  - Comparing the streamlined workflow to manual multi-variable loading

**Links**:

- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/derived_variables_demo.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=derived_variables_demo.ipynb) Launch on Binder

---

## Climate Profiles

### Custom Climate Profiles

**Generate annualized hourly climate profiles for energy system modeling and planning.**

- **Level**: Intermediate
- **Key Topics**: Climate profiles • Annualized Hourly Climate Profile • Standard year • Typical Meteorological Year • Energy planning
- **What You'll Learn**:
  - How to generate a custom Standard Year climate profile for a location
  - Customizing profiles by variable, scenario, and warming level
  - Exporting profiles to CSV for use in energy system models

**Links**:

- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/custom_climate_profiles.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=custom_climate_profiles.ipynb) Launch on Binder

---

## Interactive Development Environment

Want to develop and test new notebooks with climakitae? Launch a full development environment on Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/climakitae/main?urlpath=lab)

This provides:  

- ✅ Jupyter Lab with full IDE features  
- ✅ climakitae installed in editable mode (source changes live-reload)  
- ✅ All documentation build tools (mkdocs, mkdocstrings)  
- ✅ Example notebooks from cae-notebooks  
- ✅ Complete development environment (pytest, black, isort, git)  
- ⏱️ Up to 6 hours of continuous usage per session  

Perfect for:  
- Testing notebook examples  
- Developing new climate analysis workflows  
- Contributing to climakitae or cae-notebooks  
- Learning the climakitae API interactively  

---

## Running Notebooks Locally

### Option 1: Binder (No Installation Required)

Click any "Launch on Binder" button above to run notebooks in your browser without local setup. Binder automatically installs all dependencies.

**Advantages**:  
- ✅ No installation needed  
- ✅ Works from any browser  
- ✅ Temporary session (changes not saved)  

**Disadvantages**:  
- ⚠️ Limited computational resources  
- ⚠️ Session times out after 10 minutes of inactivity  
- ⚠️ Changes are not persisted  

### Option 2: Local Installation

For persistent work or larger analyses, install climakitae and dependencies locally:

```bash
# Clone the repository
git clone https://github.com/cal-adapt/cae-notebooks.git
cd cae-notebooks

# Install with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or install with conda
conda create -n cae -f conda-linux-64.lock
conda activate cae

# Start Jupyter
jupyter lab
```

### Option 3: Cal-Adapt Analytics Engine

Access pre-installed notebooks on the [Cal-Adapt Analytics Engine JupyterHub](https://analytics.cal-adapt.org/) with no setup required.

---

## Notebook Difficulty Progression

**Suggested learning path**:  

1. **Start**: Basic Climate Data Access (understand the data model)  
2. **Next**: Global Warming Levels (key climakitae feature)  
3. **Then**: Threshold Exceedance or Renewables Data Access (real-world applications)  
4. **Advanced**: Custom Derived Variables or Custom Climate Profiles (extend the framework)

---

## For More Information

- **climakitae Documentation**: this site (cal-adapt.github.io/climakitae)  
- **cae-notebooks Repository**: [https://github.com/cal-adapt/cae-notebooks](https://github.com/cal-adapt/cae-notebooks)  
- **Cal-Adapt Analytics Engine**: [https://analytics.cal-adapt.org/](https://analytics.cal-adapt.org/)  
  - [Example applications](https://analytics.cal-adapt.org/analytics/applications/example)  
  - [Methods](https://analytics.cal-adapt.org/analytics/methods)  
  - [Glossary](https://analytics.cal-adapt.org/guidance/glossary)  
- **Cal-Adapt Overview**: [https://cal-adapt.org/](https://cal-adapt.org/)  

---

## Contributing

Have a notebook example you'd like to share? Contributions are welcome! See the [cae-notebooks CONTRIBUTING guide](https://github.com/cal-adapt/cae-notebooks/blob/main/README.md) for details.

---

## Binder Configuration

The Binder environment is configured in `.binder/` with:  
- `runtime.txt`: Python 3.12  
- `environment.yml`: Conda dependencies (scientific computing, geospatial, Jupyter, documentation tools)  
- `postBuild`: Installs climakitae in editable mode, configures Jupyter Lab  

For details, see [.binder/README.md](https://github.com/cal-adapt/climakitae/blob/main/.binder/README.md).
