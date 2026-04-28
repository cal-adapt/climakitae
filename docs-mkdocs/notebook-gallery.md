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
- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/data-access/basic_data_access.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=data-access/basic_data_access.ipynb) Launch on Binder

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
- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/analysis/warming_level_methods.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=analysis/warming_level_methods.ipynb) Launch on Binder

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
- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/analysis/thresholds_tools.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=analysis/thresholds_tools.ipynb) Launch on Binder

---

### Model Uncertainty: Understanding Multi-Model Ensembles

**Explore sources of uncertainty in climate projections from multiple climate models.**

- **Level**: Intermediate
- **Duration**: 20-25 minutes
- **Key Topics**: Ensemble uncertainty • Model spread • Climate variability • Ensemble statistics
- **What You'll Learn**:
  - Why different climate models produce different results
  - How to compute ensemble mean and spread
  - Visualizing model uncertainty with ensemble statistics
  - When to use ensemble mean vs. individual models

**Links**:
- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/analysis/model_uncertainty.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=analysis/model_uncertainty.ipynb) Launch on Binder

---

### Time Series Transformations & Analysis

**Transform and analyze climate time series data with different temporal aggregations and statistics.**

- **Level**: Intermediate
- **Duration**: 20-25 minutes
- **Key Topics**: Temporal aggregation • Percentile computation • Moving averages • Anomaly calculation
- **What You'll Learn**:
  - Resampling data to different time resolutions
  - Computing percentiles and anomalies
  - Calculating rolling statistics for extreme event detection
  - Comparing different time-based analyses

**Links**:
- [View on GitHub](https://github.com/cal-adapt/cae-notebooks/blob/main/analysis/timeseries_transformations.ipynb)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/cae-notebooks/main?filepath=analysis/timeseries_transformations.ipynb) Launch on Binder

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

1. **Start**: Basic Climate Data Access (understand data model)
2. **Next**: Global Warming Levels (key climakitae feature)
3. **Then**: Threshold Exceedance or Model Uncertainty (real-world applications)
4. **Advanced**: Time Series Transformations (custom analyses)

---

## For More Information

- **climakitae Documentation**: [https://climakitae.readthedocs.io/](https://climakitae.readthedocs.io/)
- **cae-notebooks Repository**: [https://github.com/cal-adapt/cae-notebooks](https://github.com/cal-adapt/cae-notebooks)
- **Cal-Adapt Analytics Engine**: [https://analytics.cal-adapt.org/](https://analytics.cal-adapt.org/)
- **Cal-Adapt Overview**: [https://cal-adapt.org/](https://cal-adapt.org/)

---

## Contributing

Have a notebook example you'd like to share? Contributions are welcome! See the [cae-notebooks CONTRIBUTING guide](https://github.com/cal-adapt/cae-notebooks/blob/main/README.md) for details.
