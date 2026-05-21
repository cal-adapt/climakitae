# Binder Environment for climakitae

This directory contains the configuration files for running climakitae on Binder, a free cloud-based Jupyter environment.

## Configuration Files

### `runtime.txt`
Specifies the Python version (3.12) that Binder should use.

### `environment.yml`
Conda environment specification with all climakitae dependencies:
- Scientific computing: numpy, scipy, pandas, xarray, dask
- Geospatial: geopandas, cartopy, shapely, pyproj
- Data access: intake, intake-xarray, netcdf4, zarr
- Visualization: matplotlib, seaborn, bokeh
- Jupyter: jupyterlab, notebook, ipywidgets
- Documentation: mkdocs, mkdocs-material, mkdocstrings-python
- Development: pytest, black, isort, flake8

### `postBuild`
Post-build script that runs after Conda environment setup:
- Installs climakitae in editable mode (`pip install -e .`)
- Installs pip-only dependencies (param, pydantic, tqdm)
- Configures Jupyter Lab extensions
- Creates a kernel display name "Python (climakitae)"

## Launching on Binder

Click the badge below to launch an interactive Binder session:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cal-adapt/climakitae/main?urlpath=lab)

This will:
1. Build the environment from `.binder/environment.yml`
2. Run `postBuild` to install climakitae
3. Launch Jupyter Lab with access to:
   - climakitae source code
   - Example notebooks from cae-notebooks (via git submodule or clone)
   - Documentation build tools

## Manual Binder Commands

To launch Binder with specific notebooks or paths:

```bash
# Launch with Jupyter Lab
https://mybinder.org/v2/gh/cal-adapt/climakitae/main?urlpath=lab

# Launch with Jupyter Notebook
https://mybinder.org/v2/gh/cal-adapt/climakitae/main?urlpath=notebook

# Open a specific notebook
https://mybinder.org/v2/gh/cal-adapt/climakitae/main?urlpath=lab/tree/examples/access_new_wrf.ipynb
```

Replace `main` with a specific branch or tag name to launch different versions.

## Building Locally

To test the Binder environment locally before deploying:

```bash
# Option 1: Use repo2docker (what Binder uses)
pip install repo2docker
repo2docker --no-run .

# Option 2: Manually build the Conda environment
conda env create -f .binder/environment.yml
conda activate climakitae-binder
bash .binder/postBuild
```

## Adding New Dependencies

To add new packages to the Binder environment:

1. **Conda packages**: Add to `environment.yml` under `dependencies:`
   ```yaml
   dependencies:
     - package-name
   ```

2. **Pip-only packages**: Add to `postBuild` script:
   ```bash
   pip install package-name
   ```

3. **System dependencies**: Create `apt.txt` file with one package per line:
   ```
   libsomething
   build-essential
   ```

## More Information

- [Binder Documentation](https://mybinder.readthedocs.io/)
- [repo2docker Configuration](https://repo2docker.readthedocs.io/)
- [climakitae Repository](https://github.com/cal-adapt/climakitae)
