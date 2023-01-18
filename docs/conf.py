# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Climakitae"
copyright = "2022, Cal-Adapt Analytics Engine"
author = "Cal-Adapt Analytics Engine Team"
# The full version, including alpha/beta/rc tags
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon", 
    "sphinx.ext.intersphinx", 
    "sphinx.ext.extlinks"
]

# Generate autosummary of package
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"
# Theme options are theme-specific and customize the look and feel of a theme
# further. For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "repository_url": "https://github.com/cal-adapt/climakitae",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
}
# The name of an image file (relative to this directory) to place at the top of
# the sidebar.
html_logo = "_static/cae-logo.svg"
html_favicon = "_static/cae-logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Napoleon configurations
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_type_aliases = {
"matplotlib colormap name": ":doc:`matplotlib colormap name <matplotlib:gallery/color/colormap_reference>`", 
'array-like': ':term:`array-like <array_like>`'
} 

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "iris": ("https://scitools-iris.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "cftime": ("https://unidata.github.io/cftime", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest", None),
    "sparse": ("https://sparse.pydata.org/en/latest/", None),
    "sparse": ("https://sparse.pydata.org/en/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None)
}
