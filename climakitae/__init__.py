from climakitae.core.data_load import load
from climakitae.core.data_export import export, remove_zarr

try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version  # type: ignore[no-redef]

try:
    __version__ = _version("climakitae")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__all__ = (
    # Methods
    "load",
    "export",
    "remove_zarr",
    # Constants
    "__version__",
)
