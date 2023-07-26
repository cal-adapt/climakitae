from climakitae.core.data_interface import DataInterface
from climakitae.ui.select import Select
from climakitae.core.data_loaders import load

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
    # Classes
    "DataInterface",
    "Select",
    # Methods
    "load"
    # Constants
    "__version__",
)
