import warnings

from climakitae.core.data_export import export, remove_zarr
from climakitae.core.data_load import load
from climakitae._version import __version__
from climakitae.new_core.user_interface import ClimateData

# By default, ignore warnings coming from modules outside this package so that
# noisy third-party warnings (for example pandas/dateutil parsing fallbacks)
# don't appear for users who set logging to WARNING. Keep warnings that
# originate from the `climakitae` package visible. Tests or applications can
# still override this behavior by calling `warnings.filterwarnings(...)`.
#
# We use a negative lookahead to match module names that do NOT start with
# 'climakitae' and ignore those warnings; modules beginning with
# 'climakitae' will not be ignored by this rule. Match full module names.
warnings.filterwarnings("ignore", module=r"^(?!climakitae).*")

__all__ = (
    # Methods
    "load",
    "export",
    "remove_zarr",
    # Classes
    "ClimateData",
    # Constants
    "__version__",
)
