import warnings

# Configure warning filters BEFORE importing submodules so that warnings
# emitted during import of dependencies are also suppressed. By default,
# ignore all warnings, then re-enable default handling for warnings
# originating from this package. This ensures noisy third-party warnings
# (e.g., pandas/dateutil parsing fallbacks, tqdm, pkg_resources deprecation)
# don't clutter user output while keeping climakitae package warnings visible.
warnings.filterwarnings("ignore")
warnings.filterwarnings("default", module=r"^climakitae(\.|$)")

# Suppress specific common third-party warnings that are particularly noisy
warnings.filterwarnings("ignore", message="Could not infer format")  # pandas
warnings.filterwarnings("ignore", message="IProgress not found")  # tqdm
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")  # intake_esm
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")  # xarray

from climakitae._version import __version__  # noqa: E402

# Import submodules after configuring filters (noqa: E402 suppresses linter warnings)
from climakitae.core.data_export import export, remove_zarr  # noqa: E402
from climakitae.core.data_load import load  # noqa: E402
from climakitae.new_core.user_interface import ClimateData  # noqa: E402

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
