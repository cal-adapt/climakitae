import matplotlib.colors as mcolors
import matplotlib
import numpy as np

from climakitae.core.paths import (
    ae_orange,
    ae_diverging,
    ae_blue,
    ae_diverging_r,
    categorical_cb,
)
from climakitae.util.utils import _package_file_path


def read_ae_colormap(cmap="ae_orange", cmap_hex=False):
    """Read in AE colormap by name

    Parameters
    ----------
    cmap: str
        one of ["ae_orange","ae_blue","ae_diverging"]
    cmap_hex: boolean
        return RGB or hex colors?

    Returns
    -------
    one of either

    cmap_data: matplotlib.colors.LinearSegmentedColormap
        used for matplotlib (if cmap_hex == False)
    cmap_data: list
        used for hvplot maps (if cmap_hex == True)

    """

    if cmap == "ae_orange":
        cmap_data = ae_orange
    elif cmap == "ae_diverging":
        cmap_data = ae_diverging
    elif cmap == "ae_blue":
        cmap_data = ae_blue
    elif cmap == "ae_diverging_r":
        cmap_data = ae_diverging_r
    elif cmap == "categorical_cb":
        cmap_data = categorical_cb

    # Load text file
    cmap_np = np.loadtxt(_package_file_path(cmap_data), dtype=float)

    # RBG to hex
    if cmap_hex:
        cmap_data = [matplotlib.colors.rgb2hex(color) for color in cmap_np]
    else:
        cmap_data = mcolors.LinearSegmentedColormap.from_list(cmap, cmap_np, N=256)
    return cmap_data
