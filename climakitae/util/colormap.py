import matplotlib
import matplotlib.colors as mcolors
import numpy as np

from climakitae.core.paths import (
    AE_BLUE,
    AE_DIVERGING,
    AE_DIVERGING_R,
    AE_ORANGE,
    CATEGORICAL_CB,
)
from climakitae.util.utils import _package_file_path


def read_ae_colormap(
    cmap: str = "ae_orange", cmap_hex: bool = False
) -> matplotlib.colors.LinearSegmentedColormap | list:
    """Read in AE colormap by name

    Parameters
    ----------
    cmap : str
        one of ["ae_orange", "ae_diverging", "ae_blue", "ae_diverging_r", "categorical_cb"]
    cmap_hex : boolean
        return RGB or hex colors?

    Returns
    -------
    one of either

    cmap_data : matplotlib.colors.LinearSegmentedColormap
        used for matplotlib (if cmap_hex == False)
    cmap_data : list
        used for hvplot maps (if cmap_hex == True)

    """

    match cmap:
        case "ae_orange":
            cmap_data = AE_ORANGE
        case "ae_diverging":
            cmap_data = AE_DIVERGING
        case "ae_blue":
            cmap_data = AE_BLUE
        case "ae_diverging_r":
            cmap_data = AE_DIVERGING_R
        case "categorical_cb":
            cmap_data = CATEGORICAL_CB
        case _:
            raise ValueError(
                'cmap needs to be one of ["ae_orange", "ae_diverging", "ae_blue", "ae_diverging_r", "categorical_cb"]'
            )

    # Load text file
    cmap_np = np.loadtxt(_package_file_path(cmap_data), dtype=float)

    # RBG to hex
    if cmap_hex:
        cmap_data = [matplotlib.colors.rgb2hex(color) for color in cmap_np]
    else:
        cmap_data = mcolors.LinearSegmentedColormap.from_list(cmap, cmap_np, N=256)
    return cmap_data
