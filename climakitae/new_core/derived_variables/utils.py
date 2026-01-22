"""Helper utilities for derived variable computations.

Provides runtime helpers for resolving parameter precedence for derived
variable calculations (explicit args -> dataset-level overrides -> global defaults).
"""

import logging
from typing import Optional

from climakitae.core.constants import DEFAULT_DEGREE_DAY_THRESHOLD_K
from climakitae.util.utils import f_to_k

logger = logging.getLogger(__name__)


def get_derived_threshold(
    ds,
    derived_var_name: Optional[str] = None,
    threshold_k: Optional[float] = None,
    threshold_c: Optional[float] = None,
    threshold_f: Optional[float] = None,
) -> float:
    """Resolve a degree-day threshold in Kelvin.

    Precedence:
    1. Explicit function arguments (`threshold_k`, `threshold_c`, `threshold_f`)
    2. Per-dataset attribute overrides (see dataset.attrs keys below)
    3. Global default `DEFAULT_DEGREE_DAY_THRESHOLD_K`

    Dataset attribute conventions supported (checked in order):
    - `derived_variable_overrides` : dict mapping derived var name -> params dict
      e.g. dataset.attrs['derived_variable_overrides'] = {'CDD_wrf': {'threshold_f': 75}}
    - Top-level attrs: `threshold_k`, `threshold_c`, `threshold_f`

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that may contain attribute-based overrides.
    derived_var_name : str, optional
        Name of the derived variable to look up per-variable overrides.
    threshold_k, threshold_c, threshold_f : optional
        Explicit threshold values supplied by caller. If provided, these take
        precedence.

    Returns
    -------
    float
        Threshold value in Kelvin.
    """

    # 1) explicit args
    if threshold_k is not None:
        return float(threshold_k)
    if threshold_c is not None:
        return float(threshold_c) + 273.15
    if threshold_f is not None:
        return float(f_to_k(threshold_f))

    # 2) dataset-level overrides (attrs)
    attrs = getattr(ds, "attrs", None) or {}

    # per-derived-variable overrides
    if derived_var_name:
        overrides = attrs.get("derived_variable_overrides") or attrs.get(
            "derived_variable_params"
        )
        if isinstance(overrides, dict) and derived_var_name in overrides:
            params = overrides[derived_var_name] or {}
            if params.get("threshold_k") is not None:
                return float(params.get("threshold_k"))
            if params.get("threshold_c") is not None:
                return float(params.get("threshold_c")) + 273.15
            if params.get("threshold_f") is not None:
                return float(f_to_k(params.get("threshold_f")))

    # top-level attrs
    if attrs.get("threshold_k") is not None:
        return float(attrs.get("threshold_k"))
    if attrs.get("threshold_c") is not None:
        return float(attrs.get("threshold_c")) + 273.15
    if attrs.get("threshold_f") is not None:
        return float(f_to_k(attrs.get("threshold_f")))

    # 3) global default
    logger.debug(
        "No threshold override found for '%s'; using default %s K",
        derived_var_name,
        DEFAULT_DEGREE_DAY_THRESHOLD_K,
    )
    return float(DEFAULT_DEGREE_DAY_THRESHOLD_K)
