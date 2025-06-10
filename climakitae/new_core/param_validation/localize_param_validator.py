"""
Validator for parameters provided to Localize Processor.
"""

from __future__ import annotations

import warnings
from typing import Any

from climakitae.core.constants import UNSET
from climakitae.core.paths import STATIONS_CSV_PATH
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)
from climakitae.new_core.param_validation.param_validation_tools import (
    _get_closest_options,
)
from climakitae.util.utils import read_csv_file

STATIONS_DF = read_csv_file(STATIONS_CSV_PATH)


@register_processor_validator("localize")
def validate_localize_param(
    value: str | list[str] | dict[str, Any],
) -> bool:
    """
    Validate the parameters provided to the Localize Processor.

    This function checks the value provided to the Localize Processor and ensures that it
    meets the expected criteria. Will raise a user warning and return false if the value
    is not valid.

    Parameters
    ----------
    value : Any
        The value to validate. It can be a string, list of strings, or a dict
    """
    ret = True
    match value:
        case str() | list():
            # convert to list
            stations = []
            if isinstance(value, list):
                stations = value
            else:
                stations.append(value)
            stations = [s.strip() for s in stations]

            # check if all values are valid station names or IDs
            available_stations = STATIONS_DF["station"].tolist()
            for s in stations:
                if len(s) == 4:
                    # this is a 4 letter airport code, find the station name
                    # and tell the user to replace it with the station name
                    mask = STATIONS_DF["station"].contains(s)
                    if not mask.any():
                        warnings.warn(
                            f"\n\nStation ID '{s}' not found in list of stations."
                        )
                        ret = False
                    elif sum(mask) > 1:
                        warnings.warn(f"\n\nMultiple stations found for ID '{s}'. ")
                        ret = False
                    else:
                        warnings.warn(
                            f"\n\n The query probably contains airport codes."
                            f"\nPlease replace {s} with {STATIONS_DF[mask]['station'].values[0]}"
                        )

                else:
                    if s not in available_stations:
                        closest = _get_closest_options(s, available_stations)
                        if closest:
                            warnings.warn(
                                f"\n\nStation '{s}' not found in list of stations. "
                                f"Did you mean any of these: {', '.join(closest)}?"
                            )
                        else:
                            msg = "\n\t".join(
                                station for station in sorted(available_stations)
                            )
                            warnings.warn(
                                f"\n\nStation '{s}' not found in list of stations."
                                f"\nAvailable stations are: \n {msg}\n"
                            )
                        ret = False
        case dict():
            # check if the dict has the correct keys
            stations = value.get("stations", UNSET)
            if stations is UNSET:
                ret = False
                warnings.warn(
                    "\n\nNo 'stations' key found in Localize Processor parameters."
                )
            else:
                ret = validate_localize_param(stations)

            bias_correction = value.get("bias_correction", UNSET)
            match bias_correction:
                case object():
                    pass
                case bool():
                    pass
                case _:
                    ret = False
                    warnings.warn(
                        "\n\nInvalid 'bias_correction' parameter type. "
                        "Expected a boolean or no key at all."
                    )

            method = value.get("method", UNSET)
            if method is not UNSET:
                if method != "quantile_delta_mapping":
                    ret = False
                    warnings.warn(
                        "\n\nInvalid 'method' parameter value. "
                        "Expected 'quantile_delta_mapping', or no key at all."
                    )

            window = value.get("window", UNSET)
            if window is UNSET:
                pass
            elif not isinstance(window, int) or window < 1:
                ret = False
                warnings.warn(
                    "\n\nInvalid 'window' parameter type. "
                    "Expected a positive integer greater than 1 or no key at all."
                )

            quantiles = value.get("nquantiles", UNSET)
            if quantiles is UNSET:
                pass
            elif not isinstance(quantiles, int) or quantiles < 1 or quantiles > 99:
                ret = False
                warnings.warn(
                    "\n\nInvalid 'nquantiles' parameter type. "
                    "Expected a positive integer between 1 and 100 or no key at all."
                )

        case _:
            warnings.warn(
                "\n\nInvalid parameter type for Localize Processor. "
                "Expected a string, list of strings, or a dict."
            )
            ret = False

    return ret
