"""Builtin derived variables for ClimakitAE.

This package contains pre-registered derived variables for common climate
calculations. These are automatically registered when the derived_variables
module is imported.

Modules
-------
wind
    Wind speed and direction derived from U/V components.
humidity
    Relative humidity, dew point, and specific humidity.
temperature
    Heat index, wind chill, and degree days.

Notes
-----
All builtin derived variables are registered at import time using the
@register_derived decorator. Users can override these by registering
their own function with the same variable name.

"""

# Import all builtin modules to trigger registration
from climakitae.new_core.derived_variables.builtin import humidity, temperature, wind

__all__ = ["wind", "humidity", "temperature"]
