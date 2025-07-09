"""
Tools for validating user input
"""

import datetime
import difflib
import warnings
from collections.abc import Iterable
from typing import Any

import pandas as pd


def _get_closest_options(val, valid_options, cutoff=0.59):
    """If the user inputs a bad option, find the closest option from a list of valid options

    Parameters
    ----------
    val: str
        User input
    valid_options: list
        Valid options for that key from the catalog
    cutoff: a float in the range [0, 1]
        See difflib.get_close_matches
        Possibilities that don't score at least that similar to word are ignored.

    Returns
    -------
    closest_options: list or None
        List of best guesses, or None if nothing close is found
    """

    # Perhaps the user just capitalized it wrong?
    is_it_just_capitalized_wrong = [
        i for i in valid_options if val.lower() == i.lower()
    ]
    if len(is_it_just_capitalized_wrong) > 0:
        return is_it_just_capitalized_wrong

    # Perhaps the input is a substring of a valid option?
    is_it_a_substring = [i for i in valid_options if val.lower() in i.lower()]
    if len(is_it_a_substring) > 0:
        return is_it_a_substring

    # Use difflib package to make a guess for what the input might have been
    # For example, if they input "statistikal" instead of "Statistical", difflib will find "Statistical"
    # Change the cutoff to increase/decrease the flexibility of the function
    maybe_difflib_can_find_something = difflib.get_close_matches(
        val, valid_options, cutoff=cutoff
    )
    if len(maybe_difflib_can_find_something) > 0:
        return maybe_difflib_can_find_something

    return None


def _validate_experimental_id_param(
    value: list[str] | None,
    valid_experiment_ids: list[str],
) -> bool:
    """
    Validate the experiment_id parameter.

    This function checks if the provided value is valid for the experiment_id parameter.
    It performs a greedy match against a predefined list of valid experiment IDs,
    replacing partial matches with the full valid ID.

    Parameters
    ----------
    value : str | list[str] | None
        The experiment_id parameter to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.

    Notes
    -----
    Modifies input value in place if it contains a single string that matches
    multiple valid experiment IDs. If the value is a single string that does not
    match any valid experiment ID, it will attempt to find the closest match
    from the valid_experiment_ids list and issue a warning.
    """

    if value is None:
        return False  # No value provided

    if isinstance(value, str):
        # If a single string is provided, convert it to a list
        value = [value]

    match len(value):
        case 0:
            return False  # Empty list is not valid

        case 1:
            # Single value, check if it matches a valid experiment ID
            # If not, check if multiple experimental IDs match, in which case we re-set
            # value to the whole list of matching IDs
            # Otherwise, check for closest match, and issue a warning
            v = value[0]
            if v in valid_experiment_ids:
                return True  # Valid single experiment ID

            if any(v in valid for valid in valid_experiment_ids):
                # If any part of the value matches a valid experiment ID
                # We assume the user meant to greedy match multiple IDs
                # i.e. "hist" or "ssp" could match "historical" or all "ssp" experiments
                value.extend([valid for valid in valid_experiment_ids if v in valid])
                value.pop(0)  # Remove the original value
                return True

            # If no match, try to find the closest valid experiment ID
            closest = _get_closest_options(v, valid_experiment_ids)
            if closest:
                warnings.warn(
                    f"\n\nExperiment ID '{v}' not found."
                    f"\nDid you mean any of the following '{closest[0]}'?"
                )
            else:
                warnings.warn(
                    f"\n\nExperiment ID '{v}' not found."
                    "\nPlease check the available experiment IDs."
                )
            return False

        case _:
            # Multiple values, check each one
            ret = True
            for v in value:
                if v not in valid_experiment_ids:
                    closest = _get_closest_options(v, valid_experiment_ids)
                    if closest:
                        warnings.warn(
                            f"Experiment ID '{v}' not found. Did you mean '{closest[0]}'?"
                        )

                    ret = False
            return ret  # All values are valid


def _coerce_to_dates(value: Iterable[Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Coerce the values to date-like objects.

    Parameters
    ----------
    value : tuple
        The value to coerce.

    Returns
    -------
    tuple
        The coerced values.
    """
    ret = []
    for x in value:
        match x:
            case str() | int() | float() | datetime.date() | datetime.datetime():
                ret.append(pd.to_datetime(x))
            case pd.Timestamp():
                ret.append(x)
            case pd.DatetimeIndex():
                ret.append(x[0])
            case _:
                warnings.warn(
                    f"\n\nValue '{x}' is not a date-like object. "
                    "\nExpected a string, int, float, datetime.date, datetime.datetime, or pd.Timestamp."
                )
                return None
    return tuple(ret)
