"""Tools for validating user input"""

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
    val : str
        User input
    valid_options : list
        Valid options for that key from the catalog
    cutoff : a float in the range [0, 1]
        See difflib.get_close_matches
        Possibilities that don't score at least that similar to word are ignored.

    Returns
    -------
    closest_options : list or None
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
    """Validate the experiment_id parameter against a list of valid experiment IDs.

    This function checks if the provided experiment_id value(s) are valid by comparing
    them against a predefined list of valid experiment IDs. It supports partial matching
    for convenience and provides helpful error messages with suggestions for invalid inputs.

    Parameters
    ----------
    value : list[str] | None
        The experiment_id parameter to validate. Can be None, a single string converted
        to a list internally, or a list of strings representing experiment IDs.
    valid_experiment_ids : list[str]
        A list of valid experiment ID strings to validate against.

    Returns
    -------
    bool :
        True if all provided experiment IDs are valid or can be matched, False otherwise.
        Returns False for None or empty inputs.

    Warnings
    --------
    UserWarning
        Issued when an experiment ID is not found, either with suggestions for
        the closest matches or a general message to check available IDs.

    Notes
    -----
    This function modifies the input value list in place under certain conditions:

    - For single string inputs that partially match multiple valid experiment IDs,
      the function performs a greedy match, replacing the partial string with all
      matching full experiment IDs (e.g., "ssp" might expand to ["ssp126", "ssp245", "ssp585"]).

    - For invalid experiment IDs, the function attempts to find the closest match
      using fuzzy matching and issues warnings with suggestions.

    - For multiple values, each is validated individually, and warnings are issued
      for any invalid entries.
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
                    f"\nDid you mean any of the following '{closest[0]}'?",
                    UserWarning,
                    stacklevel=999,
                )
            else:
                warnings.warn(
                    f"\n\nExperiment ID '{v}' not found."
                    "\nPlease check the available experiment IDs.",
                    UserWarning,
                    stacklevel=999,
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
                            f"Experiment ID '{v}' not found. Did you mean '{closest[0]}'?",
                            UserWarning,
                            stacklevel=999,
                        )

                    ret = False
            return ret  # All values are valid


def _coerce_to_dates(value: Iterable[Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Coerce the values to date-like objects.

    Parameters
    ----------
    value : Iterable[Any]
        An iterable containing exactly 2 date-like objects to coerce.
        Each element can be a string, int, float, datetime.date,
        datetime.datetime, pd.Timestamp, or pd.DatetimeIndex.

        For integer/float values in the range 1900-2200, they are treated as years:
        - First position: Start of year (January 1st)
        - Second position: End of year (December 31st)

        Other numeric values are treated as Unix timestamps.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        A tuple containing exactly 2 pd.Timestamp objects.

    Raises
    ------
    ValueError
        If the iterable doesn't contain exactly 2 elements or if any
        element cannot be coerced to a date-like object.

    Examples
    --------
    >>> _coerce_to_dates([2020, 2021])
    (Timestamp('2020-01-01 00:00:00'), Timestamp('2021-12-31 00:00:00'))

    >>> _coerce_to_dates(["2020-01-01", "2021-06-15"])
    (Timestamp('2020-01-01 00:00:00'), Timestamp('2021-06-15 00:00:00'))

    """
    # Convert to list to check length and iterate
    value_list = list(value)

    if len(value_list) != 2:
        raise ValueError(f"Expected exactly 2 date-like values, got {len(value_list)}")

    ret = []
    for i, x in enumerate(value_list):
        try:
            match x:
                case str() | datetime.date() | datetime.datetime():
                    ret.append(pd.to_datetime(x))
                case int() | float() if 1900 <= x <= 2200:
                    # Handle year integers/floats - interpret as year ranges
                    if i == 0:
                        # First position: start of year (Jan 1st)
                        ret.append(pd.Timestamp(year=int(x), month=1, day=1))
                    else:
                        # Second position: end of year (Dec 31st)
                        ret.append(pd.Timestamp(year=int(x), month=12, day=31))
                case int() | float():
                    # Handle other numeric values (Unix timestamps, etc.)
                    ret.append(pd.to_datetime(x))
                case pd.Timestamp():
                    ret.append(x)
                case pd.DatetimeIndex():
                    if len(x) == 0:
                        raise ValueError(f"Empty DatetimeIndex at position {i}")
                    ret.append(x[0])
                case _:
                    raise ValueError(
                        f"Value '{x}' at position {i} is not a date-like object. "
                        f"Expected a string, int, float, datetime.date, datetime.datetime, "
                        f"pd.Timestamp, or pd.DatetimeIndex."
                    )
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
            raise ValueError(
                f"Cannot coerce value '{x}' at position {i} to datetime: {e}"
            ) from e

    return tuple(ret)
