"""
Tools for validating user input
"""

import difflib


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
