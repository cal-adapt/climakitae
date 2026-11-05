"""QA sheet validation for bias_adjust_model_to_station's training window.

Checks a requested (station, variable, center year) combination for
`bias_adjust_model_to_station` against a precomputed per-station/variable/year
data-quality sheet (`STATION_QA_CENTERED_FILE`). The center year is the
midpoint of `historical_slice` -- the training reference window the
processor bias-adjusts against -- since that's the historical station record
whose quality actually matters, independent of whatever future/output period
`time_slice` or `warming_level` requests.

The sheet's QA tests are split into two severity tiers:

- Error tier (`total-time-window`, `window_missing_nans`): an overall
  coverage problem severe enough to block the query.
- Warning tier (everything else -- the twelve `month_NN_nans` tests and the
  two diurnal tests): logged but non-blocking.

Each row in the sheet is one `(source-id, variable, test)` combination, and
each year column is a standalone 0/1 flag for *that one test* at that center
year. So for any failing test (either tier), the suggested alternative year
is the nearest year where *that specific row* is clear -- not a tier-wide
"all tests pass" condition. A center year can carry both an error and a
warning at once if different rows fail independently.

A station/variable/year the sheet doesn't cover at all is treated as
"unknown", not "failing" -- consistent with the rest of this validator
module's "can't validate, assume OK" convention.
"""

from __future__ import annotations

import calendar
import logging
from functools import lru_cache
from typing import Any

import pandas as pd

from climakitae.core.constants import PROC_KEY, UNSET
from climakitae.core.paths import STATION_QA_CENTERED_FILE
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.bias_adjust_model_to_station import (
    _VARIABLE_ID_TO_HDP_VARIABLE,
)
from climakitae.new_core.processors.processor_utils import (
    find_station_match,
    is_station_identifier,
)
from climakitae.util.utils import read_csv_file

# Module logger
logger = logging.getLogger(__name__)

# Matches BiasAdjustModelToStation's own default historical_slice.
_DEFAULT_HISTORICAL_SLICE = (1980, 2014)

# Overall-coverage tests block the query outright; every other test name in
# the sheet (per-month + diurnal) is a warning-only signal.
_ERROR_TESTS = frozenset({"total-time-window", "window_missing_nans"})

_TEST_LABELS = {
    "total-time-window": "overall time coverage",
    "window_missing_nans": "overall missing-data window",
    "diurnal_day_nans": "daytime coverage",
    "diurnal_night_nans": "nighttime coverage",
}


@lru_cache(maxsize=1)
def _load_qa_table() -> dict[tuple[str, str], dict[int, dict[str, bool]]]:
    """Load the QA sheet into {(source_id, variable): {year: {test: failed}}}.

    Returns
    -------
    dict[tuple[str, str], dict[int, dict[str, bool]]]
        Nested lookup: station/variable -> year -> test name -> whether that
        test is flagged as failed (True) for that year.
    """
    df = read_csv_file(STATION_QA_CENTERED_FILE)
    year_cols = [c for c in df.columns if c not in ("source-id", "variable", "test")]

    table: dict[tuple[str, str], dict[int, dict[str, bool]]] = {}
    for (source_id, variable), group in df.groupby(["source-id", "variable"]):
        year_map: dict[int, dict[str, bool]] = {int(y): {} for y in year_cols}
        for _, row in group.iterrows():
            for year in year_cols:
                year_map[int(year)][row["test"]] = bool(int(row[year]))
        table[(source_id, variable)] = year_map
    return table


def _year_qa_status(source_id: str, variable: str, year: int) -> tuple[list[str], list[str]]:
    """Determine which tests a station/variable/year fails, per tier.

    Parameters
    ----------
    source_id : str
        Airport-code-style station identifier (the sheet's `source-id`).
    variable : str
        HDP variable name (e.g. "tas", "tdps").
    year : int
        Calendar year to check.

    Returns
    -------
    tuple[list[str], list[str]]
        `(error_tests, warning_tests)`: the names of the failing tests in
        each tier. Both are empty if the sheet doesn't cover this
        station/variable/year at all, or if nothing failed.
    """
    flags = _load_qa_table().get((source_id, variable), {}).get(year)
    if flags is None:
        return [], []
    error_tests = sorted(t for t in _ERROR_TESTS if flags.get(t, False))
    warning_tests = sorted(
        t for t, failed in flags.items() if failed and t not in _ERROR_TESTS
    )
    return error_tests, warning_tests


def _nearest_passing_years(source_id: str, variable: str, test: str, year: int) -> list[int]:
    """Find the nearest year(s) around `year` where one specific test passes.

    Parameters
    ----------
    source_id : str
        Airport-code-style station identifier.
    variable : str
        HDP variable name.
    test : str
        The single test name (sheet row) to check -- not a tier.
    year : int
        Calendar year the caller requested (and where `test` failed).

    Returns
    -------
    list[int]
        Up to two years (the nearest before and after `year`) among the
        sheet's own coverage for this station/variable where `test` is not
        flagged. Empty if no such year exists.
    """
    year_map = _load_qa_table().get((source_id, variable), {})
    passing_years = sorted(y for y, flags in year_map.items() if not flags.get(test, False))
    before = max((y for y in passing_years if y < year), default=None)
    after = min((y for y in passing_years if y > year), default=None)
    return [y for y in (before, after) if y is not None]


def _describe_test(test: str) -> str:
    """Translate a test name into a human-readable label."""
    if test.startswith("month_"):
        return calendar.month_name[int(test.split("_")[1])]
    return _TEST_LABELS.get(test, test)


def _format_test_hint(source_id: str, variable: str, test: str, year: int) -> str:
    """Describe one failing test plus its own nearest passing year(s)."""
    nearby = _nearest_passing_years(source_id, variable, test, year)
    label = _describe_test(test)
    if nearby:
        return f"{label} (nearest clear year(s): {' or '.join(map(str, nearby))})"
    return f"{label} (no nearby year is clear of this)"


def resolve_station_to_source_id(
    identifier: str, legacy_stations_df: pd.DataFrame
) -> str | None:
    """Map a `stations` entry to the airport-code-style `source-id` used in
    the QA sheet.

    Parameters
    ----------
    identifier : str
        A single entry from `bias_adjust_model_to_station`'s `stations` list
        -- either a legacy airport code/name (e.g. "KSAN") or an HDP
        `station_id` (e.g. "ASOSAWOS_72290023188").
    legacy_stations_df : pd.DataFrame
        The legacy lookup table (`DataCatalog()["stations"]`), with `ID` and
        `station id` columns.

    Returns
    -------
    str | None
        The matching `source-id` (uppercased airport code), or None if the
        station has no resolvable airport-code equivalent.
    """
    if is_station_identifier(identifier):
        match = find_station_match(identifier, legacy_stations_df)
        return str(match.iloc[0]["ID"]).upper() if len(match) == 1 else None

    _, _, numeric_id = identifier.partition("_")
    try:
        numeric_id_int = int(numeric_id)
    except ValueError:
        return None
    match = legacy_stations_df[legacy_stations_df["station id"] == numeric_id_int]
    return str(match.iloc[0]["ID"]).upper() if len(match) == 1 else None


def _historical_slice_center_year(historical_slice: tuple[int, int]) -> int:
    """Compute the center year of a `historical_slice` training window.

    Parameters
    ----------
    historical_slice : tuple[int, int]
        `(start_year, end_year)` of the training reference window.

    Returns
    -------
    int
        The midpoint year, via integer floor division.
    """
    start, end = historical_slice
    return (start + end) // 2


def validate_station_qa_for_requested_years(query: dict[str, Any]) -> bool:
    """Validate requested (station, variable, center year) combinations
    against the QA sheet.

    No-op (returns True) unless `bias_adjust_model_to_station` is present
    with resolvable stations/variables. The center year checked is always
    the midpoint of `historical_slice` (the processor's own default,
    `(1980, 2014)`, if not given) -- independent of `time_slice` or
    `warming_level`, which only govern the output window that gets returned,
    not what's checked here.

    Parameters
    ----------
    query : dict[str, Any]
        Full query dictionary.

    Returns
    -------
    bool
        False if any station/variable fails an error-tier QA test at the
        center year (logged via `logger.error`, with a suggested
        alternative year). True otherwise -- including when only
        warning-tier tests fail (logged via `logger.warning`, but the query
        still proceeds).
    """
    bias_params = query.get(PROC_KEY, {}).get("bias_adjust_model_to_station")
    if not bias_params:
        return True

    stations = bias_params.get("stations", [])
    variable_ids = query.get("variable_id", UNSET)
    if variable_ids is UNSET:
        variable_ids = []
    elif not isinstance(variable_ids, list):
        variable_ids = [variable_ids]
    hdp_variables = {
        _VARIABLE_ID_TO_HDP_VARIABLE[v]
        for v in variable_ids
        if v in _VARIABLE_ID_TO_HDP_VARIABLE
    }
    if not stations or not hdp_variables:
        return True

    legacy_stations_df = DataCatalog()["stations"]
    source_ids = set()
    for s in stations:
        source_id = resolve_station_to_source_id(s, legacy_stations_df)
        if source_id is None:
            logger.info(
                "No QA-sheet coverage for station '%s' (no airport-code "
                "equivalent); skipping QA check for this station.",
                s,
            )
            continue
        source_ids.add(source_id)

    if not source_ids:
        return True

    historical_slice = bias_params.get("historical_slice", _DEFAULT_HISTORICAL_SLICE)
    year = _historical_slice_center_year(historical_slice)

    has_any_error = False
    for source_id in sorted(source_ids):
        for hdp_variable in sorted(hdp_variables):
            error_tests, warn_tests = _year_qa_status(source_id, hdp_variable, year)
            if error_tests:
                has_any_error = True
                hints = "; ".join(
                    _format_test_hint(source_id, hdp_variable, t, year)
                    for t in error_tests
                )
                logger.error(
                    "error: too much missing data overall for station "
                    "'%s', variable '%s', center year %d (historical_slice="
                    "%s): %s.",
                    source_id,
                    hdp_variable,
                    year,
                    historical_slice,
                    hints,
                )
            if warn_tests:
                hints = "; ".join(
                    _format_test_hint(source_id, hdp_variable, t, year)
                    for t in warn_tests
                )
                logger.warning(
                    "warning: station '%s' variable '%s' center year %d "
                    "(historical_slice=%s) has missing-data flags for %s; "
                    "data may be biased.",
                    source_id,
                    hdp_variable,
                    year,
                    historical_slice,
                    hints,
                )

    return not has_any_error
