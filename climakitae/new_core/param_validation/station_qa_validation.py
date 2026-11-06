"""QA sheet validation for bias_adjust_model_to_station's training window.

Checks a requested (station, variable, center year) combination for
`bias_adjust_model_to_station` against a precomputed per-station/variable/year
data-quality sheet (`WEATHER_STATION_QA_FILE`). The center year is the
midpoint of `historical_slice`, which is the training reference window the
processor bias-adjusts against.

`total-time-window` is the sole blocking check, a `1` means a hard
error. Every other row (`window_missing_nans`, the twelve `month_NN_nans`
tests, and the two diurnal tests) is non-blocking: when `total-time-window`
fails, the other flagged rows are surfaced as the *reasons* for that
failure; when it passes, any of those other rows still flagged are
surfaced as warnings instead. Either way, the suggested alternative is
`historical_slice`-shaped: the nearest year(s) where `total-time-window`
itself passes, re-expressed as a same-width range centered on that year.

A station/variable/year that the sheet doesn't cover at all is treated
as "unknown", not "failing".
"""

from __future__ import annotations

import calendar
import logging
from functools import lru_cache
from typing import Any, Callable

import pandas as pd

from climakitae.core.constants import PROC_KEY, UNSET
from climakitae.core.paths import WEATHER_STATION_QA_FILE
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.bias_adjust_model_to_station import (
    _VARIABLE_ID_TO_HDP_VARIABLE,
)
from climakitae.new_core.processors.processor_utils import (
    find_station_match,
    is_station_identifier,
)
from climakitae.util.utils import read_csv_file

logger = logging.getLogger(__name__)

# Matches BiasAdjustModelToStation's own default historical_slice.
_DEFAULT_HISTORICAL_SLICE = (1980, 2014)

# The sole blocking check. Every other row in the sheet is informational --
# a "reason" when this one fails, a plain warning when it doesn't.
_ERROR_TEST = "total-time-window"

_TEST_LABELS = {
    "total-time-window": "overall time coverage",
    "window_missing_nans": "overall missing-data window",
    "diurnal_day_nans": "daytime coverage",
    "diurnal_night_nans": "nighttime coverage",
    "diurnal_night_winter_nans": "winter nighttime coverage",
}

# Rows whose name ends in "_nans" are missing-value checks and get grouped
# under one "too many missing station values" phrase in messages. A future
# test type that isn't about missing values won't end in this suffix, and
# is described on its own instead.
_MISSING_VALUE_TEST_SUFFIX = "_nans"


@lru_cache(maxsize=1)
def _load_qa_table() -> dict[tuple[str, str], dict[int, dict[str, bool]]]:
    """Load the sheet into {(source_id, variable): {year: {test: failed}}}."""
    df = read_csv_file(WEATHER_STATION_QA_FILE)
    year_cols = [c for c in df.columns if c not in ("source-id", "variable", "test")]

    table: dict[tuple[str, str], dict[int, dict[str, bool]]] = {}
    for (source_id, variable), group in df.groupby(["source-id", "variable"]):
        year_map: dict[int, dict[str, bool]] = {int(y): {} for y in year_cols}
        for _, row in group.iterrows():
            for year in year_cols:
                year_map[int(year)][row["test"]] = bool(int(row[year]))
        table[(source_id, variable)] = year_map
    return table


def _year_qa_status(source_id: str, variable: str, year: int) -> tuple[bool, list[str]]:
    """`(has_error, other_tests)` for one station/variable/year: whether
    `total-time-window` is flagged, and every other flagged row at that
    year. Both empty/False if the sheet doesn't cover this combination."""
    flags = _load_qa_table().get((source_id, variable), {}).get(year)
    if flags is None:
        return False, []
    has_error = flags.get(_ERROR_TEST, False)
    other_tests = sorted(
        t for t, failed in flags.items() if failed and t != _ERROR_TEST
    )
    return has_error, other_tests


def _nearest_passing_years(
    source_id: str, variable: str, test: str, year: int
) -> list[int]:
    """Nearest year(s) before/after `year`, among the sheet's own coverage
    for this station/variable, where `test` (a single row, not a tier) is
    not flagged. Empty if no such year exists."""
    year_map = _load_qa_table().get((source_id, variable), {})
    passing_years = sorted(
        y for y, flags in year_map.items() if not flags.get(test, False)
    )
    before = max((y for y in passing_years if y < year), default=None)
    after = min((y for y in passing_years if y > year), default=None)
    return [y for y in (before, after) if y is not None]


def _describe_test(test: str) -> str:
    """Human-readable label for a test name, e.g. "month_07_nans" -> "July"."""
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


def _group_missing_value_tests(tests: list[str]) -> tuple[list[str], list[str]]:
    """Split failing test names into (missing-value tests, other tests)."""
    missing = [t for t in tests if t.endswith(_MISSING_VALUE_TEST_SUFFIX)]
    other = [t for t in tests if not t.endswith(_MISSING_VALUE_TEST_SUFFIX)]
    return missing, other


def _describe_failing_tests(tests: list[str], describe: Callable[[str], str]) -> str:
    """Build a human-readable clause for a list of failing tests, grouping
    missing-value rows under one "too many missing station values" phrase
    (since that's the common, expected cause) and describing any other
    kind of failing row on its own (it may have a different cause).

    `describe` is called per test name to render its label, e.g.
    `_describe_test` (plain) or a closure adding a nearest-year hint.
    """
    missing, other = _group_missing_value_tests(tests)
    clauses = []
    if missing:
        clauses.append(
            "Too many missing station values for "
            + ", ".join(describe(t) for t in missing)
        )
    clauses.extend(describe(t) for t in other)
    return "; ".join(clauses)


def resolve_station_to_source_id(
    identifier: str, legacy_stations_df: pd.DataFrame
) -> str | None:
    """Map a `stations` entry (airport code/name, or HDP `station_id`) to
    the airport-code-style `source-id` used in the QA sheet, via the legacy
    lookup table (`DataCatalog()["stations"]`). None if unmappable."""
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
    """Midpoint year of a `(start_year, end_year)` training window."""
    start, end = historical_slice
    return (start + end) // 2


def _recenter_historical_slice(
    year: int, historical_slice: tuple[int, int]
) -> tuple[int, int]:
    """Re-apply `historical_slice`'s exact width, centered on `year`
    instead. Offsets are computed separately (not assumed symmetric) so an
    odd total span round-trips to the same width."""
    start, end = historical_slice
    center = _historical_slice_center_year(historical_slice)
    return year - (center - start), year + (end - center)


def _check_source_variable(
    source_id: str, variable: str, year: int, historical_slice: tuple[int, int]
) -> bool:
    """Check one station/variable at the center year; log an error or
    warning as appropriate. Returns True if it's an error (blocks)."""
    has_error, other_tests = _year_qa_status(source_id, variable, year)
    if not has_error:
        if other_tests:
            hints = _describe_failing_tests(
                other_tests,
                lambda t: _format_test_hint(source_id, variable, t, year),
            )
            logger.warning(
                "warning: station '%s' variable '%s' center year %d "
                "(historical_slice=%s): %s; resulting localized data may be biased.",
                source_id,
                variable,
                year,
                historical_slice,
                hints,
            )
        return False

    reasons = (
        _describe_failing_tests(other_tests, _describe_test)
        if other_tests
        else "no other checks are flagged for this time range"
    )
    nearby_years = _nearest_passing_years(source_id, variable, _ERROR_TEST, year)
    if nearby_years:
        ranges = (_recenter_historical_slice(y, historical_slice) for y in nearby_years)
        suggestion = "Closest usable time ranges: " + " or ".join(
            f"{s}-{e}" for s, e in ranges
        )
    else:
        suggestion = (
            "No usable time range was found nearby in the QA sheet for this "
            "station/variable."
        )
    slice_start, slice_end = historical_slice
    logger.error(
        "Error: Too much missing data for station '%s', variable '%s', "
        "from %d-%d. Related issues: %s. %s",
        source_id,
        variable,
        slice_start,
        slice_end,
        reasons,
        suggestion,
    )
    return True


def validate_station_request_qa(query: dict[str, Any]) -> bool:
    """Validate a `bias_adjust_model_to_station` request's center year(s)
    against the QA sheet.

    No-op (True) unless `bias_adjust_model_to_station` is present with
    resolvable stations/variables. The center year is always the midpoint
    of `historical_slice` (default `(1980, 2014)`) -- independent of
    `time_slice`/`warming_level`, which only govern the output window.

    Returns False if any station/variable fails `total-time-window` at
    that year (logged via `logger.error`); True otherwise, including when
    only non-blocking rows are flagged (logged via `logger.warning`).
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

    # A list, not a generator, so every combination is checked (and logged)
    # rather than stopping at the first error.
    results = [
        _check_source_variable(source_id, hdp_variable, year, historical_slice)
        for source_id in sorted(source_ids)
        for hdp_variable in sorted(hdp_variables)
    ]
    return not any(results)
