"""ClimakitAE MCP Server — generates climakitae code, never executes it.

A lightweight MCP server that helps AI agents produce correct climakitae
Python code by providing:
- Searchable variable/catalog metadata (from static CSVs)
- Parameter validation against the real catalog
- Code generation for ClimateData queries
- Documentation resources for API patterns and domain knowledge

No climakitae imports. No data connections. No AWS credentials needed.
"""

import csv
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ── Paths ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "climakitae" / "data"
DOCS_DIR = ROOT / ".github" / "instructions"

# ── Server ─────────────────────────────────────────────────────────────

mcp = FastMCP(
    "climakitae-assistant",
    instructions=(
        "You are a Cal-Adapt climate data code assistant. You help users "
        "write correct Python code using the climakitae library. You NEVER "
        "execute climate data queries — you only generate code for the user "
        "to run.\n\n"
        "WORKFLOW:\n"
        "1. Understand what the user wants (variable, region, timeframe, analysis)\n"
        "2. Use lookup tools to verify parameter names and availability\n"
        "3. Consult resource docs for correct API patterns\n"
        "4. Return well-commented Python code the user can copy and run\n\n"
        "RULES:\n"
        "- Always use: from climakitae.new_core.user_interface import ClimateData\n"
        "- WRF variables: t2max, t2min, t2, prec, dew_point (dynamical)\n"
        "- LOCA2 variables: tasmax, tasmin, pr (statistical, CMIP6 naming)\n"
        "- Minimum required: catalog, variable, table_id, grid_label\n"
        "- Never use the legacy core.data_interface\n"
        "- Include imports, comments, and explain what the code does"
    ),
)

# ── Static data (loaded once at startup) ───────────────────────────────


def _load_variable_catalog() -> list[dict]:
    path = DATA_DIR / "variable_descriptions.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_catalog_summary() -> dict:
    path = DATA_DIR / "catalogs.csv"
    if not path.exists():
        return {"summary": {}, "variables_by_activity": {}}

    fields = {
        "catalogs": "catalog",
        "activity_ids": "activity_id",
        "variable_ids": "variable_id",
        "table_ids": "table_id",
        "grid_labels": "grid_label",
        "experiment_ids": "experiment_id",
        "institution_ids": "institution_id",
        "source_ids": "source_id",
    }
    sets = {k: set() for k in fields}
    var_by_activity: dict[str, set] = {}

    with open(path) as f:
        for row in csv.DictReader(f):
            for key, col in fields.items():
                val = row.get(col, "")
                if val:
                    sets[key].add(val)
            aid = row.get("activity_id", "")
            vid = row.get("variable_id", "")
            if aid and vid:
                var_by_activity.setdefault(aid, set()).add(vid)

    return {
        "summary": {k: sorted(v) for k, v in sets.items()},
        "variables_by_activity": {k: sorted(v) for k, v in var_by_activity.items()},
    }


VARIABLES = _load_variable_catalog()
CATALOG = _load_catalog_summary()

# The CSV uses internal names ("data", "renewables") but the user-facing
# ClimateData API uses friendlier names ("cadcat", "renewable energy generation").
# Map both directions so validation accepts what users actually type.
_CATALOG_ALIASES = {
    "cadcat": "data",
    "renewable energy generation": "renewables",
}
_CATALOG_ALIASES_REV = {v: k for k, v in _CATALOG_ALIASES.items()}

# Add user-facing names to the summary so validation accepts them
for user_name, csv_name in _CATALOG_ALIASES.items():
    if csv_name in CATALOG.get("summary", {}).get("catalogs", []):
        CATALOG["summary"]["catalogs"].append(user_name)
        CATALOG["summary"]["catalogs"] = sorted(set(CATALOG["summary"]["catalogs"]))


# ═══════════════════════════════════════════════════════════════════════
# RESOURCES — documentation grounding
# ═══════════════════════════════════════════════════════════════════════


def _read_doc(name: str) -> str:
    path = DOCS_DIR / name
    if path.exists():
        return path.read_text()
    return f"(doc not found: {name})"


@mcp.resource("climakitae://docs/notebook-guide")
def notebook_guide() -> str:
    """Complete reference for ClimateData queries, processors, and examples."""
    return _read_doc("notebook-analysis.instructions.md")


@mcp.resource("climakitae://docs/domain-knowledge")
def domain_knowledge() -> str:
    """Climate data hierarchy, catalogs, WRF vs LOCA2, grid labels, architecture."""
    return _read_doc("climakitae.instructions.md")


@mcp.resource("climakitae://docs/processors")
def processor_docs() -> str:
    """Processor reference: clip, time_slice, warming_level, export, bias correction."""
    return _read_doc("processors.instructions.md")


# ═══════════════════════════════════════════════════════════════════════
# TOOLS — catalog lookups (metadata only, no data)
# ═══════════════════════════════════════════════════════════════════════


@mcp.tool()
def lookup_variables(activity_id: str = "", search: str = "") -> str:
    """Look up available climate variables.

    Parameters
    ----------
    activity_id : str
        Filter by downscaling method ("WRF" or "LOCA2").
        If empty, shows all.
    search : str
        Filter variable names/descriptions (e.g., "temperature", "wind").
    """
    valid_vars = None
    if activity_id:
        valid_vars = set(CATALOG.get("variables_by_activity", {}).get(activity_id, []))

    results = []
    for var in VARIABLES:
        vid = var.get("variable_id", "")
        if valid_vars is not None and vid not in valid_vars:
            continue
        if search:
            haystack = (
                f"{vid} {var.get('display_name', '')} "
                f"{var.get('extended_description', '')}"
            ).lower()
            if search.lower() not in haystack:
                continue
        results.append(
            {
                "variable_id": vid,
                "display_name": var.get("display_name", ""),
                "unit": var.get("unit", ""),
                "timescale": var.get("timescale", ""),
                "downscaling_method": var.get("downscaling_method", ""),
                "description": var.get("extended_description", ""),
            }
        )

    if not results:
        return "No matching variables found."
    return json.dumps(results, indent=2)


@mcp.tool()
def lookup_options(option_type: str) -> str:
    """List all available values for a query parameter.

    Parameters
    ----------
    option_type : str
        One of: catalogs, activity_ids, variable_ids, table_ids,
        grid_labels, experiment_ids, institution_ids, source_ids
    """
    summary = CATALOG.get("summary", {})
    if option_type not in summary:
        return json.dumps(
            {"error": f"Unknown '{option_type}'", "available": list(summary.keys())}
        )
    return json.dumps(summary[option_type], indent=2)


@mcp.tool()
def validate_query(
    catalog: str,
    variable: str,
    activity_id: str = "",
    table_id: str = "",
    grid_label: str = "",
) -> str:
    """Check whether a set of query parameters is valid.

    Parameters
    ----------
    catalog : str
        Data catalog name
    variable : str
        Variable ID
    activity_id : str
        Downscaling method
    table_id : str
        Temporal resolution
    grid_label : str
        Spatial resolution
    """
    summary = CATALOG.get("summary", {})
    issues = []

    checks = [
        ("catalog", catalog, "catalogs"),
        ("variable", variable, "variable_ids"),
        ("activity_id", activity_id, "activity_ids"),
        ("table_id", table_id, "table_ids"),
        ("grid_label", grid_label, "grid_labels"),
    ]
    for label, val, key in checks:
        if val and val not in summary.get(key, []):
            issues.append(f"Unknown {label} '{val}'. Valid: {summary.get(key, [])}")

    if activity_id and variable:
        avail = CATALOG.get("variables_by_activity", {}).get(activity_id, [])
        if variable not in avail:
            issues.append(
                f"'{variable}' not available for {activity_id}. " f"Try: {avail[:15]}"
            )

    if issues:
        return json.dumps({"valid": False, "issues": issues}, indent=2)
    return json.dumps({"valid": True})


@mcp.tool()
def list_derived_functions() -> str:
    """List derived-variable and climate-index functions in climakitae.

    Returns function names, signatures, required inputs, and import paths.
    These are functions the USER applies to data they have already fetched.
    """
    fns = [
        {
            "name": "compute_hdd_cdd",
            "import": "from climakitae.tools.derived_variables import compute_hdd_cdd",
            "signature": "compute_hdd_cdd(t2, hdd_threshold, cdd_threshold)",
            "inputs": "Air Temperature at 2m (°F)",
            "returns": "(HDD, CDD) DataArrays",
        },
        {
            "name": "compute_hdh_cdh",
            "import": "from climakitae.tools.derived_variables import compute_hdh_cdh",
            "signature": "compute_hdh_cdh(t2, hdh_threshold, cdh_threshold)",
            "inputs": "Air Temperature at 2m (°F)",
            "returns": "(HDH, CDH) DataArrays",
        },
        {
            "name": "compute_dewpointtemp",
            "import": "from climakitae.tools.derived_variables import compute_dewpointtemp",
            "signature": "compute_dewpointtemp(t2, rh)",
            "inputs": "Temperature (K), Relative Humidity (%)",
            "returns": "Dew point temperature DataArray",
        },
        {
            "name": "compute_relative_humidity",
            "import": "from climakitae.tools.derived_variables import compute_relative_humidity",
            "signature": "compute_relative_humidity(t2, q2, psfc)",
            "inputs": "Temperature (K), Specific Humidity (kg/kg), Surface Pressure (Pa)",
            "returns": "Relative Humidity DataArray (0-100%)",
        },
        {
            "name": "compute_wind_mag",
            "import": "from climakitae.tools.derived_variables import compute_wind_mag",
            "signature": "compute_wind_mag(u, v)",
            "inputs": "U10 and V10 wind components",
            "returns": "Wind speed DataArray",
        },
        {
            "name": "compute_wind_dir",
            "import": "from climakitae.tools.derived_variables import compute_wind_dir",
            "signature": "compute_wind_dir(u, v)",
            "inputs": "U10 and V10 wind components",
            "returns": "Wind direction DataArray (degrees)",
        },
        {
            "name": "effective_temp",
            "import": "from climakitae.tools.indices import effective_temp",
            "signature": "effective_temp(T)",
            "inputs": "Daily air temperature (any units)",
            "returns": "Effective temperature DataArray",
        },
        {
            "name": "noaa_heat_index",
            "import": "from climakitae.tools.indices import noaa_heat_index",
            "signature": "noaa_heat_index(T, RH)",
            "inputs": "Temperature (°F), Relative Humidity (0-100%)",
            "returns": "Heat Index DataArray (°F)",
        },
        {
            "name": "fosberg_fire_index",
            "import": "from climakitae.tools.indices import fosberg_fire_index",
            "signature": "fosberg_fire_index(t2_F, rh_percent, windspeed_mph)",
            "inputs": "Temperature (°F), Relative Humidity (%), Wind Speed (mph)",
            "returns": "FFWI DataArray (0-100)",
        },
    ]
    return json.dumps(fns, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# TOOL — code generation
# ═══════════════════════════════════════════════════════════════════════


@mcp.tool()
def generate_code(
    catalog: str,
    variable: str,
    table_id: str,
    grid_label: str,
    activity_id: str = "",
    experiment_id: str = "",
    institution_id: str = "",
    source_id: str = "",
    time_slice_start: str = "",
    time_slice_end: str = "",
    clip: str = "",
    clip_points: str = "",
    warming_levels: str = "",
    warming_level_window: int = 0,
    export_filename: str = "",
    export_format: str = "NetCDF",
) -> str:
    """Generate ready-to-run Python code for a ClimateData query.

    Parameters
    ----------
    catalog : str
        "cadcat" or "renewable energy generation"
    variable : str
        Climate variable (e.g., "tasmax", "t2max")
    table_id : str
        "1hr", "day", "mon"
    grid_label : str
        "d01" (45km), "d02" (9km), "d03" (3km)
    activity_id : str
        "WRF" or "LOCA2"
    experiment_id : str
        SSP scenario(s), comma-separated (e.g., "historical,ssp245").
        NOT for warming levels — use warming_levels param instead.
    institution_id : str
        Institution filter
    source_id : str
        Climate model filter
    time_slice_start : str
        Start date (e.g., "2030-01-01")
    time_slice_end : str
        End date
    clip : str
        Boundary name(s). Use semicolons for multiple:
        "Los Angeles" or "Los Angeles;Sacramento;San Francisco"
    clip_points : str
        Lat/lon as "lat1,lon1;lat2,lon2"
    warming_levels : str
        Global Warming Levels as comma-separated floats (e.g., "1.5,2.0,3.0").
        These are temperature thresholds in °C, NOT experiment/scenario IDs.
    warming_level_window : int
        Years around GWL crossing (default 0 uses library default of 15).
    export_filename : str
        Adds export processor if provided
    export_format : str
        "NetCDF", "CSV", or "Zarr" (default: "NetCDF")
    """
    lines = [
        "from climakitae.new_core.user_interface import ClimateData",
        "",
        "cd = ClimateData()",
        "",
        "data = (cd",
        f'    .catalog("{catalog}")',
    ]

    if activity_id:
        lines.append(f'    .activity_id("{activity_id}")')
    if institution_id:
        lines.append(f'    .institution_id("{institution_id}")')
    if source_id:
        lines.append(f'    .source_id("{source_id}")')
    if experiment_id:
        if "," in experiment_id:
            exps = [e.strip() for e in experiment_id.split(",")]
            lines.append(f"    .experiment_id({exps})")
        else:
            lines.append(f'    .experiment_id("{experiment_id}")')

    lines.append(f'    .table_id("{table_id}")')
    lines.append(f'    .grid_label("{grid_label}")')
    lines.append(f'    .variable("{variable}")')

    # Build processes dict
    procs: list[str] = []
    if time_slice_start and time_slice_end:
        procs.append(
            f'        "time_slice": ("{time_slice_start}", "{time_slice_end}")'
        )
    if clip:
        if ";" in clip:
            boundaries = [b.strip() for b in clip.split(";")]
            boundary_list = ", ".join(f'"{b}"' for b in boundaries)
            procs.append(f'        "clip": [{boundary_list}]')
        else:
            procs.append(f'        "clip": "{clip}"')
    if clip_points:
        pts = []
        for pt in clip_points.split(";"):
            lat, lon = pt.strip().split(",")
            pts.append(f"({lat.strip()}, {lon.strip()})")
        val = pts[0] if len(pts) == 1 else f"[{', '.join(pts)}]"
        procs.append(f'        "clip": {val}')
    if warming_levels:
        wl = [float(x.strip()) for x in warming_levels.split(",")]
        wl_dict = f'"warming_levels": {wl}'
        if warming_level_window:
            wl_dict += f', "warming_level_window": {warming_level_window}'
        procs.append(f'        "warming_level": {{{wl_dict}}}')
    if export_filename:
        procs.append(
            f'        "export": {{"filename": "{export_filename}", '
            f'"file_format": "{export_format}"}}'
        )

    if procs:
        lines.append("    .processes({")
        lines.append(",\n".join(procs))
        lines.append("    })")

    lines.append("    .get())")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run()
