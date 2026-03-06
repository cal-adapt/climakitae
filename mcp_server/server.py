"""ClimakitAE MCP Server — generates climakitae code, never executes it.

A lightweight MCP server that helps AI agents produce correct climakitae
Python code by providing:
- Searchable variable/catalog metadata (from static CSVs)
- Parameter validation against the real catalog
- Boundary, station, unit, and warming-level lookups
- Code generation for ClimateData queries (including processors)
- Documentation resources for API patterns and domain knowledge

No climakitae imports. No data connections. No AWS credentials needed.
"""

import csv
import html as _html
import json
import re
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "climakitae" / "data"
DOCS_DIR = ROOT / ".github" / "instructions"
SITE_PAGES = ROOT.parent / "cae-website" / "src" / "pages"
SITE_DATA = ROOT.parent / "cae-website" / "src" / "data"

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


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _load_variable_catalog() -> list[dict]:
    return _load_csv_rows(DATA_DIR / "variable_descriptions.csv")


def _load_catalog_summary() -> dict:
    path = DATA_DIR / "catalogs.csv"
    rows = _load_csv_rows(path)
    if not rows:
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

    for row in rows:
        for key, col in fields.items():
            val = row.get(col, "")
            if val:
                sets[key].add(val)

        activity = row.get("activity_id", "")
        variable = row.get("variable_id", "")
        if activity and variable:
            var_by_activity.setdefault(activity, set()).add(variable)

    return {
        "summary": {k: sorted(v) for k, v in sets.items()},
        "variables_by_activity": {k: sorted(v) for k, v in var_by_activity.items()},
    }


def _load_stations() -> list[dict]:
    return _load_csv_rows(DATA_DIR / "hadisd_stations.csv")


def _load_gwl_timing() -> list[dict]:
    rows = _load_csv_rows(DATA_DIR / "gwl_timing_table.csv")
    normalized = []
    for row in rows:
        row_copy = dict(row)
        warming_level = row_copy.pop("", None)
        if warming_level is not None:
            row_copy["warming_level"] = warming_level
        normalized.append(row_copy)
    return normalized


VARIABLES = _load_variable_catalog()
CATALOG = _load_catalog_summary()
STATIONS = _load_stations()
GWL_TIMING = _load_gwl_timing()

# CSV contains internal names while user-facing API accepts friendlier names
_CATALOG_ALIASES = {
    "cadcat": "data",
    "renewable energy generation": "renewables",
}
_CATALOG_ALIASES_REV = {v: k for k, v in _CATALOG_ALIASES.items()}

# Add user-facing catalog aliases to summary
for user_name, csv_name in _CATALOG_ALIASES.items():
    if csv_name in CATALOG.get("summary", {}).get("catalogs", []):
        CATALOG["summary"]["catalogs"].append(user_name)
        CATALOG["summary"]["catalogs"] = sorted(set(CATALOG["summary"]["catalogs"]))


UNIT_OPTIONS = {
    "K": ["K", "degC", "degF"],
    "degF": ["K", "degC", "degF"],
    "degC": ["K", "degC", "degF"],
    "hPa": ["Pa", "hPa", "mb", "inHg"],
    "Pa": ["Pa", "hPa", "mb", "inHg"],
    "m/s": ["m/s", "mph", "knots"],
    "m s-1": ["m s-1", "mph", "knots"],
    "[0 to 100]": ["[0 to 100]", "fraction"],
    "mm": ["mm", "inches"],
    "mm/d": ["mm/d", "inches/d"],
    "mm/h": ["mm/h", "inches/h"],
    "kg/kg": ["kg/kg", "g/kg"],
    "kg kg-1": ["kg kg-1", "g kg-1"],
    "kg m-2 s-1": ["kg m-2 s-1", "mm", "inches"],
    "g/kg": ["g/kg", "kg/kg"],
}


BOUNDARY_TYPES = {
    "us_states": {
        "description": "US western states (11 states)",
        "examples": ["CA", "OR", "WA", "NV", "AZ", "UT"],
    },
    "ca_counties": {
        "description": "California counties",
        "examples": ["Alameda County", "Los Angeles County", "Humboldt County"],
    },
    "ca_watersheds": {
        "description": "California HUC8 watersheds",
        "examples": ["San Francisco Bay", "San Pablo Bay"],
    },
    "ious_pous": {
        "description": "California electric load serving entities",
        "examples": [
            "Pacific Gas & Electric",
            "Southern California Edison",
            "San Diego Gas & Electric",
        ],
    },
    "forecast_zones": {
        "description": "California electricity demand forecast zones",
        "examples": [],
    },
    "electric_balancing_areas": {
        "description": "California electric balancing authority areas",
        "examples": [],
    },
    "ca_census_tracts": {
        "description": "California census tracts (GEOID)",
        "examples": ["06071010300"],
    },
}


PROCESSOR_OPTIONS = {
    "concat": {
        "description": "Concatenate historical + scenario data",
        "options": {"value": ["time", "sim"]},
    },
    "filter_unadjusted_models": {
        "description": "Include/exclude unadjusted WRF models",
        "options": {"value": ["yes", "no"]},
    },
    "drop_leap_days": {
        "description": "Remove Feb 29 entries",
        "options": {"value": ["yes", "no"]},
    },
    "time_slice": {
        "description": "Subset by date/year range",
        "options": {"value": "(start, end)"},
    },
    "warming_level": {
        "description": "Subset by global warming levels",
        "options": {
            "warming_levels": [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
            "warming_level_window": "int years",
        },
    },
    "clip": {
        "description": "Clip to boundary / point / bbox",
        "options": {
            "boundary": "string or list[str]",
            "point": "(lat, lon)",
            "points": "[(lat, lon), ...]",
            "bbox": "((lat_min, lat_max), (lon_min, lon_max))",
            "separated": [True, False],
        },
    },
    "convert_units": {
        "description": "Convert variable units",
        "options": {"target_unit": "see lookup_unit_conversions"},
    },
    "metric_calc": {
        "description": "Calculate metrics, percentiles, or 1-in-X return values",
        "options": {
            "metric": ["min", "max", "mean", "median", "sum"],
            "percentiles": "list[0-100]",
            "dim": "string or list[string]",
            "one_in_x.distribution": ["gev", "genpareto", "gamma"],
            "one_in_x.extremes_type": ["max", "min"],
        },
    },
    "bias_adjust_model_to_station": {
        "description": "Bias-correct WRF data to station observations",
        "options": {
            "stations": "list[str]",
            "historical_slice": "(start_year, end_year)",
        },
    },
    "export": {
        "description": "Export data to file",
        "options": {
            "filename": "str",
            "file_format": ["NetCDF", "CSV", "Zarr"],
            "separated": [True, False],
            "location_based_naming": [True, False],
        },
    },
}


# ── Guidance RAG index (loaded once at startup from cae-website) ──────

_GUIDANCE_CHUNKS: list[dict] = []
_GUIDANCE_VECTORIZER: TfidfVectorizer | None = None
_GUIDANCE_MATRIX = None


def _strip_html(text: str) -> str:
    """Remove HTML tags, collapse whitespace, and decode entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return _html.unescape(text).strip()


def _extract_astro_chunks() -> list[dict]:
    """Parse .astro files, split by heading level, return text chunks."""
    chunks = []
    for fpath in sorted(SITE_PAGES.rglob("*.astro")):
        raw = fpath.read_text(errors="ignore")
        m = re.search(r"<Markdown>(.*?)</Markdown>", raw, re.DOTALL)
        if not m:
            continue
        md = m.group(1)

        # Build a clean URL from the file path
        rel = fpath.relative_to(SITE_PAGES).with_suffix("")
        url = "/" + rel.as_posix().replace("/index", "")

        # Split on h1/h2/h3 headings, keeping the heading text as chunk title.
        # Astro template indents headings 4 spaces, so allow optional leading whitespace.
        parts = re.split(r"\n[ \t]*(#{1,3} [^\n]+)", md)
        current_heading = fpath.stem.replace("_", " ").replace("-", " ")

        for part in parts:
            part = part.strip()
            if re.match(r"^#{1,3} ", part):
                current_heading = re.sub(r"^#{1,3} ", "", part).strip()
                # Strip Astro template vars like {title}
                current_heading = re.sub(r"\{[^}]+\}", "", current_heading).strip()
            elif len(part) > 120:
                clean = _strip_html(part)
                if len(clean) > 120:
                    chunks.append(
                        {
                            "url": url,
                            "title": current_heading,
                            "text": clean,
                            "source": "guidance-page",
                        }
                    )
    return chunks


def _extract_accordion_chunks() -> list[dict]:
    """Load Q&A accordion JSON files (decision-making guidance)."""
    chunks = []
    for fpath in sorted(SITE_DATA.glob("*accordion*.json")):
        try:
            items = json.loads(fpath.read_text())
        except Exception:
            continue
        for item in items:
            text = _strip_html(item.get("description", ""))
            title = item.get("title", "")
            if len(text) > 120:
                chunks.append(
                    {
                        "url": "/guidance/using_in_decision_making",
                        "title": title,
                        "text": text,
                        # accordion titles ARE questions — they are our synthetic questions
                        "source": "decision-making-faq",
                    }
                )
    return chunks


def _build_guidance_index() -> None:
    """Build TF-IDF index over all guidance content at startup."""
    global _GUIDANCE_CHUNKS, _GUIDANCE_VECTORIZER, _GUIDANCE_MATRIX
    if not SITE_PAGES.exists():
        return

    chunks = _extract_astro_chunks() + _extract_accordion_chunks()
    if not chunks:
        return

    # Index text: title doubled (for term weight) + full text.
    # For accordion items the title IS a question, so repeating it biases
    # retrieval toward question-shaped user queries.
    index_texts = [
        f"{c['title']} {c['title']} {c['text']}" for c in chunks
    ]

    vec = TfidfVectorizer(
        stop_words="english",
        max_features=10_000,
        ngram_range=(1, 2),
    )
    mat = vec.fit_transform(index_texts)

    _GUIDANCE_CHUNKS = chunks
    _GUIDANCE_VECTORIZER = vec
    _GUIDANCE_MATRIX = mat


_build_guidance_index()


# ── Notebook code example index ────────────────────────────────────────
#
# Directories to search for .ipynb / .py example files.
# Paths are resolved relative to the workspace root (ROOT.parent).
# Add new sources here — missing paths are silently skipped.

_NOTEBOOK_DIRS: list[Path] = [
    ROOT.parent / "scratch" / "developers" / "neil" / "new_core_demos",
    ROOT.parent / "cae-notebooks" / "data-access",
    ROOT.parent / "cae-notebooks" / "analysis",
    ROOT.parent / "cae-notebooks" / "climate-profiles",
    ROOT.parent / "cae-notebooks" / "collaborative",
]

_EXAMPLE_CHUNKS: list[dict] = []
_EXAMPLE_VECTORIZER: TfidfVectorizer | None = None
_EXAMPLE_MATRIX = None

_IMPORT_ONLY_RE = re.compile(
    r"^(%(?:load_ext|autoreload|matplotlib)|import |from \w+ import |\s*#|\s*$)",
    re.MULTILINE,
)


def _is_setup_cell(code: str) -> bool:
    """Return True if a code cell is pure imports/magic with no real logic."""
    lines = [l for l in code.strip().splitlines() if l.strip()]
    if not lines:
        return True
    meaningful = [l for l in lines if not _IMPORT_ONLY_RE.match(l)]
    return len(meaningful) == 0


def _extract_notebook_chunks(path: Path) -> list[dict]:
    """Extract (description, code) pairs from a single .ipynb file.

    Strategy:
    - Each markdown cell becomes a description.
    - The consecutive code cells that follow (up to the next markdown) form
      the paired code block.
    - Whole-notebook title + full code is also emitted as a single summary
      chunk so users can ask "show me a complete X example".
    - Setup-only code cells (pure imports/magic) are excluded from pairs
      but included in the whole-notebook chunk.
    """
    try:
        nb = json.loads(path.read_text())
    except Exception:
        return []

    cells = nb.get("cells", [])
    notebook_name = path.stem
    source_url = f"scratch/new_core_demos/{path.name}" if "neil" in str(path) else f"cae-notebooks/{path.relative_to(ROOT.parent / 'cae-notebooks')}"

    chunks = []
    current_heading = notebook_name.replace("_", " ")
    notebook_title = current_heading
    pending_code: list[str] = []

    def flush(description: str, code_blocks: list[str]) -> None:
        code = "\n\n".join(code_blocks).strip()
        if code and not _is_setup_cell(code) and len(description) > 20:
            chunks.append(
                {
                    "notebook": notebook_name,
                    "section": description[:120],
                    "description": description,
                    "code": code,
                    "source": source_url,
                }
            )

    for cell in cells:
        ctype = cell.get("cell_type", "")
        src = "".join(cell.get("source", [])).strip()
        if not src:
            continue

        if ctype == "markdown":
            # Flush any accumulated code under the previous heading
            if pending_code:
                flush(current_heading, pending_code)
                pending_code = []
            # Strip markdown formatting for the heading/description
            clean = re.sub(r"^#{1,4}\s+", "", src, flags=re.MULTILINE)
            clean = re.sub(r"```.*?```", "", clean, flags=re.DOTALL).strip()
            if not clean:
                continue
            # Use first line as heading, full text as description
            first_line = clean.splitlines()[0].strip()
            if re.match(r"^#\s", src):  # top-level h1 = notebook title
                notebook_title = first_line
            current_heading = first_line
            # Store full markdown as description instead of just heading
            current_desc = clean
        elif ctype == "code":
            if ctype == "code" and "markdown" not in [
                c.get("cell_type") for c in cells[: cells.index(cell)]
            ]:
                # Before any markdown — treat as setup, skip pairing
                pending_code.append(src)
            else:
                pending_code.append(src)
            # Update current_desc binding (Python scoping workaround)
            # description used comes from last markdown seen

    # Flush final section
    if pending_code:
        flush(current_heading, pending_code)

    # Emit a whole-notebook summary chunk
    all_code = "\n\n".join(
        "".join(c.get("source", [])).strip()
        for c in cells
        if c.get("cell_type") == "code" and "".join(c.get("source", [])).strip()
    )
    if all_code:
        chunks.append(
            {
                "notebook": notebook_name,
                "section": f"Complete example: {notebook_title}",
                "description": notebook_title,
                "code": all_code[:4000],  # cap to avoid huge TF-IDF vectors
                "source": source_url,
            }
        )

    return chunks


def _extract_py_chunks(path: Path) -> list[dict]:
    """Extract a single chunk from a .py script file."""
    try:
        src = path.read_text()
    except Exception:
        return []
    if len(src.strip()) < 50:
        return []

    # Use first docstring or comment block as description
    m = re.search(r'^"""(.*?)"""', src, re.DOTALL)
    if not m:
        m = re.match(r"((?:^#[^\n]*\n)+)", src, re.MULTILINE)
    description = m.group(1).strip() if m else path.stem.replace("_", " ")

    source_url = f"scratch/new_core_demos/{path.name}" if "neil" in str(path) else str(path.name)
    return [
        {
            "notebook": path.stem,
            "section": path.stem.replace("_", " "),
            "description": description,
            "code": src[:4000],
            "source": source_url,
        }
    ]


def _build_example_index() -> None:
    """Build TF-IDF index over all notebook code examples at startup."""
    global _EXAMPLE_CHUNKS, _EXAMPLE_VECTORIZER, _EXAMPLE_MATRIX

    chunks: list[dict] = []
    for nb_dir in _NOTEBOOK_DIRS:
        if not nb_dir.exists():
            continue
        for fpath in sorted(nb_dir.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in str(fpath):
                continue
            chunks.extend(_extract_notebook_chunks(fpath))
        for fpath in sorted(nb_dir.rglob("*.py")):
            chunks.extend(_extract_py_chunks(fpath))

    if not chunks:
        return

    # Index on: section heading (doubled) + first 500 chars of description
    index_texts = [
        f"{c['section']} {c['section']} {c['description'][:500]}"
        for c in chunks
    ]

    vec = TfidfVectorizer(
        stop_words="english",
        max_features=10_000,
        ngram_range=(1, 2),
    )
    mat = vec.fit_transform(index_texts)

    _EXAMPLE_CHUNKS = chunks
    _EXAMPLE_VECTORIZER = vec
    _EXAMPLE_MATRIX = mat


_build_example_index()


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


@mcp.resource("climakitae://docs/data-access")
def data_access_docs() -> str:
    """Data access patterns and boundaries."""
    return _read_doc("data-access.instructions.md")


@mcp.resource("climakitae://docs/param-validation")
def param_validation_docs() -> str:
    """Parameter validation patterns."""
    return _read_doc("param-validation.instructions.md")


@mcp.resource("climakitae://docs/new-core")
def new_core_docs() -> str:
    """New core architecture patterns and examples."""
    return _read_doc("new-core.instructions.md")


@mcp.resource("climakitae://data/stations")
def stations_resource() -> str:
    """HadISD station metadata used for clipping and bias correction."""
    return json.dumps(STATIONS, indent=2)


@mcp.resource("climakitae://data/gwl-timing")
def gwl_timing_resource() -> str:
    """Global warming level timing table by SSP scenario."""
    return json.dumps(GWL_TIMING, indent=2)


@mcp.resource("climakitae://examples/common-workflows")
def common_workflows() -> str:
    """Copy/paste prompt and code examples for common climate workflows."""
    return _COMMON_WORKFLOWS


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
        variable_id = var.get("variable_id", "")

        if valid_vars is not None and variable_id not in valid_vars:
            continue

        if search:
            haystack = (
                f"{variable_id} {var.get('display_name', '')} "
                f"{var.get('extended_description', '')}"
            ).lower()
            if search.lower() not in haystack:
                continue

        results.append(
            {
                "variable_id": variable_id,
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
def lookup_processors() -> str:
    """List available processors and key parameter options."""
    return json.dumps(PROCESSOR_OPTIONS, indent=2)


@mcp.tool()
def lookup_boundaries(boundary_type: str = "") -> str:
    """List available boundary types (and examples) for clip processor.

    Parameters
    ----------
    boundary_type : str
        Optional specific type. One of:
        us_states, ca_counties, ca_watersheds, ious_pous,
        forecast_zones, electric_balancing_areas, ca_census_tracts
    """
    if not boundary_type:
        return json.dumps(BOUNDARY_TYPES, indent=2)

    if boundary_type not in BOUNDARY_TYPES:
        return json.dumps(
            {
                "error": f"Unknown boundary_type '{boundary_type}'",
                "available": list(BOUNDARY_TYPES.keys()),
            },
            indent=2,
        )

    return json.dumps(
        {
            "boundary_type": boundary_type,
            **BOUNDARY_TYPES[boundary_type],
            "note": (
                "For a complete runtime list, use "
                f'ClimateData().show_boundary_options("{boundary_type}").'
            ),
        },
        indent=2,
    )


@mcp.tool()
def lookup_stations(search: str = "") -> str:
    """Look up weather stations for clip and bias_adjust_model_to_station.

    Parameters
    ----------
    search : str
        Optional station ID/city/state search (e.g., "KSAC", "Sacramento", "CA").
    """
    results = []

    for station in STATIONS:
        station_id = station.get("ID", "")
        station_name = station.get("station", "")
        city = station.get("city", "")
        state = station.get("state", "")

        if search:
            haystack = f"{station_id} {station_name} {city} {state}".lower()
            if search.lower() not in haystack:
                continue

        results.append(
            {
                "ID": station_id,
                "station": station_name,
                "city": city,
                "state": state,
                "lat": station.get("LAT_Y", ""),
                "lon": station.get("LON_X", ""),
                "elevation_m": station.get("elevation", ""),
            }
        )

    if not results:
        return "No matching stations found."
    return json.dumps(results, indent=2)


@mcp.tool()
def lookup_unit_conversions(source_unit: str = "") -> str:
    """List valid unit conversions for convert_units processor.

    Parameters
    ----------
    source_unit : str
        Optional source unit (e.g., "K", "mm", "m/s").
        If omitted, returns full map.
    """
    if not source_unit:
        return json.dumps(UNIT_OPTIONS, indent=2)

    if source_unit not in UNIT_OPTIONS:
        return json.dumps(
            {
                "error": f"Unknown source_unit '{source_unit}'",
                "available": list(UNIT_OPTIONS.keys()),
            },
            indent=2,
        )

    return json.dumps(
        {
            "source_unit": source_unit,
            "targets": UNIT_OPTIONS[source_unit],
        },
        indent=2,
    )


@mcp.tool()
def lookup_warming_level_timing(warming_level: str = "") -> str:
    """Look up timing of warming-level thresholds across SSP scenarios.

    Parameters
    ----------
    warming_level : str
        Optional level in °C (e.g., "1.5", "2.0", "3.0").
        If omitted, returns full table.
    """
    if not GWL_TIMING:
        return "No warming-level timing data found."

    if not warming_level:
        return json.dumps(GWL_TIMING, indent=2)

    filtered = [
        row
        for row in GWL_TIMING
        if str(row.get("warming_level", "")).strip() == str(warming_level).strip()
    ]

    if not filtered:
        return json.dumps(
            {
                "error": f"No timing row found for warming_level='{warming_level}'",
                "available": [row.get("warming_level", "") for row in GWL_TIMING],
            },
            indent=2,
        )

    return json.dumps(filtered[0], indent=2)


@mcp.tool()
def search_guidance(query: str, top_k: int = 4) -> str:
    """Search Cal-Adapt scientific guidance for context to inform code generation.

    Call this BEFORE generate_code when the user's question involves any of:
    - Which dataset to use (WRF vs LOCA2-Hybrid)
    - Which models or ensemble members to select
    - Bias correction choices and caveats (a-priori vs post-hoc)
    - Temporal scale choices (hourly vs daily, sampling windows)
    - Spatial scale choices (resolution, aggregation tradeoffs)
    - Reference periods and global warming levels
    - How to handle extreme events or return periods (1-in-X)
    - General best practices for interpreting climate projections

    Use the returned guidance to:
    1. Inform parameter choices in generate_code (e.g., LOCA2 for 1-in-X analyses)
    2. Add explanatory comments citing the methodology to generated code
    3. Provide the user with caveats or recommendations alongside code

    Parameters
    ----------
    query : str
        Natural-language question, e.g. "should I use WRF or LOCA2 for
        extreme heat return period analysis?"
    top_k : int
        Number of guidance sections to return (default 4, max 8).
    """
    if _GUIDANCE_VECTORIZER is None or _GUIDANCE_MATRIX is None:
        return (
            "Guidance index not available. "
            "Ensure the cae-website repo exists at "
            f"{SITE_PAGES} (currently missing)."
        )

    top_k = min(max(top_k, 1), 8)
    q_vec = _GUIDANCE_VECTORIZER.transform([query])
    scores = cosine_similarity(q_vec, _GUIDANCE_MATRIX).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k * 2]  # over-fetch to deduplicate

    results = []
    seen_titles: set[str] = set()
    for i in top_idx:
        if len(results) >= top_k:
            break
        if scores[i] < 0.04:
            break
        c = _GUIDANCE_CHUNKS[i]
        # Deduplicate chunks with identical titles
        if c["title"] in seen_titles:
            continue
        seen_titles.add(c["title"])
        # Truncate at a word boundary
        snippet = c["text"][:800]
        if len(c["text"]) > 800:
            snippet = snippet.rsplit(" ", 1)[0] + "..."
        source_label = "FAQ" if c["source"] == "decision-making-faq" else "Guidance"
        results.append(
            f"### {c['title']}\n"
            f"*{source_label} — "
            f"[cal-adapt.org{c['url']}](https://analytics.cal-adapt.org{c['url']})*\n\n"
            f"{snippet}"
        )

    if not results:
        return "No relevant guidance found for that query."

    return (
        "## Relevant Cal-Adapt Guidance\n\n"
        + "\n\n---\n\n".join(results)
        + "\n\n---\n\n*Incorporate the above into code comments and parameter "
        "choices when calling generate_code.*"
    )


@mcp.tool()
def search_code_examples(query: str, top_k: int = 3) -> str:
    """Search real climakitae notebook examples for working code snippets.

    Call this when the user asks for an example of a specific pattern and
    generate_code alone may miss nuance — e.g.:
    - Complete end-to-end analyses (extreme heat, degree days, LA county)
    - Plotting patterns (violin, ridgeline, spatial maps with warming levels)
    - Derived variables usage (HDD/CDD, custom thresholds, factory pattern)
    - Export patterns (NetCDF, CSV, Zarr, location-based naming)
    - Clip patterns (multi-boundary, station, census tract, separated mode)
    - Bias correction / station localization workflows

    Returns matched code sections from real notebooks with source file info.
    Use returned code as reference when writing generate_code output or as a
    direct snippet to show the user.

    Parameters
    ----------
    query : str
        Description of what you're looking for, e.g.
        "extreme heat exceedance days warming level violin plot" or
        "export data to NetCDF with location-based filenames".
    top_k : int
        Number of examples to return (default 3, max 6).
    """
    if _EXAMPLE_VECTORIZER is None or _EXAMPLE_MATRIX is None:
        return (
            "Code example index not available. "
            f"Ensure at least one notebook directory exists under {ROOT.parent}."
        )

    top_k = min(max(top_k, 1), 6)
    q_vec = _EXAMPLE_VECTORIZER.transform([query])
    scores = cosine_similarity(q_vec, _EXAMPLE_MATRIX).flatten()
    top_idx = np.argsort(scores)[::-1][: top_k * 2]

    results = []
    seen: set[str] = set()
    for i in top_idx:
        if len(results) >= top_k:
            break
        if scores[i] < 0.04:
            break
        c = _EXAMPLE_CHUNKS[i]
        key = f"{c['notebook']}::{c['section']}"
        if key in seen:
            continue
        seen.add(key)

        # Truncate code at a reasonable length for LLM context
        code = c["code"]
        if len(code) > 1500:
            code = code[:1500].rsplit("\n", 1)[0] + "\n# ... (truncated)"

        results.append(
            f"### {c['section']}\n"
            f"*Source: `{c['source']}`*\n\n"
            f"```python\n{code}\n```"
        )

    if not results:
        return "No matching code examples found."

    return (
        "## Matching Code Examples\n\n"
        + "\n\n---\n\n".join(results)
        + "\n\n---\n\n*Adapt these snippets as needed. "
        "All examples use the ClimateData new_core interface.*"
    )


@mcp.tool()
def validate_query(
    catalog: str,
    variable: str,
    activity_id: str = "",
    table_id: str = "",
    grid_label: str = "",
    experiment_id: str = "",
    institution_id: str = "",
    processors: str = "",
) -> str:
    """Validate core ClimateData query parameters and processor names.

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
    experiment_id : str
        SSP experiment id
    institution_id : str
        Institution id
    processors : str
        Comma-separated processor names (e.g., "warming_level,clip,export")
    """
    summary = CATALOG.get("summary", {})
    issues = []
    warnings = []

    checks = [
        ("catalog", catalog, "catalogs"),
        ("variable", variable, "variable_ids"),
        ("activity_id", activity_id, "activity_ids"),
        ("table_id", table_id, "table_ids"),
        ("grid_label", grid_label, "grid_labels"),
        ("experiment_id", experiment_id, "experiment_ids"),
        ("institution_id", institution_id, "institution_ids"),
    ]

    for label, value, key in checks:
        if value and value not in summary.get(key, []):
            issues.append(f"Unknown {label} '{value}'. Valid: {summary.get(key, [])}")

    if activity_id and variable:
        available = CATALOG.get("variables_by_activity", {}).get(activity_id, [])
        if variable not in available:
            issues.append(
                f"'{variable}' not available for {activity_id}. Try: {available[:15]}"
            )

    if processors:
        requested = [x.strip() for x in processors.split(",") if x.strip()]
        valid = set(PROCESSOR_OPTIONS.keys())
        for proc in requested:
            if proc not in valid:
                issues.append(
                    f"Unknown processor '{proc}'. Valid: {sorted(PROCESSOR_OPTIONS.keys())}"
                )

        if "warming_level" in requested and experiment_id:
            warnings.append(
                "warming_level processor usually does not need experiment_id; it uses "
                "all simulations that reach the requested warming level"
            )

        if "bias_adjust_model_to_station" in requested:
            if activity_id and activity_id != "WRF":
                issues.append("bias_adjust_model_to_station is only supported for WRF")
            if table_id and table_id != "1hr":
                warnings.append("bias_adjust_model_to_station typically uses table_id='1hr'")
            if variable and variable != "t2":
                warnings.append("bias_adjust_model_to_station currently supports variable='t2'")

    payload = {"valid": len(issues) == 0}
    if issues:
        payload["issues"] = issues
    if warnings:
        payload["warnings"] = warnings

    return json.dumps(payload, indent=2)


@mcp.tool()
def list_derived_functions() -> str:
    """List derived-variable and climate-index functions in climakitae.

    Returns function names, signatures, required inputs, and import paths.
    These are functions the USER applies to data they have already fetched.
    """
    functions = [
        {
            "name": "compute_hdd_cdd",
            "import": "from climakitae.tools.derived_variables import compute_hdd_cdd",
            "signature": "compute_hdd_cdd(t2, hdd_threshold, cdd_threshold)",
            "inputs": "Air temperature at 2m (°F)",
            "returns": "(HDD, CDD) DataArrays",
        },
        {
            "name": "compute_hdh_cdh",
            "import": "from climakitae.tools.derived_variables import compute_hdh_cdh",
            "signature": "compute_hdh_cdh(t2, hdh_threshold, cdh_threshold)",
            "inputs": "Air temperature at 2m (°F)",
            "returns": "(HDH, CDH) DataArrays",
        },
        {
            "name": "compute_dewpointtemp",
            "import": "from climakitae.tools.derived_variables import compute_dewpointtemp",
            "signature": "compute_dewpointtemp(t2, rh)",
            "inputs": "Temperature (K), Relative humidity (%)",
            "returns": "Dew point temperature DataArray",
        },
        {
            "name": "compute_relative_humidity",
            "import": "from climakitae.tools.derived_variables import compute_relative_humidity",
            "signature": "compute_relative_humidity(t2, q2, psfc)",
            "inputs": "Temperature (K), specific humidity (kg/kg), surface pressure (Pa)",
            "returns": "Relative humidity DataArray",
        },
        {
            "name": "compute_specific_humidity",
            "import": "from climakitae.tools.derived_variables import compute_specific_humidity",
            "signature": "compute_specific_humidity(tdps, pressure, name='q2_derived')",
            "inputs": "Dew-point temperature (K), pressure (Pa)",
            "returns": "Specific humidity DataArray (g/kg)",
        },
        {
            "name": "compute_sea_level_pressure",
            "import": "from climakitae.tools.derived_variables import compute_sea_level_pressure",
            "signature": "compute_sea_level_pressure(psfc, t2, q2, elevation, lapse_rate=0.0065, average_t2=True)",
            "inputs": "Surface pressure (Pa), temperature (K), mixing ratio, elevation (m)",
            "returns": "Sea level pressure DataArray (Pa)",
        },
        {
            "name": "compute_geostrophic_wind",
            "import": "from climakitae.tools.derived_variables import compute_geostrophic_wind",
            "signature": 'compute_geostrophic_wind(geopotential_height, gridlabel="d01")',
            "inputs": "Geopotential height data on WRF grid",
            "returns": "Tuple of (U, V) geostrophic wind DataArrays",
        },
        {
            "name": "compute_wind_mag",
            "import": "from climakitae.tools.derived_variables import compute_wind_mag",
            "signature": "compute_wind_mag(u, v)",
            "inputs": "U and V wind components",
            "returns": "Wind speed DataArray",
        },
        {
            "name": "compute_wind_dir",
            "import": "from climakitae.tools.derived_variables import compute_wind_dir",
            "signature": "compute_wind_dir(u, v)",
            "inputs": "U and V wind components",
            "returns": "Wind direction DataArray",
        },
        {
            "name": "effective_temp",
            "import": "from climakitae.tools.indices import effective_temp",
            "signature": "effective_temp(T)",
            "inputs": "Daily air temperature",
            "returns": "Effective temperature DataArray",
        },
        {
            "name": "noaa_heat_index",
            "import": "from climakitae.tools.indices import noaa_heat_index",
            "signature": "noaa_heat_index(T, RH)",
            "inputs": "Temperature (°F), relative humidity (0-100)",
            "returns": "Heat Index DataArray (°F)",
        },
        {
            "name": "fosberg_fire_index",
            "import": "from climakitae.tools.indices import fosberg_fire_index",
            "signature": "fosberg_fire_index(t2_F, rh_percent, windspeed_mph)",
            "inputs": "Temperature (°F), relative humidity (%), wind speed (mph)",
            "returns": "Fosberg fire weather index DataArray",
        },
    ]
    return json.dumps(functions, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# TOOL — code generation
# ═══════════════════════════════════════════════════════════════════════


def _split_csv(value: str, cast=None) -> list:
    if not value:
        return []
    items = [x.strip() for x in value.split(",") if x.strip()]
    if cast is None:
        return items
    return [cast(x) for x in items]


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
    clip_bbox: str = "",
    clip_separated: bool = False,
    warming_levels: str = "",
    warming_level_window: int = 0,
    convert_units: str = "",
    metric: str = "",
    metric_dims: str = "",
    percentiles: str = "",
    one_in_x_return_periods: str = "",
    one_in_x_distribution: str = "gev",
    one_in_x_extremes_type: str = "max",
    bias_stations: str = "",
    bias_historical_slice: str = "",
    filter_unadjusted_models: str = "",
    drop_leap_days: bool = False,
    concat_dim: str = "",
    export_filename: str = "",
    export_format: str = "NetCDF",
    export_separated: bool = False,
    export_location_naming: bool = False,
    verbosity: int = 0,
) -> str:
    """Generate ready-to-run Python code for a ClimateData query.

    Supports common and advanced processors in a single generated query.

    IMPORTANT: Before calling this tool, call search_guidance(query) if the
    user's request involves any methodology decision:
    - Dataset choice (WRF vs LOCA2) → search_guidance("WRF vs LOCA2 for <use case>")
    - Extreme events / return periods → search_guidance("extreme value analysis model selection")
    - Temporal/spatial scale → search_guidance("temporal scale hourly daily sampling window")
    - Bias correction → search_guidance("bias correction WRF a-priori")
    - Model/ensemble selection → search_guidance("how many models ensemble members")

    Also call search_code_examples(query) if the user wants a complete example,
    plot, or workflow pattern — the returned snippets can supplement or replace
    generated code for complex analyses.

    Use the guidance results to:
    1. Choose correct parameters (e.g., LOCA2 for 1-in-X, 30+ yr windows for extremes)
    2. Add # comments in the generated code citing the methodology rationale
    """
    lines = [
        "from climakitae.new_core.user_interface import ClimateData",
        "",
        f"cd = ClimateData(verbosity={verbosity})" if verbosity != 0 else "cd = ClimateData()",
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
        experiments = [x.strip() for x in experiment_id.split(",") if x.strip()]
        if len(experiments) == 1:
            lines.append(f'    .experiment_id("{experiments[0]}")')
        else:
            lines.append(f"    .experiment_id({experiments})")

    lines.append(f'    .table_id("{table_id}")')
    lines.append(f'    .grid_label("{grid_label}")')
    lines.append(f'    .variable("{variable}")')

    process_lines = []

    if concat_dim:
        process_lines.append(f'        "concat": "{concat_dim}"')

    if filter_unadjusted_models:
        process_lines.append(
            f'        "filter_unadjusted_models": "{filter_unadjusted_models}"'
        )

    if drop_leap_days:
        process_lines.append('        "drop_leap_days": "yes"')

    if time_slice_start and time_slice_end:
        if time_slice_start.isdigit() and time_slice_end.isdigit():
            process_lines.append(
                f'        "time_slice": ({time_slice_start}, {time_slice_end})'
            )
        else:
            process_lines.append(
                f'        "time_slice": ("{time_slice_start}", "{time_slice_end}")'
            )

    if warming_levels:
        levels = _split_csv(warming_levels, float)
        wl_parts = [f'"warming_levels": {levels}']
        if warming_level_window:
            wl_parts.append(f'"warming_level_window": {warming_level_window}')
        process_lines.append(f'        "warming_level": {{{", ".join(wl_parts)}}}')

    if clip:
        boundaries = [x.strip() for x in clip.split(";") if x.strip()]
        if len(boundaries) == 1 and not clip_separated:
            process_lines.append(f'        "clip": "{boundaries[0]}"')
        elif clip_separated:
            process_lines.append(
                f'        "clip": {{"boundaries": {boundaries}, "separated": True}}'
            )
        else:
            process_lines.append(f'        "clip": {boundaries}')
    elif clip_points:
        points = []
        for pair in [x.strip() for x in clip_points.split(";") if x.strip()]:
            lat, lon = [value.strip() for value in pair.split(",", maxsplit=1)]
            points.append(f"({lat}, {lon})")

        if len(points) == 1 and not clip_separated:
            process_lines.append(f'        "clip": {points[0]}')
        elif clip_separated:
            process_lines.append(
                f'        "clip": {{"points": [{", ".join(points)}], "separated": True}}'
            )
        else:
            process_lines.append(f'        "clip": [{", ".join(points)}]')
    elif clip_bbox:
        values = [x.strip() for x in clip_bbox.split(",") if x.strip()]
        if len(values) == 4:
            lat_min, lat_max, lon_min, lon_max = values
            process_lines.append(
                f'        "clip": (({lat_min}, {lat_max}), ({lon_min}, {lon_max}))'
            )

    if convert_units:
        process_lines.append(f'        "convert_units": "{convert_units}"')

    if bias_stations:
        stations = [x.strip() for x in bias_stations.split(",") if x.strip()]
        bias_parts = [f'"stations": {stations}']

        if bias_historical_slice:
            bounds = [x.strip() for x in bias_historical_slice.split(",") if x.strip()]
            if len(bounds) == 2:
                start, end = bounds
                if start.isdigit() and end.isdigit():
                    bias_parts.append(f'"historical_slice": ({start}, {end})')
                else:
                    bias_parts.append(f'"historical_slice": ("{start}", "{end}")')

        process_lines.append(
            f'        "bias_adjust_model_to_station": {{{", ".join(bias_parts)}}}'
        )

    if metric or percentiles or one_in_x_return_periods:
        metric_parts = []

        if metric:
            metric_parts.append(f'"metric": "{metric}"')

        if metric_dims:
            dims = [x.strip() for x in metric_dims.split(",") if x.strip()]
            if len(dims) == 1:
                metric_parts.append(f'"dim": "{dims[0]}"')
            else:
                metric_parts.append(f'"dim": {dims}')

        if percentiles:
            pct = _split_csv(percentiles, float)
            metric_parts.append(f'"percentiles": {pct}')

        if one_in_x_return_periods:
            periods = _split_csv(one_in_x_return_periods, int)
            one_in_x_parts = [f'"return_periods": {periods}']

            if one_in_x_distribution and one_in_x_distribution != "gev":
                one_in_x_parts.append(f'"distribution": "{one_in_x_distribution}"')
            if one_in_x_extremes_type and one_in_x_extremes_type != "max":
                one_in_x_parts.append(f'"extremes_type": "{one_in_x_extremes_type}"')

            metric_parts.append(f'"one_in_x": {{{", ".join(one_in_x_parts)}}}')

        process_lines.append(f'        "metric_calc": {{{", ".join(metric_parts)}}}')

    if export_filename:
        export_parts = [f'"filename": "{export_filename}"']

        if export_format:
            export_parts.append(f'"file_format": "{export_format}"')
        if export_separated:
            export_parts.append('"separated": True')
        if export_location_naming:
            export_parts.append('"location_based_naming": True')

        process_lines.append(f'        "export": {{{", ".join(export_parts)}}}')

    if process_lines:
        lines.append("    .processes({")
        lines.append(",\n".join(process_lines))
        lines.append("    })")

    lines.append("    .get())")
    return "\n".join(lines)


_COMMON_WORKFLOWS = """# Common Workflow Prompts

Use these prompts with `generate_code` or as direct templates:

1) Basic county time-slice query
- WRF monthly temperature for Los Angeles County, 2030-2060 in °F

2) Warming-level analysis
- WRF daily t2 over Humboldt County at 1.5, 2.0, 3.0°C

3) Multi-point comparison
- Extract WRF t2max at Los Angeles + San Francisco + San Diego points

4) Percentiles and thresholds
- Compute 90/95/98th percentiles at 1.2°C warming for a county

5) 1-in-X extreme value analysis
- Estimate 1-in-10/50/100 daily max t2 return values for Alameda County

6) Bias-correct to station observations
- WRF hourly t2 bias-adjusted to KSAC for 1980-2014

7) Export data
- Save clipped warming-level data to NetCDF or Zarr

Code skeleton:

from climakitae.new_core.user_interface import ClimateData

cd = ClimateData(verbosity=-1)
data = (
    cd.catalog("cadcat")
      .activity_id("WRF")
      .institution_id("UCLA")
      .table_id("day")
      .grid_label("d03")
      .variable("t2")
      .processes({
          "warming_level": {"warming_levels": [1.5, 2.0]},
          "clip": "Los Angeles County",
          "convert_units": "degF",
      })
      .get()
)
"""


if __name__ == "__main__":
    mcp.run()
