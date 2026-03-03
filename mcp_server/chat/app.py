"""FastAPI backend for climakitae chat assistant.

Bridges a React chat UI to Ollama, with climakitae MCP tools
available as function calls. The LLM generates code — never executes it.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Import MCP tools as plain Python functions ─────────────────────────
# Add the mcp_server directory to path so we can import server.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import (
    CATALOG,
    VARIABLES,
    generate_code,
    list_derived_functions,
    lookup_options,
    lookup_variables,
    validate_query,
)

# Also load the doc resources for the system prompt
from server import _read_doc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen2.5:3b"  # Good tool-calling support, fast on CPU

# ── Tool registry ──────────────────────────────────────────────────────
# Map tool names to their Python callables and Ollama-compatible schemas.

TOOL_FUNCTIONS: dict[str, Any] = {
    "lookup_variables": lookup_variables,
    "lookup_options": lookup_options,
    "validate_query": validate_query,
    "list_derived_functions": list_derived_functions,
    "generate_code": generate_code,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_variables",
            "description": "Search available climate variables. Filter by downscaling method and/or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_id": {
                        "type": "string",
                        "description": 'Filter by downscaling method: "WRF" or "LOCA2". Empty = all.',
                    },
                    "search": {
                        "type": "string",
                        "description": 'Keyword to search variable names/descriptions (e.g., "temperature", "wind").',
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_options",
            "description": "List all available values for a query parameter type.",
            "parameters": {
                "type": "object",
                "required": ["option_type"],
                "properties": {
                    "option_type": {
                        "type": "string",
                        "description": "One of: catalogs, activity_ids, variable_ids, table_ids, grid_labels, experiment_ids, institution_ids, source_ids",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_query",
            "description": "Check if a combination of query parameters is valid before generating code.",
            "parameters": {
                "type": "object",
                "required": ["catalog", "variable"],
                "properties": {
                    "catalog": {
                        "type": "string",
                        "description": 'Data catalog (e.g., "cadcat")',
                    },
                    "variable": {
                        "type": "string",
                        "description": 'Variable ID (e.g., "tasmax", "t2max")',
                    },
                    "activity_id": {
                        "type": "string",
                        "description": '"WRF" or "LOCA2"',
                    },
                    "table_id": {
                        "type": "string",
                        "description": '"1hr", "day", "mon"',
                    },
                    "grid_label": {
                        "type": "string",
                        "description": '"d01", "d02", "d03"',
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_derived_functions",
            "description": "List all derived variable and climate index functions (HDD/CDD, heat index, fire index, etc.) with import paths and signatures.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_code",
            "description": "Generate ready-to-run Python code for a ClimateData query. Supports time slices, spatial clipping, warming levels, and export.",
            "parameters": {
                "type": "object",
                "required": ["catalog", "variable", "table_id", "grid_label"],
                "properties": {
                    "catalog": {
                        "type": "string",
                        "description": '"cadcat" or "renewable energy generation"',
                    },
                    "variable": {
                        "type": "string",
                        "description": 'Climate variable (e.g., "tasmax", "t2max")',
                    },
                    "table_id": {
                        "type": "string",
                        "description": '"1hr", "day", "mon"',
                    },
                    "grid_label": {
                        "type": "string",
                        "description": '"d01" (45km), "d02" (9km), "d03" (3km)',
                    },
                    "activity_id": {
                        "type": "string",
                        "description": '"WRF" or "LOCA2"',
                    },
                    "experiment_id": {
                        "type": "string",
                        "description": 'SSP scenario ID(s) comma-separated. Values: "historical", "ssp245", "ssp370", "ssp585". NOT for warming levels.',
                    },
                    "time_slice_start": {
                        "type": "string",
                        "description": "Start date (e.g., 2030-01-01)",
                    },
                    "time_slice_end": {
                        "type": "string",
                        "description": "End date (e.g., 2060-12-31)",
                    },
                    "clip": {
                        "type": "string",
                        "description": 'Boundary name(s), semicolon-separated. Single: "Los Angeles". Multiple: "Los Angeles;Sacramento;San Francisco"',
                    },
                    "clip_points": {
                        "type": "string",
                        "description": 'Lat/lon points as "lat,lon" or "lat1,lon1;lat2,lon2"',
                    },
                    "warming_levels": {
                        "type": "string",
                        "description": 'Global Warming Level thresholds in °C, comma-separated. E.g., "1.5,2.0,3.0". These are temperature thresholds, NOT scenario IDs.',
                    },
                    "warming_level_window": {
                        "type": "integer",
                        "description": "Years around GWL crossing point (default: 15)",
                    },
                    "export_filename": {
                        "type": "string",
                        "description": "Filename for export processor (no extension)",
                    },
                    "export_format": {
                        "type": "string",
                        "description": '"NetCDF", "CSV", or "Zarr"',
                    },
                },
            },
        },
    },
]

# ── System prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Cal-Adapt Climate Data Assistant. You help users write correct Python code using the climakitae library.

You NEVER execute code. You generate code the user copies and runs themselves.

WORKFLOW:
1. Understand what the user wants (variable, region, timeframe, analysis)
2. Use lookup_variables or lookup_options if you need to discover available data
3. Use generate_code to produce the final Python code
4. ALWAYS include the generated code in your response inside a ```python code block
5. Add a brief explanation of what the code does

CRITICAL: When generate_code returns Python code, you MUST include that EXACT code in a ```python code block in your reply. Never just describe the code — always show it.

EXAMPLE: If user asks "monthly max temp for LA, 2030-2060", call generate_code with:
  catalog="cadcat", variable="tasmax", table_id="mon", grid_label="d03", time_slice_start="2030-01-01", time_slice_end="2060-12-31", clip="Los Angeles"
Include ALL relevant parameters — don't omit time_slice or clip if the user specified them.

WARMING LEVELS vs EXPERIMENT IDS — these are DIFFERENT things:
- warming_levels: Temperature thresholds in °C (e.g., "1.5,2.0,3.0"). Use when user says "GWL", "warming level", or "°C warming".
- experiment_id: SSP scenario IDs (e.g., "ssp245", "ssp585"). Use when user says "SSP", "scenario", or "RCP".
- NEVER put warming level numbers in experiment_id. NEVER put SSP values in warming_levels.

MULTIPLE REGIONS: When user asks about multiple counties/cities, use semicolons in clip:
  clip="Los Angeles;Sacramento;San Francisco"

KEY RULES:
- Always use: from climakitae.new_core.user_interface import ClimateData
- WRF uses dynamical downscaling: variables are t2max, t2min, t2, prec, dew_point
- LOCA2 uses statistical downscaling: variables are tasmax, tasmin, pr (CMIP6 naming)
- Never mix WRF variables with LOCA2 activity_id (or vice versa)
- Minimum required params: catalog, variable, table_id, grid_label
- grid_label d01=45km, d02=9km, d03=3km
- table_id: 1hr=hourly, day=daily, mon=monthly
- experiment_id: historical, ssp245, ssp370, ssp585

For derived variables (heat index, degree days, fire index), use list_derived_functions to get the correct imports and signatures.

Be concise. Always show code in a ```python block."""

# ── FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(title="climakitae Chat Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict] = []


def _call_tool(name: str, args: dict) -> str:
    """Execute a tool function by name and return its string result."""
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _ollama_chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Call Ollama's chat API."""
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Handle a chat request with tool-calling loop."""
    # Build message history with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in req.messages:
        messages.append(
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
        )

    tool_calls_log = []

    # Tool-calling loop (max 5 rounds to prevent infinite loops)
    for _ in range(5):
        result = await _ollama_chat(messages, tools=TOOL_SCHEMAS)
        msg = result.get("message", {})

        # If the model made tool calls, execute them and continue
        if msg.get("tool_calls"):
            # Add assistant message with tool calls to history
            messages.append(msg)

            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                logger.info("Tool call: %s(%s)", fn_name, json.dumps(fn_args)[:200])

                tool_result = _call_tool(fn_name, fn_args)
                tool_calls_log.append({"name": fn_name, "args": fn_args})

                # Add tool response to history
                messages.append({"role": "tool", "content": tool_result})

            continue  # Let the model process tool results

        # No tool calls — we have the final response
        return ChatResponse(
            reply=msg.get("content", ""),
            tool_calls=tool_calls_log,
        )

    # Fallback if we hit the loop limit
    return ChatResponse(
        reply=msg.get("content", "I had trouble processing that. Could you rephrase?"),
        tool_calls=tool_calls_log,
    )


@app.get("/api/health")
async def health():
    """Health check — also verifies Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            return {
                "status": "ok",
                "ollama": True,
                "models": models,
                "active_model": MODEL,
            }
    except Exception as e:
        return {"status": "degraded", "ollama": False, "error": str(e)}
