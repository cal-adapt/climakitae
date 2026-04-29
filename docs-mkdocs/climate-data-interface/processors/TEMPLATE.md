# Template: Processor Documentation Page

Use this template when creating a new processor documentation page. Replace `processor_name` and customize the algorithm flowchart, code links, and parameters table.

```markdown
# Processor: Processor Name

**Priority:** N | **Category:** [Temporal / Spatial / Conversion / Aggregation / Export]

Brief description of what the processor does, when to use it, and key transformations it performs.

## Algorithm

Visual flowchart showing the execution flow:

\`\`\`mermaid
flowchart TD
    Start([Input: xr.Dataset]) --> A["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/processor_name.py#L50'>Initialize parameters</a>"]
    A --> B["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/processor_name.py#L75'>Validate inputs</a>"]
    B --> C{Handle input type?}
    C -->|dict| D["<a href='...'>Process each key</a>"]
    C -->|Dataset| E["<a href='...'>Process single dataset</a>"]
    C -->|list/tuple| F["<a href='...'>Process each item</a>"]
    D --> G["<a href='...'>Update context</a>"]
    E --> G
    F --> G
    G --> End([Output: xr.Dataset])
\`\`\`

### Execution Flow

1. **Initialization** (lines X–Y): Parse and validate processor parameters
2. **Input Routing** (lines Y–Z): Determine if input is dict, Dataset, list, or tuple
3. **Core Processing** (lines Z–W): Apply transformation logic
4. **Context Update** (lines W–V): Record operation in dataset attributes
5. **Return** (line V): Return transformed data

## Parameters

| Parameter | Type | Required | Default | Description | Constraints |
|-----------|------|----------|---------|-------------|-------------|
| `param1` | str / int / list | ✓ | — | Description | Range, enum, or requirements |
| `param2` | bool | | False | Description | — |

## Code References

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | [50–70](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/processor_name.py#L50) | Parse and store parameters |
| `execute` | [75–150](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/processor_name.py#L75) | Route input and apply transformation |
| `_helper_method` | [155–190](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/processor_name.py#L155) | Core transformation logic |
| `update_context` | [195–210](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/processor_name.py#L195) | Record metadata |

## Example Usage

Basic usage via `ClimateData.processes()`:

\`\`\`python
from climakitae.new_core.user_interface import ClimateData

data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .table_id("mon")
    .grid_label("d03")
    .processes({
        "processor_name": {
            "param1": "value1",
            "param2": True
        }
    })
    .get())
```

Advanced: Multiple operations in sequence:

\`\`\`python
data = (ClimateData()
    # ... basic query ...
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "clip": "Los Angeles",
        "processor_name": {"param1": "value1"}
    })
    .get())
\`\`\`

## Implementation Details

### Data Type Handling

The processor handles multiple input types:

- **`xr.Dataset`**: Single dataset transformation
- **`dict`**: Key-value mapping (e.g., from `separated` output)
- **`list` / `tuple`**: Multiple datasets (e.g., multi-point results)

The output type matches the input type for consistency.

### Context Metadata

Operations are recorded in the `_NEW_ATTRS_KEY` context dictionary for provenance tracking:

\`\`\`python
context["new_attrs"]["processor_name"] = "Description of operation..."
\`\`\`

This is later attached to output dataset attributes and stored in exported files.

### Error Handling

- **Invalid parameters**: Raise `ValueError` with descriptive message during `__init__`
- **Processing errors**: Log warning, attempt graceful degradation or re-raise
- **Type mismatches**: Log and handle, return unmodified data if not applicable

## See Also

- [Processor Index](index.md)
- [Architecture → Extension Guide](../architecture.md#add-a-processor-4-step-guide)
- [How-To Guides](../howto.md)
```

## Creating a Processor Page

1. **Copy this template** to `docs-mkdocs/new-core/processors/processor_name.md`
2. **Read the source code** (`climakitae/new_core/processors/processor_name.py`)
3. **Customize sections**:
   - Replace `processor_name` throughout
   - Sketch the algorithm flowchart with key decision points
   - Add GitHub code links (adjust line numbers)
   - List all parameters with constraints
   - Provide real example usage
4. **Validate links**: Open HTML build locally and verify boxes link correctly
5. **Commit and update navigation**: Add to `mkdocs.yml` under `Processors` section

## Flowchart Best Practices

- **One box per decision or major operation** — Don't over-granularize
- **Include line numbers in links** — e.g., `#L50` points to specific code
- **Show branching logic** — If/else, match statements, type routing
- **Label outputs clearly** — Show what data structure is returned
- **Use semantic colors** (optional): 
  - Blue: I/O operations
  - Green: Data transformations
  - Orange: Validation/branching
  - Red: Error handling

