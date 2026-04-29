# Data Processors

The data processor pipeline transforms query results in priority order. Each processor is registered against a string key and runs at a fixed priority. The full registry, parameter shapes, and per-processor narratives live under the [Processors section](../new-core/processors/index.md).

## Where to look

| You want… | Read |
|-----------|------|
| Per-processor parameter shapes and examples | [Processors index](../new-core/processors/index.md) and the individual `processors/<name>.md` pages |
| Base class, registry decorator, and helpers (auto-generated) | [Processor Utilities (Detailed)](processor-utilities.md) |
| The fluent query API (`ClimateData`) | [Main Entrypoint](new-core.md) |
| How processors fit into pipeline execution | [Architecture](../new-core/architecture.md#registry-pattern) |

## See also

- [How-To Guides](../new-core/howto.md) — task-oriented examples that chain multiple processors
- [Migration: Legacy → ClimateData](../migration/legacy-to-new-core.md) — mapping from `DataParameters` fields to processors
