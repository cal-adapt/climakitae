# Data Processors

The data processor pipeline transforms query results in priority order. Each processor is registered against a string key and runs at a fixed priority. The full registry, parameter shapes, and per-processor narratives live under the [Processors section](../climate-data-interface/processors/index.md).

## Where to look

| You want… | Read |
|-----------|------|
| Per-processor parameter shapes and examples | [Processors index](../climate-data-interface/processors/index.md) and the individual `processors/<name>.md` pages |
| Base class, registry decorator, and helpers (auto-generated) | [Processor Utilities (Detailed)](processor-utilities.md) |
| The fluent query API (`ClimateData`) | [ClimateData class](climate-data.md) |
| How processors fit into pipeline execution | [Architecture](../climate-data-interface/architecture.md#registry-pattern) |

## See also

- [How-To Guides](../climate-data-interface/howto.md) — task-oriented examples that chain multiple processors
- [Migration: Legacy → ClimateData](../migration/legacy-to-climate-data.md) — mapping from `DataParameters` fields to processors
