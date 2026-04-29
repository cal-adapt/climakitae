# Legacy API Status

`climakitae.core` (the legacy `DataParameters` / `get_data` interface) is **still supported** for backward compatibility, but is now in **maintenance-only** mode. All new development happens in `climakitae.new_core`.

## TL;DR

- ✅ Legacy code keeps working — no breaking changes are planned in the next minor release series.
- ⚠️ All new features (processors, catalogs, validation, derived variables) ship to `new_core` only.
- 📘 New tutorials, notebooks, and documentation use `ClimateData` (ClimateData interface) exclusively.
- 🚦 A multi-phase deprecation plan has been drafted; see the deprecation timeline below.

## Deprecation timeline (planned)

The phased migration is tracked in the `climakitae-d91` epic on the project's issue tracker. A summary of the planned phases:

| Phase | Approximate window | What changes for users |
|------:|-------------------|------------------------|
| 1 | Now → mid-2026 | Audit, freeze legacy feature set, publish migration guide. **No runtime changes.** |
| 2 | Jul – Dec 2026 | `DeprecationWarning` emitted on first import / first call to legacy `get_data`. Migration helper script published. |
| 3 | Jan – Sep 2027 | Documentation site marks legacy pages as *deprecated*; banners in notebooks. Legacy still functional. |
| 4 | Oct – Dec 2027 | Legacy `climakitae.core.data_interface` is removed. Final pre-removal release pinned for users who cannot migrate. |

Exact dates may shift based on community uptake and feedback. Watch the changelog for binding announcements.

## Recommended action

- **New projects** — start with [`ClimateData`](../new-core/index.md). Follow [Get Started in 5 Minutes](../getting-started.md).
- **Existing projects** — read the [Legacy → ClimateData migration guide](../migration/legacy-to-new-core.md), pin a known-good `climakitae` version, and migrate at your own pace before Phase 4.
- **Notebook authors** — replace `from climakitae.core.data_interface import get_data` with `from climakitae.new_core.user_interface import ClimateData`. The migration guide has a side-by-side comparison.

## Where the legacy API still appears in the wild

Some Cal-Adapt resources continue to demonstrate the legacy API while we update them:

- The [Cal-Adapt Analytics Engine — Methods page](https://analytics.cal-adapt.org/analytics/methods) documents the GWL workflow using `get_data`.
- Several `cae-notebooks` are mid-migration; the Notebook Gallery flags any that have not yet been ported.

If you spot legacy usage that should be modernized, please open an issue on [`cal-adapt/climakitae`](https://github.com/cal-adapt/climakitae/issues).

## See also

- [Legacy → ClimateData migration guide](../migration/legacy-to-new-core.md)
- [Legacy API reference](../api/core.md)
- [ClimateData Interface overview](../new-core/index.md)
