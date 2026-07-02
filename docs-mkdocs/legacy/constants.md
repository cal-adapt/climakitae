# Legacy Constants

The **`climakitae.core.constants` module** holds the shared sentinels and lookup
constants used across the legacy stack. These are also imported by parts of the
modern interface, so they remain a stable reference.

!!! note
    Unlike most legacy modules, `constants` is shared with the modern interface.
    The values documented here are current, not deprecated.

## Key constants

| Constant | Purpose |
|----------|---------|
| `UNSET` | Sentinel for "not yet set by the user", distinct from `None` (explicitly set to nothing). |
| `WARMING_LEVELS` | The supported global warming levels: `[0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]`. |
| `SSPS` | The Shared Socioeconomic Pathway scenario labels. |
| `_NEW_ATTRS_KEY` | Key used to write metadata into processor context dictionaries. |
| `WRF_BA_MODELS` / `NON_WRF_BA_MODELS` | WRF model lists with and without a-priori bias adjustment. |

---

## Public API

::: climakitae.core.constants
    options:
      docstring_style: numpy
      show_source: true

---

## Related legacy modules

- [Legacy API Overview](index.md)
- [Data Interface](data-interface.md)
- [Paths](paths.md)
