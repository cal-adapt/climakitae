# New Core Overview

The new core API is the primary interface for climakitae.

## Design goals

- Fluent method chaining for readable query construction
- Explicit processors for transformations
- Lazy xarray/dask-backed execution

## Primary entrypoint

```python
from climakitae.new_core.user_interface import ClimateData
```

In this docs overhaul, all new tutorials and API examples will prioritize the new core.
