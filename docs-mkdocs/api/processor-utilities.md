# Processor Utilities

Internal utilities for data processor implementation.

## Overview

The processor utilities modules provide base classes and helper functions for implementing data processors in the new core architecture:
- **Abstract base class** — `DataProcessor` base class and registry
- **Processor utilities** — Helper functions for common operations

## Data Processor Base Class

::: climakitae.new_core.processors.abc_data_processor.DataProcessor
    options:
      docstring_style: numpy
      show_source: true
      members_order: source

## Processor Registry

::: climakitae.new_core.processors.abc_data_processor.register_processor
    options:
      docstring_style: numpy
      show_source: true

## Processor Utilities

::: climakitae.new_core.processors.processor_utils
    options:
      docstring_style: numpy
      show_source: true

## Implementation Guide

To create a new processor, follow this template:

```python
from climakitae.new_core.processors.abc_data_processor import DataProcessor, register_processor

@register_processor(key="my_processor", priority=50)
class MyProcessor(DataProcessor):
    """Process climate data in a specific way.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with processor-specific parameters
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        # Validation
        
    def execute(self, data, context):
        """Execute the processing step.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            Input climate data
        context : dict
            Shared processing context
            
        Returns
        -------
        xr.Dataset or xr.DataArray
            Processed data with same lazy evaluation properties
        """
        # Processing logic
        return processed_data
    
    def update_context(self, context):
        """Update shared processing context with metadata.
        
        Parameters
        ----------
        context : dict
            Shared context to update with processor metadata
        """
        # Store processor-specific metadata
        pass
    
    def set_data_accessor(self, catalog):
        """Configure data accessor if needed.
        
        Parameters
        ----------
        catalog : DataCatalog
            Data catalog instance for this processor
        """
        pass
```

## See Also

- [Key Processor Patterns](../new-core/architecture.md#key-design-patterns)
- [Adding a Processor](../new-core/architecture.md#add-a-processor-4-step-guide)
