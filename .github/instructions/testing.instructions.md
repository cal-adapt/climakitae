---
applyTo: '**'
---
# ClimakitAE Testing Instructions

This guide provides comprehensive instructions for writing pytest unit tests for the ClimakitAE project, focusing on best practices, mocking strategies, and documentation standards.

## Table of Contents
- [Test Organization]
- [Fixtures and Setup]
- [Mocking and Patching]
- [Documentation Standards]
- [Testing Patterns]
- [Common Pitfalls]
- [Performance Testing]
- [Test Data Management]
- [Parameterized Testing]
- [Integration Testing]
- [Debugging Failed Tests]
- [Test Coverage Configuration]
- [Running Tests]

## Test Organization

### 1. Class-Based Test Structure

Organize tests using classes that group related test cases by function and purpose:

```python
class TestClimateDataInit:
    """Test class for ClimateData initialization."""
    
    def test_init_successful(self):
        """Test successful initialization."""
        pass
    
    def test_init_with_factory_error(self):
        """Test initialization when DatasetFactory raises an exception."""
        pass


class TestClimateDataParameterSetters:
    """Test class for parameter setting methods."""
    
    def test_catalog_valid(self):
        """Test catalog setter with valid input."""
        pass
    
    def test_catalog_invalid_empty_string(self):
        """Test catalog setter with empty string."""
        pass
```

**Naming Conventions:**
- Test classes: `Test<ClassName><Purpose>` or `Test<FunctionName><Behavior>`
  - Examples: `TestClimateDataInit`, `TestDataProcessorValidation`, `TestExportForNetCDF`
- Test methods: `test_<method_name>_<scenario>`
  - Examples: `test_get_successful_execution`, `test_validate_missing_parameters`

### 2. File Structure

Mirror the source code structure in your test directory:

```
tests/
├── conftest.py              # Shared fixtures
├── core/
│   ├── test_data_interface.py
│   └── test_data_loader.py
├── new_core/
│   ├── test_user_interface.py
│   ├── test_dataset_factory.py
│   └── test_processors.py
└── utils/
    └── test_export.py
```

## Fixtures and Setup

### 1. Use `setup_method` for Test Class Initialization

```python
class TestClimateDataParameterSetters:
    """Test class for parameter setting methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        mock_factory_instance = MagicMock()
        with patch('climakitae.new_core.user_interface.DatasetFactory',
                  return_value=mock_factory_instance), \
             patch('climakitae.new_core.user_interface.read_csv_file',
                  return_value=pd.DataFrame()), \
             patch('builtins.print'):
            self.climate_data = ClimateData()
```

### 2. Create Reusable Fixtures in `conftest.py`

```python
# conftest.py
import pytest
import xarray as xr
import pandas as pd
from unittest.mock import MagicMock

@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset for testing.
    
    Returns
    -------
    xr.Dataset
        Sample climate dataset with temperature data.
    """
    return xr.Dataset({
        'tasmax': (['time', 'lat', 'lon'], 
                   np.random.rand(10, 5, 5)),
        'time': pd.date_range('2020-01-01', periods=10),
        'lat': np.linspace(32, 42, 5),
        'lon': np.linspace(-124, -114, 5)
    })

@pytest.fixture
def mock_catalog():
    """Create a mock intake-esm catalog.
    
    Returns
    -------
    MagicMock
        Mock catalog with basic query functionality.
    """
    catalog = MagicMock()
    catalog.search.return_value = catalog
    catalog.to_dataset_dict.return_value = {'key': MagicMock()}
    return catalog
```

## Mocking and Patching

### 1. Proper Patching Strategies

**Patch at the import location, not the definition location:**

```python
# CORRECT: Patch where the function is imported
@patch('climakitae.new_core.user_interface.DatasetFactory')
def test_something(mock_factory):
    pass

# INCORRECT: Don't patch at definition location
@patch('climakitae.new_core.dataset_factory.DatasetFactory')
def test_something(mock_factory):
    pass
```

### 2. Complex Mocking Patterns

**Mock with spec for type safety:**

```python
from climakitae.new_core.dataset_factory import DatasetFactory

@patch('climakitae.new_core.user_interface.DatasetFactory', 
       spec=DatasetFactory)
def test_with_spec(mock_factory):
    """Using spec ensures mock has same interface as real object."""
    mock_instance = mock_factory.return_value
    mock_instance.create_dataset.return_value = MagicMock()
```

**Mock chain of method calls:**

```python
def test_chained_methods(self):
    """Test method chaining with proper mocking."""
    mock_dataset = MagicMock()
    mock_dataset.spatial_average.return_value = mock_dataset
    mock_dataset.temporal_average.return_value = mock_dataset
    mock_dataset.execute.return_value = xr.Dataset()
    
    self.factory.create_dataset.return_value = mock_dataset
```

### 3. Context Manager Mocking

**Mock multiple dependencies:**

```python
def test_complex_initialization(self):
    """Test with multiple mocked dependencies."""
    with patch('module.DataCatalog') as mock_catalog, \
         patch('module.ParameterValidator') as mock_validator, \
         patch('module.read_csv_file') as mock_csv, \
         patch('builtins.print') as mock_print:
        
        # Setup mock returns
        mock_catalog.return_value.get_catalog.return_value = {}
        mock_validator.return_value.validate.return_value = True
        mock_csv.return_value = pd.DataFrame()
        
        # Execute test
        obj = MyClass()
        
        # Assertions
        mock_catalog.assert_called_once()
        assert mock_print.call_count > 0
```

### 4. Mock Data Structures

**Inspect actual data structures carefully:**

```python
def test_climate_data_structure(self):
    """Mock complex xarray Dataset structure."""
    # Create realistic mock data matching actual structure
    mock_data = MagicMock(spec=xr.Dataset)
    mock_data.dims = {'time': 365, 'lat': 100, 'lon': 100}
    mock_data.coords = {
        'time': pd.date_range('2020-01-01', periods=365),
        'lat': np.linspace(32, 42, 100),
        'lon': np.linspace(-124, -114, 100)
    }
    mock_data.data_vars = {'tasmax': MagicMock()}
    mock_data.attrs = {
        'source': 'CMIP6',
        'experiment': 'ssp245',
        'institution': 'CNRM'
    }
    mock_data.nbytes = 29200000  # Realistic size
```

## Documentation Standards

### 1. Module Docstrings

```python
"""
Unit tests for climakitae/new_core/user_interface.py

This module contains comprehensive unit tests for the ClimateData class
that provide the high-level interface for accessing climate data.
"""
```

### 2. Class Docstrings

```python
class TestClimateDataValidation:
    """Test class for parameter validation.
    
    This class tests all validation methods in ClimateData including
    required parameter checks, type validation, and boundary conditions.
    
    Attributes
    ----------
    climate_data : ClimateData
        Instance of ClimateData class for testing.
    mock_factory : MagicMock
        Mock DatasetFactory instance.
    """
```

### 3. Function/Method Docstrings (NumPy Style)

```python
def test_validate_experiment_id(self):
    """Test experiment_id validation with various inputs.
    
    Tests validation of experiment_id parameter including:
    - Valid string inputs
    - Valid list inputs  
    - Invalid types
    - Empty values
    
    Raises
    ------
    AssertionError
        If validation doesn't behave as expected.
    """
```

## Testing Patterns

### 1. Test Both Success and Failure Cases

```python
def test_parameter_valid(self):
    """Test with valid parameter."""
    result = self.obj.set_parameter("valid_value")
    assert result.parameter == "valid_value"

def test_parameter_invalid_empty(self):
    """Test with invalid empty parameter."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        self.obj.set_parameter("")

def test_parameter_invalid_type(self):
    """Test with invalid parameter type."""
    with pytest.raises(TypeError, match="must be string"):
        self.obj.set_parameter(123)
```

### 2. Test Edge Cases and Boundaries

```python
def test_date_range_boundary(self):
    """Test date range at boundaries."""
    # Test minimum date
    result = self.obj.set_date_range("1950-01-01", "1950-12-31")
    assert result is not None
    
    # Test maximum date  
    result = self.obj.set_date_range("2100-01-01", "2100-12-31")
    assert result is not None
    
    # Test invalid range (end before start)
    with pytest.raises(ValueError):
        self.obj.set_date_range("2020-12-31", "2020-01-01")
```

### 3. Test Method Chaining

```python
def test_method_chaining(self):
    """Test that methods can be chained together."""
    result = (self.climate_data
              .catalog("climate")
              .variable("tas")
              .table_id("day")
              .grid_label("d03"))
    
    # Should return the same instance
    assert result is self.climate_data
    
    # Parameters should be set correctly
    assert self.climate_data._query["catalog"] == "climate"
    assert self.climate_data._query["variable_id"] == "tas"
```

### 4. Test Error Handling

```python
def test_get_with_connection_error(self):
    """Test get method when connection fails."""
    self.mock_catalog.search.side_effect = ConnectionError("Network error")
    
    with patch('builtins.print') as mock_print:
        result = self.climate_data.get()
    
    assert result is None
    printed = "".join(str(call) for call in mock_print.call_args_list)
    assert "Connection failed" in printed
```

## Common Pitfalls

### 1. Avoid Over-Mocking

```python
# BAD: Over-mocking internal implementation details
@patch('module._private_method')
@patch('module._another_private')
@patch('module._internal_helper')
def test_overmocked(mock1, mock2, mock3):
    # Too many mocks, test is brittle
    pass

# GOOD: Mock external dependencies only
@patch('module.external_api_call')
def test_focused(mock_api):
    # Test actual logic, mock only external calls
    pass
```

### 2. Handle Async/Lazy Operations

```python
def test_lazy_loaded_data(self):
    """Test with dask/lazy-loaded xarray data."""
    # Create lazy dataset
    lazy_data = xr.Dataset({
        'var': (['x', 'y'], da.random.random((100, 100)))
    })
    
    # Mock to return lazy data
    self.mock_loader.load_data.return_value = lazy_data
    
    # Test doesn't trigger computation
    result = self.obj.get_lazy_data()
    assert isinstance(result['var'].data, da.Array)
    
    # Explicitly test computation if needed
    computed = result.compute()
    assert isinstance(computed['var'].data, np.ndarray)
```

### 3. Test Print Output Correctly

```python
def test_print_output(self):
    """Test printed output messages."""
    with patch('builtins.print') as mock_print:
        self.obj.display_info()
    
    # Collect all printed text
    all_prints = [str(call[0][0]) for call in mock_print.call_args_list]
    full_output = "\n".join(all_prints)
    
    # Check for expected messages
    assert "Expected message" in full_output
    assert mock_print.call_count == 3
```

### 4. Mark Advanced Tests

```python
@pytest.mark.advanced
def test_requires_external_connection(self):
    """Test that requires external data connection.
    
    This test is marked as advanced and skipped in basic test runs.
    Run with: pytest -m advanced
    """
    # Test with real external connection
    pass

@pytest.mark.skipif(not has_gpu(), reason="Requires GPU")
def test_gpu_processing(self):
    """Test GPU-accelerated processing."""
    pass
```

### 5. Use Meaningful Assertions

```python
# BAD: Generic assertion
assert result is not None

# GOOD: Specific assertion with context
assert result is not None, "Dataset creation should return valid xarray.Dataset"
assert isinstance(result, xr.Dataset), f"Expected xr.Dataset, got {type(result)}"
assert 'tasmax' in result.data_vars, "Dataset missing required 'tasmax' variable"
```

## Performance Testing

### 1. Test Performance-Critical Operations

```python
import pytest
import time

def test_large_dataset_processing_performance(self, benchmark):
    """Test performance with large datasets using pytest-benchmark."""
    large_data = create_large_dataset(size=(1000, 500, 500))
    
    # Use pytest-benchmark for accurate timing
    result = benchmark(self.processor.process, large_data)
    assert result is not None

@pytest.mark.timeout(30)
def test_operation_timeout(self):
    """Ensure operation completes within reasonable time."""
    result = self.climate_data.compute_expensive_operation()
    assert result is not None
```

### 2. Memory Usage Testing

```python
import tracemalloc

def test_memory_usage(self):
    """Test that operations don't leak memory."""
    tracemalloc.start()
    
    # Run operation multiple times
    for _ in range(100):
        result = self.processor.process_data()
        del result
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Assert memory usage is reasonable (e.g., < 1GB)
    assert peak < 1_000_000_000, f"Peak memory usage too high: {peak:,} bytes"
```

## Test Data Management

### 1. Use Test Data Factories

```python
# tests/factories.py
class TestDataFactory:
    """Factory for creating consistent test data."""
    
    @staticmethod
    def create_climate_dataset(
        variables=['tasmax'],
        time_periods=10,
        lat_points=5,
        lon_points=5,
        **kwargs
    ):
        """Create standardized test dataset."""
        data = {}
        for var in variables:
            data[var] = (['time', 'lat', 'lon'], 
                        np.random.rand(time_periods, lat_points, lon_points))
        
        return xr.Dataset(
            data,
            coords={
                'time': pd.date_range('2020-01-01', periods=time_periods),
                'lat': np.linspace(32, 42, lat_points),
                'lon': np.linspace(-124, -114, lon_points)
            },
            attrs={'source': 'test', **kwargs}
        )
```

### 2. Test Data Snapshots

```python
def test_data_transformation(snapshot):
    """Test using pytest-snapshot for complex data validation."""
    result = transform_complex_data(input_data)
    
    # Snapshot testing for complex outputs
    snapshot.assert_match(result.to_dict(), 'transformed_data.json')
```

## Parameterized Testing

### 1. Use pytest.mark.parametrize Effectively

```python
@pytest.mark.parametrize("catalog,expected_type", [
    ("CMIP6", "climate"),
    ("ERA5", "reanalysis"),
    ("CORDEX", "regional"),
    pytest.param("INVALID", None, marks=pytest.mark.xfail),
])
def test_catalog_types(self, catalog, expected_type):
    """Test different catalog types with parameterization."""
    result = self.climate_data.catalog(catalog)
    if expected_type:
        assert result._catalog_type == expected_type
```

### 2. Parameterize Complex Scenarios

```python
@pytest.mark.parametrize("scenario", [
    {
        "name": "historical",
        "params": {"experiment": "historical", "start": "1950", "end": "2014"},
        "expected_vars": ["tas", "pr"],
        "should_fail": False
    },
    {
        "name": "future_projection",
        "params": {"experiment": "ssp245", "start": "2015", "end": "2100"},
        "expected_vars": ["tasmax", "tasmin"],
        "should_fail": False
    },
    {
        "name": "invalid_dates",
        "params": {"experiment": "ssp245", "start": "2100", "end": "2000"},
        "expected_vars": [],
        "should_fail": True
    },
], ids=lambda s: s["name"])
def test_scenarios(self, scenario):
    """Test various climate scenarios."""
    if scenario["should_fail"]:
        with pytest.raises(ValueError):
            self.climate_data.setup(**scenario["params"])
    else:
        result = self.climate_data.setup(**scenario["params"])
        for var in scenario["expected_vars"]:
            assert var in result.variables
```

## Integration Testing

### 1. Test Component Integration

```python
class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.integration
    def test_full_pipeline(self, tmp_path):
        """Test complete data pipeline from input to output."""
        # Setup
        climate_data = ClimateData()
        
        # Configure
        climate_data.catalog("CMIP6") \
                   .variable("tas") \
                   .experiment("ssp245") \
                   .timeframe("2020", "2050")
        
        # Process
        dataset = climate_data.get()
        processed = climate_data.process(dataset)
        
        # Export
        output_file = tmp_path / "output.nc"
        climate_data.export(processed, output_file)
        
        # Verify
        assert output_file.exists()
        loaded = xr.open_dataset(output_file)
        assert 'tas' in loaded.data_vars
```

## Debugging Failed Tests

### 1. Add Debug Information

```python
def test_complex_calculation(self, caplog):
    """Test with debug logging enabled."""
    import logging
    caplog.set_level(logging.DEBUG)
    
    result = self.calculator.complex_calculation()
    
    # Check debug messages if test fails
    if result is None:
        print("Debug logs:")
        for record in caplog.records:
            print(f"{record.levelname}: {record.message}")
    
    assert result is not None
```

### 2. Use pytest Fixtures for Debugging

```python
@pytest.fixture
def debug_on_failure(request):
    """Drop into debugger on test failure."""
    yield
    if request.node.rep_call.failed:
        import pdb; pdb.set_trace()
```

## Test Coverage Configuration

### 1. Add .coveragerc Configuration

```ini
# .coveragerc
[run]
source = climakitae
omit = 
    */tests/*
    */test_*.py
    */__init__.py
    */conftest.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstract
```

### 2. Coverage Commands

```bash
# Run with branch coverage
pytest --cov=climakitae --cov-branch --cov-report=term-missing

# Generate XML report for CI
pytest --cov=climakitae --cov-report=xml

# Fail if coverage below threshold
pytest --cov=climakitae --cov-fail-under=80
```

### 3. Coverage Goals

- Aim for at least 80% test coverage across all modules.
- Prioritize covering critical and complex code paths.
- Regularly review and update tests to maintain coverage as code evolves.

## Running Tests

### 1. Environment Configuration

Be sure to configure the environment. The three most common environments are `uv`, 
`mamba`, and `conda`. 

```bash
# uv activation
source .venv/bin/activate

# mamba activation
mamba activate climakitae

# conda activation
conda activate climakitae
```

### 2. Running tests

Always use `python -m pytest` to run the tests. If the test environment is `uv` prepend
the `python -m pytest` command with `uv run` for tests that look like `uv run python -m pytest ...`

```bash
# Run basic tests (no external dependencies)
python -m pytest -m "not advanced"

# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=climakitae --cov-report=html

# Run specific test class
python -m pytest tests/new_core/test_user_interface.py::TestClimateDataInit

# Run with verbose output
python -m pytest -v

# Run and stop on first failure
python -m pytest -x
```
