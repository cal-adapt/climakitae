# Testing Philosophy and Best Practices for Junior Developers
## A Comprehensive Guide Using ClimakitAE Examples

---

## Agenda

1. **Testing Philosophy: The Why**
   - Unit vs Integration Tests
   - Benefits and Trade-offs

2. **Testing Tools: The How**
   - Mocking and Patching
   - Fixtures and Setup
   - Test Structure

3. **Testing Strategy: The What**
   - What to Test
   - Test Organization
   - Best Practices

4. **Practical Examples**
   - Real code from ClimakitAE
   - Common patterns and pitfalls

---

# Part 1: Testing Philosophy - The Why

---

## Why Do We Write Unit Tests?

### 🎯 **Primary Goals**
- **Catch bugs early** - Find issues before they reach production
- **Enable refactoring** - Change code confidently without breaking functionality
- **Document behavior** - Tests serve as living documentation of how code should work
- **Enable rapid development** - Quick feedback loop on changes

### 📊 **Real Impact**
```python
# Without tests: "Does this change break anything?"
# With tests: "The test suite tells me exactly what broke"
```

---

## Why Do We Write Unit Tests? (Continued)

### 🔍 **Example from our Boundaries class**
```python
def test_init_with_valid_catalog(self):
    """Test initialization with a valid catalog."""
    mock_catalog = Mock()
    for attr in ["states", "counties", "huc8", "utilities", "dfz", "eba"]:
        setattr(mock_catalog, attr, Mock())

    boundaries = Boundaries(mock_catalog)
    assert boundaries._cat is mock_catalog
    assert boundaries._lookup_cache == {}
```

**This test ensures:**
- Constructor works with valid input
- Initial state is correct
- We catch initialization bugs immediately

---

## Unit Tests vs Integration Tests

### 🧩 **Unit Tests**
- Test **individual components** in isolation
- Fast execution (milliseconds)
- Use mocks/stubs for dependencies
- High coverage, pinpoint failures

### 🔗 **Integration Tests**
- Test **components working together**
- Slower execution (seconds/minutes)
- Use real dependencies when possible
- Test realistic workflows

---

## Unit vs Integration: Practical Example

### Unit Test (Fast, Isolated)
```python
def test_process_ca_counties(self):
    """Test CA counties data processing (sorting)."""
    boundaries = Boundaries.__new__(Boundaries)  # Skip __init__
    test_df = pd.DataFrame({"NAME": ["San Francisco", "Alameda", "Los Angeles"]})

    result = boundaries._process_ca_counties(test_df)
    
    # Should be sorted by NAME
    expected = test_df.sort_values("NAME")
    pd.testing.assert_frame_equal(result, expected)
```

### Integration Test (Realistic, Slower)
```python
@pytest.mark.integration
def test_full_boundary_loading_pipeline(self):
    """Test complete boundary loading from catalog to final data."""
    # Use real catalog, real data, test end-to-end
    real_catalog = intake.open_catalog('boundaries.yaml')
    boundaries = Boundaries(real_catalog)
    
    # Test the full pipeline
    boundary_dict = boundaries.boundary_dict()
    assert 'CA counties' in boundary_dict
    assert len(boundary_dict['CA counties']) > 50  # CA has 58 counties
```

---

# Part 2: Testing Tools - The How

---

## Mocking and Patching: Why?

### 🎭 **Core Problem**: External Dependencies
```python
class Boundaries:
    def __init__(self, boundary_catalog):
        self._cat = boundary_catalog  # External dependency!
        
    @property 
    def _us_states(self):
        if self.__us_states is None:
            # This could fail due to network, permissions, etc.
            self.__us_states = self._cat.states.read()
        return self.__us_states
```

**Without mocking:** Tests fail due to network issues, missing files, etc.
**With mocking:** Tests focus on our logic, not external systems

---

## Mocking: Controlling Dependencies

### 🎯 **Mock Objects Replace Real Dependencies**
```python
def test_us_states_lazy_loading(self, mock_boundaries):
    """Test lazy loading of US states data."""
    # Initially not loaded - check private attribute
    assert getattr(mock_boundaries, "_Boundaries__us_states", None) is None

    # Access triggers loading - this is what we're actually testing
    states = mock_boundaries._us_states
    assert states is not None
    assert isinstance(states, pd.DataFrame)
    
    # Subsequent access uses cached data - test caching behavior
    states2 = mock_boundaries._us_states
    assert states is states2  # Same object reference = cached
```

The `mock_boundaries` fixture provides a controlled environment where we can test our caching logic without external dependencies.

---

## Mocking: Key Principles

### ✅ **Good Mocking**
```python
# Mock at the boundary - where YOUR code calls external code
@patch('climakitae.new_core.data_access.boundaries.pd.DataFrame')
def test_with_controlled_data(mock_dataframe):
    mock_dataframe.return_value = expected_data
    # Test our logic with predictable input
```

### ❌ **Over-Mocking**
```python
# Don't mock everything - you're not testing anything real
@patch('boundaries.sort_values')
@patch('boundaries.copy') 
@patch('boundaries.loc')
def test_overmocked(mock1, mock2, mock3):
    # This doesn't test actual behavior
```

---

## Patching: When and Where

### 🎯 **Patch at Import Location, Not Definition**
```python
# ✅ CORRECT: Patch where it's imported/used
@patch('climakitae.new_core.data_access.boundaries.pd.read_parquet')
def test_loading(mock_read):
    # Test our code that calls pd.read_parquet
    
# ❌ WRONG: Patching at definition site
@patch('pandas.read_parquet')  # Too broad, might not work
```

### 🔧 **Real Example from Our Tests**
```python
@pytest.fixture
def mock_boundaries(self):
    """Create a Boundaries instance with mocked catalog."""
    mock_catalog = Mock()
    for attr in ["states", "counties", "huc8", "utilities", "dfz", "eba"]:
        catalog_entry = Mock()
        catalog_entry.read.return_value = self._create_mock_dataframe(attr)
        setattr(mock_catalog, attr, catalog_entry)

    return Boundaries(mock_catalog)
```

---

## Fixtures: Reusable Test Setup

### 🏗️ **Why Use Fixtures?**
- **Eliminate code duplication** across tests
- **Consistent test setup** 
- **Clean teardown** (automatic cleanup)
- **Dependency injection** for tests

### 📝 **Example: Complex Setup Made Simple**
```python
@pytest.fixture
def mock_boundaries_with_data(self):
    """Create boundaries with mock data already loaded."""
    boundaries = Boundaries.__new__(Boundaries)
    boundaries._lookup_cache = {}

    # Mock data for different boundary types
    setattr(boundaries, "_Boundaries__us_states",
            pd.DataFrame({
                "abbrevs": ["CA", "OR", "WA", "NV"],
                "name": ["California", "Oregon", "Washington", "Nevada"],
            }, index=[10, 11, 12, 13]))
    
    # ... more setup ...
    return boundaries
```

---

## Fixtures: Scope and Lifecycle

### ⏱️ **Fixture Scopes**
```python
@pytest.fixture(scope="function")  # Default: new for each test
def fresh_data():
    return create_test_data()

@pytest.fixture(scope="class")     # Shared across test class
def expensive_setup():
    return load_large_dataset()

@pytest.fixture(scope="session")   # Once per test session
def database_connection():
    return setup_test_db()
```

### 🔄 **Setup and Teardown**
```python
@pytest.fixture
def temp_file():
    # Setup
    filename = "test_file.txt"
    with open(filename, 'w') as f:
        f.write("test data")
    
    yield filename  # Provide to test
    
    # Teardown (automatic cleanup)
    os.remove(filename)
```

---

# Part 3: Testing Strategy - The What

---

## Test Structure: Organize for Success

### 📁 **Class-Based Organization**
```python
class TestBoundariesInitialization:
    """Test Boundaries class initialization and validation."""
    
    def test_init_with_valid_catalog(self):
        """Test successful initialization."""
        
    def test_init_with_missing_catalog_entries(self):
        """Test initialization fails with missing catalog entries."""

class TestBoundariesProperties:
    """Test lazy loading properties and data processing."""
    
    def test_us_states_lazy_loading(self):
        """Test lazy loading of US states data."""
```

**Benefits:**
- Related tests grouped together
- Shared setup with `setup_method`
- Clear test categories

---

## What Properties Should You Test?

### 🎯 **Essential Test Categories**

1. **Happy Path** - Normal, expected usage
2. **Edge Cases** - Boundary conditions, empty data
3. **Error Conditions** - Invalid input, system failures
4. **Performance** - Memory usage, timing constraints
5. **Integration Points** - How components work together

---

## Testing Properties: Concrete Examples

### 1. **Happy Path Testing**
```python
def test_init_with_valid_catalog(self):
    """Test initialization with a valid catalog."""
    mock_catalog = Mock()
    for attr in ["states", "counties", "huc8", "utilities", "dfz", "eba"]:
        setattr(mock_catalog, attr, Mock())

    boundaries = Boundaries(mock_catalog)
    assert boundaries._cat is mock_catalog
    assert boundaries._lookup_cache == {}
```

### 2. **Edge Case Testing**
```python
def test_empty_dataframes(self):
    """Test handling of empty DataFrames."""
    boundaries = Boundaries.__new__(Boundaries)
    
    # Test with empty DataFrame that has the required column
    empty_df = pd.DataFrame(columns=["NAME"])
    result = boundaries._process_ca_counties(empty_df)
    assert len(result) == 0
    assert "NAME" in result.columns
```

---

## Testing Properties: Error Conditions

### 3. **Error Condition Testing**
```python
def test_init_with_missing_catalog_entries(self):
    """Test initialization fails with missing catalog entries."""
    mock_catalog = Mock(spec=[])  # Empty spec = no attributes
    setattr(mock_catalog, "states", Mock())
    # Missing: huc8, utilities, dfz, eba

    with pytest.raises(ValueError) as excinfo:
        Boundaries(mock_catalog)

    assert "Missing required catalog entries" in str(excinfo.value)
    assert "huc8" in str(excinfo.value)
```

### 4. **Error Handling in Operations**
```python
def test_us_states_loading_error(self):
    """Test error handling during US states loading."""
    mock_catalog = Mock()
    catalog_entry = Mock()
    catalog_entry.read.side_effect = Exception("Catalog read error")
    setattr(mock_catalog, "states", catalog_entry)

    boundaries = Boundaries(mock_catalog)
    with pytest.raises(RuntimeError) as excinfo:
        _ = boundaries._us_states

    assert "Failed to load US states data" in str(excinfo.value)
```

---

## Testing Properties: Performance and Memory

### 5. **Performance Testing**
```python
def test_get_memory_usage_with_loaded_data(self):
    """Test memory usage with some data loaded."""
    boundaries = Boundaries.__new__(Boundaries)
    boundaries._lookup_cache = {"test": {"key": 0}}

    # Mock DataFrame with realistic memory usage
    mock_df1 = Mock()
    mock_memory_usage = Mock()
    mock_memory_usage.sum.return_value = 1024
    mock_df1.memory_usage.return_value = mock_memory_usage

    setattr(boundaries, "_Boundaries__us_states", mock_df1)
    result = boundaries.get_memory_usage()

    assert result["us_states"] == 1024
    assert result["total_bytes"] == 1024
    assert result["loaded_datasets"] == 1
```

---

## Testing Behavior: State Changes

### 🔄 **Testing State Transitions**
```python
def test_us_states_lazy_loading(self, mock_boundaries):
    """Test lazy loading of US states data."""
    # Initially not loaded - check private attribute  
    assert getattr(mock_boundaries, "_Boundaries__us_states", None) is None

    # Access triggers loading
    states = mock_boundaries._us_states
    assert states is not None
    assert isinstance(states, pd.DataFrame)
    assert getattr(mock_boundaries, "_Boundaries__us_states", None) is not None

    # Subsequent access uses cached data
    states2 = mock_boundaries._us_states
    assert states is states2  # Test caching works
```

**This tests:**
- Initial state (not loaded)
- State change (loading triggered)
- Final state (cached)
- Behavior consistency (caching works)

---

## Testing Caching Behavior

### 💾 **Cache Testing Pattern**
```python
def test_get_us_states_caching(self, mock_boundaries_with_data):
    """Test US states lookup dictionary caching."""
    boundaries = mock_boundaries_with_data

    # First call builds cache
    result1 = boundaries._get_us_states()
    assert "us_states" in boundaries._lookup_cache
    assert isinstance(result1, dict)

    # Second call uses cache - THIS IS THE KEY TEST
    result2 = boundaries._get_us_states()
    assert result1 is result2  # Same object reference = cached
```

**Why test caching?**
- Performance impact
- Memory management
- Consistency guarantees

---

# Part 4: Advanced Testing Patterns

---

## Parameterized Testing: Test Multiple Cases

### 🔄 **Testing Multiple Scenarios Efficiently**
```python
@pytest.mark.parametrize("dataset_type,expected_column", [
    ("states", "abbrevs"),
    ("counties", "NAME"), 
    ("huc8", "Name"),
    ("utilities", "Utility"),
])
def test_dataset_structure(self, dataset_type, expected_column):
    """Test that each dataset has expected column structure."""
    mock_data = self._create_mock_dataframe(dataset_type)
    assert expected_column in mock_data.columns
```

**Benefits:**
- Single test logic, multiple scenarios
- Clear test case identification
- Easy to add new cases

---

## Testing Complex Data Transformations

### 🔄 **Testing Data Processing Logic**
```python
def test_process_ca_forecast_zones(self):
    """Test CA forecast zones data processing (replace 'Other' names)."""
    boundaries = Boundaries.__new__(Boundaries)
    test_df = pd.DataFrame({
        "FZ_Name": ["North Bay", "Other", "South Bay"],
        "FZ_Def": ["North Bay", "Alameda County", "South Bay"],
    })

    result = boundaries._process_ca_forecast_zones(test_df)

    # Test specific transformation: 'Other' -> actual county name
    assert result.loc[1, "FZ_Name"] == "Alameda County"
    assert result.loc[0, "FZ_Name"] == "North Bay"  # Unchanged
    assert result.loc[2, "FZ_Name"] == "South Bay"  # Unchanged
```

**Key principles:**
- Test the transformation logic, not pandas itself
- Use realistic but controlled data
- Verify both changed and unchanged data

---

## Memory and Resource Management Testing

### 💾 **Testing Resource Cleanup**
```python
def test_clear_cache(self, mock_boundaries_public):
    """Test clear_cache resets all cached data."""
    # Set some data first
    mock_boundaries_public._lookup_cache["test"] = {"key": "value"}
    setattr(mock_boundaries_public, "_Boundaries__us_states", Mock())

    mock_boundaries_public.clear_cache()

    # Verify complete cleanup
    assert mock_boundaries_public._lookup_cache == {}
    private_attrs = [
        "_Boundaries__us_states",
        "_Boundaries__ca_counties", 
        # ... etc
    ]
    for attr in private_attrs:
        assert getattr(mock_boundaries_public, attr, None) is None
```

---

## Testing Deprecation and Backwards Compatibility

### ⚠️ **Testing Deprecation Warnings**
```python
def test_load_deprecated_warning(self, mock_boundaries_public):
    """Test that load() method issues deprecation warning."""
    with patch.object(mock_boundaries_public, "preload_all") as mock_preload:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mock_boundaries_public.load()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            mock_preload.assert_called_once()
```

**Why test deprecation?**
- Ensure warnings are actually issued
- Verify fallback behavior works
- Document migration path

---

# Part 5: Best Practices and Common Pitfalls

---

## Best Practices: Test Naming and Documentation

### 📝 **Descriptive Test Names**
```python
# ✅ GOOD: Clear, specific, tells a story
def test_init_with_missing_catalog_entries(self):
    """Test initialization fails with missing catalog entries."""

def test_us_states_lazy_loading_uses_cache_on_second_access(self):
    """Test that accessing US states twice uses cached data."""

# ❌ BAD: Vague, doesn't explain behavior
def test_init(self):
    """Test init method."""

def test_states(self):
    """Test states."""
```

### 📖 **Documentation Standards**
```python
def test_process_ca_electric_balancing_areas(self):
    """Test CA electric balancing areas processing (remove tiny CALISO).
    
    The CALISO polygon has two options where one is super tiny with 
    negligible area. This test ensures the tiny polygon is removed 
    and only the large one is kept.
    """
```

---

## Best Practices: Assertion Quality

### ✅ **Meaningful Assertions**
```python
# ✅ GOOD: Specific, explains what's expected
assert result.loc[1, "FZ_Name"] == "Alameda County"
assert len(result) == 2
assert 0 not in result.index  # Tiny CALISO removed
assert 1 in result.index      # Large CALISO kept

# ❌ BAD: Generic, doesn't explain intent
assert result is not None
assert len(result) > 0
```

### 🔍 **Use Appropriate Assertion Methods**
```python
# For DataFrames
pd.testing.assert_frame_equal(result, expected)

# For dictionaries  
assert "us_states" in boundaries._lookup_cache

# For exceptions with specific messages
with pytest.raises(ValueError) as excinfo:
    Boundaries(invalid_catalog)
assert "Missing required catalog entries" in str(excinfo.value)
```

---

## Common Pitfalls: Over-Mocking

### ❌ **Don't Mock What You're Testing**
```python
# BAD: Mocking the behavior we want to test
@patch('boundaries._process_ca_counties')
def test_ca_counties_processing(mock_process):
    mock_process.return_value = expected_result
    result = boundaries._process_ca_counties(input_data)
    # This doesn't test anything - we mocked the method!
```

### ✅ **Mock Dependencies, Test Your Logic**
```python
# GOOD: Mock external dependency, test our processing
def test_ca_counties_processing(self):
    boundaries = Boundaries.__new__(Boundaries)
    test_df = pd.DataFrame({"NAME": ["San Francisco", "Alameda", "Los Angeles"]})
    
    result = boundaries._process_ca_counties(test_df)
    
    # Test our actual sorting logic
    expected = test_df.sort_values("NAME")
    pd.testing.assert_frame_equal(result, expected)
```

---

## Common Pitfalls: Fragile Tests

### ❌ **Brittle Test Setup**
```python
# BAD: Too specific, breaks easily
def test_boundary_dict_structure(self):
    result = boundaries.boundary_dict()
    # This will break if we add new boundary types
    assert len(result) == 8  
    assert list(result.keys()) == ["none", "lat/lon", "states", ...]
```

### ✅ **Robust Test Assertions**
```python
# GOOD: Test behavior, not implementation details
def test_boundary_dict_contains_required_categories(self):
    result = boundaries.boundary_dict()
    
    required_categories = ["states", "CA counties", "CA watersheds"]
    for category in required_categories:
        assert category in result
        assert isinstance(result[category], dict)
```

---

## Performance Testing Guidelines

### ⏱️ **When and How to Test Performance**
```python
@pytest.mark.timeout(30)  # Fail if takes too long
def test_preload_all_performance(self):
    """Test that preloading all data completes in reasonable time."""
    boundaries = Boundaries(mock_catalog)
    
    start_time = time.time()
    boundaries.preload_all()
    elapsed = time.time() - start_time
    
    # Should complete quickly with mocked data
    assert elapsed < 5.0  # 5 seconds max

@pytest.mark.advanced  # Skip in basic test runs
def test_memory_usage_realistic(self):
    """Test memory usage with real data."""
    # Use real catalog for this test
    boundaries = Boundaries(real_catalog)
    boundaries.preload_all()
    
    usage = boundaries.get_memory_usage()
    # Memory should be reasonable
    assert usage['total_bytes'] < 100_000_000  # Less than 100MB
```

---

## Integration Testing Strategy

### 🔗 **When to Write Integration Tests**

1. **Critical user workflows**
```python
@pytest.mark.integration
def test_complete_boundary_selection_workflow(self):
    """Test full workflow from catalog to boundary selection."""
    boundaries = Boundaries(real_catalog)
    
    # User selects boundary type
    boundary_options = boundaries.boundary_dict()
    
    # User selects specific boundary
    counties = boundary_options['CA counties']
    alameda_index = counties['Alameda']
    
    # System retrieves boundary data
    county_data = boundaries._ca_counties.iloc[alameda_index]
    
    assert county_data['NAME'] == 'Alameda'
```

2. **Cross-component interactions**
3. **External system integrations**
4. **Performance with real data**

---

## Test Organization: File Structure

### 📁 **Mirror Source Code Structure**
```
tests/
├── conftest.py              # Shared fixtures
├── new_core/
│   ├── test_boundaries.py   # Mirrors new_core/data_access/boundaries.py
│   ├── test_dataset.py      # Mirrors new_core/dataset.py
│   └── data_access/
│       └── test_boundaries.py
└── core/
    ├── test_data_interface.py
    └── test_data_loader.py
```

### 🏷️ **Test Categorization**
```python
# Mark different test types
@pytest.mark.unit
def test_individual_method():
    pass

@pytest.mark.integration  
def test_component_interaction():
    pass

@pytest.mark.advanced     # Requires external resources
def test_with_real_data():
    pass

@pytest.mark.slow         # Takes significant time
def test_performance():
    pass
```

---

# Summary: Testing Checklist

---

## ✅ **Unit Test Checklist**

### **For Each Method/Function:**
- [ ] Happy path with valid inputs
- [ ] Edge cases (empty data, boundary values)
- [ ] Error conditions (invalid inputs, exceptions)
- [ ] State changes (if applicable)
- [ ] Caching behavior (if applicable)

### **For Each Class:**
- [ ] Constructor with valid/invalid inputs  
- [ ] Property access and modification
- [ ] Method chaining (if applicable)
- [ ] Resource cleanup
- [ ] Memory usage patterns

---

## ✅ **Integration Test Checklist**

### **End-to-End Workflows:**
- [ ] Complete user scenarios
- [ ] Cross-component communication
- [ ] External dependency integration
- [ ] Performance with realistic data
- [ ] Error propagation through system

### **System Behavior:**
- [ ] Configuration changes
- [ ] Resource exhaustion
- [ ] Concurrent access
- [ ] Data consistency

---

## 🎯 **Key Takeaways**

1. **Unit tests** catch bugs early and enable refactoring
2. **Mock dependencies**, not the code you're testing
3. **Test behavior**, not implementation details
4. **Use fixtures** to eliminate duplication
5. **Organize tests** by functionality, not file structure
6. **Write descriptive names** and documentation
7. **Test the unhappy path** - error conditions matter
8. **Integration tests** verify components work together
9. **Performance tests** catch scalability issues
10. **Maintain tests** like production code

---

## 📚 **Further Reading**

- **pytest documentation**: https://docs.pytest.org/
- **unittest.mock guide**: https://docs.python.org/3/library/unittest.mock.html
- **Testing best practices**: Clean Code by Robert Martin
- **ClimakitAE testing guidelines**: `/.github/instructions/testing.instructions.md`

---

## Questions?

**Remember:** Good tests are an investment in code quality, developer productivity, and system reliability. They're not just about catching bugs - they're about enabling fearless refactoring and confident deployment.

**Happy Testing! 🧪**
