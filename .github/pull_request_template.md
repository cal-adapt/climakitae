# Description of PR

### Summary of changes and related issue
[What's changed in this PR?]

### Relevant motivation and context
[Why did you change this and what applicable context is needed to understand why this change is needed?]

### How to test 
[How should reviewer's test the changes?] 

### Type of change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## Definition of Done Checklist

#### Practical
- [ ] Unit tests
  - [ ] Existing unit tests are passing
  - [ ] If relevant, new unit tests are written (required 80% unit test coverage)
- [ ] Documentation
  - [ ] All functions/adjusted functions documented in the [readthedocs](https://climakitae.readthedocs.io/en/latest/).
  - [ ] Intent of all functions included
  - [ ] Complex code commented
  - [ ] Functions include [NumPy style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) 
- [ ] Naming conventions followed
  - [ ] Helper functions hidden with `_` before the name
- [ ] Any notebooks known to utilize the affected functions are still working
- [ ] Black formatting has been utilized
