**Please manually remove unnecessary checkboxes if they're not applicable to your PR**

## Summary of changes and related issue
[What's changed in this PR?]

## Link to corresponding Jira ticket(s)
[What Jira ticket(s) describe further context/detail for this PR?]

## Testing
- [ ] Unit tests written for new/modified code (goal: 80% coverage)
  - All public functions have unit tests
  - Functions that must produce specific values have unit tests
- [ ] Verified that notebooks utilizing affected functions still work
- [ ] Appropriate manual testing completed
- [ ] Advanced Testing label added to PR if this PR makes major changes to the codebase 

## How to Test
[How should reviewers test these changes? Demo code snippet?]

## Documentation
- [ ] Complex code includes comments explaining logic
- [ ] NumPy-style docstrings added/updated for all new or modified public functions, classes, and modules

**Which of the following need updating? Check all that apply and complete them:**
- [ ] **API reference** (`docs-mkdocs/api/`) — add or update the `.md` file for the affected module (e.g. `processors.md`, `tools.md`, `util.md`)
- [ ] **Concept / architecture docs** (`docs-mkdocs/climate-data-interface/`) — update if you changed how a processor, validator, or data access component works
- [ ] **How-to guides** (`docs-mkdocs/climate-data-interface/howto/`) — add a new guide if you introduced a non-obvious workflow
- [ ] **Getting started / migration guides** (`docs-mkdocs/getting-started.md`, `docs-mkdocs/migration/`) — update if you changed the public API or added a new entry point
- [ ] **Notebook gallery** (`docs-mkdocs/notebook-gallery.md`) — add a link if this PR introduces a new example notebook
- [ ] **`cae-notebooks`** — update or add a notebook in `cae-notebooks/` that demonstrates the new or changed functionality
- [ ] **`MAINTAINERS.md`** — update release notes, secrets inventory, or CI table if you changed workflows or infrastructure

## Code Quality
- [ ] Follows PEP8 naming and style conventions
- [ ] Helper functions prefixed with underscore `_`
- [ ] Linting completed and all issues resolved
- [ ] Does not replicate existing functionality
- [ ] Aligns with general coding standards of existing codebase
- [ ] Code generalized for multiple uses (unless too complex/time-intensive)

## Review Process
- [ ] PR review instructions provided:
  - [ ] Type of review requested (scientific/technical/debugging)
  - [ ] Level of review detail needed

## Administrative Reminders
  - Jira ticket moved to "Review" when PR created
  - Jira ticket will be moved to "Done" when complete
  - PR branch will be deleted after merge
