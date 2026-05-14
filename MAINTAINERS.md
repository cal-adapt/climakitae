# Maintainers Guide

This document covers release tagging, deployment, third-party integrations, and ongoing maintenance responsibilities for **climakitae**.

---

## Contents

- [Release Cadence](#release-cadence)
- [Versioning](#versioning)
- [Tagging a Release](#tagging-a-release)
- [Publishing to PyPI](#publishing-to-pypi)
- [Zenodo (DOI Archiving)](#zenodo-doi-archiving)
- [Codecov (Coverage Tracking)](#codecov-coverage-tracking)
- [ReadTheDocs](#readthedocs)
- [CI Workflows Overview](#ci-workflows-overview)
- [Secrets and Credentials](#secrets-and-credentials)
- [Dependency Management](#dependency-management)
- [Branch and PR Strategy](#branch-and-pr-strategy)
- [Checklist: Full Release](#checklist-full-release)

---

## Release Cadence

Publish a new release **approximately every two months**, or sooner when any of the following conditions are met:

- A significant new feature lands in `main`
- A security vulnerability or critical bug is fixed
- A breaking change to the public API is merged
- A required dependency releases a breaking update that we need to track

Minor patch releases (e.g. `1.4.1`) may be issued at any time for regressions or documentation fixes without waiting for the two-month window.

---

## Versioning

climakitae uses **[`setuptools_scm`](https://setuptools-scm.readthedocs.io/)** — the version is derived automatically from git tags. There is no version number to edit manually.

Version format follows [Semantic Versioning](https://semver.org/):

| Change type | Example bump |
|-------------|-------------|
| Bug fix, documentation | `1.4.0` → `1.4.1` |
| New feature, backward-compatible | `1.4.1` → `1.5.0` |
| Breaking API change | `1.5.0` → `2.0.0` |

---

## Tagging a Release

1. **Ensure `main` is green.** All CI checks on the `main` branch must pass before tagging.

2. **Write release notes.** Summarize what changed since the last tag. Include:
   - New features and their notebook/API examples
   - Bug fixes with issue references (e.g. `Fixes #123`)
   - Breaking changes (if any) with migration guidance
   - Dependency bumps that affect users

3. **Create and push the tag:**

   ```bash
   git checkout main
   git pull origin main
   git tag -a v1.5.0 -m "Release v1.5.0 — <one-line summary>"
   git push origin v1.5.0
   ```

   The annotated tag message becomes the default release title on GitHub.

4. **Create a GitHub Release** from the tag:
   - Go to **Releases → Draft a new release** on GitHub.
   - Select the tag you just pushed.
   - Paste the release notes.
   - Click **Publish release**.

   Publishing the release triggers the [`publish.yml`](.github/workflows/publish.yml) workflow, which builds the package and uploads it to PyPI automatically.

5. **Verify PyPI.** After the workflow completes (~5 min), confirm the new version appears at <https://pypi.org/project/climakitae/>.

---

## Publishing to PyPI

The [`publish.yml`](.github/workflows/publish.yml) workflow runs automatically on every published GitHub release. It:

1. Runs the basic test suite (`-m "not advanced"`)
2. Builds the wheel and sdist with `python -m build`
3. Validates the artifacts with `twine check`
4. Publishes via **[PyPA Trusted Publishing](https://docs.pypi.org/trusted-publishers/)** — no API token is stored in the repo

**Trusted Publishing** is configured directly on PyPI under the `climakitae` project. If you need to add a new publisher (e.g. a different org or repo), update the trusted publisher settings at <https://pypi.org/manage/project/climakitae/settings/publishing/>.

To trigger a publish manually without creating a release (e.g. testing the workflow), use the `workflow_dispatch` trigger from the Actions tab.

---

## Zenodo (DOI Archiving)

climakitae is archived on **[Zenodo](https://zenodo.org/)** to provide citable DOIs for academic work.

### How it works

Zenodo is connected to this GitHub repository via the Zenodo GitHub integration. Each time a **GitHub Release** is published, Zenodo automatically archives a snapshot and mints a new DOI for that version. The top-level DOI (the "concept DOI") always resolves to the latest version.

### First-time setup (already done — for reference)

1. Log into Zenodo with the org GitHub account.
2. Go to **Settings → GitHub** and toggle the `cal-adapt/climakitae` repository on.
3. Optionally add a `.zenodo.json` metadata file to the repo root to pre-populate author list, license, keywords, etc. (see below).

### `.zenodo.json`

To control the metadata that Zenodo captures, add a `.zenodo.json` file to the repo root. Example:

```json
{
  "title": "climakitae: Climate Data Analysis Toolkit for Cal-Adapt",
  "description": "A Python toolkit for accessing and analyzing downscaled CMIP6 climate data for California via the Cal-Adapt Analytics Engine.",
  "license": "BSD-3-Clause",
  "creators": [
    {
      "name": "Cal-Adapt Analytics Engine Team",
      "affiliation": "Eagle Rock Analytics / Lawrence Berkeley National Laboratory"
    }
  ],
  "keywords": ["climate", "California", "CMIP6", "downscaling", "WRF", "LOCA2", "xarray"],
  "related_identifiers": [
    {
      "identifier": "https://pypi.org/project/climakitae/",
      "relation": "isSupplementTo",
      "scheme": "url"
    }
  ]
}
```

### Citing

After each release, update the citation badge in `README.md` with the new version DOI from Zenodo.

---

## Codecov (Coverage Tracking)

Test coverage is reported to **[Codecov](https://app.codecov.io/gh/cal-adapt/climakitae)** on every push to `main`.

### How it works

The [`ci-main.yml`](.github/workflows/ci-main.yml) workflow runs `pytest --cov --cov-branch --cov-report=xml` and then uploads the `coverage.xml` artifact to Codecov using `CODECOV_TOKEN`.

### Secrets

`CODECOV_TOKEN` is stored as a repository secret in GitHub Settings → Secrets and Variables → Actions. If the token expires or is rotated, update it there. The token can be found or regenerated at <https://app.codecov.io/gh/cal-adapt/climakitae/settings>.

### Coverage targets

- **Soft target**: ≥ 80% overall coverage
- PRs that drop coverage by more than a few percentage points should include new tests to cover the added code

### Reading coverage reports

- The Codecov bot posts a summary comment on each PR.
- The [`.coveragerc`](.coveragerc) file controls which files are included and excluded from measurement.

---

## ReadTheDocs

Documentation is hosted at <https://climakitae.readthedocs.io/> and built automatically by ReadTheDocs on every push to `main` and on each tagged release.

- Configuration: [`.readthedocs.yaml`](.readthedocs.yaml)
- Local preview: `cd docs-mkdocs && mkdocs serve`
- Full build: `mkdocs build`

For releases, ReadTheDocs creates a versioned docs snapshot matched to the tag. Make sure all new public symbols have NumPy-style docstrings before tagging — the docs build will fail otherwise.

---

## CI Workflows Overview

| Workflow | File | Trigger | What it does |
|----------|------|---------|--------------|
| `ci-main` | [`ci-main.yml`](.github/workflows/ci-main.yml) | Push to `main` | Black lint + full test suite on Python 3.12 & 3.13 + Codecov upload |
| `ci-not-main` | `.github/workflows/ci-not-main.yml` | Push to any non-`main` branch | Black lint + basic tests (`not advanced`); advanced tests only if PR labeled **"Advanced Testing"** |
| `publish` | [`publish.yml`](.github/workflows/publish.yml) | GitHub Release published (or manual dispatch) | Build + publish to PyPI via Trusted Publishing |
| `docs-check` | `.github/workflows/docs-check.yml` | Pull requests | Validates docs build succeeds |

---

## Secrets and Credentials

| Secret name | Used by | Notes |
|-------------|---------|-------|
| `CODECOV_TOKEN` | `ci-main.yml` | Codecov upload; rotate at Codecov settings |
| *(none for PyPI)* | `publish.yml` | Uses PyPA Trusted Publishing — no token needed |

Secrets are managed under **GitHub → Settings → Secrets and Variables → Actions**. Do not commit credentials or tokens to the repository.

---

## Dependency Management

climakitae uses **[uv](https://docs.astral.sh/uv/)** for environment management. The lock file (`uv.lock`) pins all transitive dependencies for reproducible installs.

- After adding or removing a dependency in `pyproject.toml`, run `uv sync` to update the lock file and commit both files together.
- Keep `uv.lock` committed so CI and contributors get the exact same environment.
- Review the `dependencies` list in `pyproject.toml` before each release. Remove version upper-bounds that are no longer necessary, and tighten any that are required for correctness.

---

## Branch and PR Strategy

- `main` is the stable, deployable branch. Direct pushes are prohibited; all changes come through PRs.
- Feature branches: `feature/<short-description>`
- Bug fix branches: `fix/<short-description>`
- Documentation branches: `docs/<issue-or-description>`
- PRs require at least one review approval before merge.
- Squash-merge is preferred to keep the `main` history readable.

---

## Checklist: Full Release

Use this checklist when cutting a release:

- [ ] All CI checks on `main` are green
- [ ] `uv.lock` is up to date and committed
- [ ] CHANGELOG / release notes drafted
- [ ] NumPy docstrings present on all new public symbols
- [ ] `black climakitae/ tests/` and `isort climakitae/ tests/` pass locally
- [ ] Advanced tests pass (`uv run python -m pytest -m "advanced"`)
- [ ] ReadTheDocs build preview looks correct (`mkdocs serve`)
- [ ] `.zenodo.json` author list and metadata is accurate
- [ ] Version bump is correct per semver (setuptools_scm reads the tag automatically)
- [ ] Annotated git tag pushed: `git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z`
- [ ] GitHub Release published from the tag (triggers PyPI publish)
- [ ] New version visible on PyPI within ~5 minutes
- [ ] New Zenodo DOI visible after release (check <https://zenodo.org/search?q=climakitae>)
- [ ] Codecov badge and citation DOI in `README.md` updated if needed
- [ ] Relevant `cae-notebooks` notebooks tested against the new release
