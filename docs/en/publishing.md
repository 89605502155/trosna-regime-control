# Publishing

[Русская версия](../ru/publishing.md) · [README](../../README.md)

The package is built with `uv_build`. The source distribution also carries the
tests, documentation and examples.

## Local checks

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv build
uvx twine check dist/*
```

## Commit hook

Commit subjects have at most 8 characters and no body. The hook is a symlink to
`scripts/commit-msg`, installed once per clone:

```sh
sh scripts/install_hooks.sh
```

## Release

1. Update the version in `pyproject.toml`, `src/trosna_regime_control/__init__.py`,
   `CITATION.cff` and `CHANGELOG.md`.
2. Commit, then tag `vX.Y.Z` and push the tag.
3. The workflow `.github/workflows/publish.yml` builds the package and publishes
   it to PyPI with trusted publishing, so no API token is stored.

The CI workflow `ci.yml` runs lint, the commit-message check, the tests on
Python 3.11–3.14 and the build on each push and pull request.
