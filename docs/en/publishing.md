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

The package is published to PyPI by hand. The repository has no automated
publishing workflow, and no API token is stored in it.

1. Update the version in `pyproject.toml`, `CITATION.cff` and `CHANGELOG.md`.
   `trosna_regime_control.__version__` is read from the installed package metadata.
2. Run the local checks above, then `uv build`.
3. Upload with your own token: `uv publish --token <PyPI token>`. Do not commit the token.
4. Commit, then tag the release commit and push it:
   `git tag v0.0.1` and `git push origin main --tags`.

The repository is `https://github.com/89605502155/trosna-regime-control`.
