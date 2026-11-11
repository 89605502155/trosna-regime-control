# Публикация

[English version](../en/publishing.md) · [README](../../README.ru.md)

Пакет собирается через `uv_build`. Исходный дистрибутив содержит также тесты,
документацию и примеры.

## Локальные проверки

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv build
uvx twine check dist/*
```

## Хук сообщений коммитов

Тема коммита не длиннее 8 символов, тела нет. Хук — символическая ссылка на
`scripts/commit-msg`, ставится один раз на клон:

```sh
sh scripts/install_hooks.sh
```

## Выпуск

1. Обновить версию в `pyproject.toml`, `src/trosna_regime_control/__init__.py`,
   `CITATION.cff` и `CHANGELOG.md`.
2. Закоммитить, поставить тег `vX.Y.Z` и отправить тег.
3. Рабочий процесс `.github/workflows/publish.yml` соберёт пакет и опубликует его
   на PyPI через доверенную публикацию, токен не хранится.

Workflow `ci.yml` при каждом push и pull request запускает проверку стиля,
проверку сообщений коммитов, тесты на Python 3.11–3.14 и сборку.
