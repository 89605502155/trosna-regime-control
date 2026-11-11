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

Пакет публикуется в PyPI вручную. В репозитории нет автоматической публикации,
API-токен в репозитории не хранится.

1. Укажите версию в `pyproject.toml`, `CITATION.cff` и добавьте запись в `CHANGELOG.md`.
   `trosna_regime_control.__version__` читается из метаданных установленного пакета.
2. Выполните локальные проверки из раздела выше, затем `uv build`.
3. Загрузите своим токеном: `uv publish --token <токен PyPI>`. Токен в репозиторий не добавляйте.
4. Закоммитьте, поставьте тег на коммит релиза и отправьте его:
   `git tag v0.0.1` и `git push origin main --tags`.

Репозиторий: `https://github.com/89605502155/trosna-regime-control`.
