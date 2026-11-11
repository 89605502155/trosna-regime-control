# trosna-regime-control

[![PyPI](https://img.shields.io/pypi/v/trosna-regime-control.svg)](https://pypi.org/project/trosna-regime-control/)
[![Python](https://img.shields.io/pypi/pyversions/trosna-regime-control.svg)](https://pypi.org/project/trosna-regime-control/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/89605502155/trosna-regime-control/blob/main/LICENSE)

**Определение режима по решениям о перераспределении при нерегулярных проверках, масштабно-инвариантные интервалы мониторинга и распределение памяти на симплексе (геометрия Эйчисона) для адаптивного управления памятью KVM.**

Адаптивный распределитель памяти виртуальных машин проверяет спрос через
интервалы и перераспределяет память баллунингом, когда спрос изменился. После
каждой проверки остаётся один бит решения: 1, если память перераспределяли, и 0,
если нет. trosna-regime-control превращает этот поток битов в три вещи:

* **режим**: стационарна ли интенсивность решений или меняется. Используется
  детектор Ширяева–Робертса с гарантированным средним временем до ложной
  тревоги. Он работает с нерегулярными интервалами, которые задаёт сама политика;
* **следующий интервал**: по оценке интенсивности решений в реальном времени с
  экспоненциальным забыванием и по масштабно-инвариантным правилам
  `Δt = p*/λ̂` и `Δt = √(2C₁/(λ̂C₂))`, а не по ручной таблице;
* **распределение**: доли памяти как точки симплекса, возмущаемые в clr-координатах,
  с минимумами, максимумами и зоной нечувствительности δ против лишних перераспределений.

Библиотека также диагностирует эвристику из пяти битов, которую заменяет: точную
частоту переключений режима через цепь Маркова и воспроизведение обоих методов
на записанном журнале.

Английская версия: [README.md](https://github.com/89605502155/trosna-regime-control/blob/main/README.md).
Документация: [English](https://github.com/89605502155/trosna-regime-control/tree/main/docs/en) ·
[Русский](https://github.com/89605502155/trosna-regime-control/tree/main/docs/ru) ·
[Журнал изменений](https://github.com/89605502155/trosna-regime-control/blob/main/CHANGELOG.md) ·
[Задачи](https://github.com/89605502155/trosna-regime-control/issues)

Библиотека сопровождает статью о группах Ли и мартингалах в адаптивном управлении
памятью KVM. Авторы — Андрей Ферубко и Олег Дмитриевич Казаков, Брянский государственный
инженерно-технологический университет (БГИТУ). Теория описана в
[docs/ru/theory.md](https://github.com/89605502155/trosna-regime-control/blob/main/docs/ru/theory.md).

## Установка

```sh
pip install trosna-regime-control
# или с uv
uv add trosna-regime-control
```

Нужны Python 3.11 или новее и NumPy. Дополнительный набор `plot` добавляет
matplotlib для скриптов экспериментов.

## Быстрый старт

```python
from trosna_regime_control import ControllerConfig, RegimeController

controller = RegimeController(ControllerConfig(rule="A", p_star=0.3))
t = 0.0
interval = controller.start(t)  # секунды до первой проверки
for _ in range(20):
    t += interval
    moved = False  # реальное решение системы: True, если память перераспределяли
    step = controller.record_decision(t, moved)
    interval = step.next_dt
print(f"следующая проверка через {step.next_dt:.1f} с, нестационарный режим: {step.nonstationary}")
```

По записанному журналу (`checks.csv`, см. [docs/ru/log_format.md](https://github.com/89605502155/trosna-regime-control/blob/main/docs/ru/log_format.md)):

```sh
trosna-regime-control analyze checks.csv --rule A --p-star 0.3
trosna-regime-control flicker --heuristic config/heuristic.json
```

## Чем отличается от существующих пакетов

Библиотеки для композиционных данных, например `scikit-bio`
(`skbio.stats.composition`), `composition_stats` и `pyCoDA`, реализуют
центрированное и изометрическое логарифмические преобразования. Они не
распределяют ресурсы: в них нет ограничений на доли, зоны нечувствительности
против лишних перераспределений и интервала мониторинга по решениям. Библиотеки
поиска разладки, например `ruptures`, `ocpdet` и `changepoint-online`, считают
выборку регулярной и, как правило, данные гауссовыми. Ни одна из них не работает
с бернуллиевскими решениями при нерегулярных интервалах, которые задаёт политика,
и не даёт гарантии частоты ложных тревог.

| Возможность | trosna-regime-control | библиотеки композиционных данных | библиотеки поиска разладки |
|---|---|---|---|
| clr, softmax, расстояние Эйчисона | да | да | нет |
| Распределение памяти на симплексе с ограничениями и зоной δ | да | нет | нет |
| Интервал мониторинга по масштабно-инвариантным правилам, модель затрат `C₁/C₂` | да | нет | нет |
| Интенсивность решений в реальном времени с забыванием и смещением | да | нет | нет |
| Ширяев–Робертс и CUSUM для бернуллиевских решений при нерегулярных интервалах | да | нет | нет, они считают выборку регулярной |
| Нижняя граница ARL для адаптивных интервалов | да, при модели | нет | нет |
| Точное мигание эвристики из пяти битов (цепь Маркова) | да | нет | нет |
| Потоковый контроллер с состоянием в JSON и командная строка | да | нет | частично |

Пакет не оценивает рост памяти интервалами Азюмы–Хёффдинга и ставочными
мартингалами. Это задача `azarovo-martingale`. trosna-regime-control относится к режиму
потока решений.

## Разработка

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest            # быстрые тесты; Монте-Карло помечены как slow
uv run pytest -m slow    # Монте-Карло проверка ARL (минуты)
```

Сообщения коммитов в этом репозитории: тема не длиннее 8 символов, без тела.
Хук ставится один раз на клон командой `scripts/install_hooks.sh`.

## Цитирование

См. [CITATION.cff](https://github.com/89605502155/trosna-regime-control/blob/main/CITATION.cff)
или кнопку «Cite this repository» на GitHub.

## Авторы

- Ферубко Андрей, [ORCID 0009-0006-5625-137X](https://orcid.org/0009-0006-5625-137X)
- Казаков Олег Дмитриевич, [ORCID 0000-0001-9665-8138](https://orcid.org/0000-0001-9665-8138)

Брянский государственный инженерно-технологический университет (БГИТУ).

## Лицензия

MIT. Copyright (c) 2026 Andrey Ferubko, Oleg Kazakov. См. [LICENSE](https://github.com/89605502155/trosna-regime-control/blob/main/LICENSE).
