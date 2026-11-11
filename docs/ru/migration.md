# Переход: от эвристики из пяти битов к λ̂ и детектору

[English version](../en/migration.md) · [README](../../README.ru.md)

Текущая система пишет один бит на проверку и задаёт интервал и режим по сумме `S`
последних пяти битов. Переход сохраняет тот же журнал, поэтому его можно сделать
в три шага, и каждый шаг обратим.

## Имена и уровень доверия базовых уровней (версия 0.0.1)

Первый выпуск — версия 0.0.1. Потоковые методы называются так, как показано ниже.
Устаревших синонимов нет: черновик не публиковался.

| Черновик | Выпуск (0.0.1) |
|---|---|
| `RegimeController.update(t, decision)` | `record_decision(t, decision)` |
| `Scheduler.update`, `FixedInterval.update`, `LegacyHeuristic.update`, `UniformIntervals.update` | `record_decision` |
| `ForgettingRate.update(decision, dt)` | `record_check(decision, dt)` |
| `ShiryaevRoberts.update`, `Cusum.update`, `TwoSided.update` | `record_check` |
| `upper_naive(z)`, `lower_naive(z)` | `upper_naive()`, `lower_naive()`; `z` задаётся в конструкторе |
| `ControllerConfig(baseline_z=1.645)` | `confidence=0.95` (по умолчанию, `z = 1,6449`), или `z=` |
| `RegimeController(config)` | также `RegimeController(config, confidence=…)` или `z=…` |
| состояние версии 1 | состояние версии 2: `z` в состоянии оценки, `confidence` и `z` в настройках |

Состояние версии 1 не восстанавливается. Контроллер запускают заново и прогоняют
журнал (шаг 2). Новый `record_decision` не принимает дробное решение и время, которое
не растёт: это `TypeError` и `ValueError`.

Уровень по умолчанию заменяет прежнюю константу `z = 1,645`. Новое `z` при 95 %
равно `1,6449`, базовые уровни меняются меньше чем на 0,05 %.

Чтобы изменить уровень:

```python
from trosna_regime_control import ControllerConfig, RegimeController

strict = RegimeController(ControllerConfig(rule="A"), confidence=0.99)  # меньше тревог
```

В командной строке: `trosna-regime-control analyze checks.csv --confidence 0.99`, либо
`--z 2.33`, чтобы задать число стандартных отклонений напрямую.

## Шаг 1. Измерить текущую систему

1. Задать в файле настроек эвристики реальные таблицу и порог, поставить
   `is_real = true`, затем запустить `trosna-regime-control flicker`. Так получают
   переключения в час для наблюдаемых интенсивностей.
2. Вести журнал `checks.csv` по [log_format.md](log_format.md).
3. Запустить `trosna-regime-control analyze checks.csv`. Модель должна проходить
   проверку (`model consistent yes`). Если нет, гарантия детектора приближённая, и
   отчёт это показывает.

## Шаг 2. Воспроизвести на записанном журнале

Подать те же решения новому контроллеру и сравнить с записанным режимом.
Систему пока не меняем.

```python
import pathlib

from trosna_regime_control import (
    ControllerConfig,
    LegacyHeuristic,
    RegimeController,
    flicker_stats,
    read_checks,
    replay,
)

pathlib.Path("checks.csv").write_text(  # заглушка вместо записанного журнала системы
    "t_unix,dt_s,b\n1763200000.0,10.0,0\n1763200010.0,10.0,1\n1763200018.2,8.2,0\n",
    encoding="utf-8",
)
trace = read_checks("checks.csv")
new = replay(RegimeController(ControllerConfig(rule="A", p_star=0.3)), trace)
old = replay(LegacyHeuristic(), trace)
print(sum(s.nonstationary for s in new), sum(s.nonstationary for s in old))
print(
    round(flicker_stats(0.05).switches_per_hour, 1),
    "переключений в час, текущее правило при 0,05 1/с",
)
```

На журнале новый режим должен меняться заметно реже старого.

## Шаг 3. Заменить правило в системе

Соответствие старых частей новым:

| Было | Стало | Примечание |
|---|---|---|
| таблица `S → Δt` | интервал `RegimeController`, правило А или Б | `rule_a`, `rule_b` |
| `S ≥ 3` → нестационарный | тревога `TwoSided`, `hold_s` | `arl_s` задаёт уровень ложных тревог |
| история из пяти битов | `ForgettingRate` с `tau_s` | состояние — один словарь |
| ручная `1 с` после изменения | `bounds.lo` в нестационарном режиме | |

Внутри цикла мониторинга:

```python
from trosna_regime_control import ControllerConfig, RegimeController

controller = RegimeController(ControllerConfig(rule="A", p_star=0.3, arl_s=3600.0))
interval = controller.start(0.0)


def on_check(now: float, moved: bool) -> float:
    """Вернуть интервал до следующей проверки."""
    global interval
    step = controller.record_decision(now, moved)
    interval = step.next_dt
    return interval
```

Состояние между перезапусками сохраняют через `controller.to_json()` и
восстанавливают через `RegimeController.from_json(text)`.

## Что меняется в результатах

* Интервал зависит от оценённой интенсивности, а не от таблицы.
* Режим меняется только по тревоге, а не на каждой проверке. Частоту задаёт `arl_s`.
* Правилам А и Б нужна оценка `λ̂`. Её память `tau_s` должна быть в несколько раз
  больше самого длинного значимого интервала, примерно 30 минут в документе.

## Проверка перед переключением

* Трасса нового контроллера на журнале остаётся в границах.
* Проверка модели согласована на стационарной части журнала.
* Число тревог до известного изменения нагрузки около нуля при выбранном `arl_s`.
  Для уровня ложных тревог используют `run_lengths`.
