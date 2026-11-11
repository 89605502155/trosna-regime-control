# Быстрый старт

[English version](../en/quickstart.md) · [README](../../README.ru.md)

Примеры ниже запускаются как есть. Каждый блок выполняется во временной папке.

## 1. Интенсивность решений в реальном времени

Бит решения равен 1, если на проверке память перераспределяли. Каждая проверка
приходит с реальным интервалом от предыдущей. `ForgettingRate` хранит три числа
и забывает старые проверки в масштабе реального времени.

```python
from trosna_regime_control import ForgettingRate

est = ForgettingRate(tau=1800.0)  # память около 30 минут
for moved, dt in [(0, 10.0), (1, 8.2), (0, 6.4), (1, 4.6)]:
    est.record_check(moved, dt)
print(f"наивная {est.naive:.4f} 1/с, с поправкой {est.debiased:.4f} 1/с")
```

Наивная оценка `N/T` занижена при большом `λΔt`. Оценка с поправкой обращает
`P(b = 1) = 1 − e^{−λΔt}` и является основной.

## 2. Следующий интервал по правилу

Правило А делает долю `p*` проверок перераспределяющими. Правило Б минимизирует
стоимость проверок и задержки. По умолчанию оба ограничены отрезком `[1 с, 10 с]`.

```python
from trosna_regime_control import rule_a, rule_b

print(float(rule_a(0.05, p_star=0.3)))  # 6.0 с
print(round(float(rule_b(0.05, c1_over_c2=2.0)), 2))  # 8.94 с
```

## 3. Детектор смены режима

`ShiryaevRoberts.for_arl` задаёт порог по нужному среднему времени до ложной
тревоги в секундах. Статистика использует реальные интервалы.

```python
from trosna_regime_control import ShiryaevRoberts

det = ShiryaevRoberts.for_arl(lam0=0.05, kappa=2.0, arl=3600.0)
alarms = [det.record_check(decision, dt) for decision, dt in [(1, 5.0), (0, 5.0), (1, 5.0)]]
print(alarms, round(det.stat, 3))
```

## 4. Весь цикл: потоковый контроллер

`RegimeController` объединяет оценку, детектор и правило. Его запускают один раз,
а затем обновляют на каждой проверке. Он возвращает следующий интервал и режим.

```python
from trosna_regime_control import ControllerConfig, RegimeController

ctrl = RegimeController(ControllerConfig(rule="A", p_star=0.3))
t = 0.0
due = ctrl.start(t)
for moved in [False, True, False, False, True]:
    t += due  # в реальной системе: время только что выполненной проверки
    step = ctrl.record_decision(t, moved)
    due = step.next_dt
print(f"следующая проверка через {step.next_dt:.2f} с, тревога {step.alarm}")
```

Состояние хранится в простом JSON. Сохраняют его через `ctrl.to_json()`, а
восстанавливают через `RegimeController.from_json(text)`.

## 5. Доли памяти на симплексе

Доли положительны и в сумме дают единицу. Групповая операция — `perturb`, её
логарифм — `clr`. `plan` перераспределяет память, только если цель дальше зоны
нечувствительности `delta`.

```python
from trosna_regime_control import allocate, plan_reallocation

decision = plan_reallocation([0.5, 0.5], [0.7, 0.3], delta=0.05)
print(decision.reallocate, [round(x, 2) for x in decision.shares])
alloc = allocate(4096.0, decision.shares, m_min=[512.0, 512.0], m_max=[3584.0, 3584.0])
print([round(x, 1) for x in alloc.memory])
```

## 6. Записанный журнал

```python
import pathlib

from trosna_regime_control import analyze, read_checks

path = pathlib.Path("checks.csv")
path.write_text(
    "t_unix,dt_s,b\n"
    "1763200000.0,10.0,0\n"
    "1763200010.0,10.0,1\n"
    "1763200018.2,8.2,0\n"
    "1763200026.4,8.2,0\n"
    "1763200034.6,8.2,1\n",
    encoding="utf-8",
)
report = analyze(read_checks(path))
print(report.n_checks, round(report.share_moves, 2))
```

Та же сводка из командной строки: `trosna-regime-control analyze checks.csv`.
Параметры — в [cli.md](cli.md), столбцы — в [log_format.md](log_format.md).

## Что дальше

* [theory.md](theory.md): идеи простыми словами и с формулами.
* [api.md](api.md): все публичные имена.
* [migration.md](migration.md): переход с эвристики из пяти битов.
