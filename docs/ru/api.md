# API

[English version](../en/api.md) · [README](../../README.ru.md)

Все публичные имена импортируются из `trosna_regime_control`. Массивы — NumPy.
Время в секундах, интенсивности в 1/с, переключения в час.

## Симплекс и распределение (идея 1)

| Имя | Назначение |
|---|---|
| `closure_shares(x)` | нормировка положительных весов в доли |
| `perturb(w, p)` | групповая операция `w ⊕ p` |
| `clr(w)`, `clr_inv(z)` | логарифм (центрированное преобразование) и экспонента (softmax) |
| `aitchison_distance(w1, w2)` | расстояние `‖clr(w1) − clr(w2)‖` |
| `plan_reallocation(current, target, delta)` | решение с зоной нечувствительности, возвращает `Reallocation` |
| `allocate(total, shares, m_min, m_max)` | память по ВМ с ограничениями, возвращает `Allocation` |

```python
import numpy as np
from trosna_regime_control import allocate, clr, clr_inv, perturb

w = np.array([0.5, 0.3, 0.2])
p = np.array([1.2, 1.0, 1.0])
assert np.allclose(clr(perturb(w, p)), clr(w) + clr(p))
assert np.allclose(clr_inv(clr(w)), w)
print(allocate(1000.0, w, [100.0, 100.0, 100.0]).memory)
```

## Интервалы (идея 2)

| Имя | Назначение |
|---|---|
| `IntervalBounds(lo, hi)` | границы интервала, по умолчанию 1 с … 10 с |
| `rule_a(rate, p_star)` | `Δt = p*/λ̂` |
| `rule_b(rate, c1_over_c2)` | `Δt = √(2C₁/(λ̂C₂))` |
| `rule_c(drift, delta)` | `Δt = δ/v` |
| `exact_interval(rate, c1_over_c2)` | точный оптимум затрат, бисекция, без SciPy |

```python
from trosna_regime_control import IntervalBounds, exact_interval, rule_a, rule_b

bounds = IntervalBounds(lo=1.0, hi=10.0)
print(float(rule_a(0.05, 0.3, bounds)), round(float(rule_b(0.05, 2.0, bounds)), 2))
print(round(exact_interval(0.05, 2.0), 2))
```

## Интенсивность (идея 3)

`ForgettingRate(tau, prior_rate=None, prior_time=0.0, prior_dt=5.0)`:

* `update(decision, dt) -> float`: учесть проверку, вернуть наивную оценку;
* `naive`, `debiased`: оценки, 1/с (NaN до данных);
* `upper_naive(z)`, `lower_naive(z)`: доверительные границы наивной оценки;
* `to_dict()`, `from_dict(data)`: состояние для хранения.

`naive_bias(rate, dt)` даёт относительное смещение наивной оценки.

## Детекторы (идея 4)

| Имя | Назначение |
|---|---|
| `lr_factor(decision, dt, lam0, kappa)` | множитель отношения правдоподобия `f` |
| `ShiryaevRoberts(lam0, kappa, threshold, time_weighted=True)` | статистика `R`, тревога при `R ≥ threshold`, затем сброс |
| `ShiryaevRoberts.for_arl(lam0, kappa, arl)` | порог, равный нужному ARL, в секундах |
| `Cusum(lam0, kappa, h)` | статистика CUSUM |
| `cusum_threshold_for_arl(arl)` | `h = ln(ARL)` |
| `TwoSided(lam0_up, threshold, kappa_up, kappa_down, time_weighted, lam0_down)` | рост и спад; `update` возвращает `+1`, `−1` или `0` |
| `run_lengths(kind, threshold, lam_true, lam0, kappa, dt, n_runs, seed)` | Монте-Карло длины пробега, векторизовано |

Функции `lorden_delay_bound`, `kl_rate_poisson` и `kl_per_check` лежат в
`trosna_regime_control.detectors`.

```python
from trosna_regime_control import ShiryaevRoberts

det = ShiryaevRoberts.for_arl(0.05, 2.0, arl=3600.0)
print(det.update(1, 10.0), round(det.stat, 3))
```

## Мигание эвристики (идея 3)

| Имя | Назначение |
|---|---|
| `flicker_stats(rate, config=None)` | точные переключения в час, доля нестационарных проверок, средний интервал |
| `switches_per_hour(decisions, dt, config, burn_in)` | то же по записанной последовательности |
| `HeuristicConfig(table, threshold, n_bits, is_real)` | таблица правила из пяти битов |
| `load_heuristic_config(path)` | чтение таблицы из JSON |
| `default_config()`, `DEFAULT_TABLE` | условная таблица документа |
| `window_sums(bits)` | сумма `S` после каждой проверки |
| `LegacyHeuristic(config)` | правило из пяти битов как планировщик |

```python
from trosna_regime_control import flicker_stats

print(round(flicker_stats(0.05).switches_per_hour, 1))
```

## Контроллер

`ControllerConfig` содержит поля `tau_s`, `rule` (`"A"` или `"B"`), `p_star`,
`c1_over_c2`, `bounds`, `arl_s`, `kappa_up`, `kappa_down`, `hold_s`,
`warmup_s`, `tau_fast_s`, `min_rate`, `baseline_z`, `prior_rate`,
`prior_time_s`. Описание каждого поля — в докстринге.

`RegimeController(config)`:

* `start(t0) -> float`: начать, вернуть первый интервал;
* `update(t, decision) -> Step`: учесть проверку;
* `baselines`, `rate`, `n_checks`, `n_alarms`: состояние;
* `to_dict()`, `from_dict(data)`, `to_json()`, `from_json(text)`: сохранение и восстановление.

`Step` содержит `t_next`, `next_dt`, `nonstationary`, `alarm` (`"rise"`, `"fall"`
или `None`) и `rate`.

## Планировщики и моделирование

`Scheduler` — протокол с методами `start` и `update`. Его реализуют `FixedInterval(dt)`,
`LegacyHeuristic` и `RegimeController`. `replay(scheduler, trace)` подаёт
записанную трассу. `simulate(scheduler, StepIntensity(...), horizon_s, seed,
carryover)` прогоняет цикл на пуассоновском спросе. `UniformIntervals(lo, hi, seed)`
даёт случайные интервалы, которые нужны для проверки модели.

## Трассы и анализ

`read_checks(path) -> Trace` читает `checks.csv`. `Trace` содержит массивы `t`,
`dt`, `decision` и необязательные `legacy_mode` и `s_sum`.
`analyze(trace, controller_config, legacy_config, burn_in) -> TraceReport` даёт
сводку. `fit_cloglog(decisions, dt, burn_in) -> CloglogFit` проверяет модель.
`ones_share_by_interval` даёт долю перераспределений по группам интервалов.

## Ошибки

Неверные параметры вызывают `ValueError` с сообщением. Вызов `update` до `start`
вызывает `RuntimeError`. Файл неверного формата вызывает `ValueError` из `read_checks`.
