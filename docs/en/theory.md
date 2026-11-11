# Theory

[Русская версия](../ru/theory.md) · [README](../../README.md)

This page explains the four ideas of the article and what the code does with
them. The formulas follow the document `idei_gruppy_li_martingaly.md`. The
theory is in the document. Here we keep only what the code relies on.

## The problem in plain words

A manager moves memory between virtual machines by ballooning. Each move costs
time. The manager checks the demand at intervals. The current rule sets the
next interval from the number of moves in the last five checks, and it sets
the regime from the same number. Both thresholds are hand-made.

The library replaces the two decisions with methods that have a rationale:

| Decision | Current rule | Replacement |
|---|---|---|
| interval to the next check | table by `S` (5 bits) | `Δt = p*/λ̂` or `Δt = √(2C₁/(λ̂C₂))` |
| regime | `S ≥ 3` | Shiryaev–Roberts detector on the decisions |
| shares of memory | not specified | clr coordinates with bounds and a dead band |

## Idea 1. Memory shares on the simplex

Shares `w` of the pool are positive and sum to one. The operation
`(w ⊕ p)_i = w_i p_i / Σ_j w_j p_j` (the function `perturb`) is a group
operation, the neutral element is equal shares. The logarithm of the group is
the centred log-ratio transform, the exponential is the softmax:

    z_i = ln w_i − mean(ln w),      w = softmax(z)

In these coordinates a step needs no constraints, and softmax always returns
valid shares. The Aitchison distance `d = ‖clr(w₁) − clr(w₂)‖` (function
`aitchison_distance`) does not depend on the total memory. It measures a
relative change: a share growing from 1 % to 2 % gives `d ≈ 0.50`, from 40 %
to 41 % only `d ≈ 0.03`. The dead band `δ` is therefore a distance in
clr coordinates, and it has to be calibrated on traces.

The group is commutative, so the geometry is `ℝⁿ⁻¹` in disguise. The Lie
language justifies the method and makes the formulas short. It brings no new
results.

## Idea 2. The interval from scale invariance and costs

If all memory sizes and all units of time are multiplied by a constant, the
rule must give the same relative result. So the rule uses dimensionless
quantities and a cost model.

* **Rule A**: `Δt = p* / λ̂`. The share of checks with a move tends to `p*`.
  The Poisson model is not needed for this.
* **Rule B**: `Δt = √(2 C₁ / (λ̂ C₂))`. It minimises the cost rate
  `J(Δt) = C₁/Δt + λC₂Δt/2`, where `C₁` is the cost of a check and `C₂` the
  cost of a second of delay. Only the ratio `C₁/C₂` matters. The exact
  optimum of Barlow, Hunter and Proschan solves `e^{λx} − 1 − λx = λC₁/C₂`
  (function `exact_interval`).
* **Rule C**: `Δt = δ / v`, with `v` the drift of the shares in the Aitchison
  distance.

All rules are clipped to `[1 s, 10 s]`, as the current table is.

## Idea 3. Decision intensity with forgetting in real time

The sum of the five bits is invariant under permutations, so it ignores the
order of the events. Its expectation also grows with the interval: at
`λ = 0.05 1/s` it is about 0.24 at `Δt = 1 s` and about 1.97 at `Δt = 10 s`.
So the sum sets the interval, and the interval changes the sum. The regime
switches often even at constant load. The exact count is in
`flicker_stats`: 57 switches per hour at `λ = 0.05 1/s` and 137 at
`λ = 0.1 1/s`, for the conditional table of the document.

The replacement is an estimate with exponential forgetting in real time:

    N_k = e^{−Δt_k/τ} N_{k−1} + b_k
    T_k = e^{−Δt_k/τ} T_{k−1} + Δt_k
    λ̂_k = N_k / T_k

Exponential weights are the only ones that match a linear recursion with one
number at any intervals (Cauchy's functional equation). The naive ratio is
biased low when `λΔt` is large. Its relative bias is `(1 − e^{−x})/x − 1` with
`x = λΔt`: −2.5 % at `Δt = 1 s`, −11.5 % at `5 s`, −21 % at `10 s` for
`λ = 0.05 1/s`. The debiased estimate is `−ln(1 − p̂)/Δ̄`, with `p̂ = N/K` the
share of moves and `Δ̄ = T/K` the mean interval. It is exact for a constant
interval (`ForgettingRate.debiased`).

## Idea 4. Martingale detector of the regime

Under the model `P(b_k = 1 | past) = 1 − e^{−λΔt_k}`, the factor of the
likelihood ratio between `λ = κλ₀` and `λ = λ₀` is

    f_k = (1 − e^{−κλ₀Δt_k}) / (1 − e^{−λ₀Δt_k})   if b_k = 1
    f_k = e^{−(κ−1)λ₀Δt_k}                          if b_k = 0

Under the null its conditional expectation is 1. So `Λ_n = f_1 ⋯ f_n` is a
martingale, and this holds even when the intervals are chosen from the past.
The library uses the time-weighted Shiryaev–Roberts statistic:

    R_k = (Δt_k + R_{k−1}) f_k

Its mean time to a false alarm is at least the threshold `A` in seconds
(Pollak, 1985, and the stopping theorem). The CUSUM `S_k = max(0, S_{k−1} + ln f_k)`
with `h = ln A` has the same kind of guarantee, and it is more conservative.
Two detectors run together: a rise with `κ > 1` and a fall with `κ < 1`.

Three remarks keep the guarantee honest.

1. **The model.** The guarantee holds under `P(b = 1) = 1 − e^{−λΔt}`, that is,
   without carry-over of unresolved demand. `trosna-regime-control analyze` checks it
   with `fit_cloglog`: the slope of `ln Δt` must be near 1 and the coefficient
   of the previous bit near 0. A carry-over of 0.4 fails the check.
2. **The baseline `λ₀`.** It is estimated from the same data, so the guarantee
   is approximate. The controller takes the upper confidence bound `λ₀⁺` for
   the rise detector and the lower bound `λ₀⁻` for the fall detector. For the
   rise the null is `λ ≤ λ₀`, for the fall it is `λ ≥ λ₀`, and the expected
   factor respects both (checked numerically).
3. **Speed.** The Lorden bound `ln(ARL)/I` is asymptotic. At moderate ARL the
   real delay is smaller, sometimes twice as small. Measure it with
   `run_lengths` for the chosen threshold.

## How they fit together: the controller

`RegimeController` takes a check `(t, b)` and does this:

1. It updates `λ̂` with the real interval `Δt = t − t_prev`.
2. It updates the detectors. After an alarm the regime is non-stationary for
   `hold_s`, the interval is the lower bound (1 s), and the baselines are reset
   to the fast estimate.
3. In the stationary regime the interval comes from rule A or rule B.

The guarantees apply to the detector. Rule A and rule B are the design of the
interval, not a claim about the optimal policy under changing demand.

## What is not claimed

* The group structure is commutative and adds no new theorems.
* Optimality of CUSUM and Shiryaev–Roberts holds for independent observations
  at a constant step. With adaptive intervals the guarantee is the mean time to
  a false alarm, not optimality.
* The method is not a memory growth estimator. That is `azarovo-martingale`.

## References

Aitchison (1986); Pawlowsky-Glahn and Egozcue (2001); Kivinen and Warmuth
(1997); Barlow, Hunter and Proschan (1963); Anta and Tabuada (2010); Shiryaev
(1976); Page (1954); Lorden (1971); Pollak (1985); Polunchenko and Tartakovsky
(2010); Ramdas et al. (2023). The full list is in the document.
