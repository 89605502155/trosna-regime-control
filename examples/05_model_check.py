"""Check of the decision model P(b = 1) = 1 − e^{−λΔt} with the cloglog regression."""

from trosna_regime_control import StepIntensity, UniformIntervals, fit_cloglog, simulate

clean = simulate(UniformIntervals(1.0, 10.0, seed=6), StepIntensity(0.05), 60_000.0, seed=7)
fit = fit_cloglog(clean.decision, clean.dt, burn_in=10)
print("clean model")
print(f"  slope of ln dt       {fit.slope_ln_dt.estimate:.3f}  (model: 1)")
print(f"  previous bit         {fit.coef_prev_bit.estimate:.3f}  (model: 0)")
print(f"  rate                 {fit.rate_hat:.4f} 1/s  (true 0.05)")
print(f"  consistent           {fit.model_consistent}")

dirty = simulate(
    UniformIntervals(1.0, 10.0, seed=8),
    StepIntensity(0.05),
    60_000.0,
    seed=9,
    carryover=0.4,
)
bad = fit_cloglog(dirty.decision, dirty.dt, burn_in=10)
print("demand carried over to the next check")
print(f"  previous bit         {bad.coef_prev_bit.estimate:.3f}  (model: 0)")
print(f"  consistent           {bad.model_consistent}")
