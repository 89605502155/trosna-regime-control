"""Real-time intensity of decisions: naive versus debiased estimate."""

from trosna_regime_control import (
    ForgettingRate,
    IntervalBounds,
    StepIntensity,
    UniformIntervals,
    rule_a,
    simulate,
)
from trosna_regime_control.scheduler import FixedInterval

# A fixed 5 s interval makes the naive estimate visibly low: naive_bias(0.05, 5) = -11.5 %.
trace = simulate(FixedInterval(5.0), StepIntensity(0.05), horizon_s=40_000.0, seed=1)
est = ForgettingRate(tau=1800.0)
for bit, dt in zip(trace.decision, trace.dt, strict=True):
    est.update(float(bit), float(dt))
print("true rate            0.05000 1/s")
print(f"naive N/T            {est.naive:.5f} 1/s")
print(f"debiased             {est.debiased:.5f} 1/s")

# Rule A: the interval that makes a share p* = 0.3 of the checks move memory.
bounds = IntervalBounds(lo=1.0, hi=10.0)
print(f"rule A interval      {float(rule_a(est.debiased, 0.3, bounds)):.2f} s")

# With variable intervals the same estimator works without a correction.
variable = simulate(UniformIntervals(1.0, 10.0, seed=2), StepIntensity(0.05), 40_000.0, seed=3)
est2 = ForgettingRate(tau=1800.0)
for bit, dt in zip(variable.decision, variable.dt, strict=True):
    est2.update(float(bit), float(dt))
print(f"variable intervals   {est2.debiased:.5f} 1/s (debiased)")
