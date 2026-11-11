"""Why the 5-bit heuristic flickers: exact Markov analysis and a simulation check."""

from trosna_regime_control import (
    LegacyHeuristic,
    StepIntensity,
    flicker_stats,
    simulate,
    switches_per_hour,
)

print("rate 1/s   switches/h (exact)   share non-stationary")
for rate in (0.02, 0.05, 0.10):
    stats = flicker_stats(rate)
    print(f"{rate:8.2f}   {stats.switches_per_hour:18.1f}   {stats.share_nonstationary:20.3f}")

# The same number from a simulated run of the heuristic itself.
run = simulate(LegacyHeuristic(), StepIntensity(0.05), horizon_s=60_000.0, seed=4)
observed = switches_per_hour(run.decision, run.dt, burn_in=100)
print(f"simulated, 60000 s   {observed:.1f} switches/h at rate 0.05 1/s")
