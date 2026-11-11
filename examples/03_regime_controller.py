"""Streaming regime controller on a fourfold rise of the decision rate."""

from trosna_regime_control import (
    ControllerConfig,
    RegimeController,
    StepIntensity,
    replay,
    simulate,
)

config = ControllerConfig(
    rule="A",
    p_star=0.3,
    arl_s=1e5,
    warmup_s=600.0,
    hold_s=120.0,
    tau_s=900.0,
    tau_fast_s=120.0,
)
change = 3000.0
intensity = StepIntensity(0.02, 0.08, t_change=change)
trace = simulate(RegimeController(config), intensity, horizon_s=6000.0, seed=5)

controller = RegimeController(config)
steps = replay(controller, trace)
alarms = [t for t, s in zip(trace.t, steps, strict=True) if s.alarm == "rise"]
print(f"checks                 {len(trace)}")
print(f"rate changes at        {change:.0f} s")
print(f"first rise alarm       {alarms[0]:.0f} s" if alarms else "no alarm")
print(f"delay                  {alarms[0] - change:.0f} s" if alarms else "")
print(f"alarms before change   {sum(1 for t in alarms if t <= change)}")

# The state is plain JSON: a daemon can stop and continue without losing the statistics.
state = controller.to_json()
restored = RegimeController.from_json(state)
print(f"state size             {len(state)} characters, restored checks {restored.n_checks}")
