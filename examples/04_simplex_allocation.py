"""Memory shares on the simplex: perturbation, clr coordinates and bounded allocation."""

import numpy as np

from trosna_regime_control import aitchison_distance, allocate, clr, clr_inv, perturb
from trosna_regime_control.simplex import plan

shares = np.array([0.5, 0.3, 0.2])
bump = np.array([1.2, 1.0, 1.0])  # VM 1 asks for 20 % more relative to the others
new = perturb(shares, bump)
print("perturbed shares      ", np.round(new, 4).tolist())
print("clr adds up           ", bool(np.allclose(clr(new), clr(shares) + clr(bump))))
print("back from clr         ", np.round(clr_inv(clr(new)), 4).tolist())

distance = float(aitchison_distance(shares, new))
print(f"Aitchison distance    {distance:.4f}")

# Dead band: move memory only when the target is farther than delta.
decision = plan(shares, new, delta=0.05)
print("reallocate?           ", decision.reallocate)

# Allocation of a 100 GiB-like pool (MiB) with minimum and maximum per VM.
alloc = allocate(
    total=16384.0,
    shares=new,
    m_min=[2048.0, 2048.0, 2048.0],
    m_max=[6144.0, 8192.0, 8192.0],
)
print("memory per VM, MiB    ", np.round(alloc.memory, 1).tolist())
print("unallocated, MiB      ", round(alloc.unallocated, 1))
