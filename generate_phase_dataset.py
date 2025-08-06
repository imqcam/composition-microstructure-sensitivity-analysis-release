#!/usr/bin/env python
"""
Generate (or enlarge) a phase-fraction dataset for Ni-base superalloys
using Thermo-Calc’s TC-Python batch-equilibrium interface.
"""

# ------------------ CONFIG ---------------------------------------------------
N_SAMPLES = 500_000  # change for ‘more data’
TEMPERATURE_K = 923.15  # 650 °C
DB_NAME = "TCNI12"  # thermodynamic database
OUTFILE = "Phase_fraction_data_v2.hdf5"
RANGE_PCT = dict(  # ±% variation around a baseline
    Ni=5, Cr=5, Nb=5, Al=10, Ti=10, Co=50, Mo=10
)
BASE_MASS = dict(  # baseline IN-718 in mass-fraction
    Ni=0.55, Cr=0.19, Nb=0.05, Al=0.006, Ti=0.007, Co=0.004, Mo=0.03
)
# -----------------------------------------------------------------------------

import numpy as np, h5py, tc_python, itertools, tqdm, math

elements = list(BASE_MASS.keys())  # order matters!
dependent_element = "Fe"


def random_compositions(n=N_SAMPLES):
    """Return (n, 7) array of mass fractions that sum ≤1 (Fe balances)."""
    rng = np.random.default_rng()
    comps = np.empty((n, len(elements)), np.float32)
    for i, el in enumerate(elements):
        base = BASE_MASS[el]
        pct = RANGE_PCT[el] / 100
        comps[:, i] = base * (1 + rng.uniform(-pct, pct, n))
    # normalise + scale so Σ≤1
    total = comps.sum(axis=1)
    comps /= total[:, None] * (1 / (1 - 1e-6))
    return comps


def run_batch(compositions):
    """Return phase-fraction arrays for γ, γ′, γ′′."""
    with tc_python.TCPython() as s:
        s.select_database(DB_NAME)
        system = (
            s.start_system()
            .with_elements(elements + [dependent_element])
            .select_phase("FCC_L12#1")  # gamma matrix
            .select_phase("FCC_L12#2")  # gamma prime
            .select_phase("NI3TA_D0A#1")  # gamma double prime
            .get_system()
        )
        eq = (
            system.with_batch_equilibrium_calculation()
            .set_condition("T", TEMPERATURE_K)
            .disable_global_minimization()
        )

        cond_blocks = [
            [(f"W({el})", x) for el, x in zip(elements, row)] for row in compositions
        ]
        eq.set_conditions_for_equilibria(cond_blocks)

        q = [
            tc_python.ThermodynamicQuantity.mole_fraction_of_a_phase("FCC_L12#1"),
            tc_python.ThermodynamicQuantity.mole_fraction_of_a_phase("FCC_L12#2"),
            tc_python.ThermodynamicQuantity.mole_fraction_of_a_phase("NI3TA_D0A#1"),
        ]
        res = eq.calculate(q, len(compositions))
        return [res.get_values_of(qi) for qi in q]  # γ, γ′, γ′′


def main():
    conc = random_compositions()
    γ, γp, γpp = run_batch(conc)

    with h5py.File(OUTFILE, "w") as f:
        f.create_dataset("conc", data=conc, dtype="f4", compression="gzip")
        f.create_dataset("Matrix", data=γ, dtype="f4", compression="gzip")
        f.create_dataset("g_prime", data=γp, dtype="f4", compression="gzip")
        f.create_dataset("g_dprime", data=γpp, dtype="f4", compression="gzip")

    print(f"Wrote {len(conc):,} rows → {OUTFILE}")


if __name__ == "__main__":
    main()
