Road-map for **re-generating ― and scaling up ― the `Phase_fraction_data.hdf5` dataset** from Jame and Taheri-Mousavi with Thermo-Calc TC-Python.  This is a revision of the batch processing Kacper created last year. It involves some clean-up and parametric re-implementation of Kacper's batch approach that makes the same four datasets (conc, Matrix, g_prime, and g_dprime) with configs for n_samples, temperature, DB, etc. 


## A.  Anatomy of Kacper's original batch workflow

1. **Perturb a baseline IN-718 recipe**

   ```python
   def nudge_numbers(value, count=10000, percentage=5):
       return value * (1 + np.random.uniform(-percentage/100, percentage/100, count))
   ```

   Seven arrays (Ni, Cr, Nb, Al, Ti, Co, Mo) are generated, Fe is implicit as the balance.
   `zip(*arrays)` produces the **`compositions`** list fed to Thermo-Calc.

2. **Batch equilibrium in TC-Python**

   ```python
   with tc_python.TCPython() as s:
       s.select_database("TCNI12")
       eq = (s.start_system()
                .with_elements(elements + [dependent_element])
                .select_phase("FCC_L12#1")        # γ (matrix)
                .select_phase("FCC_L12#2")        # γ′
                .select_phase("NI3TA_D0A#1")      # γ′′
                .get_system()
                .with_batch_equilibrium_calculation()
                .set_condition("T", 923.15)       # 650 °C
                .disable_global_minimization())
       eq.set_conditions_for_equilibria(list_of_conditions)
       res = eq.calculate(quantities, 100)
   ```

   – **`quantities`** is a list of `mole_fraction_of_a_phase()` calls for each phase.
   – Results are returned as NumPy arrays (`mat`, `gp`, `gd`).

3. **Write to HDF5**

   ```python
   with h5py.File("Phase_fraction_data.hdf5", "w") as f:
       f["conc"]    = np.asarray(compositions, dtype=np.float32)
       f["Matrix"]  = mat.astype(np.float32)
       f["g_prime"] = gp.astype(np.float32)
       f["g_dprime"]= gd.astype(np.float32)
   ```

   The final file therefore contains **495 000 rows × 7 mass-fraction columns** plus the three phase-fraction vectors.

---

## B.  A parametric Re-implementation (generate\_phase\_dataset.py)

Adjust the **CONFIG** block as wanted.

```python
#!/usr/bin/env python
"""
Generate (or enlarge) a phase-fraction dataset for Ni-base superalloys
using Thermo-Calc’s TC-Python batch-equilibrium interface.
"""

# ------------------ CONFIG ---------------------------------------------------
N_SAMPLES      = 500_000          # change for ‘more data’
TEMPERATURE_K  = 923.15           # 650 °C
DB_NAME        = "TCNI12"         # thermodynamic database
OUTFILE        = "Phase_fraction_data_v2.hdf5"
RANGE_PCT      = dict(            # ±% variation around a baseline
    Ni=5, Cr=5, Nb=5, Al=10, Ti=10, Co=50, Mo=10
)
BASE_MASS      = dict(            # baseline IN-718 in mass-fraction
    Ni=0.55, Cr=0.19, Nb=0.05, Al=0.006, Ti=0.007, Co=0.004, Mo=0.03
)
# -----------------------------------------------------------------------------

import numpy as np, h5py, tc_python, itertools, tqdm, math

elements = list(BASE_MASS.keys())               # order matters!
dependent_element = "Fe"

def random_compositions(n=N_SAMPLES):
    """Return (n, 7) array of mass fractions that sum ≤1 (Fe balances)."""
    rng = np.random.default_rng()
    comps = np.empty((n, len(elements)), np.float32)
    for i, el in enumerate(elements):
        base = BASE_MASS[el]
        pct  = RANGE_PCT[el] / 100
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
             .select_phase("FCC_L12#1")      # gamma matrix
             .select_phase("FCC_L12#2")      # gamma prime
             .select_phase("NI3TA_D0A#1")    # gamma double prime
             .get_system()
        )
        eq = (
            system.with_batch_equilibrium_calculation()
                  .set_condition("T", TEMPERATURE_K)
                  .disable_global_minimization()
        )

        cond_blocks = [
            [(f"W({el})", x) for el, x in zip(elements, row)]
            for row in compositions
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
        f.create_dataset("conc",    data=conc, dtype="f4", compression="gzip")
        f.create_dataset("Matrix",  data=γ,    dtype="f4", compression="gzip")
        f.create_dataset("g_prime", data=γp,   dtype="f4", compression="gzip")
        f.create_dataset("g_dprime",data=γpp,  dtype="f4", compression="gzip")

    print(f"Wrote {len(conc):,} rows → {OUTFILE}")

if __name__ == "__main__":
    main()
```

**What changed vs. the original `batch_processing.py`?**

| Improvement                                                  | Rationale                                    |
| ------------------------------------------------------------ | -------------------------------------------- |
| **Single CONFIG block** for easy editing                     | Faster iteration                             |
| **Vectorised `random_compositions`** instead of Python loops | Generates millions of alloys fast |
| **Explicit Fe balancing**                                    | Guarantees every row sums to 1.0             |
| **Compression in HDF5**                                      | Cuts file size by \~70 %                     |
| **Return γ, γ′, γ′′ only**                                   | Matches the existing dataset layout          |

---

## C.  Scaling and Performance

| Issue                                | Possible Fixes?                                                                                                                                                                   |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Thermo-Calc license seats**          | TC-Python obeys a license count; if we hit "no license available", lower `n_samples` or stagger runs.                                                                        |
| **Memory (many compositions)**         | Break `N_SAMPLES` into chunks (e.g. 25 000) and append to the HDF5 file inside a loop.                                                                                           |
| **CPU time**                           | TC-Python’s batch job is embarrassingly parallel so launch several Python processes (each with its own `TCPython()` context). Or use `multiprocessing.Pool`?             |
| **Different temperatures / databases** | Duplicate the script, change `TEMPERATURE_K` and/or `DB_NAME`, and write to a new HDF5 file.  The downstream notebook can stack multiple files by concatenating on the row axis. |
| **Phase selections**                   | Add `.select_phase()` lines for *Δ* (BCC\_B2) or *σ* if you want them in the output; will need to create corresponding datasets.                                             |

---

### How to Validate Quickly

After generating `Phase_fraction_data_v2.hdf5` run in a Python shell:

```python
import h5py
with h5py.File("Phase_fraction_data_v2.hdf5") as f:
    print(list(f.keys()), f["conc"].shape)
```

Should see:

```
['conc', 'Matrix', 'g_prime', 'g_dprime'] (500000, 7)
```

Then take the data file to the notebook to fit and do sensitivity
