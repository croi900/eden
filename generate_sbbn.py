"""
generate_sbbn.py – Monte Carlo SBBN abundance samples (no new physics)
======================================================================
Generates n samples of standard big-bang nucleosynthesis abundances by
varying only the nuisance parameters (tau_n, Omegabh2, p_npdg, p_dpHe3g)
from their NS priors. No EDE/new physics (Lambda = 0 effectively).
Output is suitable for an SBBN corner plot (Yp, D/H, He3/H, Li7/H).

Usage
-----
  uv run generate_sbbn.py [--n 1000] [--output sbbn_samples.csv]
  uv run generate_sbbn.py --n 2000 --output data/sbbn_samples.csv

Output
------
  CSV with columns: tau_n, Omegabh2, p_npdg, p_dpHe3g, Yp, DoH, He3oH, Li7oH
  (DoH = D/H × 10^5, He3oH = ³He/H × 10^5, Li7oH = ⁷Li/H × 10^10)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from eden_model import BBN_NUISANCE, make_model

# SBBN: no EDE – use CC model with Lambda_MeV4 negligible
LAMBDA_SBBN_MeV4 = 1.0e-25


def sample_nuisance(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw n samples of (tau_n, Omegabh2, p_npdg, p_dpHe3g) from NS priors."""
    if rng is None:
        rng = np.random.default_rng()
    tau_n_prior = BBN_NUISANCE["tau_n"][0]   # (879.4, 0.6) norm
    Omegabh2_prior = BBN_NUISANCE["Omegabh2"][0]  # (0.02230, 0.00015) norm
    p_npdg_prior = BBN_NUISANCE["p_npdg"][0]     # (0, 1) norm
    p_dpHe3g_prior = BBN_NUISANCE["p_dpHe3g"][0] # (0, 1) norm

    tau_n = rng.normal(tau_n_prior[0], tau_n_prior[1], size=n)
    Omegabh2 = rng.normal(Omegabh2_prior[0], Omegabh2_prior[1], size=n)
    p_npdg = rng.normal(p_npdg_prior[0], p_npdg_prior[1], size=n)
    p_dpHe3g = rng.normal(p_dpHe3g_prior[0], p_dpHe3g_prior[1], size=n)
    return np.column_stack([tau_n, Omegabh2, p_npdg, p_dpHe3g])


def run_sbbn_one(params: np.ndarray) -> np.ndarray | None:
    """Run PRyM for one nuisance set (SBBN only). Returns [Yp, DoH, He3oH, Li7oH] or None."""
    model = make_model("CC")
    tau_n, Omegabh2, p_npdg, p_dpHe3g = params
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.abundances(LAMBDA_SBBN_MeV4, tau_n, Omegabh2, p_npdg, p_dpHe3g)
        if res is None or not np.all(np.isfinite(res)):
            return None
        # res = [Neff, Omeganurel, 1/Omeganunr, YPCMB, YPBBN, DoHx1e5, He3oHx1e5, Li7oHx1e10]
        Yp = res[4]
        DoH = res[5]
        He3oH = res[6]
        Li7oH = res[7]
        return np.array([Yp, DoH, He3oH, Li7oH])
    except Exception:
        return None


def generate_sbbn(n: int, seed: int | None = None, n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n SBBN samples. Returns (nuisance_params, abundances).
    nuisance_params: (n, 4) tau_n, Omegabh2, p_npdg, p_dpHe3g
    abundances: (n, 4) Yp, DoH, He3oH, Li7oH
    """
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    nuisance = sample_nuisance(n, rng)

    if n_jobs == 1:
        abds = []
        for i in range(n):
            a = run_sbbn_one(nuisance[i])
            abds.append(a if a is not None else np.full(4, np.nan))
            if (i + 1) % 100 == 0 or i == n - 1:
                print(f"    SBBN run {i + 1}/{n}")
        abds = np.array(abds)
    else:
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=10)(
            delayed(run_sbbn_one)(nuisance[i]) for i in range(n)
        )
        abds = np.array([r if r is not None else np.full(4, np.nan) for r in results])

    return nuisance, abds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Monte Carlo SBBN abundance samples (nuisance only, no new physics)."
    )
    parser.add_argument("--n", type=int, default=1000, help="Number of samples (default: 1000)")
    parser.add_argument("--output", "-o", type=str, default="sbbn_samples.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Joblib parallel jobs (-1 = all cores, 1 = serial)")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} SBBN samples (nuisance only, Lambda=0) ...")
    nuisance, abds = generate_sbbn(args.n, seed=args.seed, n_jobs=args.n_jobs)

    # Build full table: tau_n, Omegabh2, p_npdg, p_dpHe3g, Yp, DoH, He3oH, Li7oH
    table = np.hstack([nuisance, abds])
    header = "tau_n,Omegabh2,p_npdg,p_dpHe3g,Yp,DoH,He3oH,Li7oH"
    with open(out_path, "w") as f:
        f.write("# " + header + "\n")
        np.savetxt(f, table, delimiter=",", fmt="%.6e")
    print(f"  → {out_path} ({args.n} rows)")
    n_ok = np.sum(np.all(np.isfinite(abds), axis=1))
    print(f"  {n_ok}/{args.n} samples finite")
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
