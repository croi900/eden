"""
mc.py – Simple Monte Carlo sampling over EDE priors
===================================================

Usage
-----
  uv run mc.py MODEL N_SAMPLES
  uv run mc.py CC 1000
  uv run mc.py Linear 500 --runs-dir runs

This script:
  * takes an EDE model name and its parameter priors (from `eden_model`),
  * draws Monte Carlo samples uniformly from those priors
    (log–uniform for ``\"log\"`` priors, Gaussian for ``\"norm\"`` priors),
  * runs PRyM once per sample to obtain BBN abundances,
  * writes results under ``runs/mc_<MODEL>_<timestamp>/`` as ``mc_samples.csv``.

The output CSV has columns::

  <params...>, Yp, DoHx1e5, He3oHx1e5, Li7oHx1e10

matching the conventions used in ``ns.py``.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from eden_model import BaseEDEModel, make_model


def _build_run_dir(model: BaseEDEModel, runs_dir: str = "runs", label: str = "") -> Path:
    """Create a run directory of the form runs/mc_<MODEL>_<label>_<timestamp>/."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"_{label}" if label else ""
    name = f"mc_{model.model_name}{tag}_{timestamp}"
    run_dir = Path(runs_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _sample_from_priors(model: BaseEDEModel, n: int, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """
    Draw Monte Carlo samples from the model priors.

    For a prior entry (lo, hi, scale):
      * scale == \"lin\":  x ~ Uniform[lo, hi]
      * scale == \"log\":  log10(x) ~ Uniform[lo, hi]  → x = 10**u
      * scale == \"norm\": x ~ Normal(mean=lo, sigma=hi)
    This mirrors the conventions used in `ns.py`.
    """
    priors = model.PRIORS
    param_names = list(priors.keys())
    ndim = len(param_names)

    samples = np.zeros((n, ndim), dtype=float)

    for j, name in enumerate(param_names):
        (a, b), scale = priors[name]
        if scale == "lin":
            samples[:, j] = rng.uniform(a, b, size=n)
        elif scale == "log":
            exp = rng.uniform(a, b, size=n)
            samples[:, j] = 10.0**exp
        elif scale == "norm":
            samples[:, j] = rng.normal(loc=a, scale=b, size=n)
        else:
            raise ValueError(f"Unknown prior scale '{scale}' for parameter '{name}'")

    return samples, param_names


def _run_abundances_one(model_name: str, param_row: np.ndarray) -> np.ndarray | None:
    """Picklable worker: build model and run PRyM for one parameter set."""
    try:
        from eden_model import make_model as _make_model

        model = _make_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.abundances(*param_row)
        return np.asarray(res, dtype=float)
    except Exception:
        return None


def _compute_mc_table(model_name: str, samples: np.ndarray) -> np.ndarray:
    """
    Run PRyM for each sample (in parallel) and collect abundances.

    Returns an array with shape (N, ndim + 4):
      [params..., Yp, DoHx1e5, He3oHx1e5, Li7oHx1e10]
    """
    n, ndim = samples.shape
    out = np.zeros((n, ndim + 4), dtype=float)
    out[:, :ndim] = samples

    results = Parallel(n_jobs=-1, backend="multiprocessing", verbose=55)(
        delayed(_run_abundances_one)(model_name, samples[i]) for i in range(n)
    )

    for i, res in enumerate(results):
        if res is not None and res.size >= 8 and np.all(np.isfinite(res[4:8])):
            out[i, -4:] = res[4:8]
        else:
            out[i, -4:] = np.nan

    return out


def _save_mc_csv(run_dir: Path, param_names: list[str], data: np.ndarray) -> None:
    """Save Monte Carlo samples + abundances to mc_samples.csv in run_dir."""
    cols = param_names + ["Yp", "DoHx1e5", "He3oHx1e5", "Li7oHx1e10"]
    header = "# " + " ".join(cols) + "\n"
    fn = run_dir / "mc_samples.csv"
    with open(fn, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.8e")
    print(f"  → {fn} ({data.shape[0]} rows)")


def _write_metadata(run_dir: Path, model: BaseEDEModel, n_samples: int) -> None:
    """Write a small metadata.txt akin to ns.py."""
    import json

    meta = model.metadata()
    meta["n_mc_samples"] = int(n_samples)
    with open(run_dir / "metadata.txt", "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {json.dumps(v)}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo over EDE priors → abundances.")
    parser.add_argument(
        "model",
        type=str,
        help="EDE model name (e.g. CC, Linear, Polytropic).",
    )
    parser.add_argument(
        "n_samples",
        type=int,
        help="Number of Monte Carlo samples to draw from the priors.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base directory for output runs (default: runs).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label to include in the mc run directory name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--poly-gamma",
        type=float,
        default=None,
        help="If model=Polytropic, set PRyMini.gamma to this value before sampling.",
    )

    args = parser.parse_args(argv)

    # Build model
    model = make_model(args.model)

    # Optional: configure Polytropic gamma, mirroring ns.py behaviour
    if args.model == "Polytropic" and args.poly_gamma is not None:
        import PRyM.PRyM_init as PRyMini

        PRyMini.gamma = float(args.poly_gamma)

    rng = np.random.default_rng(args.seed)

    print(
        f"Monte Carlo: model={model.model_name}, "
        f"n_samples={args.n_samples}, priors={model.PRIORS}"
    )

    run_dir = _build_run_dir(model, runs_dir=args.runs_dir, label=args.label)
    print(f"Output directory: {run_dir}")

    # Sample parameters from priors
    samples, param_names = _sample_from_priors(model, args.n_samples, rng)

    # Run PRyM and collect abundances
    data = _compute_mc_table(model.model_name, samples)

    # Save results
    _save_mc_csv(run_dir, param_names, data)
    _write_metadata(run_dir, model, args.n_samples)


if __name__ == "__main__":
    main()

