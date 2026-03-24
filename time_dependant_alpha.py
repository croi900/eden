"""
time_dependant_alpha.py
=========================

Parameter-space scan for the temperature-dependent EDE model (`TempDependent`):

* Sample nuisance parameters (tau_n, Omegabh2, p_npdg, p_dpHe3g) from their priors
  either fixed-to-mean or via Monte Carlo draws.
* For each (rho0_MeV4, alpha) point, run PRyM to obtain the background and compute
  H(t) = (1/a) da/dt from the returned `a_of_t(t)` interpolation.
* Extract log10H(t) extrema:
    - log10H_min = min_t log10(H(t))
    - log10H_max = max_t log10(H(t))
* Produce a 2D heatmap with axes:
    - x: rho0_MeV4 (log scale)
    - y: alpha (linear)
  and color = aggregated log10H_max (max over time, aggregated over nuisance draws).

This is intended to help identify regions where the model yields "normal" H
values (not astronomically large ~1e120).
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np


def _get_param_bounds(priors: dict, name: str) -> tuple[float, float, str]:
    lo_hi, scale = priors[name]
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return lo, hi, str(scale)


def _draw_from_prior(
    rng: np.random.Generator, prior: tuple[float, float, str]
) -> float:
    lo, hi, scale = prior
    if scale == "lin":
        return float(rng.uniform(lo, hi))
    if scale == "log":
        # Model convention: prior is uniform in log10(x) between lo..hi.
        return float(10.0 ** rng.uniform(lo, hi))
    if scale == "norm":
        return float(rng.normal(loc=lo, scale=hi))
    raise ValueError(f"Unsupported prior scale {scale!r}")


def _draw_nuisance_fixed(model_priors: dict) -> dict[str, float]:
    # "Known Gaussian nuisance": use the mean of each Normal prior.
    # This matches typical prior-center behaviour in other parts of the pipeline.
    out = {}
    for k, (lo_hi, scale) in model_priors.items():
        if k not in ("tau_n", "Omegabh2", "p_npdg", "p_dpHe3g"):
            continue
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        if scale != "norm":
            continue
        out[k] = lo
    return out


def _draw_nuisance_mc(rng: np.random.Generator, model_priors: dict) -> dict[str, float]:
    out = {}
    for k in ("tau_n", "Omegabh2", "p_npdg", "p_dpHe3g"):
        lo_hi, scale = model_priors[k]
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        if scale != "norm":
            raise ValueError(f"Expected Normal prior for {k}, got scale={scale!r}")
        out[k] = float(rng.normal(loc=lo, scale=hi))
    return out


@dataclass(frozen=True)
class HStats:
    log10H_min: float
    log10H_max: float


def _aggregate(values: list[float], mode: str) -> float:
    if not values:
        return float("inf")
    arr = np.array(values, dtype=float)
    if mode == "min":
        return float(np.nanmin(arr))
    if mode == "median":
        return float(np.nanmedian(arr))
    if mode == "max":
        return float(np.nanmax(arr))
    raise ValueError(f"Unknown aggregate mode {mode!r}")

def _cell_worker(
    iy: int,
    ix: int,
    rho0_MeV4: float,
    alpha: float,
    nuisance_draws: list[dict[str, float]],
    n_sampling: int,
    agg: str,
) -> tuple[int, int, float, float]:
    """
    Evaluate one (rho0, alpha) cell across nuisance draws.

    Returns:
      (iy, ix, cell_log10Hmax_agg, cell_log10Hmin_agg)
    """
    from eden_model import make_model

    model = make_model("TempDependent")
    log10Hmax_list: list[float] = []
    log10Hmin_list: list[float] = []
    for nuis in nuisance_draws:
        # Configure per nuisance draw because PRyM init reads PRyMini globals.
        stats = _compute_h_stats(
            model,
            rho0_MeV4,
            float(alpha),
            nuis,
            n_sampling=n_sampling,
        )
        if np.isfinite(stats.log10H_max):
            log10Hmax_list.append(stats.log10H_max)
            log10Hmin_list.append(stats.log10H_min)

    cell_max = _aggregate(log10Hmax_list, agg)
    cell_min = _aggregate(log10Hmin_list, agg) if log10Hmin_list else float("inf")
    return iy, ix, cell_max, cell_min


def _compute_h_stats(model, rho0_MeV4: float, alpha: float, nuisance: dict[str, float], *, n_sampling: int | None = None) -> HStats:
    """
    Run PRyM for one (rho0, alpha) and one nuisance set; return log10(H) extrema.
    """
    import PRyM.PRyM_init as PRyMini
    from PRyM.PRyM_main import PRyMclass

    if n_sampling is not None:
        # Reduce runtime while still returning a background adequate for H(t).
        # PRyM uses this for the thermodynamics background sampling.
        PRyMini.n_sampling = int(n_sampling)

    model.configure(
        rho0_MeV4,
        alpha,
        nuisance["tau_n"],
        nuisance["Omegabh2"],
        nuisance["p_npdg"],
        nuisance["p_dpHe3g"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = PRyMclass(
            my_rho_NP=model.rho_NP,
            my_p_NP=model.p_NP,
            my_drho_NP_dT=model.drho_NP_dT,
            my_rho_EDE=model.rho_EDE,
        )

    t_vec = np.array(getattr(solver, "_t_vec"), dtype=float)
    if t_vec.size == 0 or not np.all(np.isfinite(t_vec)):
        return HStats(float("inf"), float("inf"))

    order = np.argsort(t_vec)
    t_vec = t_vec[order]
    # a_of_t accepts an array; returns a(t)
    a_vec = np.array(solver.a_of_t(t_vec), dtype=float)

    # Numerical Hubble: H = (1/a) da/dt
    if np.any(a_vec <= 0.0) or np.any(~np.isfinite(a_vec)):
        return HStats(float("inf"), float("inf"))

    da_dt = np.gradient(a_vec, t_vec)
    H_vec = da_dt / a_vec
    # Physical H should be > 0.0. Filter aggressively to avoid log10(negative).
    mask = np.isfinite(H_vec) & (H_vec > 0.0)
    if not np.any(mask):
        return HStats(float("inf"), float("inf"))

    log10H = np.log10(H_vec[mask])
    return HStats(float(np.min(log10H)), float(np.max(log10H)))


def main() -> None:
    from eden_model import BBN_NUISANCE, make_model

    parser = argparse.ArgumentParser(
        description="Scan TempDependent (rho0_MeV4, alpha) and heatmap log10(H_max)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--nuisance-mode", choices=["fixed", "mc"], default="mc")
    parser.add_argument("--n-nuisance", type=int, default=3, help="Monte Carlo draws of nuisance per grid cell (if mc)")
    parser.add_argument("--n-rho", type=int, default=18, help="Number of rho grid points")
    parser.add_argument("--n-alpha", type=int, default=18, help="Number of alpha grid points")
    parser.add_argument("--mc-points", type=int, default=80, help="Random prior samples (diagnostics, not used for heatmap)")
    parser.add_argument("--agg", choices=["min", "median", "max"], default="median", help="Aggregate log10(H_max) across nuisance draws per cell")
    parser.add_argument("--n-sampling", type=int, default=500, help="Override PRyM_init.n_sampling to reduce runtime")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for grid-cell evaluation (-1 = all cores).")
    parser.add_argument(
        "--backend",
        type=str,
        default="multiprocessing",
        help="joblib backend for Parallel: multiprocessing, loky, threading, or auto-fallback (default: multiprocessing).",
    )
    parser.add_argument("--alpha-lo", type=float, default=None, help="Override alpha lower bound")
    parser.add_argument("--alpha-hi", type=float, default=None, help="Override alpha upper bound")
    parser.add_argument("--rho-log10-lo", type=float, default=None, help="Override log10(rho0_MeV4) lower bound")
    parser.add_argument("--rho-log10-hi", type=float, default=None, help="Override log10(rho0_MeV4) upper bound")
    parser.add_argument("--hubble-log10-clip", type=float, default=200.0, help="Clip plotted color scale to this value")
    parser.add_argument("--out", type=str, default="time_dependent_alpha_heatmap.png", help="Output PNG filename")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    model = make_model("TempDependent")
    priors = model.PRIORS

    rho_lo, rho_hi, rho_scale = _get_param_bounds(priors, "rho0_MeV4")
    alpha_lo, alpha_hi, alpha_scale = _get_param_bounds(priors, "alpha")
    if rho_scale != "log":
        raise ValueError(f"Expected log prior for rho0_MeV4, got scale={rho_scale!r}")
    if alpha_scale != "lin":
        raise ValueError(f"Expected lin prior for alpha, got scale={alpha_scale!r}")

    if args.rho_log10_lo is not None:
        rho_lo = float(args.rho_log10_lo)
    if args.rho_log10_hi is not None:
        rho_hi = float(args.rho_log10_hi)
    if args.alpha_lo is not None:
        alpha_lo = float(args.alpha_lo)
    if args.alpha_hi is not None:
        alpha_hi = float(args.alpha_hi)

    if rho_hi <= rho_lo:
        raise ValueError("Require rho-log10 upper > lower")
    if alpha_hi <= alpha_lo:
        raise ValueError("Require alpha upper > lower")

    # Axes
    rho_grid_log10 = np.linspace(rho_lo, rho_hi, args.n_rho)
    rho_grid = 10.0 ** rho_grid_log10
    alpha_grid = np.linspace(alpha_lo, alpha_hi, args.n_alpha)

    # Prepare nuisance sampler
    if args.nuisance_mode == "fixed":
        nuisance_draws = [_draw_nuisance_fixed(priors)]
    else:
        nuisance_draws = [_draw_nuisance_mc(rng, priors) for _ in range(args.n_nuisance)]

    grid_log10Hmax = np.zeros((args.n_alpha, args.n_rho), dtype=float)
    grid_log10Hmin = np.zeros((args.n_alpha, args.n_rho), dtype=float)

    print(f"Grid: n_rho={args.n_rho}, n_alpha={args.n_alpha}, nuisance_mode={args.nuisance_mode}, n_nuisance={len(nuisance_draws)}")

    # Evaluate grid
    from joblib import Parallel, delayed

    jobs = []
    for iy, alpha in enumerate(alpha_grid):
        for ix, rho0 in enumerate(rho_grid):
            jobs.append((iy, ix, float(rho0), float(alpha)))

    print(f"Evaluating {len(jobs)} grid cells with n_jobs={args.n_jobs} ...")
    # Try requested backend; if blocked in this environment, fall back to threading.
    try:
        results = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
            delayed(_cell_worker)(
                iy,
                ix,
                rho0_MeV4=rho0,
                alpha=alpha,
                nuisance_draws=nuisance_draws,
                n_sampling=args.n_sampling,
                agg=args.agg,
            )
            for (iy, ix, rho0, alpha) in jobs
        )
    except Exception as exc:
        print(f"[WARN] Parallel backend {args.backend!r} failed ({exc}); falling back to threading.")
        results = Parallel(n_jobs=args.n_jobs, backend="threading", verbose=10)(
            delayed(_cell_worker)(
                iy,
                ix,
                rho0_MeV4=rho0,
                alpha=alpha,
                nuisance_draws=nuisance_draws,
                n_sampling=args.n_sampling,
                agg=args.agg,
            )
            for (iy, ix, rho0, alpha) in jobs
        )
    for iy, ix, cell_max, cell_min in results:
        grid_log10Hmax[iy, ix] = cell_max
        grid_log10Hmin[iy, ix] = cell_min

    # Heatmap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_Z = np.clip(grid_log10Hmax, -50.0, args.hubble_log10_clip)
    plt.figure(figsize=(8, 6), facecolor="white")
    extent = [rho_grid_log10[0], rho_grid_log10[-1], alpha_grid[0], alpha_grid[-1]]
    # origin="lower" so y increases upward
    im = plt.imshow(
        plot_Z,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    plt.colorbar(im, label=r"$\log_{10}(H_{\max})$")
    plt.xlabel(r"$\log_{10}(\rho_{0,\mathrm{DE}}~[\mathrm{MeV}^4])$")
    plt.ylabel(r"$\alpha$")
    plt.title(r"TempDependent: heatmap of $\log_{10}(H_{\max})$")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved heatmap → {args.out}")

    # Monte Carlo diagnostics on random EDE points from the priors
    print("\nMonte Carlo diagnostics (random EDE params):")
    rho_prior_bounds = (rho_lo, rho_hi, rho_scale)
    alpha_prior_bounds = (alpha_lo, alpha_hi, alpha_scale)
    for i in range(args.mc_points):
        rho0 = _draw_from_prior(rng, rho_prior_bounds)
        alpha = _draw_from_prior(rng, alpha_prior_bounds)
        # For MC diagnostics, use nuisance draws (same nuisance_draws as the grid)
        log10Hmax_list: list[float] = []
        log10Hmin_list: list[float] = []
        for nuis in nuisance_draws:
            stats = _compute_h_stats(model, rho0, float(alpha), nuis, n_sampling=args.n_sampling)
            if np.isfinite(stats.log10H_max):
                log10Hmax_list.append(stats.log10H_max)
                log10Hmin_list.append(stats.log10H_min)
        if not log10Hmax_list:
            log10Hmax_agg = float("inf")
            log10Hmin_agg = float("inf")
        else:
            log10Hmax_agg = _aggregate(log10Hmax_list, args.agg)
            log10Hmin_agg = _aggregate(log10Hmin_list, args.agg)
        print(
            f"  {i+1:4d}/{args.mc_points}: rho0={rho0:.3e}, alpha={alpha:.3e}, "
            f"log10H_min~{log10Hmin_agg:.3f}, log10H_max~{log10Hmax_agg:.3f}"
        )


if __name__ == "__main__":
    main()

