"""
freezeout_analysis.py – Weak freeze-out time and temperature from posterior
============================================================================
For each unweighted posterior sample, runs PRyM with those parameters and
records t_weak = t_of_T(T_weak) and T_weak = T_of_t(t_weak) from that run.
Results are cached in freezeout.csv; plots are saved in run_dir/plots/.
Uses joblib with multiprocessing backend to parallelize PRyM calls.

Usage
-----
  uv run freezeout_analysis.py RUN_DIR
  uv run freezeout_analysis.py runs/CC_2026-03-10_23-24-10
  uv run freezeout_analysis.py RUN_DIR --force   # recompute even if cache exists

Output
------
  run_dir/freezeout.csv   – param columns + t_weak [s], T_weak_MeV
  run_dir/plots/freezeout_t_weak.png (.pdf, .eps)
  run_dir/plots/freezeout_T_weak.png (.pdf, .eps)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#  Load run data (posterior_unweighted = physical parameter values per sample)
# ---------------------------------------------------------------------------


def load_metadata(run_dir: Path) -> dict:
    import json
    meta = {}
    mf = run_dir / "metadata.txt"
    if not mf.exists():
        return meta
    for line in mf.read_text().splitlines():
        if not line.strip() or ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        try:
            meta[k.strip()] = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            if v == "True":
                meta[k.strip()] = True
            elif v == "False":
                meta[k.strip()] = False
            else:
                meta[k.strip()] = v
    return meta


def load_posterior_unweighted(run_dir: Path):
    """Load posterior_unweighted.csv (one row per sample, physical params)."""
    fn = run_dir / "posterior_unweighted.csv"
    if not fn.exists():
        return None, None
    header_line = ""
    delimiter = None
    with open(fn) as f:
        for line in f:
            if line.startswith("#"):
                header_line = line.lstrip("#").strip()
            elif delimiter is None:
                delimiter = "," if "," in line else None
                break
    if delimiter == ",":
        cols = header_line.replace(",", " ").split()
    else:
        cols = header_line.split()
    data = np.loadtxt(fn, comments="#", delimiter=delimiter)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data, cols


def load_freezeout_cache(run_dir: Path, param_names: list[str], n_expected: int):
    """Load freezeout.csv if it exists and matches param names and row count."""
    fn = run_dir / "freezeout.csv"
    if not fn.exists():
        return None
    with open(fn) as f:
        first = f.readline()
    if not first.startswith("#"):
        return None
    header_line = first.lstrip("#").strip()
    cols = [c.strip() for c in header_line.split(",")]
    data = np.loadtxt(fn, comments="#", delimiter=",", ndmin=2)
    if data.shape[0] != n_expected:
        return None
    if cols[: len(param_names)] != param_names or "t_weak" not in cols or "T_weak_MeV" not in cols:
        return None
    return data, cols


def save_freezeout_cache(run_dir: Path, param_names: list[str], data: np.ndarray) -> None:
    """Write freezeout.csv with param columns + t_weak, T_weak_MeV."""
    cols = param_names + ["t_weak", "T_weak_MeV"]
    header = "# " + ",".join(cols) + "\n"
    with open(run_dir / "freezeout.csv", "w") as f:
        f.write(header)
        np.savetxt(f, data, delimiter=",", fmt="%.6e")
    print(f"  → freezeout.csv ({data.shape[0]} rows)")


# ---------------------------------------------------------------------------
#  Run PRyM and get t_weak, T_weak for one parameter set
# ---------------------------------------------------------------------------

def _run_prym_one(model_name: str, param_row: np.ndarray) -> tuple[float, float] | None:
    """Picklable worker: build model and run PRyM for one sample; return (t_weak [s], T_weak [MeV]) or None."""
    import PRyM.PRyM_init as PRyMini
    from PRyM.PRyM_main import PRyMclass
    from eden_model import make_model

    try:
        model = make_model(model_name)
        model.configure(*param_row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solver = PRyMclass(
                my_rho_NP=model.rho_NP,
                my_p_NP=model.p_NP,
                my_drho_NP_dT=model.drho_NP_dT,
                my_rho_EDE=model.rho_EDE,
            )
        T_weak_MeV = PRyMini.T_weak / PRyMini.MeV_to_Kelvin
        t_weak = float(solver.t_of_T(T_weak_MeV))
        T_weak_out = float(solver.T_of_t(t_weak))
        return t_weak, T_weak_out
    except Exception:
        return None


def compute_freezeout_table(run_dir: Path, model_name: str, samples: np.ndarray, param_names: list[str]) -> np.ndarray:
    """Compute t_weak and T_weak_MeV for each sample in parallel; return array with params + t_weak + T_weak_MeV."""
    from joblib import Parallel, delayed

    n = samples.shape[0]
    out = np.zeros((n, samples.shape[1] + 2))
    out[:, : samples.shape[1]] = samples
    results = Parallel(n_jobs=-1, backend="multiprocessing", verbose=55)(
        delayed(_run_prym_one)(model_name, samples[i]) for i in range(n)
    )
    for i, res in enumerate(results):
        if res is not None:
            out[i, -2], out[i, -1] = res
        else:
            out[i, -2], out[i, -1] = np.nan, np.nan
    return out


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

PLOT_EXTENSIONS = (".png", ".pdf", ".eps")


def _param_log_label(name: str) -> str:
    """LaTeX label for log10 of a MeV^4 parameter."""
    if name == "Lambda_MeV4":
        return r"$\log_{10}(\Lambda/\mathrm{MeV}^4)$"
    if name == "rho0_MeV4":
        return r"$\log_{10}(\rho_0/\mathrm{MeV}^4)$"
    return name


def plot_freezeout(run_dir: Path, data: np.ndarray, param_names: list[str], x_col: int = 0) -> None:
    """Create freezeout_t_weak and freezeout_T_weak plots. Axes swapped: x = t_weak/T_weak, y = param."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    x = data[:, x_col]
    t_weak = data[:, -2]
    T_weak = data[:, -1]
    valid = np.isfinite(t_weak) & np.isfinite(T_weak) & (t_weak > 0) & (T_weak > 0)
    xv, twv, Twv = x[valid], t_weak[valid], T_weak[valid]

    xlabel = param_names[x_col]
    if xlabel in ("Lambda_MeV4", "rho0_MeV4"):
        param_vals = np.log10(xv)
        ylabel_plot = _param_log_label(xlabel)
    else:
        param_vals = xv
        ylabel_plot = xlabel

    # Disable offset text (the "1e-12+7...e-1" artifact)
    def no_offset(ax):
        for a in (ax.xaxis, ax.yaxis):
            f = ScalarFormatter()
            f.set_useOffset(False)
            a.set_major_formatter(f)

    # ---- t_weak plot: x = t_weak, y = param (swapped) ----
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.scatter(twv, param_vals, s=4, alpha=0.6, c="#1f77b4")
    ax.set_xlabel(r"$t_{\mathrm{weak}}$ [s]", fontsize=14)
    ax.set_ylabel(ylabel_plot, fontsize=14)
    ax.tick_params(labelsize=12)
    no_offset(ax)
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        out = plot_dir / ("freezeout_t_weak" + ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  → {out.name}")
    plt.close(fig)

    # ---- Freeze-out time modification: (t_weak - t_ref)/t_ref [%] vs param ----
    # T_weak is fixed at 1 MeV by definition; the quantity that varies with new physics is t_weak.
    # This plot shows how much the freeze-out *time* shifts (like Genesys t_NP vs t_GR comparison).
    t_ref = np.median(twv)
    t_weak_rel_pct = (twv - t_ref) / t_ref * 100.0

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.scatter(param_vals, t_weak_rel_pct, s=4, alpha=0.6, c="#1f77b4")
    ax.axhline(0, color="k", ls="--", alpha=0.5, lw=1)
    ax.set_xlabel(ylabel_plot, fontsize=14)
    ax.set_ylabel(r"$(t_{\mathrm{weak}} - t_{\mathrm{ref}})/t_{\mathrm{ref}}$ [%]", fontsize=14)
    ax.tick_params(labelsize=12)
    no_offset(ax)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in PLOT_EXTENSIONS:
        out = plot_dir / ("freezeout_T_weak" + ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  → {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def process_run(run_dir: Path, force_recompute: bool = False) -> None:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        print(f"[ERROR] Not a directory: {run_dir}")
        return

    meta = load_metadata(run_dir)
    model_name = meta.get("model")
    if not model_name:
        print("[ERROR] No model in metadata.txt; cannot run freezeout analysis.")
        return

    samples, param_names = load_posterior_unweighted(run_dir)
    if samples is None:
        print("[ERROR] No posterior_unweighted.csv in run directory.")
        return

    # Use cache if present and consistent
    data = None
    if not force_recompute:
        cached = load_freezeout_cache(run_dir, param_names, samples.shape[0])
        if cached is not None:
            data, _ = cached
            print("  Using cached freezeout.csv")
    if data is None:
        print("  Running PRyM for each unweighted posterior sample (joblib multiprocessing) ...")
        data = compute_freezeout_table(run_dir, model_name, samples, param_names)
        save_freezeout_cache(run_dir, param_names, data)

    print("  Plotting ...")
    plot_freezeout(run_dir, data, param_names, x_col=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute weak freeze-out t_weak and T_weak for posterior samples; cache and plot."
    )
    parser.add_argument("run_dir", type=str, help="Run directory (e.g. runs/CC_2026-03-10_23-24-10)")
    parser.add_argument("--force", action="store_true", help="Recompute freezeout even if freezeout.csv exists")
    args = parser.parse_args()
    process_run(Path(args.run_dir), force_recompute=args.force)
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
