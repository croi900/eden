"""
hubble_analysis.py – Background evolution (H(t), T(t), a(t)) for 95% UL, 68% UL, SBBN
====================================================================================

Runs PRyM only for three cases:
  - 95% upper limit (posterior sample at 95% UL of first parameter)
  - 68% upper limit (posterior sample at 68% UL of first parameter)
  - SBBN (no new physics: first param at zero, nuisance at posterior median)

Results are cached to run_dir/hubble_background.npz. If the file exists,
analysis is skipped unless --force is given.

Usage
-----
  uv run hubble_analysis.py RUN_DIR
  uv run hubble_analysis.py RUN_DIR --force

Output
------
  run_dir/hubble_background.npz  – t, T_MeV (3, n_t), a (3, n_t), H (3, n_t), labels
  run_dir/plots/hubble_*.{png,pdf,eps} – three curves: 95% UL, 68% UL, SBBN
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np


def load_metadata(run_dir: Path) -> dict:
    """Parse metadata.txt → dict (copied from plot_ns helper)."""
    import json

    meta: dict = {}
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


def load_summary(run_dir: Path) -> dict:
    """Parse summary.txt for median, 68% CI, 95% UL per parameter."""
    sf = run_dir / "summary.txt"
    result = {"logZ": None, "logZerr": None, "params": {}}
    if not sf.exists():
        return result
    for line in sf.read_text().splitlines():
        if line.startswith("logZ "):
            result["logZ"] = float(line.split("=")[1])
        elif line.startswith("logZerr"):
            result["logZerr"] = float(line.split("=")[1])
        elif line and not line.startswith("logZ"):
            parts = line.split(":")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            rest = parts[1].strip()
            d: dict = {}
            for seg in rest.split("  "):
                seg = seg.strip()
                if seg.startswith("median="):
                    d["median"] = float(seg[7:])
                elif seg.startswith("68CI=["):
                    inner = seg[6:].rstrip("]")
                    lo, hi = inner.split(",")
                    d["lo68"] = float(lo)
                    d["hi68"] = float(hi)
                elif seg.startswith("95UL="):
                    d["ul95"] = float(seg[5:])
            if d:
                result["params"][name] = d
    return result


def build_three_param_sets(
    samples: np.ndarray, param_names: list, summary: dict, model_name: str
):
    """
    Build three parameter vectors: 95% UL, 68% UL, SBBN.
    - 95% CI: each EDE param = 95th percentile (marginal), nuisance = median.
    - 68% CI: each EDE param = 84th percentile (upper edge of 68% CI), nuisance = median.
    - SBBN: all EDE params = 0, nuisance = median.
    """
    n_ede = n_ede_params(model_name)
    n_tot = samples.shape[1]
    nuisance_median = np.median(samples[:, n_ede:], axis=0)

    # 95% CI: marginal 95th percentile for each EDE param
    params_95 = np.empty(n_tot)
    for j in range(n_ede):
        if param_names and param_names[j] in summary.get("params", {}):
            val = summary["params"][param_names[j]].get("ul95")
            if val is not None:
                params_95[j] = val
                continue
        params_95[j] = np.percentile(samples[:, j], 95)
    params_95[n_ede:] = nuisance_median

    # 68% CI: marginal 84th percentile (upper 68% CI) for each EDE param
    params_68 = np.empty(n_tot)
    for j in range(n_ede):
        if param_names and param_names[j] in summary.get("params", {}):
            val = summary["params"][param_names[j]].get("hi68")
            if val is not None:
                params_68[j] = val
                continue
        params_68[j] = np.percentile(samples[:, j], 84)
    params_68[n_ede:] = nuisance_median

    # SBBN: all EDE params = 0, nuisance = median
    params_sbbn = np.concatenate([np.zeros(n_ede), nuisance_median]).astype(float)

    return np.array([params_95, params_68, params_sbbn]), ["95% CI", "68% CI", "SBBN"]


def n_ede_params(model_name: str) -> int:
    """Number of EDE/NP parameters (all params minus 4 BBN nuisance)."""
    from eden_model import make_model

    return make_model(model_name).ndim - 4


def compute_background_for_sample(model_name: str, params: np.ndarray):
    """
    Run PRyM for a single posterior sample and return (t_vec, T_MeV, a_vec, H_vec).

    Requires that PRyMclass exposes:
      - solver._t_vec      (time grid in seconds)
      - solver._Tg_vec     (T_gamma in MeV)
      - solver._a_vec      (scale factor sampled at Tg_vec)
      - solver.a_of_t(t)   (interpolating function a(t))
    """
    import PRyM.PRyM_init as PRyMini
    from PRyM.PRyM_main import PRyMclass
    from eden_model import make_model

    model = make_model(model_name)
    model.configure(*params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = PRyMclass(
            my_rho_NP=model.rho_NP,
            my_p_NP=model.p_NP,
            my_drho_NP_dT=model.drho_NP_dT,
            my_rho_EDE=model.rho_EDE,
        )

    t_vec = np.array(getattr(solver, "_t_vec"))
    T_MeV = np.array(getattr(solver, "_Tg_vec"))
    # Use a(t) evaluated on the same time grid
    a_vec = np.array(solver.a_of_t(t_vec))

    # Numerical Hubble H(t) = (1/a) da/dt (in 1/s)
    # Use central differences where possible
    da_dt = np.gradient(a_vec, t_vec)
    H_vec = da_dt / a_vec

    # Ensure strictly increasing t for later plotting/storage
    order = np.argsort(t_vec)
    t_vec = t_vec[order]
    T_MeV = T_MeV[order]
    a_vec = a_vec[order]
    H_vec = H_vec[order]

    return t_vec, T_MeV, a_vec, H_vec


def run_hubble_analysis(run_dir: Path, force: bool) -> None:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        print(f"[ERROR] Not a directory: {run_dir}")
        return

    meta = load_metadata(run_dir)
    model_name = meta.get("model")
    if not model_name:
        print("[ERROR] No model in metadata.txt; cannot run Hubble analysis.")
        return

    samples, param_names = load_posterior_unweighted(run_dir)
    if samples is None:
        print("[ERROR] No posterior_unweighted.csv in run directory.")
        return

    summary = load_summary(run_dir)
    param_list = param_names if isinstance(param_names, list) else list(param_names)
    params_three, labels = build_three_param_sets(
        samples, param_list, summary, model_name
    )
    n_cases = 3

    cache_file = run_dir / "hubble_background.npz"
    use_cache = False
    if cache_file.exists() and not force:
        data = np.load(cache_file, allow_pickle=True)
        if data["a"].shape[0] == 3:
            use_cache = True
            t_vec = data["t"]
            T_MeV = data["T_MeV"]
            a = data["a"]
            H = data["H"]
            labels = list(data["labels"]) if "labels" in data else ["95% UL", "68% UL", "SBBN"]
            print(f"  Using cached background from {cache_file.name} ({labels})")
        else:
            print(f"  Cache has {data['a'].shape[0]} samples (expected 3); recomputing.")

    if not use_cache:
        print("  Running PRyM for 95% UL, 68% UL, SBBN.")
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=3, backend="loky", verbose=10)(
            delayed(compute_background_for_sample)(model_name, params_three[i])
            for i in range(n_cases)
        )

        t_vec, T_MeV_0, a0, H0 = results[0]
        n_t = t_vec.size
        T_MeV = np.zeros((n_cases, n_t))
        a = np.zeros((n_cases, n_t))
        H = np.zeros((n_cases, n_t))
        T_MeV[0] = T_MeV_0
        a[0] = a0
        H[0] = H0

        for i in range(1, n_cases):
            t_i, T_i, a_i, H_i = results[i]
            if not np.allclose(t_i, t_vec):
                a_i = np.interp(t_vec, t_i, a_i)
                H_i = np.interp(t_vec, t_i, H_i)
                T_i = np.interp(t_vec, t_i, T_i)
            T_MeV[i] = T_i
            a[i] = a_i
            H[i] = H_i

        np.savez_compressed(
            cache_file,
            t=t_vec,
            T_MeV=T_MeV,
            a=a,
            H=H,
            params=params_three,
            param_names=np.array(param_list),
            labels=np.array(labels),
        )
        print(f"  → {cache_file.name} (3 curves, n_t={n_t})")

    # Legend: include all EDE parameter names and values in LaTeX in parentheses
    params_for_legend = data["params"] if use_cache else params_three
    n_ede = n_ede_params(model_name)
    ede_names = param_list[:n_ede]

    def _latex_param_name(name: str) -> str:
        if name == "Lambda_MeV4":
            return r"\Lambda"
        if name == "rho0_MeV4":
            return r"\rho_0"
        return name

    def _fmt_val_tex(x: float, mev4: bool) -> str:
        # Exact zero: just "0" (no scientific notation)
        if np.isclose(x, 0.0, atol=1e-14, rtol=0.0):
            base = "0"
        else:
            exp = int(np.floor(np.log10(abs(x))))
            mant = x / (10.0**exp)
            base = rf"{mant:.2f}\times 10^{{{exp}}}"
        if mev4:
            return rf"{base}\,[\mathrm{{MeV}}^4]"
        return base

    display_labels = []
    for i in range(3):
        if labels[i] == "SBBN":
            # Pure SBBN label, no parameter values
            display_labels.append(labels[i])
            continue
        parts = []
        for j in range(n_ede):
            name = ede_names[j]
            name_tex = _latex_param_name(name)
            val_tex = _fmt_val_tex(params_for_legend[i, j], mev4=name.endswith("MeV4"))
            parts.append(rf"${name_tex} = {val_tex}$")
        display_labels.append(f"{labels[i]} ({', '.join(parts)})")

    # Plotting: three curves with legend
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Ensure T_MeV is (3, n_t) for legacy cache that might have (n_t,)
    if T_MeV.ndim == 1:
        T_MeV = np.broadcast_to(T_MeV[np.newaxis, :], (3, len(T_MeV)))

    styles = [
        {"color": "C0", "ls": "-", "lw": 1.5},
        {"color": "C1", "ls": "--", "lw": 1.5},
        {"color": "C2", "ls": "-.", "lw": 1.5},
    ]

    def plot_three(ax, y_arr, ylabel, log_y=False):
        for i in range(3):
            sty = styles[i]
            if log_y:
                mask = (t_vec > 0) & (y_arr[i] > 0)
                if np.any(mask):
                    ax.plot(t_vec[mask], np.log10(y_arr[i][mask]), label=display_labels[i], **sty)
            else:
                ax.plot(t_vec, y_arr[i], label=display_labels[i], **sty)
        ax.set_xlabel("t [s]", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.set_facecolor("white")

    # a(t)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    plot_three(ax, a, "a(t)")
    fig.tight_layout()
    for ext in (".png", ".pdf", ".eps"):
        out = plot_dir / ("hubble_a_t" + ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  → {out.name}")
    plt.close(fig)

    # log10 a vs t
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    plot_three(ax, a, r"$\log_{10}(a)$", log_y=True)
    fig.tight_layout()
    for ext in (".png", ".pdf", ".eps"):
        out = plot_dir / ("hubble_log_a_t" + ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  → {out.name}")
    plt.close(fig)

    # T(t)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    plot_three(ax, T_MeV, r"$T_\gamma$ [MeV]")
    fig.tight_layout()
    for ext in (".png", ".pdf", ".eps"):
        out = plot_dir / ("hubble_T_t" + ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  → {out.name}")
    plt.close(fig)

    # log10 T vs t
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    plot_three(ax, T_MeV, r"$\log_{10}(T_\gamma\,[\mathrm{MeV}])$", log_y=True)
    fig.tight_layout()
    for ext in (".png", ".pdf", ".eps"):
        out = plot_dir / ("hubble_log_T_t" + ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  → {out.name}")
    plt.close(fig)

    # H(t)
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    plot_three(ax, H, "H(t) [1/s]")
    fig.tight_layout()
    for ext in (".png", ".pdf", ".eps"):
        out = plot_dir / ("hubble_H_t" + ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  → {out.name}")
    plt.close(fig)

    # log10 H vs t
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    plot_three(ax, H, r"$\log_{10}(H\,[\mathrm{s}^{-1}])$", log_y=True)
    fig.tight_layout()
    for ext in (".png", ".pdf", ".eps"):
        out = plot_dir / ("hubble_log_H_t" + ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  → {out.name}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute H(t), T(t), a(t) for 95%% UL, 68%% UL, and SBBN only."
    )
    parser.add_argument("run_dir", type=str, help="Run directory (e.g. runs/CC_2026-03-10_23-24-10)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute background even if hubble_background.npz exists.",
    )
    args = parser.parse_args()
    run_hubble_analysis(Path(args.run_dir), force=args.force)
    print("\n✓ Done.")


if __name__ == "__main__":
    main()

