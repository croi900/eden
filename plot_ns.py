from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree  # ty:ignore[unresolved-import]

import dynesty
from dynesty import plotting as dyplot

try:
    from getdist import MCSamples, plots as gdplots

    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    print("[WARNING] getdist not found   parameter corner via GetDist skipped.")

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

OBS_COLOR = "#d62728"
TRUTH_COLOR = "#ff7f0e"
BAND_ALPHA = 0.20

OBS = {
    "Yp": (0.245, 0.003, r"$Y_P$ (PDG)"),
    "DoH": (2.547, 0.029, r"$D/H \times 10^5$ (PDG)"),
    "He3oH": (1.08, 0.12, r"$^3He/H \times 10^5$ (Bania+02)"),
    "Li7oH": (1.6, 0.3, r"$^7Li/H \times 10^{10}$ (Sbordone+10)"),
}


def _mk_plotdir(run_dir: Path) -> Path:
    pd = run_dir / "plots"
    pd.mkdir(parents=True, exist_ok=True)
    return pd


PLOT_EXTENSIONS = (".png", ".pdf", ".eps")


def _save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    """Save figure to PNG, PDF, and EPS (stem taken from path)."""
    if tight:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    stem = path.with_suffix("").name if path.suffix else path.name
    base = path.parent / stem
    for ext in PLOT_EXTENSIONS:
        out = path.parent / (stem + ext)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"    {out.name}")
    plt.close(fig)


class DummyResults(dict):
    def __init__(self, samples, logwt, logl):
        super().__init__()
        self["samples"] = samples
        self["logwt"] = logwt
        self["logl"] = logl
        self["logz"] = np.linspace(logl.min(), logl.max(), len(logl))
        self["logzerr"] = np.zeros_like(self["logz"])
        weights = np.exp(logwt - logwt.max())
        self["logvol"] = np.linspace(0, -20, len(logl))
        self["niter"] = len(logl)

        self["samples_id"] = np.arange(len(logl))
        self["samples_batch"] = np.zeros(len(logl), dtype=int)
        self["samples_it"] = np.arange(len(logl))
        self["samples_u"] = np.random.uniform(size=samples.shape)
        nlive_const = 500
        self["samples_n"] = np.ones(len(logl), dtype=int) * nlive_const
        decay_idx = int(len(logl) * 0.9)
        decay_sz = len(logl) - decay_idx
        if decay_sz > 0:
            self["samples_n"][decay_idx:] = np.linspace(
                nlive_const, 1, decay_sz
            ).astype(int)

        self.__dict__.update(self)


def load_metadata(run_dir: Path) -> dict:
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


def load_summary(run_dir: Path) -> dict:
    sf = run_dir / "summary.txt"
    result = {"logZ": None, "logZerr": None, "params": {}}
    if not sf.exists():
        return result
    for line in sf.read_text().splitlines():
        if line.startswith("logZ "):
            result["logZ"] = float(line.split("=")[1])  # ty:ignore[invalid-assignment]
        elif line.startswith("logZerr"):
            result["logZerr"] = float(line.split("=")[1])  # ty:ignore[invalid-assignment]
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


def load_posterior_weighted(run_dir: Path):
    fn = run_dir / "posterior_weighted.csv"
    if not fn.exists():
        return None, None, None, None
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
    param_names = cols[:-2]
    ndim = len(param_names)
    samples_phys = data[:, :ndim]
    logwt = data[:, ndim]
    logl = data[:, ndim + 1]
    return samples_phys, logwt, logl, param_names


def load_posterior_unweighted(run_dir: Path):
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


def load_samples_csv(run_dir: Path):
    fn = run_dir / "samples.csv"
    if not fn.exists():
        return None, None, None, None
    header_line = ""
    with open(fn) as f:
        for line in f:
            if line.startswith("#"):
                header_line = line.lstrip("#").strip()
            else:
                break
    delimiter = None
    with open(fn) as f:
        for line in f:
            if not line.startswith("#"):
                if "," in line:
                    delimiter = ","
                break

    if delimiter == ",":
        cols = header_line.replace(",", " ").split()
    else:
        cols = header_line.split()

    data = np.loadtxt(fn, comments="#", delimiter=delimiter)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    abd_names = ["Yp", "D/H", "He3/H", "Li7/H", "logL"]
    n_abd = 5
    n_params = len(cols) - n_abd
    param_names = cols[:n_params]
    abd_cols = slice(n_params, n_params + n_abd)
    return data, param_names, abd_cols, abd_names


def plot_dynesty_summary(run_dir: Path, plot_dir: Path, meta: dict) -> None:
    samples_phys, logwt, logl, param_names = load_posterior_weighted(run_dir)
    if samples_phys is None:
        return

    pkl_file = run_dir / "results.pkl"
    if pkl_file.exists():
        import pickle

        with open(pkl_file, "rb") as f:
            res = pickle.load(f)
    else:
        print("No results.pkl file, skipping dynesty summary")
        return 

    try:
        fig, axes = dyplot.runplot(res)
        for obj in fig.findobj(mtext.Text):
            try:
                obj.set_fontsize(obj.get_fontsize() * 2)
            except (TypeError, ValueError):
                pass
        _save(fig, plot_dir / "summary_runplot.png", tight=True)
    except Exception as e:
        print(f"  [ERROR] dyplot.runplot failed: {e}")


def plot_traceplot(run_dir: Path, plot_dir: Path, meta: dict) -> None:
    try:
        data, s_names, abd_cols, abd_names = load_samples_csv(run_dir)
        if data is None:
            print("  [SKIP] traceplot - no samples.csv")
            return

        logL_vals = data[:, -1]

        i = 0

        fig = plt.figure(figsize=(12, 4), facecolor="white")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])

        cmap = plt.get_cmap("plasma")

        ax_trace = fig.add_subplot(gs[0])
        ax_marg = fig.add_subplot(gs[1])

        x_vals = data[:, i]

        ax_trace.scatter(logL_vals, x_vals, c=logL_vals, cmap=cmap, s=0.8, alpha=0.9)
        ax_trace.set_yscale("log")
        ax_trace.set_ylabel(s_names[i], fontsize=32)
        ax_trace.set_xlabel(r"$\log L$", fontsize=32)
        ax_trace.tick_params(labelsize=28)

        samples_phys, logwt, logl, param_names = load_posterior_weighted(run_dir)
        if samples_phys is not None and i < samples_phys.shape[1]:
            p_vals = samples_phys[:, i]
            mask = p_vals > 0
            p_vals = p_vals[mask]

            w = np.exp(logwt - logl)
            w = np.exp(logwt - logwt.max())
            w /= w.sum()
            w = w[mask]

            from scipy.stats import gaussian_kde

            log_p_vals = np.log10(p_vals)
            kde = gaussian_kde(log_p_vals, weights=w)

            log_y_d = np.linspace(log_p_vals.min(), log_p_vals.max(), 200)
            x_d = kde(log_y_d)
            y_d = 10**log_y_d

            ax_marg.fill_betweenx(y_d, 0, x_d, alpha=0.6, color="blue")

            median = np.percentile(p_vals, 50)
            ax_marg.axhline(median, color="red", lw=1.5)
        else:
            hist_sz = int(len(x_vals) * 0.2)
            if hist_sz > 0:
                p_vals = x_vals[-hist_sz:]
                mask = p_vals > 0
                p_vals = p_vals[mask]

                log_p_vals = np.log10(p_vals)
                bins = np.histogram_bin_edges(log_p_vals, bins="auto")
                counts, _ = np.histogram(log_p_vals, bins=bins, density=True)
                y_d = 10 ** (0.5 * (bins[:-1] + bins[1:]))

                ax_marg.fill_betweenx(
                    y_d, 0, counts, alpha=0.6, color="gray", step="mid"
                )

                median = np.percentile(p_vals, 50)
                ax_marg.axhline(median, color="red", lw=1.5)

        ax_marg.set_yscale("log")
        ax_marg.set_ylim(ax_trace.get_ylim())

        ax_marg.set_xlabel("Density", fontsize=28)
        ax_marg.set_yticklabels([])
        ax_marg.tick_params(labelsize=24)

        _save(fig, plot_dir / "trace_plot.png", tight=True)
    except Exception as e:
        print(f"  [ERROR] Custom traceplot failed: {e}")


def plot_3d_scatter(run_dir: Path, plot_dir: Path, meta: dict) -> None:

    samples_phys, logwt, logl, param_names = load_posterior_weighted(run_dir)
    is_preliminary = False

    if samples_phys is None or samples_phys.shape[1] < 1:
        data, param_names, _, _ = load_samples_csv(run_dir)
        if data is None or data.shape[1] < 1:
            return
        is_preliminary = True
        samples_phys = data[:, : len(param_names)]
        logwt = data[:, -1]

    model_name = meta.get("model", run_dir.name)
    ndim = samples_phys.shape[1]

    if is_preliminary:
        weights = logwt
        w_sizes = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
    else:
        weights = np.exp(logwt - logwt.max())
        weights /= weights.sum()
        w_sizes = weights / weights.max()

    n_show = min(ndim, 3)
    if n_show < 3:
        return

    x_idx, y_idx, z_idx = 1, 2, 0

    x = samples_phys[:, x_idx]
    y = samples_phys[:, y_idx]
    z = samples_phys[:, z_idx]

    labels = [param_names[x_idx], param_names[y_idx], param_names[z_idx]]

    fig = plt.figure(figsize=(10, 8), facecolor="white")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    sort_idx = np.argsort(weights)
    x = x[sort_idx]
    y = y[sort_idx]
    z = z[sort_idx]
    w_sorted = weights[sort_idx]
    wsz_sorted = w_sizes[sort_idx]

    s = 10 + 200 * wsz_sorted
    sc = ax.scatter(
        x,
        y,
        z,
        c=w_sorted,
        cmap="viridis",
        s=s,
        alpha=0.7,
        linewidths=0.5,
        edgecolors="white",
        depthshade=True,
    )

    ax.set_xlabel(labels[0], labelpad=10, fontsize=28)
    ax.set_ylabel(labels[1], labelpad=10, fontsize=28)
    ax.set_zlabel(labels[2], labelpad=10, fontsize=28)
    ax.tick_params(labelsize=24)

    ax.view_init(elev=20, azim=45)

    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.08)
    cb.ax.tick_params(labelsize=24)
    if is_preliminary:
        cb.set_label("log Likelihood (Preliminary)", fontsize=28)
    else:
        cb.set_label("Posterior weight", fontsize=28)

    _save(fig, plot_dir / "samples_3d_weights.png")


ABD_NAMES = ["Yp", "DoH", "He3oH", "Li7oH"]

ABD_LABELS = [
    r"Y_P",
    r"\mathrm{D}/\mathrm{H}",
    r"^3\mathrm{He}/\mathrm{H}",
    r"^7\mathrm{Li}/\mathrm{H}",
]
N_ABD = 4


def _load_sbbn_abundances() -> np.ndarray | None:
    """
    Load SBBN abundances from sbbn_samples.csv (if present).

    Expected columns (see generate_sbbn.py):
      tau_n, Omegabh2, p_npdg, p_dpHe3g, Yp, DoH, He3oH, Li7oH
    Returns array of shape (Ns, 4) with [Yp, DoH, He3oH, Li7oH].
    """
    root = Path(__file__).resolve().parent
    fn = root / "sbbn_samples.csv"
    if not fn.exists():
        return None
    try:
        data = np.loadtxt(fn, comments="#", delimiter=",")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.shape[1] < 8:
            return None
        return data[:, 4:8]
    except Exception:
        return None


def plot_abundance_corner(run_dir: Path, plot_dir: Path, meta: dict) -> None:
    """
    Corner of Yp, D/H, He3/H, Li7/H mapped from the valid posterior (no rerun).
    Li7/H is always included. Uses GetDist triangle; fallback: matplotlib.
    """
    uw_samples, uw_names = load_posterior_unweighted(run_dir)
    data, param_names, abd_cols, abd_names = load_samples_csv(run_dir)

    if data is None:
        print("  [SKIP] abundance corner   no samples.csv found.")
        return

    model_name = meta.get("model", run_dir.name)
    model_label = "Linear w(T)" if model_name == "TempDependent" else model_name

    if uw_samples is None:
        N = max(1, int(len(data) * 0.3))
        mapped_abundances = data[-N:, abd_cols]
    else:
        ndim = len(param_names)
        sampled_params = data[:, :ndim]
        tree = cKDTree(sampled_params)
        distances, indices = tree.query(uw_samples, k=1)
        mapped_abundances = data[indices, abd_cols]

    active_cols = [0, 1, 2, 3]
    abd = mapped_abundances[:, active_cols]

    abd_sbbn = _load_sbbn_abundances()

    if HAS_GETDIST:
        try:
            mcs = []
            labels = []
            if abd_sbbn is not None:
                mcs.append(
                    MCSamples(
                        samples=abd_sbbn,
                        names=ABD_NAMES,
                        labels=ABD_LABELS,
                        label="SBBN",
                        settings={"smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
                    )
                )
                labels.append("SBBN")
            mcs.append(
                MCSamples(
                    samples=abd,
                    names=ABD_NAMES,
                    labels=ABD_LABELS,
                    label=model_label,
                    settings={"smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
                )
            )
            labels.append(model_label)

            with plt.style.context("default"):
                g = gdplots.getSubplotPlotter(width_inch=3 * N_ABD)
                g.settings.axes_fontsize = 20
                g.settings.axes_labelsize = 24
                g.settings.lab_fontsize = 24

                g.settings.solid_colors = ["#d62728", "#1f77b4"]
                g.settings.num_plot_contours = 2
                g.settings.figure_legend_loc = "upper right"
                g.triangle_plot(mcs, filled=True, legend_labels=labels)
            gfig = g.fig
            for legend in gfig.legends:  # ty:ignore[unresolved-attribute]
                for text in legend.get_texts():
                    text.set_fontsize(text.get_fontsize() * 2)

            means = np.mean(abd, axis=0)
            sigmas = np.std(abd, axis=0)
            for i in range(N_ABD):
                diag_ax = g.subplots[i][i]  # ty:ignore[not-subscriptable]
                txt = f"{means[i]:.3g} $\\pm$ {sigmas[i]:.3g}"
                diag_ax.text(
                    0.5,
                    1.08,
                    txt,
                    transform=diag_ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=15,
                    clip_on=False,
                )
            stem = "abundance_corner"
            for ext in PLOT_EXTENSIONS:
                out = plot_dir / (stem + ext)
                gfig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")  # ty:ignore[unresolved-attribute]
                print(f"    {out.name}")
            plt.close(gfig)
        except Exception as exc:
            print(
                f"  [WARNING] GetDist abundance corner failed: {exc}, using matplotlib fallback"
            )
            _plot_abundance_corner_matplotlib(abd, model_label, plot_dir, abd_sbbn)
    else:
        _plot_abundance_corner_matplotlib(abd, model_label, plot_dir, abd_sbbn)


def _plot_abundance_corner_matplotlib(
    abd: np.ndarray,
    model_name: str,
    plot_dir: Path,
    abd_sbbn: np.ndarray | None = None,
) -> None:
    """Matplotlib-only abundance corner with KDE smoothing (fallback when GetDist unavailable).

    Blue: posterior abundances; optional red overlay: SBBN samples.
    """
    from scipy.stats import gaussian_kde

    n_abd = abd.shape[1]
    fig, axes = plt.subplots(
        n_abd, n_abd, figsize=(3 * n_abd, 3 * n_abd), facecolor="white"
    )
    if n_abd == 1:
        axes = np.array([[axes]])

    labels = [f"${lab}$" for lab in ABD_LABELS[:n_abd]]

    lab_fs, tick_fs = 20, 24
    for row in range(n_abd):
        for col in range(n_abd):
            ax = axes[row, col]
            ax.set_facecolor("white")
            if col > row:
                ax.set_visible(False)
                continue
            xdata = abd[:, col]
            ydata = abd[:, row]
            x_sbbn = abd_sbbn[:, col] if abd_sbbn is not None else None
            y_sbbn = abd_sbbn[:, row] if abd_sbbn is not None else None
            if row == col:
                try:
                    kde = gaussian_kde(xdata, bw_method=0.2)
                    x_min, x_max = xdata.min(), xdata.max()
                    pad = (x_max - x_min) * 0.1 or 1e-10
                    xs = np.linspace(x_min - pad, x_max + pad, 200)
                    ax.fill_between(
                        xs, kde(xs), alpha=0.8, color="#1f77b4", edgecolor="none"
                    )
                except Exception:
                    ax.hist(
                        xdata,
                        bins=40,
                        density=True,
                        color="#1f77b4",
                        alpha=0.8,
                        edgecolor="none",
                    )

                if x_sbbn is not None:
                    try:
                        kde_s = gaussian_kde(x_sbbn, bw_method=0.2)
                        xs_s = np.linspace(x_sbbn.min(), x_sbbn.max(), 200)
                        ax.plot(xs_s, kde_s(xs_s), color="#d62728", lw=1.5, alpha=0.9)
                    except Exception:
                        ax.hist(
                            x_sbbn,
                            bins=40,
                            density=True,
                            histtype="step",
                            color="#d62728",
                            alpha=0.9,
                        )
                mean, sigma = np.mean(xdata), np.std(xdata)
                ax.text(
                    0.5,
                    1.08,
                    f"{mean:.3g} $\\pm$ {sigma:.3g}",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=15,
                    clip_on=False,
                )
                if col == 0:
                    ax.set_ylabel("density", fontsize=lab_fs)
            else:
                try:
                    kde2 = gaussian_kde(np.vstack([xdata, ydata]), bw_method=0.25)
                    x_min, x_max = xdata.min(), xdata.max()
                    y_min, y_max = ydata.min(), ydata.max()
                    px = (x_max - x_min) * 0.1 or 1e-10
                    py = (y_max - y_min) * 0.1 or 1e-10
                    xs = np.linspace(x_min - px, x_max + px, 80)
                    ys = np.linspace(y_min - py, y_max + py, 80)
                    X, Y = np.meshgrid(xs, ys)
                    Z = kde2(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                    ax.contourf(X, Y, Z, levels=12, cmap="Blues", alpha=0.8)
                except Exception:
                    ax.scatter(
                        xdata, ydata, c="#1f77b4", s=2.5, alpha=0.3, linewidths=0
                    )

                if x_sbbn is not None and y_sbbn is not None:
                    try:
                        kde2s = gaussian_kde(
                            np.vstack([x_sbbn, y_sbbn]), bw_method=0.25
                        )
                        xs_s = np.linspace(x_sbbn.min(), x_sbbn.max(), 60)
                        ys_s = np.linspace(y_sbbn.min(), y_sbbn.max(), 60)
                        Xs, Ys = np.meshgrid(xs_s, ys_s)
                        Zs = kde2s(np.vstack([Xs.ravel(), Ys.ravel()])).reshape(
                            Xs.shape
                        )
                        ax.contour(
                            Xs,
                            Ys,
                            Zs,
                            levels=4,
                            colors="#d62728",
                            linewidths=1.0,
                            alpha=0.9,
                        )
                    except Exception:
                        ax.scatter(
                            x_sbbn, y_sbbn, c="#d62728", s=2.0, alpha=0.4, linewidths=0
                        )
            if row == n_abd - 1:
                ax.set_xlabel(labels[col], fontsize=lab_fs)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != col:
                ax.set_ylabel(labels[row], fontsize=lab_fs)
            elif col != 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=tick_fs)
    _save(fig, plot_dir / "abundance_corner.png")



CORNER_MODELS = {"Linear", "TempDependent", "Polytropic", "Poly"}


def process_run(run_dir: Path) -> None:
    sf = run_dir / "summary.txt"
    samples_f = run_dir / "samples.csv"
    if not samples_f.exists():
        print(f"[SKIP] {run_dir.name}   no samples.csv")
        return

    is_preliminary = not sf.exists()

    meta = load_metadata(run_dir)
    if not meta:
        meta = {"model": run_dir.name.split("_")[0]}

    summary = load_summary(run_dir) if not is_preliminary else {}
    model = meta.get("model", "")

    if is_preliminary:
        print(f"\n   {run_dir.name}  [{model}] (PRELIMINARY - still running)")
        pd = run_dir / "preliminary_plots"
        pd.mkdir(exist_ok=True)
        print("  [1/3] Dynesty Summary runplot   skipped (preliminary)")
    else:
        pd = _mk_plotdir(run_dir)
        print(f"\n   {run_dir.name}  [{model}]")
        print("  [1/3] Dynesty Summary runplot ...")
        plot_dynesty_summary(run_dir, pd, meta)

    print("  [2/3] Custom Traceplot ...")
    plot_traceplot(run_dir, pd, meta)

    print(
        "  [3/3] Abundance corner with raw samples.csv ..."
    ) if is_preliminary else print("  [3/3] Abundance corner ...")
    plot_abundance_corner(run_dir, pd, meta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate post-processing plots for Eden nested-sampling runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        metavar="RUN_DIR",
        help="One or more run directories.  "
        "Defaults to every subdirectory of ./runs/ containing a samples.csv.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Base runs directory to scan when no explicit paths are given.",
    )
    args = parser.parse_args()

    if args.run_dirs:
        targets = [Path(d) for d in args.run_dirs]
    else:
        base = Path(args.runs_dir)
        if not base.exists():
            print(f"[ERROR] runs dir {base} not found")
            sys.exit(1)
        targets = sorted(
            d for d in base.iterdir() if d.is_dir() and (d / "samples.csv").exists()
        )
        if not targets:
            print(f"[INFO] No completed runs found in {base}")
            sys.exit(0)

    print(f"Processing {len(targets)} run(s) ...")
    for rd in targets:
        process_run(rd)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
