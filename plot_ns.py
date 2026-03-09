#!/usr/bin/env python

import sys
import os
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings("ignore")


try:
    from getdist import MCSamples, plots as gd_plots

    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    print(
        "Note: getdist not found; corner plots will use matplotlib. "
        "Install with: uv add getdist"
    )

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print(
        "Note: seaborn not found; some plots may look plainer. "
        "Install with: uv add seaborn"
    )


PALETTE = {
    "blue": "#3a86ff",
    "orange": "#ff6600",
    "green": "#38b000",
    "red": "#e63946",
    "purple": "#7b2d8b",
    "grey": "#adb5bd",
}
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


LABELS = {
    "log10_Lambda": r"$\log_{10}(\Lambda_{\rm CC}/{\rm MeV}^4)$",
    "Lambda_MeV4": r"$\Lambda_{\rm CC}$ [MeV$^4$]",
    "tau_n": r"$\tau_n$ [s]",
    "Omegabh2": r"$\Omega_b h^2$",
    "p_npdg": r"$p_{n p d\gamma}$",
    "p_dpHe3g": r"$p_{d p \,^3\!\rm He\,\gamma}$",
    "Yp": r"$Y_P$ (Helium-4)",
    "DoHx1e5": r"$10^5 \cdot D/H$",
    "He3oHx1e5": r"$10^5 \cdot {}^3\rm He/H$",
    "Li7oHx1e10": r"$10^{10} \cdot {}^7\rm Li/H$",
    "logL": r"$\ln\mathcal{L}$",
}


OBS = {
    "Yp": (0.245, 0.003),
    "DoHx1e5": (2.527, 0.030),
    "He3oHx1e5": (1.100, 0.200),
    "Li7oHx1e10": (1.58, 0.31),
}


def load_csv(path: str) -> pd.DataFrame:
    with open(path) as f:
        first = f.readline().strip()
    has_header = first.startswith("#")
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if has_header:
        cols = first.lstrip("#").split()
        if len(cols) == df.shape[1]:
            df.columns = cols

    if "logL" not in df.columns and "loglike" in df.columns:
        df = df.rename(columns={"loglike": "logL"})
    if "logL" not in df.columns:
        df.columns = list(df.columns[:-1]) + ["logL"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["logL"])
    df = df[df["logL"] > -1e10].reset_index(drop=True)
    print(f"  Loaded {len(df)} valid evaluations.")
    return df


def parse_summary_txt(summary_path: pathlib.Path) -> dict:
    result = {}
    if not summary_path.exists():
        return result
    print(f"  Loading summary from: {summary_path.name}")
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("logZ "):
                try:
                    result["logZ"] = float(line.split("=")[1])
                except:
                    pass
            elif line.startswith("logZerr"):
                try:
                    result["logZerr"] = float(line.split("=")[1])
                except:
                    pass
            elif ":" in line and "median" in line:
                try:
                    param = line.split(":")[0].strip()
                    parts = line.split(":", 1)[1].strip()
                    median_val = float(parts.split("median=")[1].split()[0])
                    ci_part = parts.split("68CI=[")[1].split("]")[0]
                    lo, hi = [float(x) for x in ci_part.split(",")]
                    ul95 = float(parts.split("95UL=")[1].split()[0])
                    result[param] = {
                        "median": median_val,
                        "68CI": (lo, hi),
                        "95UL": ul95,
                    }
                except:
                    pass
    return result


def _good(df: pd.DataFrame) -> pd.DataFrame:
    return df


def savefig(fig, out_dir: pathlib.Path, name: str):
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


def plot_lambda_dist(df: pd.DataFrame, out_dir: pathlib.Path, summary: dict):
    col = "log10_Lambda"
    data = df[col].dropna()

    lam_summary = summary.get("Lambda_MeV4", {})
    ul95_val = lam_summary.get("95UL", None)
    ul68_val = lam_summary.get("68CI", (None, None))[1]
    ul95 = np.log10(ul95_val) if ul95_val else np.percentile(data, 95)
    ul68 = np.log10(ul68_val) if ul68_val else np.percentile(data, 68)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(r"Cosmological Constant $\Lambda_{\rm CC}$ Posterior", fontsize=14)

    ax = axes[0]
    if HAS_SEABORN:
        sns.histplot(
            data,
            bins=50,
            kde=True,
            color=PALETTE["blue"],
            alpha=0.55,
            ax=ax,
            stat="density",
        )
    else:
        ax.hist(data, bins=50, density=True, color=PALETTE["blue"], alpha=0.55)
    ax.axvline(
        ul68,
        color=PALETTE["orange"],
        lw=2,
        ls="--",
        label=f"68% UL = {10**ul68:.2e} MeV⁴",
    )
    ax.axvline(
        ul95, color=PALETTE["red"], lw=2, ls="--", label=f"95% UL = {10**ul95:.2e} MeV⁴"
    )
    ax.set_xlabel(LABELS["log10_Lambda"])
    ax.set_ylabel("Density")
    ax.set_title(r"$\log_{10}\Lambda$ distribution")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, cdf, color=PALETTE["blue"], lw=2)
    ax2.axhline(0.68, color=PALETTE["orange"], lw=1.5, ls="--", alpha=0.8, label="68%")
    ax2.axhline(0.95, color=PALETTE["red"], lw=1.5, ls="--", alpha=0.8, label="95%")
    ax2.axvline(ul68, color=PALETTE["orange"], lw=1, ls=":")
    ax2.axvline(ul95, color=PALETTE["red"], lw=1, ls=":")
    ax2.set_xlabel(LABELS["log10_Lambda"])
    ax2.set_ylabel("CDF")
    ax2.set_title("Cumulative distribution")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=9)

    ax2.annotate(
        f"95% UL\n{10**ul95:.2e}",
        xy=(ul95, 0.95),
        xytext=(ul95 - 3, 0.75),
        fontsize=9,
        color=PALETTE["red"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["red"]),
    )

    savefig(fig, out_dir, "01_lambda_distribution.pdf")


def plot_logl_vs_lambda(df: pd.DataFrame, out_dir: pathlib.Path, summary: dict):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sc = ax.scatter(
        df["log10_Lambda"],
        df["logL"],
        c=df["logL"],
        cmap="viridis",
        s=4,
        alpha=0.6,
        vmin=df["logL"].quantile(0.05),
        vmax=df["logL"].max(),
    )
    plt.colorbar(sc, ax=ax, label=LABELS["logL"])
    lam_summary = summary.get("Lambda_MeV4", {})
    ul95_val = lam_summary.get("95UL", None)
    ul95 = np.log10(ul95_val) if ul95_val else np.percentile(df["log10_Lambda"], 95)
    ax.axvline(
        ul95,
        color=PALETTE["red"],
        lw=2,
        ls="--",
        label=f"95% UL ({10**ul95:.2e} MeV\u2074)",
    )
    if "logZ" in summary:
        ax.set_title(
            rf"$\ln Z = {summary['logZ']:.3f} \pm {summary.get('logZerr', 0):.3f}$ — plateau and cliff"
        )
    else:
        ax.set_title(
            r"Log-likelihood vs $\log_{10}\Lambda$ — plateau and cliff structure"
        )
    ax.set_xlabel(LABELS["log10_Lambda"])
    ax.set_ylabel(LABELS["logL"])
    ax.legend(fontsize=9)
    savefig(fig, out_dir, "02_logL_vs_lambda.pdf")


def plot_running_ul(df: pd.DataFrame, out_dir: pathlib.Path, summary: dict):
    sorted_df = df.sort_values("logL", ascending=False).reset_index(drop=True)
    steps = np.arange(50, min(len(sorted_df), 20000) + 1, max(1, len(sorted_df) // 200))
    ul95s, ul68s = [], []
    for n in steps:
        subset = sorted_df.head(n)["log10_Lambda"]
        ul95s.append(np.percentile(subset, 95))
        ul68s.append(np.percentile(subset, 68))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, ul95s, color=PALETTE["red"], lw=2, label="95% UL (cumulative)")
    ax.plot(
        steps,
        ul68s,
        color=PALETTE["orange"],
        lw=2,
        ls="--",
        label="68% UL (cumulative)",
    )

    lam_summary = summary.get("Lambda_MeV4", {})
    if "95UL" in lam_summary:
        auth_ul = np.log10(lam_summary["95UL"])
        ax.axhline(
            auth_ul,
            color=PALETTE["red"],
            lw=1.5,
            ls=":",
            label=f"Summary 95% UL = {lam_summary['95UL']:.2e}",
        )
    ax.axhline(ul95s[-1], color=PALETTE["red"], lw=1, ls=":", alpha=0.3)
    ax.axhline(ul68s[-1], color=PALETTE["orange"], lw=1, ls=":", alpha=0.3)
    ax.set_xlabel("Number of samples used (sorted by logL)")
    ax.set_ylabel(r"$\log_{10}\Lambda_{\rm UL}$")
    ax.set_title("Running upper-limit convergence")
    ax.legend()
    savefig(fig, out_dir, "03_running_ul_convergence.pdf")


def plot_abundance_corner(df: pd.DataFrame, out_dir: pathlib.Path):
    abundance_cols = ["Yp", "DoHx1e5", "He3oHx1e5"]
    if "Li7oHx1e10" in df.columns:
        abundance_cols.append("Li7oHx1e10")
    abundance_cols = [c for c in abundance_cols if c in df.columns]
    if not abundance_cols:
        print("  skipping 04  (no abundance columns in this file)")
        return
    good = df

    if HAS_GETDIST:
        names = abundance_cols
        labels = [LABELS[c].replace("$", "") for c in abundance_cols]
        samples = MCSamples(
            samples=good[abundance_cols].values,
            names=names,
            labels=labels,
            settings={"smooth_scale_2D": 0.4},
        )
        g = gd_plots.get_subplot_plotter(subplot_size=2.5)
        g.triangle_plot(
            [samples], filled=True, contour_colors=[PALETTE["blue"]], title_limit=1
        )
        for i, col in enumerate(abundance_cols):
            if col in OBS:
                mu, sig = OBS[col]
                ax = g.get_axes_for_params(col, col)
                if ax:
                    ax.axvline(mu - sig, color=PALETTE["red"], lw=1, ls="--")
                    ax.axvline(mu, color=PALETTE["red"], lw=1.5)
                    ax.axvline(mu + sig, color=PALETTE["red"], lw=1, ls="--")
        g.fig.suptitle("BBN Abundance Posteriors (getdist)", y=1.01, fontsize=13)  # ty:ignore[unresolved-attribute]
        g.fig.savefig(out_dir / "04_abundance_corner.pdf", bbox_inches="tight")
        plt.close(g.fig)
        print("  saved → 04_abundance_corner.pdf")
    else:
        n = len(abundance_cols)
        fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
        for i, ci in enumerate(abundance_cols):
            for j, cj in enumerate(abundance_cols):
                ax = axes[i][j]
                if i == j:
                    ax.hist(
                        good[ci],
                        bins=40,
                        color=PALETTE["blue"],
                        alpha=0.7,
                        density=True,
                    )
                    if ci in OBS:
                        mu, sig = OBS[ci]
                        ax.axvline(mu, color=PALETTE["red"], lw=1.5)
                        ax.axvspan(mu - sig, mu + sig, color=PALETTE["red"], alpha=0.15)
                elif i > j:
                    ax.scatter(
                        good[cj], good[ci], s=2, alpha=0.3, color=PALETTE["blue"]
                    )
                else:
                    ax.axis("off")
                if j == 0:
                    ax.set_ylabel(LABELS.get(ci, ci), fontsize=8)
                if i == n - 1:
                    ax.set_xlabel(LABELS.get(cj, cj), fontsize=8)
        fig.suptitle("BBN Abundance Posteriors", fontsize=13)
        fig.tight_layout()
        savefig(fig, out_dir, "04_abundance_corner.pdf")


def plot_abundances_vs_lambda(df: pd.DataFrame, out_dir: pathlib.Path):
    abund_cols = ["Yp", "DoHx1e5", "He3oHx1e5"]
    if "Li7oHx1e10" in df.columns:
        abund_cols.append("Li7oHx1e10")
    abund_cols = [c for c in abund_cols if c in df.columns]
    if not abund_cols:
        print("  skipping 05  (no abundance columns in this file)")
        return

    ul95 = np.percentile(df["log10_Lambda"], 95)

    fig, axes = plt.subplots(1, len(abund_cols), figsize=(4.5 * len(abund_cols), 4))
    fig.suptitle(
        r"BBN Abundances vs $\log_{10}\Lambda$ — showing the constraint cliff",
        fontsize=12,
    )

    for ax, col in zip(axes, abund_cols):
        sc = ax.scatter(
            df["log10_Lambda"],
            df[col],
            c=df["logL"],
            cmap="viridis",
            s=3,
            alpha=0.45,
            vmin=df["logL"].quantile(0.1),
            vmax=df["logL"].max(),
        )
        ax.axvline(
            ul95, color=PALETTE["red"], lw=2, ls="--", label=f"95% UL = {10**ul95:.1e}"
        )
        if col in OBS:
            mu, sig = OBS[col]
            ax.axhline(mu, color=PALETTE["orange"], lw=1.5, ls="-", label=r"Obs. $\mu$")
            ax.axhspan(
                mu - sig,
                mu + sig,
                color=PALETTE["orange"],
                alpha=0.18,
                label=r"Obs. $1\sigma$",
            )
        ax.set_xlabel(LABELS["log10_Lambda"])
        ax.set_ylabel(LABELS.get(col, col))
        ax.legend(fontsize=8)
    plt.colorbar(sc, ax=axes[-1], label=LABELS["logL"])
    fig.tight_layout()
    savefig(fig, out_dir, "05_abundances_vs_lambda.pdf")


def plot_nuisance_params(df: pd.DataFrame, out_dir: pathlib.Path):
    nuisance = [
        c for c in ["tau_n", "Omegabh2", "p_npdg", "p_dpHe3g"] if c in df.columns
    ]
    if not nuisance:
        return
    good = df

    ncols = 2
    nrows = (len(nuisance) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, nuisance):
        data = good[col]
        if HAS_SEABORN:
            sns.histplot(
                data,
                bins=40,
                kde=True,
                color=PALETTE["purple"],
                alpha=0.55,
                ax=ax,
                stat="density",
            )
        else:
            ax.hist(data, bins=40, density=True, color=PALETTE["purple"], alpha=0.55)
        med = data.median()
        lo, hi = np.percentile(data, [16, 84])
        ax.axvline(med, color=PALETTE["blue"], lw=2, label=f"Median = {med:.4g}")
        ax.axvline(lo, color=PALETTE["grey"], lw=1.5, ls="--")
        ax.axvline(
            hi,
            color=PALETTE["grey"],
            lw=1.5,
            ls="--",
            label=f"68% CI [{lo:.4g}, {hi:.4g}]",
        )
        ax.set_xlabel(LABELS.get(col, col))
        ax.set_ylabel("Density")
        ax.set_title(f"Nuisance: {LABELS.get(col, col)}")
        ax.legend(fontsize=8)

    for ax in axes[len(nuisance) :]:
        ax.set_visible(False)

    fig.suptitle("Nuisance Parameter Posteriors", fontsize=13)
    fig.tight_layout()
    savefig(fig, out_dir, "06_nuisance_params.pdf")


def plot_sensitivity(df: pd.DataFrame, out_dir: pathlib.Path):
    if "tau_n" not in df.columns:
        print("  skipping 07  (tau_n column absent)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Sensitivity Analysis — plateau structure visualised", fontsize=12)

    ax = axes[0]
    sc = ax.scatter(
        df["log10_Lambda"],
        df["tau_n"],
        c=df["logL"],
        cmap="RdYlGn",
        s=4,
        alpha=0.5,
        vmin=df["logL"].quantile(0.1),
        vmax=df["logL"].max(),
    )
    plt.colorbar(sc, ax=ax, label=LABELS["logL"])
    ax.set_xlabel(LABELS["log10_Lambda"])
    ax.set_ylabel(LABELS["tau_n"])
    ax.set_title(r"$\tau_n$ vs $\log_{10}\Lambda$ coloured by $\ln\mathcal{L}$")

    ax2 = axes[1]
    if "Yp" in df.columns and "Omegabh2" in df.columns:
        sc2 = ax2.scatter(
            df["Omegabh2"],
            df["Yp"],
            c=df["log10_Lambda"],
            cmap="plasma",
            s=4,
            alpha=0.5,
        )
        if "Yp" in OBS:
            mu, sig = OBS["Yp"]
            ax2.axhline(mu, color=PALETTE["red"], lw=1.5)
            ax2.axhspan(
                mu - sig,
                mu + sig,
                color=PALETTE["red"],
                alpha=0.12,
                label=r"Obs. $Y_P\,1\sigma$",
            )
        plt.colorbar(sc2, ax=ax2, label=LABELS["log10_Lambda"])
        ax2.set_xlabel(LABELS["Omegabh2"])
        ax2.set_ylabel(LABELS["Yp"])
        ax2.set_title(r"$Y_P$ vs $\Omega_b h^2$ colour = $\log_{10}\Lambda$")
        ax2.legend(fontsize=9)
    elif "Omegabh2" in df.columns:
        ax2.hist(df["Omegabh2"], bins=40, color=PALETTE["purple"], alpha=0.6)
        ax2.set_xlabel(LABELS["Omegabh2"])
        ax2.set_title(r"$\Omega_b h^2$ distribution")

    fig.tight_layout()
    savefig(fig, out_dir, "07_sensitivity_analysis.pdf")


def plot_chi2_landscape(df: pd.DataFrame, out_dir: pathlib.Path):
    abund_obs = {
        k: v
        for k, v in {"Yp": OBS["Yp"], "DoHx1e5": OBS["DoHx1e5"]}.items()
        if k in df.columns
    }
    if not abund_obs:
        print("  skipping 08  (no abundance columns in this file)")
        return

    fig, axes = plt.subplots(1, len(abund_obs), figsize=(6.5 * len(abund_obs), 4.5))
    fig.suptitle(
        r"$\chi^2$ landscape: which abundance drives the upper limit on $\Lambda$",
        fontsize=11,
    )

    for ax, (col, (mu, sig)) in zip(axes, abund_obs.items()):
        chi2_col = ((df[col] - mu) / sig) ** 2
        sc = ax.scatter(
            df["log10_Lambda"],
            chi2_col,
            c=chi2_col,
            cmap="hot_r",
            s=4,
            alpha=0.5,
            vmin=0,
            vmax=10,
        )
        plt.colorbar(sc, ax=ax, label=r"$\chi^2$ contribution")
        ax.axvline(
            np.percentile(df["log10_Lambda"], 95),
            color=PALETTE["red"],
            lw=2,
            ls="--",
            label="95% UL",
        )
        ax.axhline(
            1.0, color=PALETTE["blue"], lw=1.5, ls="--", label=r"$1\sigma$ ($\chi^2=1$)"
        )
        ax.axhline(
            4.0,
            color=PALETTE["purple"],
            lw=1.5,
            ls="--",
            label=r"$2\sigma$ ($\chi^2=4$)",
        )
        ax.set_xlabel(LABELS["log10_Lambda"])
        ax.set_ylabel(rf"$\chi^2$ from {LABELS.get(col, col)}")
        ax.set_title(f"{LABELS.get(col, col)}")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.2, 12)

    fig.tight_layout()
    savefig(fig, out_dir, "08_chi2_landscape.pdf")


def plot_summary_table(df: pd.DataFrame, out_dir: pathlib.Path):
    abund_defs = {
        k: v
        for k, v in {
            "Yp": {"label": r"$Y_P$", "obs": OBS["Yp"]},
            "DoHx1e5": {"label": r"$10^5 \cdot D/H$", "obs": OBS["DoHx1e5"]},
        }.items()
        if k in df.columns
    }
    if not abund_defs:
        print("  skipping 09  (no abundance columns in this file)")
        return

    ul95 = np.percentile(df["log10_Lambda"], 95)
    ul99 = np.percentile(df["log10_Lambda"], 99)

    sbbn_mask = df["log10_Lambda"] < (df["log10_Lambda"].quantile(0.02))
    ede_mask = (df["log10_Lambda"] > ul95 - 0.5) & (df["log10_Lambda"] < ul95 + 0.5)

    fig, axes = plt.subplots(1, len(abund_defs), figsize=(6 * len(abund_defs), 4))
    fig.suptitle("SBBN vs EDE prediction vs. Observation", fontsize=13)

    for ax, (col, meta) in zip(axes, abund_defs.items()):
        mu, sig = meta["obs"]
        label = meta["label"]
        sbbn_mu = df.loc[sbbn_mask, col].mean()
        sbbn_sig = df.loc[sbbn_mask, col].std()
        ede_mu = df.loc[ede_mask, col].mean() if ede_mask.sum() > 0 else np.nan
        ede_sig = df.loc[ede_mask, col].std() if ede_mask.sum() > 0 else np.nan

        y_positions = [1, 1.5, 2]
        cats = ["Observation", r"SBBN ($\Lambda\to 0$)", r"EDE (at 95% UL)"]
        mus = [mu, sbbn_mu, ede_mu]
        sigs = [sig, sbbn_sig, ede_sig]
        colors = [PALETTE["green"], PALETTE["blue"], PALETTE["red"]]

        for y, cat, m, s, c in zip(y_positions, cats, mus, sigs, colors):
            if np.isfinite(m) and np.isfinite(s):
                ax.errorbar(
                    m, y, xerr=s, fmt="o", color=c, capsize=5, markersize=7, label=cat
                )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(cats, fontsize=10)
        ax.set_xlabel(label)
        ax.set_title(f"Constraint on {label}")
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    savefig(fig, out_dir, "09_sbbn_vs_ede_summary.pdf")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive NS analysis plotter")
    parser.add_argument("file", help="NS samples CSV / posterior TXT file")
    parser.add_argument(
        "--nbest",
        type=int,
        default=0,
        help="Keep only the best N samples by logL. Default 0 = use all.",
    )
    args = parser.parse_args()

    csv_path = pathlib.Path(args.file)
    if not csv_path.exists():
        print(f"Error: '{csv_path}' not found.")
        sys.exit(1)

    out_dir = pathlib.Path(csv_path.stem)
    out_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {out_dir}/\n")

    print("Loading samples …")
    df = load_csv(str(csv_path))
    if args.nbest > 0:
        print(f"  Trimming to top {args.nbest} samples by logL …")
        df = (
            df.sort_values("logL", ascending=False)
            .head(args.nbest)
            .reset_index(drop=True)
        )
        print(f"  Using {len(df)} samples.")

    stem = csv_path.stem

    for suffix in ["_summary.txt", "_summary"]:
        candidate = csv_path.parent / (
            stem.replace("_samples", "").replace("_posterior", "") + suffix
        )
        if candidate.exists():
            summary = parse_summary_txt(candidate)
            break
    else:
        print("  Note: no companion _summary.txt found; ULs computed from data.")
        summary = {}

    if "Lambda_MeV4" not in df.columns:
        print("Error: 'Lambda_MeV4' column not found. Check CSV format.")
        sys.exit(1)

    lam = df["Lambda_MeV4"].copy().where(df["Lambda_MeV4"] > 0)
    df["log10_Lambda"] = np.log10(lam)

    lam_summary = summary.get("Lambda_MeV4", {})
    ul95 = lam_summary.get("95UL", 10 ** np.percentile(df["log10_Lambda"].dropna(), 95))
    ul68 = lam_summary.get(
        "68CI", (None, 10 ** np.percentile(df["log10_Lambda"].dropna(), 68))
    )[1]
    logZ = summary.get("logZ", None)
    logZerr = summary.get("logZerr", None)

    print(f"\n{'=' * 55}")
    if logZ is not None:
        print(f"  ln Z  = {logZ:.4f} ± {logZerr:.4f}")
    print(f"  68% Upper Limit: {ul68:.4e} MeV⁴  (log10 = {np.log10(ul68):.3f})")
    print(f"  95% Upper Limit: {ul95:.4e} MeV⁴  (log10 = {np.log10(ul95):.3f})")
    print(f"  Total samples:   {len(df)}")
    print(f"{'=' * 55}\n")

    print("Generating plots …")
    plot_lambda_dist(df, out_dir, summary)
    plot_logl_vs_lambda(df, out_dir, summary)
    plot_running_ul(df, out_dir, summary)
    plot_abundance_corner(df, out_dir)
    plot_abundances_vs_lambda(df, out_dir)
    plot_nuisance_params(df, out_dir)
    plot_sensitivity(df, out_dir)
    plot_chi2_landscape(df, out_dir)
    plot_summary_table(df, out_dir)

    print(
        f"\nAll done! {len(list(out_dir.glob('*.pdf')))} plots saved to '{out_dir}/'."
    )


if __name__ == "__main__":
    main()
