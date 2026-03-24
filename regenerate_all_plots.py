"""
regenerate_all_plots.py – Convenience wrapper to refresh key plots
===================================================================

Runs `plot_ns.py` and `hubble_analysis.py` on a fixed set of runs so you
do not have to type the commands manually each time.

Usage
-----
  uv run regenerate_all_plots.py
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_ns


# Hardcoded list of runs to refresh.
RUNS = [
    {
        "label": "CC",
        "path": Path("runs/CC_2026-03-10_23-24-10"),
        "model": "CC",
        "poly_gamma": None,
    },
    {
        "label": "Linear",
        "path": Path("runs/Linear_2026-03-10_23-23-45"),
        "model": "Linear",
        "poly_gamma": None,
    },
    {
        "label": "Polytropic",
        "path": Path("runs/Polytropic_2026-03-16_19-18-28"),
        "model": "Polytropic",
        "poly_gamma": 4.0 / 3.0,
    },
    {
        "label": r"Polytropic $(\gamma = 2)$",
        "path": Path("runs/Polytropic_2026-03-17_12-28-05"),
        "model": "Polytropic",
        "poly_gamma": 4.0 / 2.0,
    },
    {
        "label": "TempDependent",
        "path": Path("runs/TempDependent_2026-03-20_09-33-10"),
        "model": "TempDependent",
        "poly_gamma": None,
    },
]


def run_cmd(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _archive_members_for_run(run_dir: Path) -> list[tuple[Path, Path]]:
    """
    Return files to archive for one run.

    Keep lightweight run results only:
      - summary.txt
      - metadata.txt
      - plots/*.pdf
      - plots/*.eps
    Exclude heavy/raw data products such as csv, npz, pkl, png.
    """
    members: list[tuple[Path, Path]] = []
    for name in ("summary.txt", "metadata.txt"):
        f = run_dir / name
        if f.exists():
            members.append((f, run_dir.name / f.name))

    plot_dir = run_dir / "plots"
    if plot_dir.is_dir():
        for ext in (".pdf", ".eps"):
            for f in sorted(plot_dir.glob(f"*{ext}")):
                members.append((f, run_dir.name / "plots" / f.name))

    return members


def make_results_archive(here: Path) -> None:
    """Zip all configured run results, excluding heavy data files."""
    archive_path = here / "run_results_plots_only.zip"
    existing_runs = [here / run["path"] for run in RUNS if (here / run["path"]).is_dir()]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for run_dir in existing_runs:
            for src, arc in _archive_members_for_run(run_dir):
                zf.write(src, arcname=str(arc))
                print(f"  → archived {arc}")
    print(f"\n✓ Wrote archive: {archive_path}")


def _mapped_abundances(run_dir: Path) -> np.ndarray | None:
    """Map posterior samples to abundance columns using plot_ns helpers."""
    uw_samples, _ = plot_ns.load_posterior_unweighted(run_dir)
    data, param_names, abd_cols, _ = plot_ns.load_samples_csv(run_dir)
    if data is None:
        return None

    if uw_samples is None:
        n = max(1, int(len(data) * 0.3))
        mapped = data[-n:, abd_cols]
    else:
        ndim = len(param_names)
        sampled_params = data[:, :ndim]
        tree = plot_ns.cKDTree(sampled_params)
        _, indices = tree.query(uw_samples, k=1)
        mapped = data[indices, abd_cols]
    return mapped[:, [0, 1, 2, 3]]


def make_polytropic_overlay_plot(run_a: dict, run_b: dict, here: Path) -> None:
    """Create a comparison abundance corner for the two Polytropic runs plus SBBN."""
    run_a_dir = here / run_a["path"]
    run_b_dir = here / run_b["path"]
    if not run_a_dir.is_dir() or not run_b_dir.is_dir():
        return

    abd_a = _mapped_abundances(run_a_dir)
    abd_b = _mapped_abundances(run_b_dir)
    abd_sbbn = plot_ns._load_sbbn_abundances()
    if abd_a is None or abd_b is None:
        print("  [SKIP] Polytropic overlay plot – missing abundance data.")
        return

    labels = [
        "Polytropic ($\\gamma = 4/3$)",
        "Polytropic ($\\gamma = 2$)",
    ]
    stem = "abundance_corner_polytropes_overlay"
    target_dirs = [run_a_dir / "plots", run_b_dir / "plots"]
    for pd in target_dirs:
        pd.mkdir(parents=True, exist_ok=True)

    if plot_ns.HAS_GETDIST:
        try:
            mcs = []
            legend_labels = []
            colors = []
            if abd_sbbn is not None:
                mcs.append(
                    plot_ns.MCSamples(
                        samples=abd_sbbn,
                        names=plot_ns.ABD_NAMES,
                        labels=plot_ns.ABD_LABELS,
                        label="SBBN",
                        settings={"smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
                    )
                )
                legend_labels.append("SBBN")
                colors.append("#d62728")

            mcs.append(
                plot_ns.MCSamples(
                    samples=abd_a,
                    names=plot_ns.ABD_NAMES,
                    labels=plot_ns.ABD_LABELS,
                    label=labels[0],
                    settings={"smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
                )
            )
            legend_labels.append(labels[0])
            colors.append("#1f77b4")

            mcs.append(
                plot_ns.MCSamples(
                    samples=abd_b,
                    names=plot_ns.ABD_NAMES,
                    labels=plot_ns.ABD_LABELS,
                    label=labels[1],
                    settings={"smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
                )
            )
            legend_labels.append(labels[1])
            colors.append("#2ca02c")

            with plt.style.context("default"):
                g = plot_ns.gdplots.getSubplotPlotter(width_inch=3 * plot_ns.N_ABD)
                g.settings.axes_fontsize = 20
                g.settings.axes_labelsize = 24
                g.settings.lab_fontsize = 24
                g.settings.solid_colors = colors
                g.settings.num_plot_contours = 2
                g.settings.figure_legend_loc = "upper right"
                g.triangle_plot(mcs, filled=True, legend_labels=legend_labels)

            gfig = g.fig
            for legend in gfig.legends:
                for text in legend.get_texts():
                    text.set_fontsize(text.get_fontsize() * 2)
            for pd in target_dirs:
                for ext in plot_ns.PLOT_EXTENSIONS:
                    out = pd / f"{stem}{ext}"
                    gfig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
                    print(f"  → {out}")
            plt.close(gfig)
            return
        except Exception as exc:
            print(f"  [WARNING] GetDist overlay failed: {exc}, using matplotlib fallback")

    # Fallback: 2D abundance projection overlay (D/H vs Yp)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.scatter(abd_a[:, 1], abd_a[:, 0], s=5, alpha=0.18, label=labels[0], color="#1f77b4")
    ax.scatter(abd_b[:, 1], abd_b[:, 0], s=5, alpha=0.18, label=labels[1], color="#2ca02c")
    if abd_sbbn is not None:
        ax.scatter(abd_sbbn[:, 1], abd_sbbn[:, 0], s=5, alpha=0.12, label="SBBN", color="#d62728")
    ax.set_xlabel(r"$\mathrm{D}/\mathrm{H}$")
    ax.set_ylabel(r"$Y_P$")
    ax.legend(fontsize=18)
    ax.set_facecolor("white")
    ax.grid(alpha=0.25)
    for pd in target_dirs:
        plot_ns._save(fig, pd / f"{stem}.png")
        break
    # _save closes fig; copy generated files to second folder
    first_dir = target_dirs[0]
    second_dir = target_dirs[1]
    for ext in plot_ns.PLOT_EXTENSIONS:
        src = first_dir / f"{stem}{ext}"
        if src.exists():
            (second_dir / src.name).write_bytes(src.read_bytes())


def main() -> None:
    here = Path.cwd()
    for run in RUNS:
        run_dir = here / run["path"]
        if not run_dir.is_dir():
            print(f"[SKIP] {run['label']}: {run_dir} does not exist")
            continue

        print(f"\n=== {run['label']} – {run_dir} ===")

        # 1) Nested-sampling post-processing plots
        run_cmd(["uv", "run", "plot_ns.py", str(run_dir)])

        # 2) Hubble / background plots (95% CI, 68% CI, Observational Data)
        hubble_cmd = ["uv", "run", "hubble_analysis.py", str(run_dir), "--force"]
        if run["model"] == "Polytropic" and run["poly_gamma"] is not None:
            hubble_cmd += ["--poly-gamma", f"{run['poly_gamma']:.6f}"]
        run_cmd(hubble_cmd)

    poly_runs = [run for run in RUNS if run["model"] == "Polytropic"]
    if len(poly_runs) >= 2:
        print("\n=== Polytropic overlay abundance plot ===")
        make_polytropic_overlay_plot(poly_runs[0], poly_runs[1], here)

    print("\n=== Packaging results archive ===")
    # make_results_archive(here)

    print("\n✓ All configured plots regenerated.")


if __name__ == "__main__":
    main()

