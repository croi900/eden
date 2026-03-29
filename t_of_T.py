from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np


def _t_seconds_at_T(
    T_query: np.ndarray,
    t_vec: np.ndarray,
    T_MeV: np.ndarray,
) -> np.ndarray:
    """Interpolate t [s] at photon temperature T [MeV] along one background curve."""
    T = np.asarray(T_MeV, dtype=float).ravel()
    t = np.asarray(t_vec, dtype=float).ravel()
    if T.shape != t.shape:
        raise ValueError("t_vec and T_MeV must have the same shape")
    order = np.argsort(T)
    T_sorted = T[order]
    t_sorted = t[order]
    uniq_T, inv = np.unique(T_sorted, return_inverse=True)
    counts = np.bincount(inv)
    t_unique = np.bincount(inv, weights=t_sorted) / np.maximum(counts, 1)
    Tq = np.asarray(T_query, dtype=float)
    flat = Tq.ravel()
    out = np.interp(flat, uniq_T, t_unique, left=np.nan, right=np.nan)
    return out.reshape(Tq.shape)


def _t_gr_minus_t_ede_over_t_gr(t_gr: np.ndarray, t_ede: np.ndarray) -> np.ndarray:
    """(t_GR - t_EDE) / t_GR; NaN where t_GR <= 0 or undefined."""
    t_gr = np.asarray(t_gr, dtype=float)
    t_ede = np.asarray(t_ede, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (t_gr - t_ede) / t_gr
    return np.where(np.isfinite(f) & (t_gr > 0.0), f, np.nan)


def _t_gr_minus_t_ede_over_t_ede(t_gr: np.ndarray, t_ede: np.ndarray) -> np.ndarray:
    """(t_GR - t_EDE) / t_EDE; NaN where t_EDE <= 0 or undefined."""
    t_gr = np.asarray(t_gr, dtype=float)
    t_ede = np.asarray(t_ede, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (t_gr - t_ede) / t_ede
    return np.where(np.isfinite(f) & (t_ede > 0.0), f, np.nan)


def load_or_compute_backgrounds(
    run_dir: Path,
    *,
    poly_gamma: float | None = None,
    force_compute: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
 
    from hubble_analysis import (
        build_three_param_sets,
        compute_background_for_sample,
        load_metadata,
        load_posterior_unweighted,
        load_summary,
    )

    run_dir = Path(run_dir)
    meta = load_metadata(run_dir)
    model_name = meta.get("model")
    if not model_name:
        raise ValueError(f"No 'model' in {run_dir / 'metadata.txt'}")

    effective_poly_gamma = None
    if model_name == "Polytropic":
        if poly_gamma is not None:
            effective_poly_gamma = float(poly_gamma)
        elif meta.get("gamma_fixed") is not None:
            effective_poly_gamma = float(meta["gamma_fixed"])
        else:
            raise ValueError(
                "Polytropic runs need --poly-gamma or gamma_fixed in metadata.txt"
            )

    samples, param_names = load_posterior_unweighted(run_dir)
    if samples is None:
        raise FileNotFoundError(f"Missing {run_dir / 'posterior_unweighted.csv'}")
    param_list = param_names if isinstance(param_names, list) else list(param_names)

    summary = load_summary(run_dir)
    params_three, labels_three = build_three_param_sets(
        samples, param_list, summary, model_name
    )
    # Index 0: 95% CI, 1: 68% CI, 2: SBBN (GR)
    idx_gr = 2
    idx_95 = 0
    idx_68 = 1

    cache_file = run_dir / "hubble_background.npz"
    if cache_file.is_file() and not force_compute:
        data = np.load(cache_file, allow_pickle=True)
        t_vec = np.asarray(data["t"], dtype=float)
        T_all = np.asarray(data["T_MeV"], dtype=float)
        if T_all.ndim == 1:
            T_all = T_all[np.newaxis, :]
        if T_all.shape[0] < 3:
            raise ValueError(f"Unexpected T_MeV shape in {cache_file}")
        T_gr = T_all[idx_gr]
        T_ede_95 = T_all[idx_95]
        T_ede_68 = T_all[idx_68]
    else:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        res_gr = compute_background_for_sample(
            model_name,
            params_three[idx_gr],
            effective_poly_gamma,
            labels_three[idx_gr],
        )
        res_95 = compute_background_for_sample(
            model_name,
            params_three[idx_95],
            effective_poly_gamma,
            labels_three[idx_95],
        )
        res_68 = compute_background_for_sample(
            model_name,
            params_three[idx_68],
            effective_poly_gamma,
            labels_three[idx_68],
        )
        t_vec = res_gr[0]
        T_gr = res_gr[1]
        # Same time grid as GR (matches hubble_analysis when grids differ)
        T_ede_95 = np.interp(t_vec, res_95[0], res_95[1])
        T_ede_68 = np.interp(t_vec, res_68[0], res_68[1])

    label_gr = labels_three[idx_gr]
    return (t_vec, T_gr, T_ede_95, T_ede_68, label_gr)


def run(
    run_dir: Path,
    *,
    ede_case: str = "95",
    poly_gamma: float | None = None,
    force_compute: bool = False,
    n_T: int = 400,
    out_csv: Path | None = None,
) -> Path:
    """Write CSV with T, t_GR, t_EDE, and ``t_GR - t_EDE``. Returns path to CSV."""
    run_dir = Path(run_dir)
    (t_vec, T_gr, T_ede_95, T_ede_68, label_gr) = load_or_compute_backgrounds(
        run_dir, poly_gamma=poly_gamma, force_compute=force_compute
    )

    if ede_case == "95":
        T_ede = T_ede_95
        ede_label = "EDE (95% UL)"
    elif ede_case == "68":
        T_ede = T_ede_68
        ede_label = "EDE (68% UL)"
    else:
        raise ValueError("ede_case must be '95' or '68'")

    T_min = float(max(np.nanmin(T_gr), np.nanmin(T_ede)))
    T_max = float(min(np.nanmax(T_gr), np.nanmax(T_ede)))
    if not np.isfinite(T_min) or not np.isfinite(T_max) or T_min >= T_max:
        raise RuntimeError(
            f"Invalid overlapping T range [{T_min}, {T_max}] MeV for GR vs EDE."
        )

    T_grid = np.geomspace(T_min, T_max, n_T)

    t_gr = _t_seconds_at_T(T_grid, t_vec, T_gr)
    t_ede = _t_seconds_at_T(T_grid, t_vec, T_ede)
    delta_t_gr_minus_ede = t_gr - t_ede
    frac_gr = _t_gr_minus_t_ede_over_t_gr(t_gr, t_ede)
    frac_ede = _t_gr_minus_t_ede_over_t_ede(t_gr, t_ede)

    out_csv = out_csv or (run_dir / "t_of_T_gr_ede.csv")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    hdr = (
        f"# GR = {label_gr}; EDE = {ede_label}; delta_t_s = t_GR - t_EDE; "
        "cols: /t_GR then /t_EDE\n"
        "T_MeV,t_s_GR,t_s_EDE,delta_t_s,tGR_minus_tEDE_over_tGR,tGR_minus_tEDE_over_tEDE"
    )
    np.savetxt(
        out_csv,
        np.column_stack(
            [T_grid, t_gr, t_ede, delta_t_gr_minus_ede, frac_gr, frac_ede]
        ),
        delimiter=",",
        header=hdr,
        comments="",
    )

    return out_csv


def short_model_legend(run_dir: Path, *, poly_gamma: float | None = None) -> str:
    from hubble_analysis import load_metadata

    meta = load_metadata(Path(run_dir))
    m = meta.get("model", "")
    if m == "CC":
        return "CC"
    if m == "Linear":
        return "Linear"
    if m == "TempDependent":
        return "LinearT"
    if m == "Polytropic":
        g = poly_gamma if poly_gamma is not None else meta.get("gamma_fixed")
        if g is None:
            return "Polytropic"
        gf = float(g)
        if np.isclose(gf, 4.0 / 3.0, rtol=1e-4):
            return r"Polytropic ($\gamma = 4/3$)"
        if np.isclose(gf, 2.0, rtol=1e-5):
            return r"Polytropic ($\gamma = 2$)"
        return rf"Polytropic ($\gamma = {gf:g}$)"
    return m or run_dir.name


def _legend_linear_t_last(handles, labels: list[str]):
    n = len(labels)
    lt = [i for i, lb in enumerate(labels) if lb == "LinearT"]
    rest = [i for i in range(n) if i not in lt]
    order = rest + lt
    return [handles[i] for i in order], [labels[i] for i in order]


def compare_runs_overlay(
    run_dirs: list[Path],
    *,
    ede_case: str = "95",
    poly_gamma: float | None = None,
    force_compute: bool = False,
    n_T: int = 400,
    out_path: Path | None = None,
) -> Path:
    r"""Single figure: $T_\gamma$ (log x) vs $(t_\mathrm{GR} - t_\mathrm{EDE}) / t_\mathrm{GR}$ for each run."""
    if len(run_dirs) < 2:
        raise ValueError("compare_runs_overlay needs at least two run directories")

    curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    T_mins: list[float] = []
    T_maxs: list[float] = []

    for rd in run_dirs:
        rd = Path(rd)
        (t_vec, T_gr, T_95, T_68, _lg) = load_or_compute_backgrounds(
            rd, poly_gamma=poly_gamma, force_compute=force_compute
        )
        T_ede = T_95 if ede_case == "95" else T_68
        T_lo = float(max(np.nanmin(T_gr), np.nanmin(T_ede)))
        T_hi = float(min(np.nanmax(T_gr), np.nanmax(T_ede)))
        if not np.isfinite(T_lo) or not np.isfinite(T_hi) or T_lo >= T_hi:
            raise RuntimeError(f"{rd}: invalid overlapping T range [{T_lo}, {T_hi}] MeV")
        T_mins.append(T_lo)
        T_maxs.append(T_hi)
        label = short_model_legend(rd)
        curves.append((label, t_vec, T_gr, T_ede))

    T_min = max(T_mins)
    T_max = min(T_maxs)
    if T_min >= T_max:
        raise RuntimeError(
            f"No common T interval across runs: max(T_min)={T_min}, min(T_max)={T_max}"
        )
    T_grid = np.geomspace(T_min, T_max, n_T)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(7.0, 4.5), facecolor="white")
    cmap = plt.get_cmap("tab10")
    case_tex = "95" if ede_case == "95" else "68"
    for i, pack in enumerate(curves):
        label, t_vec, T_gr, T_ede = pack
        t_gr = _t_seconds_at_T(T_grid, t_vec, T_gr)
        t_ede = _t_seconds_at_T(T_grid, t_vec, T_ede)
        frac = _t_gr_minus_t_ede_over_t_gr(t_gr, t_ede)
        ax.plot(
            T_grid,
            frac,
            color=cmap(i % 10),
            lw=1.8,
            label=label,
        )

    ax.axhline(0.0, color="0.45", lw=0.9, zorder=0)
    ax.set_xlabel(r"$T_\gamma\ [\mathrm{MeV}]$")
    ax.set_ylabel(
        rf"$(t_\mathrm{{GR}} - t_\mathrm{{EDE}}) / t_\mathrm{{GR}}$"
        f"\n({case_tex}% UL EDE vs SBBN)"
    )
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    
    ax.set_xlim(T_max, T_min)
    h, lab = ax.get_legend_handles_labels()
    h, lab = _legend_linear_t_last(h, lab)
    ax.legend(h, lab, loc="best", fontsize=9, framealpha=0.92)
    ax.set_facecolor("white")
    fig.tight_layout()

    default = Path(run_dirs[0]) / "plots" / "ede_t_shift"
    p = Path(out_path) if out_path is not None else default
    if p.suffix.lower() in {".pdf", ".png", ".eps"}:
        stem = p.parent / p.stem
    else:
        stem = p
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pdf_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="CSV: t(T) for one run. Several runs: ede_t_shift "
        "(T_gamma vs (t_GR - t_EDE) / t_GR) with short model labels."
    )
    p.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        help="One run directory, or several to build a single overlay plot.",
    )
    p.add_argument(
        "--ede-case",
        choices=("95", "68"),
        default="95",
        help="Which EDE parameter set to compare (default: 95%% UL).",
    )
    p.add_argument(
        "--poly-gamma",
        type=float,
        default=None,
        help="Polytropic gamma override (if the run model is Polytropic).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore hubble_background.npz and recompute backgrounds.",
    )
    p.add_argument(
        "--n-T",
        type=int,
        default=400,
        help="Number of log-spaced T points in the output table.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path: CSV for one run; figure for several runs (default: <first>/plots/ede_t_shift).",
    )
    args = p.parse_args(argv)

    if len(args.run_dirs) >= 2:
        out = compare_runs_overlay(
            list(args.run_dirs),
            ede_case=args.ede_case,
            poly_gamma=args.poly_gamma,
            force_compute=args.force,
            n_T=args.n_T,
            out_path=args.output,
        )
        print(f"Wrote {out} and {out.with_suffix('.png')}")
        return

    path = run(
        args.run_dirs[0],
        ede_case=args.ede_case,
        poly_gamma=args.poly_gamma,
        force_compute=args.force,
        n_T=args.n_T,
        out_csv=args.output,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
