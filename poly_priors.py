"""
poly_priors.py – Interactive Polytropic prior explorer
======================================================

Builds the standard-BBN background once, extracts the scale-factor curve a(t),
and then provides interactive sliders for the Polytropic parameters a_t and rho_t.
For each slider choice it plots rho_EDE evaluated on that fixed SBBN a(t)
trajectory. If the Polytropic density is invalid at any point, the curve falls
back to y = 0 there.

Usage
-----
  uv run poly_priors.py
  uv run poly_priors.py --gamma 1.333333
  uv run poly_priors.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.widgets import Slider

import PRyM.PRyM_init as PRyMini
from PRyM.PRyM_main import PRyMclass
from eden_model import BBN_NUISANCE, make_model


@dataclass(frozen=True)
class SliderSpec:
    name: str
    lo: float
    hi: float
    scale: str

    def to_physical(self, slider_value: float) -> float:
        if self.scale == "log":
            return 10.0**slider_value
        if self.scale == "lin":
            return slider_value
        raise ValueError(f"Unsupported slider prior scale for {self.name}: {self.scale}")

    def label(self) -> str:
        if self.scale == "log":
            return f"log10({self.name})"
        return self.name

    def initial_slider_value(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def format_value(self, physical_value: float) -> str:
        if physical_value == 0.0:
            return "0"
        exp = int(np.floor(np.log10(abs(physical_value))))
        mant = physical_value / (10.0**exp)
        return f"{mant:.2f}e{exp:+d}"


def build_sbbn_background() -> tuple[np.ndarray, np.ndarray]:
    """Run PRyM once for standard BBN and return (t_vec [s], a_vec)."""
    cc = make_model("CC")
    tau_n = BBN_NUISANCE["tau_n"][0][0]
    Omegabh2 = BBN_NUISANCE["Omegabh2"][0][0]
    p_npdg = BBN_NUISANCE["p_npdg"][0][0]
    p_dpHe3g = BBN_NUISANCE["p_dpHe3g"][0][0]

    # SBBN background: zero cosmological-constant contribution.
    cc.configure(0.0, tau_n, Omegabh2, p_npdg, p_dpHe3g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = PRyMclass(
            my_rho_NP=cc.rho_NP,
            my_p_NP=cc.p_NP,
            my_drho_NP_dT=cc.drho_NP_dT,
            my_rho_EDE=cc.rho_EDE,
        )

    t_vec = np.asarray(getattr(solver, "_t_vec"), dtype=float)
    a_vec = np.asarray(solver.a_of_t(t_vec), dtype=float)

    order = np.argsort(t_vec)
    return t_vec[order], a_vec[order]


def load_slider_specs() -> tuple[SliderSpec, SliderSpec]:
    """Read a_t and rho_t prior bounds directly from the Polytropic model definition."""
    priors = make_model("Polytropic").PRIORS
    a_t_bounds, a_t_scale = priors["a_t"]
    rho_t_bounds, rho_t_scale = priors["rho_t_MeV4"]
    return (
        SliderSpec("a_t", float(a_t_bounds[0]), float(a_t_bounds[1]), a_t_scale),
        SliderSpec("rho_t_MeV4", float(rho_t_bounds[0]), float(rho_t_bounds[1]), rho_t_scale),
    )


def _rho_chunk(a_chunk: np.ndarray, a_t_value: float, rho_t_value: float, gamma: float) -> np.ndarray:
    """Compute rho_EDE on one chunk; invalid points become 0."""
    PRyMini.gamma = float(gamma)
    model = make_model("Polytropic")
    model._a_t = float(a_t_value)  # type: ignore[attr-defined]
    model._rho_t = float(rho_t_value)  # type: ignore[attr-defined]
    PRyMini.a_t = float(a_t_value)  # type: ignore[attr-defined]
    PRyMini.rho_t_MeV4 = float(rho_t_value)  # type: ignore[attr-defined]

    out = np.zeros_like(a_chunk, dtype=float)
    for i, a_val in enumerate(a_chunk):
        try:
            rho = float(model.rho_EDE(0.0, float(a_val)))
            out[i] = rho if np.isfinite(rho) else 0.0
        except Exception:
            out[i] = 0.0
    return out


def compute_rho_curve(
    a_vec: np.ndarray,
    a_t_value: float,
    rho_t_value: float,
    gamma: float,
    n_jobs: int,
) -> np.ndarray:
    """Compute rho_EDE(a(t)) in parallel; any error returns a zero curve."""
    try:
        chunks = [chunk for chunk in np.array_split(a_vec, max(1, abs(n_jobs) if n_jobs != -1 else 8)) if len(chunk) > 0]
        parts = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_rho_chunk)(chunk, a_t_value, rho_t_value, gamma) for chunk in chunks
        )
        rho = np.concatenate(parts) if parts else np.zeros_like(a_vec, dtype=float)
        if rho.shape != a_vec.shape or not np.all(np.isfinite(rho)):
            return np.zeros_like(a_vec, dtype=float)
        return rho
    except Exception:
        return np.zeros_like(a_vec, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive sliders for Polytropic (a_t, rho_t) on the SBBN a(t) background."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=getattr(PRyMini, "gamma", 1.333333),
        help="Fixed Polytropic gamma (default: current PRyMini.gamma or 1.333333).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for rho_EDE evaluation (-1 = all cores).",
    )
    args = parser.parse_args()

    PRyMini.gamma = float(args.gamma)
    t_vec, a_vec = build_sbbn_background()
    a_t_spec, rho_t_spec = load_slider_specs()

    a_t0_slider = a_t_spec.initial_slider_value()
    rho_t0_slider = rho_t_spec.initial_slider_value()
    a_t0 = a_t_spec.to_physical(a_t0_slider)
    rho_t0 = rho_t_spec.to_physical(rho_t0_slider)
    rho0 = compute_rho_curve(a_vec, a_t0, rho_t0, args.gamma, args.n_jobs)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.28)

    (line,) = ax.plot(t_vec, rho0, lw=2.0, color="C0")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$\rho_{\mathrm{EDE}}(a(t))$")
    ax.set_title("Polytropic rho_EDE on the SBBN background")
    ax.set_facecolor("white")
    ax.set_yscale("symlog", linthresh=1e-300)
    ax.grid(alpha=0.25)

    value_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax_a_t = plt.axes((0.12, 0.14, 0.76, 0.04), facecolor="white")
    ax_rho_t = plt.axes((0.12, 0.08, 0.76, 0.04), facecolor="white")

    a_t_slider = Slider(
        ax=ax_a_t,
        label=a_t_spec.label(),
        valmin=a_t_spec.lo,
        valmax=a_t_spec.hi,
        valinit=a_t0_slider,
    )
    rho_t_slider = Slider(
        ax=ax_rho_t,
        label=rho_t_spec.label(),
        valmin=rho_t_spec.lo,
        valmax=rho_t_spec.hi,
        valinit=rho_t0_slider,
    )

    def refresh_plot() -> None:
        a_t_val = a_t_spec.to_physical(a_t_slider.val)
        rho_t_val = rho_t_spec.to_physical(rho_t_slider.val)
        rho = compute_rho_curve(a_vec, a_t_val, rho_t_val, args.gamma, args.n_jobs)

        line.set_ydata(rho)
        positive = rho[rho > 0.0]
        if positive.size > 0:
            ymin = max(np.min(positive), 1e-300)
            ymax = max(np.max(positive), ymin * 10.0)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(-1.0, 1.0)

        value_text.set_text(
            f"gamma = {args.gamma:.6g}\n"
            f"a_t = {a_t_spec.format_value(a_t_val)}\n"
            f"rho_t = {rho_t_spec.format_value(rho_t_val)}"
        )
        fig.canvas.draw_idle()

    def on_slider_change(_value: float) -> None:
        refresh_plot()

    a_t_slider.on_changed(on_slider_change)
    rho_t_slider.on_changed(on_slider_change)
    refresh_plot()
    plt.show()


if __name__ == "__main__":
    main()
