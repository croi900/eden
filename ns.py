import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import threading
from datetime import datetime
from pathlib import Path

import dynesty
import numpy as np
from dynesty import utils as dyfunc
from scipy.stats import norm
from tqdm import tqdm

from eden_model import BaseEDEModel, make_model


#  BBN observables 
Yp_obs, Yp_sig       = 0.245,  0.003
DoH_obs, DoH_sig     = 2.547,  0.029
He3oH_obs, He3oH_sig = 1.08,   0.12
Li7oH_obs, Li7oH_sig = 1.6,    0.3


#  Module-level globals for subprocess workers 
_GLOBAL_MODEL: BaseEDEModel | None = None
_SAMPLES_FILE: str = ""
_FILE_LOCK = threading.Lock()


def _worker_init(model_name: str, samples_file: str) -> None:
    global _GLOBAL_MODEL, _SAMPLES_FILE, _FILE_LOCK
    _GLOBAL_MODEL  = make_model(model_name)
    _SAMPLES_FILE  = samples_file
    _FILE_LOCK     = threading.Lock()


#  NestedSampler 
class NestedSampler:

    def __init__(
        self,
        model: BaseEDEModel,
        nlive: int = 200,
        use_Li7: bool = False,
        dlogz: float = 0.05,
        pool=None,
        nthreads: int = 1,
        run_label: str = "",
        runs_dir: str = "runs",
    ):
        self.model     = model
        self.nlive     = nlive
        self.use_Li7   = use_Li7
        self.dlogz     = dlogz
        self.pool      = pool
        self.nthreads  = nthreads
        self.run_label = run_label

        priors = model.PRIORS
        self.param_names = model.param_names
        self.ndim        = model.ndim

        self.scales = []
        self.lo, self.hi = [], []
        for k in self.param_names:
            param_args, scale = priors[k]
            self.scales.append(scale)
            if scale == "log":
                # param_args are already base-10 exponents, e.g. (-20, -2) => 10^-20 to 10^-2
                self.lo.append(param_args[0])
                self.hi.append(param_args[1])
            else:
                self.lo.append(param_args[0])
                self.hi.append(param_args[1])

        self.lo = np.array(self.lo)
        self.hi = np.array(self.hi)

        #  run directory 
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tag = f"_{run_label}" if run_label else ""
        run_name = f"{model.model_name}{tag}_{timestamp}"
        self.run_dir = Path(runs_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.samples_file      = str(self.run_dir / "samples.csv")
        self.posterior_w_file  = str(self.run_dir / "posterior_weighted.csv")
        self.posterior_uw_file = str(self.run_dir / "posterior_unweighted.csv")
        self.summary_file      = str(self.run_dir / "summary.txt")
        self.metadata_file     = str(self.run_dir / "metadata.txt")

    #  prior helpers 

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        v = np.empty_like(u)
        for i, scale in enumerate(self.scales):
            if scale in ("lin", "log"):
                v[i] = self.lo[i] + u[i] * (self.hi[i] - self.lo[i])
            elif scale == "norm":
                v[i] = norm.ppf(u[i], loc=self.lo[i], scale=self.hi[i])
        return v

    def _physical_params(self, theta: np.ndarray) -> np.ndarray:
        physical = np.array(theta, dtype=float)
        for i, scale in enumerate(self.scales):
            if scale == "log":
                physical[i] = 10.0 ** theta[i]
        return physical

    #  likelihood 

    def log_likelihood(self, theta: np.ndarray) -> float:
        model    = _GLOBAL_MODEL
        physical = self._physical_params(theta)
        try:
            res = model.abundances(*physical)  # ty:ignore[unresolved-attribute]
            Yp, DoH, He3oH, Li7oH = res[4], res[5], res[6], res[7]
        except Exception:
            return -1e300

        if not (0.0 < Yp < 1.0):
            return -1e300

        chi2 = (
            (Yp    - Yp_obs)    ** 2 / Yp_sig    ** 2
            + (DoH  - DoH_obs)  ** 2 / DoH_sig   ** 2
            + (He3oH - He3oH_obs) ** 2 / He3oH_sig ** 2
        )
        if self.use_Li7:
            chi2 += (Li7oH - Li7oH_obs) ** 2 / Li7oH_sig ** 2

        logl = -0.5 * chi2

        param_str = " ".join(f"{p:.8e}" for p in physical)
        line = f"{param_str} {Yp:.8e} {DoH:.8e} {He3oH:.8e} {Li7oH:.8e} {logl:.8e}\n"
        with _FILE_LOCK:
            with open(_SAMPLES_FILE, "a") as fh:
                fh.write(line)

        return logl

    #  tqdm driver 

    def _run_with_tqdm(self, sampler: dynesty.NestedSampler) -> None:
        pbar = tqdm(desc="Nested sampling", unit=" iter", dynamic_ncols=True)
        for _, res in enumerate(sampler.sample(dlogz=self.dlogz)):
            logl_new = res[3]
            logz = (
                sampler.results.logz[-1]
                if hasattr(sampler, "results")
                else float("nan")
            )
            pbar.set_postfix(logL=f"{logl_new:.2f}", logZ=f"{logz:.2f}", refresh=False)
            pbar.update(1)
        for _, res in enumerate(sampler.add_live_points()):
            logl_new = res[3]
            pbar.set_postfix(logL=f"{logl_new:.2f}", live=True, refresh=False)
            pbar.update(1)
        pbar.close()

    #  main run 

    def run(self) -> dynesty.results.Results:
        global _GLOBAL_MODEL, _SAMPLES_FILE, _FILE_LOCK
        print(
            f"nested sampler: model={self.model.model_name}, ndim={self.ndim}, "
            f"nlive={self.nlive}"
        )
        print(f"parameters: {self.param_names}")
        print(f"run dir:    {self.run_dir}")
        sys.stdout.flush()

        header = " ".join(self.param_names) + " Yp DoHx1e5 He3oHx1e5 Li7oHx1e10 logL"
        with open(self.samples_file, "w") as fh:
            fh.write(f"#{header}\n")

        sampler_kwargs = dict(ndim=self.ndim, nlive=self.nlive, sample="rwalk")

        self._write_metadata()

        #  pool / single-thread branch
        if self.pool is not None:
            _GLOBAL_MODEL = self.model
            _SAMPLES_FILE = self.samples_file
            _FILE_LOCK    = threading.Lock()
            pool_size = getattr(self.pool, "_processes", None) or self.nthreads
            sampler = dynesty.NestedSampler(
                self.log_likelihood,
                self.prior_transform,
                pool=self.pool,
                queue_size=pool_size,
                **sampler_kwargs,
            )
            self._run_with_tqdm(sampler)

        elif self.nthreads > 1:
            with mp.Pool(
                processes=self.nthreads,
                initializer=_worker_init,
                initargs=(self.model.model_name, self.samples_file),
            ) as pool:
                sampler = dynesty.NestedSampler(
                    self.log_likelihood,
                    self.prior_transform,
                    pool=pool,
                    queue_size=self.nthreads,
                    **sampler_kwargs,
                )
                self._run_with_tqdm(sampler)

        else:
            _GLOBAL_MODEL = self.model
            _SAMPLES_FILE = self.samples_file
            _FILE_LOCK    = threading.Lock()
            sampler = dynesty.NestedSampler(
                self.log_likelihood,
                self.prior_transform,
                **sampler_kwargs,
            )
            self._run_with_tqdm(sampler)

        results = sampler.results
        self._print_summary(results)
        self._save(results)
        return results

    #  output helpers 

    def _print_summary(self, results: dynesty.results.Results) -> None:
        print("\n" + "=" * 60)
        print(f"ln Z  = {results.logz[-1]:.3f} ± {results.logzerr[-1]:.3f}")
        print("=" * 60)
        weights      = np.exp(results.logwt - results.logz[-1])
        samples_phys = np.array([self._physical_params(s) for s in results.samples])
        for i, name in enumerate(self.param_names):
            col = samples_phys[:, i]
            q16, q50, q84, q95 = dyfunc.quantile(
                col, [0.16, 0.50, 0.84, 0.95], weights=weights
            )
            print(f"  {name}:")
            print(f"    median          = {q50:.4e}")
            print(f"    68% CI          = [{q16:.4e}, {q84:.4e}]")
            print(f"    95% upper limit = {q95:.4e}")
        print("=" * 60)
        sys.stdout.flush()

    def _save(self, results: dynesty.results.Results) -> None:
        weights      = np.exp(results.logwt - results.logz[-1])
        samples_phys = np.array([self._physical_params(s) for s in results.samples])

        #  weighted posterior (CSV) 
        w_header = " ".join(self.param_names) + " logwt loglike"
        w_data   = np.column_stack([samples_phys, results.logwt, results.logl])
        np.savetxt(
            self.posterior_w_file, w_data, header=w_header, delimiter=",",
            comments="#", fmt="%.10e",
        )
        print(f"Weighted posterior   → {self.posterior_w_file}")

        #  unweighted / resampled posterior (CSV) 
        rng = np.random.default_rng(0)
        n_resample = len(weights)
        idx = rng.choice(len(weights), size=n_resample, p=weights / weights.sum(), replace=True)
        uw_samples = samples_phys[idx]
        uw_header  = " ".join(self.param_names)
        np.savetxt(
            self.posterior_uw_file, uw_samples, header=uw_header, delimiter=",",
            comments="#", fmt="%.10e",
        )
        print(f"Unweighted posterior → {self.posterior_uw_file}")

        #  summary text 
        with open(self.summary_file, "w") as f:
            f.write(f"logZ    = {results.logz[-1]:.6f}\n")
            f.write(f"logZerr = {results.logzerr[-1]:.6f}\n")
            for i, name in enumerate(self.param_names):
                col = samples_phys[:, i]
                q16, q50, q84, q95 = dyfunc.quantile(
                    col, [0.16, 0.50, 0.84, 0.95], weights=weights
                )
                f.write(
                    f"{name}: median={q50:.6e}  "
                    f"68CI=[{q16:.6e},{q84:.6e}]  "
                    f"95UL={q95:.6e}\n"
                )
        print(f"Summary              → {self.summary_file}")

    def _write_metadata(self) -> None:
        import dynesty as _dynesty
        pool_type = (
            type(self.pool).__qualname__ if self.pool is not None
            else ("multiprocessing.Pool" if self.nthreads > 1 else "None")
        )
        model_meta = self.model.metadata()
        meta = {
            **model_meta,
            "nlive":        self.nlive,
            "dlogz":        self.dlogz,
            "use_Li7":      self.use_Li7,
            "nthreads":     self.nthreads,
            "pool_type":    pool_type,
            "run_label":    self.run_label,
            "run_dir":      str(self.run_dir),
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "dynesty_v":    _dynesty.__version__,
            "python_v":     platform.python_version(),
            "platform":     platform.platform(),
        }
        with open(self.metadata_file, "w") as f:
            for k, v in meta.items():
                f.write(f"{k}: {json.dumps(v) if isinstance(v, (dict, list)) else v}\n")
        print(f"Metadata             → {self.metadata_file}")


#  CLI 
def main() -> None:
    from eden_model import MODEL_REGISTRY

    parser = argparse.ArgumentParser(
        description="Dynesty nested sampler for EDE BBN constraints"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY),
        default="CC",
        help="EDE model to constrain (default: CC)",
    )
    parser.add_argument(
        "--nlive", type=int, default=200, help="Number of live points (default: 200)"
    )
    parser.add_argument(
        "--nthreads", type=int, default=16, help="Parallel worker processes (default: 1)"
    )
    parser.add_argument(
        "--dlogz",
        type=float,
        default=0.05,
        help="Evidence stopping tolerance dlogZ (default: 0.05)",
    )
    parser.add_argument(
        "--use-li7",
        action="store_true",
        help="Include ⁷Li/H in the likelihood (off by default)",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label appended to the run folder name",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Base directory for run outputs (default: runs/)",
    )
    args = parser.parse_args()

    model = make_model(args.model)
    ns = NestedSampler(
        model,
        nlive=args.nlive,
        nthreads=args.nthreads,
        use_Li7=args.use_li7,
        dlogz=args.dlogz,
        run_label=args.label,
        runs_dir=args.runs_dir,
    )
    ns.run()


if __name__ == "__main__":
    main()
