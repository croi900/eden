#!/usr/bin/env python3
"""
Serverless PRyMordial via Modal (ns_cloud.py)

Deploys the PRyMordial ODE BBN calculations to Modal's serverless infrastructure
utilising a container pool to evaluate dynesty candidate points dynamically.

Run with: `uv run ns_cloud.py --model Linear`
"""

import argparse
import datetime
import pathlib
import sys
import threading

import dynesty
import numpy as np

# ====== Modal Application Setup ===============================================

import modal

modal.enable_output()
app = modal.App("prymordial-sampler")

# Create the exact Docker container environment remotely without a Dockerfile!
# We install numba and pre-compile PRyMordial during the image build process
# so that every cold start on Modal already holds the compiled C-binaries!
image = (
    modal.Image.debian_slim(python_version="3.13")
    # 1. Install system dependencies: git (for your repo) AND julia (for PyJulia/diffeqpy)
    .apt_install("git", "build-essential", "cmake")
    # 2. Install all Python dependencies via pip
    .pip_install(
        "numpy",
        "boto3",
        "dynesty",
        "git+https://github.com/croi900/NumbaQuadpack",  # Custom package
        "scipy",  # Prymordial mandatory
        "numba",  # Prymordial recommended
        "numdifftools",  # Prymordial recommended
        "vegas",  # Prymordial recommended
        "diffeqpy",  # Prymordial optional (Python bridge)
    )

    # 4. Set up the working directory
    .workdir("/app")
    # 5. Copy your local files directly into the built image
    .add_local_dir("PRyM", remote_path="/app/PRyM", copy=True)
    .add_local_dir("PRyMrates", remote_path="/app/PRyMrates", copy=True)
    .add_local_file("eden_model.py", remote_path="/app/eden_model.py", copy=True)
    # 6. Warm up your Numba caches
    .run_commands("python eden_model.py")
)

# ====== Global State Functions for Dynesty ====================================

_GLOBAL_MODEL = None
_SAMPLES_FILE = None
_FILE_LOCK = None

Yp_obs = 0.245
Yp_sig = 0.003
DoH_obs = 2.527
DoH_sig = 0.030
He3oH_obs = 1.100
He3oH_sig = 0.200
Li7oH_obs = 1.58
Li7oH_sig = 0.31


def _physical_params(theta, scales) -> list[float]:
    """Convert sampled parameters to physical values."""
    return [(10**t if s == "log" else t) for t, s in zip(theta, scales)]


def _log_likelihood_local(theta) -> float:
    """Fallback likelihood calculator for the master node."""
    global _GLOBAL_MODEL
    physical = _physical_params(
        theta,
        getattr(
            _GLOBAL_MODEL.PRIORS,
            "scales",
            [s[1] for s in _GLOBAL_MODEL.PRIORS.values()],
        ),
    )
    try:
        res = _GLOBAL_MODEL.abundances(*physical)
        chi2 = (
            (res[0] - Yp_obs) ** 2 / Yp_sig**2
            + (res[1] - DoH_obs) ** 2 / DoH_sig**2
            + (res[2] - He3oH_obs) ** 2 / He3oH_sig**2
        )
        if _GLOBAL_MODEL.use_Li7:
            Li7oH = res[3] if len(res) > 3 else 0.0
            chi2 += (Li7oH - Li7oH_obs) ** 2 / Li7oH_sig**2
        return float(-0.5 * chi2)
    except Exception:
        return -1e300


def _prior_transform_local(u):
    """Fallback prior transform for the master node."""
    global _GLOBAL_MODEL
    return _GLOBAL_MODEL.prior_transform_dynesty(u)


def _write_live_samples(results: dynesty.results.Results) -> None:
    """Periodically flush accepted samples to disk."""
    global _SAMPLES_FILE, _FILE_LOCK, _GLOBAL_MODEL
    if _SAMPLES_FILE is None or _FILE_LOCK is None or results is None:
        return

    with _FILE_LOCK:
        try:
            samples = results.samples
            logwt = results.logwt
            logl = results.logl

            with open(_SAMPLES_FILE, "a") as fh:
                s = samples[-1]
                phys = _physical_params(
                    s,
                    getattr(
                        _GLOBAL_MODEL.PRIORS,
                        "scales",
                        [v[1] for v in _GLOBAL_MODEL.PRIORS.values()],
                    ),
                )
                try:
                    res = _GLOBAL_MODEL.abundances(*phys)
                except Exception:
                    res = [0.0, 0.0, 0.0, 0.0]

                s_str = " ".join(f"{x:.6e}" for x in s)
                res_str = " ".join(f"{x:.6e}" for x in res[:4])
                fh.write(f"{s_str} {res_str} {logl[-1]:.6e}\n")
        except Exception as e:
            print(f"Write error: {e}", file=sys.stderr)


# ====== Modal Remote Class ====================================================


@app.cls(
    image=image,
    timeout=600,  # Maximum execution time per batch
    scaledown_window=300,  # Keep the container alive briefly between batches
    max_containers=10,  # Maximum number of 16GB containers to spawn at once
)
@modal.concurrent(max_inputs=6)
class PRyMordialNode:
    """
    A persistent Python object that lives inside a 16GB Modal container.
    """

    @modal.enter()
    def setup(self):
        """Called ONCE when the Modal container spins up."""
        import sys

        sys.path.append("/root")

        # We import the exact same local model file we pushed up
        from eden_model import make_model

        # Instantiate all models to keep them warm in memory
        self.models = {
            "CC": make_model("CC"),
            "Linear": make_model("Linear"),
            "TempDependent": make_model("TempDependent"),
            "Polytropic": make_model("Polytropic"),
        }

    @modal.method()
    def evaluate_theta(self, record: dict) -> float:
        """
        Calculates the likelihood of a single candidate theta point using the
        warn model held perfectly in RAM.
        """
        theta = record["theta"]
        model_name = record["model_name"]
        scales = record["scales"]
        use_li7 = record.get("use_Li7", False)

        physical = _physical_params(theta, scales)
        model = self.models[model_name]

        try:
            res = model.abundances(*physical)
            Yp, DoH, He3oH = res[0], res[1], res[2]
            Li7oH = res[3] if len(res) > 3 else 0.0
        except Exception:
            return -1e300

        chi2 = (
            (Yp - Yp_obs) ** 2 / Yp_sig**2
            + (DoH - DoH_obs) ** 2 / DoH_sig**2
            + (He3oH - He3oH_obs) ** 2 / He3oH_sig**2
        )
        if use_li7:
            chi2 += (Li7oH - Li7oH_obs) ** 2 / Li7oH_sig**2

        return float(-0.5 * chi2)


# ====== Duck Typed Map Pool for Dynesty =======================================


class ModalPool:
    """
    Acts identically to `mp.Pool.map` for Dynesty, intercepting calls and
    dispatching them to Modal's map function.
    """

    def __init__(self, model_name: str, scales: list, use_li7: bool):
        self.node = PRyMordialNode()
        self.model_name = model_name
        self.scales = scales
        self.use_li7 = use_li7

        # Start tracking cost variables
        self.total_invocations = 0

    def map(self, func, iterable):
        # Package every candidate into a dictionary context for the cloud
        tasks = []
        for theta in iterable:
            tasks.append(
                {
                    "theta": np.array(theta),
                    "model_name": self.model_name,
                    "scales": self.scales,
                    "use_Li7": self.use_li7,
                }
            )

        # Fire them to the cloud! `map` automatically parallelizes them across
        # thousands of machines and yields the results in exact order.
        results = list(self.node.evaluate_theta.map(tasks, order_outputs=True))

        self.total_invocations += len(tasks)
        if self.total_invocations % 100 == 0:
            print(f"[Cloud Update] Computed {self.total_invocations} points via Modal.")

        # Sometimes Modal exceptions return None instead of raising locally depending on handles
        return [-1e300 if r is None else r for r in results]


# ====== Main Script Orchestrator ==============================================


def main():
    parser = argparse.ArgumentParser(description="Modal Serverless PRyMordial Sampler")
    parser.add_argument("--model", type=str, default="Linear", help="Model name")
    parser.add_argument("--nlive", type=int, default=50, help="Live points for dynesty")
    parser.add_argument(
        "--dlogz", type=float, default=0.5, help="Stop tolerance (Evidence)"
    )
    parser.add_argument(
        "--use-li7", action="store_true", help="Include Li7 in likelihood"
    )
    parser.add_argument("--label", type=str, default="", help="Optional run label")
    parser.add_argument(
        "--runs-dir", type=str, default="runs", help="Output directory folder"
    )
    parser.add_argument(
        "--concurrency", type=int, default=60, help="Dynesty batch size limit"
    )
    args = parser.parse_args()

    # Create run directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.model}_{timestamp}"
    if args.label:
        run_name += f"_{args.label}"

    run_dir = pathlib.Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_file = run_dir / "samples.csv"

    # Save metadata locally
    from eden_model import make_model

    model = make_model(args.model)
    model.use_Li7 = args.use_li7

    meta = {
        "model": args.model,
        "nlive": args.nlive,
        "dlogz": args.dlogz,
        "use_Li7": args.use_li7,
        "nthreads": args.concurrency,
        "run_dir": str(run_dir),
        "serverless": "Modal.com",
    }
    with open(run_dir / "metadata.txt", "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")

    # Start the Modal cloud connection wrapper!
    print(f"🚀 Deploying to Modal serverless...")
    with app.run():
        global _GLOBAL_MODEL, _SAMPLES_FILE, _FILE_LOCK
        _GLOBAL_MODEL = model
        _SAMPLES_FILE = samples_file
        _FILE_LOCK = threading.Lock()

        header = " ".join(model.param_names) + " Yp DoHx1e5 He3oHx1e5 Li7oHx1e10 logL"
        with open(samples_file, "w") as fh:
            fh.write(f"#{header}\n")

        scales = getattr(model.PRIORS, "scales", [s[1] for s in model.PRIORS.values()])

        # Instantiate our super-simple Modal pool
        cloud_pool = ModalPool(args.model, scales, args.use_li7)

        print(f"Nested sampling starting...")
        print(f"Targeting dlogz: {args.dlogz}")

        # dynesty DynamicNestedSampler does not accept 'queue_size' in modern versions directly if sample="rwalk".
        # the pool handles the concurrency
        sampler_kwargs = dict(ndim=len(model.param_names), sample="rwalk")

        sampler = dynesty.NestedSampler(
            _log_likelihood_local,
            _prior_transform_local,
            pool=cloud_pool,
            queue_size=args.concurrency,
            **sampler_kwargs,
        )

        # Bind the live-writer to intercept samples being saved
        from dynesty.utils import resample_equal

        try:
            for it, res in enumerate(sampler.sample(dlogz=args.dlogz)):
                # Progress is printed dynamically by Modal
                # Note: `res` is a tuple inside the generator, we intercept valid saved points
                _write_live_samples(sampler.results)

                # Custom terminal progress bar to prove it's iterating over the map
                if it % 10 == 0:
                    sys.stdout.write(
                        f"\rDynesty Iterator [iter {it:5d}] | logZ: {sampler.results.logz[-1]:.3f}"
                    )
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\nManually Halting Sampling...")

        print("\n\nComputing final results...")
        sampler.add_live_points()
        results = sampler.results

        print("\n" + "=" * 60)
        print(f"ln Z  = {results.logz[-1]:.3f} ± {results.logzerr[-1]:.3f}")
        print("=" * 60)

        # Save exact posterior array equivalent of what ns.py did
        try:
            import dynesty.utils as dyfunc

            weights = np.exp(results.logwt - results.logz[-1])
            samples_phys = np.array(
                [_physical_params(s, scales) for s in results.samples]
            )
            with open(run_dir / "posterior_weighted.csv", "w") as f:
                header = " ".join(model.param_names) + " weight logL"
                f.write(f"#{header}\n")
                for s, w, l in zip(samples_phys, weights, results.logl):
                    f.write(f"{' '.join(f'{x:.6e}' for x in s)} {w:.6e} {l:.6e}\n")

            samples_equal = dyfunc.resample_equal(samples_phys, weights)
            with open(run_dir / "posterior_unweighted.csv", "w") as f:
                header = " ".join(model.param_names)
                f.write(f"#{header}\n")
                for s in samples_equal:
                    f.write(f"{' '.join(f'{x:.6e}' for x in s)}\n")

            print(f"Saved completed posterior arrays to {run_dir.name}")
        except Exception as e:
            print(f"Failed to run summary calculations locally: {e}")


if __name__ == "__main__":
    main()
