import emcee
import numpy as np
from schwimmbad import MultiPool
from datetime import datetime
import sys


Yp_obs = 0.245
Yp_sig = 0.003


DoH_obs = 2.547
DoH_sig = 0.029


He3oH_obs = 1.08
He3oH_sig = 0.12


Li7oH_obs = 1.6
Li7oH_sig = 0.3


class MCMC:
    def __init__(
        self, model, priors, nsteps=10000, nwalkers=16, nthreads=16, use_Li7=False
    ):
        self.model = model
        self.priors = priors
        self.param_names = list(priors.keys())
        self.ndim = len(self.param_names)

        self.log_space = []
        self.bounds_low = []
        self.bounds_high = []
        for k in self.param_names:
            entry = priors[k]
            if len(entry) == 3 and entry[2] == "log":
                self.log_space.append(True)
                self.bounds_low.append(np.log10(entry[0]))
                self.bounds_high.append(np.log10(entry[1]))
            else:
                self.log_space.append(False)
                self.bounds_low.append(entry[0])
                self.bounds_high.append(entry[1])
        self.log_space = np.array(self.log_space)
        self.bounds_low = np.array(self.bounds_low)
        self.bounds_high = np.array(self.bounds_high)

        self.nsteps = nsteps
        self.nwalkers = max(nwalkers, 2 * self.ndim + 2)
        self.nthreads = nthreads
        self.use_Li7 = use_Li7

        model_name = getattr(model, "model_name", "unknown")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_name = f"ede_{model_name}_{timestamp}.csv"
        self.chain_file = f"chain_{self.file_name}"

    def _to_physical(self, theta):
        physical = np.array(theta, dtype=float)
        physical[self.log_space] = 10.0 ** theta[self.log_space]
        return physical

    def log_prior(self, theta):
        if np.all(theta > self.bounds_low) and np.all(theta < self.bounds_high):
            return 0.0
        return -np.inf

    def log_likelihood(self, theta):
        physical = self._to_physical(theta)
        try:
            res = self.model.abundances(*physical)
            Yp, DoH, He3oH, Li7oH = res[4], res[5], res[6], res[7]
        except Exception:
            return -np.inf

        if not (0.0 < Yp < 1.0):
            return -np.inf

        param_str = " ".join(f"{p:.8e}" for p in physical)
        with open(self.file_name, "a") as f:
            f.write(f"{param_str} {Yp:.8e} {DoH:.8e} {He3oH:.8e} {Li7oH:.8e}\n")

        chi2 = (
            (Yp - Yp_obs) ** 2 / Yp_sig**2
            + (DoH - DoH_obs) ** 2 / DoH_sig**2
            + (He3oH - He3oH_obs) ** 2 / He3oH_sig**2
        )
        if self.use_Li7:
            chi2 += (Li7oH - Li7oH_obs) ** 2 / Li7oH_sig**2

        return -0.5 * chi2

    def log_prob(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def run(self):
        print(
            f"MCMC: model={getattr(self.model, 'model_name', '?')}, "
            f"ndim={self.ndim}, nwalkers={self.nwalkers}, nsteps={self.nsteps}"
        )
        print(f"Parameters: {self.param_names}")
        print(f"Priors: {dict(self.priors)}")
        print(f"Output: {self.file_name}")
        sys.stdout.flush()

        pos = np.random.uniform(
            low=self.bounds_low,
            high=self.bounds_high,
            size=(self.nwalkers, self.ndim),
        )

        header = " ".join(self.param_names) + " Yp DoHx1e5 He3oHx1e5 Li7oHx1e10"
        with open(self.file_name, "w") as f:
            f.write(f"#{header}\n")

        with MultiPool(processes=self.nthreads) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                self.log_prob,
                pool=pool,
                moves=emcee.moves.StretchMove(a=2.0),
            )
            sampler.run_mcmc(pos, self.nsteps, progress=True)

            burn_in = min(100, self.nsteps // 5)
            raw_samples = sampler.get_chain(discard=burn_in, flat=True)

            all_samples = raw_samples.copy()
            all_samples[:, self.log_space] = 10.0 ** raw_samples[:, self.log_space]
            np.savetxt(self.chain_file, all_samples, header=" ".join(self.param_names))

        print(f"\nChain saved to {self.chain_file}")
        print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
        return sampler
