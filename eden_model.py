import traceback
import numpy as np

import PRyM.PRyM_init as PRyMini
from PRyM.PRyM_main import PRyMclass


BBN_NUISANCE = {
    "tau_n": ((879.4, 0.6), "norm"),
    "Omegabh2": ((0.02230, 0.00015), "norm"),
    "p_npdg": ((0, 1), "norm"),
    "p_dpHe3g": ((0, 1), "norm"),
}


class BaseEDEModel:
    model_name: str = ""
    dynamical_a: bool = False
    PRIORS: dict = {}

    compute_nTOp_flag: bool = False
    compute_bckg_flag: bool = True
    nacreii_flag: bool = True
    smallnet_flag: bool = True

    def __init__(self):
        if not self.model_name:
            raise TypeError("BaseEDEModel must be subclassed.")

    @property
    def param_names(self) -> list[str]:
        return list(self.PRIORS.keys())

    @property
    def ndim(self) -> int:
        return len(self.PRIORS)

    @property
    def param_priors(self) -> dict:
        return dict(self.PRIORS)

    def _configure_prym_flags(self):
        PRyMini.compute_bckg_flag = self.compute_bckg_flag  # ty:ignore[invalid-assignment]
        PRyMini.compute_nTOp_flag = self.compute_nTOp_flag  # ty:ignore[invalid-assignment]
        PRyMini.nacreii_flag = self.nacreii_flag  # ty:ignore[invalid-assignment]
        PRyMini.smallnet_flag = self.smallnet_flag  # ty:ignore[invalid-assignment]
        PRyMini.dynamical_a_flag = self.dynamical_a  # ty:ignore[invalid-assignment]

    def _configure_nuisance(self, tau_n, Omegabh2, p_npdg, p_dpHe3g):
        PRyMini.tau_n = tau_n * PRyMini.second
        PRyMini.Omegabh2 = Omegabh2
        PRyMini.eta0b = PRyMini.Omegabh2_to_eta0b * Omegabh2
        PRyMini.p_npdg = p_npdg
        PRyMini.p_dpHe3g = p_dpHe3g

    def rho_NP(self, T, a=None) -> float:
        return 0.0

    def p_NP(self, T, a=None) -> float:
        return 0.0

    def drho_NP_dT(self, T, a=None) -> float:
        return 0.0

    def rho_EDE(self, Tg, a) -> float:
        # Non-interacting dark energy; only enters Hubble(). Override in subclasses.
        return 0.0

    def configure(self, *params):
        raise NotImplementedError

    def abundances(self, *params) -> np.ndarray:
        self.configure(*params)
        try:
            result = PRyMclass(
                my_rho_NP=self.rho_NP,
                my_p_NP=self.p_NP,
                my_drho_NP_dT=self.drho_NP_dT,
                my_rho_EDE=self.rho_EDE,
            )
            return result.res
        except Exception:
            traceback.print_exc()
            return np.full(8, np.nan)

    def metadata(self) -> dict:
        return {
            "model": self.model_name,
            "dynamical_a": self.dynamical_a,
            "nacreii_flag": self.nacreii_flag,
            "smallnet_flag": self.smallnet_flag,
            "compute_nTOp": self.compute_nTOp_flag,
            "compute_bckg": self.compute_bckg_flag,
            "params": self.param_names,
            "priors": self.param_priors,
        }


class CCModel(BaseEDEModel):
    model_name = "CC"
    dynamical_a = False
    PRIORS = {
        "Lambda_MeV4": ((-20.0, -4.0), "log"),
        **BBN_NUISANCE,
    }

    def configure(self, Lambda_MeV4, tau_n, Omegabh2, p_npdg, p_dpHe3g):  # ty:ignore[invalid-method-override]
        self._configure_prym_flags()
        PRyMini.model = "CC"  # ty:ignore[invalid-assignment]
        self._Lambda_MeV4 = Lambda_MeV4  # store for NP closures
        PRyMini.Lambda_MeV4 = Lambda_MeV4
        self._configure_nuisance(tau_n, Omegabh2, p_npdg, p_dpHe3g)

    def rho_EDE(self, Tg, a) -> float:
        # CC: energy density is constant regardless of a
        return getattr(self, "_Lambda_MeV4", PRyMini.Lambda_MeV4)

    def p_NP(self, T, a=None) -> float:
        return -self.rho_EDE(T, a)  # w = -1

    def drho_NP_dT(self, T, a=None) -> float:
        return 0.0

    def metadata(self) -> dict:
        d = super().metadata()
        d["Lambda_MeV4_default"] = PRyMini.Lambda_MeV4
        return d


class LinearModel(BaseEDEModel):
    model_name = "Linear"
    dynamical_a = True
    PRIORS = {
        "rho0_MeV4": ((-30.0, -2.0), "log"),
        "w": ((-1.0, 0.0), "lin"),
        **BBN_NUISANCE,
    }

    def configure(self, rho0_MeV4, w, tau_n, Omegabh2, p_npdg, p_dpHe3g):  # ty:ignore[invalid-method-override]
        self._configure_prym_flags()
        PRyMini.model = "Linear"  # ty:ignore[invalid-assignment]
        self._rho0 = rho0_MeV4
        self._w = w
        PRyMini.rho0_MeV4 = rho0_MeV4
        PRyMini.w = w
        self._configure_nuisance(tau_n, Omegabh2, p_npdg, p_dpHe3g)

    def rho_EDE(self, Tg, a) -> float:
        # Linear EoS: rho(a) = rho0 * a^{-3(1+w)}
        if a is None:
            return 0.0
        rho0 = getattr(self, "_rho0", PRyMini.rho0_MeV4)
        w = getattr(self, "_w", PRyMini.w)
        return rho0 * a ** (-3.0 * (1.0 + w))

    def p_NP(self, T, a=None) -> float:
        return getattr(self, "_w", PRyMini.w) * self.rho_EDE(T, a)

    def drho_NP_dT(self, T, a=None) -> float:
        return  0

    def metadata(self) -> dict:
        d = super().metadata()
        d["rho0_MeV4_default"] = PRyMini.rho0_MeV4
        d["w_default"] = PRyMini.w
        return d


class PolytropicModel(BaseEDEModel):
    model_name = "Polytropic"
    dynamical_a = True
    # Polytropic/Chaplygin-like EoS with fixed gamma (set via PRyMini.gamma):
    #   p = K rho^gamma,  rho(a) determined by (C, K, gamma).
    # We now sample (C, K) while gamma is a fixed PRyMini flag.
    # To avoid the nested sampler wandering into regions with pathological
    # backgrounds (rho_EDE huge or complex), keep C positive and K mildly
    # negative so EDE remains a small perturbation.
    PRIORS = {
        # C = 10^[−6,0] ∈ [1e−6,1], strictly positive
        "C": ((-30.0, -1.0), "log"),
        # K ∈ [−1,0], modestly negative
        "K": ((-1.0, 1.0), "lin"),
        **BBN_NUISANCE,
    }

    def configure(self, C, K, tau_n, Omegabh2, p_npdg, p_dpHe3g):  # ty:ignore[invalid-method-override]
        self._configure_prym_flags()
        PRyMini.model = "Polytropic"  # ty:ignore[invalid-assignment]
        self._C = C
        self._K = K
        PRyMini.K = K
        self._configure_nuisance(tau_n, Omegabh2, p_npdg, p_dpHe3g)

    def rho_EDE(self, Tg, a) -> float:
        # Polytropic EoS: generalized Chaplygin-like dark energy
        if a is None:
            return 0.0
        C = getattr(self, "_C", getattr(PRyMini, "C", 0.0))
        K = getattr(self, "_K", PRyMini.K)
        gamma = PRyMini.gamma

        # Guard against invalid scalar powers that would yield NaN/complex results.
        # We explicitly check the base and exponent and raise if they are not usable.
        denom = 1.0 - gamma
        if denom == 0.0 or not np.isfinite(denom):
            raise ValueError(f"Invalid gamma={gamma!r} in Polytropic rho_EDE (1-gamma=0).")

        a_power = 3.0 / denom
        if not np.isfinite(a_power):
            raise ValueError(f"Invalid a exponent 3/(1-gamma) for gamma={gamma!r}.")

        if a <= 0.0 or not np.isfinite(a):
            raise ValueError(f"Scale factor a must be positive and finite, got a={a!r}.")

        base = C / (a**a_power) - K
        exponent = 1.0 / denom

        if not np.isfinite(base) or not np.isfinite(exponent):
            raise ValueError(
                f"Non-finite base/exponent in Polytropic rho_EDE: base={base!r}, exponent={exponent!r}"
            )

        # If the base is negative and the exponent is non-integer, the scalar power
        # will produce a complex/NaN; treat this as an error so callers can catch it.
        if base < 0.0 and not np.isclose(exponent, round(exponent)):
            raise ValueError(
                f"Invalid scalar power in Polytropic rho_EDE: base={base!r} < 0 with "
                f"non-integer exponent={exponent!r}."
            )

        return base**exponent

    def p_NP(self, T, a=None) -> float:
        rho = self.rho_EDE(T, a)
        gamma = PRyMini.gamma
        K = getattr(self, "_K", PRyMini.K)
        return K * rho**gamma

    def drho_NP_dT(self, T, a=None) -> float:
        return 0.0

    def metadata(self) -> dict:
        d = super().metadata()
        d["C_default"] = getattr(PRyMini, "C", 0.0)
        d["K_default"] = PRyMini.K
        d["gamma_fixed"] = PRyMini.gamma
        return d


MODEL_REGISTRY: dict[str, type[BaseEDEModel]] = {
    "CC": CCModel,
    "Linear": LinearModel,
    "Polytropic": PolytropicModel,
}


def make_model(name: str) -> BaseEDEModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown EDE model {name!r}. Choose from: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]()
