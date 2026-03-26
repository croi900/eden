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
        if a is None:
            return 0.0
        rho0 = getattr(self, "_rho0", PRyMini.rho0_MeV4)
        w = getattr(self, "_w", PRyMini.w)
        return rho0 * a ** (-3.0 * (1.0 + w))

    def p_NP(self, T, a=None) -> float:
        return getattr(self, "_w", PRyMini.w) * self.rho_EDE(T, a)

    def drho_NP_dT(self, T, a=None) -> float:
        return 0

    def metadata(self) -> dict:
        d = super().metadata()
        d["rho0_MeV4_default"] = PRyMini.rho0_MeV4
        d["w_default"] = PRyMini.w
        return d


class TempDependentModel(BaseEDEModel):
    model_name = "TempDependent"
    dynamical_a = False
    PRIORS = {
        "rho0_MeV4": ((-40.0, -12.0), "log"),
        "alpha": ((0.0, 0.095), "lin"),
        **BBN_NUISANCE,
    }

    @staticmethod
    def _T0_MeV() -> float:
        return PRyMini.T0CMB / PRyMini.MeV_to_Kelvin

    def _w_of_T(self, T_MeV: float) -> float:
        alpha_coeff = getattr(self, "_alpha", getattr(PRyMini, "alpha", 0.0))
        beta = getattr(self, "_beta", getattr(PRyMini, "beta", 1.0))
        if not np.isfinite(alpha_coeff) or not np.isfinite(beta):
            raise ValueError(
                f"TempDependent model needs finite alpha and beta, got alpha={alpha_coeff!r}, beta={beta!r}"
            )
        if T_MeV < 0.0 or not np.isfinite(T_MeV):
            raise ValueError(
                f"TempDependent model needs finite non-negative temperature, got T={T_MeV!r}"
            )
        if alpha_coeff == 0.0:
            return -1.0
        return -1.0 + alpha_coeff * T_MeV**beta

    def configure(self, rho0_MeV4, alpha, tau_n, Omegabh2, p_npdg, p_dpHe3g):  # ty:ignore[invalid-method-override]
        self._configure_prym_flags()
        PRyMini.model = "TempDependent"  # ty:ignore[invalid-assignment]
        self._rho0 = rho0_MeV4
        self._alpha = alpha
        self._beta = getattr(PRyMini, "beta", 1.0)
        PRyMini.rho0_MeV4 = rho0_MeV4
        PRyMini.alpha = alpha  # ty:ignore[invalid-assignment]
        PRyMini.beta = self._beta  # ty:ignore[invalid-assignment]
        self._configure_nuisance(tau_n, Omegabh2, p_npdg, p_dpHe3g)

    def rho_EDE(self, Tg, a) -> float:
        if Tg is None or Tg <= 0.0 or not np.isfinite(Tg):
            return 0.0

        rho0 = getattr(self, "_rho0", PRyMini.rho0_MeV4)
        alpha_coeff = getattr(self, "_alpha", getattr(PRyMini, "alpha", 0.0))
        beta = getattr(self, "_beta", getattr(PRyMini, "beta", 1.0))

        if rho0 == 0.0:
            return 0.0
        if rho0 < 0.0 or not np.isfinite(rho0):
            raise ValueError(
                f"TempDependent rho0_MeV4 must be non-negative and finite, got {rho0!r}."
            )
        if not np.isfinite(alpha_coeff) or not np.isfinite(beta):
            raise ValueError(
                f"TempDependent model needs finite alpha and beta, got alpha={alpha_coeff!r}, beta={beta!r}."
            )

        T0 = self._T0_MeV()
        if T0 <= 0.0 or not np.isfinite(T0):
            raise ValueError(
                f"TempDependent T0 must be positive and finite, got {T0!r}."
            )

        w_eff = self._w_of_T(Tg)
        log_rho = np.log(rho0) + 3.0 * (1.0 + w_eff) * (np.log(Tg) - np.log(T0))

        if not np.isfinite(log_rho):
            raise ValueError(
                f"Non-finite log rho in TempDependent rho_EDE: rho0={rho0!r}, alpha={alpha_coeff!r}, beta={beta!r}, T={Tg!r}"
            )
        if log_rho > 700.0:
            raise ValueError(
                f"TempDependent rho_EDE overflow: log_rho={log_rho!r}, rho0={rho0!r}, alpha={alpha_coeff!r}, beta={beta!r}, T={Tg!r}"
            )

        return float(np.exp(log_rho))

    def p_NP(self, T, a=None) -> float:
        if T is None or T <= 0.0 or not np.isfinite(T):
            return 0.0
        return self._w_of_T(T) * self.rho_EDE(T, None)

    def drho_NP_dT(self, T, a=None) -> float:
        return 0.0

    def metadata(self) -> dict:
        d = super().metadata()
        d["rho0_MeV4_default"] = PRyMini.rho0_MeV4
        d["alpha_default"] = getattr(PRyMini, "alpha", 0.0)
        d["beta_fixed"] = getattr(PRyMini, "beta", 1.0)
        return d


class PolytropicModel(BaseEDEModel):
    model_name = "Polytropic"
    dynamical_a = True

    PRIORS = {
        "a_t": ((-15.0, -2.0), "log"),
        "rho_t_MeV4": ((-20.0, 10.0), "log"),
        **BBN_NUISANCE,
    }

    def configure(self, a_t, rho_t_MeV4, tau_n, Omegabh2, p_npdg, p_dpHe3g):  # ty:ignore[invalid-method-override]
        self._configure_prym_flags()
        PRyMini.model = "Polytropic"  # ty:ignore[invalid-assignment]
        self._a_t = a_t
        self._rho_t = rho_t_MeV4
        PRyMini.a_t = a_t  # ty:ignore[invalid-assignment]
        PRyMini.rho_t_MeV4 = rho_t_MeV4  # ty:ignore[invalid-assignment]
        self._configure_nuisance(tau_n, Omegabh2, p_npdg, p_dpHe3g)

    def rho_EDE(self, Tg, a) -> float:
        if a is None:
            return 0.0
        a_t = getattr(self, "_a_t", getattr(PRyMini, "a_t", 0.0))
        rho_t = getattr(self, "_rho_t", getattr(PRyMini, "rho_t_MeV4", 0.0))
        gamma = PRyMini.gamma

        if a_t == 0.0 and rho_t == 0.0:
            return 0.0

        denom = 1.0 - gamma
        if denom == 0.0 or not np.isfinite(denom):
            raise ValueError(
                f"Invalid gamma={gamma!r} in Polytropic rho_EDE (1-gamma=0)."
            )

        if a <= 0.0 or not np.isfinite(a):
            raise ValueError(
                f"Scale factor a must be positive and finite, got a={a!r}."
            )
        if a_t <= 0.0 or not np.isfinite(a_t):
            raise ValueError(
                f"Transition scale a_t must be positive and finite, got a_t={a_t!r}."
            )
        if rho_t <= 0.0 or not np.isfinite(rho_t):
            raise ValueError(
                f"Plateau density rho_t_MeV4 must be positive and finite, got rho_t_MeV4={rho_t!r}."
            )

        alpha = 3.0 * (gamma - 1.0)
        exponent = 1.0 / (1.0 - gamma)
        if not np.isfinite(alpha) or not np.isfinite(exponent):
            raise ValueError(
                f"Invalid alpha/exponent in Polytropic rho_EDE: alpha={alpha!r}, exponent={exponent!r}"
            )

        log_ratio = np.log(a) - np.log(a_t)
        log_x = alpha * log_ratio
        softplus = np.log1p(np.exp(-abs(log_x))) + max(log_x, 0.0)
        log_rho = np.log(rho_t) + exponent * softplus

        if not np.isfinite(log_rho):
            raise ValueError(
                f"Non-finite log rho in Polytropic rho_EDE: log_ratio={log_ratio!r}, log_rho={log_rho!r}"
            )

        return float(np.exp(log_rho))

    def p_NP(self, T, a=None) -> float:
        rho = self.rho_EDE(T, a)
        gamma = PRyMini.gamma
        rho_t = getattr(self, "_rho_t", getattr(PRyMini, "rho_t_MeV4", 0.0))
        if rho_t == 0.0 and rho == 0.0:
            return 0.0
        if rho_t <= 0.0 or not np.isfinite(rho_t):
            raise ValueError(
                f"Plateau density rho_t_MeV4 must be positive and finite for pressure, got rho_t_MeV4={rho_t!r}."
            )
        k_eff = -(rho_t ** (1.0 - gamma))
        return k_eff * rho**gamma

    def drho_NP_dT(self, T, a=None) -> float:
        return 0.0

    def metadata(self) -> dict:
        d = super().metadata()
        d["a_t_default"] = getattr(PRyMini, "a_t", 0.0)
        d["rho_t_MeV4_default"] = getattr(PRyMini, "rho_t_MeV4", 0.0)
        d["gamma_fixed"] = PRyMini.gamma
        return d


MODEL_REGISTRY: dict[str, type[BaseEDEModel]] = {
    "CC": CCModel,
    "Linear": LinearModel,
    "TempDependent": TempDependentModel,
    "Polytropic": PolytropicModel,
}


def make_model(name: str) -> BaseEDEModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown EDE model {name!r}. Choose from: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]()
