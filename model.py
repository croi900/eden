import traceback
import warnings, sys, numpy as np, multiprocessing as mp
from scipy.special import zeta
import PRyM.PRyM_main as PRyMmain
import PRyM.PRyM_init as PRyMini


class Model:
    def __init__(self, mc_key_rates=True):
        PRyMini.NP_e_flag = True  # ty:ignore[invalid-assignment]
        PRyMini.numba_flag = True
        PRyMini.nacreii_flag = PRyMini.smallnet_flag = True
        PRyMini.compute_nTOp_flag = False  # ty:ignore[invalid-assignment]
        self.mc_key_rates = mc_key_rates

    @staticmethod
    def rho_np(Tg, xi):
        return xi * Tg**4

    @staticmethod
    def p_np(Tg, xi, w):
        return w * Model.rho_np(Tg, xi)

    @staticmethod
    def drho_np_dT(Tg, xi):
        return 4 * xi * Tg**3

    def abundances(self, xi, w):
        rho_w = lambda Tg: self.rho_np(Tg, xi)
        p_w = lambda Tg: self.p_np(Tg, xi, w)
        drho_dt_w = lambda Tg: self.drho_np_dT(Tg, xi)

        if self.mc_key_rates:
            PRyMini.tau_n = float(np.random.normal(PRyMini.tau_n, 0.5))
            PRyMini.Omegabh2 = float(np.random.normal(PRyMini.Omegabh2, 2e-4))
            PRyMini.eta0b = PRyMini.Omegabh2_to_eta0b * PRyMini.Omegabh2
            (
                PRyMini.p_npdg,
                PRyMini.p_dpHe3g,
                PRyMini.p_ddHe3n,
                PRyMini.p_ddtp,
                PRyMini.p_tpag,
                PRyMini.p_tdan,
                PRyMini.p_taLi7g,
                PRyMini.p_He3ntp,
                PRyMini.p_He3dap,
                PRyMini.p_He3aBe7g,
                PRyMini.p_Be7nLi7p,
                PRyMini.p_Li7paa,
            ) = np.random.normal(0, 1, 12)
        PRyMini.ReloadKeyRates()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                solver = PRyMmain.PRyMclass(rho_w, p_w, drho_dt_w)
                prym_results = solver.PRyMresults()

                return prym_results
            except Exception as err:
                print("Error!")
                sys.stdout.flush()
                return np.array([np.nan] * 8)
