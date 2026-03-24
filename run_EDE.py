import numpy as np
import sys
import PRyM.PRyM_init as PRyMini
from PRyM.PRyM_main import PRyMclass
from eden_model import make_model


def run_tests():
    print("=== Testing Standard Model ===")
    PRyMini.model = ""
    PRyMini.compute_bckg_flag = True

    PRyMini.t_end = 1.0e7
    sm = PRyMclass()
    sm_Yp = sm.YPBBN()
    sm_Neff = sm.Neff_f
    print(f"SM   Yp: {sm_Yp:.5f}, Neff: {sm_Neff:.4f}")

    print(f"SM   t(1 MeV): {sm.t_of_T(1.0):.3e} s, t(0.1 MeV): {sm.t_of_T(0.1):.3e} s")

    print("\n=== Testing Cosmological Constant ===")
    PRyMini.model = "CC"  # ty:ignore[invalid-assignment]
    PRyMini.compute_bckg_flag = True

    PRyMini.t_end = 1.0e4
    cc = PRyMclass()
    cc_Yp = cc.YPBBN()
    print(f"CC   Yp: {cc_Yp:.5f}, Neff: {cc.Neff_f:.4f}")
    print(f"CC   t(1 MeV): {cc.t_of_T(1.0):.3e} s, t(0.1 MeV): {cc.t_of_T(0.1):.3e} s")
    if cc_Yp <= sm_Yp:
        print("FAIL: CC model did not increase Yp (higher H -> higher Yp)")
    else:
        print("PASS: CC model increased Yp as expected")

    print("\n=== Testing Linear Model ===")
    PRyMini.model = "Linear"  # ty:ignore[invalid-assignment]
    PRyMini.w = -1.0
    lin = PRyMclass()
    lin_Yp = lin.YPBBN()
    print(f"Lin  Yp: {lin_Yp:.5f}, Neff: {lin.Neff_f:.4f}")
    if not np.isclose(lin_Yp, cc_Yp, rtol=1e-3):
        print("FAIL: Linear(w=-1) does not match CC")
    else:
        print("PASS: Linear model (w=-1) matches CC")

    print("\n=== Testing Temperature-Dependent Model ===")
    temp_model = make_model("TempDependent")
    temp_model.configure(
        PRyMini.Lambda_MeV4,
        0.0,
        PRyMini.tau_n / PRyMini.second,
        PRyMini.Omegabh2,
        PRyMini.p_npdg,
        PRyMini.p_dpHe3g,
    )
    temp = PRyMclass(
        my_rho_NP=temp_model.rho_NP,
        my_p_NP=temp_model.p_NP,
        my_drho_NP_dT=temp_model.drho_NP_dT,
        my_rho_EDE=temp_model.rho_EDE,
    )
    temp_Yp = temp.YPBBN()
    print(f"Temp Yp: {temp_Yp:.5f}, Neff: {temp.Neff_f:.4f}")
    if not np.isclose(temp_Yp, cc_Yp, rtol=1e-3):
        print("FAIL: TempDependent(alpha=0) does not match CC")
    else:
        print("PASS: TempDependent model (alpha=0) matches CC")

    print("\n=== Testing Polytropic Model ===")
    PRyMini.gamma = 4.0 / 3.0
    poly_model = make_model("Polytropic")
    poly_model.configure(1.0e-8, 1.0e-6, PRyMini.tau_n / PRyMini.second, PRyMini.Omegabh2, PRyMini.p_npdg, PRyMini.p_dpHe3g)
    poly = PRyMclass(
        my_rho_NP=poly_model.rho_NP,
        my_p_NP=poly_model.p_NP,
        my_drho_NP_dT=poly_model.drho_NP_dT,
        my_rho_EDE=poly_model.rho_EDE,
    )
    poly_Yp = poly.YPBBN()
    print(f"Poly Yp: {poly_Yp:.5f}, Neff: {poly.Neff_f:.4f}")
    print(f"Poly t(1 MeV): {poly.t_of_T(1.0):.3e} s, t(0.1 MeV): {poly.t_of_T(0.1):.3e} s")
    if np.isnan(poly_Yp) or poly_Yp <= 0.0:
        print("FAIL: Polytropic model failed integration")
    else:
        print("PASS: Polytropic model completed")


if __name__ == "__main__":
    run_tests()
