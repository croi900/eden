import numpy as np
import scipy as sp
import PRyM.PRyM_init as PRyMini
import multiprocessing as mp
from NumbaQuadpack import quadpack_sig, dqags


# cummulative multiprocessed quadrature integration


def _integrate_subinterval(mp_args):
    return sp.integrate.quad(*mp_args)


def cum_quad_mp(
    func,
    a,
    b,
    args=(),
    full_output=0,
    epsabs=1.49e-08,
    epsrel=1.49e-08,
    limit=50,
    points=None,
    weight=None,
    wvar=None,
    wopts=None,
    maxp1=50,
    limlst=50,
    complex_func=False,
):
    x = np.linspace(a, b, PRyMini.cmq_cores + 1)
    args = [
        (
            func,
            x[i],
            x[i + 1],
            args,
            full_output,
            epsabs,
            epsrel,
            limit,
            points,
            weight,
            wvar,
            wopts,
            maxp1,
            limlst,
            complex_func,
        )
        for i in range(PRyMini.cmq_cores)
    ]

    integrals = PRyMini.cmq_pool.map(_integrate_subinterval, args)

    return np.array(integrals).sum(axis=0)


class interp1d(sp.interpolate.interp1d):
    def __call__(self, x):
        x_new = np.asarray(x)

        if self._kind == "linear":
            return np.interp(
                x_new, self.x, self.y, left=self.fill_value, right=self.fill_value
            )
        elif self._kind == "nearest":
            # Use scipy's interp1d for nearest-neighbor interpolation
            interp_func = interp1d(
                self.x, self.y, kind="nearest", fill_value=self.fill_value
            )
            return interp_func(x_new)
        else:
            return super().__call__(x)


def optimize_flags():
    PRyMini.numba_flag = True
    PRyMini.nacreii_flag = True
    PRyMini.tau_n_flag = True
    PRyMini.aTid_flag = False
    PRyMini.smallnet_flag = True
    PRyMini.largenet_flag = not PRyMini.smallnet_flag
    PRyMini.compute_nTOp_flag = False
    PRyMini.save_nTOp_flag = False
    PRyMini.print_ivp = False
