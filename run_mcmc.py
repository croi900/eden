#!/usr/bin/env python

import argparse
from eden_model import make_model
from mcmc import MCMC


def main():
    from eden_model import MODEL_REGISTRY

    parser = argparse.ArgumentParser(
        description="MCMC parameter estimation for Early Dark Energy models"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY),
        required=True,
        help="EDE model to constrain",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=10000,
        help="Number of MCMC steps per walker (default: 10000)",
    )
    parser.add_argument(
        "--nwalkers",
        type=int,
        default=16,
        help="Number of ensemble walkers (default: 16)",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=16,
        help="Number of parallel threads (default: 16)",
    )
    parser.add_argument(
        "--use-li7",
        action="store_true",
        help="Include 7Li/H in the likelihood (disabled by default)",
    )
    args = parser.parse_args()

    model = make_model(args.model)
    # Build priors dict for MCMC in the format it expects: name -> (lo, hi[, "log"])
    priors = {}
    for name, (bounds, scale) in model.PRIORS.items():
        if scale == "log":
            priors[name] = (bounds[0], bounds[1], "log")
        elif scale == "norm":
            # treat norm-priors as wide uniform for MCMC (MH sampler doesn't do ppf)
            lo = bounds[0] - 5 * bounds[1]
            hi = bounds[0] + 5 * bounds[1]
            priors[name] = (lo, hi)
        else:
            priors[name] = bounds

    sampler = MCMC(
        model,
        priors=priors,
        nsteps=args.nsteps,
        nwalkers=args.nwalkers,
        nthreads=args.nthreads,
        use_Li7=args.use_li7,
    )
    sampler.run()


if __name__ == "__main__":
    main()
