from model import Model
from mcmc import MCMC
import time


def main():
    test = Model(mc_key_rates=False)

    mcmc = MCMC(test, 1000, 16, 16)
    mcmc.run()


if __name__ == "__main__":
    main()
