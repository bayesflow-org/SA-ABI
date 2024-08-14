import os
import sys
import numpy as np
from functools import partial
sys.path.extend([os.path.abspath(os.path.join("../../../BayesFlow_dev/BayesFlow/"))])
import bayesflow as bf
RNG = np.random.default_rng(2023)

# Power-scaling
def alpha_gen(lower_bound=0.5, upper_bound=2.0):
    """
    Generates power-scaling parameters from a uniform distribution.
    Samples in log space to have evenly distributed samples below and above 1 (= no scaling).
    Uses base10 to minimize rounding precision loss.
    """
    log_samples = RNG.uniform(np.log10(lower_bound), np.log10(upper_bound), size=5)
    return np.power(10, log_samples)


def alpha_gen_fixed(fix_alpha):
    """Generates fixed power-scaling parameters."""
    return np.repeat(fix_alpha, repeats=5)


# SIR model
def model_prior(alpha):
    """Generates random draws from the prior given an alpha scaling parameter. Each marginal
    prior has its own scaling parameter. See the paper linked below for details:

    https://arxiv.org/abs/2107.14054
    """

    lambd = RNG.lognormal(mean=np.log(0.4), sigma=0.5 / np.sqrt(alpha[0]))
    mu = RNG.lognormal(mean=np.log(1 / 8), sigma=0.2 / np.sqrt(alpha[1]))
    D = RNG.lognormal(mean=np.log(8), sigma=0.2 / np.sqrt(alpha[2]))
    I0 = RNG.gamma(shape=2 * alpha[3] - alpha[3] + 1, scale=20 / alpha[3])
    psi = RNG.exponential(5 / alpha[4])
    return np.array([lambd, mu, D, I0, psi])


def convert_params(mu, phi):
    """Helper function to convert mean/dispersion parameterization of a negative binomial to N and p,
    as expected by numpy.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """

    r = phi
    var = mu + 1 / r * mu**2
    p = (var - mu) / var
    return r, 1 - p


def stationary_SIR(params, N, T, eps=1e-5):
    """Performs a forward simulation from the stationary SIR model given a random draw from the prior,"""

    # Extract parameters and round I0 and D
    lambd, mu, D, I0, psi = params
    I0 = np.ceil(I0)
    D = int(round(D))

    # Initial conditions
    S, I, R = [N - I0], [I0], [0]

    # Reported new cases
    C = [I0]

    # Simulate T-1 timesteps
    for t in range(1, T + D):

        # Calculate new cases
        I_new = lambd * (I[-1] * S[-1] / N)

        # SIR equations
        S_t = S[-1] - I_new
        I_t = np.clip(I[-1] + I_new - mu * I[-1], 0.0, N)
        R_t = np.clip(R[-1] + mu * I[-1], 0.0, N)

        # Track
        S.append(S_t)
        I.append(I_t)
        R.append(R_t)
        C.append(I_new)

    reparam = convert_params(np.clip(np.array(C[D:]), 0, N) + eps, psi)
    C_obs = RNG.negative_binomial(reparam[0], reparam[1])
    return C_obs[:, np.newaxis]


def setup_simulator(config, fix_alpha=None):
    """ Wrapper function to quickly setup the simulator with an optionally given power scaling factor."""
    if fix_alpha:
        prior_context = bf.simulation.ContextGenerator(batchable_context_fun=partial(alpha_gen_fixed, fix_alpha=fix_alpha))
    else:
        prior_context = bf.simulation.ContextGenerator(batchable_context_fun=alpha_gen)
    prior = bf.simulation.Prior(prior_fun=model_prior, context_generator=prior_context)
    simulator = bf.simulation.Simulator(simulator_fun=partial(stationary_SIR, T=config["T"], N=config["N"]))
    model = bf.simulation.GenerativeModel(prior, simulator, name=f"alpha_covid_simulator_{fix_alpha}", skip_test=True)
    return prior, simulator, model