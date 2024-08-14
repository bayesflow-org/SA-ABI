import os
import sys

sys.path.append(os.path.abspath(os.path.join('../../../BayesFlow/')))
sys.path.append(os.path.abspath(os.path.join('../..')))  # access sibling directories

from src.python.helpers import find_parameter_spans, get_var_from_time_to_threshold, TinySummaryNet, DenseSummaryNet
from src.python.settings import plotting_settings, plotting_update

from functools import partial

import numpy as np
import scipy.stats as stats
import xarray as xr
import tensorflow as tf

import matplotlib.pyplot as plt
plt.rcParams.update(plotting_update)
import seaborn as sns
from collections import OrderedDict
import logging

from bayesflow.amortizers import AmortizedPosterior
from bayesflow.simulation import GenerativeModel, Prior, Simulator, ContextGenerator
from bayesflow.networks import InvertibleNetwork
from bayesflow.trainers import Trainer
import bayesflow.diagnostics as diag




def load_datasets(config, data_path):
    datasets = OrderedDict((name[18:-3], xr.open_dataset(data_path+name).sel(year=slice(*config['year_bounds'])))
                        for name in config['filenames_sims'] if 'tas_anual_preproc' in name)
    return datasets

def build_generative_model(config, datasets, RNG):

    if 'prior_range_override' in config:
        prior_low, prior_high = config['prior_range_override']
    else:
        prior_low, prior_high = find_parameter_spans(datasets, threshold=config['threshold'])

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if 'prior' in config['context']:
        def flat_prior_func():
            time_to_threshold = RNG.uniform(low=prior_low, high=prior_high)
            noise = RNG.normal(loc=0, scale=1)
            return np.array([time_to_threshold, noise])
        def informative_prior_func():
            # sample from truncated normal distribution between 0 and the highest possible value (prior_high)
            prior_mean = config['informative_prior']['mean']
            prior_std = config['informative_prior']['std']
            time_to_threshold = stats.truncnorm.rvs((0 - prior_mean) / prior_std, (prior_high - prior_mean) / prior_std, loc=prior_mean, scale=prior_std)
            noise = RNG.normal(loc=0, scale=1)
            return np.array([time_to_threshold, noise])
        def b_gen():
            return RNG.binomial(1, 0.5)
        def prior_func(b):
            if b:
                return informative_prior_func()
            else:
                return flat_prior_func()

        prior_context = ContextGenerator(batchable_context_fun=b_gen)
        prior = Prior(prior_fun=prior_func, context_generator=prior_context)
        logger.info(
            f"Using uniform and truncated normal, {config['informative_prior']['mean'], config['informative_prior']['std']}, prior on time-to-threshold between {prior_low} and {prior_high}."
        )

    else:
        def flat_prior_func():
            time_to_threshold = RNG.uniform(low=prior_low, high=prior_high)
            noise = RNG.normal(loc=0, scale=1)
            return np.array([time_to_threshold, noise])

        prior = Prior(prior_fun=flat_prior_func)
        logger.info(
            f"Using uniform prior on time-to-threshold between {prior_low} and {prior_high}."
        )

    model_names = list(datasets.keys())

    def model_gen(**kwargs):
        """Chooses a random model name from a uniform distribution."""
        idx = RNG.integers(len(model_names))
        return np.eye(len(model_names))[idx]

    simulator_context = ContextGenerator(batchable_context_fun=model_gen)

    logger.info(
        f"Using the following climate models: {model_names}"
    )

    def likelihood_simulator_lookup(params, model_onehot, threshold, state='train', verbose=False):
        assert len(params)==2, 'Expecting two parameters: time_to_threshold, dummy'
        assert np.sum(model_onehot)==1, 'Expecting one-hot encoding of model index'

        time_to_threshold = params[0]
        model_idx = np.argmax(model_onehot)

        options = get_var_from_time_to_threshold(datasets[model_names[model_idx]], 'TAS', threshold=threshold, time=time_to_threshold)
        options = options - datasets[model_names[model_idx]].TAS_baseline
        n_ensembles = len(np.unique(options.ensemble))

        # Choose a random member.
        # For example, if 10 members are available, first 7 members are used for training,
        # next 2 for validation, last one for testing. Loads split from config.
        assert state in ['train', 'val', 'test'], 'state must be train, test, or val'
        random_option = options.isel(ensemble=RNG.integers(n_ensembles), member=RNG.choice(config['member_split'][state]))

        if verbose:
            logger.info(f"Testing: Looked up data in state {state} ({config['member_split'][state]}) for model {model_names[np.argmax(model_onehot)]}.")

        return random_option

    simulator = Simulator(simulator_fun=partial(likelihood_simulator_lookup, threshold=config["threshold"]), context_generator=simulator_context)
    model = GenerativeModel(prior, simulator)

    logger.info(
        f"Built generative model for temperature maps parametrized by the years before the warming threshold {config['threshold']}°C is reached."
    )

    return model

def build_amortizer(config):
    if config['summary_net']['type'] == 'dense':
        summary_net = DenseSummaryNet(**config['summary_net']['kwargs'])
    elif config['summary_net']['type'] == 'tiny':
        summary_net = TinySummaryNet(**config['summary_net']['kwargs'])
    else:
        raise ValueError('Unknown summary net type')

    if config['inference_net']['type'] == 'bf-invertible':
        inference_net = InvertibleNetwork(**config['inference_net']['kwargs'])
    else:
        raise ValueError('Unknown inference net type')

    amortizer = AmortizedPosterior(inference_net=inference_net, summary_net=summary_net)

    return amortizer



