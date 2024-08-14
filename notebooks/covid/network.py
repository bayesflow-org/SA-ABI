import os
import sys
import numpy as np
import tensorflow as tf
from functools import partial
sys.path.extend([os.path.abspath(os.path.join("../../../BayesFlow_dev/BayesFlow/"))])
import bayesflow as bf


def configure_input(forward_dict, prior_means=None, prior_stds=None, is_fix_alpha=False):
    """Function to configure the simulated quantities (i.e., simulator outputs)
    into a neural network-friendly (BayesFlow) format.
    """

    # Prepare placeholder dict
    out_dict = {}

    # Convert data to logscale
    logdata = np.log1p(forward_dict["sim_data"]).astype(np.float32)

    # Extract scaling parameter and convert to array
    alphas = np.array(forward_dict["prior_batchable_context"]).astype(np.float32)

    # Extract prior draws and z-standardize with previously computed means
    params = forward_dict["prior_draws"].astype(np.float32)
    if prior_means is not None and prior_stds is not None: # Standardize only when prior_means and prior_stds are given
        params = (params - prior_means) / prior_stds

    # Remove a batch if it contains nan, inf or -inf
    idx_keep = np.all(np.isfinite(logdata), axis=(1, 2))
    if not np.all(idx_keep):
        print("Invalid value encountered...removing from batch")

    # Add to keys
    out_dict["summary_conditions"] = logdata[idx_keep]
    if not is_fix_alpha:
        out_dict["direct_conditions"] = alphas[idx_keep]
    out_dict["parameters"] = params[idx_keep]

    return out_dict


def setup_network(generative_model, checkpoint_path, prior_means=None, prior_stds=None, is_fix_alpha=False):
    """ Set up a neural network for parameter estimation. 
    is_fix_alpha takes care of removing direct_context for unscaled networks. 
    Input is only standardized when both prior_means and prior_stds are given.
    """

    tf.keras.backend.clear_session()
    summary_net = tf.keras.Sequential(
        [tf.keras.layers.GRU(32, kernel_regularizer=tf.keras.regularizers.L2(1e-4), return_sequences=True), 
        tf.keras.layers.GRU(32, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.L2(1e-5))]
    )
    inference_net = bf.inference_networks.InvertibleNetwork(
        num_params=5, 
        num_coupling_layers=6, 
        coupling_settings=dict(
            dropout_prob=0.1,
            num_dense=2,
            dense_args={'units': 64, 'activation': 'swish', 'kernel_regularizer': tf.keras.regularizers.L2(1e-5)}
        )
    )
    amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
    trainer = bf.trainers.Trainer(
        amortizer=amortizer, 
        generative_model=generative_model, 
        configurator=partial(configure_input, prior_means=prior_means, prior_stds=prior_stds, is_fix_alpha=is_fix_alpha), 
        checkpoint_path=checkpoint_path,
        skip_checks=True
    )
    return amortizer, trainer