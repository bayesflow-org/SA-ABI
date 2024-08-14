import os
import sys

# Set paths
sys.path.extend([
    os.path.abspath(os.path.join("../..")),
    os.path.abspath(os.path.join("../../../BayesFlow_dev/BayesFlow/"))
])

import argparse
import time
import pickle
import bayesflow as bf
import numpy as np
import pandas as pd
import tensorflow as tf

from simulation import setup_simulator
from network import setup_network


def train_network(i, m, config, checkpoint_folder, alpha_values, sim_budget, num_epochs):
    """
    Trains a network.

    Parameters
    ----------
    i : int
        Index of the network.
    m : int
        Index of the ensemble member.
    config : dict
        Configuration parameters.
    checkpoint_folder : str
        Path to the folder where checkpoints will be saved.
    alpha_values : list
        List of alpha values for powerscaling.
    sim_budget : int
        Simulation budget.
    num_epochs : int
        Number of epochs for training.

    Returns
    -------
    network_name : str
        Name of the network.
    training_time : float
        Training time.
    """
    network_name, network_num, checkpoint_path, is_fix_alpha = get_network_info(i, m, checkpoint_folder, alpha_values)
    
    if i == 0: # Powerscaled network
        prior, simulator, model = setup_simulator(config=config)
        print(f"Starting training of {network_name}_{network_num}...")
    else: # Unscaled networks
        prior, simulator, model = setup_simulator(config=config, fix_alpha=alpha_values[i-1])
        print(f"Starting training of {network_name}_{network_num} with alpha of {alpha_values[i-1]}...")

    start_time = time.time()

    amortizer, trainer = setup_network(
        generative_model=model,
        checkpoint_path=checkpoint_path,
        is_fix_alpha=is_fix_alpha
    )

    # Generate training and validation data
    offline_data = model(sim_budget)
    val_data = model(500)

    with tf.device('/cpu:0'):  # Faster for these small networks
        h = trainer.train_offline(offline_data, epochs=num_epochs, batch_size=32, validation_sims=val_data)

    end_time = time.time()
    training_time = end_time - start_time

    save_loss_trajectory(h, network_name, network_num)

    return network_name, network_num, training_time


def get_network_info(i, m, checkpoint_folder, alpha_values):
    """
    Gets information differing between the powerscaled/unscaled setting.

    Parameters
    ----------
    i : int
        Index of the network.
    m : int
        Index of the ensemble member.
    checkpoint_folder : str
        Path to the folder where checkpoints will be saved.
    alpha_values : list
        List of alpha values for powerscaling.

    Returns
    -------
    network_name : str
        Name of the network.
    checkpoint_path: str
        Path to the checkpoint.
    is_fix_alpha : bool
        Whether alpha is fixed.
    """

    if i == 0:  # Powerscaled network
        network_name = "powerscaled"
        network_num = f"net{m}"
        checkpoint_path = f"{checkpoint_folder}/powerscaled/{network_num}"
        is_fix_alpha = False

    else:  # Unscaled networks
        alpha_net = alpha_values[i-1]
        network_name = f"unscaled_alpha_{alpha_net}"
        network_num = f"net{m}"
        checkpoint_path = f"{checkpoint_folder}/unscaled/alpha_{alpha_net}/{network_num}"
        is_fix_alpha = True

    return network_name, network_num, checkpoint_path, is_fix_alpha


def save_loss_trajectory(h, network_name, network_num):
    """Saves a plot of the loss trajectory."""
    f = bf.diagnostics.plot_losses(h["train_losses"], h["val_losses"], moving_average=True)
    fig_path = f"figures/benchmark/{BUDGET_SETTING}"
    os.makedirs(fig_path, exist_ok=True)
    f.savefig(f"{fig_path}/losses_{network_name}_{network_num}.png")


def main():
    """Main function that trains NUM_ENSEMBLE_MEMBERS networks for each setting and saves training times"""
    training_times = {}
    for i in range(len(ALPHA_VALUES) + 1):  # Loop over powerscaled + len(ALPHA_VALUES) unscaled networks
        for m in range(NUM_ENSEMBLE_MEMBERS):
            network_name, network_num, training_time = train_network(
                i,
                m,
                config,
                CHECKPOINT_FOLDER,
                ALPHA_VALUES,
                SIM_BUDGET,
                NUM_EPOCHS
            )
            net = f"{network_name}_{network_num}"
            training_times[net] = training_time

    with open(f"{CHECKPOINT_FOLDER}/training_times.pkl", "wb") as f:
        pickle.dump(training_times, f)


if __name__ == "__main__":
    # Parse SIM_BUDGET, NUM_EPOCHS, and NUM_ENSEMBLE_MEMBERS from command line 
    parser = argparse.ArgumentParser(description='Train benchmark')
    parser.add_argument('--sim_budget', type=int, default=2**14, help='Simulation budget')
    parser.add_argument('--num_epochs', type=int, default=75, help='Number of epochs')
    parser.add_argument('--num_ensemble_members', type=int, default=2, help='Number of ensemble members')
    args = parser.parse_args()

    # Initialize constants
    SIM_BUDGET = args.sim_budget
    NUM_EPOCHS = args.num_epochs
    NUM_ENSEMBLE_MEMBERS = args.num_ensemble_members
    BUDGET_SETTING = f"{SIM_BUDGET}_budget"
    ALPHA_VALUES = [0.5, 1.0, 2.0]
    config = {"T": 14, "N": 83e6}
    CHECKPOINT_FOLDER = f"checkpoints/benchmark/{BUDGET_SETTING}"

    # Execute training loop
    main()
