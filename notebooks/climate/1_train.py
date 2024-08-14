import argparse
import importlib

import time
import numpy as np
import os
import sys
from pprint import pprint

from collections import OrderedDict
from functools import partial

sys.path.append(os.path.abspath(os.path.join('../../../BayesFlow/')))
from bayesflow.trainers import Trainer
import bayesflow.diagnostics as diag

sys.path.append(os.path.abspath(os.path.join('../..')))  # access sibling directories
from src.python.helpers import _configure_input
from src.python.helpers import estimate_data_means_and_stds, _configure_input
from src.python.settings import plotting_settings, plotting_update

import matplotlib.pyplot as plt
plt.rcParams.update(plotting_update)

import setup
import pickle

def import_config(name):
    try:
        # Dynamically import the module
        module = importlib.import_module(f"config.{name}")
        return module.config
    except ModuleNotFoundError:
        print(f"No module named 'config.{name}'")
        return None
    except AttributeError:
        print(f"Module 'config.{name}' does not have a 'config' attribute")
        return None

def main():
    parser = argparse.ArgumentParser(description='Train a model using the given config file.')
    parser.add_argument('--config', type=str, required=True, help='Name of the module in the config folder')
    args = parser.parse_args()

    config = import_config(args.config)
    if config:
        print("Successfully imported config:")
        pprint(config)
    else:
        print("Failed to import config.")

    RNG = np.random.default_rng(config['rng_seed'])

    if 'datapath' in config.keys():
        DATAPATH = config['datapath']
    else:
        DATAPATH = '../../../climate/sim-data/preproc/'

    datasets = setup.load_datasets(config, data_path=DATAPATH)
    model_names = list(datasets.keys())

    model = setup.build_generative_model(config, datasets, RNG)

    train_data_dict = OrderedDict((key, datasets[key].TAS.isel(member=config['member_split']['train']) - datasets[key].TAS_baseline) for key in datasets.keys())
    data_means, data_stds = estimate_data_means_and_stds(train_data_dict)
    prior_means, prior_stds = model.prior.estimate_means_and_stds()

    context_aware = config['context_aware'] if 'context_aware' in config else True
    configure_input = partial(_configure_input, prior_means=prior_means, prior_stds=prior_stds, data_means=data_means, data_stds=data_stds, context=config['context'], context_aware=context_aware)

    # Test train-val-test split
    for state in ['train', 'val', 'test']:
        sims = model(batch_size=2, sim_args={'state': state, 'verbose': True})

    if 'presimulate_path' in config.keys():

        if os.path.isdir(config['presimulate_path']):
            print(f"Presimulate path {config['presimulate_path']} already exists, skipping presimulation")
            val_sims = pickle.load(open(os.path.join(config['presimulate_path'], 'presim_val.pkl'), 'rb'))
            print(f"Loaded val sims from {config['presimulate_path']}. Be sure to check that the sims are correct!")

        else:
            os.mkdir(config['presimulate_path'])

            states = ['train', 'val', 'test']
            for state in states:
                sims = model(batch_size=5000, sim_args={'state': state})
                pickle.dump(sims, open(os.path.join(config['presimulate_path'], f'presim_{state}.pkl'), 'wb'))
                if state == 'val':
                    val_sims = sims

                print(f"Pregenerated and saved {state} sims to {config['presimulate_path']}")

    else:
        val_sims = 1000

    starttime = time.time()
    amortizer = setup.build_amortizer(config)

    trainer = Trainer(
        amortizer=amortizer, configurator=configure_input, checkpoint_path=config['checkpoint_path'],
        generative_model=model, memory=True, reuse_optimizer=True,
    )
    init_time = time.time()
    print(f"Initialization took {init_time - starttime:.1f} seconds")

    history = trainer.train_online(epochs=config['epochs'], iterations_per_epoch=config['iterations_per_epoch'], batch_size=config['batch_size'], validation_sims=val_sims, val_model_args={'sim_args': {'state':'val'}})

    endtime = time.time()
    print(f"Training took {endtime - starttime:.1f} seconds")
    # Save training time with pickle
    with open(os.path.join(config['checkpoint_path'], 'training_time.pkl'), 'wb') as file:
        pickle.dump(endtime - starttime, file)

    f = diag.plot_losses(history["train_losses"], history["val_losses"])
    f.savefig(os.path.join(config['checkpoint_path'], 'losses.pdf'), bbox_inches='tight')


if __name__ == "__main__":
    main()
