import os
import sys
import pickle
from functools import partial
from src.python.settings import summary_meta_diffusion, probability_meta_diffusion

# Set paths
sys.path.append(os.path.abspath(os.path.join('../../../BayesFlow_dev/BayesFlow/')))
import bayesflow as bf


# Levy comparison
def load_training_data(scaling_name, goal):
    """ Loads data for the training process according to power scaling setting and training goal. """
    data_path = os.path.abspath(f"../../data/levy_comparison/{scaling_name}/{goal}.pkl")
    with open(data_path, "rb") as file:
        return pickle.load(file)


def setup_network():
    """ Sets up a hierarchical model comparison network with standardized settings. """
    summary_net = bf.summary_networks.HierarchicalNetwork([
        bf.networks.DeepSet(
            dense_s1_args=summary_meta_diffusion['level_1']['dense_s1_args'],
            dense_s2_args=summary_meta_diffusion['level_1']['dense_s2_args'],
            dense_s3_args=summary_meta_diffusion['level_1']['dense_s3_args']
        ),
        bf.networks.DeepSet(
            summary_dim=64,
            dense_s1_args=summary_meta_diffusion['level_2']['dense_s1_args'],
            dense_s2_args=summary_meta_diffusion['level_2']['dense_s2_args'],
            dense_s3_args=summary_meta_diffusion['level_2']['dense_s3_args']
        )])

    probability_net = bf.inference_networks.PMPNetwork(
        num_models=4,
        dense_args=probability_meta_diffusion['dense_args'],
        dropout=True,
        dropout_prob=0.1
    )

    amortizer = bf.amortizers.AmortizedModelComparison(
        probability_net,
        summary_net,
        loss_fun=partial(bf.losses.log_loss, label_smoothing=None)
    )

    return summary_net, probability_net, amortizer
