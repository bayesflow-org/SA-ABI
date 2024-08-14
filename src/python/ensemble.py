import os
import sys
import numpy as np
import tensorflow as tf

# Set paths
sys.path.extend(os.path.abspath(os.path.join("../../../BayesFlow_dev/BayesFlow/")))

# Import from relative paths
from src.python.training import setup_network
import bayesflow as bf


def get_ensemble_predictions(
    path,
    data,
    num_models,
    summary_dim,
    predict_in_chunks=False,
    num_chunks=None,
    report_progress=True,
    return_pmps_only=False
):
    """
    Generates predictions from an ensemble of neural network models.

    Parameters:
    -----------
    path              : string
        The path to the directory containing checkpoints of the ensemble models in subfolders.
    data              : dict
        Input dictionary containing at least the `summary_conditions` key:
        `summary_conditions` - the conditioning variables that are first passed through a summary network
        `direct_conditions`  - the conditioning variables that the directly passed to the inference network
    summary_dim       : int
        The number of learned summary statistics.
    predict_in_chunks : bool
        Whether to predict in chunks (use when many data sets are to be processed).
    num_chunks        : int
        The number of data chunks to split predictions into for memory efficiency.
        Only has an effect if ``predict_in_chunks=True``.
    report_progress : bool
        Whether to report when a network is finished.
    return_pmps_only : bool
        Whether to return only pmps or also embeddings and logits.

    Returns:
    --------
    embeddings      : numpy.ndarray of shape (num_ensemble_members, num_data_sets, num_summary_dim)
        Summary network embeddings for each ensemble member and data set.
    pmps            : numpy.ndarray of shape (num_ensemble_members, num_data_sets, num_models)
        Predicted posterior model probabilities for each ensemble member and data set.
    logits          : numpy.ndarray of shape (num_ensemble_members, num_data_sets, num_models)
        Predicted logits for each ensemble member and data set.
    """
    # Get a list of subfolders in the specified directory
    checkpoint_folders = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    sorted_checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x[3:]))  # Sort from net1 to net[num_ensemble_members]
    num_ensemble_members = len(sorted_checkpoint_folders)

    # Extract constants
    num_ensemble_members = len(sorted_checkpoint_folders)
    num_data_sets = data["summary_conditions"].shape[0]

    # Initialize arrays to store results
    embeddings = np.zeros((num_ensemble_members, num_data_sets, summary_dim))
    pmps = np.zeros((num_ensemble_members, num_data_sets, num_models))
    logits = np.zeros((num_ensemble_members, num_data_sets, num_models))

    for i, network in enumerate(sorted_checkpoint_folders):
        # Load network
        tf.keras.backend.clear_session()
        summary_net, probability_net, amortizer = setup_network()
        checkpoint_path = os.path.join(path, network, "finetune")  # Finetune is specific to this setup
        _ = bf.trainers.Trainer(
            amortizer=amortizer,
            checkpoint_path=checkpoint_path
        )

        # Get predictions in chunks or as a whole
        if predict_in_chunks:
            split_summary_conditions = tf.split(data["summary_conditions"], num_chunks)
            net_embeddings = [summary_net(x_chunk) for x_chunk in split_summary_conditions]
            del split_summary_conditions  # Free up memory
            if data["direct_conditions"] is not None:
                split_direct_conditions = tf.split(data["direct_conditions"], num_chunks)
                inference_net_inputs = [
                    np.concatenate((emb, cond), axis=1) for emb, cond in zip(net_embeddings, split_direct_conditions)
                ]
                del split_direct_conditions  # Free up memory
            else:
                inference_net_inputs = net_embeddings

            embeddings[i, ...] = np.concatenate(net_embeddings)
            pmps[i, ...] = np.concatenate([probability_net.posterior_probs(input) for input in inference_net_inputs])
            logits[i, ...] = np.concatenate([probability_net.logits(input) for input in inference_net_inputs])

        else:
            net_embeddings = summary_net(data["summary_conditions"])
            if data["direct_conditions"] is not None:
                inference_net_inputs = np.concatenate([net_embeddings, data["direct_conditions"]], axis=-1)
            else:
                inference_net_inputs = net_embeddings
            embeddings[i, ...] = net_embeddings
            pmps[i, ...] = probability_net.posterior_probs(inference_net_inputs)
            logits[i, ...] = probability_net.logits(inference_net_inputs)

        if report_progress:
            print(f"{network} finished!")

    if return_pmps_only:
        return pmps
    return embeddings, pmps, logits


def get_pmps_under_scaling(factors, data, num_ensemble_members, path):
    """
    Generates predictions from an ensemble of neural network models under a range of powerscaling settings for two
    scaling factors.

    Parameters:
    -----------
    factors : numpy.ndarray of shape(num_factors)
        The factors that should be applied for the power scaling, independently for both components.
    data : numpy.ndarray of 4D shape
        The data used as summary_conditions.
    num_ensemble_members : int
        The number of members in the neural network ensemble.
    path : str
        The path to the directory containing the trained ensemble members.

    Returns:
    --------
    results_under_scaling : numpy.ndarray of shape
    (num_data_sets, len(factors) * len(factors) * num_ensemble_members, num_models + 3)
    or (len(factors) * len(factors) * num_ensemble_members, 7) if num_data_sets=1.
        An array containing predictions generated under different powerscaling settings.
        Each row of the array represents a combination of scaling factors and ensemble member predictions:
        - Column 0: Power-scaling factor of the location hyperparameter
        - Column 1: Power-scaling factor of the Scale hyperparameter
        - Column 2: Ensemble member number
        - Column 3: Predicted value for model 1
        - Column 4: Predicted value for model 2
        - Column 5: Predicted value for model 3
        - Column 6: Predicted value for model 4
    """
    num_scaling_settings = len(factors) * len(factors) * num_ensemble_members
    num_data_sets = data.shape[0]
    results_under_scaling = np.zeros((num_data_sets, num_scaling_settings, 7))

    # Get number of ensemble members
    checkpoint_folders = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    sorted_checkpoint_folders = sorted(checkpoint_folders, key=lambda x: int(x[3:]))  # Sort from net1 to net[num_ensemble_members]
    num_ensemble_members = len(sorted_checkpoint_folders)

    row_index = 0

    for i, network in enumerate(sorted_checkpoint_folders):
        # Load network
        tf.keras.backend.clear_session()
        summary_net, probability_net, amortizer = setup_network()
        checkpoint_path = os.path.join(path, network, "finetune")  # Finetune is specific to this setup
        _ = bf.trainers.Trainer(
            amortizer=amortizer,
            checkpoint_path=checkpoint_path
        )

        for f1 in factors:
            for f2 in factors:
                scaling_factors = np.array([f1, f2], dtype=np.float32)[np.newaxis, :]
                scaling_factors = np.repeat(scaling_factors, num_data_sets, axis=0)  # Handles num_data_sets > 1 cases
                data_dict = {"summary_conditions": data, "direct_conditions": scaling_factors}

                net_embeddings = summary_net(data_dict["summary_conditions"])
                inference_net_inputs = np.concatenate([net_embeddings, data_dict["direct_conditions"]], axis=-1)
                pmps = probability_net.posterior_probs(inference_net_inputs)

                results_under_scaling[:, row_index, 0] = f1
                results_under_scaling[:, row_index, 1] = f2
                results_under_scaling[:, row_index, 2] = i
                results_under_scaling[:, row_index, 3:7] = pmps

                row_index += 1

    if num_data_sets == 1:  # Simplify output if only one data set was processed
        results_under_scaling = results_under_scaling[0, ...]

    return results_under_scaling


def get_pmps_under_scaling_bootstrapping(factors, empirical_data, num_ensemble_members, level, num_bootstrap):
    """Get predictions on the bootstrapped data sets with power scaling.

    Parameters
    ----------
    factors : numpy.ndarray of shape(num_factors)
        The factors that should be applied for the power scaling, independently for both components.
    empirical_data : numpy.ndarray of 4D shape
        The empirical data used as summary_conditions.
    num_ensemble_members : int
        The number of members in the ensemble of models.
    level : string
        Indicating the level to bootstrap; either 'participants' or 'trials'.
    num_bootstrap : int
        Number of bootstrap repetitions.

    Returns
    -------
    results_under_scaling_bootstrapping : np.array of shape
    (num_bootstrap, len(factors) * len(factors) * num_ensemble_members, num_models + 3)
        Predictions on the bootstrapped data sets with power scaling.
        See get_pmps_under_scaling docstring for details.
    """

    if level == "participants":
        n = empirical_data.shape[1]
    elif level == "trials":
        n = empirical_data.shape[2]

    bootstrapped_data = np.zeros((
        num_bootstrap,
        empirical_data.shape[1],
        empirical_data.shape[2],
        empirical_data.shape[3]
    ))

    for b in range(num_bootstrap):
        b_idx = np.random.choice(np.arange(n), size=n, replace=True)
        if level == "participants":
            bootstrapped_data[b, ...] = empirical_data[:, b_idx, :, :]
        elif level == "trials":
            bootstrapped_data[b, ...] = empirical_data[:, :, b_idx, :]

    results_under_scaling_bootstrapping = get_pmps_under_scaling(
        factors=factors,
        data=bootstrapped_data,
        num_ensemble_members=num_ensemble_members,
        path="ensemble_checkpoints/sim_powerscaled"
    )

    return results_under_scaling_bootstrapping
