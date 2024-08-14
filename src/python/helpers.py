import os, sys
sys.path.append(os.path.abspath(os.path.join('../../../BayesFlow_dev/BayesFlow/')))
sys.path.append(os.path.abspath(os.path.join("../../../BayesFlow/")))
import bayesflow as bf
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import copy
import warnings
from scipy.stats import truncnorm


# General helpers
def get_latex_results_table(input_df, precision=2, print_table=True):
    """
    Prints or returns a LaTeX table containing the input DataFrame and summary statistics.

    Parameters
    ----------
    input_df : pandas.DataFrame
        The input DataFrame to be summarized.
    precision : int, optional
        The number of decimal places to include in the table. Default is 2.
    print_table : bool, optional
        If True, the function will print the LaTeX table. If False, it will return the LaTeX table as a string.
        Default is True.

    Returns
    -------
    str or None
        A string containing the LaTeX code for the formatted table, if print_table is False.
        If print_table is True, returns None.
    """
    # Calculate summary metrics
    means = input_df.mean()
    std_devs = input_df.std()

    # Create a new row for each metric
    summary_row_mean = pd.DataFrame(means).T
    summary_row_mean.index = ['Average']
    summary_row_std = pd.DataFrame(std_devs).T
    summary_row_std.index = ['SD']

    # Concatenate the input DataFrame with the summary rows
    full_df = pd.concat([input_df, summary_row_mean, summary_row_std])

    # Apply the formatting function to the DataFrame
    formatted_table = full_df.style.format(precision=precision)

    # Convert the formatted DataFrame to LaTeX
    latex_table = formatted_table.to_latex(position="h", position_float="centering", hrules=True)

    # Print or return the LaTeX table
    if print_table:
        print(latex_table)
    else:
        return latex_table


# Levy comparison: Load and transform data
class MaskingConfigurator:

    def __init__(self, power_scaling):
        """
        Initializes a MaskingConfigurator instance using the BayesFlow default DefaultModelComparisonConfigurator.
        Includes a custom combiner for transforming context data.
        """
        self.power_scaling = power_scaling
        self.default_config = bf.configuration.DefaultModelComparisonConfigurator(
            num_models=4,
            combiner=self.custom_combiner
        )

    def custom_combiner(self, forward_dict):
        """ Replaces BayesFlow's DefaultCombiner in order to transform simulated context to direct_conditions."""
        if self.power_scaling:
            out_dict = {
                "summary_conditions": forward_dict["sim_data"],
                "direct_conditions": np.array(forward_dict["prior_batchable_context"], dtype=np.float32),
            }
        else:
            out_dict = {
                "summary_conditions": forward_dict["sim_data"],
            }
        return out_dict

    def __call__(self, forward_dict):
        """ Masks the simulated data with missing values and configures it for network training. """
        masked_dict = mask_inputs(forward_dict, missings_mean=28.5, missings_sd=13.5)
        config = self.default_config(masked_dict)
        return config


def load_empirical_rt_data(load_dir):
    """
    Reads single subject datasets from a folder and transforms into list of 4D-arrays
    which allows for a variable number of observations between participants.
    Assumes data files have a three-column csv format (Condition, Response, Response Time).
    ----------

    Arguments:
    load_dir : str -- a string indicating the directory from which to load the data
    --------

    Returns:
    X: list of length (n_clusters), containing tf.Tensors of shape (1, 1, n_obs, 3)
        -- variable order now (Condition, Response Time, Response)
    """

    data_files = os.listdir(load_dir)
    X = []

    # Loop through data files and estimate
    for data_file in data_files:
        ### Read in and transform data
        data = pd.read_csv(os.path.join(load_dir, data_file), header=None, sep=" ")
        data = data[[0, 2, 1]].values  # reorder columns
        X_file = tf.convert_to_tensor(data, dtype=tf.float32)[np.newaxis][
            np.newaxis
        ]  # get 4D tensor
        X.append(X_file)

    return X


def mask_inputs(
    simulations_dict,
    missings_mean,
    missings_sd,
    missing_value=-1,
    missing_rts_equal_mean=True,
    tolerance=3,
    insert_additional_missings=False,
):
    """Masks some training inputs so that training leads to a robust net that can handle missing data

    Parameters
    ----------
    simulations_dict : dict
        simulated training data sets in the BayesFlow format
    missings_mean : float
        empirical mean of missings per person
    missings_sd : float
        empirical sd of missings per person
    missing_value : int or float
        value to be used as a replacement for masked data
    missing_rts_equal_mean : bool
        indicates whether missing reaction time data should be imputed with the person mean instead of missing_value
    tolerance : int
        Maximum deviation between average amount of masks per person and missings_mean that is tolerated
    insert_additional_missings : bool
        indicates whether additional missings should be inserted, which results in disabling the faithfulness check

    Returns
    -------
    data_sets : dict
        simulated training data sets with masked inputs
    """

    masked_dict = copy.deepcopy(simulations_dict)
    n_models = len(masked_dict['model_outputs'])
    n_persons = masked_dict['model_outputs'][0]['sim_data'].shape[1]
    n_trials = masked_dict['model_outputs'][0]['sim_data'].shape[2]

    # Create truncated normal parameterization in accordance with scipy documentation
    a, b = (0 - missings_mean) / missings_sd, (n_trials - missings_mean) / missings_sd

    total_masks_per_person = []

    for m in range(n_models):
        n_data_sets_per_model = masked_dict['model_outputs'][m]['sim_data'].shape[0]
        for d in range(n_data_sets_per_model):
            # Draw number of masked values per person from truncated normal distribution
            masks_per_person = (
                truncnorm.rvs(a, b, loc=missings_mean, scale=missings_sd, size=n_persons)
                .round()
                .astype(int)
            )
            total_masks_per_person.append(masks_per_person)
            # Assign the specific trials to be masked within each person
            mask_positions = [
                np.random.choice(n_trials, size=j, replace=False)
                for j in masks_per_person
            ]
            # Iterate over each person and mask their trials according to mask_positions
            for j in range(n_persons):
                if missing_rts_equal_mean: # set rts to person mean and decisions to missing_value
                    masked_dict['model_outputs'][m]['sim_data'][d, j, :, 2][mask_positions[j]] = missing_value
                    masked_dict['model_outputs'][m]['sim_data'][d, j, :, 1][mask_positions[j]] = np.mean(
                        masked_dict['model_outputs'][m]['sim_data'][d, j, :, 1]
                    )
                else:  # set rts and decisions to missing_value
                    masked_dict['model_outputs'][m]['sim_data'][d, j, :, 1:3][mask_positions[j]] = missing_value

    # Calculate the average amount of masks per person across all models and data sets
    avg_masks_per_person = np.mean(total_masks_per_person)

    # Check if the deviation between average amount of masks and missings_mean is greater than the specified tolerance
    if insert_additional_missings == False:
        deviation = abs(avg_masks_per_person - missings_mean)
        if deviation > tolerance:
            warnings.warn(f"Average amount of masks per person deviates by {deviation} from missings_mean!")

    return masked_dict


def join_and_fill_missings(
    color_data, lexical_data, n_trials, missings_value=-1, missing_rts_equal_mean=True
):
    """Joins data from color discrimination and lexical decision task per person and fills missings

    Parameters
    ----------
    color_data : tf.Tensor
    lexical_data : tf.Tensor
    n_trials : int
    missings_value : float
        specifies the value that codes missings
    missing_rts_equal_mean : boolean
        specifies whether missing rt should be coded with missings_value or imputed with the participant mean

    Returns
    -------
    empirical_data : list of tf.Tensors
    """

    n_clusters = len(color_data)
    n_trials_per_cond = int(n_trials / 2)
    empirical_data = []

    for j in range(n_clusters):
        # Join data
        joint_data = tf.concat([color_data[j], lexical_data[j]], axis=2).numpy()
        # Extract information about trial
        n_trials_obs = joint_data.shape[2]
        n_missings = n_trials - n_trials_obs
        n_condition_1 = int(joint_data[0, 0, :, 0].sum())
        mean_rt = np.mean(joint_data[0, 0, :, 1])
        # replace all missings with missings_value
        npad = ((0, 0), (0, 0), (0, n_missings), (0, 0))
        joint_data = np.pad(
            joint_data, npad, "constant", constant_values=missings_value
        )
        # replace missing condition indices
        cond_indices = np.array(
            [0] * (n_trials_per_cond - (n_trials_obs - n_condition_1))
            + [1] * (n_trials_per_cond - n_condition_1)
        )
        np.random.shuffle(cond_indices)
        joint_data[0, 0, -(n_missings):, 0] = cond_indices
        # replace missing rts with mean rt
        if missing_rts_equal_mean:
            joint_data[0, 0, :, 1] = np.select(
                [joint_data[0, 0, :, 1] == missings_value],
                [mean_rt],
                joint_data[0, 0, :, 1],
            )
        # Append
        empirical_data.append(joint_data)

    # Transform to np.array
    empirical_data = np.reshape(
        np.asarray(empirical_data), (1, n_clusters, n_trials, 3)
    )

    # Assert that the number of coded missings equals the real number of missings
    deviation = abs(
        (
            (empirical_data == missings_value).sum()
            / (n_clusters * (2 - missing_rts_equal_mean))
        )
        - 28.5
    )
    assert deviation < 1, "number of filled and existing missings does not match"

    return empirical_data


# Climate application: preprocessing of gridded temperature data

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell

    Parameters
    ----------
    lat : np.array
        vector of latitude in degrees
    lon : np.array
        vector of longitude in degrees

    Returns
    -------
    area : xr.DataArray
        grid-cell area in square-meters with dimensions, (lat,lon)
        Area is in square meters

    Notes
    -------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """


    xlon, ylat = np.meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = np.deg2rad(np.gradient(ylat, axis=0))
    dlon = np.deg2rad(np.gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * np.cos(np.deg2rad(ylat))

    area = xr.DataArray(
        dy * dx,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return area

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84

    Parameters
    ----------
    lat : np.array
        vector or latitudes in degrees

    Returns
    -------
    r : np.array
        vector of radius in meters

    Notes
    -------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = np.deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5)
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5
        )

    return r

# Climate application: helper functions for streamlining formulation of the generative model

def find_parameter_spans(datasets, threshold=1.5):
    """
    Find the maximum shared parameter span for which all datasets have data.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.
    threshold : float
        Critical warming threshold, one of [1.1, 1.5, 2.0, 3.0].

    Returns
    -------
    lower : float
        Lower bound of shared parameter span.
    higher : float
        Upper bound of shared parameter span.
    """
    lower = np.array([(datasets[key][f"pass_{threshold*10:.0f}"] - datasets[key].year).min() for key in datasets.keys()]).max()
    higher = np.array([(datasets[key][f"pass_{threshold*10:.0f}"] - datasets[key].year).max() for key in datasets.keys()]).min()
    return lower, higher

def get_var_from_time_to_threshold(ds, var, threshold, time):
    """
    Get a variable from a dataset at a given time to a given threshold.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable.
    var : str
        Variable name, like 'TAS' or 'TAS_global', that varies over time.
    threshold : float
        Critical warming threshold, one of [1.1, 1.5, 2.0, 3.0].

    Returns
    -------
    match : xr.DataArray
        Variable at the given time to the given threshold.
    """

    time = int(time+.5)  # round to nearest integer
    assert threshold in [1.1, 1.5, 2.0, 3.0]
    assert var in ds
    tidx = (ds[f'pass_{int(threshold*10)}'] - ds.year == time).idxmax(dim='year')
    match = ds[var].loc[{'year': tidx}]

    return match

'''def get_time_to_threshold(ds, threshold, var='TAS_global'):
    """
    Get the time to a given threshold for a given variable.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable.
    threshold : float
        Critical warming threshold, one of [1.1, 1.5, 2.0, 3.0].

    Returns
    -------
    time : int
        Time to the given threshold.
    """

    assert threshold in [1.1, 1.5, 2.0, 3.0]
    time = (ds[f'pass_{int(threshold*10)}'] - ds.year).values

    return time

'''

def estimate_data_means_and_stds(data_dict):
    """
    Estimate the means and standard deviations of the data for standardization.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing xarray.DataArrays.
        For construction see `preprocess-all-sims.ipynb`.

    Returns
    -------
    mean : np.array
        Means of the data for standardization.
    std : np.array
        Standard deviations of the data for standardization.
    """

    means = np.mean(
        [data_dict[key].mean(dim=('ensemble', 'member', 'year')) for key in data_dict.keys()],
        axis=0)
    stds = np.sqrt(np.mean(
        [((data_dict[key] - means)**2).mean(dim=('ensemble', 'member', 'year')).values for key in data_dict.keys()],
        axis=0))

    return means, stds

def _configure_input(forward_dict, prior_means, prior_stds, data_means=None, data_stds=None, context='ssp&model', context_aware=True):
    """
    Function to configure the simulated quantities (i.e., simulator outputs)
    into a neural network-friendly (BayesFlow) format.

    Parameters
    ----------
    forward_dict : dict
        Dictionary of simulated quantities.
    prior_means : np.array
        Means of the prior distributions for standardization.
    prior_stds : np.array
        Standard deviations of the prior distributions for standardization.
    data_means : np.array
        Means of the data for standardization.
        If None, the data is not standardized.
    data_stds : np.array
        Standard deviations of the data for standardization.
        If None, the data is not standardized.

    Returns
    -------
    out_dict : dict
        Dictionary of simulated quantities in a neural network-friendly format.
    """

    # Prepare placeholder dict
    out_dict = {}

    # Enforce np.float16 on one hot encoding
    models = np.array(forward_dict["sim_batchable_context"]).astype(np.float16)
    if context == 'ssp&model':
        direct_conditions = models
    elif context == 'ssp&model&prior':
        priors = np.array(forward_dict["prior_batchable_context"]).astype(np.float16)
        direct_conditions = np.concatenate([models, priors[:,None]], axis=1)

    # Extract prior draws and z-standardize with previously computed means
    params = forward_dict["prior_draws"].astype(np.float32)
    params = (params - prior_means) / prior_stds

    # Extract data and z-standardize with previously computed means if available
    sim_data = forward_dict["sim_data"].astype(np.float32)
    if data_means is not None and data_stds is not None:
        sim_data = (sim_data - data_means) / data_stds

    # Remove a batch if it contains nan, inf or -inf
    idx_keep = np.all(np.isfinite(sim_data), axis=(1, 2))
    if not np.all(idx_keep):
        print("Invalid value encountered...removing from batch")

    # Add to keys
    out_dict["summary_conditions"] = sim_data[idx_keep]
    if context_aware:
        out_dict["direct_conditions"] = direct_conditions[idx_keep]
    out_dict["parameters"] = params[idx_keep]

    return out_dict


# Climate application: hand crafted summary statistics

def tiny_summary_func_tf(dat):
    mean = tf.math.reduce_mean(dat, axis=(1,2))
    std = tf.math.reduce_std(dat, axis=(1,2))
    point1 = dat[:,60,60]
    summary = tf.stack([mean, std, point1], axis=1)
    return summary

class TinySummaryNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return tiny_summary_func_tf(x)

# Climate application: learned summary statistics

class DenseSummaryNet(tf.keras.Model):
    """
    Dense network for learning summary statistics.
    Architecture similar to heteroscedastic regressor in [1].

    Input layer: (batch_size, latitude, longitude)
    Output layer: (batch_size, output_dim)

    [1] Diffenbaugh, Noah S., and Elizabeth A. Barnes. “Data-Driven Predictions of the Time Remaining until Critical Global Warming Thresholds Are Reached.”
        Proceedings of the National Academy of Sciences 120, no. 6 (February 7, 2023): e2207183120. https://doi.org/10.1073/pnas.2207183120.

    """
    def __init__(self, hidden_units, output_dim, zeroth_layer=None, *args, **kwargs):
        """
        Parameters
        ----------
        hidden_units : array-like
            Number of hidden units per layer.
        output_dim : int
            Dimensionality of the output layer.
        """
        super().__init__(*args, **kwargs)

        self.hidden_units = hidden_units
        self.output_dim = output_dim

        self.flatten = tf.keras.layers.Flatten()
        self.dense_layers = []
        regularizer = None
        if zeroth_layer:
            if isinstance(zeroth_layer, tf.keras.layers.Layer):
                self.dense_layers.append(zeroth_layer)
            elif isinstance(zeroth_layer, dict):
                if 'dropout' in zeroth_layer.keys():
                    self.dense_layers.append(tf.keras.layers.Dropout(zeroth_layer['dropout']))
                elif 'l2' in zeroth_layer.keys():
                    regularizer = tf.keras.regularizers.l2(zeroth_layer['l2'])
                else:
                    raise KeyError("zeroth_layer dict must have key 'dropout'/'l2'.")
            else:
                raise TypeError("zeroth_layer must be an instance of tf.keras.layers.Layer or dict.")

        self.dense_layers = self.dense_layers + [
            tf.keras.layers.Dense(units, activation='relu') for units in hidden_units
        ]
        self.dense_layers[0].kernel_regularizer = regularizer   # only regularize first layer

        self.dense_layers.append(tf.keras.layers.Dense(output_dim))

    def call(self, x):
        x = self.flatten(x)
        for layer in self.dense_layers:
            x = layer(x)

        return x


# Convert ssp&model lowercase shortform to uppercase longform

def format_names(full_name):
    """
    Parameters
    ----------
    full_name : str
        String of format like 'ssp370_CESM2'

    Returns
    -------
    formatted_name : str
        String of format 'SSP3-7.0 CESM2'
    """
    ssp, model_name = full_name.split('_')
    ssp = ssp[3:]
    return f'SSP{ssp[0]}-{float(ssp[1:])/10:.1f} {model_name}'


