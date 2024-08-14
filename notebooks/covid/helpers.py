import os
import sys
import numpy as np
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.extend([os.path.abspath(os.path.join("../../../BayesFlow_dev/BayesFlow/"))])
import bayesflow as bf
from bayesflow.computational_utilities import simultaneous_ecdf_bands


def get_metrics(parameters, draws, agg_fun_draws=np.median, agg_fun=np.median):
    """
    Get various approximation performance metrics for a given set of parameters and posterior draws.
    Intermediary output shapes vary by metric, but if aggregation is necessary, agg_fun is applied.

    Parameters:
    ----------
    parameters : np.array of shape (n_data_sets, n_parameters)
        The true parameter values (standardized by net).
    draws : np.array of shape (n_data_sets, n_draws, n_parameters)
        The posterior draws (standardized by net).
    agg_fun_draws : function, optional
        The aggregation function to apply over the posterior draws to get a point estimate per data set and parameter. 
        Default is np.median.
    agg_fun : function, optional
        The aggregation function to reduce the computed metrics to a single number if necessary. Default is np.median.

    Returns:
    -------
    rmse : float
        The root mean squared error.
    mae : float
        The mean absolute error.
    r2 : float
        The R-squared coefficient.
    corr : float
        The correlation coefficient.
    ece : float
        The expected calibration error.
    mmd : float
        The maximum mean discrepancy.
    post_contraction : float
        The posterior contraction.
    """

    def root_mean_squared_error(x_true, x_pred):
        """ Gets the RMSE between true parameters and posterior draws for each test data set, draw and parameter. """
        return np.sqrt(np.mean(np.square(x_true[:, np.newaxis, :] - x_pred)))

    def mean_absolute_error(x_true, x_pred):
        """ Gets the MAE between true parameters and posterior draws for each test data set, draw and parameter. """
        return np.mean(np.abs((x_true[:, np.newaxis, :] - x_pred)))

    def mmd_over_datasets(x_true, x_pred):
        """ Gets the MMD between true parameters and posterior draws for each test data set. """
        mmd_array = np.empty(draws.shape[0], dtype=np.float32)
        for i in range(x_pred.shape[0]):
            mmd_array[i] = bf.computational_utilities.maximum_mean_discrepancy(
                x_true.astype('float32')[i, :][np.newaxis, :], # align shapes to (1, n_parameters)
                x_pred[i, :, :], # align shapes to (num_draws, n_parameters)
            )
        return mmd_array
    
    def post_cont(x_true, x_pred):
        """ Gets the posterior contraction between true parameters and posterior draws for each test data set and parameter. """
        post_vars = x_pred.var(axis=1, ddof=1)
        prior_vars = x_true.var(axis=0, keepdims=True, ddof=1)
        #prior_vars = np.repeat(1.0, repeats=x_pred.shape[-1]) # take an array of 1 as params are standardized        
        return 1 - (post_vars / prior_vars)

    agg_draws = agg_fun_draws(draws, axis=1) # point estimates for r2 and correlation

    rmse = root_mean_squared_error(parameters, draws) 
    mae = mean_absolute_error(parameters, draws) 
    r2 = r2_score(parameters, agg_draws)
    corr = agg_fun([np.corrcoef(parameters[:, i], np.median(draws, axis=1)[:, i])[0, 1] for i in range(parameters.shape[-1])])
    ece = agg_fun(
        bf.computational_utilities.posterior_calibration_error(prior_samples=parameters, posterior_samples=draws)
    ) # shape before aggregation: (num_params)
    #mmd = agg_fun(mmd_over_datasets(parameters, draws)) # shape before aggregation: (num_datasets)
    post_contraction = agg_fun(post_cont(parameters, draws)) # shape before aggregation: (num_datasets, num_params)

    #return rmse, mae, r2, corr, ece, mmd, post_contraction
    return rmse, mae, r2, corr, ece, post_contraction


def load_standardization_params(STANDARDIZATION_PATH, network_name):
    """Helper function to load standardization parameters for a given network."""
    with open(STANDARDIZATION_PATH, "rb") as f:
        standardization_params = pickle.load(f)
    return standardization_params[network_name]


def get_avg_time(times, prefix):
    """
    Helper function to get the average time for a given prefix in a times dictionary.

    Parameters
    ----------
    times : dict
        A dictionary where keys are string prefixes and values are time measurements.
    prefix : str
        The prefix to filter the keys in the times dictionary.

    Returns
    -------
    float
        The average of the values in the times dictionary that have keys starting with the given prefix.
    """
    return np.mean([value for key, value in times.items() if key.startswith(prefix)])


def get_total_time(time_setting, training_time, inference_time, scaling):
    """
    Helper function to calculate the total training + inference time for a given time and scaling setting.

    Parameters
    ----------
    time_setting : int
        The time setting factor to scale the training (if not powerscaled) and inference times.
    training_time : float
        The average training time.
    inference_time : float
        The average inference time.
    scaling : str
        The scaling type. If 'unscaled', the training time is multiplied by the time setting.

    Returns
    -------
    float
        The total time (in minutes) for training and inference, scaled by the time setting.
    """
    if scaling == 'unscaled':  # Training must only be repeated for unscaled networks
        training_time = training_time * time_setting
    inference_time = inference_time * time_setting
    total_time = (training_time + inference_time) / 60
    return total_time


def calculate_setting_times(time_settings, training_times, inference_times, scaling):
    """
    Calculate the total training and inference times for a range of time settings.

    Parameters
    ----------
    time_settings : list
        A list of time settings to calculate the total times for.
    training_times : dict
        A dictionary where keys are string prefixes and values are training time measurements.
    inference_times : dict
        A dictionary where keys are string prefixes and values are inference time measurements.
    scaling : str
        The scaling type. Used to filter the keys in the training_times and inference_times dictionaries.

    Returns
    -------
    dict
        A dictionary where keys are string representations of the time settings and values are the total times for
        training and inference.
    """
    training_avg = get_avg_time(training_times, scaling)
    inference_avg = get_avg_time(inference_times, scaling)
    time_data = {}
    for time_setting in time_settings:
        time_data[str(time_setting)] = get_total_time(time_setting, training_avg, inference_avg, scaling)
    return time_data


def custom_plot_sbc_ecdf(
    post_samples,
    prior_samples,
    ax,
    add_bounds=True,
    difference=False,
    stacked=False,
    fig_size=None,
    param_names=None,
    label_fontsize=16,
    legend_fontsize=14,
    title_fontsize=18,
    tick_fontsize=12,
    rank_ecdf_color="#a34f4f",
    fill_color="grey",
    **kwargs,
):
    """Creates the empirical CDFs for each marginal rank distribution and plots it against
    a uniform ECDF. ECDF simultaneous bands are drawn using simulations from the uniform,
    as proposed by [1].
    For models with many parameters, use `stacked=True` to obtain an idea of the overall calibration
    of a posterior approximator.
    [1] Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022). Graphical test for discrete uniformity and
    its applications in goodness-of-fit evaluation and multiple sample comparison. Statistics and Computing,
    32(2), 1-21. https://arxiv.org/abs/2103.10522
    """

    # Store reference to number of parameters
    n_params = post_samples.shape[-1]

    # Compute fractional ranks (using broadcasting)
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1) / post_samples.shape[1]

    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(post_samples.shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))

    # Difference, if specified
    if difference:
        L -= z
        H -= z
        ylab = "ECDF difference"
    else:
        ylab = "ECDF"
        
    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        if stacked:
            if j == 0: 
                ax.plot(xx, yy, color=rank_ecdf_color[j], alpha=0.95, linewidth=2, label="Rank ECDFs")
            else:
                ax.plot(xx, yy, color=rank_ecdf_color[j], alpha=0.95, linewidth=2)
        else:
            ax.flat[j].plot(xx, yy, color=rank_ecdf_color, alpha=0.95 ,linewidth=2, label="Rank ECDF")

    if add_bounds:
        # Add simultaneous bounds
        if stacked:
            titles = [None]
            axes = [ax]
        else:
            axes = ax.flat
            if param_names is None:
                titles = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
            else:
                titles = param_names

        for _ax, title in zip(axes, titles):
            _ax.fill_between(z, L, H, color=fill_color, alpha=0.2,
                             label=rf"{int((1-alpha) * 100)}$\%$ Confidence Bands")


# Custom to deactivate legend
def custom_plot_mmd_hypothesis_test(
    mmd_null,
    mmd_observed=None,
    alpha_level=0.05,
    null_color=(0.16407, 0.020171, 0.577478),
    observed_color="red",
    alpha_color="orange",
    truncate_vlines_at_kde=False,
    xmin=None,
    xmax=None,
    bw_factor=1.5,
):
    """

    Parameters
    ----------
    mmd_null       : np.ndarray
        The samples from the MMD sampling distribution under the null hypothesis "the model is well-specified"
    mmd_observed   : float
        The observed MMD value
    alpha_level    : float
        The rejection probability (type I error)
    null_color     : str or tuple
        The color of the H0 sampling distribution
    observed_color : str or tuple
        The color of the observed MMD
    alpha_color    : str or tuple
        The color of the rejection area
    truncate_vlines_at_kde: bool
        true: cut off the vlines at the kde
        false: continue kde lines across the plot
    xmin           : float
        The lower x-axis limit
    xmax           : float
        The upper x-axis limit
    bw_factor      : float, optional, default: 1.5
        bandwidth (aka. smoothing parameter) of the kernel density estimate

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    def draw_vline_to_kde(x, kde_object, color, label=None, **kwargs):
        kde_x, kde_y = kde_object.lines[0].get_data()
        idx = np.argmin(np.abs(kde_x - x))
        plt.vlines(x=x, ymin=0, ymax=kde_y[idx], color=color, linewidth=3, label=label, **kwargs)

    def fill_area_under_kde(kde_object, x_start, x_end=None, **kwargs):
        kde_x, kde_y = kde_object.lines[0].get_data()
        if x_end is not None:
            plt.fill_between(kde_x, kde_y, where=(kde_x >= x_start) & (kde_x <= x_end), interpolate=True, **kwargs)
        else:
            plt.fill_between(kde_x, kde_y, where=(kde_x >= x_start), interpolate=True, **kwargs)

    f = plt.figure(figsize=(8, 4))

    kde = sns.kdeplot(mmd_null, fill=False, linewidth=0, bw_adjust=bw_factor)
    sns.kdeplot(mmd_null, fill=True, alpha=0.12, color=null_color, bw_adjust=bw_factor)

    if truncate_vlines_at_kde:
        draw_vline_to_kde(x=mmd_observed, kde_object=kde, color=observed_color, label=r"Observed data")
    else:
        plt.vlines(
            x=mmd_observed,
            ymin=0,
            ymax=plt.gca().get_ylim()[1],
            color=observed_color,
            linewidth=3,
            label=r"Observed data",
        )

    mmd_critical = np.quantile(mmd_null, 1 - alpha_level)
    fill_area_under_kde(
        kde, mmd_critical, color=alpha_color, alpha=0.5, label=rf"{int(alpha_level*100)}% rejection area"
    )

    if truncate_vlines_at_kde:
        draw_vline_to_kde(x=mmd_critical, kde_object=kde, color=alpha_color)
    else:
        plt.vlines(x=mmd_critical, color=alpha_color, linewidth=3, ymin=0, ymax=plt.gca().get_ylim()[1])

    sns.kdeplot(mmd_null, fill=False, linewidth=3, color=null_color, label=r"$H_0$", bw_adjust=bw_factor)

    plt.xlabel(r"MMD", fontsize=20)
    plt.ylabel("")
    plt.yticks([])
    plt.xlim(xmin, xmax)
    plt.tick_params(axis="both", which="major", labelsize=16)

    #plt.legend(fontsize=20)
    sns.despine()

    return f
