import numpy as np
import matplotlib.pyplot as plt
from src.python.settings import plotting_settings
from scipy.stats import entropy
import matplotlib.patches as patches
import seaborn as sns
import random
from collections import OrderedDict


# Levy comparison

def plot_color_grid(
    x_grid,
    y_grid,
    z_grid,
    cmap="viridis",
    xlabel="x",
    ylabel="y",
    cbar_title="z",
    xticks=None,
    yticks=None,
    show_yticks=True,
    ax=None
):
    """
    Plots a 2-dimensional color grid (adapted from bayesflow/sensitivity.py).

    Parameters
    ----------
    x_grid : np.ndarray
        Meshgrid of x values.
    y_grid : np.ndarray
        Meshgrid of y values.
    z_grid : np.ndarray
        Meshgrid of z values (coded by color in the plot).
    cmap : str, optional
        Color map for the fill. Default is 'viridis'.
    xlabel : str, optional
        X label text. Default is 'x'.
    ylabel : str, optional
        Y label text. Default is 'y'.
    cbar_title : str, optional
        Title of the color bar legend. Default is 'z'.
    xticks : list, optional
        List of x ticks. None results in dynamic ticks.
    yticks : list, optional
        List of y ticks. None results in dynamic ticks.
    show_yticks : bool, optional
        Whether to show y ticks. Default is True.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If not provided, a new figure will be created.

    Returns
    -------
    f : plt.Figure
        The figure instance for optional saving.
    """

    # Construct plot
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 6.5))

    im = ax.pcolormesh(x_grid, y_grid, z_grid, shading="nearest", cmap=cmap)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(labelsize=20)

    ticks_labels = [xtick.round(1) for xtick in xticks]
    ax.set_xticks(xticks, ticks_labels, rotation=45)
    if show_yticks:
        ax.set_yticks(yticks, ticks_labels)
    else:
        ax.set_yticks([])

    ax.minorticks_off()

    cbar_ticks = np.linspace(np.min(z_grid), np.max(z_grid), 4)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", location="top", pad=0.05, ticks=cbar_ticks, shrink=0.8)
    cbar.ax.set_xticklabels(cbar_ticks.round(2), fontsize=20)
    cbar.ax.set_xlabel(cbar_title, fontsize=28, labelpad=15)


def plot_prior_sensitivity(
    factors_1,
    factors_2,
    factor_list,
    pmps,
    num_ensemble_members,
    cmap,
    xlabel,
    ylabel,
    show_yticks,
    cbar_title,
    ax,
    target_model_for_avg=None,
    reference_pmps=None
):
    """
    Plots the sensitivity of the prior to different factors.
    Whether reference_pmps is provided determines whether the plot shows the average PMPs for one model
    or the average KL divergence between all model probabilities.

    Parameters
    ----------
    factors_1 : numpy.ndarray
        Array of shape (num_samples,) containing the first factor values.
    factors_2 : numpy.ndarray
        Array of shape (num_samples,) containing the second factor values.
    factor_list : list
        List of factor names.
    pmps : numpy.ndarray
        Array of shape (num_samples, num_ensemble_members, 4) containing the PMPs.
        Expects a vector for average probabilities for one model and an array for average KL divergence
        between all model probabilities.
    num_ensemble_members : int
        Number of ensemble members.
    cmap : matplotlib.colors.Colormap
        Colormap to use for the plot.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    show_yticks : bool
        Whether to show y-axis ticks.
    cbar_title : str
        Title for the colorbar.
    ax : matplotlib.axes.Axes
        Axes object to use for the plot.
    target_model_for_avg : int, optional
        Integer specifying the model for which the average PMPs should be calculated (using 1-based indexing).
        Only used for average PMP calculation.
    reference_pmps : numpy.ndarray, optional
        Array of shape (num_ensemble_members, 4) containing the reference PMPs.
        Only used for KL divergence calculation.

    Returns
    -------
    None
    """

    unique_factors = np.unique(factors_1)
    num_factors = len(unique_factors)

    # indexing argument prevents axis switching due to default cartesian indexing
    x_grid, y_grid = np.meshgrid(unique_factors, unique_factors, indexing="ij")
    z_grid = np.zeros((num_factors, num_factors))

    # Loop through unique combinations and calculate the mean for each ensemble
    for i, f1 in enumerate(unique_factors):
        for j, f2 in enumerate(unique_factors):
            mask = (factors_1 == f1) & (factors_2 == f2)
            if not reference_pmps:  # get avg. pmps
                z_grid[i, j] = np.mean(pmps[mask, target_model_for_avg - 1])
            if reference_pmps:  # Get avg. kl div to reference_pmps
                z_grid[i, j] = np.mean([
                                    entropy(reference_pmps, pmps[mask][net, :]) for net in range(num_ensemble_members)
                                ])

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=plotting_settings["figsize"])

    plot_color_grid(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        cmap=cmap,
        xlabel=xlabel,
        ylabel=ylabel,
        xticks=factor_list,
        yticks=factor_list,
        show_yticks=show_yticks,
        cbar_title=cbar_title,
        ax=ax
    )


def plot_sensitivity_posteriors(
    single_pmps,
    sensitivity_pmps,
    labels,
    color,
    alpha,
    fontsize_labels,
    fontsize_ticks,
    fontsize_title,
    ylabel=True,
    title=None,
    save=False,
    ax=None
):
    """
    Creates a combined violin and scatter plot to visualize sensitivity in model posteriors.

    Parameters:
    -----------
    single_pmps : np.array of shape (num_models)
        An array of posterior model probabilities from a single network.
    sensitivity_pmps : np.array of shape (num_ensemble_members, num_models)
        An array of posterior model probabilities.
    labels : list of str
        Model labels.
    color : str
        Color used for the violin plot and scatter points.
    alpha : float
        Transparency level for the violin plot.
    fontsize_labels : int
        Font size for the axis labels.
    fontsize_title : int
        Font size for the plot title (if provided).
    ylabel : bool, optional
        Whether to include a y-axis label for the plot.
    title : str or None, optional
        Title for the plot. If None, no title is displayed.
    save : bool, optional
        Whether to save the plot as a PDF file.
    ax : matplotlib.axes.Axes or None, optional
        An existing matplotlib Axes object to plot on. If None, a new figure
        and Axes will be created.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the plot.
    """

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=plotting_settings["figsize"])

    ax.grid(alpha=0.25)

    violin_plot = ax.violinplot(
        sensitivity_pmps,
        showextrema=True,
        showmedians=True
    )

    for vp in violin_plot["bodies"]:
        vp.set_facecolor(color)
        vp.set_edgecolor(color)
        vp.set_alpha(alpha)

    for partname in ["cbars", "cmins", "cmaxes", "cmedians"]:
        violin_plot[partname].set_edgecolor(color)
        violin_plot[partname].set_alpha(alpha)

    ax.scatter([1, 2, 3, 4], single_pmps, color=color)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(labels, fontsize=fontsize_labels)
    ax.yaxis.set_tick_params(labelsize=fontsize_ticks)
    if ylabel:
        ax.set_ylabel("Posterior model probability", fontsize=fontsize_labels)
    if not ylabel:
        ax.set_yticklabels([])
        ax.set_yticks(np.linspace(0, 1, 6))
    eps = 0.03
    ax.set_ylim([0 - eps, 1 + eps])

    if title:
        ax.set_title(title, fontsize=fontsize_title)#, pad=15)

    if save:
        plt.savefig("levy_sensitivity_posterior.pdf", dpi=300, bbox_inches="tight")


# Climate application: helper functions for plotting

def build_climate_model_color_dict(names, seed=42424):
    colors = sns.color_palette("bright", len(names)+1)
    del colors[-1]
    random.seed(seed)
    random.shuffle(colors)
    model_color_dict = OrderedDict((name, color) for name, color in zip(sorted(names), colors))
    return model_color_dict


def sensitivity_ridge_plot(df, x, palette, category, overlap=0.4, alpha=1, major=None, minor=None, label_func=None, major_hspace=0.3):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing samples from the posterior for different categories given by columns.
    x : str
        Column name of the variable to plot.
    palette : list
        List of colors to use for the different categories.
    category : str
        Column name of the category variable.
    overlap : float, optional
        Vertical overlap of category axes.
    major : str, optional
        Column name of the category variable to use for the grouping.
        If None, no grouping to major and minor category is used.
    minor : str, optional
        Column name of the category variable to use for the grouping.
        If None, no grouping to major and minor category is used.
    label_func : callable, optional
        Function to use for labeling the categories. Use this to define how to do major labelling.
        If label_func is None AND major&minor are None aswell the category labels are used.
        If label_func is None AND major&minor are not None, use category naming convention such that df['category'] contains strings like '<major> <minor>'.
    """
    assert category in df.columns, f'category={category} not in df.columns={df.columns}'
    assert x in df.columns, f'x={x} not in df.columns={df.columns}'
    if major is not None: assert major in df.columns, f'major={major} not in df.columns={df.columns}'
    if minor is not None: assert minor in df.columns, f'minor={minor} not in df.columns={df.columns}'
    assert (major is None) == (minor is None), 'either both major and minor must be None or not None'

    g = sns.FacetGrid(df, palette=palette, row=category, hue=category, aspect=9, height=1)

    g.map_dataframe(sns.kdeplot, x=x, fill=True, alpha=alpha)
    g.map_dataframe(sns.kdeplot, x=x, color='black')


    if label_func is None:
        if major is None:
            def label_func(x, color, label):
                ax = plt.gca()
                ax.text(0, (1-overlap)/2, label, color='black', fontsize=13,
                        ha='left', va='center', transform=ax.transAxes)
        else:
            started_groups = []
            def label_func(x, color, label):
                ax = plt.gca()
                ma, mi = label.split(' ')
                if ma in started_groups:
                    pass
                else:
                    ax.text(-major_hspace, (1-overlap)/2, ma, color='black', fontsize=24,
                            rotation=0, horizontalalignment='left',
                            ha='left', va='center', transform=ax.transAxes)
                    started_groups.append(ma)
                ax.text(0, (1-overlap)/2, mi, color='black', fontsize=13,
                        ha='left', va='center', transform=ax.transAxes)
    else:
        assert callable(label_func)

    g.map(label_func, category)

    g.fig.subplots_adjust(hspace=-overlap)
    g.set_titles('')
    g.set(yticks=[], xlabel=x)
    g.set(ylabel='')
    g.despine(left=True)

    return g

def group_facets(g, df, category, major, major_hspace=0.3, hpadding=0.02, overlap=0.4):
    """
    Draw boxes around groups of facets.

    Parameters
    ----------
    g : sns.FacetGrid
        FacetGrid object.
    df : pd.DataFrame
        Dataframe containing samples from the posterior for different categories given by columns.
    category : str
        Column name of the category variable.
    major : str
        Column name of the category variable to use for the major grouping.
    major_hspace : float, optional
        Horizontal space left of axes for major title.
    hpadding : float, optional
        Horizontal padding of box around group.
    overlap : float, optional
        Vertical overlap of category axes.

    """


    # Group them based on unique majors and put boxes around each group.

    previous_major = None
    y_upper = 0.97

    for ax, row in zip(g.axes.flat, df[category].unique()):
        current_major = df[df[category]==row][major].iloc[0]

        if previous_major is None:
            trans_major_hspace = major_hspace * (ax.get_position().bounds[2])
        elif current_major != previous_major:
            x_lower, y_lower, width = previous_ax.get_position().bounds[0:3]
            # draw box around previous group
            rect = patches.Rectangle(
                (x_lower-trans_major_hspace -hpadding, y_upper), trans_major_hspace+width + 2*hpadding, y_lower-y_upper,
                transform=g.fig.transFigure,  # Use figure coordinates
                color='black',
                fill=False,
                linewidth=2,
                zorder=-1  # Place below everything else
            )
            g.fig.patches.append(rect)
            y_upper = y_lower

        previous_ax = ax
        previous_major = current_major

    rect = patches.Rectangle(
        (x_lower-trans_major_hspace -hpadding, y_upper), trans_major_hspace+width + 2*hpadding, -y_upper,
        transform=g.fig.transFigure,  # Use figure coordinates
        color='black',
        fill=False,
        linewidth=2,
        zorder=-1  # Place below everything else
    )
    g.fig.patches.append(rect)