# Plotting settings
# General settings
plotting_settings = {
    "figsize": (5, 5),
    "colors": ["#800000", "#000080", "#008000", "#800080"],
    "alpha_plot": 0.9,
    "alpha_50%CI": 0.5,
    "alpha_95%CI": 0.3,
    "alpha_grid": 0.25,
    "fontsize_labels": 16,
    "fontsize_title": 24,
    "fontsize_legend": 16,
}

# Update: Pass to plt.rcParams.update() to adjust plotting for whole file
plotting_update = {
        "axes.labelsize": 24,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.sans-serif": "Palatino",
        "text.latex.preamble": r"\usepackage{{amsmath}}",
    }

# Network settings
# Levy comparison
summary_meta_diffusion = {
    "level_1": {
        "dense_s1_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s2_args": dict(
                units=64, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s3_args": dict(
                units=32, activation="elu", kernel_initializer="glorot_normal"
            )
    },
    "level_2": {
        "dense_s1_args": dict(
                units=64, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s2_args": dict(
                units=128, activation="elu", kernel_initializer="glorot_normal"
            ),
        "dense_s3_args": dict(
                units=64, activation="elu", kernel_initializer="glorot_normal"
            )
    }
}

probability_meta_diffusion = {
    "dense_args": dict(units=128, activation="elu", kernel_initializer="glorot_normal")
}
