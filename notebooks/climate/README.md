# Preprocessing

The Jupyter notebooks `preprocess-all-sims.ipynb` and `preprocess-observation.ipynb` guide through the preprocessing of climate model simulation runs and observations.

# Reproducing results

The scripts and notebooks to reproduce all results from the paper are organized in four major steps: train, validate, illicit sensitivities and benchmark.
Configuration of datasets and hyperparameters are separated from the code and shared accross steps. Find them in the `config` folder.

1. To train a network, simply run
```
$ python 1_train.py --config <config-file-name>
```

2. The Jupyter notebook `2_validate.ipynb` is used to inspect the trained network, including recall and simulation based calibration.

3. `3_model&prior-sensitivity.ipynb` contains the full code to do sensitivity-aware inference on the preprocessed observations and reproduce the respective plots.

4. Finally, the Jupyter notebook `4_benchmark.ipynb` loads in several networks given their corresponding config files and aggregates benchmark metrics.

# Additional figures

`plot_priors.ipynb` and `plot_temp_map.ipynb` each produce additional plots used in the paper.
