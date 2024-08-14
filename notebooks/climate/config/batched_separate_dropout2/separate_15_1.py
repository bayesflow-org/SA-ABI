config = {
    "threshold": 1.5,
    "datapath": "../../../climate/sim-data/preproc/",
    "filenames_sims": [
        "tas_anual_preproc_ssp370_IPSL-CM6A-LR.nc"
    ],
    "context": "ssp&model",
    "context_aware": False,
    "prior_range_override": [
        -40,
        41
    ],
    "year_bounds": [
        1970,
        2100
    ],
    "checkpoint_path": "checkpoints/batched_separate_dropout2/separate_15_1",
    "presimulate_path": "checkpoints/presims-15/",
    "n": 1,
    "summary_net": {
        "type": "dense",
        "kwargs": {
            "hidden_units": [
                25,
                25
            ],
            "output_dim": 4,
            "zeroth_layer": {
                "dropout": 0.4
            }
        }
    },
    "inference_net": {
        "type": "bf-invertible",
        "kwargs": {
            "num_params": 2,
            "num_coupling_layers": 1
        }
    },
    "member_split": {
        "train": [
            0,
            1,
            2,
            3,
            4,
            5,
            6
        ],
        "val": [
            7,
            8
        ],
        "test": [
            9
        ]
    },
    "epochs": 15,
    "iterations_per_epoch": 60,
    "batch_size": 32,
    "rng_seed": 2023
}