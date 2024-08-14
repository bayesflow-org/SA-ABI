import os
import sys
import tensorflow as tf
import shutil

# Silence tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set paths
os.chdir(os.path.dirname(__file__))
sys.path.extend([
    os.path.abspath(os.path.join("../..")),
    os.path.abspath(os.path.join("../../../BayesFlow_dev/BayesFlow/"))
])

# Import from relative paths
from src.python.helpers import MaskingConfigurator
from src.python.training import load_training_data, setup_network
import bayesflow as bf

# Setup training
GOALS = ["pretrain", "finetune"]
POWER_SCALINGS = [True, False]
LEARNING_RATES = {"pretrain": 0.0005, "finetune": 0.00005}
BATCH_SIZE = 32
N_EPOCHS = 30  # equal for both goals
N_ENSEMBLE_MEMBERS = 10

# Start training loop
if __name__ == "__main__":
    for goal in GOALS:
        for power_scaling in POWER_SCALINGS:

            scaling_name = "sim_powerscaled" if power_scaling else "sim_unscaled"

            # Load data
            train_data = load_training_data(scaling_name, goal)
            val_data = load_training_data(scaling_name, "validate")

            # Handle existing networks via last_network (gets last ensemble nets' number to continue from there)
            last_network = 0
            while os.path.exists(f"ensemble_checkpoints/{scaling_name}/net{last_network}/{goal}"):
                last_network += 1

            for network in range(last_network, last_network + N_ENSEMBLE_MEMBERS):

                print(f'Starting to {goal} network{network} using {scaling_name}...')
                # Set up network
                tf.keras.backend.clear_session()
                summary_net, probability_net, amortizer = setup_network()

                # Set up trainer
                if goal == "finetune":
                    # Copy pretrained checkpoints to continue training in new folder
                    shutil.copytree(
                        f"ensemble_checkpoints/{scaling_name}/net{network}/pretrain",
                        f"ensemble_checkpoints/{scaling_name}/net{network}/finetune"
                    )
                checkpoint_path = f"ensemble_checkpoints/{scaling_name}/net{network}/{goal}"
                masking_configurator = MaskingConfigurator(power_scaling=power_scaling)
                trainer = bf.trainers.Trainer(
                    amortizer=amortizer,
                    configurator=masking_configurator,
                    checkpoint_path=checkpoint_path,
                    default_lr=LEARNING_RATES[goal]
                )

                # Train
                losses = trainer.train_offline(
                    simulations_dict=train_data,
                    epochs=N_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_sims=val_data,
                    **{"sim_dataset_args": {"batch_on_cpu": True}}
                )
