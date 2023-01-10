from pathlib import Path
import multiprocessing as mp
import os
from utils.CustomCallbacks import CustomCallbacks

ROOT = Path(__file__).parents[1].absolute()

MODEL_CONFIG = {
    "layer_size": 319,
    "layer_nb": 2,
}

ENV_CONFIG = {
    "instance_path": ROOT / "data/instances_preprocessed/MK01.fjs",
    "energy_data_path": ROOT / "data/electric_price/price_MinMax_sept_14_days.npy",
    "power_consumption_machines": {
        "4": [8, 19, 17, 7],
        "5": [12, 8, 3, 4, 7],
        "6": [14, 14, 7, 6, 19, 5],
        "8": [2, 8, 13, 14, 16, 5, 19, 18],
        "10": [11, 11, 3, 19, 10, 16, 12, 15, 13, 17],
        "15": [12, 3, 19, 10, 5, 13, 11, 18, 17, 16, 14, 12, 3, 10, 18],
        "20": [3, 8, 19, 6, 1, 19, 13, 9, 5, 18, 9, 1, 15, 9, 13, 14, 7, 9, 2, 3],
    },
    "alpha": 0.75,
    "loose_noop_restrictions": False,
    "seed": 2023,
}
MODIFIED_CONFIG_PPO = {
    "env": "FjspEnv-v0",
    "env_config": ENV_CONFIG,
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "callbacks": CustomCallbacks,
    "model": {
        "custom_model": "fc_masked_model_tf",
        "fcnet_activation": "relu",
        "fcnet_hiddens": [
            MODEL_CONFIG["layer_size"] for k in range(MODEL_CONFIG["layer_nb"])
        ],
        "vf_share_layers": False,
    },
    "framework": "tf",
    "ignore_worker_failures": True,
    "log_level": "WARN",
    "disable_env_checking": True,
}

DEFAULT_CONFIG_PPO = {
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE (lambda) parameter.
    "lambda": 1.0,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # Size of batches collected from each worker.
    "rollout_fragment_length": 704,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": mp.cpu_count() * 4 * 704,  # Default: 4000,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,
    # Stepsize of SGD.
    "lr": 5e-5,
    # Learning rate schedule.
    "lr_schedule": None,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers=True inside your model's config.
    "vf_loss_coeff": 1.0,
    "model": {
        # Share layers for value function. If you set this to True, it's
        # important to tune vf_loss_coeff.
        "vf_share_layers": False,
    },
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.0,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Target value for KL divergence.
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
    # Deprecated keys:
    # Share layers for value function. If you set this to True, it's important
    # to tune vf_loss_coeff.
    # Use config.model.vf_share_layers instead.
    # "vf_share_layers": DEPRECATED_VALUE,
}
