import random
import time
from pathlib import Path

import numpy as np
import ray
import ray.tune.integration.wandb as wandb_tune
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import register_env
from ray.tune.utils import flatten_dict

import wandb
from algorithms.fcnn_model import FCMaskedActionsModelTF
from fjsp_env.envs.fjsp_env import FjspEnv
from utils.config import MODIFIED_CONFIG_PPO
from utils.CustomCallbacks import CustomCallbacks

# TODO: Move this to RLLib do not use tune or integrate wandb for the training
# TODO: save the model locally

ROOT = Path(__file__).parent.absolute()

tf1, tf, tfv = try_import_tf()
# Use these result keys to update `wandb.config`
_config_results = [
    "trial_id", "experiment_tag", "node_ip", "experiment_id", "hostname",
    "pid", "date",
]
_exclude_results = ["done", "should_checkpoint", "config"]


def _handle_result(result) :
    config_update = result.get("config", {}).copy()
    log = {}
    flat_result = flatten_dict(result, delimiter="/")

    for k, v in flat_result.items():
        if any(
                k.startswith(item + "/") or k == item
                for item in _config_results):
            config_update[k] = v
        elif any(
                k.startswith(item + "/") or k == item
                for item in _exclude_results):
            continue
        else:
            log[k] = v

    config_update.pop("callbacks", None)  # Remove callbacks
    return log, config_update


if __name__ == "__main__":
    ray.init()
    wandb.init(config=MODIFIED_CONFIG_PPO)
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    register_env("FjspEnv-v0", lambda config: FjspEnv(config))
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(MODIFIED_CONFIG_PPO)
    wandb.config.update(ppo_config)
    trainer = ppo.PPO(config=ppo_config)
    
    start_time = time.time()
    
    stop = {
        "time_total_s": 10 * 60,
    }
    while start_time + stop['time_total_s'] > time.time():
        result = trainer.train()        
        result = wandb_tune._clean_log(result)
        log, config_update = _handle_result(result)
        wandb.log(result["custom_metrics"])
        wandb.log(result["sampler_results"])
        wandb.config.update(config_update, allow_val_change=True)

    trainer.save(ROOT / "checkpoints" / str(wandb.run.name))
    ray.shutdown()    