from pathlib import Path

import ray
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from algorithms.fcnn_model import FCMaskedActionsModelTF
from fjsp_env.envs.fjsp_env import FjspEnv
from utils.config import ENV_CONFIG, MODIFIED_CONFIG_PPO

ROOT = Path(__file__).parent.absolute()

if __name__ == "__main__":
    checkpoint_dir = ROOT / "checkpoints/skilled-paper-19/checkpoint_000095"
    ray.init()
    register_env("FjspEnv-v0", lambda config: FjspEnv(config))
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(MODIFIED_CONFIG_PPO)
    print("Training completed. Restoring new Trainer for action inference.")
    # Get the last checkpoint from the above training run.
    # Create new Trainer and restore its state from the last checkpoint.
    algo = ppo.PPO(config=ppo_config)
    algo.restore(checkpoint_dir)

    # Create the env to do inference in.
    env = FjspEnv(ENV_CONFIG)
    obs = env.reset()

    num_episodes = 0
    episode_reward = 0.0
    done = False
    while not done:
        # Compute an action (`a`).
        a = algo.compute_single_action(
            observation=obs,
            explore=False,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `a` to the env.
        obs, reward, done, _ = env.step(a)
        episode_reward += reward
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            print(env.solution)
            obs = env.reset()
            num_episodes += 1
            episode_reward = 0.0

    ray.shutdown()
