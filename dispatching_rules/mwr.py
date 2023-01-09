import random
import wandb
import gym
import numpy as np

from utils.config import ENV_CONFIG
from fjsp_env.envs.fjsp_env import FjspEnv

def MWR_worker(config):
    wandb.init(config=config)
    config = wandb.config
    env = FjspEnv(env_config=config)
    env.seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    done = False
    state = env.reset()
    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.operations, 9))
        max_duration_jobs = np.repeat(env.jobs_max_duration, env.max_alternatives)
        remaining_time = (reshaped[:, 3] * env.max_time_jobs) / max_duration_jobs
        illegal_actions = np.invert(legal_actions)
        mask = illegal_actions * 1e8
        remaining_time += mask
        MTWR_action = np.argmin(remaining_time)
        assert legal_actions[MTWR_action]
        state, reward, done, _ = env.step(MTWR_action)
    energy_costs = env.total_energy_costs
    env.reset()
    make_span = env.last_time_step
    wandb.log({"nb_episodes": 1, "make_span": make_span, "noop_actions": 0, "energy_costs": energy_costs}, )


if __name__ == "__main__":
    MWR_worker(ENV_CONFIG)