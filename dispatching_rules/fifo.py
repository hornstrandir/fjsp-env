import random
import numpy as np
import wandb
from fjsp_env.envs.fjsp_env import FjspEnv
from utils.config import ENV_CONFIG


def FIFO_worker(env_config):
    wandb.init(config=env_config)
    env = FjspEnv(env_config)
    env.seed(2023)
    random.seed(2023)
    np.random.seed(2023)
    done = False
    state = env.reset()
    while not done:
        real_state = np.copy(state['real_obs'])
        legal_actions = state['action_mask'][:-1]
        reshaped = np.reshape(real_state, (env.operations, 9))
        remaining_time = reshaped[:, 5]
        illegal_actions = np.invert(legal_actions)
        mask = illegal_actions * -1e8
        remaining_time += mask
        FIFO_action = np.argmax(remaining_time)
        assert legal_actions[FIFO_action]
        state, reward, done, _ = env.step(FIFO_action)
    make_span = env.last_time_step
    total_energy_costs = env.total_energy_costs
    print(env.solution)
    env.reset()
    wandb.log({"nb_episodes": 1, "make_span": make_span, "total_energy_costs": total_energy_costs})

if __name__ == "__main__":
    FIFO_worker(ENV_CONFIG)