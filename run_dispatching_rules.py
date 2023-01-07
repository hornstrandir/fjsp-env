from dispatching_rules.fifo import FIFO_worker
from utils.config import ENV_CONFIG

if __name__ == "__main__":
    FIFO_worker(ENV_CONFIG)
