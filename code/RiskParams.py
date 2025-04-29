import os
from multiprocessing import Pool
from itertools import product
import multiprocessing
import tensorflow as tf

# limit TF GPU memory growth to avoid OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from RiskEnv import RiskEnv
from RiskLearn import DDQNAgent

# Configuration constants
NUM_EPISODES = 200
NUM_EVALS    = 100
MAX_TURNS    = 40

# Hyperparameter grid
param_grid = {
    'batch_size':    [64, 128, 256],
    'gamma':         [0.9, 0.99, 1.0],
    'lr':            [1e-5, 1e-6, 1e-7],
    'epsilon_decay': [500, 1000, 2000],
}

# Ensure output directory exists
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(BASE_DIR, exist_ok=True)

# Worker function for one hyperparameter configuration
def run_config(kwargs):
    print(f"\n=== Running config: {kwargs} ===")
    env = RiskEnv(max_turns=MAX_TURNS)
    agent = DDQNAgent(env, **kwargs)
    agent.train(NUM_EPISODES)
    agent.evaluate_agent(NUM_EVALS)

if __name__ == '__main__':
    # Generate all combinations of hyperparameters
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*param_grid.values())]

    # Parallel execution across CPU cores
    with Pool() as pool:
        pool.map(run_config, combos)

    print("\nAll hyperparameter sweeps completed.")