import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RiskEnv(gym.Env):
    
    def __init__(self):
        super().__init__()

        self.adj = np.array([
            #0 1 2 3 4 5 6 7
            [0,1,1,1,0,0,0,0],  # 0
            [1,0,1,1,1,0,0,0],  # 1
            [1,1,0,1,0,0,1,0],  # 2
            [1,1,1,0,0,0,0,0],  # 3
            [0,1,0,0,0,1,0,0],  # 4
            [0,0,0,0,1,0,1,0],  # 5
            [0,0,1,0,0,1,0,1],  # 6
            [0,0,0,0,0,0,1,0],  # 7
        ], dtype=np.int8)

        self.neighbors = {
            i: np.where(self.adj[i] == 1)[0].tolist()
            for i in range(8)}
        
        self.continents = {
            'A': [0,1,2,3],
            'B': [4,5],
            'C': [6,7],}
        
        self.MAX_ARMY = 10

        self.observation_space = spaces.Box(
            low=np.array([0, 0]*8),
            high=np.array([2,self.MAX_ARMY]*8),
            shape=(8,2),
            dtype=np.int8)
        
        self.attacks = []

        for i in range(8):
            for j in self.neighbors[i]:
                self.attacks.append((i,j))
        
        self.action_space = spaces.Discrete(len(self.attacks) + 1)

        self.reset()

def reset(self, *, seed=None, options=None):
    super().reset(seed=seed, options=options)

    state = np.zeros((8, 2), dtype=np.int8)

    territories = list(range(8))
    p1 = self.np_random.choice(territories, size=4, replace=False).tolist()
    p2 = [t for t in territories if t not in p1]

    for t in p1:
        state[t, 0] = 1
        state[t, 1] = 1
    for t in p2:
        state[t, 0] = 2
        state[t, 1] = 1

    for player_id in (1, 2):
        owned = np.where(state[:, 0] == player_id)[0]
        for _ in range(2):
            drop = int(self.np_random.choice(owned))
            state[drop, 1] += 1

    self.state = state

    return state, {}

