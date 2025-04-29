import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from numba import njit

# Numba helpers
@njit
def can_attack_nb(adj, owners, armies, player, src, tgt):
    return (adj[src, tgt] == 1 and
            owners[src] == player and
            owners[tgt] != player and
            armies[src] > 1)

@njit
def can_fortify_nb(adj, owners, armies, player, src, tgt):
    return (adj[src, tgt] == 1 and
            owners[src] == player and
            owners[tgt] == player and
            armies[src] > 1)

@njit
def terminal_nb(armies, owners):
    p0 = 0
    p1 = 0
    for i in range(armies.shape[0]):
        if owners[i] == 0:
            p0 += armies[i]
        else:
            p1 += armies[i]
    if p1 == 0 and p0 > 0:
        return 1.0, True
    if p0 == 0 and p1 > 0:
        return -1.0, True
    return 0.0, False

class RiskEnv(gym.Env):
    """
    Simplified Risk environment with unified normalized-difference reward.
    Phases: reinforce, attack, fortify; max turns triggers done.
    Reward each non-terminal step is (p0 - p1)/total_army_cap in [-1,1],
    and on a true conquest (terminal), returns exactly +1 or -1.
    """
    metadata = {"render.modes": ["human"]}

    PHASE_REINFORCE = 0
    PHASE_ATTACK    = 1
    PHASE_FORTIFY   = 2

    def __init__(self, num_players=2, total_army_cap=64, max_turns=100):
        super().__init__()
        self.num_players     = num_players
        self.total_army_cap  = total_army_cap
        self.max_turns       = max_turns
        self.turn_count      = 0

        # adjacency matrix for 8 territories
        self.adj = np.array([
            [0,1,1,1,0,0,0,0],
            [1,0,1,1,1,0,0,0],
            [1,1,0,1,0,0,1,0],
            [1,1,1,0,0,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,0,0,1,0,1,0],
            [0,0,1,0,0,1,0,1],
            [0,0,0,0,0,0,1,0],
        ], np.int8)

        self.continents = {'A': list(range(4)), 'B': [4,5], 'C': [6,7]}
        self.continent_bonus = {'A':2,'B':1,'C':1}

        self.observation_space = spaces.Dict({
            'player':         spaces.Discrete(self.num_players),
            'phase':          spaces.Discrete(3),
            'owners':         spaces.MultiDiscrete([self.num_players]*8),
            'armies':         spaces.Box(0, self.total_army_cap, shape=(8,), dtype=np.int8),
            'reinforcements': spaces.Discrete(100),
            'turn':           spaces.Discrete(self.max_turns+1),
        })
        self.action_space = spaces.Dict({
            'type':   spaces.Discrete(3),
            'source': spaces.Discrete(8),
            'target': spaces.Discrete(8),
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # only seed the env’s RNG for dice rolls, not global random()
        self.owners        = np.array([0,1,0,1,0,1,0,1], dtype=np.int8)
        self.armies        = np.ones(8, dtype=np.int8)
        self.current_player= 0
        self.phase         = self.PHASE_REINFORCE
        self.reinforcements= self._compute_reinforcements(self.current_player)
        self.turn_count    = 0
        return self._get_obs(), {}

    def step(self, action):
        a_type = action['type']
        src    = action['source']
        tgt    = action['target']

        # --- reinforce phase ---
        if self.phase == self.PHASE_REINFORCE:
            if a_type == self.PHASE_REINFORCE and self.owners[src] == self.current_player:
                owned_total = int(self.armies[self.owners == self.current_player].sum())
                allowed     = min(self.reinforcements, self.total_army_cap - owned_total)
                self.armies[src] += allowed
            self.phase = self.PHASE_ATTACK

        # --- attack phase ---
        elif self.phase == self.PHASE_ATTACK:
            if (a_type == self.PHASE_ATTACK and
                can_attack_nb(self.adj, self.owners, self.armies,
                              self.current_player, src, tgt)):
                self._attack(src, tgt)
            self.phase = self.PHASE_FORTIFY

        # --- fortify & end-turn phase ---
        else:
            if (a_type == self.PHASE_FORTIFY and
                can_fortify_nb(self.adj, self.owners, self.armies,
                               self.current_player, src, tgt)):
                move_amt = int(self.armies[src] - 1)
                self.armies[src] -= move_amt
                self.armies[tgt] += move_amt
            # rotate player
            self.current_player = (self.current_player + 1) % self.num_players
            self.phase          = self.PHASE_REINFORCE
            self.reinforcements = self._compute_reinforcements(self.current_player)
            self.turn_count    += 1

        # --- reward & done check ---
        p0 = int(self.armies[self.owners == 0].sum())
        p1 = int(self.armies[self.owners == 1].sum())
        # base reward: normalized difference
        reward = (p0 - p1) / self.total_army_cap
        reward = max(min(reward, 1.0), -1.0)

        # if conquest, override with ±1
        term_r, term_done = terminal_nb(self.armies, self.owners)
        if term_done and term_r != 0.0:
            reward = term_r

        done = bool(term_done or self.turn_count >= self.max_turns)
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return {
            'player':         self.current_player,
            'phase':          self.phase,
            'owners':         self.owners.copy(),
            'armies':         self.armies.copy(),
            'reinforcements': self.reinforcements,
            'turn':           self.turn_count,
        }

    def get_legal_actions(self):
        legal = []
        p, ph = self.current_player, self.phase
        if ph == self.PHASE_REINFORCE:
            for s in range(8):
                if self.owners[s] == p:
                    legal.append({'type':ph,'source':s,'target':s})
        elif ph == self.PHASE_ATTACK:
            for s in range(8):
                for t in range(8):
                    if can_attack_nb(self.adj,self.owners,self.armies,p,s,t):
                        legal.append({'type':ph,'source':s,'target':t})
            # allow skip
            legal.append({'type':ph,'source':0,'target':0})
        else:
            for s in range(8):
                for t in range(8):
                    if can_fortify_nb(self.adj,self.owners,self.armies,p,s,t):
                        legal.append({'type':ph,'source':s,'target':t})
            legal.append({'type':ph,'source':0,'target':0})
        return legal

    def render(self, mode='human'):
        print(f"Player {self.current_player}, Phase {self.phase}, Turn {self.turn_count}/{self.max_turns}")
        print("Owners:", self.owners)
        print("Armies:", self.armies)

    def _compute_reinforcements(self, player):
        cnt  = int((self.owners == player).sum())
        base = min(1, cnt // 3)
        bonus= 0
        for cont, terr in self.continents.items():
            if all(self.owners[t] == player for t in terr):
                bonus += self.continent_bonus[cont]
        return base + bonus

    def _attack(self, src, tgt):
        a = min(3, int(self.armies[src] - 1))
        d = min(2, int(self.armies[tgt]))
        atk_rolls = sorted([random.randint(1,6) for _ in range(a)], reverse=True)
        def_rolls = sorted([random.randint(1,6) for _ in range(d)], reverse=True)
        for av, dv in zip(atk_rolls, def_rolls):
            if av > dv:
                self.armies[tgt] = max(0, int(self.armies[tgt] - 1))
            else:
                self.armies[src] = max(1, int(self.armies[src] - 1))
        if self.armies[tgt] == 0:
            self.owners[tgt] = self.current_player
            self.armies[tgt] = 1
            self.armies[src] -= 1

    def close(self):
        pass

def main():
    env = RiskEnv(max_turns=50)
    obs, _ = env.reset()
    done = False
    while not done:
        action  = random.choice(env.get_legal_actions())
        obs, r, done, *_ = env.step(action)
        env.render()
    print("Game ended with final reward:", r)

if __name__ == '__main__':
    main()
