import math
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from numba import njit
import time

# --- Numba-compiled helpers ---

@njit
def _sort_desc(a):
    for i in range(a.shape[0]):
        for j in range(i+1, a.shape[0]):
            if a[i] < a[j]:
                tmp = a[i]; a[i] = a[j]; a[j] = tmp
    return a

@njit
def resolve_combat_jit(def_armies, atk_armies, ask_dice):
    max_attacker = atk_armies - 1
    ad = min(ask_dice, 3, max_attacker)
    if ad < 1:
        return 0, 0
    dd = 2 if def_armies >= 2 else def_armies
    att = np.empty(ad, np.int64)
    df  = np.empty(dd, np.int64)
    for i in range(ad):
        att[i] = np.random.randint(1, 7)
    for i in range(dd):
        df[i]  = np.random.randint(1, 7)
    a_s = _sort_desc(att.copy())
    d_s = _sort_desc(df.copy())
    al = 0; dl = 0
    m = ad if ad < dd else dd
    for i in range(m):
        if a_s[i] > d_s[i]:
            dl += 1
        else:
            al += 1
    return al, dl

@njit
def path_owned_jit(adj, owner, player, src, dst):
    T = adj.shape[0]
    seen = np.zeros(T, np.int8)
    queue = np.empty(T, np.int64)
    h = 0; t = 0
    seen[src] = 1
    queue[t] = src; t += 1
    while h < t:
        u = queue[h]; h += 1
        for v in range(T):
            if adj[u, v] == 1 and owner[v] == player:
                if v == dst:
                    return True
                if seen[v] == 0:
                    seen[v] = 1
                    queue[t] = v; t += 1
    return False

@njit
def step_jit(state, phase, current_player, action_flat,
             reinf, fortify_steps, t1, t2, turns_elapsed,
             flat_phase, flat_src, flat_dst, flat_cnt,
             adj, continent_of, cont_bonus,
             intermediate_scale, max_reinf, MAX_ARMY, max_turns):
    T = state.shape[0]

    # --- AUTO-SKIP EMPTY PHASES ---
    # If no reinforcements left, skip from Reinforce → Attack
    if phase == 0 and reinf == 0:
        phase = 1

    # If in Fortify but no legal fortify moves, skip → End
    elif phase == 2:
        can_fortify = False
        for s in range(T):
            if state[s,0] == current_player and state[s,1] > 1:
                for d in range(T):
                    if adj[s,d] == 1 and state[d,0] == current_player:
                        if path_owned_jit(adj, state[:,0], current_player, s, d):
                            can_fortify = True
                            break
                if can_fortify:
                    break
        if not can_fortify:
            phase = 3
    # --- end auto-skip ---

    # if no attacks possible, skip Attack → Fortify
    if phase == 1:
        can_attack = False
        for s in range(T):
            if state[s,0]==current_player and state[s,1]>1:
                for d in range(T):
                    if adj[s,d]==1 and state[d,0]!=current_player:
                        can_attack = True
                        break
                if can_attack: break
        if not can_attack:
            phase = 2

    # decode action
    act_phase = flat_phase[action_flat]
    src       = flat_src[action_flat]
    dst       = flat_dst[action_flat]
    k         = flat_cnt[action_flat]

    # Phase 0: Reinforce
    if phase == 0:
        if act_phase == 0:
            if k == 0 and reinf == 0:
                phase = 1
            elif 1 <= k <= reinf and state[dst,0] == current_player:
                state[dst,1] += k
                reinf -= k

    # Phase 1: Attack
    elif phase == 1:
        if act_phase == 2 and k == 0:
            phase = 2
        elif (act_phase == 1 and adj[src,dst]==1
              and state[src,0]==current_player
              and state[dst,0]!=current_player
              and 1 <= k < state[src,1]):
            al, dl = resolve_combat_jit(state[dst,1], state[src,1], k)
            state[src,1] -= al
            state[dst,1] -= dl
            if state[dst,1] <= 0:
                state[dst,0] = current_player
                move = k - al
                if move < 1: move = 1
                if move > state[src,1]: move = state[src,1]
                state[src,1] -= move
                state[dst,1] += move

    # Phase 2: Fortify
    elif phase == 2:
        if fortify_steps >= 28:
            if act_phase == 3 and k == 0:
                phase = 3
        else:
            if act_phase == 3 and k == 0:
                phase = 3
            elif (act_phase == 2 and 0 < k < state[src,1]
                  and state[src,0]==current_player
                  and state[dst,0]==current_player):
                if path_owned_jit(adj, state[:,0], current_player, src, dst):
                    state[src,1] -= k
                    state[dst,1] += k
                    fortify_steps += 1

    # Phase 3: End-turn
    else:
        current_player = 3 - current_player
        owned = (state[:,0] == current_player).sum()
        base = max(1, owned // 3)
        for ci in range(cont_bonus.shape[0]):
            complete = True
            for t in range(T):
                if continent_of[t] == ci and state[t,0] != current_player:
                    complete = False; break
            if complete:
                base += cont_bonus[ci]
        # go-second bonus
        if current_player == 2 and t2 == 0:
            base += 1
        reinf = base
        if current_player == 1:
            t1 += 1
        else:
            t2 += 1
        phase = 0
        turns_elapsed += 1
        fortify_steps = 0

    # check for game-over
    done = False
    if turns_elapsed >= max_turns:
        done = True
    else:
        all1 = True; all2 = True
        for i in range(T):
            if state[i,0] != 1: all1 = False
            if state[i,0] != 2: all2 = False
        if all1 or all2:
            done = True

    # differential-based raw reward
    owned1 = (state[:,0]==1).sum()
    owned2 = (state[:,0]==2).sum()
    b1 = max(1, owned1//3)
    b2 = max(1, owned2//3)
    for ci in range(cont_bonus.shape[0]):
        same1 = True; same2 = True
        for t in range(T):
            if continent_of[t] == ci:
                if state[t,0] != 1: same1 = False
                if state[t,0] != 2: same2 = False
        if same1: b1 += cont_bonus[ci]
        if same2: b2 += cont_bonus[ci]
    diff       = b1 - b2
    raw_reward = diff / max_reinf

    # override on true win/loss
    if done:
        all1 = True; all2 = True
        for i in range(T):
            if state[i,0] != 1: all1 = False
            if state[i,0] != 2: all2 = False
        if all1 or all2:
            raw_reward = 1.0 if all1 else -1.0

    # only deliver non-zero reward in phase 3
    scaled_reward = raw_reward if phase == 3 else 0.0

    return (state, phase, current_player,
            reinf, fortify_steps, t1, t2, turns_elapsed,
            scaled_reward, raw_reward, done)

# --- RiskEnv class ---

class RiskEnv(gym.Env):
    def __init__(self, max_turns=20):
        super().__init__()
        self.T                 = 8
        self.MAX_ARMY          = 40
        self.max_turns         = max_turns
        self.intermediate_scale= 0.1
        self.max_reinf         = math.ceil(self.T/3) + sum((2,1,1))

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

        self.continent_of = np.array([0,0,0,0,1,1,2,2], np.int8)
        self.cont_bonus    = np.array([2,1,1], np.int8)

        # Observation now includes the board, the current player, and the phase
        low  = np.tile([0,0], (self.T,1))
        high = np.tile([2,self.MAX_ARMY], (self.T,1))
        self.observation_space = spaces.Dict({
            "state":          spaces.Box(low, high, dtype=np.int8),
            "current_player": spaces.Discrete(3),  # 1 or 2
            "phase":          spaces.Discrete(4),  # 0..3
        })

        self.action_space = spaces.Dict({
            "phase": spaces.Discrete(4),
            "src":   spaces.Discrete(self.T),
            "dst":   spaces.Discrete(self.T),
            "count": spaces.Discrete(self.MAX_ARMY+1),
        })

        # build flat‐action lookup tables
        self.flat2action = []
        for ph in range(4):
            for s in range(self.T):
                for d in range(self.T):
                    for c in range(self.MAX_ARMY+1):
                        self.flat2action.append({
                            "phase":ph, "src":s, "dst":d, "count":c
                        })
        self.action2flat = {
            (a["phase"],a["src"],a["dst"],a["count"]): i
            for i,a in enumerate(self.flat2action)
        }
        self.flat_phase = np.array([a["phase"] for a in self.flat2action], np.int8)
        self.flat_src   = np.array([a["src"]   for a in self.flat2action], np.int8)
        self.flat_dst   = np.array([a["dst"]   for a in self.flat2action], np.int8)
        self.flat_cnt   = np.array([a["count"] for a in self.flat2action], np.int8)

        self.reset()

    def _start_reinforcement(self):
        owned = int((self.state[:,0] == self.current_player).sum())
        base  = max(1, owned // 3)
        for ci in range(self.cont_bonus.shape[0]):
            complete = True
            for t in range(self.T):
                if self.continent_of[t] == ci and self.state[t,0] != self.current_player:
                    complete = False; break
            if complete:
                base += self.cont_bonus[ci]
        if self.current_player == 2 and self.turn_count[2] == 0:
            base += 1
        self.turn_count[self.current_player] += 1
        self.reinforcements = base
        self.fortify_steps  = 0
        self.phase          = 0

    def reset(self):
        # initialize territories
        self.state = np.zeros((self.T,2), np.int8)
        terrs = list(range(self.T))
        p1 = random.sample(terrs,4)
        p2 = [t for t in terrs if t not in p1]
        for t in p1: self.state[t] = [1,1]
        for t in p2: self.state[t] = [2,1]
        for pl in (1,2):
            owned = [i for i,o in enumerate(self.state[:,0]) if o==pl]
            for _ in range(2):
                self.state[random.choice(owned),1] += 1

        self.current_player = 1
        self.turn_count      = {1:0, 2:0}
        self.phase           = 0
        self.reinforcements  = 0
        self.fortify_steps   = 0
        self.turns_elapsed   = 0

        self._start_reinforcement()

        return {
            "state":          self.state.copy(),
            "current_player": int(self.current_player),
            "phase":          int(self.phase),
        }

    def step(self, action):
        idx = self.action2flat[
            (action["phase"], action["src"], action["dst"], action["count"])
        ]
        (new_state, new_phase, new_current_player,
         new_reinf, new_fortify, new_t1, new_t2,
         new_turns, scaled_r, raw_r, done) = step_jit(
            self.state, self.phase, self.current_player, idx,
            self.reinforcements, self.fortify_steps,
            self.turn_count[1], self.turn_count[2], self.turns_elapsed,
            self.flat_phase, self.flat_src, self.flat_dst, self.flat_cnt,
            self.adj, self.continent_of, self.cont_bonus,
            self.intermediate_scale, self.max_reinf,
            self.MAX_ARMY, self.max_turns
        )

        self.state          = new_state
        self.phase          = new_phase
        self.current_player = new_current_player
        self.reinforcements = new_reinf
        self.fortify_steps  = new_fortify
        self.turn_count[1]  = new_t1
        self.turn_count[2]  = new_t2
        self.turns_elapsed  = new_turns

        obs = {
            "state":          self.state.copy(),
            "current_player": int(self.current_player),
            "phase":          int(self.phase),
        }
        info = {"raw_reward": float(raw_r)}
        return obs, float(scaled_r), done, info

    def render(self):
        print(f"\nPlayer {self.current_player}'s turn (phase {self.phase})")
        print("Territory | Owner | Armies")
        for i, (o,a) in enumerate(self.state):
            print(f"{i:9d} | {o:5d} | {a:6d}")
        print("")

    def legal_actions(self):
        cp, st, ph = self.current_player, self.state, self.phase
        out = []
        if ph == 0:  # Reinforce
            owned = np.where(st[:,0] == cp)[0]
            total = int(st[st[:,0] == cp,1].sum())
            for d in owned:
                for k in range(1, self.reinforcements+1):
                    if total + k <= self.MAX_ARMY:
                        out.append({'phase':0,'src':d,'dst':d,'count':k})
            if not out:
                out.append({'phase':0,'src':0,'dst':0,'count':0})

        elif ph == 1:  # Attack
            owned = np.where(st[:,0] == cp)[0]
            for s in owned:
                m = st[s,1] - 1
                if m > 0:
                    for d in range(self.T):
                        if self.adj[s,d] == 1 and st[d,0] != cp:
                            for k in range(1, m+1):
                                out.append({'phase':1,'src':s,'dst':d,'count':k})
            out.append({'phase':2,'src':0,'dst':0,'count':0})

        elif ph == 2:  # Fortify
            if self.fortify_steps >= 28:
                out.append({'phase':3,'src':0,'dst':0,'count':0})
            else:
                owned = np.where(st[:,0] == cp)[0]
                for s in owned:
                    m = st[s,1] - 1
                    if m > 0:
                        for d in range(self.T):
                            if self.adj[s,d] == 1 and st[d,0] == cp:
                                if path_owned_jit(self.adj, st[:,0], cp, s, d):
                                    for k in range(1, m+1):
                                        out.append({'phase':2,'src':s,'dst':d,'count':k})
                out.append({'phase':3,'src':0,'dst':0,'count':0})

        else:  # End‐turn
            out.append({'phase':3,'src':0,'dst':0,'count':0})

        return out

    def watch_random(self, delay=0.0):
        phase_names = ["Reinforce", "Attack", "Fortify", "End"]
        self.reset()
        done = False
        end_printed = False
        last_reward = 0.0

        while not done:
            print(f"\n** NOW PHASE: {phase_names[self.phase]} **")
            self.render()
            act = random.choice(self.legal_actions())
            print(f"CHOSEN ACTION PHASE: {phase_names[act['phase']]} "
                  f"(src={act['src']}, dst={act['dst']}, cnt={act['count']})")
            _, r, done, _ = self.step(act)
            print(f"reward = {last_reward}")
            last_reward = r
            if self.phase == 3 and not end_printed:
                print(f"\n** NOW PHASE: {phase_names[self.phase]} **")
                self.render()
                print("CHOSEN ACTION PHASE: End (src=0, dst=0, cnt=0)")
                print(f"reward = {r:.3f}")
                end_printed = True
            if done:
                print("\n*** Game Over ***")
                break
            time.sleep(delay)

    def perspective(self, state, player):
        return state.astype(np.float32)

def main():
    env = RiskEnv(max_turns=10)
    env.watch_random(delay=0)

if __name__ == "__main__":
    main()
