import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
from RiskEnv import RiskEnv

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def build_q_network(input_dim, output_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim)
    ])
    return model

class DDQNAgent:
    def __init__(
        self, env,
        buffer_size=10000, batch_size=128, gamma=0.99,
        lr=1e-6, target_update_freq=2,
        epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500
    ):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        # input: player, phase, 8 owners, 8 armies, reinf, turn_norm
        self.input_dim = 1 + 1 + 8 + 8 + 1 + 1
        self.output_dim = (
            env.action_space['type'].n *
            env.action_space['source'].n *
            env.action_space['target'].n
        )

        self.policy_net = build_q_network(self.input_dim, self.output_dim)
        self.target_net = build_q_network(self.input_dim, self.output_dim)
        self.target_net.set_weights(self.policy_net.get_weights())

        self.optimizer = optimizers.Adam(lr, clipnorm=0.5)
        self.loss_fn = losses.Huber(delta=1.0)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.loss_history = []

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # parameter prefix for files (include epsilon_decay)
        gamma_str = str(self.gamma).replace('.', 'p')
        lr_str = str(lr).replace('.', 'p')
        epsdec_str = str(self.epsilon_decay)
        self.params = f"bs{batch_size}_epsdec{epsdec_str}_gamma{gamma_str}_lr{lr_str}"

    def encode_state(self, obs):
        turn_norm = obs['turn'] / self.env.max_turns
        return np.concatenate([
            [obs['player']], [obs['phase']],
            obs['owners'], obs['armies'],
            [obs['reinforcements']], [turn_norm]
        ]).astype(np.float32)

    @tf.function
    def greedy_action(self, state, legal_mask):
        q = self.policy_net(state)
        neg_inf = tf.constant(-1e9, q.dtype)
        masked = tf.where(legal_mask, q, neg_inf)
        return tf.argmax(masked, axis=1, output_type=tf.int32)

    def select_action(self, state, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        mask = np.zeros(self.output_dim, dtype=bool)
        mask[legal_actions] = True
        state_tf = tf.constant(state[None, :], dtype=tf.float32)
        mask_tf = tf.constant(mask[None, :])
        return int(self.greedy_action(state_tf, mask_tf)[0])

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        next_q = self.policy_net(next_states)
        next_actions = tf.argmax(next_q, axis=1, output_type=tf.int32)
        next_target_q = self.target_net(next_states)
        idx = tf.range(tf.shape(states)[0], dtype=tf.int32)
        next_q_sel = tf.gather_nd(next_target_q, tf.stack([idx, next_actions], axis=1))
        target_q = rewards + (1.0 - dones) * self.gamma * next_q_sel

        with tf.GradientTape() as tape:
            q_vals = self.policy_net(states)
            q_sel = tf.gather_nd(q_vals, tf.stack([idx, actions], axis=1))
            loss = self.loss_fn(target_q, q_sel)
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        return loss

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)
        states = tf.constant(s, dtype=tf.float32)
        actions = tf.constant(a, dtype=tf.int32)
        rewards = tf.constant(r, dtype=tf.float32)
        next_states = tf.constant(ns, dtype=tf.float32)
        dones = tf.constant(d, dtype=tf.float32)
        loss = self.train_step(states, actions, rewards, next_states, dones)
        return float(loss.numpy())

    def train(self, num_episodes=500):
        base_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        os.makedirs(base_dir, exist_ok=True)

        episode_rewards = []
        for ep in tqdm(range(1, num_episodes+1), desc="Training"):
            obs, _ = self.env.reset(seed=random.randrange(1_000_000_000))
            state = self.encode_state(obs)
            done = False
            final_reward = 0.0
            while not done:
                legal = self.env.get_legal_actions()
                flat = [a['type']*64 + a['source']*8 + a['target'] for a in legal] or [0]
                if obs['phase'] == self.env.PHASE_ATTACK:
                    action = self.select_action(state, flat)
                else:
                    action = random.choice(flat)
                a_type, src, tgt = action//64, (action%64)//8, action%8

                obs_next, reward, done, _, _ = self.env.step({
                    'type':a_type, 'source':src, 'target':tgt
                })
                next_state = self.encode_state(obs_next)
                loss = self.optimize()
                if loss is not None:
                    self.loss_history.append(loss)
                self.replay_buffer.push(state, action, reward, next_state, float(done))
                obs, state = obs_next, next_state
                final_reward = reward

            episode_rewards.append(final_reward)
            if ep % self.target_update_freq == 0:
                self.target_net.set_weights(self.policy_net.get_weights())
            self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(-ep / self.epsilon_decay)

        weights_path = os.path.join(base_dir, f"{self.params}_weights.h5")
        loss_png = os.path.join(base_dir, f"{self.params}_loss.png")
        loss_txt = os.path.join(base_dir, f"{self.params}_loss.txt")
        rewards_txt = os.path.join(base_dir, f"{self.params}_episode_rewards.txt")

        self.policy_net.save_weights(weights_path)
        plt.figure()
        plt.plot(self.loss_history)
        plt.savefig(loss_png)
        plt.close()
        np.savetxt(loss_txt, np.array(self.loss_history), fmt='%.6f')
        np.savetxt(rewards_txt, np.array(episode_rewards), fmt='%.6f')

        print(f"Saved weights to {weights_path}")
        print(f"Saved loss curve to {loss_png} and data to {loss_txt}")
        print(f"Saved episode rewards to {rewards_txt}")

    def evaluate_agent(self, num_games=100):
        base_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        eval_rewards = []
        wins = 0.0
        total_reward = 0.0

        for _ in tqdm(range(num_games), desc="Agent Eval"):
            obs, _ = self.env.reset()
            state = self.encode_state(obs)
            done = False
            final_reward = 0.0

            while not done:
                legal = self.env.get_legal_actions()
                flat = [a['type']*64 + a['source']*8 + a['target'] for a in legal] or [0]
                mask = np.zeros(self.output_dim, dtype=bool)
                mask[flat] = True
                state_tf = tf.constant(state[None, :], dtype=tf.float32)
                mask_tf = tf.constant(mask[None, :])
                action = int(self.greedy_action(state_tf, mask_tf)[0])

                a_type, src, tgt = action//64, (action%64)//8, action%8
                obs, reward, done, _, _ = self.env.step({
                    'type':a_type, 'source':src, 'target':tgt
                })
                state = self.encode_state(obs)
                final_reward = reward

            eval_rewards.append(final_reward)
            total_reward += final_reward
            if final_reward > 0:
                wins += 1.0
            elif final_reward == 0.0:
                wins += 0.5

        # save evaluation rewards
        eval_txt = os.path.join(base_dir, f"{self.params}_eval_rewards.txt")
        np.savetxt(eval_txt, np.array(eval_rewards), fmt='%.6f')
        print(f"Saved evaluation rewards to {eval_txt}")

        print(f"\nAgent Win rate: {wins/num_games:.2f}")
        print(f"Average final reward: {total_reward/num_games:.4f}")

    def evaluate_random(self, num_games=100):
        wins = 0.0
        total_reward = 0.0

        for _ in tqdm(range(num_games), desc="Random Eval"):
            obs, _ = self.env.reset()
            done = False
            final_reward = 0.0

            while not done:
                action = random.choice(self.env.get_legal_actions())
                obs, reward, done, _, _ = self.env.step(action)
                final_reward = reward

            total_reward += final_reward
            if final_reward > 0:
                wins += 1.0
            elif final_reward == 0.0:
                wins += 0.5

        print(f"\nRandom Win rate: {wins/num_games:.2f}")
        print(f"Random Avg final reward: {total_reward/num_games:.4f}")

    def visualize_game(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        state = self.encode_state(obs)
        done = False
        step = 1
        print("=== Visualization ===")
        self.env.render()
        while not done:
            legal = self.env.get_legal_actions()
            flat = [a['type']*64 + a['source']*8 + a['target'] for a in legal] or [0]
            if obs['phase'] == self.env.PHASE_ATTACK and obs['player'] == 0:
                action = self.select_action(state, flat)
            else:
                action = random.choice(flat)
            a_type, src, tgt = action//64, (action%64)//8, action%8
            obs, reward, done, _, _ = self.env.step({
                'type':a_type, 'source':src, 'target':tgt
            })
            state = self.encode_state(obs)
            self.env.render()
            step += 1


def sweep_params(param_grid, num_episodes=50, num_evals=50, max_turns=40):
    base_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(base_dir, exist_ok=True)
    keys = list(param_grid.keys())
    for vals in product(*param_grid.values()):
        kwargs = dict(zip(keys, vals))
        print(f"\n=== Sweep: {kwargs} ===")
        env = RiskEnv(max_turns=max_turns)
        agent = DDQNAgent(env, **kwargs)
        agent.train(num_episodes)
        agent.evaluate_agent(num_evals)

if __name__ == '__main__':
    env = RiskEnv(max_turns=40)
    agent = DDQNAgent(env)
    # agent.train(20)
    # agent.evaluate_agent(20)
    sweep_params({'batch_size': [64, 128, 256], 'epsilon_decay': [250, 500, 1000], 'gamma': [0.9, 0.99, 1], 'lr': [1e-5, 1e-6, 1e-7],}, num_episodes=50, num_evals=50)