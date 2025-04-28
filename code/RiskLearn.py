import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import RiskEnv

class RiskLearn:
    def __init__(
        self,
        env=None,
        max_turns=16,
        replay_size=50_000,
        batch_size=64,
        gamma=0.99,
        lr=1e-5,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=2e-5,
        tau=1e-4,
        hidden_units=(64,64)
    ):
        # Environment
        self.env = env or RiskEnv.RiskEnv(max_turns=max_turns)
        self.T   = self.env.T                # # of territories
        self.N   = len(self.env.flat_phase)  # # of flat actions

        # Replay buffer
        self.replay     = deque(maxlen=replay_size)
        self.batch_size = batch_size

        # Discount & exploration
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau           = tau

        # Loss tracking — switch to Mean Squared Error (quadratic loss)
        self.mse_loss     = tf.keras.losses.MeanSquaredError()
        self.loss_history = []

        # Build networks
        self.q_model       = self._build_model(hidden_units, lr)
        self.target_model  = models.clone_model(self.q_model)
        self.target_model.set_weights(self.q_model.get_weights())

        # Compile the replay-update into a single tf.function
        # Now accepts pre-built NumPy arrays for states
        self._compiled_replay_update = tf.function(
            self._replay_update_step, reduce_retracing=True
        )

    def _build_model(self, hidden_units, lr):
        m = models.Sequential()
        m.add(layers.Input(shape=(self.T,2)))
        m.add(layers.Flatten())
        for u in hidden_units:
            m.add(layers.Dense(u, activation='relu'))
        m.add(layers.Dense(self.N))
        m.compile(optimizer=optimizers.Adam(lr), loss=self.mse_loss)
        return m

    def flatten(self, action):
        return self.env.action2flat[
            (action['phase'], action['src'], action['dst'], action['count'])
        ]

    def unflatten(self, idx):
        return self.env.flat2action[idx]

    def legal_actions_flat(self):
        return [self.flatten(a) for a in self.env.legal_actions()]

    def train_episode(self):
        obs        = self.env.reset()
        state_arr  = obs["state"]
        done       = False
        last_scaled= 0.0
        last_raw   = 0.0
        steps      = 0

        while not done:
            cp    = self.env.current_player
            valid = self.legal_actions_flat()

            if cp == 1:
                # player 1: ε-greedy
                if random.random() < self.epsilon:
                    idx = random.choice(valid)
                else:
                    qv   = self.q_model(self.env.perspective(state_arr,1)[None]).numpy()[0]
                    mask = np.full(self.N, -1e9, dtype=np.float32)
                    mask[valid] = qv[valid]
                    idx = int(mask.argmax())
                action = self.unflatten(idx)
            else:
                # player 2: random
                idx    = random.choice(valid)
                action = self.unflatten(idx)

            next_obs, scaled_r, done, info = self.env.step(action)
            raw_r         = info["raw_reward"]
            next_state_arr= next_obs["state"]

            last_scaled = scaled_r
            last_raw    = raw_r

            if cp == 1:
                # store & train only on player 1
                self.replay.append((
                    state_arr.copy(), idx, scaled_r,
                    next_state_arr.copy(), done
                ))
                if len(self.replay) >= self.batch_size:
                    self._replay_update()

            state_arr = next_state_arr
            steps += 1
            self.epsilon = max(self.epsilon_min,
                               self.epsilon - self.epsilon_decay)

        # soft update target network
        for w, wt in zip(self.q_model.trainable_variables,
                         self.target_model.trainable_variables):
            wt.assign(self.tau * w + (1.0 - self.tau) * wt)

        return last_scaled, last_raw, steps

    def _replay_update(self):
        # sample a batch
        batch = random.sample(self.replay, self.batch_size)
        sb, ab, rb, nsb, db = zip(*batch)

        # build NumPy state arrays outside the tf.function
        s_t_np  = np.stack([self.env.perspective(s,1) for s in sb])
        ns_t_np = np.stack([self.env.perspective(s,1) for s in nsb])

        # convert the other lists into fixed-shape NumPy arrays
        ab_np = np.array(ab, dtype=np.int32)
        rb_np = np.array(rb, dtype=np.float32)
        db_np = np.array(db, dtype=bool)

        # call the compiled update with pure NumPy inputs
        self._compiled_replay_update(s_t_np, ab_np, rb_np, ns_t_np, db_np)

    def _replay_update_step(self, s_t_np, ab, rb, ns_t_np, db):
        # convert inputs to tensors
        s_t       = tf.convert_to_tensor(s_t_np, dtype=tf.float32)
        ns_t      = tf.convert_to_tensor(ns_t_np, dtype=tf.float32)
        ab_tensor = tf.convert_to_tensor(ab, dtype=tf.int32)
        rb_tensor = tf.convert_to_tensor(rb, dtype=tf.float32)
        done_mask = tf.convert_to_tensor(db, dtype=tf.bool)

        # Double-DQN target calculation
        qn_next   = self.q_model(ns_t)
        best_next = tf.argmax(qn_next, axis=1)
        qn_target = self.target_model(ns_t)

        future_q = tf.reduce_sum(
            tf.one_hot(best_next, self.N) * qn_target, axis=1
        )
        y = rb_tensor + self.gamma * tf.where(done_mask, 0.0, future_q)

        with tf.GradientTape() as tape:
            q_vals = self.q_model(s_t, training=True)
            q_sel  = tf.reduce_sum(
                q_vals * tf.one_hot(ab_tensor, self.N), axis=1
            )
            loss   = tf.reduce_mean(tf.square(y - q_sel))

        grads = tape.gradient(loss, self.q_model.trainable_variables)
        self.q_model.optimizer.apply_gradients(
            zip(grads, self.q_model.trainable_variables)
        )
        self.loss_history.append(loss)

    def train(self, episodes=256, show=False):
        rewards, steps = [], []
        for ep in tqdm(range(1, episodes+1)):
            _, raw_r, s = self.train_episode()
            rewards.append(raw_r)
            steps.append(s)
            print(f"Episode {ep}/{episodes} | epsilon={self.epsilon:.3f} | reward={raw_r:+.3f} | steps={s}")
            # render end phase
            self.env.phase = 3
            print("** NOW PHASE: End **")
            self.env.render()
            print(f"reward = {raw_r:+.3f}")
            print("-"*40)

        # plot & save loss curve
        plt.figure()
        plt.plot(self.loss_history)
        plt.xlabel("Replay-updates")
        plt.ylabel("MSE loss")
        plt.title("Loss Curve")
        out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "loss_curve.png"))
        if show:
            plt.show()
        plt.close()

        # save model weights
        self.q_model.save_weights(
            os.path.join(out_dir, "weights.weights.h5")
        )

        return rewards, steps

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs found:", gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU detected, will run on CPU.")

    learner = RiskLearn(max_turns=20, epsilon_decay=1e-5)
    learner.train(episodes=528, show=True)

if __name__ == "__main__":
    main()
