"""
Double DQN agent for Atari Breakout (TensorFlow/Keras + Gymnasium).

Commands:
  python breakout_dqn.py baseline --episodes 30 --gif    # score of a random agent
  python breakout_dqn.py train --steps 1000000           # train the agent
  python breakout_dqn.py eval --model checkpoints/best.keras --episodes 30 --gif
"""
import argparse
import csv
import os
import random
import time
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

gym.register_envs(ale_py)

ENV_ID = "ALE/Breakout-v5"
FIRE = 1  # Breakout needs FIRE to launch the ball after a reset or lost life


# ---------------------------------------------------------------- environment
def make_env(render=False):
    env = gym.make(
        ENV_ID,
        frameskip=1,  # AtariPreprocessing handles frame skipping
        repeat_action_probability=0.0,
        render_mode="rgb_array" if render else None,
    )
    # Grayscale, resize to 84x84, skip 4 frames per action, random no-op starts
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    # Stack the last 4 frames so the agent can see ball direction and speed
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


def to_hwc(obs):
    """(4, 84, 84) frame stack -> (84, 84, 4) channels-last for Conv2D."""
    return np.transpose(np.asarray(obs, dtype=np.uint8), (1, 2, 0))


# ---------------------------------------------------------------- model
def build_q_network(n_actions):
    """Convolutional network from DeepMind's 2015 DQN paper."""
    inputs = keras.Input(shape=(84, 84, 4))
    x = keras.layers.Rescaling(1.0 / 255)(inputs)
    x = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(x)
    x = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    outputs = keras.layers.Dense(n_actions)(x)
    return keras.Model(inputs, outputs)


class ReplayBuffer:
    """Fixed-size buffer stored as uint8 numpy arrays to keep memory down."""

    def __init__(self, capacity, obs_shape=(84, 84, 4)):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ids = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[ids].astype(np.float32),
            self.actions[ids],
            self.rewards[ids],
            self.next_obs[ids].astype(np.float32),
            self.dones[ids],
        )


class DQNAgent:
    def __init__(self, n_actions, lr=1e-4, gamma=0.99):
        # Gymnasium reports the action count as a numpy integer; Keras needs a plain int
        self.n_actions = int(n_actions)
        self.gamma = gamma
        self.q = build_q_network(self.n_actions)
        self.target_q = build_q_network(self.n_actions)
        self.update_target()
        self.optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=10.0)
        self.loss_fn = keras.losses.Huber()

    def update_target(self):
        self.target_q.set_weights(self.q.get_weights())

    @tf.function
    def _predict(self, x):
        return self.q(x, training=False)

    def act(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q_values = self._predict(tf.convert_to_tensor(obs[None], dtype=tf.float32))
        return int(tf.argmax(q_values[0]))

    @tf.function
    def train_step(self, obs, actions, rewards, next_obs, dones):
        # Double DQN: online network picks the next action, target network scores it
        next_actions = tf.argmax(self.q(next_obs, training=False), axis=1)
        next_q = self.target_q(next_obs, training=False)
        next_q = tf.reduce_sum(next_q * tf.one_hot(next_actions, self.n_actions), axis=1)
        targets = rewards + self.gamma * (1.0 - dones) * next_q

        with tf.GradientTape() as tape:
            q_values = self.q(obs, training=True)
            q_taken = tf.reduce_sum(q_values * tf.one_hot(actions, self.n_actions), axis=1)
            loss = self.loss_fn(targets, q_taken)
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_variables))
        return loss


def epsilon_at(step, start=1.0, end=0.05, decay_steps=250_000):
    """Linear decay from start to end over decay_steps, then constant."""
    frac = min(step / decay_steps, 1.0)
    return start + frac * (end - start)


# ---------------------------------------------------------------- plotting
def plot_rewards(log_path, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, rewards, avgs = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            avgs.append(float(row["avg_100"]))
    if not episodes:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label="Episode reward")
    plt.plot(episodes, avgs, linewidth=2, label="100-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Breakout DQN training")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------- progress GIFs
def record_progress_gif(agent, env, path, max_frames=2000):
    """Play one game with the current model and save it as a GIF."""
    import imageio

    obs, info = env.reset()
    obs = to_hwc(obs)
    lives, need_fire, total, done = info["lives"], True, 0.0, False
    frames = []
    while not done and len(frames) < max_frames:
        if need_fire:
            action, need_fire = FIRE, False
        else:
            action = agent.act(obs, 0.05)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = to_hwc(obs)
        total += reward
        done = terminated or truncated
        if info["lives"] < lives:
            need_fire = True
        lives = info["lives"]
        frames.append(env.render())
    imageio.mimsave(path, frames, fps=30)
    print(f"Saved progress GIF: {path} (score {total:.0f})")


# ---------------------------------------------------------------- training
def train(args):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("gifs/progress", exist_ok=True)
    log_path = "plots/rewards.csv"

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    env = make_env()
    gif_env = make_env(render=True)  # separate env so recording doesn't disturb training
    agent = DQNAgent(env.action_space.n, lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size)

    obs, info = env.reset(seed=args.seed)
    obs = to_hwc(obs)
    lives = info["lives"]
    need_fire = True
    episode, episode_reward = 0, 0.0
    recent = deque(maxlen=100)
    best_avg = -float("inf")
    start_time = time.time()

    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["episode", "step", "reward", "epsilon", "avg_100"])

        for step in range(1, args.steps + 1):
            eps = epsilon_at(step, decay_steps=args.eps_decay_steps)
            if need_fire:
                action, need_fire = FIRE, False
            else:
                action = agent.act(obs, eps)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = to_hwc(next_obs)
            episode_reward += reward

            # Treat a lost life as "done" for learning so the agent feels the penalty
            life_lost = info["lives"] < lives
            lives = info["lives"]
            buffer.add(obs, action, np.sign(reward), next_obs, float(terminated or life_lost))
            obs = next_obs
            if life_lost:
                need_fire = True

            if step > args.learning_starts and step % args.train_freq == 0:
                agent.train_step(*buffer.sample(args.batch_size))
            if step % args.target_update == 0:
                agent.update_target()

            if terminated or truncated:
                episode += 1
                recent.append(episode_reward)
                avg = float(np.mean(recent))
                writer.writerow([episode, step, episode_reward, round(eps, 4), round(avg, 2)])
                log_file.flush()

                if episode % 10 == 0:
                    speed = step / (time.time() - start_time)
                    print(f"ep {episode:5d} | step {step:8d} | reward {episode_reward:5.1f} | "
                          f"avg100 {avg:6.2f} | eps {eps:.3f} | {speed:.0f} steps/s")

                if step > args.learning_starts and len(recent) >= 20 and avg > best_avg:
                    best_avg = avg
                    agent.q.save("checkpoints/best.keras")

                obs, info = env.reset()
                obs = to_hwc(obs)
                lives = info["lives"]
                need_fire = True
                episode_reward = 0.0

            if step % args.checkpoint_every == 0:
                agent.q.save("checkpoints/latest.keras")
                plot_rewards(log_path, "plots/training_rewards.png")
                print(f"Checkpoint saved at step {step}")

            if args.gif_every and step % args.gif_every == 0:
                record_progress_gif(agent, gif_env, f"gifs/progress/step_{step:07d}.gif")

    agent.q.save("checkpoints/latest.keras")
    plot_rewards(log_path, "plots/training_rewards.png")
    print(f"Done. Best 100-episode average: {best_avg:.2f}")
    env.close()
    gif_env.close()


# ---------------------------------------------------------------- evaluation
def run_episodes(env, policy, n_episodes, gif_path=None, max_gif_frames=3000):
    scores, frames = [], []
    for ep in range(n_episodes):
        obs, info = env.reset()
        obs = to_hwc(obs)
        lives, need_fire, total, done = info["lives"], True, 0.0, False
        while not done:
            if need_fire:
                action, need_fire = FIRE, False
            else:
                action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            obs = to_hwc(obs)
            total += reward
            done = terminated or truncated
            if info["lives"] < lives:
                need_fire = True
            lives = info["lives"]
            if gif_path and ep == 0 and len(frames) < max_gif_frames:
                frames.append(env.render())
        scores.append(total)
        print(f"Episode {ep + 1}: {total:.0f}")

    if gif_path and frames:
        import imageio

        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved GIF: {gif_path}")

    print(f"\nAverage over {n_episodes} episodes: {np.mean(scores):.2f} "
          f"(std {np.std(scores):.2f}, max {np.max(scores):.0f})")
    return scores


def baseline(args):
    env = make_env(render=args.gif)
    gif_path = "gifs/random_baseline.gif" if args.gif else None
    run_episodes(env, lambda obs: env.action_space.sample(), args.episodes, gif_path)
    env.close()


def evaluate(args):
    env = make_env(render=args.gif)
    agent = DQNAgent(env.action_space.n)
    agent.q = keras.models.load_model(args.model)
    gif_path = "gifs/eval.gif" if args.gif else None
    # Small epsilon (standard for Atari evaluation) avoids getting stuck in loops
    run_episodes(env, lambda obs: agent.act(obs, 0.05), args.episodes, gif_path)
    env.close()


def main():
    parser = argparse.ArgumentParser(description="DQN for Atari Breakout")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=1_000_000, help="agent steps (each = 4 frames)")
    t.add_argument("--buffer-size", type=int, default=50_000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--learning-starts", type=int, default=20_000)
    t.add_argument("--train-freq", type=int, default=4)
    t.add_argument("--target-update", type=int, default=10_000)
    t.add_argument("--eps-decay-steps", type=int, default=250_000)
    t.add_argument("--checkpoint-every", type=int, default=50_000)
    t.add_argument("--gif-every", type=int, default=250_000, help="0 to disable progress GIFs")
    t.add_argument("--seed", type=int, default=42)

    b = sub.add_parser("baseline")
    b.add_argument("--episodes", type=int, default=30)
    b.add_argument("--gif", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--model", default="checkpoints/best.keras")
    e.add_argument("--episodes", type=int, default=30)
    e.add_argument("--gif", action="store_true")

    args = parser.parse_args()
    {"train": train, "baseline": baseline, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
