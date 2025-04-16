#!/usr/bin/env python
# coding: utf-8

# # 1) Boxing Game (Easy)

# In[1]:


import os
import re
import glob
import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import gym

from gym.wrappers import AtariPreprocessing, FrameStack
from tf_agents.environments import gym_wrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent

# Scan the Checkpoints Directory
checkpoints_dir = "checkpoints/boxing"
ckpt_index_files = glob.glob(os.path.join(checkpoints_dir, "ckpt-*.index"))

if not ckpt_index_files:
    print(f"No checkpoint files found in {checkpoints_dir}. Exiting.")
    exit()

pattern = re.compile(r"ckpt-(\d+)\.index")
steps = []
for path in ckpt_index_files:
    match = pattern.search(os.path.basename(path))
    if match:
        step_num = int(match.group(1))
        steps.append(step_num)
steps.sort() 
print("Found checkpoint steps:", steps)

ckpt_paths = [os.path.join(checkpoints_dir, f"ckpt-{step}") for step in steps]

# Create the Environment
def create_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, grayscale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

env_name = "BoxingNoFrameskip-v4"
py_env = create_env(env_name)
env = gym_wrapper.GymWrapper(py_env)
tf_env = TFPyEnvironment(env)

# Recreate the DQN Agent
def create_agent(env):
    train_step = tf.Variable(0)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=2.5e-4, rho=0.95, epsilon=1e-5, centered=True
    )
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=200000,
        end_learning_rate=0.01
    )
    preprocessing_layer = tf.keras.layers.Lambda(
        lambda obs: tf.cast(tf.transpose(obs, [0, 2, 3, 1]), tf.float32) / 255.0
    )
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]
    q_net = QNetwork(
        env.observation_spec(),
        env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params
    )
    agent = DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=500,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=0.99,
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step)
    )
    agent.initialize()
    return agent

agent = create_agent(tf_env)

# Evaluate Each Checkpoint
def evaluate_agent(tf_env, agent, num_episodes=5):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = tf_env.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = agent.policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            episode_return += time_step.reward.numpy()[0]
        total_return += episode_return
    return total_return / num_episodes

avg_returns = []
checkpoint = tf.train.Checkpoint(agent=agent)
for ckpt_path in ckpt_paths:
    checkpoint.restore(ckpt_path).expect_partial()
    avg_return = evaluate_agent(tf_env, agent, num_episodes=5)
    avg_returns.append(avg_return)
    print(f"Checkpoint {ckpt_path} => Average Return = {avg_return:.2f}")

# Plot Training Curve
plt.figure(figsize=(8,5))
plt.plot(steps, avg_returns, marker='o')
plt.xlabel("Checkpoint Step")
plt.ylabel("Average Return")
plt.title("Training Curve for Boxing Agent (Checkpoints)")
plt.savefig("trainingCurve-Boxing.png")
plt.show()
print("Saved training curve as 'trainingCurve-Boxing.png'.")

# Automatically Select Poor, Intermediate, Best Checkpoints
checkpoint_info = list(zip(ckpt_paths, steps, avg_returns))
sorted_info = sorted(checkpoint_info, key=lambda x: x[2])
poor_ckpt = sorted_info[0][0] 
best_ckpt = sorted_info[-1][0]
intermediate_ckpt = sorted_info[len(sorted_info)//2][0] 

print("Selected checkpoints based on performance:")
print("  Poor:", poor_ckpt)
print("  Intermediate:", intermediate_ckpt)
print("  Best:", best_ckpt)

# Record Gameplay Videos for Selected Checkpoints
def record_gameplay(agent, tf_env, ckpt_path, video_name, num_episodes=1, fps=30):
    checkpoint.restore(ckpt_path).expect_partial()
    print(f"Recording gameplay for checkpoint '{ckpt_path}' -> {video_name}")
    with imageio.get_writer(video_name, fps=fps) as video:
        for episode in range(num_episodes):
            time_step = tf_env.reset()
            while not time_step.is_last():
                action_step = agent.policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                frame = tf_env.pyenv.envs[0].render(mode="rgb_array")
                frame = np.array(frame, dtype=np.uint8)
                if frame.shape[0] == 1:
                    frame = frame.squeeze(0)
                video.append_data(frame)
    print(f"Video saved as {video_name}")

# Generate videos for poor, intermediate, and best checkpoints.
record_gameplay(agent, tf_env, poor_ckpt, "myAgentPlays-Boxing-poor.mp4")
record_gameplay(agent, tf_env, intermediate_ckpt, "myAgentPlays-Boxing-intermediate.mp4")
record_gameplay(agent, tf_env, best_ckpt, "myAgentPlays-Boxing-best.mp4")


# In[ ]:





# # 2) Seaquest Game(Harder)

# In[1]:


import os
import re
import glob
import gc
import numpy as np
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import gym

from gym.wrappers import AtariPreprocessing, FrameStack
from tf_agents.environments import gym_wrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.utils.common import function
from tf_agents.policies.policy_saver import PolicySaver

# Define Auto-Fire Wrapper for Atari Environments
class AtariPreprocessingWithAutoFire(AtariPreprocessing):
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        obs, _, done, info = super().step(1)
        return obs

    def step(self, action):
        lives_before = self.ale.lives()
        obs, reward, done, info = super().step(action)
        if self.ale.lives() < lives_before and not done:
            obs, _, done, info = super().step(1)
        return obs, reward, done, info

# Scan the Checkpoints Directory (for Seaquest)
checkpoints_dir = "checkpoints/seaquest"
ckpt_index_files = glob.glob(os.path.join(checkpoints_dir, "ckpt-*.index"))

if not ckpt_index_files:
    print(f"No checkpoint files found in {checkpoints_dir}. Exiting.")
    exit()

pattern = re.compile(r"ckpt-(\d+)\.index")
steps = []
for path in ckpt_index_files:
    match = pattern.search(os.path.basename(path))
    if match:
        step_num = int(match.group(1))
        steps.append(step_num)
steps.sort()
print("Found checkpoint steps:", steps)

ckpt_paths = [os.path.join(checkpoints_dir, f"ckpt-{step}") for step in steps]

# Create the Environment (Seaquest with Auto-Fire)
def create_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessingWithAutoFire(env, grayscale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

env_name = "SeaquestNoFrameskip-v4"
py_env = create_env(env_name)
env = gym_wrapper.GymWrapper(py_env)
tf_env = TFPyEnvironment(env)

# Recreate the DQN Agent
def create_agent(env):
    train_step = tf.Variable(0)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=1e-4, rho=0.95, epsilon=1e-5, centered=True
    )
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=500000,
        end_learning_rate=0.01
    )
    preprocessing_layer = tf.keras.layers.Lambda(
        lambda obs: tf.cast(tf.transpose(obs, [0, 2, 3, 1]), tf.float32) / 255.0
    )
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]

    q_net = QNetwork(
        env.observation_spec(),
        env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params
    )

    agent = DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=2000,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=0.99,
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step)
    )
    agent.initialize()
    return agent

agent = create_agent(tf_env)

# Evaluate Each Checkpoint
def evaluate_agent(tf_env, agent, num_episodes=20):
    total_return = 0.0
    for ep in range(num_episodes):
        time_step = tf_env.reset()
        episode_return = 0.0
        step_count = 0
        while not time_step.is_last():
            action_step = agent.policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            r = time_step.reward.numpy()[0]
            episode_return += r
            step_count += 1
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return

avg_returns = []
checkpoint = tf.train.Checkpoint(agent=agent)
for ckpt_path in ckpt_paths:
    checkpoint.restore(ckpt_path).expect_partial()
    avg_return = evaluate_agent(tf_env, agent, num_episodes=20)
    avg_returns.append(avg_return)
    print(f"Checkpoint {ckpt_path} => Average Return = {avg_return:.2f}")

# Plot Training Curve
plt.figure(figsize=(8,5))
plt.plot(steps, avg_returns, marker='o')
plt.xlabel("Checkpoint Step")
plt.ylabel("Average Return")
plt.title("Training Curve for Seaquest Agent (Checkpoints)")
plt.savefig("trainingCurve-Seaquest.png")
plt.show()
print("Saved training curve as 'trainingCurve-Seaquest.png'.")

# Automatically Select Poor, Intermediate, Best Checkpoints
checkpoint_info = list(zip(ckpt_paths, steps, avg_returns))
sorted_info = sorted(checkpoint_info, key=lambda x: x[2])
poor_ckpt = sorted_info[0][0] 
best_ckpt = sorted_info[-1][0]
intermediate_ckpt = sorted_info[len(sorted_info) // 2][0] 

print("Selected checkpoints based on performance:")
print("  Poor:", poor_ckpt)
print("  Intermediate:", intermediate_ckpt)
print("  Best:", best_ckpt)

# Record Gameplay Videos for Selected Checkpoints
def record_gameplay(agent, tf_env, ckpt_path, video_name, num_episodes=1, fps=30):
    checkpoint.restore(ckpt_path).expect_partial()
    print(f"Recording gameplay for checkpoint '{ckpt_path}' -> {video_name}")
    with imageio.get_writer(video_name, fps=fps) as video:
        for episode in range(num_episodes):
            time_step = tf_env.reset()
            while not time_step.is_last():
                action_step = agent.policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                frame = tf_env.pyenv.envs[0].render(mode="rgb_array")
                frame = np.array(frame, dtype=np.uint8)
                if frame.shape[0] == 1:
                    frame = frame.squeeze(0)
                video.append_data(frame)
    print(f"Video saved as {video_name}")

# Generate videos for poor, intermediate, and best checkpoints.
record_gameplay(agent, tf_env, poor_ckpt, "myAgentPlays-Seaquest-poor.mp4")
record_gameplay(agent, tf_env, intermediate_ckpt, "myAgentPlays-Seaquest-intermediate.mp4")
record_gameplay(agent, tf_env, best_ckpt, "myAgentPlays-Seaquest-best.mp4")
