#!/usr/bin/env python
# coding: utf-8

# # 1) Boxing Game (Simple)

# In[1]:


import os
import gc
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.wrappers import AtariPreprocessing, FrameStack
from tf_agents.environments import gym_wrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.utils.common import function
from tf_agents.trajectories import time_step as ts
from tf_agents.policies.random_tf_policy import RandomTFPolicy

# Setup GPU & Clear Session
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled!")
    except RuntimeError as e:
        print("Error enabling GPU memory growth:", e)

tf.keras.backend.clear_session()
gc.collect()

# Recreate the Environment
def create_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, grayscale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

game_env = create_env("BoxingNoFrameskip-v4")
game_env = gym_wrapper.GymWrapper(game_env)
tf_game_env = TFPyEnvironment(game_env)

# Recreate the Q-Network
def create_q_network(env):
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
    return q_net

q_net = create_q_network(tf_game_env)

# Recreate the DQN Agent
def create_agent(env, q_net):
    train_step = tf.Variable(0)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=2.5e-4, rho=0.95, epsilon=1e-5, centered=True
    )
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=200000,
        end_learning_rate=0.01
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

agent = create_agent(tf_game_env, q_net)

# Recreate the Replay Buffer
def create_replay_buffer(agent, env, buffer_size=150000):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=1,
        max_length=buffer_size
    )

replay_buffer = create_replay_buffer(agent, tf_game_env)

# Restore Latest Checkpoint 
checkpoint_dir = "checkpoints/boxing"
checkpoint = tf.train.Checkpoint(agent=agent)

latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if not latest_ckpt:
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}. Cannot resume training.")

checkpoint.restore(latest_ckpt).expect_partial()
print(f"Restored checkpoint from: {latest_ckpt}")

# Resume Training
def resume_train_agent(agent, env, replay_buffer, n_iterations=10000):
    dataset = replay_buffer.as_dataset(sample_batch_size=32, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    collect_driver = DynamicStepDriver(env, agent.collect_policy, observers=[replay_buffer.add_batch], num_steps=4)
    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)

    rewards = []
    cumulative_steps = 0
    cumulative_episodes = 0

    for iteration in range(n_iterations):
        collect_driver.run()
        trajectories, _ = next(iterator)
        train_loss = agent.train(trajectories)

        episode_reward = np.sum(trajectories.reward.numpy())
        rewards.append(episode_reward)

        cumulative_steps += len(trajectories.reward.numpy())
        cumulative_episodes += 1

        window_size = 1000
        avg_reward = (np.mean(rewards[-window_size:])
                      if len(rewards) >= window_size
                      else np.mean(rewards) if rewards
                      else 0)

        if iteration % 1000 == 0:
            print(f"Resumed Iteration {iteration}/{n_iterations} | Loss: {train_loss.loss.numpy():.4f} | "
                  f"Avg Reward (last {window_size}): {avg_reward:.2f} | Steps: {cumulative_steps} | Episodes: {cumulative_episodes}")

print("Resuming Training Agent...")
resume_train_agent(agent, tf_game_env, replay_buffer, n_iterations=10000)
print("Resumed training Boxing completed.")


# In[ ]:





# # 2) Seaquest Game (Harder)

# In[1]:


import os
import gc
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.wrappers import AtariPreprocessing, FrameStack
from tf_agents.environments import gym_wrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.utils.common import function
from tf_agents.trajectories import time_step as ts
from tf_agents.policies.random_tf_policy import RandomTFPolicy

# Setup GPU & Clear Session
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled!")
    except RuntimeError as e:
        print("Error enabling GPU memory growth:", e)

tf.keras.backend.clear_session()
gc.collect()

# Recreate the Environment (Seaquest)
def create_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, grayscale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

# Environment Seaquest
game_env = create_env("SeaquestNoFrameskip-v4")
game_env = gym_wrapper.GymWrapper(game_env)
tf_game_env = TFPyEnvironment(game_env)

# Recreate the Q-Network
def create_q_network(env):
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
    return q_net

q_net = create_q_network(tf_game_env)

# Recreate the DQN Agent
def create_agent(env, q_net):
    train_step = tf.Variable(0)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=2.5e-4, rho=0.95, epsilon=1e-5, centered=True
    )
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=300000,
        end_learning_rate=0.01
    )

    agent = DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_period=1000,
        td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        gamma=0.99,
        train_step_counter=train_step,
        epsilon_greedy=lambda: epsilon_fn(train_step)
    )
    agent.initialize()
    return agent

agent = create_agent(tf_game_env, q_net)

# Recreate the Replay Buffer
def create_replay_buffer(agent, env, buffer_size=200000):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=1,
        max_length=buffer_size
    )

replay_buffer = create_replay_buffer(agent, tf_game_env)

# Restore Latest Checkpoint 
checkpoint_dir = "checkpoints/seaquest" 
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(agent=agent)

latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if not latest_ckpt:
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}. Cannot resume training.")

checkpoint.restore(latest_ckpt).expect_partial()
print(f"Restored checkpoint from: {latest_ckpt}")

# Resume Training
def resume_train_agent(agent, env, replay_buffer, n_iterations=10000):
    dataset = replay_buffer.as_dataset(sample_batch_size=32, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    collect_driver = DynamicStepDriver(
        env, agent.collect_policy, observers=[replay_buffer.add_batch], num_steps=4
    )
    collect_driver.run = function(collect_driver.run)
    agent.train = function(agent.train)

    rewards = []
    cumulative_steps = 0
    cumulative_episodes = 0

    window_size = 1000
    for iteration in range(n_iterations):
        collect_driver.run()
        trajectories, _ = next(iterator)
        train_loss = agent.train(trajectories)

        episode_reward = np.sum(trajectories.reward.numpy())
        rewards.append(episode_reward)

        cumulative_steps += len(trajectories.reward.numpy())
        cumulative_episodes += 1

        avg_reward = np.mean(rewards[-window_size:]) if len(rewards) >= window_size else np.mean(rewards) if rewards else 0

        if iteration % 1000 == 0:
            print(f"Resumed Iteration {iteration}/{n_iterations} | Loss: {train_loss.loss.numpy():.4f} | "
                  f"Avg Reward (last {window_size}): {avg_reward:.2f} | Steps: {cumulative_steps} | Episodes: {cumulative_episodes}")

print("Resuming Training Agent...")
resume_train_agent(agent, tf_game_env, replay_buffer, n_iterations=10000)
print("Resumed training Seaquest completed.")

