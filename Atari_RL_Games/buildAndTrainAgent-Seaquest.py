#!/usr/bin/env python
# coding: utf-8

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

# ------------------------------------------------------------
# Define Auto-Fire Wrapper for Atari Environments
# ------------------------------------------------------------
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

# Create Environment (Seaquest with Auto-Fire)
def create_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessingWithAutoFire(env, grayscale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

game_env = create_env("SeaquestNoFrameskip-v4")
game_env = gym_wrapper.GymWrapper(game_env)
tf_game_env = TFPyEnvironment(game_env)

# Define Q-Network
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

# Create the DQN Agent
def create_agent(env, q_net):
    train_step = tf.Variable(0)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=1e-4, rho=0.95, epsilon=1e-5, centered=True
    )
    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=500000,
        end_learning_rate=0.01
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

agent = create_agent(tf_game_env, q_net)

# Create Replay Buffer
def create_replay_buffer(agent, env, buffer_size=200000):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=1,
        max_length=buffer_size
    )

replay_buffer = create_replay_buffer(agent, tf_game_env)

# Warm-up Replay Buffer
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
            if self.counter % 100 == 0:
                print(f"\rWarm-up: {self.counter}/{self.total} steps collected", end="")

warmup_steps = 20000  
initial_collect_policy = RandomTFPolicy(tf_game_env.time_step_spec(), tf_game_env.action_spec())
warmup_driver = DynamicStepDriver(
    tf_game_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(warmup_steps)],
    num_steps=warmup_steps
)
final_time_step, final_policy_state = warmup_driver.run()
print("\nWarm-up of replay buffer completed!")

# Training Loop with Checkpointing & Policy Saving
checkpoint_dir = "checkpoints/seaquest"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = tf.train.Checkpoint(agent=agent)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

n_iterations = 150000  
checkpoint_intervals = [50000, 100000, 149999]

dataset = replay_buffer.as_dataset(sample_batch_size=64, num_steps=2).prefetch(3)
iterator = iter(dataset)

collect_driver = DynamicStepDriver(
    tf_game_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=4
)
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

cumulative_steps = 0
cumulative_episodes = 0
rewards = [] 

for iteration in range(n_iterations):
    collect_driver.run()
    trajectories, _ = next(iterator)
    train_loss = agent.train(trajectories)

    episode_reward = np.sum(trajectories.reward.numpy())
    rewards.append(episode_reward)

    cumulative_steps += len(trajectories.reward.numpy())
    cumulative_episodes += 1

    window_size = 2500
    if len(rewards) >= window_size:
        avg_reward = np.mean(rewards[-window_size:])
    else:
        avg_reward = np.mean(rewards) if rewards else 0

    if iteration % 2500 == 0:
        print(f"Iteration {iteration}/{n_iterations} | Loss: {train_loss.loss.numpy():.4f} | "
              f"Avg Reward (last {window_size}): {avg_reward:.2f} | "
              f"Steps: {cumulative_steps} | Episodes: {cumulative_episodes}")

    if iteration in checkpoint_intervals:
        checkpoint_manager.save()
        print(f"Saved checkpoint at iteration {iteration} (Steps: {cumulative_steps}, Episodes: {cumulative_episodes})")
        
#Save Policy
tf_policy = agent.policy

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None], dtype=tf.int32, name="step_type"),
    tf.TensorSpec(shape=[None], dtype=tf.float32, name="reward"),
    tf.TensorSpec(shape=[None], dtype=tf.float32, name="discount"),
    tf.TensorSpec(shape=[None, 4, 84, 84], dtype=tf.uint8, name="observation")
])
def serving_default_fn(step_type, reward, discount, observation):
    time_step = ts.TimeStep(
        step_type=step_type,
        reward=reward,
        discount=discount,
        observation=observation,
    )
    return tf_policy.action(time_step).action

policy_saver = PolicySaver(tf_policy, batch_size=None)
save_path = "saved_policies/seaquest"
os.makedirs(save_path, exist_ok=True)
tf.saved_model.save(tf_policy, save_path, signatures={"serving_default": serving_default_fn})
print(f"Policy successfully saved at '{save_path}' with serving_default!")
