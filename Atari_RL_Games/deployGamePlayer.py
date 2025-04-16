#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install imageio[ffmpeg]


# In[ ]:


pip install imageio[pyav]


# # 1) Boxing Game (Easy)

# In[1]:


import os
import numpy as np
import tensorflow as tf
import imageio
from gym.wrappers import AtariPreprocessing, FrameStack
from tf_agents.environments import gym_wrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories import time_step as ts
import gym

# Load the Saved Policy
policy_dir = "saved_policies/boxing"
policy = tf.saved_model.load(policy_dir)
print(" Successfully loaded policy from:", policy_dir)

# Setup Environment for Gameplay
def create_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, grayscale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

game_env = create_env("BoxingNoFrameskip-v4")
game_env = gym_wrapper.GymWrapper(game_env)
tf_game_env = TFPyEnvironment(game_env)

# Record Gameplay and Save Video
def record_gameplay(policy, env, video_name="myAgentPlays-Boxing.mp4", num_episodes=1, fps=30):
    print(" Recording gameplay...")
    
    with imageio.get_writer(video_name, fps=fps) as video:
        for episode in range(num_episodes):
            time_step = env.reset()
            while not time_step.is_last():
                formatted_time_step = ts.TimeStep(
                    step_type=tf.convert_to_tensor(time_step.step_type.numpy(), dtype=tf.int32),
                    reward=tf.convert_to_tensor(time_step.reward.numpy(), dtype=tf.float32),
                    discount=tf.convert_to_tensor(time_step.discount.numpy(), dtype=tf.float32),
                    observation=tf.convert_to_tensor(time_step.observation.numpy(), dtype=tf.uint8)
                )

                # Get action from policy using the serving_default signature
                action_step = policy.signatures["serving_default"](
                    step_type=formatted_time_step.step_type,
                    reward=formatted_time_step.reward,
                    discount=formatted_time_step.discount,
                    observation=formatted_time_step.observation
                )

                time_step = env.step(action_step['output_0'])
                frame = env.render(mode="rgb_array")
                frame = np.array(frame, dtype=np.uint8)
                if frame.shape[0] == 1:
                    frame = frame.squeeze(0)
                video.append_data(frame)

    print(f" Video saved as {video_name}")
    return video_name

# Gameplay Recording
video_file = record_gameplay(policy, tf_game_env)


# In[ ]:





# # 2) Seaquest Game (Harder)

# In[1]:


import os
import numpy as np
import tensorflow as tf
import imageio
from gym.wrappers import AtariPreprocessing, FrameStack
from tf_agents.environments import gym_wrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories import time_step as ts
import gym

# Load the Saved Policy for Seaquest
policy_dir = "saved_policies/seaquest"
policy = tf.saved_model.load(policy_dir)
print(" Successfully loaded policy from:", policy_dir)

# Setup Environment for Gameplay (Seaquest)
def create_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, grayscale_obs=True)
    env = FrameStack(env, num_stack=4)
    return env

game_env = create_env("SeaquestNoFrameskip-v4")
game_env = gym_wrapper.GymWrapper(game_env)
tf_game_env = TFPyEnvironment(game_env)

# Record Gameplay and Save Video
def record_gameplay(policy, env, video_name="myAgentPlays-Seaquest.mp4", num_episodes=1, fps=30):
    print(" Recording gameplay...")
    
    with imageio.get_writer(video_name, fps=fps) as video:
        for episode in range(num_episodes):
            time_step = env.reset()
            while not time_step.is_last():
                formatted_time_step = ts.TimeStep(
                    step_type=tf.convert_to_tensor(time_step.step_type.numpy(), dtype=tf.int32),
                    reward=tf.convert_to_tensor(time_step.reward.numpy(), dtype=tf.float32),
                    discount=tf.convert_to_tensor(time_step.discount.numpy(), dtype=tf.float32),
                    observation=tf.convert_to_tensor(time_step.observation.numpy(), dtype=tf.uint8)
                )

                # Get action from policy using the serving_default signature
                action_step = policy.signatures["serving_default"](
                    step_type=formatted_time_step.step_type,
                    reward=formatted_time_step.reward,
                    discount=formatted_time_step.discount,
                    observation=formatted_time_step.observation
                )

                time_step = env.step(action_step['output_0'])
                frame = env.render(mode="rgb_array")
                frame = np.array(frame, dtype=np.uint8)
                if frame.shape[0] == 1:
                    frame = frame.squeeze(0)
                video.append_data(frame)

    print(f" Video saved as {video_name}")
    return video_name

# Gameplay Recording for Seaquest
video_file = record_gameplay(policy, tf_game_env)
