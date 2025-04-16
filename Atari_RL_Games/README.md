# 🕹️ Atari_RL_Games

This project focuses on developing reinforcement learning agents to play Atari games like **Seaquest** and **Boxing** using **Deep Q-Networks (DQN)** implemented with **TensorFlow** and **TF-Agents**. The aim is to train agents that can learn effective policies through exploration and exploitation strategies in classic visual gaming environments.

---

## 🎯 Objectives

- Build and train DQN agents using TF-Agents framework
- Apply frame stacking and preprocessing for image-based input
- Evaluate different ε-greedy exploration strategies
- Visualize average rewards and gameplay performance

---

## Main packages used:
 - tensorflow
 - tf-agents
 - gym[atari]
 - opencv-python
 - numpy, matplotlib

**Note: Ensure you have a GPU-enabled TensorFlow version if you want accelerated training.**

---

## 🧠 Notes
 - Used AtariPreprocessing and FrameStack from TF-Agents
 - Added custom ε-greedy policy scheduling
 - Replay buffer and target network implemented for stability
 - Tuned hyperparameters (γ, ε, learning rate, etc.)

---
## 📁 Folder Structure
Atari_RL_Games/
├── train_agent.py         # Training script
├── test_agent.py          # Evaluation script
├── plot_rewards.py        # Reward visualization
├── checkpoints/           # Saved model weights
├── videos/                # Recorded gameplay
├── plots/                 # Graphs of training reward
└── README.md              # This file

Atari_RL_Games/
 ─ train_agent.py         # Training script
 ─ test_agent.py          # Evaluation script
 ─ plot_rewards.py        # Reward visualization
 ─ checkpoints/           # Saved model weights
 ─ videos/                # Recorded gameplay
 ─ plots/                 # Graphs of training reward
 ─ README.md              # This file
