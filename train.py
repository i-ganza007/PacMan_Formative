!pip install -q --force-reinstall "numpy==1.26.4" "scipy<1.13" "scikit-learn<1.6" "matplotlib<3.9"

!apt-get update -qq && apt-get install -y -qq swig cmake libopenmpi-dev zlib1g-dev

!pip install -q \
    "gymnasium[box2d,atari,accept-rom-license]" \
    "stable-baselines3[extra]>=2.0.0" \
    "ale-py" \
    "torch>=2.0" \
    "tensorflow<2.17" \
    "keras<3.0"

!pip cache purge

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import numpy as np
import random
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

CONFIG_ID = 2

configs = [
    {"lr": 0.0001, "gamma": 0.99,  "batch": 8,  "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 0.3},
    {"lr": 0.0005, "gamma": 0.95,  "batch": 16, "eps_start": 1.0, "eps_end": 0.1,  "eps_decay": 0.2},
    {"lr": 0.0002, "gamma": 0.99,  "batch": 8,  "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.4},
    {"lr": 0.0003, "gamma": 0.999, "batch": 12, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 0.25},
    {"lr": 0.00015,"gamma": 0.98,  "batch": 8,  "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 0.35},
    {"lr": 0.0004, "gamma": 0.99,  "batch": 16, "eps_start": 1.0, "eps_end": 0.1,  "eps_decay": 0.3},
    {"lr": 0.0001, "gamma": 0.995, "batch": 12, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 0.2},
    {"lr": 0.0002, "gamma": 0.98,  "batch": 8,  "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 0.5},
    {"lr": 0.00025,"gamma": 0.99,  "batch": 16, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 0.3},
    {"lr": 0.0003, "gamma": 0.999, "batch": 12, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 0.25},
]

cfg = configs[CONFIG_ID - 1]


print(f"Policy: CnnPolicy")
print(f"Hyperparameters:")
print(f"  • Learning Rate (lr): {cfg['lr']}")
print(f"  • Gamma (γ): {cfg['gamma']}")
print(f"  • Batch Size: {cfg['batch']}")
print(f"  • Epsilon Start: {cfg['eps_start']}")
print(f"  • Epsilon End: {cfg['eps_end']}")
print(f"  • Epsilon Decay: {cfg['eps_decay']}")

env_test = gym.make("ALE/SpaceInvaders-v5")
random_scores = []

for episode in range(1, 21):
    obs, _ = env_test.reset()
    done = False
    truncated = False
    score = 0
    while not (done or truncated):
        action = random.choice([0, 1, 2, 3, 4, 5])
        obs, reward, done, truncated, _ = env_test.step(action)
        score += reward
    random_scores.append(score)
    if episode % 5 == 0:
        print(f"  Episode {episode}: {score}")

avg_random_score = np.mean(random_scores)
print(f"\nRandom Agent Average Score: {avg_random_score:.2f}")
env_test.close()

env = gym.make("ALE/SpaceInvaders-v5", frameskip=4)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=cfg["lr"],
    gamma=cfg["gamma"],
    batch_size=cfg["batch"],
    buffer_size=10000,
    learning_starts=5000,
    exploration_initial_eps=cfg["eps_start"],
    exploration_final_eps=cfg["eps_end"],
    exploration_fraction=cfg["eps_decay"],
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    tensorboard_log=f"./logs/",
    verbose=1,
    device="cuda"
)

print("Model created successfully!")
print(f"   Buffer size: {model.buffer_size:,}")
print(f"   Learning starts at: {model.learning_starts:,} steps")
print(f"   Policy: CnnPolicy (Convolutional Neural Network)")

print(f"\nInitial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

try:
    model.learn(
        total_timesteps=500_000,
        tb_log_name=f"SpaceInvaders_Config{CONFIG_ID}_CnnPolicy",
        log_interval=1000,
        progress_bar=True
    )

    print("\nTraining completed successfully!")
    print(f"Max GPU memory used: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"\nOUT OF MEMORY ERROR")
        print(f"Current settings: buffer={model.buffer_size}, batch={cfg['batch']}")
        print(f"\nEMERGENCY FIX:")
        print(f"   1. Change buffer_size=5000")
        print(f"   2. Change batch_size=4")
        print(f"   3. Or use device='cpu' (slower but stable)")
    raise e

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True
)

print(f"\nResults over 10 episodes:")
print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
print(f"   Random Baseline: {avg_random_score:.2f}")
print(f"   Improvement: {mean_reward - avg_random_score:.2f}")

model_name = "dqn_model_2"
model.save(model_name)

backup_name = f"dqn_spaceinvaders_exp{CONFIG_ID}"
model.save(backup_name)

print(f"\nModels saved:")
print(f"   • {model_name}.zip (required submission)")
print(f"   • {backup_name}.zip (backup with config ID)")