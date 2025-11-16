

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import numpy as np
import time

MODEL_PATH = "Ian_Model\dqn_spaceinvaders_exp2.zip"  
NUM_EPISODES = 5              
RENDER_MODE = "human"         

print(f"\nLoading trained model from: {MODEL_PATH}")

try:
    # Load the model (this loads the best trained model)
    model = DQN.load(MODEL_PATH)
    print("Model loaded successfully!")
    print(f"   Policy type: {model.policy.__class__.__name__}")
    
except FileNotFoundError:
    print(f" ERROR: Model file not found at {MODEL_PATH}")
    print("   Make sure you've trained the model first using train.py")
    print("   Available models in current directory:")
    import glob
    models = glob.glob("*.zip")
    for m in models:
        print(f"   - {m}")
    exit(1)

env = gym.make("ALE/SpaceInvaders-v5", render_mode=RENDER_MODE, frameskip=4)

env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)




episode_rewards = []
episode_lengths = []

for episode in range(1, NUM_EPISODES + 1):
    print(f"Episode {episode}/{NUM_EPISODES}")

    
    # Reset environment
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    while not done:

        action, _states = model.predict(obs, deterministic=True)
        
        # Take action in environment
        obs, reward, done, info = env.step(action)
        
        episode_reward += reward[0]
        episode_length += 1

        if RENDER_MODE == "human":
            time.sleep(0.01) 
    
    # Episode finished
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    
    print(f"\nEpisode {episode} Results:")
    print(f"   Total Reward: {episode_reward}")
    print(f"   Episode Length: {episode_length} steps")
    print(f"   Average Reward per Step: {episode_reward/episode_length:.2f}")


mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
min_reward = np.min(episode_rewards)
max_reward = np.max(episode_rewards)
mean_length = np.mean(episode_lengths)

print(f"\nReward Statistics ({NUM_EPISODES} episodes):")
print(f"  Mean Reward:   {mean_reward:.2f}")
print(f"  Std Deviation: {std_reward:.2f}")
print(f"  Min Reward:    {min_reward}")
print(f"  Max Reward:    {max_reward}")
print(f"\nEpisode Statistics:")
print(f"  Mean Length:   {mean_length:.1f} steps")
print(f"  Total Steps:   {sum(episode_lengths)}")

print(" GAMEPLAY COMPLETE!")

env.close()

