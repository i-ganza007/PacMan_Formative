
# Space_Invaders — DQN with Stable Baselines3 and Gymnasium

Overview
--------
This repository contains an assignment implementation and scaffolding to train and evaluate a Deep Q-Network (DQN) agent on an Atari environment using Stable Baselines3 and Gymnasium. The goal is to train the agent to play an Atari game (Space Invaders) and then evaluate its performance by running the trained agent in the same environment.

This project includes:
- train.py — training script to train a DQN agent and save the policy.
- play.py — script to load a saved model and run/play episodes to visualize performance.
- A hyperparameter tuning table template to record experiments and observed behaviors.


Requirements
------------
- Python 3.8+
- PyTorch (version compatible with Stable Baselines3)
- Stable Baselines3
- Gymnasium with Atari support
- ale-py and ROMs (autorom can be used to install ROMs)

Quick install (example):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch stable-baselines3 gymnasium[atari] ale-py autorom[accept-rom-license]
python -m autorom --accept-license --install-all
```

Files
-----
- `train.py` — Train a DQN agent, compare MlpPolicy vs CnnPolicy, tune hyperparameters, save model as `dqn_model.zip`, and log metrics (reward trends, episode length).
- `play.py` — Load the saved model and run a number of episodes to visualize the agent playing the environment. Use greedy (deterministic) policy during evaluation to pick the highest-Q actions.

Training (train.py)
-------------------
Main responsibilities for train.py:
1. Create and wrap the Gymnasium Atari environment (use Monitor for logging).
2. Define a DQN agent with Stable Baselines3:
   - Compare MlpPolicy and CnnPolicy (CNNs often perform much better on raw pixel Atari).
3. Train the agent for the desired number of timesteps.
4. Save the trained model as `dqn_model.zip`.
5. Log training metrics:
   - Episode reward trends
   - Episode length

Example usage:
```bash
python train.py \
  --env "ALE/SpaceInvaders-v5" \
  --policy "CnnPolicy" \
  --timesteps 100000 \
  --output models/dqn_model.zip
```


Hyperparameter Tuning
---------------------
Each group member ran at least 10 experiments (10 different combinations of hyperparameters). 

Suggested hyperparameters to vary:
- lr (learning rate)
- gamma (discount factor)
- batch_size
- epsilon parameters (epsilon_start, epsilon_end, epsilon_decay)

Use the table below in your README to report your experiments.

Hyperparameter tuning table (template)
|  Ian Ganza | Hyperparameter Set | Noted Behavior |
|-------------|--------------------|----------------|
| CNNPOLICY_1 | lr=0.0001, gamma=0.99, batch=8, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.3 | Observations (Mean Reward: 275.00 ± 182.28 , Random Baseline: 101.50 , Improvement: 173.50 ) |
| CNNPOLICY_2| lr=0.0005, gamma=0.95, batch=16, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.2 | Observations (Mean Reward: 407.00 ± 247.00 , Random Baseline: 143.75 , Improvement: 263.25)|
| CNNPOLICY_3 | lr=0.0002, gamma=0.99, batch=8, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.4 | Observations (Mean Reward: 259.50 ± 85.19, Random Baseline: 130.75 , Improvement: 128.75) |
| CNNPOLICY_4 | lr=0.0003, gamma=0.999, batch=12, epsilon_start=1.0, epsilon_end=0.08, epsilon_decay=0.25 | Observations (Mean Reward: 285.00 ± 0.00,Random Baseline: 124.00, Improvement: 161.00) |
| CNNPOLICY_5 | lr=0.00015, gamma=0.98, batch=8, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.35 | Observations (Mean Reward: 227.00 ± 46.05, Random Baseline: 142.50, Improvement: 84.50) |
| MLPolicy_1 | lr=0.0001, gamma=0.99, batch=8, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.3 | Observations (Mean Reward: 50.00 ± 0.00, Random Baseline: 181.00,Improvement: -131.00 ) |
| MLPolicy_2 | lr=0.0005, gamma=0.95, batch=16, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.2 | Observations (Mean Reward: 236.00 ± 33.23, Random Baseline: 148.75,Improvement: 87.25)| |
| MLPolicy_3 | lr=0.0002, gamma=0.99, batch=8, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.4 | Observations (Mean Reward: 285.00 ± 0.00, Random Baseline: 174.00,Improvement: 111.00) |
| MLPolicy_4 | lr=0.0003, gamma=0.999, batch=12, epsilon_start=1.0, epsilon_end=0.08, epsilon_decay=0.25 | Observations (Mean Reward: 285.00 ± 0.00,Random Baseline: 124.00, Improvement: 161.00) |
| MLPolicy_5 | lr=0.00015, gamma=0.98, batch=8, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.35 | Observations (Mean Reward: 227.00 ± 46.05, Random Baseline: 142.50, Improvement: 84.50) |




MEMBER NAME	Policy	Learning Rate	Gamma	Batch Size	Eps Start	Eps End	Eps Decay	Mean Reward	Std Dev	Random Baseline	Improvement	Noted Behavior
Nhial Majok	CNNPolicy	0.0001	0.990	8	1.0	0.05	0.30	275.00	182.28	101.50	173.50	Observations (Mean Reward: 275.00 ± 182.28, Random Baseline: 101.50, Improvement: 173.50). Analysis: Strong improvement but high variance (±182.28). The small batch size (8) combined with a moderate learning rate results in an unstable but ultimately successful learning trajectory.
Nhial Majok	CNNPolicy	0.0005	0.950	16	1.0	0.10	0.20	407.00	247.00	143.75	263.25	Observations (Mean Reward: 407.00 ± 247.00, Random Baseline: 143.75, Improvement: 263.25). Analysis: Achieved the highest mean reward but suffered the highest variance (±247.00). The high learning rate (lr=0.0005) likely caused aggressive updates, resulting in high potential but poor stability.
Nhial Majok	CNNPolicy	0.0002	0.990	8	1.0	0.01	0.40	259.50	85.19	130.75	128.75	Observations (Mean Reward: 259.50 ± 85.19, Random Baseline: 130.75, Improvement: 128.75). Analysis: Moderate stability and performance. The high γ (0.99) promotes good long-term planning, but the small batch size (8) prevents convergence to a perfectly stable policy.
Nhial Majok	CNNPolicy	0.0003	0.999	12	1.0	0.08	0.25	285.00	0.00	124.00	161.00	Observations (Mean Reward: 285.00 ± 0.00, Random Baseline: 124.00, Improvement: 161.00). Analysis: Excellent stability (±0.00). The near-perfect discount factor (γ=0.999) and balanced learning rate result in a reliable, robust policy.
Nhial Majok	CNNPolicy	0.00015	0.980	8	1.0	0.02	0.35	227.00	46.05	142.50	84.50	Observations (Mean Reward: 227.00 ± 46.05, Random Baseline: 142.50, Improvement: 84.50). Analysis: Solid improvement, but slightly limited by the conservative learning rate and lower γ.
Nhial Majok	MLPPolicy	0.0001	0.990	8	1.0	0.05	0.30	50.00	0.00	181.00	-131.00	Observations (Mean Reward: 50.00 ± 0.00, Random Baseline: 181.00, Improvement: -131.00). Analysis: Complete failure. The MLP architecture is unable to process the spatial features of Atari pixel data, leading to performance significantly worse than a random agent.
Nhial Majok	MLPPolicy	0.0005	0.950	16	1.0	0.10	0.20	236.00	33.23	148.75	87.25	Observations (Mean Reward: 236.00 ± 33.23, Random Baseline: 148.75, Improvement: 87.25). Analysis: Surprisingly successful for an MLP. The larger batch size (16) provides stability, allowing the higher learning rate (lr=0.0005) to converge to a decent local maximum.
Nhial Majok	MLPPolicy	0.0002	0.990	8	1.0	0.01	0.40	285.00	0.00	174.00	111.00	Observations (Mean Reward: 285.00 ± 0.00, Random Baseline: 174.00, Improvement: 111.00). Analysis: Perfect stability (±0.00). The combination of γ=0.99 and a low ϵend ensures the agent exploits its learned policy consistently.
Nhial Majok	MLPPolicy	0.0003	0.999	12	1.0	0.08	0.25	285.00	0.00	124.00	161.00	Observations (Mean Reward: 285.00 ± 0.00, Random Baseline: 124.00, Improvement: 161.00). Analysis: Highest MLP improvement and perfect stability. The high γ (0.999) is the key stabilizing factor.
Nhial Majok	MLPPolicy	0.00015	0.980	8	1.0	0.02	0.35	227.00	46.05	142.50	84.50	Observations (Mean Reward: 227.00 ± 46.05, Random Baseline: 142.50, Improvement: 84.50). Analysis: Matches the CNN result, showing that the MLP's failure is not universal, but is dependent on the hyperparameter set.


Playing / Evaluation (play.py)
------------------------------
Main responsibilities for play.py:
1. Load trained model:
   ```python
   from stable_baselines3 import DQN
   model = DQN.load("models/dqn_model.zip")
   ```
2. Create the same environment used for training.
3. Run a number of episodes using a greedy policy to select actions (deterministic=True) to maximize performance.
   - If your SB3 API supports a GreedyQPolicy helper, you may use it; otherwise use `model.predict(obs, deterministic=True)` to select actions greedily.
4. Render the environment to visualize the agent:
   ```python
   obs = env.reset()
   action, _ = model.predict(obs, deterministic=True)
   obs, reward, terminated, truncated, info = env.step(action)
   env.render()
   ```
5. Optionally record a video or save rendered frames.

Example usage:
```bash
python play.py --model models/dqn_model.zip --env "ALE/SpaceInvaders-v5" --episodes 5
```

Logging and Visualization
-------------------------
- Use Monitor wrapper to log episode rewards and lengths to CSV for plotting.
- TensorBoard: you can configure Stable Baselines3 to log to TensorBoard for easy visualization of loss, reward, and other metrics.
- Save checkpoints of models periodically during training (use callbacks).


Video Playing
-----------------------
https://drive.google.com/file/d/1X3I3OAWVz4nVx5YXdEe3eLiK44Jp1-7J/view?usp=sharing

Tips for Reproducibility
------------------------
- Fix random seeds for environment, numpy, torch, and Stable Baselines3 to help with reproducibility.
- Record the exact versions of Python and key packages in a `requirements.txt` or `environment.yml`.
- Keep a log of experiments (hyperparameters, random seed, total timesteps, and a short note about performance).

Example requirements.txt (start point)
```
torch>=1.9
stable-baselines3>=2.0.0
gymnasium[atari]
ale-py
autorom[accept-rom-license]
numpy
opencv-python
tensorboard
```

Contact
-------
Repository owner: i-ganza007

License
-------
You can add a license of your choice. (Suggested: MIT)

---
Good luck training your DQN agent! Follow the templates above to run experiments, compare MLP vs CNN policies, and document the hyperparameter tuning results with clear plots and a short video showing agent behavior.
```
