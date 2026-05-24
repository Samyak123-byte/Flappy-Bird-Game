Flappy Bird AI Agent using Deep Q-Network (DQN)
This repository contains an autonomous artificial intelligence agent trained to play Flappy Bird using Deep Reinforcement Learning (DRL). The project is built using Python, Pygame, and modern Reinforcement Learning architectures (DQN/Q-learning). 

GitHub
The agent learns optimal survival strategies strictly through trial and error, adjusting its actions based on score rewards and crash penalties until it masters the game environment.
🚀 Key Features
Complete DRL Architecture: Features a custom Deep Q-Network (DQN) built to handle continuous game states.
Experience Replay: Implements a replay buffer mechanism to break state correlation, stabilizing training.
Dynamic Configuration: Tweak all hyperparameters seamlessly using a dedicated configuration file.
Performance Visualizations: Automated reward tracking and metric graphs to visualize model convergence over time.
Modular Pipeline: Clean decoupling of game engine logic, network architecture, and agent training scripts. 

GitHub
 +1
🛠️ Project Structure
bash
📂 Flappy-Bird-Game
├── 📄 agent.py               # Coordinates the RL pipeline, training steps, and action selection
├── 📄 dqn.py                 # Neural Network architecture modeling the Q-value function
├── 📄 experience_replay.py   # Replay memory buffer to store and sample state transitions
├── 📄 game_flappy_bird.py    # Custom Pygame-based Flappy Bird environment clone
├── 📄 play.py                # Wrapper script to execute and evaluate fully trained models
├── 📄 parameters.yaml        # Central configuration file for all training parameters
├── 📄 .gitignore             # Standard git exclusion configurations
└── 📄 README.md              # Project documentation and details
Use code with caution.
🧠 Reinforcement Learning Workflow
The agent learns by interacting continuously with the Pygame window environment:
   +-------------------------------------------------+

   |                                                 |
   v                                                 |
[State (s)] ---> [ Agent / DQN ] ---> [Action (a)] --+
                    ^                   |
                    |                   v
                    +--- [ Reward (r) ] + [Next State (s')]
1. State Space (
)
Instead of parsing raw pixels, the state space relies on highly optimized continuous spatial metrics: 

GitHub
 +1
The horizontal distance to the next pair of incoming pipes.
The vertical distance relative to the upper and lower pipe gaps.
The instantaneous vertical velocity of the bird. 

Reddit
·eccentric code
 +2
2. Action Space (
)
At every single game frame, the agent chooses between two distinct choices: 

GitHub
 +1
0: Do nothing (allow gravity to pull the bird downward).
1: Flap (apply upward impulse velocity). 

Reddit
·eccentric code
 +2
3. Reward Function (
)
The behavior of the agent is heavily guided by a carefully balanced feedback loop:
Survival / Progression: 
 (or custom positive score) for passing through a pipe opening.
Terminal Crash: 
 (or a severe negative penalty) upon colliding with pipes or hitting the ground. 

TIJER
⚙️ Configuration & Hyperparameters
All execution configs are fully managed inside parameters.yaml. Key variables include:
learning_rate (
): Step size adjustments for network optimization gradient descents.
discount_factor (
): Quantifies how much the agent prioritizes long-term prospective scores over short-term survival rewards.
epsilon (
-greedy): Balances exploration (random steps) and exploitation (best-known actions).
batch_size: The number of transition samples pulled from replay memory during weights optimization.
