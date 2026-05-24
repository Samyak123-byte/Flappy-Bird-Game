import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory 
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

# Device setup
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else: 
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")


    def run(self, is_training=True, render=False):
        # Render state handling
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0] # input dim (8 states)
        num_actions = env.action_space.n # output dim (2 actions)

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            # Copy weights from policy to target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
        else:
            # Load best policy for evaluation
            if os.path.exists(self.MODEL_FILE):
                policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            else:
                print(f"Error: Saved model file {self.MODEL_FILE} not found!")
                return
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            episode_reward = 0
            terminated = False

            while (not terminated and episode_reward < self.reward_threshold):
                # Action selection (Epsilon-greedy)
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() # Explore
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        # Fixed dimensional compression issue
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax(dim=-1).squeeze() # Exploit

                next_state, reward, terminated, _, _ = env.step(action.item())
                episode_reward += reward

                # Create tensors
                reward_tensor = torch.tensor(reward, dtype=torch.float, device=device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state_tensor, reward_tensor, terminated))
                    steps += 1

                    # Optimize model if memory has enough experiences
                    if len(memory) > self.mini_batch_size:
                        mini_batch = memory.sample(self.mini_batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Fixed: Sync target network inside the step loop
                    if steps >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

                state = next_state_tensor
                
            print(f"episode={episode+1} with total reward={episode_reward:.2f} & epsilon={epsilon:.4f}")

            if is_training:
                # Epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # Save best weights
                if episode_reward > best_reward:
                    log_msg = f"best reward = {episode_reward:.2f} for episode={episode+1}"
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

        env.close()

    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Unzip batch of experiences safely
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        
        # Cast terminations tuple to tensor safely
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        # Calculate target Q-values
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]
            
        # Calculate current policy prediction
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss and step optimizer
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Name of hyperparameter set from parameters.yaml')
    parser.add_argument('--train', help='Training mode flag', action='store_true')
    parser.add_argument('--render', help='Render game screen during training', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        # Pass render flag dynamically from command line
        dql.run(is_training=True, render=args.render)
    else:
        dql.run(is_training=False, render=True)
