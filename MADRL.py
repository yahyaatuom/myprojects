import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- CONFIGURATION & HYPERPARAMETERS ---
NUM_AGENTS = 3                 # Active uncoordinated GF-NOMA devices
POWER_POOL = np.linspace(0.1, 0.9, 9)  # Action Space: [0.1, 0.2, ..., 0.9]
ACTION_SIZE = len(POWER_POOL)
STATE_SIZE = NUM_AGENTS * 2    # Local CSI + Last Power Allocation for all agents
GAMMA = 0.95                   # Discount Factor
ALPHA_SHAPING = 0.1            # Weight for potential-based power penalty
SINR_TH = 1.5                  # Target operational SINR threshold (~1.76 dB)
NOISE_POWER = 1e-2             # Normalized background noise floor (N0)

# --- 1. THE SIMULATED GF-NOMA WIRELESS ENVIRONMENT ---
class GFNomaEnv:
    def __init__(self, num_agents, power_pool):
        self.num_agents = num_agents
        self.power_pool = power_pool
        self.reset()
        
    def reset(self):
        # Generate random Rayleigh fading channel gains (CSI) for the time slot
        self.channels = np.random.exponential(scale=1.0, size=self.num_agents)
        self.last_actions = np.zeros(self.num_agents)
        return self._get_state()
    
    def _get_state(self):
        # State vector: Concatenation of current channels and previous power configurations
        return np.concatenate([self.channels, self.last_actions]).astype(np.float32)
    
    def step(self, actions_indices):
        # Map indices from the agent's output network to physical transmission power values
        p_trans = np.array([self.power_pool[idx] for idx in actions_indices])
        
        # Calculate received power elements at the Base Station
        p_rx = p_trans * self.channels
        
        # Sort users by received power to simulate Successive Interference Cancellation (SIC)
        # Strongest received power is decoded first, experiencing interference from all weaker users
        sorted_idx = np.argsort(p_rx)[::-1] 
        
        sinrs = np.zeros(self.num_agents)
        rewards = np.zeros(self.num_agents)
        
        # Calculate multi-user SINR values according to the NOMA protocol decoding sequence
        for rank, u_idx in enumerate(sorted_idx):
            signal = p_rx[u_idx]
            # Interference consists of signals from all users not yet decoded
            interference = np.sum([p_rx[other] for other in sorted_idx[rank+1:]])
            sinrs[u_idx] = signal / (interference + NOISE_POWER)
            
            # --- CALCULATE UTILITY COMPONENT (Logarithmic Rate Capacity) ---
            rate = np.log2(1 + sinrs[u_idx])
            
            # --- CALCULATE POTENTIAL-BASED REWARD SHAPING GRADIENETS ---
            # Calculate potential function: penalizes severe deviations below the necessary SINR threshold
            phi_current = min(1.0, sinrs[u_idx] / SINR_TH) - ALPHA_SHAPING * (p_trans[u_idx])
            phi_previous = min(1.0, (self.last_actions[u_idx]*self.channels[u_idx]/NOISE_POWER) / SINR_TH) - ALPHA_SHAPING * self.last_actions[u_idx]
            shaping_signal = GAMMA * phi_current - phi_previous
            
            # Aggregate base reward components and penalize system outages 
            outage_penalty = -2.0 if sinrs[u_idx] < SINR_TH else 0.0
            rewards[u_idx] = rate + outage_penalty + shaping_signal

        # Update environment tracking parameters
        self.last_actions = p_trans
        self.channels = np.random.exponential(scale=1.0, size=self.num_agents) # Next-step CSI
        next_state = self._get_state()
        
        return next_state, rewards, sinrs

# --- 2. DEEP Q-NETWORK ARCHITECTURE ---
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size) # Outputs Q-values for each of the 9 power pool index steps
        )
        
    def forward(self, x):
        return self.fc(x)

# --- 3. TRAINING COORDINATOR PIPELINE ---
def train_madrl():
    env = GFNomaEnv(NUM_AGENTS, POWER_POOL)
    agents = [DQNAgent(STATE_SIZE, ACTION_SIZE) for _ in range(NUM_AGENTS)]
    optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in agents]
    memory = deque(maxlen=10000)
    
    epsilon = 1.0       # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.05
    batch_size = 32
    
    for episode in range(100): # Run training sequences
        state = env.reset()
        total_rewards = np.zeros(NUM_AGENTS)
        
        for time_step in range(50):
            actions = []
            # Every device makes an independent decision based on localized tracking observations
            for i in range(NUM_AGENTS):
                if random.random() <= epsilon:
                    actions.append(random.randint(0, ACTION_SIZE - 1))
                else:
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    q_values = agents[i](state_t)
                    actions.append(torch.argmax(q_values).item())
            
            next_state, rewards, sinrs = env.step(actions)
            memory.append((state, actions, rewards, next_state))
            state = next_state
            total_rewards += rewards
            
            # Perform Experience Replay batch optimization
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for i in range(NUM_AGENTS):
                    # Unpack transition samples
                    states_b = torch.FloatTensor([m[0] for m in minibatch])
                    actions_b = torch.LongTensor([m[1][i] for m in minibatch]).unsqueeze(1)
                    rewards_b = torch.FloatTensor([m[2][i] for m in minibatch])
                    next_states_b = torch.FloatTensor([m[3] for m in minibatch])
                    
                    # Calculate current predicted actions Q-values
                    q_eval = agents[i](states_b).gather(1, actions_b).squeeze(1)
                    # Compute standard target labels
                    q_next = agents[i](next_states_b).detach().max(1)[0]
                    q_target = rewards_b + GAMMA * q_next
                    
                    loss = nn.MSELoss()(q_eval, q_target)
                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()
                    
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 10 == 0:
            print(f"Episode: {episode} | Ep-Avg Reward Per Agent: {np.mean(total_rewards)/50:.3f} | Current Epsilon: {epsilon:.2f}")

if __name__ == "__main__":
    train_madrl()
