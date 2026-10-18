import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class RCActor(nn.Module):
    """
    Decentralized Recurrent Actor: Processes local observations and maintains 
    a hidden state (RNN memory) to handle partial observability.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(RCActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, hidden_state):
        # obs shape: (batch_size, obs_dim)
        # hidden_state shape: (batch_size, hidden_dim)
        x = torch.relu(self.fc1(obs))
        h_next = self.rnn(x, hidden_state)
        logits = self.action_head(h_next)
        dist = Categorical(logits=logits)
        return dist, h_next


class RCCritic(nn.Module):
    """
    Centralized Recurrent Critic: Takes the global state (or concatenation of all 
    agents' observations) to calculate a unified value estimate during training.
    """
    def __init__(self, global_state_dim, hidden_dim=64):
        super(RCCritic, self).__init__()
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, global_state, hidden_state):
        # global_state shape: (batch_size, global_state_dim)
        x = torch.relu(self.fc1(global_state))
        h_next = self.rnn(x, hidden_state)
        value = self.value_head(h_next)
        return value, h_next


class RCMAPPOTrainer:
    """
    A simplified single-step update execution handler for RC-MAPPO.
    """
    def __init__(self, obs_dim, global_state_dim, action_dim, num_agents, lr=3e-4, clip_eps=0.2):
        self.num_agents = num_agents
        self.clip_eps = clip_eps
        
        # In homogeneous MAPPO, agents often share policy parameters to boost sample efficiency
        self.actor = RCActor(obs_dim, action_dim)
        self.critic = RCCritic(global_state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def train_step(self, batch_data):
        """
        Executes a single optimization step using gathered batch data.
        """
        # Unpack mock replay buffer steps
        obs = batch_data["obs"]                  # (batch, num_agents, obs_dim)
        global_state = batch_data["state"]      # (batch, global_state_dim)
        actions = batch_data["actions"]          # (batch, num_agents)
        old_log_probs = batch_data["log_probs"]  # (batch, num_agents)
        returns = batch_data["returns"]          # (batch, 1)
        advantages = batch_data["advantages"]    # (batch, 1)
        
        # Initialize hidden states for RNN units
        batch_size = obs.shape[0]
        actor_h = torch.zeros(batch_size, 64)
        critic_h = torch.zeros(batch_size, 64)
        
        # 1. Update Centralized Critic
        values, _ = self.critic(global_state, critic_h)
        critic_loss = nn.MSELoss()(values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 2. Update Decentralized Actor (Iterating through agents)
        actor_loss_total = 0
        for i in range(self.num_agents):
            agent_obs = obs[:, i, :]
            agent_actions = actions[:, i]
            agent_old_log_probs = old_log_probs[:, i]
            
            # Forward pass through recurrent policy
            dist, _ = self.actor(agent_obs, actor_h)
            new_log_probs = dist.log_prob(agent_actions)
            entropy = dist.entropy().mean()
            
            # Calculate PPO Clipped Objective
            ratios = torch.exp(new_log_probs - agent_old_log_probs)
            surr1 = ratios * advantages.squeeze(-1)
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages.squeeze(-1)
            
            # Policy loss with entropy regularization for exploration
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            actor_loss_total += actor_loss
            
        self.actor_optimizer.zero_grad()
        actor_loss_total.backward()
        self.actor_optimizer.step()
        
        return actor_loss_total.item(), critic_loss.item()

# --- Execution Example ---
if __name__ == "__main__":
    # Define arbitrary dimension sizes
    N_AGENTS = 3
    OBS_DIM = 8
    STATE_DIM = N_AGENTS * OBS_DIM  # Centralized state is a concat of local observations
    ACTION_DIM = 4                 # Discrete actions (e.g., Up, Down, Left, Right)
    BATCH_SIZE = 16

    # Instantiate Trainer
    mappo = RCMAPPOTrainer(OBS_DIM, STATE_DIM, ACTION_DIM, N_AGENTS)

    # Generate synthetic tensor inputs mimicking an environment batch step
    mock_batch = {
        "obs": torch.randn(BATCH_SIZE, N_AGENTS, OBS_DIM),
        "state": torch.randn(BATCH_SIZE, STATE_DIM),
        "actions": torch.randint(0, ACTION_DIM, (BATCH_SIZE, N_AGENTS)),
        "log_probs": torch.randn(BATCH_SIZE, N_AGENTS),
        "returns": torch.randn(BATCH_SIZE, 1),
        "advantages": torch.randn(BATCH_SIZE, 1)
    }

    # Evaluate a trial gradient step
    a_loss, c_loss = mappo.train_step(mock_batch)
    print(f"Training Step Successful!\nActor Loss: {a_loss:.4f} | Critic Loss: {c_loss:.4f}")