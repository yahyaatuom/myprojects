import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


class IndustrialURLLCEnv(ParallelEnv):
    """
    Custom Parallel PettingZoo environment simulating an Indoor Factory (InF)
    RF environment with heavy multipath fading, metal racks, and impulsive noise.
    Designed for homogeneous AMR (autonomous mobile robot) resource allocation.
    """

    metadata = {"name": "industrial_urllc_v0", "render_modes": []}

    def __init__(self, num_agents=4, num_subcarriers=12, max_steps=200):
        super().__init__()
        self.possible_agents = [f"amr_{i}" for i in range(num_agents)]
        self.num_subcarriers = num_subcarriers
        self.max_steps = max_steps

        # Room dimensions (meters)
        self.room_width = 100.0
        self.room_length = 60.0

        # High-density metal racks: [x_min, y_min, x_max, y_max]
        self.metal_racks = np.array([
            [10, 10, 20, 15],
            [30, 20, 45, 25],
            [60, 10, 75, 15],
            [50, 25, 80, 30],
        ], dtype=float)

        # Access point (gNodeB) static location at center ceiling
        self.ap_pos = np.array([50.0, 30.0, 6.0], dtype=float)

        # Action space per agent: pick a subcarrier
        self.action_spaces = {
            agent: Discrete(self.num_subcarriers) for agent in self.possible_agents
        }

        # Observation: [SINR, queue, latency, x, y]
        low = np.array([-30.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array(
            [40.0, 100.0, 50.0, self.room_width, self.room_length],
            dtype=np.float32,
        )
        self.observation_spaces = {
            agent: Box(low=low, high=high, dtype=np.float32)
            for agent in self.possible_agents
        }

        # Runtime state (populated on reset)
        self.agents = []
        self.steps = 0
        self.agent_states = {}

    # ------------------------------------------------------------------ #
    # PettingZoo API                                                     #
    # ------------------------------------------------------------------ #
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.steps = 0
        self.agent_states = {}

        for agent in self.agents:
            self.agent_states[agent] = {
                "pos": np.array(
                    [
                        np.random.uniform(0, self.room_width),
                        np.random.uniform(0, self.room_length),
                    ],
                    dtype=float,
                ),
                "queue": float(np.random.randint(5, 20)),
                "latency": 0.0,
            }

        observations = self._get_all_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self.steps += 1
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # ---- 1. Compute per-agent SINR (pre-conflict) ----------------- #
        sinrs = {
            agent: self._compute_stochastic_sinr(agent) for agent in self.agents
        }

        # ---- 2. Resolve inter-agent subcarrier conflicts -------------- #
        # Count how many agents picked each subcarrier, then subtract a
        # fixed interference penalty for each additional sharer.
        subcarrier_users = {}
        for agent, act in actions.items():
            subcarrier_users.setdefault(int(act), []).append(agent)

        effective_sinr = {}
        for agent in self.agents:
            act = int(actions[agent])
            conflicts = len(subcarrier_users[act]) - 1
            effective_sinr[agent] = sinrs[agent] - conflicts * 6.0

        # ---- 3. Update state and compute rewards ---------------------- #
        for agent in self.agents:
            state = self.agent_states[agent]
            sinr = effective_sinr[agent]

            # Queue / latency dynamics based on effective SINR
            if sinr > 10.0:
                state["queue"] = max(0.0, state["queue"] - 2)
                state["latency"] = max(0.0, state["latency"] - 0.2)
            else:
                state["queue"] += float(np.random.randint(1, 3))
                state["latency"] += 0.5  # accumulate delay

            # Mobility (random walk, clipped to room bounds)
            state["pos"] += np.random.uniform(-1.0, 1.0, size=2)
            state["pos"] = np.clip(
                state["pos"], [0.0, 0.0], [self.room_width, self.room_length]
            )

            # URLLC reward: heavily penalize latency/outage breaches
            latency_penalty = -50.0 if state["latency"] > 1.0 else 5.0
            outage_penalty = -100.0 if state["queue"] > 50 else 0.0
            rewards[agent] = latency_penalty + outage_penalty + sinr * 0.1

            terminations[agent] = False
            truncations[agent] = self.steps >= self.max_steps
            infos[agent] = {
                "sinr": sinr,
                "conflicts": len(subcarrier_users[int(actions[agent])]) - 1,
            }

        # Standard PettingZoo pattern: clear agents only when episode ends
        if self.steps >= self.max_steps:
            self.agents = []

        observations = self._get_all_observations() if self.agents else {
            a: np.zeros(self.observation_space(a).shape, dtype=np.float32)
            for a in self.possible_agents
        }
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        return None

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _get_all_observations(self):
        obs = {}
        for agent in self.agents:
            sinr = self._compute_stochastic_sinr(agent)
            state = self.agent_states[agent]
            obs[agent] = np.array(
                [
                    sinr,
                    state["queue"],
                    state["latency"],
                    state["pos"][0],
                    state["pos"][1],
                ],
                dtype=np.float32,
            )
        return obs

    def _compute_stochastic_sinr(self, agent_id):
        """
        3GPP-like Indoor Factory (InF) path loss + Rician/Rayleigh fading +
        metal-rack shadowing + heavy-tailed impulsive industrial noise.
        """
        pos_2d = self.agent_states[agent_id]["pos"]
        pos_3d = np.array([pos_2d[0], pos_2d[1], 1.0])  # AMR antenna height 1 m

        distance = float(np.linalg.norm(pos_3d - self.ap_pos))
        distance = max(distance, 1.0)  # avoid log10(0)

        # 1. Simplified 3GPP InF-DH path loss at ~3.5 GHz
        path_loss = 31.84 + 21.50 * np.log10(distance) + 19.0 * np.log10(3.5)

        # 2. Shadowing: LOS intersection test against axis-aligned racks.
        #    A simple AABB-vs-segment check is overkill here; keep the
        #    bounding-box overlap used in the original, but note the caveat.
        los_blocked = False
        x1, x2 = sorted((pos_2d[0], self.ap_pos[0]))
        y1, y2 = sorted((pos_2d[1], self.ap_pos[1]))
        for rack in self.metal_racks:
            if (
                x1 <= rack[2]
                and x2 >= rack[0]
                and y1 <= rack[3]
                and y2 >= rack[1]
            ):
                los_blocked = True
                break

        shadowing = np.random.normal(0, 10.0 if los_blocked else 4.0)

        # 3. Fast fading: Rician for LOS, Rayleigh for NLOS
        if not los_blocked:
            k_factor = 3.0
            s = np.sqrt(k_factor / (k_factor + 1))
            sigma = np.sqrt(1 / (2 * (k_factor + 1)))
            fast_fading = 20 * np.log10(
                np.abs(
                    np.random.normal(s, sigma)
                    + 1j * np.random.normal(0, sigma)
                )
            )
        else:
            fast_fading = 20 * np.log10(
                np.abs(
                    np.random.normal(0, 0.707)
                    + 1j * np.random.normal(0, 0.707)
                )
            )

        # 4. Impulsive industrial noise injection
        is_impulsive_burst = np.random.rand() < 0.15
        noise_floor = -104.0 + (
            np.random.exponential(15.0) if is_impulsive_burst else 0.0
        )

        tx_power = 23.0  # dBm
        received_power = tx_power - path_loss - shadowing + fast_fading
        sinr = received_power - noise_floor
        return float(sinr)


class RecurrentHomogeneousActorCritic(nn.Module):
    """
    Homogeneous actor-critic with parameter sharing and a recurrent (GRU)
    core to handle partially observable fading-channel dynamics.

    For strict CTDE/MAPPO, expand `critic_input_dim` to take the concatenated
    global state and call `critic_head(global_features)` instead of the
    per-agent recurrent features.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs, hidden_state):
        features = self.feature_extractor(obs)
        next_hidden = self.rnn(features, hidden_state)
        action_logits = self.actor_head(next_hidden)
        state_value = self.critic_head(next_hidden)
        return action_logits, state_value, next_hidden



if __name__ == "__main__":
    print("=== IndustrialURLLCEnv smoke test ===")
    env = IndustrialURLLCEnv(num_agents=4, num_subcarriers=12, max_steps=20)

    obs, infos = env.reset(seed=42)
    print(f"Agents: {env.agents}")
    print(f"Sample obs for amr_0: {obs['amr_0']}")

    model = RecurrentHomogeneousActorCritic(obs_dim=5, action_dim=12)
    hidden = model.init_hidden(batch_size=4)
    print(f"Model hidden shape: {hidden.shape}")

    total_reward = 0.0
    for t in range(env.max_steps):
        # Dummy random policy for the smoke test
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        total_reward += sum(rewards.values())

        if t == 0:
            print(f"Step 0 rewards: {rewards}")
            print(f"Step 0 infos[amr_0]: {infos['amr_0']}")

        if not env.agents:  # episode ended
            print(f"Episode ended at step {t + 1}")
            break

    print(f"Total cumulative reward: {total_reward:.2f}")
    env.close()
    print("=== Done ===")