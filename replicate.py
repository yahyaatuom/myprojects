import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION (Table VI from the paper)
# ============================================================
class Config:
    R = 900.0                    # cell radius (m)
    alpha = 3.0                  # path loss exponent
    C_I = 1e-3                   # distance-dependent intercept
    B_s = 10e3                   # sub-channel bandwidth (Hz)
    noise_dBm = -90              # AWGN power (dBm)
    P_max = 1.0                  # max transmit power (W)
    R_th_GB = 15                 # GB required data rate (bps/Hz)
    K = 4                        # number of sub-channels
    L = 4                        # power levels per PP
    N_GF = 12                    # number of GF users
    U_GB = 4                     # one GB per sub-channel
    # Power pools (Table VI)
    PPs = np.array([
        [0.1, 0.2, 0.4, 0.6],
        [0.1, 0.2, 0.3, 0.5],
        [0.1, 0.4, 0.5, 0.6],
        [0.1, 0.3, 0.5, 0.7],
    ])
    # Training
    E = 500                      # episodes
    T = 50                       # time slots per episode
    gamma = 0.9
    lr = 1e-3
    eps_start = 1.0
    eps_min = 0.01
    eps_decay = 0.995
    batch_size = 64
    memory_size = 10000
    target_update = 2000
    rho = 0.5                    # switching threshold


# ============================================================
# CHANNEL + SIC DECODER
# ============================================================
class SGFNOMAEnvironment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.noise_W = 10 ** ((cfg.noise_dBm - 30) / 10)  # dBm -> W
        self.reset_positions()

    def reset_positions(self):
        """Randomly place GB and GF users in the cell."""
        # GB users: one per sub-channel
        r_gb = self.cfg.R * np.sqrt(np.random.rand(self.cfg.U_GB))
        theta_gb = 2 * np.pi * np.random.rand(self.cfg.U_GB)
        self.gb_pos = np.stack([r_gb * np.cos(theta_gb),
                                r_gb * np.sin(theta_gb)], axis=1)

        # GF users
        r_gf = self.cfg.R * np.sqrt(np.random.rand(self.cfg.N_GF))
        theta_gf = 2 * np.pi * np.random.rand(self.cfg.N_GF)
        self.gf_pos = np.stack([r_gf * np.cos(theta_gf),
                                r_gf * np.sin(theta_gf)], axis=1)

    def channel_gain(self, positions):
        """g = C_I * |h|^2 * d^{-alpha}"""
        d = np.linalg.norm(positions, axis=1) + 1e-6
        h = (np.random.randn(len(d)) + 1j * np.random.randn(len(d))) / np.sqrt(2)
        return self.cfg.C_I * np.abs(h) ** 2 * d ** (-self.cfg.alpha)

    def compute_reward_and_throughput(self, pp_choices, power_choices):
        """
        pp_choices:    (N_GF,) in {0..K-1}  -- which sub-channel/PP
        power_choices: (N_GF,) in {0..L-1}  -- which power level
        Returns rewards (N_GF,), throughput (int), collisions (int)
        """
        g_gf = self.channel_gain(self.gf_pos)
        g_gb = self.channel_gain(self.gb_pos)
        rewards = np.zeros(self.cfg.N_GF)
        throughput = 0
        collisions = 0

        # Check GB QoS on each sub-channel
        gb_ok = np.ones(self.cfg.K, dtype=bool)
        for k in range(self.cfg.K):
            # Sum interference from GF users on this sub-channel
            gf_on_k = np.where(pp_choices == k)[0]
            if len(gf_on_k) == 0:
                continue
            p_gf = self.cfg.PPs[k, power_choices[gf_on_k]]
            interference = np.sum(p_gf * g_gf[gf_on_k])
            sinr_gb = (self.cfg.P_max * g_gb[k]) / (interference + self.noise_W)
            rate_gb = self.cfg.B_s * np.log2(1 + sinr_gb)
            if rate_gb < self.cfg.R_th_GB * self.cfg.B_s:
                gb_ok[k] = False

        # Per sub-channel collision detection
        for k in range(self.cfg.K):
            gf_on_k = np.where(pp_choices == k)[0]
            if len(gf_on_k) == 0:
                continue
            power_levels = power_choices[gf_on_k]

            # Count how many users chose each power level
            unique, counts = np.unique(power_levels, return_counts=True)
            collision_levels = unique[counts > 1]

            for idx, n in enumerate(gf_on_k):
                if not gb_ok[k]:
                    rewards[n] = 0
                    continue
                if power_choices[n] in collision_levels:
                    rewards[n] = 0
                    collisions += 1
                else:
                    rewards[n] = 1
                    throughput += 1

        return rewards, throughput, collisions, gb_ok


# ============================================================
# DQN AGENT
# ============================================================
class DQNAgent(nn.Module):
    def __init__(self, state_dim=3, action_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class HybridMADQN:
    """One agent per GF user."""
    def __init__(self, cfg, n_agents):
        self.cfg = cfg
        self.n_agents = n_agents
        self.action_dim = cfg.K * cfg.L  # 16
        self.state_dim = 3

        self.online = [DQNAgent(self.state_dim, self.action_dim) for _ in range(n_agents)]
        self.target = [DQNAgent(self.state_dim, self.action_dim) for _ in range(n_agents)]
        for t, o in zip(self.target, self.online):
            t.load_state_dict(o.state_dict())

        self.optimizers = [optim.Adam(o.parameters(), lr=cfg.lr) for o in self.online]
        self.memories = [deque(maxlen=cfg.memory_size) for _ in range(n_agents)]
        self.eps = cfg.eps_start
        self.step_count = 0

    def select_actions(self, states):
        """states: (n_agents, 3) -> (n_agents,) actions in {0..15}"""
        actions = []
        for i in range(self.n_agents):
            if random.random() < self.eps:
                actions.append(random.randint(0, self.action_dim - 1))
            else:
                with torch.no_grad():
                    s = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0)
                    q = self.online[i](s)
                    actions.append(int(q.argmax().item()))
        return np.array(actions)

    def store(self, i, s, a, r, s_next):
        self.memories[i].append((s, a, r, s_next))

    def train_step(self):
        for i in range(self.n_agents):
            if len(self.memories[i]) < self.cfg.batch_size:
                continue
            batch = random.sample(self.memories[i], self.cfg.batch_size)
            s, a, r, s_next = zip(*batch)
            s = torch.tensor(np.array(s), dtype=torch.float32)
            a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
            r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
            s_next = torch.tensor(np.array(s_next), dtype=torch.float32)

            q = self.online[i](s).gather(1, a)
            with torch.no_grad():
                q_next = self.target[i](s_next).max(1, keepdim=True)[0]
                y = r + self.cfg.gamma * q_next

            loss = nn.functional.mse_loss(q, y)
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

            self.step_count += 1
            if self.step_count % self.cfg.target_update == 0:
                self.target[i].load_state_dict(self.online[i].state_dict())

        self.eps = max(self.cfg.eps_min, self.eps * self.cfg.eps_decay)


# ============================================================
# TRAINING LOOP (Algorithm 1)
# ============================================================
def train(cfg, mode="hybrid"):
    env = SGFNOMAEnvironment(cfg)
    learner = HybridMADQN(cfg, cfg.N_GF)

    history = {"reward": [], "throughput": [], "collisions": []}

    for e in range(cfg.E):
        # Reset user positions each episode
        env.reset_positions()

        # Previous decoding status (3rd state component)
        decode_status = np.zeros(cfg.N_GF)

        ep_reward = 0
        ep_throughput = 0
        ep_collisions = 0

        for t in range(cfg.T):
            # Build states: [last PP, last power, last decode status]
            if t == 0:
                last_pp = np.zeros(cfg.N_GF)
                last_power = np.zeros(cfg.N_GF)
            states = np.stack([last_pp, last_power, decode_status], axis=1)

            actions = learner.select_actions(states)
            pp_choices = actions // cfg.L
            power_choices = actions % cfg.L

            rewards, thr, coll, gb_ok = env.compute_reward_and_throughput(
                pp_choices, power_choices
            )

            # --- HYBRID REWARD LOGIC ---
            if mode == "competitive":
                shaped = rewards
            elif mode == "cooperative":
                global_r = 1.0 if coll == 0 else 0.0
                shaped = np.full(cfg.N_GF, global_r)
            elif mode == "hybrid":
                if e < cfg.rho * cfg.E:
                    shaped = rewards          # competitive phase
                else:
                    global_r = 1.0 if coll == 0 else 0.0
                    shaped = np.full(cfg.N_GF, global_r)  # cooperative
            else:
                shaped = rewards

            # Store transitions
            next_status = (rewards > 0).astype(float)
            next_states = np.stack([pp_choices, power_choices, next_status], axis=1)
            for i in range(cfg.N_GF):
                learner.store(i, states[i], actions[i], shaped[i], next_states[i])

            learner.train_step()

            last_pp = pp_choices
            last_power = power_choices
            decode_status = next_status
            ep_reward += shaped.mean()
            ep_throughput += thr
            ep_collisions += coll

        history["reward"].append(ep_reward / cfg.T)
        history["throughput"].append(ep_throughput / cfg.T)
        history["collisions"].append(ep_collisions / cfg.T)

        if (e + 1) % 50 == 0:
            print(f"[{mode}] Ep {e+1}/{cfg.E} | "
                  f"Reward {history['reward'][-1]:.2f} | "
                  f"Thr {history['throughput'][-1]:.2f} | "
                  f"Coll {history['collisions'][-1]:.2f}")

    return history


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    cfg = Config()
    results = {}
    for mode in ["competitive", "cooperative", "hybrid"]:
        print(f"\n=== Training {mode} ===")
        results[mode] = train(cfg, mode=mode)

    # Plot convergence (Fig. 5a)
    plt.figure(figsize=(8, 5))
    for mode, h in results.items():
        plt.plot(h["reward"], label=f"{mode.capitalize()} MA-DRL")
    plt.xlabel("No. of Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.title("Reward Comparison (reproduction of Fig. 5a)")
    plt.savefig("fig5a_reproduction.png", dpi=150)
    plt.show()