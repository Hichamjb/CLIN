import sys
import random
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.oncomarl_env_v2 import OncoMARLEnv, ACTIONS


# ============================================================
# Configuration
# ============================================================

SEED = 42
N_EPISODES = 1500
MAX_STEPS = 12

GAMMA = 0.95
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

BUFFER_SIZE = 50_000
MIN_BUFFER_SIZE = 1_000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE_EVERY = 250

MODEL_DIR = ROOT / "models" / "oncomarl_v2"
REPORT_DIR = ROOT / "reports" / "oncomarl_v2"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N_ACTIONS = len(ACTIONS)
STATE_DIM = 5

print("Device:", DEVICE)
print("State dimension:", STATE_DIM)
print("Number of actions:", N_ACTIONS)


# ============================================================
# Q-network
# ============================================================

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(
                np.stack(states),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                actions,
                dtype=torch.long,
                device=DEVICE,
            ),
            torch.tensor(
                rewards,
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                np.stack(next_states),
                dtype=torch.float32,
                device=DEVICE,
            ),
            torch.tensor(
                dones,
                dtype=torch.float32,
                device=DEVICE,
            ),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Double DQN
# ============================================================

policy_net = QNetwork(STATE_DIM, N_ACTIONS).to(DEVICE)
target_net = QNetwork(STATE_DIM, N_ACTIONS).to(DEVICE)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(
    policy_net.parameters(),
    lr=LEARNING_RATE,
)

replay_buffer = ReplayBuffer(BUFFER_SIZE)

epsilon = EPSILON_START
global_step = 0

episode_rewards = []
episode_losses = []


def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)

    with torch.no_grad():
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        q_values = policy_net(state_tensor)

        return int(q_values.argmax(dim=1).item())


def optimize_model():
    if len(replay_buffer) < max(
        BATCH_SIZE, MIN_BUFFER_SIZE
    ):
        return None

    states, actions, rewards, next_states, dones = (
        replay_buffer.sample(BATCH_SIZE)
    )

    # Q(s, a)
    current_q = policy_net(states).gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)

    # Double DQN:
    # online network selects the next action.
    # target network evaluates that action.
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(
            dim=1, keepdim=True
        )

        next_q = target_net(next_states).gather(
            1, next_actions
        ).squeeze(1)

        # Terminal transitions do not bootstrap.
        targets = rewards + GAMMA * (1.0 - dones) * next_q

    loss = nn.functional.smooth_l1_loss(
        current_q, targets
    )

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(
        policy_net.parameters(), max_norm=10.0
    )

    optimizer.step()

    return float(loss.item())


# ============================================================
# Training loop
# ============================================================

for episode in range(1, N_EPISODES + 1):

    env = OncoMARLEnv(
        seed=SEED + episode
    )

    state = env.reset(seed=SEED + episode)

    total_reward = 0.0
    losses = []

    for step in range(MAX_STEPS):

        action = choose_action(state, epsilon)

        next_state, reward, terminated, info = env.step(
            action
        )

        replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            terminated,
        )

        loss = optimize_model()

        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward
        global_step += 1

        if global_step % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(
                policy_net.state_dict()
            )

        if terminated:
            break

    epsilon = max(
        EPSILON_END,
        epsilon * EPSILON_DECAY,
    )

    episode_rewards.append(total_reward)

    mean_loss = (
        float(np.mean(losses)) if losses else float("nan")
    )

    episode_losses.append(mean_loss)

    if episode % 50 == 0:
        recent_rewards = episode_rewards[-50:]

        print(
            f"Episode {episode:4d}/{N_EPISODES} | "
            f"Reward50={np.mean(recent_rewards):.4f} | "
            f"Loss={mean_loss:.6f} | "
            f"Epsilon={epsilon:.3f} | "
            f"Buffer={len(replay_buffer)}"
        )


# ============================================================
# Save model and training history
# ============================================================

checkpoint_path = MODEL_DIR / "double_dqn_v2.pt"

torch.save(
    {
        "model_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "state_dim": STATE_DIM,
        "action_dim": N_ACTIONS,
        "gamma": GAMMA,
        "seed": SEED,
        "dataset_type": "SYNTHETIC",
        "algorithm": "Double DQN",
    },
    checkpoint_path,
)

history_path = REPORT_DIR / "training_history.csv"

with history_path.open("w", encoding="utf-8") as f:
    f.write("episode,reward,loss\n")

    for i, (reward, loss) in enumerate(
        zip(episode_rewards, episode_losses), start=1
    ):
        f.write(f"{i},{reward},{loss}\n")


print("\nTraining complete.")
print("Checkpoint:", checkpoint_path)
print("History:", history_path)
print("Final epsilon:", epsilon)
print(
    "Mean reward over last 100 episodes:",
    round(float(np.mean(episode_rewards[-100:])), 4),
)