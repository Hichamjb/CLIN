import sys
import csv
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.oncomarl_env_v2 import OncoMARLEnv, ACTIONS


# ============================================================
# Configuration
# ============================================================

SEED = 2026
N_EPISODES = 500
MAX_STEPS = 12

DEVICE = torch.device("cpu")

MODEL_PATH = (
    ROOT / "models" / "oncomarl_v2" / "double_dqn_v2.pt"
)

OUTPUT_DIR = ROOT / "reports" / "oncomarl_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Q-network: must match the training architecture
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
# Load checkpoint
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

assert checkpoint["state_dim"] == 5
assert checkpoint["action_dim"] == len(ACTIONS)

model = QNetwork(
    checkpoint["state_dim"],
    checkpoint["action_dim"],
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded:", MODEL_PATH)
print("Algorithm:", checkpoint["algorithm"])
print("Dataset type:", checkpoint["dataset_type"])


# ============================================================
# Policies
# ============================================================

def dqn_policy(state):
    with torch.no_grad():
        x = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        q_values = model(x)

        return int(q_values.argmax(dim=1).item())


def random_policy(state):
    return random.randrange(len(ACTIONS))


def make_fixed_policy(action):
    def policy(state):
        return action
    return policy


POLICIES = {
    "RANDOM": random_policy,
    "CONTINUE": make_fixed_policy(0),
    "REASSESS": make_fixed_policy(1),
    "SWITCH": make_fixed_policy(2),
    "MULTIDISCIPLINARY": make_fixed_policy(3),
    "DOUBLE_DQN": dqn_policy,
}


# ============================================================
# Evaluation
# ============================================================

episode_rows = []

for policy_name, policy in POLICIES.items():

    print(f"\nEvaluating {policy_name}...")

    for episode in range(N_EPISODES):

        # Identical initial random seed across policies.
        episode_seed = SEED + episode

        env = OncoMARLEnv(seed=episode_seed)
        state = env.reset(seed=episode_seed)

        total_reward = 0.0
        action_counts = Counter()

        for step in range(MAX_STEPS):

            action = policy(state)
            action_counts[action] += 1

            next_state, reward, terminated, info = env.step(
                action
            )

            total_reward += reward
            state = next_state

            if terminated:
                break

        episode_rows.append({
            "policy": policy_name,
            "episode": episode,
            "seed": episode_seed,
            "total_reward": total_reward,
            "steps": step + 1,
            "final_burden": float(state[0]),
            "final_toxicity": float(state[1]),
            "final_uncertainty": float(state[2]),
            "final_progression": float(state[3]),
            "actions": dict(action_counts),
        })


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 72)
print("INDEPENDENT SYNTHETIC POLICY EVALUATION")
print("=" * 72)

summary_rows = []

for policy_name in POLICIES:

    rows = [
        row for row in episode_rows
        if row["policy"] == policy_name
    ]

    rewards = np.array([
        row["total_reward"] for row in rows
    ])

    burdens = np.array([
        row["final_burden"] for row in rows
    ])

    toxicities = np.array([
        row["final_toxicity"] for row in rows
    ])

    uncertainties = np.array([
        row["final_uncertainty"] for row in rows
    ])

    summary = {
        "policy": policy_name,
        "episodes": len(rows),
        "mean_return": float(rewards.mean()),
        "std_return": float(rewards.std()),
        "median_return": float(np.median(rewards)),
        "mean_final_burden": float(burdens.mean()),
        "mean_final_toxicity": float(toxicities.mean()),
        "mean_final_uncertainty": float(uncertainties.mean()),
    }

    summary_rows.append(summary)

    print(f"\nPolicy: {policy_name}")
    print(f"  Mean return:       {rewards.mean():.4f}")
    print(f"  Std return:        {rewards.std():.4f}")
    print(f"  Median return:     {np.median(rewards):.4f}")
    print(f"  Mean final burden: {burdens.mean():.4f}")
    print(f"  Mean final toxicity: {toxicities.mean():.4f}")
    print(f"  Mean final uncertainty: {uncertainties.mean():.4f}")


# ============================================================
# Save CSV reports
# ============================================================

summary_path = OUTPUT_DIR / "evaluation_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary_rows)


episodes_path = OUTPUT_DIR / "evaluation_episodes.csv"

with episodes_path.open("w", newline="", encoding="utf-8") as f:
    fields = [
        "policy",
        "episode",
        "seed",
        "total_reward",
        "steps",
        "final_burden",
        "final_toxicity",
        "final_uncertainty",
        "final_progression",
        "actions",
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for row in episode_rows:
        row = row.copy()
        row["actions"] = str(row["actions"])
        writer.writerow(row)


print("\nEvaluation complete.")
print("Summary:", summary_path)
print("Episode results:", episodes_path)
print("\nSynthetic research only; no clinical validity established.")