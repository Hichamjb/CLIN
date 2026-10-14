import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_model import DQN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_PATH = Path("reports/dqn_training/dqn_prepared_data.npz")
METADATA_PATH = Path(
    "reports/dqn_training/dqn_feature_metadata.json"
)

OUTPUT_DIR = Path("reports/dqn_training")
MODEL_DIR = OUTPUT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
GAMMA = 0.95
LEARNING_RATE = 1e-3

REPLAY_CAPACITY = 20_000
MIN_REPLAY_SIZE = 2_000

EPISODES = 10
MAX_STEPS_PER_EPISODE = 500

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

TARGET_UPDATE_STEPS = 250

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 8.4 — DQN TRAINING")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Prepared dataset not found: {DATA_PATH}"
    )

data = np.load(DATA_PATH, allow_pickle=False)

X = data["X"].astype(np.float32)
X_next = data["X_next"].astype(np.float32)
actions = data["actions"].astype(np.int64)
rewards = data["rewards"].astype(np.float32)
terminated = data["terminated"].astype(np.float32)
patient_ids = data["patient_ids"].astype(str)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

if metadata.get("dataset_type") != "SYNTHETIC":
    raise ValueError(
        "This training script expects the synthetic dataset."
    )

if X.shape != X_next.shape:
    raise ValueError("Current and next-state shapes do not match.")

if X.shape[0] != len(actions):
    raise ValueError("Number of actions does not match states.")

if not np.isfinite(X).all() or not np.isfinite(X_next).all():
    raise ValueError("Non-finite state features detected.")

if not np.isfinite(rewards).all():
    raise ValueError("Non-finite rewards detected.")

print("\nDevice:", DEVICE)
print("Transitions:", len(X))
print("Features:", X.shape[1])
print("Actions:", len(metadata["actions"]))


# ============================================================
# SPLIT BY PATIENT
# ============================================================

unique_patients = np.unique(patient_ids)
rng = np.random.default_rng(SEED)
rng.shuffle(unique_patients)

split_point = int(0.80 * len(unique_patients))

train_patients = set(unique_patients[:split_point])
test_patients = set(unique_patients[split_point:])

train_mask = np.isin(patient_ids, list(train_patients))
test_mask = np.isin(patient_ids, list(test_patients))

train_indices = np.flatnonzero(train_mask)
test_indices = np.flatnonzero(test_mask)

assert not (train_patients & test_patients)

print("\nPatient-level split:")
print("Training patients:", len(train_patients))
print("Testing patients:", len(test_patients))
print("Training transitions:", len(train_indices))
print("Testing transitions:", len(test_indices))


# ============================================================
# INITIALIZE NETWORKS
# ============================================================

state_dim = X.shape[1]
action_dim = len(metadata["actions"])

policy_net = DQN(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=128,
).to(DEVICE)

target_net = DQN(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=128,
).to(DEVICE)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(
    policy_net.parameters(),
    lr=LEARNING_RATE,
)

loss_function = nn.SmoothL1Loss()

print("\nNetwork initialized successfully.")
print("Parameters:", sum(
    p.numel() for p in policy_net.parameters()
))


# ============================================================
# REPLAY BUFFER
# ============================================================

replay_buffer = deque(maxlen=REPLAY_CAPACITY)


def add_transition(index):
    replay_buffer.append(
        (
            X[index],
            int(actions[index]),
            float(rewards[index]),
            X_next[index],
            float(terminated[index]),
        )
    )


def optimize_model():
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None

    batch = random.sample(replay_buffer, BATCH_SIZE)

    states = torch.tensor(
        np.stack([item[0] for item in batch]),
        dtype=torch.float32,
        device=DEVICE,
    )

    action_batch = torch.tensor(
        [item[1] for item in batch],
        dtype=torch.long,
        device=DEVICE,
    )

    reward_batch = torch.tensor(
        [item[2] for item in batch],
        dtype=torch.float32,
        device=DEVICE,
    )

    next_states = torch.tensor(
        np.stack([item[3] for item in batch]),
        dtype=torch.float32,
        device=DEVICE,
    )

    terminal_batch = torch.tensor(
        [item[4] for item in batch],
        dtype=torch.float32,
        device=DEVICE,
    )

    # Q(s, a) for actions recorded in the dataset.
    current_q = policy_net(states).gather(
        1, action_batch.unsqueeze(1)
    ).squeeze(1)

    # Standard DQN target.
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values

        target_q = reward_batch + (
            GAMMA * next_q * (1.0 - terminal_batch)
        )

    loss = loss_function(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=10.0,
    )

    optimizer.step()

    return float(loss.item())


# ============================================================
# OFFLINE TRAINING
# ============================================================

print("\nStarting offline training...")
print(
    "WARNING: Training optimizes synthetic logged transitions; "
    "it does not establish clinical treatment effectiveness."
)

epsilon = EPSILON_START
global_step = 0
history = []

for episode in range(1, EPISODES + 1):

    # Add a shuffled pass through training transitions.
    episode_indices = rng.permutation(train_indices)

    for index in episode_indices:
        add_transition(int(index))
        global_step += 1

        loss = optimize_model()

        if global_step % TARGET_UPDATE_STEPS == 0:
            target_net.load_state_dict(
                policy_net.state_dict()
            )

    epsilon = max(
        EPSILON_END,
        epsilon * EPSILON_DECAY,
    )

    # Monitor Q-values on held-out patient states.
    with torch.no_grad():
        test_sample = rng.choice(
            test_indices,
            size=min(2048, len(test_indices)),
            replace=False,
        )

        test_tensor = torch.tensor(
            X[test_sample],
            dtype=torch.float32,
            device=DEVICE,
        )

        q_values = policy_net(test_tensor)

        mean_max_q = float(
            q_values.max(dim=1).values.mean().item()
        )

        mean_abs_q = float(q_values.abs().mean().item())

    history.append({
        "episode": episode,
        "epsilon": epsilon,
        "mean_max_q_test": mean_max_q,
        "mean_abs_q_test": mean_abs_q,
        "replay_size": len(replay_buffer),
        "global_step": global_step,
    })

    print(
        f"Episode {episode:02d}/{EPISODES} | "
        f"epsilon={epsilon:.4f} | "
        f"mean_max_Q_test={mean_max_q:.4f} | "
        f"mean_abs_Q_test={mean_abs_q:.4f} | "
        f"replay={len(replay_buffer)}"
    )


# ============================================================
# SAVE MODEL AND LOGS
# ============================================================

model_path = MODEL_DIR / "dqn_synthetic_baseline.pt"
history_path = OUTPUT_DIR / "dqn_training_history.csv"

torch.save(
    {
        "model_state_dict": policy_net.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": 128,
        "gamma": GAMMA,
        "seed": SEED,
        "dataset_type": "SYNTHETIC",
        "actions": metadata["actions"],
        "feature_names": metadata["feature_names"],
    },
    model_path,
)

pd.DataFrame(history).to_csv(
    history_path,
    index=False,
)

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("Model saved to:", model_path)
print("Training history saved to:", history_path)

print(
    "\nInterpretation: Q-values and training diagnostics describe "
    "fit to synthetic transitions only. They are not measures "
    "of clinical benefit or treatment effectiveness."
)