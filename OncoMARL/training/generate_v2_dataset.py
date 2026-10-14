import csv
import json
from pathlib import Path

import numpy as np
import sys
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.oncomarl_env_v2 import OncoMARLEnv, ACTIONS
from environments.oncomarl_env_v2 import OncoMARLEnv, ACTIONS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "oncomarl_v2"

N_PATIENTS = 10_000
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

transition_path = OUTPUT_DIR / "transitions.csv"
metadata_path = OUTPUT_DIR / "metadata.json"

state_dim = None
n_transitions = 0
n_episodes = 0

fieldnames = [
    "patient_id",
    "step",
    "action",
    "reward",
    "terminated",
    "state",
    "next_state",
]

with transition_path.open(
    "w", newline="", encoding="utf-8"
) as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for patient_idx in range(N_PATIENTS):
        env = OncoMARLEnv(
            seed=int(rng.integers(0, 2**31 - 1))
        )

        state = env.reset()
        terminated = False
        step_idx = 0

        while not terminated:
            # Uniform random exploration for dataset coverage.
            action = int(rng.choice(list(ACTIONS.keys())))

            next_state, reward, terminated, info = env.step(action)

            if not np.isfinite(state).all():
                raise ValueError("Non-finite state detected")

            if not np.isfinite(next_state).all():
                raise ValueError("Non-finite next_state detected")

            if not np.isfinite(reward):
                raise ValueError("Non-finite reward detected")

            if state_dim is None:
                state_dim = len(state)

            writer.writerow({
                "patient_id": f"SYN_{patient_idx:06d}",
                "step": step_idx,
                "action": action,
                "reward": reward,
                "terminated": int(terminated),
                "state": json.dumps(state.tolist()),
                "next_state": json.dumps(next_state.tolist()),
            })

            state = next_state
            step_idx += 1
            n_transitions += 1

        n_episodes += 1

metadata = {
    "dataset_type": "SYNTHETIC",
    "environment": "OncoMARL v2",
    "seed": SEED,
    "patients": n_episodes,
    "transitions": n_transitions,
    "state_dim": state_dim,
    "action_dim": len(ACTIONS),
    "actions": ACTIONS,
    "max_visits": 12,
    "policy": "uniform_random",
    "clinical_validity": False,
    "intended_use": "software and RL research only",
}

with metadata_path.open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Dataset generation complete.")
print(f"Patients: {n_episodes}")
print(f"Transitions: {n_transitions}")
print(f"State dimension: {state_dim}")
print(f"CSV: {transition_path}")
print(f"Metadata: {metadata_path}")