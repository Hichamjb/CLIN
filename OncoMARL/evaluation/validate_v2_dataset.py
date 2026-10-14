import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "oncomarl_v2"

CSV_PATH = DATA_DIR / "transitions.csv"
META_PATH = DATA_DIR / "metadata.json"

with META_PATH.open(encoding="utf-8") as f:
    metadata = json.load(f)

rows = []

with CSV_PATH.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    required = {
        "patient_id",
        "step",
        "action",
        "reward",
        "terminated",
        "state",
        "next_state",
    }

    assert required.issubset(reader.fieldnames), (
        f"Missing columns: {required - set(reader.fieldnames)}"
    )

    for row in reader:
        row["step"] = int(row["step"])
        row["action"] = int(row["action"])
        row["reward"] = float(row["reward"])
        row["terminated"] = int(row["terminated"])
        row["state"] = np.asarray(
            json.loads(row["state"]), dtype=np.float32
        )
        row["next_state"] = np.asarray(
            json.loads(row["next_state"]), dtype=np.float32
        )
        rows.append(row)

assert len(rows) == metadata["transitions"], (
    "Transition count does not match metadata"
)

assert len(rows) > 0, "Dataset is empty"

assert all(
    np.isfinite(r["state"]).all()
    and np.isfinite(r["next_state"]).all()
    and np.isfinite(r["reward"])
    for r in rows
), "Non-finite values detected"

assert all(
    len(r["state"]) == metadata["state_dim"]
    and len(r["next_state"]) == metadata["state_dim"]
    for r in rows
), "Incorrect state dimensions"

valid_actions = set(range(metadata["action_dim"]))

assert all(
    r["action"] in valid_actions for r in rows
), "Invalid action ID"

assert all(
    r["terminated"] in (0, 1) for r in rows
), "Invalid terminated flag"

by_patient = defaultdict(list)

for row in rows:
    by_patient[row["patient_id"]].append(row)

assert len(by_patient) == metadata["patients"], (
    "Patient count does not match metadata"
)

continuity_errors = 0
terminal_errors = 0

for patient_id, trajectory in by_patient.items():
    trajectory.sort(key=lambda r: r["step"])

    steps = [r["step"] for r in trajectory]

    if steps != list(range(len(trajectory))):
        raise AssertionError(
            f"Invalid step sequence for {patient_id}"
        )

    terminal_count = sum(
        r["terminated"] for r in trajectory
    )

    if terminal_count != 1:
        terminal_errors += 1

    for current, nxt in zip(
        trajectory[:-1], trajectory[1:]
    ):
        if not np.array_equal(
            current["next_state"], nxt["state"]
        ):
            continuity_errors += 1

action_counts = Counter(r["action"] for r in rows)

reward_by_action = defaultdict(list)

for row in rows:
    reward_by_action[row["action"]].append(row["reward"])

rewards = np.asarray(
    [r["reward"] for r in rows], dtype=np.float64
)

print("=" * 55)
print("ONCOMARL V2 DATASET VALIDATION")
print("=" * 55)

print(f"Patients: {len(by_patient):,}")
print(f"Transitions: {len(rows):,}")
print(f"State dimension: {metadata['state_dim']}")
print(f"Continuity errors: {continuity_errors}")
print(f"Trajectories with terminal-count errors: {terminal_errors}")

print("\nAction distribution:")

for action_id, action_name in metadata["actions"].items():
    count = action_counts[int(action_id)]

    print(
        f"{action_name}: {count:,} "
        f"({100 * count / len(rows):.2f}%)"
    )

print("\nReward statistics:")
print(f"Mean: {rewards.mean():.4f}")
print(f"Std:  {rewards.std():.4f}")
print(f"Min:  {rewards.min():.4f}")
print(f"Max:  {rewards.max():.4f}")

print("\nMean reward by action:")

for action_id, action_name in metadata["actions"].items():
    values = reward_by_action[int(action_id)]

    print(
        f"{action_name}: mean={np.mean(values):.4f}, "
        f"count={len(values):,}"
    )

assert continuity_errors == 0, "Transition continuity failed"
assert terminal_errors == 0, "Terminal flags are inconsistent"

print("\nAll structural validation checks passed.")
print("Synthetic data only; no clinical validity established.")