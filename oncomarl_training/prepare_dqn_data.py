import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/oncomarl_breast/rl_transitions.csv")
OUTPUT_DIR = Path("reports/dqn_training")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = [
    "CONTINUE_CURRENT_PATHWAY",
    "REASSESS_CURRENT_PATHWAY",
    "SWITCH_PATHWAY",
    "MULTIDISCIPLINARY_REASSESSMENT",
]

NUMERIC_FIELDS = [
    "age",
    "performance_status",
    "toxicity_grade",
    "tumor_size_mm",
    "tumor_size_trend",
    "visit_number",
]

CATEGORICAL_FIELDS = [
    "baseline_stage",
    "current_pathway",
    "disease_status",
    "previous_response",
    "tumor_subtype",
]

BOOLEAN_FIELDS = [
    "metastasis_new",
    "metastasis_status",
    "progression_flag",
    "recurrence_flag",
    "uncertainty_flag",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("STEP 8.2 — PREPARE DQN FEATURES")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {
    "patient_id",
    "state_json",
    "action",
    "reward",
    "next_state_json",
    "terminated",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {sorted(missing_columns)}")

print("\nLoaded transitions:", len(df))


# ============================================================
# PARSE STATES
# ============================================================

states = df["state_json"].apply(json.loads).tolist()
next_states = df["next_state_json"].apply(json.loads).tolist()

for i, state in enumerate(states):
    for field in (
        NUMERIC_FIELDS
        + CATEGORICAL_FIELDS
        + BOOLEAN_FIELDS
    ):
        if field not in state:
            raise ValueError(
                f"State {i} is missing required field: {field}"
            )

for i, state in enumerate(next_states):
    for field in (
        NUMERIC_FIELDS
        + CATEGORICAL_FIELDS
        + BOOLEAN_FIELDS
    ):
        if field not in state:
            raise ValueError(
                f"Next state {i} is missing required field: {field}"
            )


# ============================================================
# FIT FEATURE ENCODING
# ============================================================

# Fit categories using current and next states together.
# This prevents unseen next-state categories.

all_states = states + next_states

categories = {}

for field in CATEGORICAL_FIELDS:
    categories[field] = sorted({
        str(state[field]) for state in all_states
    })

feature_names = []

for field in NUMERIC_FIELDS:
    feature_names.append(field)

for field in BOOLEAN_FIELDS:
    feature_names.append(field)

for field in CATEGORICAL_FIELDS:
    for category in categories[field]:
        feature_names.append(f"{field}={category}")


# ============================================================
# FEATURE ENCODER
# ============================================================

def encode_state(state):
    features = []

    # Numeric features
    for field in NUMERIC_FIELDS:
        features.append(float(state[field]))

    # Boolean features
    for field in BOOLEAN_FIELDS:
        features.append(float(bool(state[field])))

    # One-hot categorical features
    for field in CATEGORICAL_FIELDS:
        value = str(state[field])

        features.extend([
            float(value == category)
            for category in categories[field]
        ])

    return np.asarray(features, dtype=np.float32)


X = np.stack([encode_state(s) for s in states])
X_next = np.stack([encode_state(s) for s in next_states])


# ============================================================
# SCALE NUMERIC FEATURES
# ============================================================

# Use current-state statistics only.
# Constant numeric fields use scale=1 to avoid division by zero.

numeric_count = len(NUMERIC_FIELDS)

numeric_mean = X[:, :numeric_count].mean(axis=0)
numeric_std = X[:, :numeric_count].std(axis=0)

numeric_std[numeric_std < 1e-8] = 1.0

X[:, :numeric_count] = (
    X[:, :numeric_count] - numeric_mean
) / numeric_std

X_next[:, :numeric_count] = (
    X_next[:, :numeric_count] - numeric_mean
) / numeric_std


# ============================================================
# ENCODE ACTIONS
# ============================================================

unknown_actions = set(df["action"]) - set(ACTIONS)

if unknown_actions:
    raise ValueError(f"Unknown actions: {sorted(unknown_actions)}")

action_to_id = {
    action: index
    for index, action in enumerate(ACTIONS)
}

action_ids = df["action"].map(action_to_id).to_numpy(
    dtype=np.int64
)

rewards = df["reward"].to_numpy(dtype=np.float32)

terminated = (
    df["terminated"]
    .astype(str)
    .str.upper()
    .isin(["TRUE", "1"])
    .to_numpy(dtype=np.float32)
)

patient_ids = df["patient_id"].astype(str).to_numpy()


# ============================================================
# VALIDATE OUTPUTS
# ============================================================

assert X.shape[0] == len(df)
assert X_next.shape == X.shape
assert action_ids.shape[0] == len(df)
assert np.isfinite(X).all()
assert np.isfinite(X_next).all()
assert np.isfinite(rewards).all()
assert set(np.unique(action_ids)).issubset({0, 1, 2, 3})

print("\nFeature matrix shape:", X.shape)
print("Next-state matrix shape:", X_next.shape)
print("Feature count:", X.shape[1])

print("\nAction mapping:")
for action, action_id in action_to_id.items():
    print(f"  {action_id}: {action}")

print("\nReward range:", float(rewards.min()), "to", float(rewards.max()))
print("Terminal transitions:", int(terminated.sum()))
print("Unique patients:", len(np.unique(patient_ids)))

print("\nNumeric feature statistics:")
for i, field in enumerate(NUMERIC_FIELDS):
    print(
        f"  {field}: "
        f"mean={X[:, i].mean():.4f}, "
        f"std={X[:, i].std():.4f}"
    )


# ============================================================
# SAVE PREPARED DATA
# ============================================================

np.savez_compressed(
    OUTPUT_DIR / "dqn_prepared_data.npz",
    X=X,
    X_next=X_next,
    actions=action_ids,
    rewards=rewards,
    terminated=terminated,
    patient_ids=np.asarray(patient_ids, dtype=str),
)

metadata = {
    "dataset_type": "SYNTHETIC",
    "num_transitions": int(len(df)),
    "num_features": int(X.shape[1]),
    "actions": action_to_id,
    "numeric_fields": NUMERIC_FIELDS,
    "boolean_fields": BOOLEAN_FIELDS,
    "categorical_fields": CATEGORICAL_FIELDS,
    "categories": categories,
    "numeric_mean": numeric_mean.tolist(),
    "numeric_std": numeric_std.tolist(),
    "feature_names": feature_names,
    "warning": (
        "Synthetic experimental data only. "
        "Not validated for clinical decision-making."
    ),
}

with open(
    OUTPUT_DIR / "dqn_feature_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=2, ensure_ascii=False)

print("\nSaved prepared dataset:")
print(" ", OUTPUT_DIR / "dqn_prepared_data.npz")
print(" ", OUTPUT_DIR / "dqn_feature_metadata.json")

print("\nSTEP 8.2 COMPLETED SUCCESSFULLY")
print("Original dataset was not modified.")