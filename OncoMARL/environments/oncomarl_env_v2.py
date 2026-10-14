from dataclasses import dataclass
import numpy as np


# Synthetic research environment only.
# These parameters are illustrative and have NO clinical validity.

CONTINUE = 0
REASSESS = 1
SWITCH = 2
MULTIDISCIPLINARY = 3

ACTIONS = {
    CONTINUE: "CONTINUE_CURRENT_PATHWAY",
    REASSESS: "REASSESS_CURRENT_PATHWAY",
    SWITCH: "SWITCH_PATHWAY",
    MULTIDISCIPLINARY: "MULTIDISCIPLINARY_REASSESSMENT",
}


@dataclass
class SyntheticState:
    tumor_burden: float
    toxicity: float
    uncertainty: float
    progression: float
    visit_index: int = 0


class OncoMARLEnv:
    """
    Interactive synthetic environment for RL experiments.

    All dynamics and reward coefficients are artificial.
    They must not be interpreted as clinical treatment effects.
    """

    def __init__(self, max_visits=12, seed=42):
        self.max_visits = max_visits
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = SyntheticState(
            tumor_burden=float(self.rng.uniform(0.2, 0.8)),
            toxicity=float(self.rng.uniform(0.0, 0.3)),
            uncertainty=float(self.rng.uniform(0.1, 0.6)),
            progression=0.0,
            visit_index=0,
        )

        return self._observe()

    def _observe(self):
        s = self.state

        return np.array([
            s.tumor_burden,
            s.toxicity,
            s.uncertainty,
            s.progression,
            s.visit_index / self.max_visits,
        ], dtype=np.float32)

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        s = self.state

        # Save the current state for outcome-based reward calculation.
        old_burden = s.tumor_burden
        old_toxicity = s.toxicity

        # Illustrative synthetic action-conditioned dynamics.
        # These coefficients are arbitrary simulation assumptions,
        # NOT estimates of real cancer-treatment effects.

        if action == CONTINUE:
            burden_delta = 0.015
            toxicity_delta = -0.01
            uncertainty_delta = -0.02

        elif action == REASSESS:
            burden_delta = 0.0
            toxicity_delta = -0.03
            uncertainty_delta = -0.12

        elif action == SWITCH:
            burden_delta = -0.025
            toxicity_delta = 0.04
            uncertainty_delta = 0.02

        else:  # MULTIDISCIPLINARY
            burden_delta = -0.005
            toxicity_delta = -0.02
            uncertainty_delta = -0.15

        # Randomness represents artificial simulator variability.
        noise = self.rng.normal(0.0, 0.02)

        s.tumor_burden = float(np.clip(
            old_burden + burden_delta + noise, 0.0, 1.0
        ))

        s.toxicity = float(np.clip(
            old_toxicity + toxicity_delta
            + self.rng.normal(0.0, 0.01),
            0.0, 1.0
        ))

        s.uncertainty = float(np.clip(
            s.uncertainty + uncertainty_delta
            + self.rng.normal(0.0, 0.02),
            0.0, 1.0
        ))

        s.progression = float(
            s.tumor_burden > old_burden + 0.03
        )

        s.visit_index += 1

        # Reward depends on simulated state changes,
        # not on a fixed reward assigned to an action.
        burden_change = s.tumor_burden - old_burden
        toxicity_change = s.toxicity - old_toxicity

        reward = (
            -1.0 * burden_change
            -0.5 * toxicity_change
            -0.2 * s.uncertainty
        )

        terminated = s.visit_index >= self.max_visits

        info = {
            "action_name": ACTIONS[action],
            "burden_change": burden_change,
            "toxicity_change": toxicity_change,
            "synthetic": True,
        }

        return self._observe(), float(reward), terminated, info