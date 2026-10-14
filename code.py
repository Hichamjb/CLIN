#!/usr/bin/env python3
"""
generate_oncomarl_breast.py
===========================

Synthetic longitudinal breast-cancer dataset generator for OncoMARL
(Oncology Multi-Agent Reinforcement Learning) methodology research.

WARNING
-------
All records produced by this script are SYNTHETIC. This dataset is for
software development, simulation, and reinforcement-learning methodology
research only. It is NOT real clinical evidence, is NOT a validated
clinical dataset, and MUST NOT be used to diagnose patients, select real
cancer treatments, or make medical decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_VERSION = "1.0.0"
SYNTHETIC_DISCLAIMER = (
    "SYNTHETIC DATA - not real clinical evidence; not validated; "
    "must not be used for diagnosis, treatment selection, or any real "
    "medical decision. For software/simulation/RL methodology research only."
)

SUBTYPES = ["HR_POS_HER2_NEG", "HER2_POS", "TRIPLE_NEG"]
STAGES = ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"]
PATHWAYS = [
    "LOCAL_MANAGEMENT_REVIEW",
    "HORMONE_RECEPTOR_PATHWAY_REVIEW",
    "HER2_PATHWAY_REVIEW",
    "SYSTEMIC_PATHWAY_REVIEW",
    "MULTIDISCIPLINARY_REASSESSMENT",
]
ACTIONS = [
    "CONTINUE_CURRENT_PATHWAY",
    "REASSESS_CURRENT_PATHWAY",
    "SWITCH_PATHWAY",
    "MULTIDISCIPLINARY_REASSESSMENT",
]
SCENARIOS = list("ABCDEFGHIJKLMNO")

DETECTION_THRESHOLD_MM = 5.0
PROGRESSION_REL_THRESHOLD = 0.20
RESPONSE_REL_THRESHOLD = 0.30
TOXICITY_REASSESS_THRESHOLD = 3
UNCERTAINTY_STD_THRESHOLD = 0.10

DEFAULT_REWARD_WEIGHTS = {
    "policy_consistency": 1.0,
    "unnecessary_switch_penalty": 0.5,
    "toxicity_handling": 0.3,
    "progression_handling": 0.4,
    "reassessment_triggered": 0.2,
}

MIN_VISITS = 8   # required minimum to keep all scenario generators well-defined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_to_str(b: bool) -> str:
    return "TRUE" if bool(b) else "FALSE"


def _safe_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Patient / profile generation
# ---------------------------------------------------------------------------

def generate_patients(n_patients: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate one baseline profile row per patient."""
    patient_ids = [f"P{i:06d}" for i in range(1, n_patients + 1)]
    ages = rng.integers(28, 86, size=n_patients)
    sexes = np.where(rng.random(n_patients) < 0.01, "M", "F")

    stages = rng.choice(STAGES, size=n_patients, p=[0.08, 0.25, 0.35, 0.22, 0.10])
    subtypes = rng.choice(SUBTYPES, size=n_patients, p=[0.65, 0.20, 0.15])

    baseline_size = np.zeros(n_patients, dtype=float)
    ln_involve = np.zeros(n_patients, dtype=bool)
    met_status = np.zeros(n_patients, dtype=bool)
    for i in range(n_patients):
        s = stages[i]
        if s == "Stage 0":
            baseline_size[i] = rng.uniform(5, 25)
        elif s == "Stage I":
            baseline_size[i] = rng.uniform(5, 20)
        elif s == "Stage II":
            baseline_size[i] = rng.uniform(20, 50)
            ln_involve[i] = rng.random() < 0.40
        elif s == "Stage III":
            baseline_size[i] = rng.uniform(30, 80)
            ln_involve[i] = rng.random() < 0.85
        else:
            baseline_size[i] = rng.uniform(20, 100)
            ln_involve[i] = rng.random() < 0.70
            met_status[i] = True
    baseline_size = np.round(baseline_size, 1)

    perf_status = rng.integers(0, 3, size=n_patients)

    comorbidity_pool = [
        "hypertension", "type_2_diabetes", "hyperlipidemia", "asthma",
        "chronic_kidney_disease", "coronary_artery_disease", "obesity",
        "hypothyroidism",
    ]
    risk_factor_pool = [
        "family_history", "brca1_like", "brca2_like", "early_menarche",
        "late_menopause", "nulliparity", "hormone_exposure", "smoking_history",
        "alcohol_use",
    ]
    history_pool = [
        "benign_breast_biopsy", "prior_contralateral_breast_cancer",
        "prior_ovarian_cyst", "hysterectomy",
    ]

    def _joined(pool: List[str], max_n: int) -> str:
        k = int(rng.integers(0, max_n + 1))
        if k == 0:
            return "none"
        picks = rng.choice(pool, size=k, replace=False)
        return ";".join(str(x) for x in picks)

    comorbidities = [_joined(comorbidity_pool, 2) for _ in range(n_patients)]
    risk_factors = [_joined(risk_factor_pool, 2) for _ in range(n_patients)]
    histories = [_joined(history_pool, 1) for _ in range(n_patients)]

    base_date = datetime(2018, 1, 1)
    diag_offsets = rng.integers(0, 365 * 5, size=n_patients)
    diag_dates = [
        (base_date + timedelta(days=int(o))).strftime("%Y-%m-%d")
        for o in diag_offsets
    ]

    scenarios = rng.choice(SCENARIOS, size=n_patients)  # uniform

    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": ages,
            "sex": sexes,
            "diagnosis_date": diag_dates,
            "baseline_stage": stages,
            "tumor_subtype": subtypes,
            "baseline_tumor_size_mm": baseline_size,
            "lymph_node_involvement": [_bool_to_str(b) for b in ln_involve],
            "baseline_metastasis_status": [_bool_to_str(b) for b in met_status],
            "performance_status": perf_status,
            "comorbidities": comorbidities,
            "risk_factors": risk_factors,
            "medical_history": histories,
            "scenario": scenarios,
            "synthetic_flag": "SYNTHETIC",
        }
    )


def generate_tumor_profiles(patients: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Baseline tumour / receptor profile per patient."""
    n = len(patients)
    er = np.zeros(n)
    pr = np.zeros(n)
    her2 = np.zeros(n, dtype=int)
    ki67 = np.zeros(n)

    for i, row in enumerate(patients.itertuples(index=False)):
        st = row.tumor_subtype
        if st == "HR_POS_HER2_NEG":
            er[i] = rng.uniform(15, 95)
            pr[i] = rng.uniform(10, 90)
            her2[i] = int(rng.integers(0, 2))
            ki67[i] = rng.uniform(5, 35)
        elif st == "HER2_POS":
            er[i] = rng.uniform(0, 60)
            pr[i] = rng.uniform(0, 50)
            her2[i] = 3 if rng.random() < 0.85 else 2
            ki67[i] = rng.uniform(15, 55)
        else:
            er[i] = 0.0
            pr[i] = 0.0
            her2[i] = 0
            ki67[i] = rng.uniform(40, 90)

    return pd.DataFrame(
        {
            "patient_id": patients["patient_id"].values,
            "tumor_subtype": patients["tumor_subtype"].values,
            "baseline_stage": patients["baseline_stage"].values,
            "baseline_tumor_size_mm": patients["baseline_tumor_size_mm"].values,
            "lymph_node_involvement": patients["lymph_node_involvement"].values,
            "baseline_metastasis_status": patients["baseline_metastasis_status"].values,
            "er_percent": np.round(er, 1),
            "pr_percent": np.round(pr, 1),
            "her2_score": her2,
            "ki67_percent": np.round(ki67, 1),
            "synthetic_flag": "SYNTHETIC",
        }
    )


# ---------------------------------------------------------------------------
# Trajectory engine (A-O)
# ---------------------------------------------------------------------------

def _trajectory_for_scenario(
    scenario: str, base_size: float, n_visits: int, rng: np.random.Generator
) -> np.ndarray:
    """Return tumour sizes (mm) for one patient under one scenario."""
    t = np.arange(n_visits, dtype=float)
    sizes = np.full(n_visits, float(base_size))

    if base_size < DETECTION_THRESHOLD_MM:
        return np.zeros(n_visits)

    if scenario == "A":
        decay = rng.uniform(0.08, 0.18)
        sizes = base_size * np.exp(-decay * t)

    elif scenario == "B":
        floor = base_size * rng.uniform(0.40, 0.60)
        k = rng.uniform(0.5, 1.0)
        sizes = floor + (base_size - floor) * np.exp(-k * t)

    elif scenario == "C":
        decay = rng.uniform(0.20, 0.35)
        sizes = base_size * np.exp(-decay * t)

    elif scenario == "D":
        recover_visit = int(rng.integers(3, 6))
        lo = max(recover_visit + 1, 4)
        hi = max(lo + 1, n_visits)
        recur_visit = int(rng.integers(lo, hi))
        decay = rng.uniform(0.25, 0.40)
        for i in range(n_visits):
            if i < recover_visit:
                sizes[i] = base_size * np.exp(-decay * i)
            elif i < recur_visit:
                sizes[i] = 0.0
            else:
                sizes[i] = base_size * rng.uniform(0.40, 0.60) * (
                    1 + 0.25 * (i - recur_visit)
                )

    elif scenario == "E":
        nadir = int(rng.integers(3, max(4, n_visits - 3)))
        decay = rng.uniform(0.15, 0.25)
        growth = rng.uniform(0.15, 0.30)
        for i in range(n_visits):
            if i <= nadir:
                sizes[i] = base_size * np.exp(-decay * i)
            else:
                sizes[i] = sizes[nadir] * np.exp(growth * (i - nadir))

    elif scenario == "F":
        sizes = base_size * (1.0 + rng.normal(0, 0.05, n_visits))

    elif scenario == "G":
        growth = rng.uniform(0.10, 0.25)
        sizes = base_size * np.exp(growth * t)

    elif scenario == "H":
        sizes = base_size * (1.0 + rng.normal(0, 0.06, n_visits))

    elif scenario == "I":
        sizes = base_size * (1.0 + rng.normal(0, 0.07, n_visits))

    elif scenario == "J":
        sizes = base_size * (1.0 + rng.normal(0, 0.05, n_visits))

    elif scenario == "K":
        prog_visit = int(rng.integers(4, max(5, n_visits - 3)))
        growth = rng.uniform(0.20, 0.30)
        for i in range(n_visits):
            if i < prog_visit:
                sizes[i] = base_size * (1.0 + rng.normal(0, 0.03))
            else:
                sizes[i] = base_size * np.exp(growth * (i - prog_visit + 1))

    elif scenario == "L":
        nadir = int(rng.integers(3, max(4, n_visits - 3)))
        decay = rng.uniform(0.15, 0.25)
        growth = rng.uniform(0.15, 0.25)
        for i in range(n_visits):
            if i <= nadir:
                sizes[i] = base_size * np.exp(-decay * i)
            else:
                sizes[i] = sizes[nadir] * np.exp(growth * (i - nadir))

    elif scenario == "M":
        freq = rng.uniform(0.7, 1.3)
        amp = rng.uniform(0.10, 0.20)
        sizes = base_size * (
            1.0 + amp * np.sin(freq * t) + rng.normal(0, 0.03, n_visits)
        )

    elif scenario == "N":
        decay = rng.uniform(0.25, 0.40)
        for i in range(n_visits):
            sizes[i] = base_size * np.exp(-decay * i) if i < 4 else 0.0

    elif scenario == "O":
        sizes = np.full(n_visits, base_size)

    sizes = np.clip(sizes, 0.0, None)
    sizes = np.where(sizes < DETECTION_THRESHOLD_MM, 0.0, sizes)
    return np.round(sizes, 1)


# ---------------------------------------------------------------------------
# Policy & reward
# ---------------------------------------------------------------------------

def _initial_pathway(subtype: str, stage: str, rng: np.random.Generator) -> str:
    if stage == "Stage 0":
        return "LOCAL_MANAGEMENT_REVIEW"
    if subtype == "HER2_POS":
        return "HER2_PATHWAY_REVIEW"
    if subtype == "HR_POS_HER2_NEG":
        return "HORMONE_RECEPTOR_PATHWAY_REVIEW"
    if subtype == "TRIPLE_NEG":
        return "SYSTEMIC_PATHWAY_REVIEW"
    return str(rng.choice(PATHWAYS))


def _choose_next_pathway(current: str, subtype: str, rng: np.random.Generator) -> str:
    candidates = [p for p in PATHWAYS if p != current]
    if not candidates:
        return current
    if subtype == "HER2_POS" and "HER2_PATHWAY_REVIEW" in candidates and rng.random() < 0.6:
        return "HER2_PATHWAY_REVIEW"
    if (subtype == "HR_POS_HER2_NEG"
            and "HORMONE_RECEPTOR_PATHWAY_REVIEW" in candidates
            and rng.random() < 0.6):
        return "HORMONE_RECEPTOR_PATHWAY_REVIEW"
    if (subtype == "TRIPLE_NEG"
            and "SYSTEMIC_PATHWAY_REVIEW" in candidates
            and rng.random() < 0.6):
        return "SYSTEMIC_PATHWAY_REVIEW"
    return str(rng.choice(candidates))


def expected_action_from_policy(state: Dict[str, Any]) -> str:
    """Deterministic synthetic decision policy using only current-visit state."""
    if state.get("metastasis_new", False) or state.get("recurrence_flag", False):
        return "MULTIDISCIPLINARY_REASSESSMENT"
    if state.get("progression_flag", False):
        return "SWITCH_PATHWAY"
    if state.get("toxicity_grade", 0) >= TOXICITY_REASSESS_THRESHOLD:
        return "REASSESS_CURRENT_PATHWAY"
    if state.get("uncertainty_flag", False):
        return "REASSESS_CURRENT_PATHWAY"
    return "CONTINUE_CURRENT_PATHWAY"


def compute_reward(
    state: Dict[str, Any], action: str, weights: Dict[str, float]
) -> float:
    """Synthetic reward, current-visit-only. NOT a clinical benefit metric."""
    r = 0.0
    expected = expected_action_from_policy(state)

    if action == expected:
        r += weights["policy_consistency"]
    else:
        r -= weights["policy_consistency"]

    if action == "SWITCH_PATHWAY" and not (
        state.get("progression_flag", False) or state.get("metastasis_new", False)
    ):
        r -= weights["unnecessary_switch_penalty"]

    if state.get("toxicity_grade", 0) >= TOXICITY_REASSESS_THRESHOLD and action in (
        "REASSESS_CURRENT_PATHWAY", "MULTIDISCIPLINARY_REASSESSMENT",
    ):
        r += weights["toxicity_handling"]

    if (
        state.get("progression_flag", False) or state.get("metastasis_new", False)
    ) and action in ("SWITCH_PATHWAY", "MULTIDISCIPLINARY_REASSESSMENT"):
        r += weights["progression_handling"]

    return round(r, 4)


# ---------------------------------------------------------------------------
# Per-patient longitudinal simulation
# ---------------------------------------------------------------------------

def _classify_response(size_now: float, size_prev: float, size_baseline: float) -> str:
    if size_now == 0.0:
        return "CR_LIKE"
    if size_baseline > 0 and size_now <= (1.0 - RESPONSE_REL_THRESHOLD) * size_baseline:
        return "PR"
    if size_prev > 0 and (size_now - size_prev) / size_prev >= PROGRESSION_REL_THRESHOLD:
        return "PD"
    return "SD"


def simulate_patient_trajectory(
    patient: Dict[str, Any],
    profile: Dict[str, Any],
    n_visits: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Simulate one patient; return (visit_records, event_records)."""
    pid = patient["patient_id"]
    scenario = patient["scenario"]
    subtype = patient["tumor_subtype"]
    stage = patient["baseline_stage"]
    base_size = float(patient["baseline_tumor_size_mm"])
    diag_date = datetime.strptime(patient["diagnosis_date"], "%Y-%m-%d")

    sizes = _trajectory_for_scenario(scenario, base_size, n_visits, rng)

    # Metastasis timeline
    metastasis = np.zeros(n_visits, dtype=bool)
    if patient["baseline_metastasis_status"] == "TRUE":
        metastasis[:] = True
    elif scenario == "H":
        met_visit = int(rng.integers(4, max(5, n_visits - 2)))
        metastasis[met_visit:] = True

    # Lymph node (mostly static)
    ln = np.full(n_visits, patient["lymph_node_involvement"] == "TRUE", dtype=bool)

    # Toxicity
    tox = np.zeros(n_visits, dtype=int)
    base_tox = int(rng.integers(0, 2))
    tox[:] = base_tox
    if scenario == "J":
        tox_visit = int(rng.integers(3, max(4, n_visits - 3)))
        tox[tox_visit] = int(rng.integers(3, 5))
        for i in range(tox_visit + 1, n_visits):
            tox[i] = max(0, tox[i - 1] - 1)
    extra = rng.random(n_visits) < 0.05
    tox = np.maximum(tox, np.where(extra, rng.integers(2, 4, n_visits), 0))

    tox_types = ["NONE", "HEMATOLOGIC", "HEPATIC", "CARDIAC",
                 "NEUROLOGIC", "GI", "DERMATOLOGIC"]

    visits: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    current_pathway = _initial_pathway(subtype, stage, rng)
    size_prev = base_size
    prev_response = "NA"

    was_undetectable = False
    prev_tox = 0
    metastasis_announced = (
        bool(metastasis[0]) and patient["baseline_metastasis_status"] == "TRUE"
    )

    for t in range(n_visits):
        visit_num = t + 1
        visit_id = f"{pid}_V{visit_num:02d}"
        visit_date = diag_date + timedelta(
            days=30 * t + int(rng.integers(-3, 4))
        )

        size_now = float(sizes[t])

        if size_prev > 0:
            rel_change = (size_now - size_prev) / size_prev
        elif size_now > 0:
            rel_change = 1.0
        else:
            rel_change = 0.0

        progression_flag = (
            size_prev > 0 and size_now > size_prev * (1 + PROGRESSION_REL_THRESHOLD)
        ) or (size_prev == 0.0 and size_now > DETECTION_THRESHOLD_MM)
        recurrence_flag = (size_now > DETECTION_THRESHOLD_MM) and was_undetectable
        metastasis_new = bool(metastasis[t]) and not metastasis_announced
        if metastasis_new:
            metastasis_announced = True

        window = sizes[max(0, t - 2): t + 1]
        if len(window) >= 2 and window.mean() > 0:
            uncertainty = float(window.std() / window.mean())
        else:
            uncertainty = 0.0
        uncertainty_flag = uncertainty > UNCERTAINTY_STD_THRESHOLD

        response_cat = _classify_response(size_now, size_prev, base_size)

        if size_now == 0.0:
            disease_status = "REMISSION_LIKE"
        elif recurrence_flag:
            disease_status = "RECURRED"
        elif progression_flag or metastasis_new:
            disease_status = "PROGRESSING"
        elif response_cat in ("PR", "CR_LIKE"):
            disease_status = "RESPONDING"
        else:
            disease_status = "STABLE"

        tox_grade = int(tox[t])
        tox_type = "NONE" if tox_grade == 0 else str(rng.choice(tox_types[1:]))

        wbc = max(0.1, round(float(rng.normal(6.0 - 0.4 * tox_grade, 1.2)), 2))
        hgb = max(4.0, round(float(rng.normal(12.5 - 0.5 * tox_grade, 1.0)), 2))
        plt = max(5.0, round(float(rng.normal(250 - 20 * tox_grade, 40)), 1))
        alt = max(1.0, round(float(rng.normal(25 + 8 * tox_grade, 8)), 1))
        ast = max(1.0, round(float(rng.normal(25 + 8 * tox_grade, 8)), 1))
        cr = max(0.1, round(float(rng.normal(0.9 + 0.05 * tox_grade, 0.15)), 2))

        state_dict: Dict[str, Any] = {
            "visit_number": int(visit_num),
            "tumor_size_mm": round(size_now, 2),
            "tumor_size_trend": round(float(rel_change), 4),
            "disease_status": disease_status,
            "metastasis_status": bool(metastasis[t]),
            "metastasis_new": bool(metastasis_new),
            "toxicity_grade": int(tox_grade),
            "current_pathway": current_pathway,
            "previous_response": prev_response,
            "age": int(patient["age"]),
            "baseline_stage": str(stage),
            "tumor_subtype": str(subtype),
            "performance_status": int(patient["performance_status"]),
            "progression_flag": bool(progression_flag),
            "recurrence_flag": bool(recurrence_flag),
            "uncertainty_flag": bool(uncertainty_flag),
        }

        action = expected_action_from_policy(state_dict)
        next_pathway = current_pathway
        if action == "SWITCH_PATHWAY":
            next_pathway = _choose_next_pathway(current_pathway, subtype, rng)
        elif action == "MULTIDISCIPLINARY_REASSESSMENT":
            next_pathway = "MULTIDISCIPLINARY_REASSESSMENT"

        # ---- events ---------------------------------------------------------
        date_str = visit_date.strftime("%Y-%m-%d")

        if t == 0:
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "INITIAL_DIAGNOSIS",
                "severity_or_magnitude": round(base_size, 2),
                "synthetic_event_description": (
                    f"Synthetic baseline diagnosis; stage={stage}, "
                    f"subtype={subtype}, size={base_size}mm."
                ),
            })

        if t > 0 and response_cat in ("PR", "CR_LIKE"):
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "TUMOR_RESPONSE",
                "severity_or_magnitude": round(float(rel_change), 4),
                "synthetic_event_description": (
                    f"Synthetic tumour response ({response_cat}); size={size_now}mm."
                ),
            })
        if t > 0 and disease_status == "STABLE":
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "STABLE_DISEASE",
                "severity_or_magnitude": round(float(rel_change), 4),
                "synthetic_event_description": "Synthetic stable disease window.",
            })
        if progression_flag:
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "DISEASE_PROGRESSION",
                "severity_or_magnitude": round(float(rel_change), 4),
                "synthetic_event_description": (
                    f"Synthetic progression detected; size={size_now}mm."
                ),
            })
        if metastasis_new:
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "METASTASIS_DETECTED",
                "severity_or_magnitude": 1.0,
                "synthetic_event_description": "Synthetic new metastasis event.",
            })
        if size_now == 0.0 and not was_undetectable:
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "REMISSION_LIKE_STATE",
                "severity_or_magnitude": 0.0,
                "synthetic_event_description": (
                    "Synthetic undetectable tumour state (does NOT imply cure)."
                ),
            })
        if recurrence_flag:
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "RECURRENCE",
                "severity_or_magnitude": round(size_now, 2),
                "synthetic_event_description": (
                    "Synthetic recurrence after undetectable state."
                ),
            })
        if tox_grade > prev_tox and tox_grade >= 2:
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "TOXICITY_INCREASE",
                "severity_or_magnitude": float(tox_grade),
                "synthetic_event_description": (
                    f"Synthetic toxicity increase to grade {tox_grade} ({tox_type})."
                ),
            })
        if action == "REASSESS_CURRENT_PATHWAY":
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "PATHWAY_REASSESSMENT",
                "severity_or_magnitude": 1.0,
                "synthetic_event_description": "Synthetic pathway reassessment.",
            })
        if action == "SWITCH_PATHWAY":
            events.append({
                "patient_id": pid, "visit_id": visit_id, "event_date": date_str,
                "event_type": "PATHWAY_SWITCH",
                "severity_or_magnitude": 1.0,
                "synthetic_event_description": (
                    f"Synthetic pathway switch: {current_pathway} -> {next_pathway}."
                ),
            })

        visits.append({
            "visit_id": visit_id,
            "patient_id": pid,
            "visit_number": int(visit_num),
            "visit_date": date_str,
            "tumor_size_mm": round(size_now, 2),
            "lymph_node_status": _bool_to_str(bool(ln[t])),
            "metastasis_status": _bool_to_str(bool(metastasis[t])),
            "disease_status": disease_status,
            "er_percent": float(profile["er_percent"]),
            "pr_percent": float(profile["pr_percent"]),
            "her2_score": int(profile["her2_score"]),
            "ki67_percent": float(profile["ki67_percent"]),
            "wbc": wbc,
            "hemoglobin": hgb,
            "platelets": plt,
            "alt": alt,
            "ast": ast,
            "creatinine": cr,
            "toxicity_grade": int(tox_grade),
            "toxicity_type": tox_type,
            "current_pathway": current_pathway,
            "response_category": response_cat,
            "recurrence_status": _bool_to_str(bool(recurrence_flag)),
            "progression_status": _bool_to_str(bool(progression_flag)),
            "synthetic_flag": "SYNTHETIC",
            # internal (dropped before writing)
            "_next_pathway": next_pathway,
            "_action": action,
            "_state_dict": state_dict,
        })

        was_undetectable = size_now == 0.0
        prev_tox = tox_grade
        size_prev = size_now
        prev_response = response_cat
        current_pathway = next_pathway

    return visits, events


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------

def assemble_tables(
    patients: pd.DataFrame,
    profiles: pd.DataFrame,
    visits_flat: List[Dict[str, Any]],
    events_flat: List[Dict[str, Any]],
    reward_weights: Dict[str, float],
    n_visits: int,
) -> Dict[str, pd.DataFrame]:
    visits_df = pd.DataFrame(visits_flat)
    visits_df = visits_df.drop(
        columns=["_next_pathway", "_action", "_state_dict"], errors="ignore"
    )

    if events_flat:
        events_df = pd.DataFrame(events_flat)
        events_df.insert(
            0, "event_id",
            [f"E{i:08d}" for i in range(1, len(events_df) + 1)],
        )
    else:
        events_df = pd.DataFrame(columns=[
            "event_id", "patient_id", "visit_id", "event_date",
            "event_type", "severity_or_magnitude", "synthetic_event_description",
        ])

    # Fast lookup
    visit_lookup: Dict[str, Dict[str, Any]] = {v["visit_id"]: v for v in visits_flat}

    # treatment_history
    hist = visits_df[["patient_id", "visit_id", "visit_number", "current_pathway"]].copy()
    hist = hist.rename(columns={"current_pathway": "pathway"})
    hist = hist.sort_values(["patient_id", "visit_number"]).reset_index(drop=True)
    hist["previous_pathway"] = (
        hist.groupby("patient_id")["pathway"].shift(1).fillna("")
    )
    hist["change_flag"] = (
        (hist["previous_pathway"] != "")
        & (hist["previous_pathway"] != hist["pathway"])
    ).map({True: "TRUE", False: "FALSE"})
    hist.insert(0, "record_id", [f"TH{i:08d}" for i in range(1, len(hist) + 1)])
    hist["synthetic_flag"] = "SYNTHETIC"

    # treatment_decisions
    decision_rows: List[Dict[str, Any]] = []
    for v in visits_flat:
        action = v["_action"]
        sd = v["_state_dict"]
        if action == "MULTIDISCIPLINARY_REASSESSMENT":
            reason = "metastasis_or_recurrence"
        elif action == "SWITCH_PATHWAY":
            reason = "progression"
        elif action == "REASSESS_CURRENT_PATHWAY":
            reason = "toxicity" if sd["toxicity_grade"] >= TOXICITY_REASSESS_THRESHOLD else "uncertainty"
        else:
            reason = "routine_continuation"
        decision_rows.append({
            "patient_id": v["patient_id"],
            "visit_id": v["visit_id"],
            "visit_number": v["visit_number"],
            "current_pathway": v["current_pathway"],
            "action": action,
            "next_pathway": v["_next_pathway"],
            "policy_reason": reason,
            "synthetic_flag": "SYNTHETIC",
        })
    decisions_df = pd.DataFrame(decision_rows)
    decisions_df.insert(
        0, "decision_id",
        [f"D{i:08d}" for i in range(1, len(decisions_df) + 1)],
    )

    # outcomes
    visits_sorted = visits_df.sort_values(
        ["patient_id", "visit_number"]
    ).reset_index(drop=True)

    response_order = {"CR_LIKE": 4, "PR": 3, "SD": 2, "PD": 1, "NA": 0}
    out_rows: List[Dict[str, Any]] = []
    for pid, g in visits_sorted.groupby("patient_id", sort=False):
        final = g.iloc[-1]
        best_resp = max(g["response_category"], key=lambda r: response_order.get(r, 0))
        out_rows.append({
            "patient_id": pid,
            "best_response": best_resp,
            "final_response": final["response_category"],
            "progression_ever": _bool_to_str((g["progression_status"] == "TRUE").any()),
            "recurrence_ever": _bool_to_str((g["recurrence_status"] == "TRUE").any()),
            "metastasis_ever": _bool_to_str((g["metastasis_status"] == "TRUE").any()),
            "max_toxicity_grade": int(g["toxicity_grade"].max()),
            "final_tumor_size_mm": float(final["tumor_size_mm"]),
            "final_disease_status": final["disease_status"],
            "n_visits": int(len(g)),
            "synthetic_flag": "SYNTHETIC",
        })
    outcomes_df = pd.DataFrame(out_rows).merge(
        patients[["patient_id", "scenario"]], on="patient_id", how="left"
    )

    # rl_transitions
    transition_rows: List[Dict[str, Any]] = []
    tid = 1
    for pid, g in visits_sorted.groupby("patient_id", sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        for i in range(n - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]
            cur_rec = visit_lookup[cur["visit_id"]]
            nxt_rec = visit_lookup[nxt["visit_id"]]
            cur_state = cur_rec["_state_dict"]
            nxt_state = nxt_rec["_state_dict"]
            action = cur_rec["_action"]
            reward = compute_reward(cur_state, action, reward_weights)
            terminated = (i == n - 2)
            transition_rows.append({
                "transition_id": f"T{tid:08d}",
                "patient_id": pid,
                "current_visit_id": cur["visit_id"],
                "next_visit_id": nxt["visit_id"],
                "current_visit_number": int(cur["visit_number"]),
                "next_visit_number": int(nxt["visit_number"]),
                "state_json": json.dumps(cur_state, sort_keys=True),
                "action": action,
                "reward": reward,
                "next_state_json": json.dumps(nxt_state, sort_keys=True),
                "terminated": _bool_to_str(terminated),
            })
            tid += 1
    transitions_df = pd.DataFrame(transition_rows)

    return {
        "patients": patients,
        "tumor_profiles": profiles,
        "clinical_visits": visits_df,
        "clinical_events": events_df,
        "treatment_history": hist,
        "treatment_decisions": decisions_df,
        "outcomes": outcomes_df,
        "rl_transitions": transitions_df,
    }


# ---------------------------------------------------------------------------
# Distribution summaries / metadata
# ---------------------------------------------------------------------------

def write_distributions(tables: Dict[str, pd.DataFrame], out_dir: str) -> None:
    visits = tables["clinical_visits"]
    decisions = tables["treatment_decisions"]
    patients = tables["patients"]

    scen = patients["scenario"].value_counts().rename_axis("scenario").reset_index(name="count")
    scen["fraction"] = (scen["count"] / scen["count"].sum()).round(6)
    scen.to_csv(os.path.join(out_dir, "scenario_distribution.csv"), index=False)

    act = decisions["action"].value_counts().rename_axis("action").reset_index(name="count")
    act["fraction"] = (act["count"] / act["count"].sum()).round(6)
    act.to_csv(os.path.join(out_dir, "action_distribution.csv"), index=False)

    path = visits["current_pathway"].value_counts().rename_axis("pathway").reset_index(name="count")
    path["fraction"] = (path["count"] / path["count"].sum()).round(6)
    path.to_csv(os.path.join(out_dir, "pathway_distribution.csv"), index=False)


def write_metadata(
    tables: Dict[str, pd.DataFrame],
    out_dir: str,
    seed: int,
    n_patients: int,
    n_visits: int,
    reward_weights: Dict[str, float],
) -> None:
    meta = {
        "dataset_name": "OncoMARL Synthetic Breast Cancer Dataset",
        "dataset_version": DATASET_VERSION,
        "generation_timestamp": datetime.utcnow().isoformat() + "Z",
        "random_seed": int(seed),
        "patient_count": int(n_patients),
        "visits_per_patient": int(n_visits),
        "clinical_visits_rows": int(len(tables["clinical_visits"])),
        "clinical_events_rows": int(len(tables["clinical_events"])),
        "rl_transitions_rows": int(len(tables["rl_transitions"])),
        "cancer_type": "breast_cancer_only",
        "synthetic_disclaimer": SYNTHETIC_DISCLAIMER,
        "reward_weights": reward_weights,
        "detection_threshold_mm": DETECTION_THRESHOLD_MM,
        "progression_rel_threshold": PROGRESSION_REL_THRESHOLD,
        "response_rel_threshold": RESPONSE_REL_THRESHOLD,
        "toxicity_reassess_threshold": TOXICITY_REASSESS_THRESHOLD,
        "uncertainty_std_threshold": UNCERTAINTY_STD_THRESHOLD,
        "pathways": PATHWAYS,
        "actions": ACTIONS,
        "subtypes": SUBTYPES,
        "stages": STAGES,
    }
    with open(os.path.join(out_dir, "dataset_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic OncoMARL breast-cancer dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--patients", type=int, default=10000)
    p.add_argument("--visits", type=int, default=12)
    p.add_argument("--out", type=str, default="data/oncomarl_breast")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward-policy-consistency", type=float,
                   default=DEFAULT_REWARD_WEIGHTS["policy_consistency"])
    p.add_argument("--reward-unnecessary-switch", type=float,
                   default=DEFAULT_REWARD_WEIGHTS["unnecessary_switch_penalty"])
    p.add_argument("--reward-toxicity", type=float,
                   default=DEFAULT_REWARD_WEIGHTS["toxicity_handling"])
    p.add_argument("--reward-progression", type=float,
                   default=DEFAULT_REWARD_WEIGHTS["progression_handling"])
    p.add_argument("--reward-reassessment", type=float,
                   default=DEFAULT_REWARD_WEIGHTS["reassessment_triggered"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.patients <= 0:
        print("ERROR: --patients must be > 0", file=sys.stderr)
        return 2
    if args.visits < MIN_VISITS:
        print(f"ERROR: --visits must be >= {MIN_VISITS}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    rng = _safe_rng(args.seed)

    print(f"[generator] patients={args.patients} visits={args.visits} seed={args.seed}")
    print(f"[generator] output directory: {args.out}")

    patients_df = generate_patients(args.patients, rng)
    profiles_df = generate_tumor_profiles(patients_df, rng)

    all_visits: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []

    patient_dicts = patients_df.to_dict(orient="records")
    profile_dicts = profiles_df.set_index("patient_id").to_dict(orient="index")

    progress_step = max(1, args.patients // 10)
    for i, p in enumerate(patient_dicts):
        prof = profile_dicts[p["patient_id"]]
        v, e = simulate_patient_trajectory(p, prof, args.visits, rng)
        all_visits.extend(v)
        all_events.extend(e)
        if (i + 1) % progress_step == 0:
            print(f"[generator] simulated {i + 1}/{args.patients} patients")

    reward_weights = {
        "policy_consistency": args.reward_policy_consistency,
        "unnecessary_switch_penalty": args.reward_unnecessary_switch,
        "toxicity_handling": args.reward_toxicity,
        "progression_handling": args.reward_progression,
        "reassessment_triggered": args.reward_reassessment,
    }

    tables = assemble_tables(
        patients_df, profiles_df, all_visits, all_events, reward_weights, args.visits
    )

    for name in [
        "patients", "tumor_profiles", "clinical_visits", "clinical_events",
        "treatment_history", "treatment_decisions", "outcomes", "rl_transitions",
    ]:
        path = os.path.join(args.out, f"{name}.csv")
        tables[name].to_csv(path, index=False)
        print(f"[generator] wrote {name}.csv  ({len(tables[name])} rows)")

    write_distributions(tables, args.out)
    write_metadata(tables, args.out, args.seed, args.patients, args.visits, reward_weights)

    print("[generator] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())