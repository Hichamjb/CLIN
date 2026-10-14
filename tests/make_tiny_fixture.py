#!/usr/bin/env python3
"""Create a tiny SYNTHETIC test fixture to exercise the analysis pipeline.

This is NOT a replacement for the user's dataset. It exists only so that
analyze_oncomarl.py and validate_dataset.py can be smoke-tested offline.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def make(out_dir: str, n_patients: int = 8, n_visits: int = 12, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    subtypes = ["HR_POS_HER2_NEG", "HER2_POS", "TRIPLE_NEG"]

    rows_p, rows_tp = [], []
    for i in range(n_patients):
        pid = f"P{i:04d}"
        stage = stages[i % len(stages)]
        subtype = subtypes[i % len(subtypes)]
        base = float(rng.uniform(10, 70))
        met = stage == "Stage IV"
        rows_p.append({
            "patient_id": pid, "age": int(rng.integers(30, 80)), "sex": "F",
            "diagnosis_date": "2020-01-01",
            "baseline_stage": stage, "tumor_subtype": subtype,
            "baseline_tumor_size_mm": round(base, 1),
            "lymph_node_involvement": "TRUE" if stage in ("Stage III", "Stage IV") else "FALSE",
            "baseline_metastasis_status": "TRUE" if met else "FALSE",
            "performance_status": int(rng.integers(0, 3)),
            "comorbidities": "none", "risk_factors": "family_history",
            "medical_history": "none", "scenario": "F", "synthetic_flag": "SYNTHETIC",
        })
        rows_tp.append({
            "patient_id": pid, "tumor_subtype": subtype, "baseline_stage": stage,
            "baseline_tumor_size_mm": round(base, 1),
            "lymph_node_involvement": "TRUE" if stage in ("Stage III", "Stage IV") else "FALSE",
            "baseline_metastasis_status": "TRUE" if met else "FALSE",
            "er_percent": 50.0, "pr_percent": 30.0, "her2_score": 1,
            "ki67_percent": 25.0, "synthetic_flag": "SYNTHETIC",
        })

    patients = pd.DataFrame(rows_p)
    tumor_profiles = pd.DataFrame(rows_tp)

    visits, events, decisions, history, outcomes, transitions = [], [], [], [], [], []
    ev_id = 1
    tr_id = 1
    for p in patients.itertuples(index=False):
        base = float(p.baseline_tumor_size_mm)
        current_pathway = (
            "HER2_PATHWAY_REVIEW" if p.tumor_subtype == "HER2_POS"
            else "HORMONE_RECEPTOR_PATHWAY_REVIEW" if p.tumor_subtype == "HR_POS_HER2_NEG"
            else "SYSTEMIC_PATHWAY_REVIEW"
        )
        prev_pathway = ""
        sizes = base * np.exp(-0.05 * np.arange(n_visits))
        sizes = np.where(sizes < 5, 0.0, sizes)
        was_undetectable = False
        state_dicts = []
        for t in range(n_visits):
            vid = f"{p.patient_id}_V{t+1:02d}"
            size = round(float(sizes[t]), 2)
            rel_change = 0.0 if t == 0 else (size - float(sizes[t-1])) / max(float(sizes[t-1]), 1e-9)
            progression = (t > 0 and float(sizes[t]) > 1.2 * float(sizes[t-1]))
            recurrence = size > 5 and was_undetectable
            state = {
                "visit_number": t + 1, "tumor_size_mm": size,
                "tumor_size_trend": round(float(rel_change), 4),
                "disease_status": "STABLE" if size > 0 else "REMISSION_LIKE",
                "metastasis_status": bool(p.baseline_metastasis_status == "TRUE"),
                "metastasis_new": False, "toxicity_grade": 0,
                "current_pathway": current_pathway, "previous_response": "NA",
                "age": int(p.age), "baseline_stage": p.baseline_stage,
                "tumor_subtype": p.tumor_subtype,
                "performance_status": int(p.performance_status),
                "progression_flag": bool(progression),
                "recurrence_flag": bool(recurrence),
                "uncertainty_flag": False,
            }
            state_dicts.append(state)
            action = ("SWITCH_PATHWAY" if progression else
                      "MULTIDISCIPLINARY_REASSESSMENT" if recurrence else
                      "CONTINUE_CURRENT_PATHWAY")
            next_pathway = current_pathway
            if action == "SWITCH_PATHWAY":
                next_pathway = "SYSTEMIC_PATHWAY_REVIEW"
            elif action == "MULTIDISCIPLINARY_REASSESSMENT":
                next_pathway = "MULTIDISCIPLINARY_REASSESSMENT"
            visits.append({
                "visit_id": vid, "patient_id": p.patient_id, "visit_number": t + 1,
                "visit_date": (datetime(2020, 1, 1) + timedelta(days=30 * t)).strftime("%Y-%m-%d"),
                "tumor_size_mm": size,
                "lymph_node_status": p.lymph_node_involvement,
                "metastasis_status": p.baseline_metastasis_status,
                "disease_status": state["disease_status"],
                "er_percent": 50.0, "pr_percent": 30.0, "her2_score": 1, "ki67_percent": 25.0,
                "wbc": 6.0, "hemoglobin": 12.5, "platelets": 250.0,
                "alt": 25.0, "ast": 25.0, "creatinine": 0.9,
                "toxicity_grade": 0, "toxicity_type": "NONE",
                "current_pathway": current_pathway,
                "response_category": "SD" if size > 0 else "CR_LIKE",
                "recurrence_status": "TRUE" if recurrence else "FALSE",
                "progression_status": "TRUE" if progression else "FALSE",
                "synthetic_flag": "SYNTHETIC",
            })
            decisions.append({
                "patient_id": p.patient_id, "visit_id": vid, "visit_number": t + 1,
                "current_pathway": current_pathway, "action": action,
                "next_pathway": next_pathway, "policy_reason": "test",
                "synthetic_flag": "SYNTHETIC",
            })
            history.append({
                "patient_id": p.patient_id, "visit_id": vid, "visit_number": t + 1,
                "pathway": current_pathway, "previous_pathway": prev_pathway,
                "change_flag": "TRUE" if prev_pathway and prev_pathway != current_pathway else "FALSE",
                "synthetic_flag": "SYNTHETIC",
            })
            if t == 0:
                events.append({
                    "event_id": f"E{ev_id:06d}", "patient_id": p.patient_id, "visit_id": vid,
                    "event_date": (datetime(2020, 1, 1)).strftime("%Y-%m-%d"),
                    "event_type": "INITIAL_DIAGNOSIS", "severity_or_magnitude": base,
                    "synthetic_event_description": "Test fixture.",
                })
                ev_id += 1
            prev_pathway = current_pathway
            current_pathway = next_pathway
            was_undetectable = size == 0.0

        outcomes.append({
            "patient_id": p.patient_id, "best_response": "SD", "final_response": "SD",
            "progression_ever": "FALSE", "recurrence_ever": "FALSE",
            "metastasis_ever": p.baseline_metastasis_status,
            "max_toxicity_grade": 0, "final_tumor_size_mm": visits[-1]["tumor_size_mm"],
            "final_disease_status": visits[-1]["disease_status"],
            "n_visits": n_visits, "scenario": "F", "synthetic_flag": "SYNTHETIC",
        })

        for i in range(n_visits - 1):
            cur = visits[-n_visits + i]
            nxt = visits[-n_visits + i + 1]
            transitions.append({
                "transition_id": f"T{tr_id:06d}", "patient_id": p.patient_id,
                "current_visit_id": cur["visit_id"], "next_visit_id": nxt["visit_id"],
                "current_visit_number": cur["visit_number"],
                "next_visit_number": nxt["visit_number"],
                "state_json": json.dumps(state_dicts[i], sort_keys=True),
                "action": decisions[-n_visits + i]["action"],
                "reward": 1.0,
                "next_state_json": json.dumps(state_dicts[i + 1], sort_keys=True),
                "terminated": "TRUE" if i == n_visits - 2 else "FALSE",
            })
            tr_id += 1

    patients.to_csv(os.path.join(out_dir, "patients.csv"), index=False)
    tumor_profiles.to_csv(os.path.join(out_dir, "tumor_profiles.csv"), index=False)
    pd.DataFrame(visits).to_csv(os.path.join(out_dir, "clinical_visits.csv"), index=False)
    pd.DataFrame(events).to_csv(os.path.join(out_dir, "clinical_events.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "treatment_history.csv"), index=False)
    pd.DataFrame(decisions).to_csv(os.path.join(out_dir, "treatment_decisions.csv"), index=False)
    pd.DataFrame(outcomes).to_csv(os.path.join(out_dir, "outcomes.csv"), index=False)
    pd.DataFrame(transitions).to_csv(os.path.join(out_dir, "rl_transitions.csv"), index=False)

    meta = {
        "dataset_version": "fixture-1.0.0",
        "random_seed": seed,
        "patient_count": n_patients,
        "visits_per_patient": n_visits,
        "cancer_type": "breast_cancer_only",
        "synthetic_disclaimer": "SYNTHETIC test fixture — not real data.",
    }
    with open(os.path.join(out_dir, "dataset_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Test fixture written to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tests/fixture_data")
    p.add_argument("--patients", type=int, default=8)
    p.add_argument("--visits", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    make(a.out, a.patients, a.visits, a.seed)