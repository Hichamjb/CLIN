#!/usr/bin/env python3

"""
OncoMARL - Step 03
Descriptive statistical analysis of a synthetic breast cancer dataset.

Outputs:
- Cohort statistics
- Patient and tumor distributions
- Clinical outcome distributions
- Tumor trajectory summaries
- Treatment and scenario distributions
- RL reward statistics
- Tumor-size outlier audit
- PNG figures

This is descriptive analysis of synthetic data, not clinical evidence.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(data_dir, filename):
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)


def save_distribution(df, column, output_path):
    """Save category counts and percentages."""
    if column not in df.columns:
        return

    result = (
        df[column]
        .fillna("MISSING")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )

    total = result["count"].sum()
    result["percentage"] = (
        result["count"] / total * 100 if total else 0
    )

    result.to_csv(output_path, index=False)


def save_bar_chart(df, column, output_path, title):
    if column not in df.columns:
        return

    counts = (
        df[column]
        .fillna("MISSING")
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(11, 6))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Number of records")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def numeric_summary(df, columns):
    results = []

    for column in columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")

        results.append({
            "column": column,
            "count": int(values.count()),
            "missing": int(values.isna().sum()),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "p95": values.quantile(0.95),
            "p99": values.quantile(0.99),
            "max": values.max(),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/oncomarl_breast",
        help="Input dataset directory",
    )

    parser.add_argument(
        "--out",
        default="reports/step_03",
        help="Output directory",
    )

    args = parser.parse_args()

    data_dir = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")

    patients = load_csv(data_dir, "patients.csv")
    profiles = load_csv(data_dir, "tumor_profiles.csv")
    visits = load_csv(data_dir, "clinical_visits.csv")
    events = load_csv(data_dir, "clinical_events.csv")
    history = load_csv(data_dir, "treatment_history.csv")
    decisions = load_csv(data_dir, "treatment_decisions.csv")
    outcomes = load_csv(data_dir, "outcomes.csv")
    transitions = load_csv(data_dir, "rl_transitions.csv")

    # -----------------------------------------------------
    # 1. Dataset overview
    # -----------------------------------------------------

    overview = pd.DataFrame([
        {"table": "patients", "rows": len(patients)},
        {"table": "tumor_profiles", "rows": len(profiles)},
        {"table": "clinical_visits", "rows": len(visits)},
        {"table": "clinical_events", "rows": len(events)},
        {"table": "treatment_history", "rows": len(history)},
        {"table": "treatment_decisions", "rows": len(decisions)},
        {"table": "outcomes", "rows": len(outcomes)},
        {"table": "rl_transitions", "rows": len(transitions)},
    ])

    overview.to_csv(
        tables_dir / "dataset_overview.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 2. Patient demographics
    # -----------------------------------------------------

    numeric_summary(
        patients,
        ["age", "baseline_tumor_size_mm", "performance_status"],
    ).to_csv(
        tables_dir / "patient_numeric_summary.csv",
        index=False,
    )

    patient_categories = [
        "sex",
        "baseline_stage",
        "tumor_subtype",
        "lymph_node_involvement",
        "baseline_metastasis_status",
        "performance_status",
        "scenario",
    ]

    for column in patient_categories:
        save_distribution(
            patients,
            column,
            tables_dir / f"patients_{column}_distribution.csv",
        )

        save_bar_chart(
            patients,
            column,
            figures_dir / f"patients_{column}.png",
            f"Patient distribution: {column}",
        )

    # -----------------------------------------------------
    # 3. Tumor profile distributions
    # -----------------------------------------------------

    tumor_numeric_columns = [
        "baseline_tumor_size_mm",
        "er_percent",
        "pr_percent",
        "her2_score",
        "ki67_percent",
    ]

    numeric_summary(
        profiles,
        tumor_numeric_columns,
    ).to_csv(
        tables_dir / "tumor_profile_numeric_summary.csv",
        index=False,
    )

    for column in tumor_numeric_columns:
        values = pd.to_numeric(profiles[column], errors="coerce")

        plt.figure(figsize=(9, 5))
        values.dropna().plot(kind="hist", bins=40)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Number of patients")
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"tumor_profile_{column}.png",
            dpi=150,
        )
        plt.close()

    # -----------------------------------------------------
    # 4. Clinical visit statistics
    # -----------------------------------------------------

    visit_numeric_columns = [
        "tumor_size_mm",
        "er_percent",
        "pr_percent",
        "her2_score",
        "ki67_percent",
        "toxicity_grade",
        "wbc",
        "hemoglobin",
        "platelets",
        "alt",
        "ast",
        "creatinine",
    ]

    numeric_summary(
        visits,
        visit_numeric_columns,
    ).to_csv(
        tables_dir / "clinical_visits_numeric_summary.csv",
        index=False,
    )

    visit_categories = [
        "disease_status",
        "response_category",
        "toxicity_type",
        "current_pathway",
        "recurrence_status",
        "progression_status",
        "metastasis_status",
    ]

    for column in visit_categories:
        save_distribution(
            visits,
            column,
            tables_dir / f"visits_{column}_distribution.csv",
        )

    # Mean tumor size at each visit number.
    trajectory = (
        visits.assign(
            tumor_size_mm=pd.to_numeric(
                visits["tumor_size_mm"], errors="coerce"
            )
        )
        .groupby("visit_number")
        .agg(
            patients=("patient_id", "nunique"),
            records=("tumor_size_mm", "count"),
            mean_tumor_size_mm=("tumor_size_mm", "mean"),
            median_tumor_size_mm=("tumor_size_mm", "median"),
            std_tumor_size_mm=("tumor_size_mm", "std"),
            p25_tumor_size_mm=(
                "tumor_size_mm",
                lambda x: x.quantile(0.25),
            ),
            p75_tumor_size_mm=(
                "tumor_size_mm",
                lambda x: x.quantile(0.75),
            ),
        )
        .reset_index()
        .sort_values("visit_number")
    )

    trajectory.to_csv(
        tables_dir / "tumor_trajectory_by_visit.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        trajectory["visit_number"],
        trajectory["mean_tumor_size_mm"],
        marker="o",
        label="Mean",
    )
    plt.plot(
        trajectory["visit_number"],
        trajectory["median_tumor_size_mm"],
        marker="s",
        label="Median",
    )
    plt.xlabel("Visit number")
    plt.ylabel("Tumor size (mm)")
    plt.title("Tumor size across clinical visits")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "tumor_trajectory.png",
        dpi=150,
    )
    plt.close()

    # -----------------------------------------------------
    # 5. Outcome statistics
    # -----------------------------------------------------

    outcome_categories = [
        "best_response",
        "final_response",
        "progression_ever",
        "recurrence_ever",
        "metastasis_ever",
        "final_disease_status",
        "scenario",
    ]

    for column in outcome_categories:
        save_distribution(
            outcomes,
            column,
            tables_dir / f"outcomes_{column}_distribution.csv",
        )

        save_bar_chart(
            outcomes,
            column,
            figures_dir / f"outcomes_{column}.png",
            f"Outcome distribution: {column}",
        )

    numeric_summary(
        outcomes,
        [
            "final_tumor_size_mm",
            "max_toxicity_grade",
            "n_visits",
        ],
    ).to_csv(
        tables_dir / "outcomes_numeric_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 6. Treatment actions and pathways
    # -----------------------------------------------------

    for table_name, df, column in [
        ("treatment_decisions", decisions, "action"),
        ("treatment_decisions", decisions, "current_pathway"),
        ("treatment_decisions", decisions, "next_pathway"),
        ("treatment_history", history, "pathway"),
        ("rl_transitions", transitions, "action"),
    ]:
        save_distribution(
            df,
            column,
            tables_dir / f"{table_name}_{column}_distribution.csv",
        )

    save_bar_chart(
        decisions,
        "action",
        figures_dir / "treatment_actions.png",
        "Treatment action distribution",
    )

    save_bar_chart(
        history,
        "pathway",
        figures_dir / "treatment_pathways.png",
        "Treatment pathway distribution",
    )

    # -----------------------------------------------------
    # 7. RL transition statistics
    # -----------------------------------------------------

    reward = pd.to_numeric(
        transitions["reward"], errors="coerce"
    )

    reward_stats = pd.DataFrame([{
        "count": int(reward.count()),
        "missing": int(reward.isna().sum()),
        "mean": reward.mean(),
        "std": reward.std(),
        "min": reward.min(),
        "p25": reward.quantile(0.25),
        "median": reward.median(),
        "p75": reward.quantile(0.75),
        "p95": reward.quantile(0.95),
        "max": reward.max(),
    }])

    reward_stats.to_csv(
        tables_dir / "rl_reward_summary.csv",
        index=False,
    )

    save_distribution(
        transitions,
        "action",
        tables_dir / "rl_action_distribution.csv",
    )

    save_distribution(
        transitions,
        "terminated",
        tables_dir / "rl_terminated_distribution.csv",
    )

    plt.figure(figsize=(9, 5))
    reward.dropna().plot(kind="hist", bins=50)
    plt.title("Distribution of RL rewards")
    plt.xlabel("Reward")
    plt.ylabel("Number of transitions")
    plt.tight_layout()
    plt.savefig(
        figures_dir / "rl_reward_distribution.png",
        dpi=150,
    )
    plt.close()

    # -----------------------------------------------------
    # 8. Audit tumor sizes above 500 mm
    # -----------------------------------------------------

    visits_size = pd.to_numeric(
        visits["tumor_size_mm"], errors="coerce"
    )

    visit_outliers = visits.loc[
        visits_size > 500
    ].copy()

    visit_outliers.to_csv(
        tables_dir / "tumor_size_over_500_visits.csv",
        index=False,
    )

    outcomes_size = pd.to_numeric(
        outcomes["final_tumor_size_mm"], errors="coerce"
    )

    outcome_outliers = outcomes.loc[
        outcomes_size > 500
    ].copy()

    outcome_outliers.to_csv(
        tables_dir / "tumor_size_over_500_outcomes.csv",
        index=False,
    )

    # Record extrema and quantiles to help interpret outliers.
    outlier_audit = []

    for table_name, series in [
        ("clinical_visits.tumor_size_mm", visits_size),
        ("outcomes.final_tumor_size_mm", outcomes_size),
    ]:
        outlier_audit.append({
            "variable": table_name,
            "count": int(series.count()),
            "over_500_count": int((series > 500).sum()),
            "maximum": series.max(),
            "p95": series.quantile(0.95),
            "p99": series.quantile(0.99),
        })

    pd.DataFrame(outlier_audit).to_csv(
        tables_dir / "tumor_size_outlier_audit.csv",
        index=False,
    )

    # -----------------------------------------------------
    # 9. Write readable summary
    # -----------------------------------------------------

    lines = [
        "=" * 68,
        "ONCOMARL - STEP 03: DESCRIPTIVE STATISTICAL ANALYSIS",
        "=" * 68,
        "",
        f"Patients: {len(patients):,}",
        f"Clinical visits: {len(visits):,}",
        f"Clinical events: {len(events):,}",
        f"Treatment decisions: {len(decisions):,}",
        f"RL transitions: {len(transitions):,}",
        "",
        "AGE SUMMARY",
        "-" * 68,
        patients["age"].describe().to_string(),
        "",
        "BASELINE TUMOR SIZE SUMMARY (mm)",
        "-" * 68,
        patients["baseline_tumor_size_mm"].describe().to_string(),
        "",
        "FINAL TUMOR SIZE SUMMARY (mm)",
        "-" * 68,
        outcomes["final_tumor_size_mm"].describe().to_string(),
        "",
        "TUMOR SIZE OUTLIERS",
        "-" * 68,
        f"Visit records > 500 mm: {(visits_size > 500).sum():,}",
        f"Outcome records > 500 mm: {(outcomes_size > 500).sum():,}",
        "",
        "RL REWARD SUMMARY",
        "-" * 68,
        reward.describe().to_string(),
        "",
        "MEAN TUMOR SIZE BY VISIT",
        "-" * 68,
        trajectory[
            ["visit_number", "patients",
             "mean_tumor_size_mm", "median_tumor_size_mm"]
        ].to_string(index=False),
        "",
        "IMPORTANT",
        "-" * 68,
        "This analysis describes synthetic data only.",
        "Observed outcome proportions are not real-world clinical rates.",
        "Descriptive associations do not prove treatment effectiveness.",
        "No original data files were modified.",
        "",
        f"Tables directory: {tables_dir}",
        f"Figures directory: {figures_dir}",
    ]

    summary_path = out_dir / "analysis_summary.txt"
    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()