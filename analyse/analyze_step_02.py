#!/usr/bin/env python3
"""
OncoMARL - Step 02
Data quality, relational integrity and RL transition validation.

Input data are read-only. Reports are written to a separate directory.
This script validates synthetic data structure; it does not establish
clinical validity or treatment effectiveness.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def load_csv(data_dir, filename):
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def add_issue(issues, category, table, column, count, details):
    if count:
        issues.append({
            "category": category,
            "table": table,
            "column": column,
            "issue_count": int(count),
            "details": details,
        })


def check_unique(df, table, column, issues):
    if column not in df.columns:
        add_issue(
            issues, "MISSING_COLUMN", table, column, 1,
            "Expected column does not exist."
        )
        return

    duplicate_count = int(df[column].duplicated().sum())
    add_issue(
        issues, "DUPLICATE_ID", table, column, duplicate_count,
        "Identifier should be unique within this table."
    )

    missing_count = int(df[column].isna().sum())
    add_issue(
        issues, "MISSING_ID", table, column, missing_count,
        "Identifier must not be missing."
    )


def check_foreign_key(child, child_table, child_col,
                      parent, parent_table, parent_col, issues):
    if child_col not in child.columns or parent_col not in parent.columns:
        add_issue(
            issues, "FK_CHECK_SKIPPED", child_table, child_col, 1,
            f"Missing column; cannot compare with "
            f"{parent_table}.{parent_col}."
        )
        return

    parent_values = set(parent[parent_col].dropna())
    unmatched = ~child[child_col].isin(parent_values)

    add_issue(
        issues, "ORPHAN_FOREIGN_KEY", child_table, child_col,
        int(unmatched.sum()),
        f"Values not found in {parent_table}.{parent_col}."
    )


def check_numeric_range(df, table, column, low, high, issues):
    if column not in df.columns:
        return

    values = pd.to_numeric(df[column], errors="coerce")
    invalid = values.notna() & (
        (values < low) | (values > high)
    )

    add_issue(
        issues, "OUT_OF_RANGE", table, column, int(invalid.sum()),
        f"Expected values between {low} and {high}, inclusive."
    )

    non_numeric = df[column].notna() & values.isna()
    add_issue(
        issues, "NON_NUMERIC_VALUE", table, column,
        int(non_numeric.sum()),
        "Non-empty values could not be converted to numbers."
    )


def check_json_column(df, table, column, issues):
    if column not in df.columns:
        add_issue(
            issues, "MISSING_COLUMN", table, column, 1,
            "JSON column does not exist."
        )
        return

    invalid_count = 0
    non_object_count = 0

    for value in df[column].dropna():
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, (dict, list)):
                non_object_count += 1
        except (json.JSONDecodeError, TypeError):
            invalid_count += 1

    add_issue(
        issues, "INVALID_JSON", table, column, invalid_count,
        "Values cannot be parsed as JSON."
    )
    add_issue(
        issues, "UNEXPECTED_JSON_TYPE", table, column,
        non_object_count,
        "Parsed JSON is neither an object nor a list."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="../data/oncomarl_breast",
        help="Directory containing the OncoMARL CSV files."
    )
    parser.add_argument(
        "--out",
        default="reports/step_02",
        help="Output directory for validation reports."
    )
    args = parser.parse_args()

    data_dir = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {data_dir}")
    print(f"Output directory: {out_dir}")

    # ---------------------------------------------------------
    # 1. Load tables
    # ---------------------------------------------------------
    filenames = [
        "patients.csv",
        "tumor_profiles.csv",
        "clinical_visits.csv",
        "clinical_events.csv",
        "treatment_history.csv",
        "treatment_decisions.csv",
        "outcomes.csv",
        "rl_transitions.csv",
    ]

    tables = {name[:-4]: load_csv(data_dir, name)
              for name in filenames}

    patients = tables["patients"]
    profiles = tables["tumor_profiles"]
    visits = tables["clinical_visits"]
    events = tables["clinical_events"]
    history = tables["treatment_history"]
    decisions = tables["treatment_decisions"]
    outcomes = tables["outcomes"]
    transitions = tables["rl_transitions"]

    issues = []
    metrics = []

    def metric(name, value, details=""):
        metrics.append({
            "metric": name,
            "value": value,
            "details": details,
        })

    # ---------------------------------------------------------
    # 2. Basic row counts and identifier uniqueness
    # ---------------------------------------------------------
    for name, df in tables.items():
        metric(f"{name}_rows", len(df))

    unique_checks = {
        "patients": ["patient_id"],
        "tumor_profiles": ["patient_id"],
        "clinical_visits": ["visit_id"],
        "clinical_events": ["event_id"],
        "treatment_history": ["record_id"],
        "treatment_decisions": ["decision_id", "visit_id"],
        "outcomes": ["patient_id"],
        "rl_transitions": ["transition_id"],
    }

    for table_name, columns in unique_checks.items():
        df = tables[table_name]
        for column in columns:
            check_unique(df, table_name, column, issues)

    # ---------------------------------------------------------
    # 3. Foreign-key integrity
    # ---------------------------------------------------------
    patient_tables = [
        "tumor_profiles",
        "clinical_visits",
        "clinical_events",
        "treatment_history",
        "treatment_decisions",
        "outcomes",
        "rl_transitions",
    ]

    for table_name in patient_tables:
        check_foreign_key(
            tables[table_name], table_name, "patient_id",
            patients, "patients", "patient_id", issues
        )

    check_foreign_key(
        events, "clinical_events", "visit_id",
        visits, "clinical_visits", "visit_id", issues
    )
    check_foreign_key(
        history, "treatment_history", "visit_id",
        visits, "clinical_visits", "visit_id", issues
    )
    check_foreign_key(
        decisions, "treatment_decisions", "visit_id",
        visits, "clinical_visits", "visit_id", issues
    )

    check_foreign_key(
        transitions, "rl_transitions", "current_visit_id",
        visits, "clinical_visits", "visit_id", issues
    )
    check_foreign_key(
        transitions, "rl_transitions", "next_visit_id",
        visits, "clinical_visits", "visit_id", issues
    )

    # ---------------------------------------------------------
    # 4. Verify patient and visit consistency
    # ---------------------------------------------------------
    visit_patient = visits[["visit_id", "patient_id"]].copy()

    for table_name, df in [
        ("clinical_events", events),
        ("treatment_history", history),
        ("treatment_decisions", decisions),
    ]:
        if {"visit_id", "patient_id"}.issubset(df.columns):
            joined = df.merge(
                visit_patient,
                on="visit_id",
                how="left",
                suffixes=("_record", "_visit"),
                indicator=True,
            )

            mismatch = (
                joined["_merge"].eq("both")
                & joined["patient_id_record"].ne(
                    joined["patient_id_visit"]
                )
            )

            add_issue(
                issues, "PATIENT_VISIT_MISMATCH",
                table_name, "patient_id",
                int(mismatch.sum()),
                "Record patient_id differs from the patient "
                "assigned to its visit_id."
            )

    # ---------------------------------------------------------
    # 5. Visit counts and visit-number sequences
    # ---------------------------------------------------------
    visit_counts = visits.groupby("patient_id").size()
    expected_patients = set(patients["patient_id"].dropna())
    observed_patients = set(visit_counts.index)

    missing_visit_patients = expected_patients - observed_patients
    unexpected_visit_patients = observed_patients - expected_patients

    add_issue(
        issues, "PATIENT_WITHOUT_VISITS", "clinical_visits",
        "patient_id", len(missing_visit_patients),
        "Patients in patients.csv without any clinical visits."
    )
    add_issue(
        issues, "UNEXPECTED_PATIENT", "clinical_visits",
        "patient_id", len(unexpected_visit_patients),
        "Visit patient IDs not present in patients.csv."
    )

    visit_count_distribution = (
        visit_counts.value_counts()
        .sort_index()
        .rename_axis("visits_per_patient")
        .reset_index(name="patient_count")
    )
    visit_count_distribution.to_csv(
        out_dir / "visit_count_distribution.csv", index=False
    )

    patients_not_12 = int((visit_counts != 12).sum())
    add_issue(
        issues, "VISIT_COUNT_NOT_12", "clinical_visits",
        "patient_id", patients_not_12,
        "Patients whose visit count differs from 12. "
        "Check metadata before deciding whether this is an error."
    )

    if "visit_number" in visits.columns:
        visit_numbers = pd.to_numeric(
            visits["visit_number"], errors="coerce"
        )

        invalid_visit_number = (
            visit_numbers.isna()
            | (visit_numbers < 1)
            | (visit_numbers > 12)
            | (visit_numbers % 1 != 0)
        )

        add_issue(
            issues, "INVALID_VISIT_NUMBER", "clinical_visits",
            "visit_number", int(invalid_visit_number.sum()),
            "Expected integer visit numbers from 1 through 12."
        )

        duplicate_patient_visit = visits.duplicated(
            ["patient_id", "visit_number"], keep=False
        )

        add_issue(
            issues, "DUPLICATE_PATIENT_VISIT_NUMBER",
            "clinical_visits", "visit_number",
            int(duplicate_patient_visit.sum()),
            "Multiple rows share the same patient_id and visit_number."
        )

        sequence_rows = []
        for patient_id, group in visits.groupby("patient_id"):
            nums = sorted(
                pd.to_numeric(
                    group["visit_number"], errors="coerce"
                ).dropna().astype(int).tolist()
            )
            expected = list(range(1, 13))
            if nums != expected:
                sequence_rows.append({
                    "patient_id": patient_id,
                    "observed_visit_numbers": json.dumps(nums),
                    "expected_visit_numbers": json.dumps(expected),
                })

        pd.DataFrame(
            sequence_rows,
            columns=[
                "patient_id",
                "observed_visit_numbers",
                "expected_visit_numbers",
            ],
        ).to_csv(
            out_dir / "visit_sequence_anomalies.csv", index=False
        )

        metric(
            "patients_with_nonstandard_visit_sequence",
            len(sequence_rows)
        )

    metric("patients_in_patients_table", patients["patient_id"].nunique())
    metric("patients_with_visits", visits["patient_id"].nunique())
    metric("total_visits", len(visits))
    metric("patients_not_having_12_visits", patients_not_12)

    # ---------------------------------------------------------
    # 6. Date parsing and chronological order
    # ---------------------------------------------------------
    date_columns = [
        ("patients", patients, "diagnosis_date"),
        ("clinical_visits", visits, "visit_date"),
        ("clinical_events", events, "event_date"),
    ]

    parsed_dates = {}

    for table_name, df, column in date_columns:
        if column not in df.columns:
            add_issue(
                issues, "MISSING_COLUMN", table_name, column, 1,
                "Expected date column does not exist."
            )
            continue

        parsed = pd.to_datetime(df[column], errors="coerce")
        parsed_dates[table_name] = parsed

        invalid_count = int(parsed.isna().sum())
        add_issue(
            issues, "INVALID_DATE", table_name, column,
            invalid_count,
            "Date could not be parsed."
        )

    if "clinical_visits" in parsed_dates:
        date_frame = visits[["patient_id", "visit_number"]].copy()
        date_frame["_parsed_date"] = parsed_dates["clinical_visits"]

        backwards_count = 0
        for _, group in date_frame.groupby("patient_id"):
            group = group.sort_values("visit_number")
            valid_dates = group["_parsed_date"].dropna()
            backwards_count += int(
                (valid_dates.diff().dropna() < pd.Timedelta(0)).sum()
            )

        add_issue(
            issues, "NON_CHRONOLOGICAL_VISITS",
            "clinical_visits", "visit_date", backwards_count,
            "A visit date is earlier than the previous dated visit "
            "for the same patient when ordered by visit_number."
        )

    # ---------------------------------------------------------
    # 7. Numeric range checks
    # ---------------------------------------------------------
    range_checks = [
        (patients, "patients", "age", 0, 120),
        (patients, "patients", "baseline_tumor_size_mm", 0, 500),
        (patients, "patients", "performance_status", 0, 5),

        (profiles, "tumor_profiles", "baseline_tumor_size_mm", 0, 500),
        (profiles, "tumor_profiles", "er_percent", 0, 100),
        (profiles, "tumor_profiles", "pr_percent", 0, 100),
        (profiles, "tumor_profiles", "ki67_percent", 0, 100),
        (profiles, "tumor_profiles", "her2_score", 0, 3),

        (visits, "clinical_visits", "tumor_size_mm", 0, 500),
        (visits, "clinical_visits", "er_percent", 0, 100),
        (visits, "clinical_visits", "pr_percent", 0, 100),
        (visits, "clinical_visits", "ki67_percent", 0, 100),
        (visits, "clinical_visits", "her2_score", 0, 3),
        (visits, "clinical_visits", "toxicity_grade", 0, 4),

        (outcomes, "outcomes", "final_tumor_size_mm", 0, 500),
        (outcomes, "outcomes", "max_toxicity_grade", 0, 4),
    ]

    for df, table_name, column, low, high in range_checks:
        check_numeric_range(
            df, table_name, column, low, high, issues
        )

    # ---------------------------------------------------------
    # 8. Check treatment history's first-visit missing values
    # ---------------------------------------------------------
    if {"previous_pathway", "visit_number"}.issubset(history.columns):
        previous_missing = history["previous_pathway"].isna()
        first_visit = pd.to_numeric(
            history["visit_number"], errors="coerce"
        ).eq(1)

        missing_on_first = int((previous_missing & first_visit).sum())
        missing_after_first = int(
            (previous_missing & ~first_visit).sum()
        )

        metric(
            "previous_pathway_missing_on_visit_1",
            missing_on_first
        )
        metric(
            "previous_pathway_missing_after_visit_1",
            missing_after_first
        )

        add_issue(
            issues, "MISSING_PREVIOUS_PATHWAY_AFTER_FIRST_VISIT",
            "treatment_history", "previous_pathway",
            missing_after_first,
            "Missing previous_pathway values on records that are "
            "not visit 1. Review whether these are expected."
        )

    # ---------------------------------------------------------
    # 9. RL transition validation
    # ---------------------------------------------------------
    visit_lookup = visits[
        ["visit_id", "patient_id", "visit_number"]
    ].copy()

    current_lookup = visit_lookup.rename(columns={
        "visit_id": "current_visit_id",
        "patient_id": "current_patient_id",
        "visit_number": "actual_current_visit_number",
    })

    next_lookup = visit_lookup.rename(columns={
        "visit_id": "next_visit_id",
        "patient_id": "next_patient_id",
        "visit_number": "actual_next_visit_number",
    })

    rl = transitions.merge(
        current_lookup,
        on="current_visit_id",
        how="left",
        indicator="_current_merge",
    )
    rl = rl.merge(
        next_lookup,
        on="next_visit_id",
        how="left",
        indicator="_next_merge",
    )

    missing_current = int((rl["_current_merge"] != "both").sum())
    missing_next = int((rl["_next_merge"] != "both").sum())

    add_issue(
        issues, "RL_CURRENT_VISIT_NOT_FOUND", "rl_transitions",
        "current_visit_id", missing_current,
        "Current visit ID does not match clinical_visits."
    )
    add_issue(
        issues, "RL_NEXT_VISIT_NOT_FOUND", "rl_transitions",
        "next_visit_id", missing_next,
        "Next visit ID does not match clinical_visits."
    )

    both_found = (
        rl["_current_merge"].eq("both")
        & rl["_next_merge"].eq("both")
    )

    patient_mismatch = both_found & (
        rl["patient_id"].ne(rl["current_patient_id"])
        | rl["patient_id"].ne(rl["next_patient_id"])
    )

    add_issue(
        issues, "RL_PATIENT_MISMATCH", "rl_transitions",
        "patient_id", int(patient_mismatch.sum()),
        "Transition patient differs from current or next visit patient."
    )

    current_number_mismatch = both_found & (
        pd.to_numeric(rl["current_visit_number"], errors="coerce")
        != pd.to_numeric(
            rl["actual_current_visit_number"], errors="coerce"
        )
    )

    next_number_mismatch = both_found & (
        pd.to_numeric(rl["next_visit_number"], errors="coerce")
        != pd.to_numeric(
            rl["actual_next_visit_number"], errors="coerce"
        )
    )

    add_issue(
        issues, "RL_CURRENT_VISIT_NUMBER_MISMATCH",
        "rl_transitions", "current_visit_number",
        int(current_number_mismatch.sum()),
        "Transition visit number differs from clinical_visits."
    )
    add_issue(
        issues, "RL_NEXT_VISIT_NUMBER_MISMATCH",
        "rl_transitions", "next_visit_number",
        int(next_number_mismatch.sum()),
        "Transition next visit number differs from clinical_visits."
    )

    consecutive = both_found & (
        pd.to_numeric(rl["next_visit_number"], errors="coerce")
        == pd.to_numeric(rl["current_visit_number"], errors="coerce") + 1
    )

    add_issue(
        issues, "RL_NONCONSECUTIVE_VISIT", "rl_transitions",
        "next_visit_number", int((both_found & ~consecutive).sum()),
        "Current and next visit numbers are not consecutive."
    )

    reward = pd.to_numeric(transitions["reward"], errors="coerce")
    add_issue(
        issues, "RL_INVALID_REWARD", "rl_transitions", "reward",
        int((reward.isna() | ~np.isfinite(reward)).sum()),
        "Reward is missing, non-numeric, or not finite."
    )

    check_json_column(
        transitions, "rl_transitions", "state_json", issues
    )
    check_json_column(
        transitions, "rl_transitions", "next_state_json", issues
    )

    # ---------------------------------------------------------
    # 10. Boolean and categorical consistency
    # ---------------------------------------------------------
    for table_name, df, column in [
        ("clinical_visits", visits, "synthetic_flag"),
        ("patients", patients, "synthetic_flag"),
        ("outcomes", outcomes, "synthetic_flag"),
        ("clinical_visits", visits, "recurrence_status"),
        ("clinical_visits", visits, "progression_status"),
        ("clinical_visits", visits, "metastasis_status"),
        ("rl_transitions", transitions, "terminated"),
    ]:
        if column in df.columns:
            metric(
                f"{table_name}_{column}_unique_values",
                json.dumps(
                    df[column].dropna().astype(str)
                    .value_counts().to_dict(),
                    ensure_ascii=False
                )
            )

    # ---------------------------------------------------------
    # 11. Save reports
    # ---------------------------------------------------------
    issues_df = pd.DataFrame(
        issues,
        columns=["category", "table", "column", "issue_count", "details"]
    )

    metrics_df = pd.DataFrame(
        metrics, columns=["metric", "value", "details"]
    )

    issues_df.to_csv(out_dir / "data_quality_issues.csv", index=False)
    metrics_df.to_csv(out_dir / "quality_metrics.csv", index=False)

    issue_summary = (
        issues_df.groupby("category", as_index=False)["issue_count"]
        .sum()
        .sort_values("category")
        if not issues_df.empty
        else pd.DataFrame(columns=["category", "issue_count"])
    )
    issue_summary.to_csv(
        out_dir / "issue_summary_by_category.csv", index=False
    )

    lines = [
        "=" * 68,
        "ONCOMARL - STEP 02: DATA QUALITY VALIDATION",
        "=" * 68,
        f"Input directory: {data_dir}",
        f"Output directory: {out_dir}",
        "",
        "TABLE ROW COUNTS",
        "-" * 68,
    ]

    for name, df in tables.items():
        lines.append(f"{name}: {len(df):,} rows")

    lines.extend([
        "",
        "VALIDATION SUMMARY",
        "-" * 68,
        f"Total issue categories/records: {len(issues_df)}",
    ])

    if issues_df.empty:
        lines.append("No issues were detected by the implemented checks.")
    else:
        for _, row in issue_summary.iterrows():
            lines.append(
                f"{row['category']}: {int(row['issue_count']):,}"
            )

    lines.extend([
        "",
        "IMPORTANT INTERPRETATION",
        "-" * 68,
        "A detected issue is a signal to investigate, not always proof "
        "of an error.",
        "Some range limits are broad screening rules and must be reviewed.",
        "Synthetic data validation does not establish clinical validity.",
        "Original data files were not modified.",
        "",
        "Generated reports:",
        "- data_quality_issues.csv",
        "- quality_metrics.csv",
        "- issue_summary_by_category.csv",
        "- visit_count_distribution.csv",
        "- visit_sequence_anomalies.csv",
    ])

    summary_path = out_dir / "validation_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()