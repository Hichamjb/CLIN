#!/usr/bin/env python3

"""
OncoMARL - Step 04
Clinical consistency and RL reward audit.

This script is read-only:
- It does not modify original dataset files.
- It exports audit reports and discrepancy tables.
- All source data are synthetic and must not be interpreted
  as real clinical evidence.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(data_dir, filename):
    path = data_dir / filename

    if not path.exists():
        print(f"[WARNING] Missing file: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def save_csv(df, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / filename, index=False)


def compare_numeric(
    left,
    right,
    left_name,
    right_name,
    tolerance=1e-6,
):
    """
    Compare two numeric Series and return a discrepancy report.
    Rows containing missing values are reported separately.
    """

    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")

    missing = left_num.isna() | right_num.isna()
    difference = (left_num - right_num).abs()

    comparable = ~missing
    mismatch = comparable & (difference > tolerance)

    report = pd.DataFrame({
        left_name: left_num,
        right_name: right_num,
        "absolute_difference": difference,
        "missing_comparison_value": missing,
        "mismatch": mismatch,
    })

    return report


def safe_bool(series):
    """
    Convert common boolean representations to pandas nullable booleans.
    Unrecognized values become <NA>.
    """

    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        1: True,
        0: False,
        "1": True,
        "0": False,
    }

    return series.map(mapping).astype("boolean")


def summarize_boolean_comparison(
    visits,
    outcomes,
    patient_col="patient_id",
    visit_col="visit_number",
    visit_flag=None,
    outcome_flag=None,
):
    """
    Compare whether a flag was ever true in visits with an
    outcome-level 'ever' flag.
    """

    if (
        visits.empty
        or outcomes.empty
        or visit_flag not in visits.columns
        or outcome_flag not in outcomes.columns
    ):
        return pd.DataFrame()

    temp = visits[[patient_col, visit_flag]].copy()
    temp[visit_flag] = safe_bool(temp[visit_flag])

    # Ignore unknown values instead of interpreting them as False.
    ever_true = (
        temp.groupby(patient_col)[visit_flag]
        .agg(lambda x: bool((x == True).any()))
        .rename("visits_ever_true")
        .reset_index()
    )

    outcome_temp = outcomes[[patient_col, outcome_flag]].copy()
    outcome_temp[outcome_flag] = safe_bool(
        outcome_temp[outcome_flag]
    )

    merged = ever_true.merge(
        outcome_temp,
        on=patient_col,
        how="outer",
        validate="one_to_one",
    )

    merged["outcome_ever_true"] = merged[outcome_flag]
    merged["flag_disagreement"] = (
        merged["outcome_ever_true"].notna()
        & (
            merged["visits_ever_true"]
            != merged["outcome_ever_true"]
        )
    )

    merged["visit_flag_name"] = visit_flag
    merged["outcome_flag_name"] = outcome_flag

    return merged


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/oncomarl_breast",
        help="Directory containing the synthetic dataset",
    )

    parser.add_argument(
        "--out",
        default="reports/step_04",
        help="Output directory for audit results",
    )

    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    tables_dir = out_dir / "tables"

    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ONCOMARL - STEP 04")
    print("Clinical consistency and RL reward audit")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    patients = load_csv(data_dir, "patients.csv")
    visits = load_csv(data_dir, "clinical_visits.csv")
    outcomes = load_csv(data_dir, "outcomes.csv")
    transitions = load_csv(data_dir, "rl_transitions.csv")
    decisions = load_csv(data_dir, "treatment_decisions.csv")

    print("\n[1] DATA LOADING")

    for name, df in [
        ("patients", patients),
        ("clinical_visits", visits),
        ("outcomes", outcomes),
        ("rl_transitions", transitions),
        ("treatment_decisions", decisions),
    ]:
        print(f"{name}: {len(df):,} rows")

    report_lines = [
        "ONCOMARL - STEP 04 AUDIT",
        "=" * 60,
        "",
        "IMPORTANT: The dataset is synthetic.",
        "This audit does not establish clinical validity or treatment efficacy.",
        "",
    ]

    # ---------------------------------------------------------
    # 2. Check last visit versus outcome
    # ---------------------------------------------------------

    print("\n[2] LAST VISIT VS FINAL OUTCOME")

    if (
        not visits.empty
        and not outcomes.empty
        and "patient_id" in visits.columns
        and "patient_id" in outcomes.columns
        and "visit_number" in visits.columns
    ):
        visits_sorted = visits.sort_values(
            ["patient_id", "visit_number"]
        )

        last_visits = visits_sorted.groupby(
            "patient_id", as_index=False
        ).tail(1)

        last_visits = last_visits.copy()

        keep_visit_cols = [
            c for c in [
                "patient_id",
                "visit_number",
                "visit_date",
                "tumor_size_mm",
                "disease_status",
                "response_category",
                "progression_status",
                "recurrence_status",
                "metastasis_status",
                "synthetic_flag",
            ]
            if c in last_visits.columns
        ]

        keep_outcome_cols = [
            c for c in [
                "patient_id",
                "final_tumor_size_mm",
                "final_disease_status",
                "final_response",
                "progression_ever",
                "recurrence_ever",
                "metastasis_ever",
                "n_visits",
                "synthetic_flag",
            ]
            if c in outcomes.columns
        ]

        last_visits = last_visits[keep_visit_cols]

        merged = last_visits.merge(
            outcomes[keep_outcome_cols],
            on="patient_id",
            how="outer",
            suffixes=("_last_visit", "_outcome"),
            indicator=True,
            validate="one_to_one",
        )

        save_csv(
            merged,
            tables_dir,
            "last_visit_vs_outcome.csv",
        )

        report_lines.append("LAST VISIT VS FINAL OUTCOME")
        report_lines.append(
            f"Patients represented in comparison: {len(merged)}"
        )
        report_lines.append(
            f"Patients missing from one side: "
            f"{int((merged['_merge'] != 'both').sum())}"
        )

        # Tumor size comparison
        if (
            "tumor_size_mm" in merged.columns
            and "final_tumor_size_mm" in merged.columns
        ):
            comparison = compare_numeric(
                merged["tumor_size_mm"],
                merged["final_tumor_size_mm"],
                "last_visit_tumor_size_mm",
                "final_tumor_size_mm",
            )

            save_csv(
                comparison,
                tables_dir,
                "tumor_size_consistency.csv",
            )

            valid = comparison[
                ~comparison["missing_comparison_value"]
            ]

            mismatch_count = int(valid["mismatch"].sum())

            print(
                "Tumor size mismatches:",
                mismatch_count,
                "/",
                len(valid),
            )

            report_lines.append(
                f"Tumor size mismatches: {mismatch_count} / {len(valid)}"
            )

            report_lines.append(
                "Tumor size comparison tolerance: 1e-6 mm"
            )

        # Disease status comparison
        if (
            "disease_status" in merged.columns
            and "final_disease_status" in merged.columns
        ):
            left = merged["disease_status"].astype("string")
            right = merged["final_disease_status"].astype("string")

            merged["disease_status_mismatch"] = (
                left.notna()
                & right.notna()
                & (left != right)
            )

            disease_mismatches = merged[
                merged["disease_status_mismatch"]
            ]

            save_csv(
                disease_mismatches,
                tables_dir,
                "disease_status_mismatches.csv",
            )

            print(
                "Disease status mismatches:",
                len(disease_mismatches),
            )

            report_lines.append(
                f"Disease status mismatches: {len(disease_mismatches)}"
            )

        # Response comparison
        if (
            "response_category" in merged.columns
            and "final_response" in merged.columns
        ):
            left = merged["response_category"].astype("string")
            right = merged["final_response"].astype("string")

            response_mismatch = (
                left.notna()
                & right.notna()
                & (left != right)
            )

            response_rows = merged.loc[
                response_mismatch
            ].copy()

            save_csv(
                response_rows,
                tables_dir,
                "response_mismatches.csv",
            )

            print(
                "Response mismatches:",
                len(response_rows),
            )

            report_lines.append(
                f"Response mismatches: {len(response_rows)}"
            )

    else:
        report_lines.append(
            "Last visit comparison skipped: required columns unavailable."
        )

    # ---------------------------------------------------------
    # 3. Compare visit-level ever flags with outcome flags
    # ---------------------------------------------------------

    print("\n[3] EVER-FLAG CONSISTENCY")

    flag_pairs = [
        ("progression_status", "progression_ever"),
        ("recurrence_status", "recurrence_ever"),
        ("metastasis_status", "metastasis_ever"),
    ]

    for visit_flag, outcome_flag in flag_pairs:
        comparison = summarize_boolean_comparison(
            visits,
            outcomes,
            visit_flag=visit_flag,
            outcome_flag=outcome_flag,
        )

        if comparison.empty:
            print(
                f"{visit_flag} vs {outcome_flag}: skipped"
            )
            continue

        filename = (
            f"consistency_{visit_flag}_vs_{outcome_flag}.csv"
        )

        save_csv(comparison, tables_dir, filename)

        disagreement_count = int(
            comparison["flag_disagreement"].sum()
        )

        comparable_count = int(
            comparison["outcome_ever_true"].notna().sum()
        )

        print(
            f"{visit_flag} vs {outcome_flag}: "
            f"{disagreement_count} disagreements / "
            f"{comparable_count} comparable patients"
        )

        report_lines.append(
            f"{visit_flag} vs {outcome_flag}: "
            f"{disagreement_count} disagreements / "
            f"{comparable_count} comparable patients"
        )

    # ---------------------------------------------------------
    # 4. RL reward audit
    # ---------------------------------------------------------

    print("\n[4] RL REWARD AUDIT")

    if not transitions.empty and "reward" in transitions.columns:
        transitions["reward_numeric"] = pd.to_numeric(
            transitions["reward"],
            errors="coerce",
        )

        reward = transitions["reward_numeric"].dropna()

        if not reward.empty:
            reward_summary = pd.DataFrame([{
                "count": int(reward.count()),
                "missing": int(
                    transitions["reward_numeric"].isna().sum()
                ),
                "mean": float(reward.mean()),
                "std": float(reward.std()),
                "min": float(reward.min()),
                "p01": float(reward.quantile(0.01)),
                "p05": float(reward.quantile(0.05)),
                "p25": float(reward.quantile(0.25)),
                "median": float(reward.median()),
                "p75": float(reward.quantile(0.75)),
                "p95": float(reward.quantile(0.95)),
                "p99": float(reward.quantile(0.99)),
                "max": float(reward.max()),
                "unique_values": int(reward.nunique()),
                "fraction_equal_to_1": float(
                    np.isclose(reward, 1.0).mean()
                ),
            }])

            save_csv(
                reward_summary,
                tables_dir,
                "rl_reward_detailed_summary.csv",
            )

            print(reward_summary.to_string(index=False))

            report_lines.append("")
            report_lines.append("RL REWARD SUMMARY")
            report_lines.append(
                reward_summary.to_string(index=False)
            )

            # Reward distribution
            reward_distribution = (
                transitions["reward_numeric"]
                .value_counts(dropna=False)
                .sort_index()
                .rename_axis("reward")
                .reset_index(name="count")
            )

            reward_distribution["percentage"] = (
                reward_distribution["count"]
                / len(transitions)
                * 100
            )

            save_csv(
                reward_distribution,
                tables_dir,
                "rl_reward_distribution.csv",
            )

            # Reward by action
            if "action" in transitions.columns:
                reward_by_action = (
                    transitions.groupby("action")[
                        "reward_numeric"
                    ]
                    .agg([
                        "count",
                        "mean",
                        "std",
                        "min",
                        "median",
                        "max",
                    ])
                    .reset_index()
                )

                save_csv(
                    reward_by_action,
                    tables_dir,
                    "rl_reward_by_action.csv",
                )

                print("\nReward by action:")
                print(reward_by_action.to_string(index=False))

                report_lines.append("")
                report_lines.append("REWARD BY ACTION")
                report_lines.append(
                    reward_by_action.to_string(index=False)
                )

    # ---------------------------------------------------------
    # 5. RL action and termination audit
    # ---------------------------------------------------------

    print("\n[5] RL ACTION AND TERMINATION AUDIT")

    if not transitions.empty:
        for column in ["action", "terminated"]:
            if column not in transitions.columns:
                continue

            distribution = (
                transitions[column]
                .value_counts(dropna=False)
                .rename_axis(column)
                .reset_index(name="count")
            )

            distribution["percentage"] = (
                distribution["count"]
                / len(transitions)
                * 100
            )

            save_csv(
                distribution,
                tables_dir,
                f"rl_{column}_distribution.csv",
            )

            print(f"\n{column}:")
            print(distribution.to_string(index=False))

    # ---------------------------------------------------------
    # 6. Extreme tumor values: audit, do not delete
    # ---------------------------------------------------------

    print("\n[6] EXTREME TUMOR VALUES")

    if (
        not visits.empty
        and "tumor_size_mm" in visits.columns
    ):
        size = pd.to_numeric(
            visits["tumor_size_mm"],
            errors="coerce",
        )

        negative = visits[size < 0].copy()
        extreme = visits[size > 500].copy()

        save_csv(
            negative,
            tables_dir,
            "negative_tumor_sizes.csv",
        )

        save_csv(
            extreme,
            tables_dir,
            "visits_tumor_size_over_500.csv",
        )

        print("Negative tumor sizes:", len(negative))
        print("Tumor sizes > 500 mm:", len(extreme))

        report_lines.append("")
        report_lines.append("EXTREME TUMOR VALUES")
        report_lines.append(
            f"Negative visit tumor sizes: {len(negative)}"
        )
        report_lines.append(
            f"Visit tumor sizes > 500 mm: {len(extreme)}"
        )

    if (
        not outcomes.empty
        and "final_tumor_size_mm" in outcomes.columns
    ):
        size = pd.to_numeric(
            outcomes["final_tumor_size_mm"],
            errors="coerce",
        )

        negative = outcomes[size < 0].copy()
        extreme = outcomes[size > 500].copy()

        save_csv(
            negative,
            tables_dir,
            "negative_final_tumor_sizes.csv",
        )

        save_csv(
            extreme,
            tables_dir,
            "outcomes_final_tumor_size_over_500.csv",
        )

        print("Negative final tumor sizes:", len(negative))
        print("Final tumor sizes > 500 mm:", len(extreme))

        report_lines.append(
            f"Negative final tumor sizes: {len(negative)}"
        )
        report_lines.append(
            f"Final tumor sizes > 500 mm: {len(extreme)}"
        )

    # ---------------------------------------------------------
    # 7. Synthetic data safeguards
    # ---------------------------------------------------------

    print("\n[7] SYNTHETIC DATA CHECK")

    for name, df in [
        ("visits", visits),
        ("outcomes", outcomes),
        ("transitions", transitions),
    ]:
        if df.empty or "synthetic_flag" not in df.columns:
            continue

        distribution = (
            df["synthetic_flag"]
            .value_counts(dropna=False)
            .rename_axis("synthetic_flag")
            .reset_index(name="count")
        )

        save_csv(
            distribution,
            tables_dir,
            f"{name}_synthetic_flag_distribution.csv",
        )

        print(f"\n{name}:")
        print(distribution.to_string(index=False))

    # ---------------------------------------------------------
    # 8. Save final report
    # ---------------------------------------------------------

    report_lines.extend([
        "",
        "INTERPRETATION RULES",
        "-" * 40,
        "1. The dataset is synthetic, not real clinical evidence.",
        "2. A mismatch is an audit finding, not automatically a data error.",
        "3. Extreme values were exported for inspection, not deleted.",
        "4. Reward associations do not establish treatment efficacy.",
        "5. Review the reward-generation logic before policy training.",
    ])

    report_path = out_dir / "step_04_audit_summary.txt"

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("STEP 04 COMPLETED")
    print(f"Report: {report_path}")
    print(f"Tables: {tables_dir}")
    print("Original dataset files were not modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()