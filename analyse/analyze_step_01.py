#!/usr/bin/env python3

"""
OncoMARL Dataset Analysis
Step 1: Dataset discovery and structural inspection.

Purpose:
    Inspect existing CSV and JSON files without modifying the original data.

Important:
    This project analyzes synthetic breast cancer data.
    It does not perform clinical analysis or recommend treatments.
"""

import argparse
import json
import logging
import sys

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

EXPECTED_FILES = [
    "patients.csv",
    "tumor_profiles.csv",
    "clinical_visits.csv",
    "clinical_events.csv",
    "treatment_history.csv",
    "treatment_decisions.csv",
    "outcomes.csv",
    "rl_transitions.csv",
    "dataset_metadata.json",
]

IDENTIFIER_COLUMNS = [
    "patient_id",
    "visit_id",
    "event_id",
    "transition_id",
    "decision_id",
    "outcome_id",
]

CSV_ENCODING_OPTIONS = [
    "utf-8",
    "utf-8-sig",
    "latin-1",
]


# ============================================================
# 2. LOGGING
# ============================================================

def configure_logging() -> None:
    """Configure terminal logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


# ============================================================
# 3. CSV LOADING
# ============================================================

def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file using several common encodings.

    Raises:
        RuntimeError if the file cannot be loaded.
    """

    errors = []

    for encoding in CSV_ENCODING_OPTIONS:
        try:
            return pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False,
            )

        except Exception as exc:
            errors.append(
                f"{encoding}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        f"Unable to load CSV {file_path.name}. "
        + " | ".join(errors)
    )


# ============================================================
# 4. JSON LOADING
# ============================================================

def load_json(file_path: Path) -> Any:
    """Load and validate a JSON file."""

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 5. FILE INVENTORY
# ============================================================

def inspect_files(data_dir: Path) -> tuple[
    list[dict],
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    """
    Discover and inspect CSV and JSON files.

    Returns:
        inventory: File-level information.
        csv_tables: Successfully loaded DataFrames.
        json_objects: Successfully loaded JSON objects.
    """

    inventory = []
    csv_tables = {}
    json_objects = {}

    files = sorted(
        file_path
        for file_path in data_dir.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in {".csv", ".json"}
    )

    logging.info("Discovered %d CSV/JSON files.", len(files))

    for file_path in files:

        relative_path = str(file_path.relative_to(data_dir))

        record = {
            "file_name": file_path.name,
            "relative_path": relative_path,
            "extension": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size,
            "status": "NOT_PROCESSED",
            "rows": None,
            "columns": None,
            "memory_bytes": None,
            "duplicate_rows": None,
            "missing_cells": None,
            "column_names": None,
            "error": None,
        }

        try:

            # ------------------------------------------------
            # CSV FILE
            # ------------------------------------------------

            if file_path.suffix.lower() == ".csv":

                df = load_csv(file_path)

                csv_tables[relative_path] = df

                record["status"] = "LOADED"
                record["rows"] = int(df.shape[0])
                record["columns"] = int(df.shape[1])

                record["memory_bytes"] = int(
                    df.memory_usage(
                        index=True,
                        deep=True,
                    ).sum()
                )

                record["duplicate_rows"] = int(
                    df.duplicated().sum()
                )

                record["missing_cells"] = int(
                    df.isna().sum().sum()
                )

                record["column_names"] = json.dumps(
                    list(df.columns),
                    ensure_ascii=False,
                )

            # ------------------------------------------------
            # JSON FILE
            # ------------------------------------------------

            elif file_path.suffix.lower() == ".json":

                obj = load_json(file_path)

                json_objects[relative_path] = obj

                record["status"] = "LOADED_JSON"

                if isinstance(obj, dict):
                    record["column_names"] = json.dumps(
                        list(obj.keys()),
                        ensure_ascii=False,
                    )

                elif isinstance(obj, list):
                    record["rows"] = len(obj)

        except Exception as exc:

            record["status"] = "FAILED"

            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            logging.warning(
                "Could not inspect %s: %s",
                relative_path,
                exc,
            )

        inventory.append(record)

    return inventory, csv_tables, json_objects


# ============================================================
# 6. SCHEMA REPORT
# ============================================================

def create_schema_report(
    csv_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Describe every column in every successfully loaded CSV."""

    records = []

    for file_name, df in csv_tables.items():

        for column in df.columns:

            series = df[column]

            records.append({
                "file_name": file_name,
                "column_name": str(column),
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_percentage": (
                    round(
                        float(series.isna().mean() * 100),
                        4,
                    )
                ),
                "unique_values": int(series.nunique(dropna=True)),
                "is_empty_column": bool(series.isna().all()),
            })

    return pd.DataFrame(records)


# ============================================================
# 7. MISSING-VALUE REPORT
# ============================================================

def create_missing_report(
    csv_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Report missing values in CSV tables."""

    records = []

    for file_name, df in csv_tables.items():

        for column in df.columns:

            missing_count = int(df[column].isna().sum())

            records.append({
                "file_name": file_name,
                "column_name": str(column),
                "total_rows": len(df),
                "missing_count": missing_count,
                "missing_percentage": (
                    round(
                        missing_count / len(df) * 100,
                        4,
                    )
                    if len(df) > 0
                    else 0.0
                ),
            })

    return pd.DataFrame(records)


# ============================================================
# 8. DUPLICATE REPORT
# ============================================================

def create_duplicate_report(
    csv_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Report duplicated rows and duplicated column names."""

    records = []

    for file_name, df in csv_tables.items():

        records.append({
            "file_name": file_name,
            "total_rows": len(df),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_column_names": int(
                pd.Index(df.columns).duplicated().sum()
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# 9. IDENTIFIER REPORT
# ============================================================

def create_identifier_report(
    csv_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Inspect known identifier columns when present."""

    records = []

    for file_name, df in csv_tables.items():

        for column in IDENTIFIER_COLUMNS:

            if column not in df.columns:
                continue

            series = df[column]

            records.append({
                "file_name": file_name,
                "identifier_column": column,
                "total_rows": len(df),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "duplicate_non_null_count": int(
                    series.dropna().duplicated().sum()
                ),
                "appears_unique": bool(
                    series.notna().all()
                    and series.is_unique
                ),
            })

    return pd.DataFrame(records)


# ============================================================
# 10. EXPECTED FILE CHECK
# ============================================================

def check_expected_files(
    data_dir: Path,
) -> list[dict]:
    """Check which expected files are present."""

    records = []

    for filename in EXPECTED_FILES:

        path = data_dir / filename

        if path.is_file():

            status = "PRESENT"

        else:

            status = "MISSING_OPTIONAL"

        records.append({
            "expected_file": filename,
            "status": status,
        })

    return records


# ============================================================
# 11. JSON METADATA SUMMARY
# ============================================================

def summarize_json(
    json_objects: dict[str, Any],
) -> list[dict]:
    """Summarize JSON root structures."""

    records = []

    for filename, obj in json_objects.items():

        if isinstance(obj, dict):

            root_type = "object"
            top_level_keys = list(obj.keys())

        elif isinstance(obj, list):

            root_type = "array"
            top_level_keys = []

        else:

            root_type = type(obj).__name__
            top_level_keys = []

        records.append({
            "file_name": filename,
            "valid_json": True,
            "root_type": root_type,
            "top_level_keys": json.dumps(
                top_level_keys,
                ensure_ascii=False,
            ),
            "looks_like_dataset_metadata": (
                "metadata" in filename.lower()
                or "dataset_version" in top_level_keys
                or "patient_count" in top_level_keys
            ),
        })

    return records


# ============================================================
# 12. SAVE REPORTS
# ============================================================

def save_reports(
    data_dir: Path,
    output_dir: Path,
    inventory: list[dict],
    csv_tables: dict[str, pd.DataFrame],
    json_objects: dict[str, Any],
) -> dict:
    """Generate and save Step 1 reports."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # File inventory
    inventory_df = pd.DataFrame(inventory)

    inventory_df.to_csv(
        output_dir / "file_inventory.csv",
        index=False,
    )

    # Schema
    schema_df = create_schema_report(csv_tables)

    schema_df.to_csv(
        output_dir / "schema_report.csv",
        index=False,
    )

    # Missing values
    missing_df = create_missing_report(csv_tables)

    missing_df.to_csv(
        output_dir / "missing_values.csv",
        index=False,
    )

    # Duplicates
    duplicates_df = create_duplicate_report(csv_tables)

    duplicates_df.to_csv(
        output_dir / "duplicate_report.csv",
        index=False,
    )

    # Identifiers
    identifiers_df = create_identifier_report(csv_tables)

    identifiers_df.to_csv(
        output_dir / "identifier_report.csv",
        index=False,
    )

    # Expected files
    expected_files = check_expected_files(data_dir)

    pd.DataFrame(expected_files).to_csv(
        output_dir / "expected_files_report.csv",
        index=False,
    )

    # JSON metadata
    json_summary = summarize_json(json_objects)

    pd.DataFrame(json_summary).to_csv(
        output_dir / "json_summary.csv",
        index=False,
    )

    # Global summary
    loaded_count = sum(
        1
        for record in inventory
        if record["status"] in {"LOADED", "LOADED_JSON"}
    )

    failed_count = sum(
        1
        for record in inventory
        if record["status"] == "FAILED"
    )

    total_rows = sum(
        len(df)
        for df in csv_tables.values()
    )

    total_columns = sum(
        len(df.columns)
        for df in csv_tables.values()
    )

    summary = {
        "analysis_step": 1,
        "purpose": "Dataset discovery and structural inspection",
        "synthetic_data": True,
        "input_directory": str(data_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "discovered_files": len(inventory),
        "csv_files_loaded": len(csv_tables),
        "json_files_loaded": len(json_objects),
        "files_loaded_successfully": loaded_count,
        "files_failed": failed_count,
        "total_csv_rows": int(total_rows),
        "total_csv_columns": int(total_columns),
        "expected_files": expected_files,
        "warnings": [
            record
            for record in inventory
            if record["status"] == "FAILED"
        ],
        "next_step": (
            "Review these reports before beginning Step 2: "
            "data-quality validation."
        ),
    }

    with (
        output_dir / "inspection_summary.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    # Text report
    lines = [
        "ONCOMARL DATASET INSPECTION",
        "=" * 60,
        "",
        "Step: 1 - Dataset discovery and structural inspection",
        "Dataset type: Synthetic breast cancer data",
        "",
        f"Input directory: {data_dir.resolve()}",
        f"Output directory: {output_dir.resolve()}",
        "",
        f"Discovered files: {len(inventory)}",
        f"Successfully loaded files: {loaded_count}",
        f"Failed files: {failed_count}",
        f"CSV files loaded: {len(csv_tables)}",
        f"JSON files loaded: {len(json_objects)}",
        f"Total CSV rows: {total_rows}",
        f"Total CSV columns: {total_columns}",
        "",
        "EXPECTED FILES",
        "-" * 60,
    ]

    for record in expected_files:

        lines.append(
            f"{record['expected_file']}: {record['status']}"
        )

    lines.extend([
        "",
        "FILES WITH ERRORS",
        "-" * 60,
    ])

    failed_records = [
        record
        for record in inventory
        if record["status"] == "FAILED"
    ]

    if failed_records:

        for record in failed_records:

            lines.append(
                f"{record['relative_path']}: "
                f"{record['error']}"
            )

    else:

        lines.append("No file-loading errors detected.")

    lines.extend([
        "",
        "IMPORTANT",
        "-" * 60,
        "This report describes file structure only.",
        "It does not establish clinical validity.",
        "Original dataset files were not modified.",
        "",
        "NEXT STEP",
        "-" * 60,
        "Review the reports before proceeding to Step 2.",
    ])

    (
        output_dir / "inspection_summary.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return summary


# ============================================================
# 13. MAIN
# ============================================================

def main() -> int:
    """Run Step 1."""

    parser = argparse.ArgumentParser(
        description=(
            "OncoMARL Step 1: inspect CSV and JSON dataset files."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the existing dataset directory.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/oncomarl_analysis/step_01"),
        help="Directory for generated inspection reports.",
    )

    args = parser.parse_args()

    configure_logging()

    data_dir = args.data.expanduser().resolve()
    output_dir = args.out.expanduser().resolve()

    if not data_dir.exists():

        logging.error(
            "Input directory does not exist: %s",
            data_dir,
        )

        return 2

    if not data_dir.is_dir():

        logging.error(
            "Input path is not a directory: %s",
            data_dir,
        )

        return 2

    try:

        logging.info("Starting Step 1.")
        logging.info("Input directory: %s", data_dir)

        inventory, csv_tables, json_objects = inspect_files(
            data_dir
        )

        summary = save_reports(
            data_dir=data_dir,
            output_dir=output_dir,
            inventory=inventory,
            csv_tables=csv_tables,
            json_objects=json_objects,
        )

        logging.info("Step 1 completed.")

        logging.info(
            "Loaded files: %d",
            summary["files_loaded_successfully"],
        )

        logging.info(
            "Failed files: %d",
            summary["files_failed"],
        )

        logging.info(
            "Reports saved to: %s",
            output_dir,
        )

        if summary["files_failed"] > 0:
            return 1

        return 0

    except Exception:

        logging.exception(
            "Step 1 failed unexpectedly."
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())