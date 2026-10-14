#!/usr/bin/env python3
"""
validate_dataset.py
===================

Focused data-quality validator for the OncoMARL synthetic dataset.

Usage
-----
    python3 validate_dataset.py --data data/oncomarl_breast_v4 --out reports/oncomarl_analysis
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from analysis import data_loading, data_quality

logger = logging.getLogger("oncomarl.validator")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the OncoMARL synthetic dataset.")
    p.add_argument("--data", required=True, help="Input dataset directory.")
    p.add_argument("--out", default="reports/oncomarl_analysis",
                   help="Output directory for the validation report.")
    p.add_argument("--patients", type=int, default=None)
    p.add_argument("--visits", type=int, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    os.makedirs(os.path.join(args.out, "summary"), exist_ok=True)

    try:
        tables, missing = data_loading.load_tables(args.data)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    meta = data_loading.load_metadata(args.data)
    if meta:
        if args.patients is None:
            args.patients = int(meta.get("patient_count")) if meta.get("patient_count") else None
        if args.visits is None:
            args.visits = int(meta.get("visits_per_patient")) if meta.get("visits_per_patient") else None

    dq = data_quality.run_all_checks(tables, args.patients, args.visits)

    text_lines = []
    for _, r in dq.iterrows():
        text_lines.append(f"[{r['status']}] {r['check']} — {r['explanation']}")
    text = "\n".join(text_lines)

    print(text)

    out_path = os.path.join(args.out, "summary", "validation_report.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    logger.info("Wrote %s", out_path)

    n_fail = int((dq["status"] == "FAIL").sum()) if not dq.empty else 0
    n_warn = int((dq["status"] == "WARNING").sum()) if not dq.empty else 0

    print(f"\nFAIL: {n_fail}   WARNING: {n_warn}   PASS: {int((dq['status']=='PASS').sum())}")
    if n_fail > 0:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())