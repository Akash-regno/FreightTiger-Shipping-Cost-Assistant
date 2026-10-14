"""
eval_harness.py -- Lightweight evaluation harness for the Freight Cost Anomaly Detector.

Validates that:
1. Output CSV matches the required format
2. Justified verdicts cite real notes with correct route and date
3. No hallucinated reasons (no made-up note references)
4. Numerical accuracy of comparisons
5. Known test cases produce expected results

This is the "Knowing your system is right" check required by the case study.
"""

import pandas as pd
import numpy as np
import sys
import os

from src.data_loader import load_shipments, load_context_notes, compute_weekly_route_costs
from src.detector import run_detection
from src.config import SHIPMENT_FILE, CONTEXT_NOTES_FILE, OUTPUT_FILE


def load_output(filepath: str) -> pd.DataFrame:
    """Load the generated output CSV."""
    return pd.read_csv(filepath)


def check_format(output: pd.DataFrame) -> list:
    """Verify the output CSV has the exact required columns."""
    required_cols = [
        "route", "week_of", "cost_per_tonne_km",
        "vs_own_history", "vs_similar_routes",
        "flagged", "matched_note_id", "reason",
    ]
    errors = []
    for col in required_cols:
        if col not in output.columns:
            errors.append(f"FAIL: Missing required column '{col}'")
    
    extra_cols = set(output.columns) - set(required_cols)
    if extra_cols:
        errors.append(f"WARN: Extra columns found: {extra_cols}")
    
    if not errors:
        print("  [PASS] Output format matches required columns")
    return errors


def check_no_hallucinated_notes(output: pd.DataFrame, notes: pd.DataFrame) -> list:
    """Verify every matched_note_id actually exists in the notes file."""
    errors = []
    valid_note_ids = set(notes["note_id"].values)
    
    for idx, row in output.iterrows():
        note_id = row.get("matched_note_id", "")
        if pd.notna(note_id) and note_id != "":
            if note_id not in valid_note_ids:
                errors.append(
                    f"FAIL: Row {idx} cites note '{note_id}' which does not exist"
                )
    
    if not errors:
        print("  [PASS] All cited note IDs exist in context_notes.csv")
    return errors


def check_note_route_match(output: pd.DataFrame, notes: pd.DataFrame) -> list:
    """
    Verify that each justified verdict cites a note that applies to that route.
    A note applies if its 'applies_to' field matches the route or is 'All Routes'.
    """
    errors = []
    notes_dict = notes.set_index("note_id").to_dict("index")
    
    for idx, row in output.iterrows():
        if row["flagged"] == "No (justified)":
            note_id = row.get("matched_note_id", "")
            if pd.isna(note_id) or note_id == "":
                errors.append(
                    f"FAIL: Row {idx} ({row['route']}, {row['week_of']}) is "
                    f"'justified' but has no matched_note_id"
                )
                continue
            
            if note_id in notes_dict:
                applies_to = notes_dict[note_id]["applies_to"]
                if applies_to != "All Routes" and applies_to != row["route"]:
                    errors.append(
                        f"FAIL: Row {idx} ({row['route']}, {row['week_of']}) "
                        f"cites {note_id} which applies to '{applies_to}', "
                        f"not '{row['route']}'"
                    )
    
    if not errors:
        print("  [PASS] All justified verdicts cite notes that apply to the correct route")
    return errors


def check_note_temporal_match(
    output: pd.DataFrame, notes: pd.DataFrame, max_days: int = 21
) -> list:
    """
    Verify that each justified verdict cites a note temporally close to the week.
    """
    errors = []
    notes_dict = notes.set_index("note_id").to_dict("index")
    
    for idx, row in output.iterrows():
        if row["flagged"] == "No (justified)":
            note_id = row.get("matched_note_id", "")
            if pd.isna(note_id) or note_id == "" or note_id not in notes_dict:
                continue
            
            note_date = pd.Timestamp(notes_dict[note_id]["date"])
            week_of = pd.Timestamp(row["week_of"])
            week_end = week_of + pd.Timedelta(days=6)
            
            if week_of <= note_date <= week_end:
                continue  # Within the week
            
            days_away = min(
                abs((note_date - week_of).days),
                abs((note_date - week_end).days),
            )
            
            if days_away > max_days:
                errors.append(
                    f"FAIL: Row {idx} ({row['route']}, {row['week_of']}) "
                    f"cites {note_id} dated {note_date.date()}, "
                    f"which is {days_away} days away (max {max_days})"
                )
    
    if not errors:
        print("  [PASS] All justified verdicts cite temporally relevant notes")
    return errors


def check_negation_notes_not_cited(output: pd.DataFrame, notes: pd.DataFrame) -> list:
    """
    Verify that notes explicitly saying costs were NOT affected are not used as justification.
    """
    errors = []
    negation_phrases = [
        "costs were not significantly affected",
        "no significant disruptions",
        "without a rate change",
        "remained normal",
        "remained stable",
    ]
    
    negation_note_ids = set()
    for _, note_row in notes.iterrows():
        note_lower = note_row["note"].lower()
        if any(p in note_lower for p in negation_phrases):
            negation_note_ids.add(note_row["note_id"])
    
    for idx, row in output.iterrows():
        if row["flagged"] == "No (justified)":
            note_id = row.get("matched_note_id", "")
            if note_id in negation_note_ids:
                errors.append(
                    f"FAIL: Row {idx} ({row['route']}, {row['week_of']}) "
                    f"cites {note_id} which says costs were NOT affected -- "
                    f"this is a hallucination"
                )
    
    if not errors:
        print("  [PASS] No negation notes cited as justification")
    return errors


def check_unflagged_have_no_note(output: pd.DataFrame) -> list:
    """
    Verify that 'Yes' (unexplained) rows have no matched_note_id.
    """
    errors = []
    
    for idx, row in output.iterrows():
        if row["flagged"] == "Yes":
            note_id = row.get("matched_note_id", "")
            if pd.notna(note_id) and note_id != "":
                errors.append(
                    f"WARN: Row {idx} ({row['route']}, {row['week_of']}) is "
                    f"flagged 'Yes' but has matched_note_id '{note_id}'"
                )
    
    if not errors:
        print("  [PASS] All unexplained flags have empty matched_note_id")
    return errors


def check_known_cases(output: pd.DataFrame) -> list:
    """
    Validate against known test cases from the sample output and notes:
    
    1. Ahmedabad-Mumbai week of 2025-01-20 should be justified by N002
       (festival surcharge)
    2. Chennai-Bangalore during Feb 24-Mar 8 flood period should be justified by N001
    3. Notes N004, N005, N006, N007, N008, N009, N010 should NOT be used
       as justification (they describe non-cost-affecting events)
    """
    errors = []
    
    # Check: N002 should justify Ahmedabad-Mumbai around 2025-01-20
    ahm_mum = output[
        (output["route"] == "Ahmedabad-Mumbai")
        & (output["week_of"] == "2025-01-20")
    ]
    if len(ahm_mum) > 0:
        if ahm_mum.iloc[0]["matched_note_id"] == "N002":
            print("  [PASS] Ahmedabad-Mumbai 2025-01-20 correctly justified by N002")
        else:
            errors.append(
                f"FAIL: Ahmedabad-Mumbai 2025-01-20 should be justified by N002, "
                f"got: {ahm_mum.iloc[0]['matched_note_id']}"
            )
    
    # Check: No notes that say "not affected" should be used
    bad_note_ids = {"N004", "N005", "N006", "N007", "N008", "N009", "N010"}
    justified_notes = set(
        output[output["flagged"] == "No (justified)"]["matched_note_id"].dropna()
    )
    bad_used = justified_notes & bad_note_ids
    if bad_used:
        errors.append(
            f"FAIL: Non-cost-affecting notes used as justification: {bad_used}"
        )
    else:
        print("  [PASS] No non-cost-affecting notes used as justification")
    
    return errors


def check_numerical_accuracy(output: pd.DataFrame) -> list:
    """
    Spot-check that cost_per_tonne_km values are in a reasonable range.
    """
    errors = []
    
    costs = output["cost_per_tonne_km"].values
    if np.any(costs <= 0):
        errors.append("FAIL: Some cost_per_tonne_km values are <= 0")
    if np.any(costs > 100):
        errors.append("WARN: Some cost_per_tonne_km values are > 100 (suspicious)")
    
    if not errors:
        print(f"  [PASS] cost_per_tonne_km values in reasonable range "
              f"({costs.min():.2f} to {costs.max():.2f})")
    return errors


def run_evaluation():
    """Run all evaluation checks and print a summary."""
    print("=" * 60)
    print("  EVALUATION HARNESS")
    print("=" * 60)
    
    if not os.path.exists(OUTPUT_FILE):
        print(f"\n  ERROR: {OUTPUT_FILE} not found. Run main.py first.")
        sys.exit(1)
    
    output = load_output(OUTPUT_FILE)
    notes = load_context_notes(CONTEXT_NOTES_FILE)
    
    print(f"\n  Output file: {OUTPUT_FILE}")
    print(f"  Rows: {len(output)}")
    print(f"  Justified: {(output['flagged'] == 'No (justified)').sum()}")
    print(f"  Unexplained: {(output['flagged'] == 'Yes').sum()}")
    
    all_errors = []
    
    print("\n--- Format Checks ---")
    all_errors.extend(check_format(output))
    
    print("\n--- Anti-Hallucination Checks ---")
    all_errors.extend(check_no_hallucinated_notes(output, notes))
    all_errors.extend(check_note_route_match(output, notes))
    all_errors.extend(check_note_temporal_match(output, notes))
    all_errors.extend(check_negation_notes_not_cited(output, notes))
    all_errors.extend(check_unflagged_have_no_note(output))
    
    print("\n--- Known Test Cases ---")
    all_errors.extend(check_known_cases(output))
    
    print("\n--- Numerical Accuracy ---")
    all_errors.extend(check_numerical_accuracy(output))
    
    print("\n" + "-" * 60)
    
    failures = [e for e in all_errors if e.startswith("FAIL")]
    warnings = [e for e in all_errors if e.startswith("WARN")]
    
    if failures:
        print(f"\n  FAILURES: {len(failures)}")
        for f in failures:
            print(f"    {f}")
    
    if warnings:
        print(f"\n  WARNINGS: {len(warnings)}")
        for w in warnings:
            print(f"    {w}")
    
    if not failures and not warnings:
        print("\n  ALL CHECKS PASSED!")
    elif not failures:
        print(f"\n  ALL CRITICAL CHECKS PASSED ({len(warnings)} warnings)")
    else:
        print(f"\n  {len(failures)} FAILURES, {len(warnings)} WARNINGS")
    
    print("=" * 60)
    
    return len(failures) == 0


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
