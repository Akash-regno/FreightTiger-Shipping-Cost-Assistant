"""
main.py — Freight Cost Anomaly Detector
========================================
Entry point for the smart assistant that watches shipping costs.

Usage:
    python main.py                    # Run analysis and produce output.csv
    python main.py --reproduce        # Run 3 times and check reproducibility
    python main.py --query "Why did Delhi-Jaipur get pricier in November?"

Author: Akash
"""

import sys
import os
import time
import hashlib
import argparse
import pandas as pd

from src.config import (
    SHIPMENT_FILE,
    CONTEXT_NOTES_FILE,
    OUTPUT_FILE,
    REPRODUCIBILITY_RUNS,
)
from src.data_loader import load_shipments, load_context_notes, compute_weekly_route_costs
from src.detector import run_detection
from src.formatter import generate_output


def run_pipeline(shipment_file: str, notes_file: str, output_file: str) -> pd.DataFrame:
    """
    Execute the full analysis pipeline:
        1. Load data
        2. Compute weekly costs per route
        3. Detect anomalies (trailing avg + peer comparison + flagging)
        4. Match notes via RAG and generate output
        5. Write CSV

    Returns the output DataFrame.
    """
    print("=" * 60)
    print("  Freight Cost Anomaly Detector")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading shipment records...")
    shipments = load_shipments(shipment_file)
    print(f"      Loaded {len(shipments)} shipments")
    print(f"      Date range: {shipments['shipment_date'].min().date()} to "
          f"{shipments['shipment_date'].max().date()}")
    print(f"      Routes: {shipments['route'].nunique()}")

    print("\n[2/5] Loading context notes...")
    notes = load_context_notes(notes_file)
    print(f"      Loaded {len(notes)} context notes")

    # Step 2: Weekly aggregation
    print("\n[3/5] Computing weekly route costs...")
    weekly = compute_weekly_route_costs(shipments)
    print(f"      Generated {len(weekly)} route-week records")

    # Step 3: Anomaly detection
    print("\n[4/5] Running anomaly detection...")
    weekly = run_detection(weekly)
    n_flagged = weekly["is_flagged"].sum()
    print(f"      Flagged {n_flagged} anomalous route-weeks out of {len(weekly)}")

    # Step 4: Note matching & output generation
    print("\n[5/5] Matching notes via RAG and generating output...")
    output = generate_output(weekly, notes)
    justified = (output["flagged"] == "No (justified)").sum()
    unexplained = (output["flagged"] == "Yes").sum()
    print(f"      Output: {len(output)} rows "
          f"({justified} justified, {unexplained} unexplained)")

    # Step 5: Write CSV
    output.to_csv(output_file, index=False)
    print(f"\n[OK] Output written to {output_file}")

    return output


def run_reproducibility_check(shipment_file: str, notes_file: str):
    """
    Run the pipeline REPRODUCIBILITY_RUNS times and verify that
    flags, numbers, and note matches are identical across runs.
    """
    print("\n" + "=" * 60)
    print("  REPRODUCIBILITY CHECK")
    print("=" * 60)
    print(f"  Running {REPRODUCIBILITY_RUNS} identical runs...\n")

    hashes = []
    outputs = []

    for i in range(REPRODUCIBILITY_RUNS):
        out_file = f"output_run_{i + 1}.csv"
        print(f"\n--- Run {i + 1}/{REPRODUCIBILITY_RUNS} ---")
        output = run_pipeline(shipment_file, notes_file, out_file)
        outputs.append(output)

        # Compute hash of deterministic columns (excluding 'reason' wording)
        deterministic_cols = [
            "route", "week_of", "cost_per_tonne_km",
            "vs_own_history", "vs_similar_routes",
            "flagged", "matched_note_id",
        ]
        det_df = output[deterministic_cols]
        csv_str = det_df.to_csv(index=False)
        h = hashlib.sha256(csv_str.encode()).hexdigest()
        hashes.append(h)
        print(f"      Deterministic hash: {h[:16]}...")

    print("\n" + "-" * 60)
    if len(set(hashes)) == 1:
        print("[OK] ALL RUNS IDENTICAL -- flags, numbers, and note matches match.")
        print(f"  SHA-256 (deterministic columns): {hashes[0]}")
    else:
        print("[FAIL] MISMATCH DETECTED across runs!")
        for i, h in enumerate(hashes):
            print(f"  Run {i + 1}: {h}")

    # Clean up intermediate files, keep final output
    for i in range(REPRODUCIBILITY_RUNS):
        out_file = f"output_run_{i + 1}.csv"
        if i < REPRODUCIBILITY_RUNS - 1:
            os.remove(out_file)
        else:
            # Keep the last run as the final output (os.replace works across Windows when dest exists)
            os.replace(out_file, OUTPUT_FILE)
            print(f"\n  Final output saved as {OUTPUT_FILE}")

    return hashes


def handle_query(query: str, shipment_file: str, notes_file: str):
    """
    Answer a plain-English question about the findings.
    Stretch goal: interactive Q&A using the analysis results.
    """
    print("\n" + "=" * 60)
    print("  QUERY MODE")
    print("=" * 60)
    print(f"  Question: {query}\n")

    # Load and run the full pipeline to get results
    shipments = load_shipments(shipment_file)
    notes = load_context_notes(notes_file)
    weekly = compute_weekly_route_costs(shipments)
    weekly = run_detection(weekly)
    output = generate_output(weekly, notes)

    # Parse the query to find relevant route and time period
    query_lower = query.lower()

    # Extract route from query
    routes = weekly["route"].unique()
    matched_route = None
    for route in routes:
        if route.lower() in query_lower or route.lower().replace("-", " ") in query_lower:
            matched_route = route
            break

    # Extract month from query
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    matched_month = None
    for month_name, month_num in months.items():
        if month_name in query_lower:
            matched_month = month_num
            break

    # Filter results
    relevant = output.copy()
    if matched_route:
        relevant = relevant[relevant["route"] == matched_route]
    if matched_month:
        relevant = relevant[
            pd.to_datetime(relevant["week_of"]).dt.month == matched_month
        ]

    if len(relevant) == 0:
        print("  No flagged anomalies found matching your query.")
        if matched_route:
            route_data = weekly[weekly["route"] == matched_route]
            if matched_month:
                route_data = route_data[route_data["week_of"].dt.month == matched_month]
            if len(route_data) > 0:
                avg_cost = route_data["cost_per_tonne_km"].mean()
                print(f"\n  However, {matched_route} had an average cost of "
                      f"INR {avg_cost:.2f}/tonne-km during that period.")
                print("  No weeks exceeded the anomaly thresholds.")
    else:
        print(f"  Found {len(relevant)} flagged entries:\n")
        context_rows = []
        for _, row in relevant.iterrows():
            print(f"  > Route: {row['route']}")
            print(f"     Week of: {row['week_of']}")
            print(f"     Cost/tonne-km: INR {row['cost_per_tonne_km']}")
            print(f"     vs own history: {row['vs_own_history']}")
            print(f"     vs similar routes: {row['vs_similar_routes']}")
            print(f"     Status: {row['flagged']}")
            if row['matched_note_id']:
                print(f"     Matched note: {row['matched_note_id']}")
            print(f"     Reason: {row['reason']}")
            print()
            context_rows.append(
                f"Route {row['route']}, Week {row['week_of']}: "
                f"Cost {row['cost_per_tonne_km']} ({row['vs_own_history']}, {row['vs_similar_routes']}). "
                f"Flag: {row['flagged']}. Reason: {row['reason']}"
            )

        # Call Groq LLM Assistant if available
        try:
            from src.llm_assistant import ask_assistant, get_groq_client
            if get_groq_client():
                print("-" * 60)
                print("  AI ASSISTANT (Groq Qwen-27B):")
                summary_ctx = "\n".join(context_rows)
                llm_res = ask_assistant(query, summary_ctx)
                print(f"  {llm_res['answer']}\n")
                trk = llm_res['tracker']
                print(f"  [Token Log] Total tokens: {trk['total_tokens']} "
                      f"(Prompt: {trk['prompt_tokens']}, Completion: {trk['completion_tokens']}) | "
                      f"Cost: ${trk['cost_usd']:.6f} USD")
                print("-" * 60)
        except Exception as e:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Freight Cost Anomaly Detector - Smart Assistant for Shipping Cost Monitoring"
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help=f"Run {REPRODUCIBILITY_RUNS} times and verify reproducibility",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Ask a question about the findings (e.g., 'Why did Delhi-Jaipur get pricier?')",
    )
    parser.add_argument(
        "--shipments",
        type=str,
        default=SHIPMENT_FILE,
        help=f"Path to shipment records CSV (default: {SHIPMENT_FILE})",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=CONTEXT_NOTES_FILE,
        help=f"Path to context notes CSV (default: {CONTEXT_NOTES_FILE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILE,
        help=f"Path for output CSV (default: {OUTPUT_FILE})",
    )

    args = parser.parse_args()

    if args.query:
        handle_query(args.query, args.shipments, args.notes)
    elif args.reproduce:
        run_reproducibility_check(args.shipments, args.notes)
    else:
        start = time.time()
        run_pipeline(args.shipments, args.notes, args.output)
        elapsed = time.time() - start
        print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
