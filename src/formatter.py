"""
Output formatting module.
Builds the final CSV matching sample_output_format_v2.csv exactly.
"""

import pandas as pd
import numpy as np
from src.note_retriever import NoteRetriever


def build_reason_text(
    row: dict,
    note_id: str | None,
    note_text: str | None,
    nearby_notes: list,
) -> str:
    """
    Build a plain-English reason for the flagged (or justified) route-week.

    Rules (from the case study guardrails):
    - If a note matches: cite the exact note, explain how it applies.
    - If no note matches: say "unexplained" and flag for review.
    - NEVER fabricate a reason.
    """
    route = row["route"]
    week_of = row["week_of"].strftime("%Y-%m-%d")
    vs_own = row["vs_own_history_pct"]
    vs_peer = row["vs_similar_routes_pct"]

    if note_id is not None and note_text is not None:
        # Justified — cite the note
        reason = (
            f"Matches note {note_id} dated "
            f"{_get_note_date(note_id, nearby_notes)}: "
            f"{_summarize_note(note_text)} "
            f"The cost rise has a clear explanation."
        )
    else:
        # Unexplained — check if there's a nearby note that does NOT justify
        if nearby_notes:
            closest = min(nearby_notes, key=lambda n: n["date_diff"])
            reason = (
                f"The closest note ({closest['note_id']}, "
                f"{closest['date'].strftime('%Y-%m-%d')}) "
                f"{_summarize_non_justifying_note(closest['note'])} "
                f"No genuine justification found; flagged for review."
            )
        else:
            reason = (
                f"No matching note found for this route or date range. "
                f"Cost rise looks unexplained and worth a human review."
            )

    return reason


def _get_note_date(note_id: str, nearby_notes: list) -> str:
    """Get the date string for a note from nearby notes list."""
    for n in nearby_notes:
        if n["note_id"] == note_id:
            return n["date"].strftime("%Y-%m-%d")
    return "unknown date"


def _summarize_note(note_text: str) -> str:
    """Create a concise summary of the note for the reason field."""
    # Keep it brief but faithful to the original note
    if len(note_text) > 200:
        return note_text[:197] + "..."
    return note_text


def _summarize_non_justifying_note(note_text: str) -> str:
    """Summarize a note that doesn't justify the cost rise."""
    text_lower = note_text.lower()
    if "no significant disruptions" in text_lower or "remained normal" in text_lower:
        return "mentions stable conditions with no major disruptions -- it does not describe a reason for a cost rise on this route."
    if "not significantly affected" in text_lower:
        return "mentions minor issues but costs were not significantly affected -- it does not justify a cost rise."
    if "remained stable" in text_lower or "no major disruptions" in text_lower:
        return "mentions stable demand with no major disruptions -- it does not describe a reason for a cost rise on this route."
    if "without a rate change" in text_lower:
        return "mentions a new mandate but costs were absorbed without a rate change -- it does not justify a cost rise."
    return f"does not describe a reason that would justify a cost rise on this route."


def format_vs_own_history(pct: float, trailing_weeks: int) -> str:
    """Format the vs_own_history column value."""
    if pd.isna(pct):
        return ""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}% vs this route's past average"


def format_vs_similar_routes(pct: float) -> str:
    """Format the vs_similar_routes column value."""
    if pd.isna(pct):
        return ""
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}% vs similar-length routes this week"


def generate_output(
    weekly: pd.DataFrame,
    notes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate the final output DataFrame matching the required format.
    Only includes flagged route-weeks (anomalous rows).

    Output columns:
        route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes,
        flagged, matched_note_id, reason
    """
    retriever = NoteRetriever(notes_df)
    flagged = weekly[weekly["is_flagged"]].copy()
    
    rows = []
    for _, row in flagged.iterrows():
        # Determine the best matching note
        cost_change = row["vs_own_history_pct"] if not pd.isna(row["vs_own_history_pct"]) else 0
        note_id, note_text, score = retriever.find_matching_note(
            route=row["route"],
            week_of=row["week_of"],
            cost_pct_change=cost_change,
        )
        
        # Get nearby notes for context in the reason
        nearby = retriever.get_nearby_notes_for_reason(
            route=row["route"],
            week_of=row["week_of"],
        )
        
        # Determine flag status
        if note_id is not None:
            flagged_str = "No (justified)"
        else:
            flagged_str = "Yes"
        
        # Build reason text
        reason = build_reason_text(
            row=row,
            note_id=note_id,
            note_text=note_text,
            nearby_notes=nearby,
        )
        
        rows.append({
            "route": row["route"],
            "week_of": row["week_of"].strftime("%Y-%m-%d"),
            "cost_per_tonne_km": round(row["cost_per_tonne_km"], 2),
            "vs_own_history": format_vs_own_history(
                row["vs_own_history_pct"], row["trailing_weeks_used"]
            ),
            "vs_similar_routes": format_vs_similar_routes(
                row["vs_similar_routes_pct"]
            ),
            "flagged": flagged_str,
            "matched_note_id": note_id if note_id else "",
            "reason": reason,
        })
    
    output_df = pd.DataFrame(rows)
    return output_df
