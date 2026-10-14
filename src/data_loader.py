"""
Data loading and preprocessing module.
Handles reading CSVs, computing derived fields, and week-bucketing.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def load_shipments(filepath: str) -> pd.DataFrame:
    """Load shipment records and add derived columns."""
    df = pd.read_csv(filepath, parse_dates=["shipment_date"])
    
    # Build a canonical route label: "Origin-Destination"
    df["route"] = df["origin"] + "-" + df["destination"]
    
    # Cost per tonne-km for each individual shipment
    df["cost_per_tonne_km"] = df["freight_cost_inr"] / (
        df["quantity_tonnes"] * df["distance_km"]
    )
    
    # Assign each shipment to its ISO week (Monday–Sunday).
    # week_of = the Monday of that week.
    df["week_of"] = df["shipment_date"].apply(
        lambda d: d - timedelta(days=d.weekday())
    )
    
    return df


def load_context_notes(filepath: str) -> pd.DataFrame:
    """Load the context notes that explain potential cost changes."""
    notes = pd.read_csv(filepath, parse_dates=["date"])
    return notes


def compute_weekly_route_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate shipments into (route, route_type, week_of) groups and
    compute the average cost_per_tonne_km for each group.

    Returns a DataFrame sorted by route then week_of with columns:
        route, route_type, week_of, cost_per_tonne_km, shipment_count
    """
    weekly = (
        df.groupby(["route", "route_type", "week_of"])
        .agg(
            cost_per_tonne_km=("cost_per_tonne_km", "mean"),
            shipment_count=("shipment_id", "count"),
        )
        .reset_index()
        .sort_values(["route", "week_of"])
        .reset_index(drop=True)
    )
    return weekly
