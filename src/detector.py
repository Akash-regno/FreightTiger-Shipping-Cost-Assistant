"""
Anomaly detection module.
Computes trailing averages, peer comparisons, and flags anomalous route-weeks.
"""

import pandas as pd
import numpy as np
from src.config import TRAILING_WINDOW_WEEKS, OWN_HISTORY_THRESHOLD_PCT, PEER_GROUP_THRESHOLD_PCT


def add_trailing_average(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    For each (route, week_of), compute the trailing 8-week rolling average of
    cost_per_tonne_km using only weeks strictly before the current week.
    
    Adds columns:
        - trailing_avg: the rolling average (NaN for the first week of a route)
        - trailing_weeks_used: how many prior weeks were available
    """
    weekly = weekly.sort_values(["route", "week_of"]).copy()
    
    trailing_avgs = []
    trailing_counts = []
    
    for _, group in weekly.groupby("route"):
        costs = group["cost_per_tonne_km"].values
        avgs = []
        counts = []
        for i in range(len(costs)):
            if i == 0:
                avgs.append(np.nan)
                counts.append(0)
            else:
                # Use up to TRAILING_WINDOW_WEEKS prior weeks (strictly before)
                start = max(0, i - TRAILING_WINDOW_WEEKS)
                window = costs[start:i]
                avgs.append(np.mean(window))
                counts.append(len(window))
        trailing_avgs.extend(avgs)
        trailing_counts.extend(counts)
    
    weekly["trailing_avg"] = trailing_avgs
    weekly["trailing_weeks_used"] = trailing_counts
    return weekly


def add_peer_average(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    For each (route, week_of), compute the average cost_per_tonne_km across
    all OTHER routes with the same route_type in the same week.
    
    Adds column:
        - peer_avg: average cost across peer routes (same route_type, same week,
                    excluding the route itself)
    """
    peer_avgs = []
    
    for idx, row in weekly.iterrows():
        mask = (
            (weekly["week_of"] == row["week_of"])
            & (weekly["route_type"] == row["route_type"])
            & (weekly["route"] != row["route"])
        )
        peers = weekly.loc[mask, "cost_per_tonne_km"]
        if len(peers) > 0:
            peer_avgs.append(peers.mean())
        else:
            peer_avgs.append(np.nan)
    
    weekly["peer_avg"] = peer_avgs
    return weekly


def compute_deviations(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage deviations from trailing average and peer average.
    
    Adds columns:
        - vs_own_history_pct: % deviation from trailing average
        - vs_similar_routes_pct: % deviation from peer average
    """
    weekly["vs_own_history_pct"] = (
        (weekly["cost_per_tonne_km"] - weekly["trailing_avg"])
        / weekly["trailing_avg"]
        * 100
    )
    weekly["vs_similar_routes_pct"] = (
        (weekly["cost_per_tonne_km"] - weekly["peer_avg"])
        / weekly["peer_avg"]
        * 100
    )
    return weekly


def flag_anomalies(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Flag route-weeks where cost is anomalously high compared to either
    its own history or peer routes.
    
    A route-week is flagged if:
        - vs_own_history_pct > OWN_HISTORY_THRESHOLD_PCT  OR
        - vs_similar_routes_pct > PEER_GROUP_THRESHOLD_PCT
    
    We only flag when the cost is HIGHER (positive deviation), not lower.
    Also, we skip weeks with no trailing history (first week of a route).
    
    Adds column:
        - is_flagged: boolean
    """
    weekly["is_flagged"] = (
        (weekly["vs_own_history_pct"] > OWN_HISTORY_THRESHOLD_PCT)
        | (weekly["vs_similar_routes_pct"] > PEER_GROUP_THRESHOLD_PCT)
    ) & (weekly["trailing_avg"].notna())  # Can't flag without history
    
    return weekly


def run_detection(weekly: pd.DataFrame) -> pd.DataFrame:
    """Full detection pipeline: trailing avg -> peer avg -> deviations -> flags."""
    weekly = add_trailing_average(weekly)
    weekly = add_peer_average(weekly)
    weekly = compute_deviations(weekly)
    weekly = flag_anomalies(weekly)
    return weekly
