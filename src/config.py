"""
Configuration constants for the Freight Cost Anomaly Detector.
All tunable parameters live here for easy adjustment and transparency.
"""

# ─── File Paths ───────────────────────────────────────────────────────────────
SHIPMENT_FILE = "shipment_records.csv"
CONTEXT_NOTES_FILE = "context_notes.csv"
OUTPUT_FILE = "output.csv"

# ─── Core Analysis Parameters ────────────────────────────────────────────────
# Trailing window size for "vs own history" comparison (weeks)
TRAILING_WINDOW_WEEKS = 8

# ─── Flagging Thresholds ─────────────────────────────────────────────────────
# A route-week is considered anomalous if EITHER of these is breached:
#   - cost is this % above its own trailing average
OWN_HISTORY_THRESHOLD_PCT = 20.0
#   - cost is this % above the peer-group (same route_type) average that week
PEER_GROUP_THRESHOLD_PCT = 20.0

# ─── RAG / Note Matching Parameters ──────────────────────────────────────────
# Embedding model used for semantic similarity (runs locally, free)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Similarity threshold for considering a note relevant
SIMILARITY_THRESHOLD = 0.25

# How many days before/after a note's date the note is still considered
# applicable to a flagged week (temporal window)
NOTE_TEMPORAL_WINDOW_DAYS = 21

# ─── Reproducibility ─────────────────────────────────────────────────────────
# Number of reproducibility runs
REPRODUCIBILITY_RUNS = 3
