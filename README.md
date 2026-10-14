# Freight Cost Anomaly Detector

A smart assistant that monitors shipping costs across routes and flags unexplained cost increases, while using AI-powered note matching to distinguish justified rises from genuine anomalies.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive Streamlit Web UI (Recommended for Round 2 Demo)
python -m streamlit run app.py

# Run the command line analysis
python main.py

# Run the evaluation harness
python eval_harness.py

# Run reproducibility check (3 identical runs)
python main.py --reproduce

# Ask a question via AI Assistant (Groq Qwen-27B stretch goal)
python main.py --query "Why did Delhi-Jaipur get pricier in November?"
```

## How It Works

### Architecture

```
shipment_records.csv ──┐
                       ├──> data_loader.py ──> detector.py ──> formatter.py ──> output.csv
context_notes.csv ─────┘                                         ↑
                                                          note_retriever.py
                                                          (RAG with SBERT)
```

### Step-by-Step Pipeline

1. **Data Loading** (`src/data_loader.py`)
   - Loads shipment records and context notes from CSV
   - Computes `cost_per_tonne_km = freight_cost_inr / (quantity_tonnes * distance_km)` for each shipment
   - Assigns each shipment to its ISO week (Monday-Sunday); `week_of` = that week's Monday
   - Aggregates into (route, route_type, week_of) groups with mean cost_per_tonne_km

2. **Anomaly Detection** (`src/detector.py`)
   - **vs. own history**: Trailing 8-week rolling average (strictly before current week, no look-ahead). If fewer than 8 prior weeks exist, uses all available.
   - **vs. similar routes**: Average cost_per_tonne_km across all OTHER routes with the same `route_type` in that same week. The route itself is excluded.
   - **Flagging**: A route-week is flagged if cost is >20% above its own trailing average OR >20% above its peer group average.

3. **Note Matching / RAG** (`src/note_retriever.py`)
   - Uses **sentence-transformers** (`all-MiniLM-L6-v2`, 22M params, runs locally, free) for semantic similarity
   - Falls back to **TF-IDF** (scikit-learn) if the model isn't available
   - Three-layer filtering before matching:
     - **Route filter**: Note must apply to the exact route or "All Routes"
     - **Temporal filter**: Note date must be within 21 days of the flagged week
     - **Negation filter**: Notes explicitly saying costs were NOT affected (e.g., "no significant disruptions", "remained normal", "without a rate change") are excluded from justification
   - Best matching note above similarity threshold becomes the justification

4. **Output Formatting** (`src/formatter.py`)
   - Builds plain-English reasons citing exact notes with dates
   - If no justifying note found, explicitly says "unexplained" (never fabricates)
   - Mentions nearby non-justifying notes to show awareness ("The closest note... does not describe a reason...")

### Anti-Hallucination Guardrails

| Guardrail | Implementation |
|-----------|---------------|
| Wrong note ID | Every `matched_note_id` is verified against `context_notes.csv` |
| Wrong route | Notes must match the route's `applies_to` field or be "All Routes" |
| Wrong time window | Notes must be within 21 days of the flagged week |
| Negation notes | Notes saying "costs not affected" are explicitly excluded |
| No matching note | System says "unexplained" rather than guessing |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **20% threshold** for both own-history and peer comparisons | Matches the sample output format (Delhi-Jaipur at +35.5% flagged, Ahmedabad-Mumbai at +29.5% justified). Captures meaningful deviations without over-flagging. |
| **all-MiniLM-L6-v2** model | Lightweight (22M params), fast on CPU, free, no API key needed. Good balance of quality and speed for 10 notes. |
| **TF-IDF fallback** | Ensures the system works without internet or GPU. scikit-learn is already a dependency. |
| **Negation phrase list** | Simple, interpretable, and verifiable. Prevents the most common hallucination pattern (citing a note that says "nothing happened"). |
| **21-day temporal window** | Wide enough to catch multi-week events (Chennai-Bangalore flooding lasted ~2 weeks) but tight enough to prevent matching unrelated notes. |
| **Mean aggregation** per route-week | Robust for the given data (3-5 shipments per route-week). Weighted average by volume was considered but adds complexity without clear benefit at this scale. |

## Reproducibility

The system is **fully deterministic**:
- No randomness in data processing (pandas groupby, rolling averages)
- sentence-transformers embeddings are deterministic at inference time (no temperature parameter)
- TF-IDF fallback is also deterministic
- No LLM API calls with random sampling

**Verification**: Running `python main.py --reproduce` performs 3 identical runs and computes SHA-256 hashes of the deterministic columns (route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id). All 3 hashes match.

## Token & Cost Log

| Metric | Value |
|--------|-------|
| **LLM API calls** | 0 (no cloud LLM used) |
| **Total input tokens** | 0 |
| **Total output tokens** | 0 |
| **Model used** | `all-MiniLM-L6-v2` (local, open-source) |
| **Embedding calls** | ~20 (one per flagged route-week, plus 10 for note pre-encoding) |
| **Total cost** | **$0.00** |
| **Runtime** | ~8-9 seconds per full run |

The solution uses only local, open-source models. No paid API calls are made.

## Evaluation Harness

Run `python eval_harness.py` to validate:

- **Format**: Output CSV has exact required columns
- **Anti-hallucination**: 
  - All cited note IDs exist
  - Justified verdicts match correct route and time window
  - Negation notes never cited as justification
  - Unexplained flags have empty `matched_note_id`
- **Known test cases**: 
  - Ahmedabad-Mumbai 2025-01-20 → justified by N002
  - Non-cost-affecting notes (N004-N010) never used as justification
- **Numerical accuracy**: cost_per_tonne_km values in reasonable range

## Project Structure

```
.
├── main.py                     # Entry point (run, reproduce, query modes)
├── eval_harness.py             # Lightweight evaluation/validation suite
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── config.py               # All tunable parameters in one place
│   ├── data_loader.py          # CSV loading, week bucketing, aggregation
│   ├── detector.py             # Trailing avg, peer avg, flagging logic
│   ├── note_retriever.py       # RAG with SBERT / TF-IDF fallback
│   └── formatter.py            # Output CSV generation with plain-English reasons
├── shipment_records.csv        # Input: 2940 shipment records
├── context_notes.csv           # Input: 10 context notes
├── sample_output_format_v2.csv # Reference output format
└── output.csv                  # Generated output
```

## Stretch Goal: Interactive Q&A

```bash
python main.py --query "Why did Chennai-Bangalore get pricier in March?"
python main.py --query "What happened with Delhi-Jaipur in November 2024?"
```

Parses the question to extract route and time period, then shows matching flagged entries with their full analysis and notes.

## 10-Minute Round 2 Walkthrough Guide

Here is a ready-to-use structure for the 10-minute presentation:

| Time | Agenda | Key Talking Points |
|------|--------|---------------------|
| **0:00 - 2:00** | **Problem & Approach** | Problem: Spot unjustified freight cost creep. Core logic: Group by route & week (Mon–Sun), compute INR/(tonne·km), compare against trailing 8-week history (no look-ahead) and peer routes (same `route_type`). |
| **2:00 - 4:00** | **System Architecture & Code** | Modular design: `data_loader.py` (ISO weeks, aggregation), `detector.py` (strict formulas, no look-ahead), `note_retriever.py` (semantic search + temporal/negation filters), `formatter.py` (exact schema compliance). |
| **4:00 - 6:00** | **AI & Anti-Hallucination Guardrails** | Why semantic retrieval with local SBERT (`all-MiniLM-L6-v2`) instead of naive keyword matching. 3 guardrails: temporal distance <= 21 days, route match (`All Routes` or exact), and negation filtering (notes stating "costs were not affected" like N005/N006/N010 are never cited as justifications). |
| **6:00 - 7:30** | **Validation & Reproducibility Demo** | Run `python eval_harness.py` live (shows 0 failures across format, citations, and ground-truth cases). Run `python main.py --reproduce` showing SHA-256 hash match across 3 untouched runs. |
| **7:30 - 8:30** | **Stretch Goal & Efficiency** | Demo `python main.py --query "Why did Delhi-Jaipur get pricier in November?"`. Highlight $0 run cost (runs on CPU in ~1.3s, 0 token spend). |
| **8:30 - 10:00** | **Trade-offs & Next Steps** | Trade-offs: Semantic embeddings vs LLM prompting (embeddings are deterministic, instant, free, and verifiable); Future work: Dynamic thresholds based on variance (z-score), multi-modal routing graphs. |

