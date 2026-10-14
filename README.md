# FreightTiger-Shipping-Cost-Assistant

A smart assistant that monitors shipping costs across freight routes, flags unexplained cost increases, and uses AI-powered note matching (RAG) to distinguish justified cost rises from genuine anomalies.

Built for the **Freight Tiger Software Engineering Intern (AI)** evaluation.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the interactive Streamlit Web UI (Recommended for Demo)
python -m streamlit run app.py

# 3. Run the core CLI pipeline (Generates output.csv)
python main.py

# 4. Run the automated evaluation suite
python eval_harness.py

# 5. Run the 3-run reproducibility check
python main.py --reproduce

# 6. Ask questions in plain English via the AI Assistant (Stretch Goal)
python main.py --query "Why did Chennai-Bangalore get pricier in March?"
```

---

## How It Works

### System Architecture

```
shipment_records.csv ──┐
                       ├──> data_loader.py ──> detector.py ──> formatter.py ──> output.csv
context_notes.csv ─────┘                                         ↑
                                                          note_retriever.py (SBERT RAG)
                                                          llm_assistant.py (Groq Qwen-27B)
```

### Step-by-Step Pipeline

1. **Data Loading & Group Aggregation** (`src/data_loader.py`)
   - Reads shipment records and context notes from CSV.
   - Assigns each shipment to its ISO week (Monday–Sunday); `week_of` represents Monday.
   - Computes group cost per tonne-km using the exact formula from the brief:
     $$\text{Cost per tonne-km} = \frac{\sum \text{freight\_cost\_inr}}{\sum (\text{quantity\_tonnes} \times \text{distance\_km})}$$
     This weighted aggregation ensures 100% numerical alignment with `sample_output_format_v2.csv`.

2. **Anomaly Detection & Comparison Baselines** (`src/detector.py`)
   - **vs. own history**: Trailing 8-week rolling average cost per tonne-km, strictly using weeks prior to the current week (no look-ahead). If fewer than 8 prior weeks exist, all available prior weeks are used without padding or extrapolation.
   - **vs. similar routes**: Average cost per tonne-km across all *other* routes sharing the same `route_type` (Short / Medium / Long) in that same week. The route itself is strictly excluded from its peer group.
   - **Flagging**: A route-week is flagged when cost is >20% above its own trailing average or >20% above its peer group average.

3. **Grounded Note Matching / RAG** (`src/note_retriever.py`)
   - Uses local **Sentence-Transformers** (`all-MiniLM-L6-v2`, 22M parameters, running locally with zero token cost) with cosine similarity.
   - Falls back automatically to TF-IDF (scikit-learn) if needed.
   - Three-layer strict filtering before matching:
     - **Route Filter**: Note must match the route or be marked `"All Routes"`.
     - **Temporal Window**: Note date must be within 21 days of the flagged week.
     - **Negation Filter**: Notes stating costs were *not* affected (e.g. `N005`, `N006`, `N010`) are strictly excluded from justification.
   - Best matching note above similarity threshold becomes the justification.

4. **Output Formatting** (`src/formatter.py`)
   - Formats columns to match `sample_output_format.csv` exactly.
   - Justified anomalies cite the exact note ID, date, and description.
   - Unexplained anomalies explicitly state no valid reason was found and suggest human review (never hallucinates).

---

## Anti-Hallucination Guardrails

| Guardrail | Implementation & Validation |
| :--- | :--- |
| **No Invented Notes** | Every `matched_note_id` is strictly validated against `context_notes.csv`. |
| **Route Applicability** | Notes must apply to the specific corridor or `"All Routes"`. |
| **Temporal Proximity** | Events must fall within 21 days of the anomalous week. |
| **Negation Filtering** | Notes with phrases like *"costs were not significantly affected"* or *"remained normal"* are never cited as justifications. |
| **Zero Guessing** | If no note genuinely justifies the rise, the verdict remains `"Yes"` (unexplained). |

---

## Reproducibility Across 3 Untouched Runs

The core pipeline is **100% deterministic**:
- Trailing averages and peer groups use stable sorting and exact arithmetic.
- Sentence embeddings are deterministic at inference.
- Verified by running `python main.py --reproduce`, which performs 3 separate untouched runs and computes the SHA-256 hash of all deterministic columns:
  ```
  [OK] ALL RUNS IDENTICAL -- flags, numbers, and note matches match.
  SHA-256: 0a31c3356fac0711e149a3acb1c4411b314c19f68d9e3856a33900464b9f9d3d
  ```

---

## Token & Cost Log

| Metric | Local Pipeline (`main.py`) | Interactive AI Q&A (`Groq`) |
| :--- | :---: | :---: |
| **LLM Provider** | Local Open-Source | Groq Cloud API |
| **Model** | `all-MiniLM-L6-v2` | `qwen/qwen3.8-27b` |
| **Total Cloud Tokens** | **0 tokens** | ~700 tokens per query |
| **API Calls** | **0** | 1 per user question |
| **Total Pipeline Cost** | **$0.00** | **~$0.0001 / query** |
| **Pipeline Latency** | **~2.0s** on CPU | **~0.8s** response |

The entire core pipeline runs with **$0.00 cost** and zero external API dependencies.

---

## Evaluation Test Suite

Run `python eval_harness.py` to execute the automated test harness:
```
============================================================
  EVALUATION HARNESS
============================================================
  Output file: output.csv
  Rows: 20 | Justified: 5 | Unexplained: 15

--- Format Checks ---
  [PASS] Output format matches required columns
--- Anti-Hallucination Checks ---
  [PASS] All cited note IDs exist in context_notes.csv
  [PASS] All justified verdicts cite notes that apply to the correct route
  [PASS] All justified verdicts cite temporally relevant notes
  [PASS] No negation notes cited as justification
  [PASS] All unexplained flags have empty matched_note_id
--- Known Test Cases ---
  [PASS] Ahmedabad-Mumbai 2025-01-20 correctly justified by N002
  [PASS] No non-cost-affecting notes used as justification
--- Numerical Accuracy ---
  [PASS] cost_per_tonne_km values in reasonable range (3.29 to 4.49)
------------------------------------------------------------
  ALL CHECKS PASSED!
============================================================
```

---

## Project Structure

```
FreightTiger-Shipping-Cost-Assistant/
├── main.py                     # Primary CLI engine (run, reproduce, query modes)
├── eval_harness.py             # Evaluation & verification test suite
├── app.py                      # Interactive Streamlit Web Dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & benchmarks
├── .env.example                # Safe environment template
├── .gitignore                  # Git exclusions for secrets and cache
├── output.csv                  # Audited output CSV matching the contract
├── shipment_records.csv        # 2,940 shipment records (input)
├── context_notes.csv           # 10 context notes (input)
├── sample_output_format_v2.csv # Freight Tiger benchmark format
├── FreightTiger_Intern_CaseStudy.pdf
└── src/
    ├── __init__.py
    ├── config.py               # Tunable constants & thresholds
    ├── data_loader.py          # Week bucketing & weighted cost aggregation
    ├── detector.py             # Trailing 8-wk baseline & peer group calculations
    ├── formatter.py            # Output formatter with plain-English reasons
    ├── note_retriever.py       # RAG module with semantic search & guardrails
    └── llm_assistant.py        # Groq-powered natural language assistant
```

---

## Stretch Goal: Natural Language Q&A

You can ask natural-language questions in two ways:

1. **Via Streamlit Web UI**: Launch `python -m streamlit run app.py` and navigate to the **"🤖 Ask AI Assistant"** tab.
2. **Via Command Line**:
   ```bash
   python main.py --query "Why did Chennai-Bangalore get pricier in March?"
   python main.py --query "Why did Delhi-Jaipur get pricier in November 2024?"
   ```

*Example Output:*
> **Question**: Why did Chennai-Bangalore get pricier in March?  
> **Answer**: According to note **N001** (dated 2025-02-24), heavy flooding on the Chennai-Bangalore highway disrupted normal truck movement from February 24 to March 8, forcing longer detours and higher trip costs. Roads became passable again by March 9, but the lingering impact elevated trip costs through March.  
> *[Token Log] Total tokens: 707 | Cost: $0.000141 USD*
