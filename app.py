"""
app.py — Streamlit Interactive Web Application
Freight Tiger AI Intern Case Study: Smart Assistant That Watches Shipping Costs

Features:
- Real-time KPI summary (Total Shipments, Anomalies, Justified vs Unexplained)
- Interactive Plotly route cost trends (Trailing 8-week baseline vs peer baseline)
- Filterable Flagged Anomalies table matching sample output format
- Natural Language AI Assistant powered by Groq (Strictly grounded RAG)
- Token and API cost tracker
- Live Evaluation Harness runner
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.config import SHIPMENT_FILE, CONTEXT_NOTES_FILE, OUTPUT_FILE
from src.data_loader import load_shipments, load_context_notes, compute_weekly_route_costs
from src.detector import run_detection
from src.formatter import generate_output
from src.llm_assistant import ask_assistant, tracker, get_groq_client

# ─── Streamlit Config & Page Header ──────────────────────────────────────────
st.set_page_config(
    page_title="Freight Tiger | Smart Cost Assistant",
    page_icon="🐅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF5A1F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4A5568;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F7FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
    }
    .badge-justified {
        background-color: #C6F6D5;
        color: #22543D;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-unexplained {
        background-color: #FED7D7;
        color: #742A2A;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🐅 Freight Tiger — Smart Shipping Cost Watchdog</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Anomaly Detection & Grounded Note Retrieval for Freight Costs</div>', unsafe_allow_html=True)


# ─── Cached Data Pipeline ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading shipments & computing baselines...")
def get_data():
    shipments = load_shipments(SHIPMENT_FILE)
    notes = load_context_notes(CONTEXT_NOTES_FILE)
    weekly = compute_weekly_route_costs(shipments)
    weekly = run_detection(weekly)
    output = generate_output(weekly, notes)
    return shipments, notes, weekly, output


shipments, notes, weekly, output = get_data()

# ─── Top Level KPI Metrics ──────────────────────────────────────────────────
total_shipments = len(shipments)
total_routes = shipments["route"].nunique()
total_weeks = weekly["week_of"].nunique()
flagged_total = len(output)
justified_total = (output["flagged"] == "No (justified)").sum()
unexplained_total = (output["flagged"] == "Yes").sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📦 Total Shipments", f"{total_shipments:,}")
col2.metric("🛣️ Active Routes", total_routes)
col3.metric("⚠️ Anomalies Flagged", flagged_total)
col4.metric("✅ Justified by Notes", justified_total, help="Real-world cause verified in context notes")
col5.metric("🚨 Unexplained (Review)", unexplained_total, help="No valid justification; action required")

st.divider()

# ─── Main Tabs ───────────────────────────────────────────────────────────────
tab_trends, tab_flags, tab_ai, tab_notes, tab_eval = st.tabs([
    "📈 Route Cost Trends",
    "📋 Flagged Anomalies Table",
    "🤖 Ask AI Assistant (Groq)",
    "📝 Context Notes",
    "🧪 Evaluation & Audits"
])

# ─── TAB 1: Route Cost Trends ────────────────────────────────────────────────
with tab_trends:
    st.subheader("Interactive Route Cost vs. Trailing 8-Week Baseline")
    
    selected_route = st.selectbox(
        "Select a Route to Inspect:",
        options=sorted(weekly["route"].unique()),
        index=0
    )
    
    route_df = weekly[weekly["route"] == selected_route].copy().sort_values("week_of")
    
    fig = go.Figure()
    
    # Actual cost line
    fig.add_trace(go.Scatter(
        x=route_df["week_of"],
        y=route_df["cost_per_tonne_km"],
        mode="lines+markers",
        name="Actual Cost / t-km",
        line=dict(color="#3182CE", width=2.5),
        marker=dict(size=5)
    ))
    
    # Trailing 8-week average
    fig.add_trace(go.Scatter(
        x=route_df["week_of"],
        y=route_df["trailing_avg"],
        mode="lines",
        name="Trailing 8-Wk History Avg",
        line=dict(color="#718096", width=2, dash="dash")
    ))
    
    # Peer group average
    fig.add_trace(go.Scatter(
        x=route_df["week_of"],
        y=route_df["peer_avg"],
        mode="lines",
        name="Peer Group Avg (Same Length)",
        line=dict(color="#CBD5E0", width=1.5, dash="dot")
    ))
    
    # Overlay Flagged Anomalies
    flagged_route = route_df[route_df["is_flagged"]]
    if not flagged_route.empty:
        fig.add_trace(go.Scatter(
            x=flagged_route["week_of"],
            y=flagged_route["cost_per_tonne_km"],
            mode="markers",
            name="Flagged Anomaly",
            marker=dict(color="#E53E3E", size=10, symbol="x", line=dict(width=2, color="#9B2C2C"))
        ))
    
    fig.update_layout(
        title=f"Cost per Tonne-Km Evolution: {selected_route}",
        xaxis_title="Week Commencing (week_of)",
        yaxis_title="Cost per Tonne-Km (INR)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── TAB 2: Flagged Anomalies Table ──────────────────────────────────────────
with tab_flags:
    st.subheader("Audited Output Records (sample_output_format_v2.csv specification)")
    
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        status_filter = st.radio(
            "Filter by verdict:",
            ["All", "Yes (Unexplained)", "No (justified)"]
        )
    
    filtered_output = output.copy()
    if status_filter == "Yes (Unexplained)":
        filtered_output = filtered_output[filtered_output["flagged"] == "Yes"]
    elif status_filter == "No (justified)":
        filtered_output = filtered_output[filtered_output["flagged"] == "No (justified)"]
        
    st.dataframe(
        filtered_output,
        use_container_width=True,
        column_config={
            "route": st.column_config.TextColumn("Route"),
            "week_of": st.column_config.TextColumn("Week Of"),
            "cost_per_tonne_km": st.column_config.NumberColumn("Cost / t-km (₹)", format="₹%.2f"),
            "vs_own_history": st.column_config.TextColumn("vs History"),
            "vs_similar_routes": st.column_config.TextColumn("vs Peers"),
            "flagged": st.column_config.TextColumn("Flagged"),
            "matched_note_id": st.column_config.TextColumn("Matched Note"),
            "reason": st.column_config.TextColumn("Reason", width="large"),
        },
        hide_index=True
    )
    
    # Download Button
    csv_bytes = output.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download output.csv",
        data=csv_bytes,
        file_name="output.csv",
        mime="text/csv",
    )

# ─── TAB 3: Natural Language AI Assistant (Groq Qwen-27B) ────────────────────
with tab_ai:
    st.subheader("🤖 Natural Language Assistant (Stretch Goal)")
    st.caption("Ask questions about why routes got pricier, verify ground truth notes, or query anomaly explanations. Powered by Groq's low-latency Qwen-27B with strict anti-hallucination grounding.")
    
    sample_queries = [
        "Why did Chennai-Bangalore get pricier in March?",
        "Why did Delhi-Jaipur get pricier in November 2024?",
        "Why was Ahmedabad-Mumbai justified in January 2025?",
        "Did diesel price hikes explain Mumbai-Pune increases?"
    ]
    
    selected_sample = st.selectbox("💡 Or choose a sample question:", ["(Select a question...)"] + sample_queries)
    
    user_query = st.text_input(
        "Enter your question:",
        value=selected_sample if selected_sample != "(Select a question...)" else "",
        placeholder="e.g. Why did Chennai-Bangalore have higher costs in late February 2025?"
    )
    
    if st.button("Ask Assistant", type="primary"):
        if not user_query.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Analyzing shipping data and searching context notes via Groq..."):
                # Prepare summary context of flagged anomalies & relevant notes
                context_lines = []
                for _, r in output.iterrows():
                    context_lines.append(
                        f"Route: {r['route']} | Week: {r['week_of']} | Cost: INR {r['cost_per_tonne_km']} "
                        f"({r['vs_own_history']}, {r['vs_similar_routes']}) | Flag: {r['flagged']} | "
                        f"Note: {r['matched_note_id']} | Reason: {r['reason']}"
                    )
                context_str = "\n".join(context_lines)
                
                res = ask_assistant(user_query, context_str)
                
                st.markdown("### Answer")
                st.info(res["answer"])
                
                # Token and Cost metrics
                trk = res["tracker"]
                st.markdown("##### 📊 Token & Cost Transparency")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("API Calls", trk["calls"])
                c2.metric("Prompt Tokens", trk["prompt_tokens"])
                c3.metric("Completion Tokens", trk["completion_tokens"])
                c4.metric("Estimated Cost", f"${trk['cost_usd']:.6f} USD")

# ─── TAB 4: Context Notes Explorer ───────────────────────────────────────────
with tab_notes:
    st.subheader("Context Notes Knowledge Base (context_notes.csv)")
    st.caption("Real-world signals (fuel, weather, toll, festivals). Notice that notes with negations (e.g. N005, N006, N010) are filtered to prevent false justifications.")
    st.dataframe(notes, use_container_width=True, hide_index=True)

# ─── TAB 5: Evaluation & Audits ──────────────────────────────────────────────
with tab_eval:
    st.subheader("🧪 Evaluation Harness & Anti-Hallucination Checks")
    st.caption("Validates output schema, verifies note citations, checks temporal & route consistency, and tests ground-truth test cases.")
    
    if st.button("Run Evaluation Suite Live", type="secondary"):
        from eval_harness import (
            check_format, check_no_hallucinated_notes, check_note_route_match,
            check_note_temporal_match, check_negation_notes_not_cited,
            check_unflagged_have_no_note, check_known_cases, check_numerical_accuracy
        )
        
        errors = []
        c1 = check_format(output)
        c2 = check_no_hallucinated_notes(output, notes)
        c3 = check_note_route_match(output, notes)
        c4 = check_note_temporal_match(output, notes)
        c5 = check_negation_notes_not_cited(output, notes)
        c6 = check_unflagged_have_no_note(output)
        c7 = check_known_cases(output)
        c8 = check_numerical_accuracy(output)
        
        total_errors = len(c1) + len(c2) + len(c3) + len(c4) + len(c5) + len(c6) + len(c7) + len(c8)
        
        if total_errors == 0:
            st.success("✅ ALL EVALUATION CHECKS PASSED (8/8 Checks Passed)")
            st.markdown("""
            - **Format Test**: Exact column names, order, and data types verified.
            - **Anti-Hallucination Check**: 100% of cited notes exist in `context_notes.csv`.
            - **Route Matching**: Justified notes match route or apply to 'All Routes'.
            - **Temporal Proximity**: Events match within strict 21-day window.
            - **Negation Guardrail**: Notes stating 'costs not affected' (e.g. N005, N006, N010) are never used as justifications.
            - **Known Test Cases**: Ahmedabad-Mumbai correctly justified by N002 (festival surcharge).
            - **Numerical Sanity**: Cost/tonne-km verified within expected bounds.
            """)
        else:
            st.error(f"❌ {total_errors} issues detected.")

st.sidebar.markdown("### 🐅 Freight Tiger Intern")
st.sidebar.markdown("**Role**: Software Engineer Intern (AI)")
st.sidebar.markdown("**Focus**: Shipping Cost Watchdog")
st.sidebar.divider()
st.sidebar.markdown("**System Architecture**")
st.sidebar.code("""
Shipments CSV 
  └─> Week Bucketing (Mon-Sun)
  └─> Trailing 8-Wk History
  └─> Peer Group (Excl. Self)
  └─> Anomaly Flags (>20%)
  └─> SBERT / Groq RAG
  └─> output.csv
""")
st.sidebar.markdown("Groq LLM Engine: `qwen/qwen3.8-27b`")
st.sidebar.markdown("Local Embeddings: `all-MiniLM-L6-v2`")
