"""
streamlit_app.py
=================

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path so we can import retrieve.*
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from sentence_transformers import SentenceTransformer

from retrieve.orchestrator import answer_query
from dashboard import render_dashboard


# Page configuration
st.set_page_config(
    page_title="Quotes Manager",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Autumn pastel theme via CSS injection
st.markdown("""
<style>
    .stApp { background-color: #FBF8F2; }
    .confidence-high { color: #587858; font-weight: 600; }
    .confidence-medium { color: #C2942F; font-weight: 600; }
    .confidence-low { color: #B85A5A; font-weight: 600; }

    /* Sample-query chips */
    .chip-row .stButton > button {
        background: #FDFAF6;
        border: 1px solid #E5DDD2;
        color: #2D2520;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 6px 12px;
        border-radius: 18px;
        transition: all 0.15s;
    }
    .chip-row .stButton > button:hover {
        background: #F5EFE5;
        border-color: #B85A5A;
        color: #B85A5A;
    }
    .chip-label {
        color: #786558;
        font-size: 0.85rem;
        margin: 0.5rem 0 0.4rem 0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Bottom-of-page footer */
    .page-footer {
        margin-top: 3rem;
        padding-top: 0.6rem;
        border-top: 1px solid #E5DDD2;
        color: #A89484;
        font-size: 0.7rem;
        line-height: 1.4;
    }
    .footer-status {
        display: flex;
        flex-wrap: wrap;
        gap: 18px;
        align-items: center;
        justify-content: flex-start;
    }
    .footer-status .label {
        font-weight: 600;
        color: #786558;
    }
    .footer-clear .stButton > button {
        background: transparent;
        border: 1px solid #E5DDD2;
        color: #786558;
        font-size: 0.7rem;
        padding: 2px 10px;
        border-radius: 6px;
        height: auto;
        min-height: 0;
        line-height: 1.2;
    }
    .footer-clear .stButton > button:hover {
        border-color: #B85A5A;
        color: #B85A5A;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def preload_models():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


with st.spinner("Loading embedding model (one-time, ~10 sec)..."):
    _ = preload_models()


# Session state initialisation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ============ SIDEBAR ============
with st.sidebar:
    st.title("Trade Manager")
    st.caption("AI assistant for Irish trades")
    # Dashboard filters get appended here by render_dashboard()


# ============ MAIN HEADER ============
st.title("Quotes and Customer Manager")
st.caption("AI assistant for Irish trades")


# ============ TABS ============
tab_dashboard, tab_chat = st.tabs(["Dashboard", "Chatbot"])


# ---------- TAB 1: BI DASHBOARD ----------
with tab_dashboard:
    render_dashboard()


# ---------- TAB 2: CHATBOT ----------
with tab_chat:

    st.markdown('<p class="chip-label">Try one of these</p>', unsafe_allow_html=True)

    sample_queries = [
        "How much for a boiler installation?",
        "Show me PO-2026-P0042",
        "What's Gerard Walsh's phone number?",
        "I need a quote for floor sanding",
        "Has cust_0001 had work done before?",
        "Show me all POs",
    ]

    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    chip_cols = st.columns(3)
    for i, q in enumerate(sample_queries):
        with chip_cols[i % 3]:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    col_chat, col_quote = st.columns([2, 1])

    # ---- Right column: Predicted Quote ----
    with col_quote:
        st.subheader("Predicted Quote")
        pred = st.session_state.last_prediction

        if not pred:
            st.info(
                "Ask a pricing question to see predictions here.\n\n"
                "Example: *How much for a boiler installation?*"
            )
        else:
            st.markdown(f"**Job:** {pred.get('job_type', '?')}")
            confidence = pred.get("confidence", "unknown")
            st.markdown(
                f"**Confidence:** "
                f"<span class='confidence-{confidence}'>{confidence.upper()}</span>",
                unsafe_allow_html=True,
            )

            st.divider()

            mat = pred.get("materials", {})
            if mat.get("subtotal", 0) > 0:
                st.markdown(f"**Materials**  €{mat['subtotal']:,.2f}")
                st.caption(f"From {mat.get('n_recipe_items', 0)} graph recipe items")
                for item in mat.get("items", [])[:5]:
                    st.markdown(
                        f"<small>• {item['item_name']}  "
                        f"({item['quantity']} × €{item['unit_price']:.2f})</small>",
                        unsafe_allow_html=True,
                    )
                if len(mat.get("items", [])) > 5:
                    st.caption(f"...and {len(mat['items']) - 5} more")

            lab = pred.get("labour", {})
            if lab.get("median_eur", 0) > 0:
                st.divider()
                st.markdown(f"**Labour**  €{lab['median_eur']:,.2f}")
                st.caption(
                    f"Median of {lab.get('n_invoices', 0)} past invoices  "
                    f"(€{lab.get('min_eur', 0):.0f}-€{lab.get('max_eur', 0):.0f})"
                )

            totals = pred.get("totals", {})
            if totals.get("total_inc_vat", 0) > 0:
                st.divider()
                st.markdown(f"**Subtotal ex VAT**  €{totals.get('subtotal_ex_vat', 0):,.2f}")
                st.markdown(f"**VAT 23%**  €{totals.get('vat_23pct', 0):,.2f}")
                st.markdown(f"**Total inc VAT**  **€{totals.get('total_inc_vat', 0):,.2f}**")

            bench = pred.get("benchmark", {})
            if bench.get("n_pos", 0) > 0:
                st.divider()
                st.caption(
                    f"**Benchmark:** €{bench.get('avg_total', 0):,.2f} "
                    f"(avg of {bench['n_pos']} similar PO(s))"
                )

            ev = pred.get("evidence", {})
            if ev.get("stores_used"):
                st.divider()
                st.caption("**Stores used:** " + ", ".join(ev["stores_used"]))

    # ---- Left column: chat ----
    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("metadata"):
                    meta = msg["metadata"]
                    with st.expander(
                        f"Sources ({len(meta.get('sources', []))}) · "
                        f"{meta.get('latency_ms', 0)} ms · "
                        f"{meta.get('routing', {}).get('intent', '?')}"
                    ):
                        if meta.get("sources"):
                            st.markdown("**Sources:**")
                            for src in meta["sources"]:
                                st.markdown(f"- {src}")
                        if meta.get("routing"):
                            st.markdown("---")
                            st.markdown("**Routing:**")
                            st.json(meta["routing"])

        typed_input = st.chat_input("Ask about a customer, quote, or job...")

        user_input = None
        if typed_input:
            user_input = typed_input
        elif st.session_state.pending_query:
            user_input = st.session_state.pending_query
            st.session_state.pending_query = None

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            result = None
            latency = 0
            with st.chat_message("assistant"):
                with st.spinner("Searching MongoDB, Neo4j, and Atlas Vector Search..."):
                    try:
                        t_start = time.time()
                        result = answer_query(user_input, verbose=True)
                        latency = round((time.time() - t_start) * 1000)
                    except Exception as e:
                        err_str = str(e).lower()
                        if "defunct" in err_str or "service unavailable" in err_str:
                            st.warning(
                                "**Neo4j AuraDB is paused.** Wake it at "
                                "[console.neo4j.io](https://console.neo4j.io), then retry."
                            )
                        elif "auth" in err_str:
                            st.error("Authentication error. Check Neo4j credentials in .env.")
                        else:
                            st.error(f"Error: {e}")

                if result is not None:
                    st.markdown(result["answer"])
                    raw_results = result.get("raw_results", {})
                    if "predict" in raw_results:
                        st.session_state.last_prediction = raw_results["predict"]

                    with st.expander(
                        f"Sources ({len(result.get('sources', []))}) · "
                        f"{result.get('latency_ms', latency)} ms · "
                        f"{result.get('routing', {}).get('intent', '?')}"
                    ):
                        if result.get("sources"):
                            st.markdown("**Sources:**")
                            for src in result["sources"]:
                                st.markdown(f"- {src}")
                        st.markdown("---")
                        st.markdown("**Routing:**")
                        st.json(result.get("routing", {}))

            if result is not None:
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result["answer"],
                    "metadata": {
                        "sources":    result.get("sources", []),
                        "latency_ms": result.get("latency_ms", latency),
                        "routing":    result.get("routing", {}),
                    },
                })
                st.rerun()


# ============ PAGE-WIDE FOOTER ============

st.markdown('<div class="page-footer">', unsafe_allow_html=True)

footer_col_status, footer_col_clear = st.columns([5, 1])

with footer_col_status:
    st.markdown(
        """
<div class="footer-status">
  <span><span class="label">MongoDB</span> 1,468 records</span>
  <span><span class="label">Neo4j</span> 537 nodes</span>
  <span><span class="label">Vector index</span> 263 chunks</span>
  <span><span class="label">Pricing engine</span> live</span>
</div>
""",
        unsafe_allow_html=True,
    )

with footer_col_clear:
    st.markdown('<div class="footer-clear">', unsafe_allow_html=True)
    if st.button("Clear chat", key="footer_clear_btn", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_prediction = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
