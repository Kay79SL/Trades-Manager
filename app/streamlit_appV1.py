"""
streamlit_app.py
=================
Phase 6: the chatbot UI for DACARag.

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


# Page configuration
st.set_page_config(
    page_title="Quotes Manager",
    page_icon="🔧",
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
</style>
""", unsafe_allow_html=True)


# Model preloading - happens once per Streamlit process
@st.cache_resource
def preload_models():
    """Load embedding model once, cache for the lifetime of the process."""
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
    st.title("🔧 DACARag")
    st.caption("AI assistant for Irish trades")

    st.divider()

    st.markdown("### Try these queries")
    sample_queries = [
    "How much for a boiler installation?",
    "Show me PO-2026-P0042",
    "What's Gerard Walsh's phone number?",
    "I need a quote for floor sanding",
    "Has cust_0001 had work done before?",
    "show me all POs",
    ]
    for i, q in enumerate(sample_queries):
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

    st.divider()

    st.markdown("### System status")
    st.markdown("""
    -  MongoDB: 1,468 records
    -  Neo4j: 537 nodes
    -  Vector index: 263 chunks
    -  Pricing engine: live
    """)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_prediction = None
        st.rerun()


# ============ MAIN HEADER ============
st.title("Quotes Manager")
st.caption("AI assistant for Irish trades")


# ============ TWO-COLUMN LAYOUT ============
col_chat, col_quote = st.columns([2, 1])


# ============ RIGHT COLUMN: PREDICTED QUOTE PANEL ============
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


# ============ LEFT COLUMN: CHAT ============
with col_chat:
    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metadata"):
                meta = msg["metadata"]
                with st.expander(
                    f"📚 Sources ({len(meta.get('sources', []))}) · "
                    f"⚡ {meta.get('latency_ms', 0)} ms · "
                    f"🧠 {meta.get('routing', {}).get('intent', '?')}"
                ):
                    if meta.get("sources"):
                        st.markdown("**Sources:**")
                        for src in meta["sources"]:
                            st.markdown(f"- {src}")
                    if meta.get("routing"):
                        st.markdown("---")
                        st.markdown("**Routing:**")
                        st.json(meta["routing"])

    # Resolve user input - either typed or pending from sidebar
    typed_input = st.chat_input("Ask about a customer, quote, or job...")

    user_input = None
    if typed_input:
        user_input = typed_input
    elif st.session_state.pending_query:
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None

    if user_input:
        # Show the user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run the orchestrator
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
                            "🌙 **Neo4j AuraDB is paused.** Wake it at "
                            "[console.neo4j.io](https://console.neo4j.io), then retry."
                        )
                    elif "auth" in err_str:
                        st.error("🔐 Authentication error. Check Neo4j credentials in .env.")
                    else:
                        st.error(f"Error: {e}")

            if result is not None:
                st.markdown(result["answer"])

                raw_results = result.get("raw_results", {})
                if "predict" in raw_results:
                    st.session_state.last_prediction = raw_results["predict"]

                with st.expander(
                    f"📚 Sources ({len(result.get('sources', []))}) · "
                    f"⚡ {result.get('latency_ms', latency)} ms · "
                    f"🧠 {result.get('routing', {}).get('intent', '?')}"
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
