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

# Preload embedding model at startup to avoid latency on first query
@st.cache_resource
def preload_models():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Note: the model is also cached inside retrieve.vector_search, so this is just to warm it up at app startup
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
tab_dashboard, tab_chat = st.tabs(["Dashboard", "Chatbot"]) # tabs for PO management, customer management, etc


#  TAB 1: BI DASHBOARD 
with tab_dashboard:
    render_dashboard()


#  TAB 2: CHATBOT 
with tab_chat:

    st.markdown('<p class="chip-label">Try one of these</p>', unsafe_allow_html=True) # predefined queries as "chips" for easy testing - these will populate the input and trigger the same logic as a typed query, 
    # but are more user-friendly than having to copy-paste from a doc or type out manually. The chips are implemented as buttons with custom CSS styling.

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

    # Right column: Predicted Quote
    # This section displays the predicted quote details after a query is answered. It shows the job type, confidence level, 
    # breakdown of materials and labour costs, totals, benchmark comparison, and sources used. 
    # The information is pulled from st.session_state.last_prediction which is updated after each query that returns a prediction.
    with col_quote:
        st.subheader("Predicted Quote")
        pred = st.session_state.last_prediction # this gets set in the raw_results of the query answer, and is expected to have a structure like:
       
        if not pred: # if there's no prediction data, show an info message prompting the user to ask a pricing question to see predictions here. 
            # This encourages interaction and lets the user know what kind of queries will populate this
            st.info(
                "Ask a pricing question to see predictions here.\n\n"
                "Example: *How much for a boiler installation?*"
            )
        else: # if there is prediction data, display the details in a structured format. The confidence level is styled with CSS classes for visual emphasis.
            st.markdown(f"**Job:** {pred.get('job_type', '?')}")
            confidence = pred.get("confidence", "unknown")
            st.markdown( # the confidence level is displayed with a colored label using CSS classes defined earlier. The confidence value is converted to uppercase for emphasis.
                f"**Confidence:** "
                f"<span class='confidence-{confidence}'>{confidence.upper()}</span>",
                unsafe_allow_html=True,
            )

            st.divider()
            # The materials section shows a breakdown of the predicted materials cost, including a subtotal, number of recipe items, 
            # and a list of the top items contributing to the cost. This gives the user insight into how the materials cost was calculated.
            
            mat = pred.get("materials", {})
            if mat.get("subtotal", 0) > 0:
                st.markdown(f"**Materials**  €{mat['subtotal']:,.2f}") # the subtotal for materials is displayed prominently, and then a breakdown of the items is shown below in smaller text. 
                                                                        # The number of recipe items is also noted to give context to the breakdown.
                st.caption(f"From {mat.get('n_recipe_items', 0)} graph recipe items")
                for item in mat.get("items", [])[:5]: # show up to 5 items from the materials breakdown, with their name, quantity, and unit price. This gives the user a sense of what materials are contributing to the cost.
                    st.markdown(
                        f"<small>• {item['item_name']}  "
                        f"({item['quantity']} × €{item['unit_price']:.2f})</small>",
                        unsafe_allow_html=True,
                    ) # if there are more than 5 items, show a caption indicating how many more items there are that are not displayed, 
                    # to give the user a sense of the full breakdown without overwhelming them with too much detail in the main view.
                if len(mat.get("items", [])) > 5:
                    st.caption(f"...and {len(mat['items']) - 5} more")

            # The labour section shows the predicted labour cost based on past invoices for similar jobs. 
            # It includes the median cost, number of invoices considered, and the range of costs from those invoices. 
            # This helps the user understand how the labour cost was derived and its variability.
            lab = pred.get("labour", {})
            if lab.get("median_eur", 0) > 0:
                st.divider()
                st.markdown(f"**Labour**  €{lab['median_eur']:,.2f}")
                st.caption(
                    f"Median of {lab.get('n_invoices', 0)} past invoices  "
                    f"(€{lab.get('min_eur', 0):.0f}-€{lab.get('max_eur', 0):.0f})"
                )

            # The totals section shows the overall predicted cost for the job, including a breakdown of the subtotal excluding VAT, 
            # the VAT amount, and the total including VAT. This gives the user a clear summary of the predicted quote for the job.
            totals = pred.get("totals", {})
            if totals.get("total_inc_vat", 0) > 0:
                st.divider()
                st.markdown(f"**Subtotal ex VAT**  €{totals.get('subtotal_ex_vat', 0):,.2f}")
                st.markdown(f"**VAT 23%**  €{totals.get('vat_23pct', 0):,.2f}")
                st.markdown(f"**Total inc VAT**  **€{totals.get('total_inc_vat', 0):,.2f}**")

            # The benchmark section compares the predicted total cost to an average total from similar purchase orders (POs) in the past. 
            # It shows the average total and the number of similar POs considered in the benchmark. 
            # This allows the user to see how the predicted quote stacks up against historical data for similar jobs, providing
            bench = pred.get("benchmark", {})
            if bench.get("n_pos", 0) > 0:
                st.divider()
                st.caption(
                    f"**Benchmark:** €{bench.get('avg_total', 0):,.2f} "
                    f"(avg of {bench['n_pos']} similar PO(s))"
                )
            # Finally, the sources section lists the data sources that were used to generate the prediction, such as specific records from MongoDB or Neo4j.
            # This provides transparency to the user about where the information is coming from and can help build trust in the prediction by showing the underlying data that informed it.
            ev = pred.get("evidence", {})
            if ev.get("stores_used"):
                st.divider()
                st.caption("**Stores used:** " + ", ".join(ev["stores_used"]))

    #  Left column:
    # This section implements the chat interface where the user can ask questions about customers, quotes, jobs, etc. 
    # The conversation history is displayed here, with user messages and assistant responses.
    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("metadata"):
                    meta = msg["metadata"]
                    with st.expander(
                        f"Sources ({len(meta.get('sources', []))}) · " # the metadata for each message includes sources, latency, and routing information. 
                                                                        # This is displayed in an expander to keep the main chat view clean, but allows the user to see the details if they want.
                        f"{meta.get('latency_ms', 0)} ms · "
                        f"{meta.get('routing', {}).get('intent', '?')}"
                    ):
                        if meta.get("sources"): # if there are sources listed in the metadata, display them in a bulleted list to show the user where the information in the assistant's response came from.
                            st.markdown("**Sources:**")
                            for src in meta["sources"]:
                                st.markdown(f"- {src}")
                        if meta.get("routing"): # if there is routing information in the metadata, display it as JSON to show the user how the query was processed 
                                                # and which components of the system were involved in generating the response. This can help users understand the inner workings of the assistant and build trust in its responses.
                            st.markdown("---")
                            st.markdown("**Routing:**")
                            st.json(meta["routing"])

        typed_input = st.chat_input("Ask about a customer, quote, or job...")

        #  The logic for handling user input checks if there is a new typed input from the user. 
        # If not, it checks if there is a pending query set by one of the sample query buttons.
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
            with st.chat_message("assistant"): #  when the user submits a query, a new chat message is created for the assistant's response. 
                # A spinner is shown while the system processes the query and searches MongoDB, Neo4j, 
                # and Atlas Vector Search for relevant information to generate an answer. The time taken to get the answer is measured to provide latency information in the response metadata.
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

                # The result from the query is expected to have an "answer" field which contains the assistant's response to the user's query. This answer is displayed in the chat interface.
                if result is not None:
                    st.markdown(result["answer"])
                    raw_results = result.get("raw_results", {})
                    if "predict" in raw_results:
                        st.session_state.last_prediction = raw_results["predict"]

                    with st.expander( # the expander for sources, latency, and routing information is also shown for the assistant's response, similar to how it's shown for past messages. 
                                     # This allows the user to see the details of how the answer was generated for each response.
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

            if result is not None: # after displaying the assistant's response and the metadata, the message is appended to the session state messages with the role of "assistant", 
                                    # the content of the answer, and the metadata including sources, latency, and routing information. This ensures that the conversation history is maintained in the session state and can be displayed in future interactions.
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

# The footer displays the current status of the system, including the number of records in MongoDB, nodes in Neo4j, chunks in the vector index, and the status of the pricing engine. 
# This information is useful for debugging and gives the user insight into the underlying data that the assistant is working
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
# The footer also includes a "Clear chat" button that allows the user to reset the conversation history and clear any stored predictions. 
# This is useful for starting a new conversation or if the user wants to clear the context after a series of interactions. 
# When the button is clicked, the session state for messages and last_prediction is reset, and the app reruns to reflect the cleared state.
with footer_col_clear:
    st.markdown('<div class="footer-clear">', unsafe_allow_html=True)
    if st.button("Clear chat", key="footer_clear_btn", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_prediction = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
