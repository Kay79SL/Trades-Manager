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

sys.path.insert(0, str(Path(__file__).parent.parent))

import gridfs
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from retrieve.orchestrator import answer_query
from dashboard import render_dashboard
from dashboard import get_mongo_db


# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quotes Manager",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #FBF8F2; }
    .confidence-high   { color: #587858; font-weight: 600; }
    .confidence-medium { color: #C2942F; font-weight: 600; }
    .confidence-low    { color: #B85A5A; font-weight: 600; }
    .chip-row .stButton > button {
        background: #FDFAF6; border: 1px solid #E5DDD2; color: #2D2520;
        font-size: 0.85rem; font-weight: 500; padding: 6px 12px;
        border-radius: 18px; transition: all 0.15s;
    }
    .chip-row .stButton > button:hover {
        background: #F5EFE5; border-color: #B85A5A; color: #B85A5A;
    }
    .chip-label {
        color: #786558; font-size: 0.85rem; margin: 0.5rem 0 0.4rem 0;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .page-footer {
        margin-top: 3rem; padding-top: 0.6rem;
        border-top: 1px solid #E5DDD2; color: #A89484;
        font-size: 0.7rem; line-height: 1.4;
    }
    .footer-status { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; }
    .footer-status .label { font-weight: 600; color: #786558; }
    .footer-clear .stButton > button {
        background: transparent; border: 1px solid #E5DDD2; color: #786558;
        font-size: 0.7rem; padding: 2px 10px; border-radius: 6px;
        height: auto; min-height: 0; line-height: 1.2;
    }
    .footer-clear .stButton > button:hover { border-color: #B85A5A; color: #B85A5A; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def preload_models():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with st.spinner("Loading embedding model (one-time, ~10 sec)..."):
    _ = preload_models()


@st.cache_resource
def get_gridfs_buckets():
    """
    Returns all three named GridFS buckets:
      csv_files   — seed CSVs
      po_files    — supplier PO PDFs
      email_files — raw .eml files
    """
    db = get_mongo_db()
    return {
        "csv":   gridfs.GridFS(db, collection="csv_files"),
        "pdf":   gridfs.GridFS(db, collection="po_files"),
        "email": gridfs.GridFS(db, collection="email_files"),
    }


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages        = []
if "last_prediction" not in st.session_state: st.session_state.last_prediction = None
if "pending_query"   not in st.session_state: st.session_state.pending_query   = None
if "ingest_log"      not in st.session_state: st.session_state.ingest_log      = []


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Trade Manager")
    st.caption("AI assistant for Irish trades")


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.title("Quotes and Customer Manager")
st.caption("AI assistant for Irish trades")


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_dashboard, tab_chat, tab_upload = st.tabs(["Dashboard", "Chatbot", "Data Upload"])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — BI DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab_dashboard:
    render_dashboard()


# ══════════════════════════════════════════════════════════════
#  TAB 2 — CHATBOT
# ══════════════════════════════════════════════════════════════
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
                    f"(€{lab.get('min_eur', 0):.0f}–€{lab.get('max_eur', 0):.0f})"
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

            result  = None
            latency = 0
            with st.chat_message("assistant"):
                with st.spinner("Searching MongoDB, Neo4j, and Atlas Vector Search..."):
                    try:
                        t_start = time.time()
                        result  = answer_query(user_input, verbose=True)
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
                    "role":     "assistant",
                    "content":  result["answer"],
                    "metadata": {
                        "sources":    result.get("sources", []),
                        "latency_ms": result.get("latency_ms", latency),
                        "routing":    result.get("routing", {}),
                    },
                })
                st.rerun()


# ══════════════════════════════════════════════════════════════
#  TAB 3 — DATA UPLOAD & INGEST ORCHESTRATOR
# ══════════════════════════════════════════════════════════════
with tab_upload:

    st.markdown("## Data Upload & Ingest Orchestrator")
    st.caption(
        "Upload CSVs or PDFs into the correct GridFS bucket, "
        "then run each ingestion step without touching the command line."
    )

    # ── Import ingest runner ──────────────────────────────────
    try:
        from ingest.ingest_runner import (
            run_load_mongo,
            run_extract_pos,
            run_extract_entities,
            run_load_neo4j,
            run_load_pos_neo4j,
            run_embed_documents,
        )
        runner_available = True
    except ImportError:
        runner_available = False

    buckets = get_gridfs_buckets()

    # ── SECTION 1: Upload ─────────────────────────────────────
    st.markdown("### Upload files")

    up_col1, up_col2 = st.columns([3, 1])
    with up_col1:
        uploaded_files = st.file_uploader(
            "Choose CSVs or PDFs",
            type=["csv", "pdf"],
            accept_multiple_files=True,
            key="gridfs_uploader",
            help="CSVs → csv_files bucket · PDFs → po_files bucket",
        )
    with up_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("**CSVs** → `csv_files` bucket  \n**PDFs** → `po_files` bucket")

    if uploaded_files:
        if st.button("Upload to GridFS", type="primary", key="btn_upload"):
            upload_results = []
            progress = st.progress(0)

            for idx, uf in enumerate(uploaded_files):
                filename    = uf.name
                is_pdf      = filename.lower().endswith(".pdf")
                bucket      = buckets["pdf"] if is_pdf else buckets["csv"]
                ctype       = "application/pdf" if is_pdf else "text/csv"
                bucket_name = "po_files" if is_pdf else "csv_files"

                if bucket.find_one({"filename": filename}):
                    upload_results.append(("skip", filename, bucket_name))
                else:
                    try:
                        file_id = bucket.put(
                            uf.getvalue(),
                            filename=filename,
                            content_type=ctype,
                        )
                        upload_results.append(("ok", filename, bucket_name, str(file_id)))
                    except Exception as exc:
                        upload_results.append(("err", filename, bucket_name, str(exc)))

                progress.progress((idx + 1) / len(uploaded_files))

            progress.empty()
            for row in upload_results:
                if row[0] == "ok":
                    st.success(f"✓ **{row[1]}** → `{row[2]}` (id: `{row[3]}`)")
                elif row[0] == "skip":
                    st.info(f"↷ **{row[1]}** already in `{row[2]}` — skipped")
                else:
                    st.error(f"✗ **{row[1]}** failed: {row[3]}")

    st.divider()

    # ── SECTION 2: Files in GridFS ────────────────────────────
    st.markdown("### Files in GridFS")

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        st.button("Refresh", key="btn_refresh_gridfs")

    try:
        bucket_map = {
            "csv_files":   buckets["csv"],
            "po_files":    buckets["pdf"],
            "email_files": buckets["email"],
        }

        rows = []
        for bucket_name, fs in bucket_map.items():
            for d in fs.find():
                rows.append({
                    "Bucket":    bucket_name,
                    "Filename":  d.filename,
                    "Type":      d.content_type or "—",
                    "Size (KB)": round(d.length / 1024, 1),
                    "Uploaded":  d.upload_date.strftime("%Y-%m-%d %H:%M") if d.upload_date else "—",
                })

        if not rows:
            st.info("No files found in any GridFS bucket.")
        else:
            df_fs = pd.DataFrame(rows).sort_values(["Bucket", "Filename"])
            st.dataframe(df_fs, use_container_width=True, hide_index=True)
            st.caption(
                f"{len(rows)} file(s) total · "
                f"{sum(r['Size (KB)'] for r in rows):.1f} KB · "
                f"across {len(bucket_map)} buckets"
            )

    except Exception as e:
        st.error(f"Could not read GridFS: {e}")

    st.divider()

    # ── SECTION 3: Full ingest pipeline ──────────────────────
    st.markdown("### Ingest pipeline")
    st.caption(
        "Run steps in order after uploading. "
        "Results show which collections were updated."
    )
    st.info("""
    **Which steps to run after uploading:**

     **New PO PDF** → run in this order: **② then ⑤ then ⑥**
    - ② extracts fields from the PDF into the `pos` collection
    - ⑤ creates the PO node in Neo4j and links to customer, job and materials
    - ⑥ embeds the PO description so the chatbot can find it via vector search

     **New CSV data** (customers, invoices, items) → run in this order: **① then ④ then ⑥**
    - ① loads CSV records into MongoDB collections
    - ④ projects the updated data into the Neo4j graph
    - ⑥ regenerates embeddings to reflect new records

     **New emails** → run in this order: **③ then ④ then ⑥**
    - ③ extracts customer and job entities from emails
    - ④ adds email nodes to the Neo4j graph
    - ⑥ embeds email bodies for vector search

     **Full rebuild** → click **▶ Run all steps**
    """)

    if not runner_available:
        st.warning(
            "`ingest/ingest_runner.py` not found — pipeline buttons disabled. "
            "Add it to the `ingest/` folder and redeploy."
        )

    # All 6 pipeline steps
    pipeline_steps = [
        {
            "key":   "step_load_mongo",
            "label": "① Load CSVs → MongoDB",
            "desc":  "Reads CSVs from `csv_files` bucket · upserts into `customers`, "
                     "`invoices`, `job_types`, `items`, `job_items`, `invoice_items`.",
            "fn":    "run_load_mongo",
        },
        {
            "key":   "step_extract_pos",
            "label": "② Extract PO PDFs → `pos`",
            "desc":  "Reads PDFs from `po_files` bucket · Claude Haiku extracts fields · "
                     "upserts structured records into `pos` collection.",
            "fn":    "run_extract_pos",
        },
        {
            "key":   "step_extract_entities",
            "label": "③ Extract entities from emails",
            "desc":  "Reads emails from `email_files` bucket · Claude Haiku extracts "
                     "customer and job entities · writes into `emails` collection.",
            "fn":    "run_extract_entities",
        },
        {
            "key":   "step_load_neo4j",
            "label": "④ Load MongoDB → Neo4j graph",
            "desc":  "Projects customers, invoices, job types and items from MongoDB "
                     "into Neo4j as nodes and relationships.",
            "fn":    "run_load_neo4j",
        },
        {
            "key":   "step_load_pos_neo4j",
            "label": "⑤ Load POs → Neo4j graph",
            "desc":  "Creates PO nodes in Neo4j · connects to Customer, JobType and "
                     "Item nodes via FOR_CUSTOMER, FOR_JOB, CONTAINS_ITEM relationships.",
            "fn":    "run_load_pos_neo4j",
        },
        {
            "key":   "step_embed",
            "label": "⑥ Generate embeddings → Vector index",
            "desc":  "Embeds email bodies, PO descriptions and customer notes using "
                     "all-MiniLM-L6-v2 · writes 384-dim vectors into `embeddings`.",
            "fn":    "run_embed_documents",
        },
    ]

    # Initialise step states
    for step in pipeline_steps:
        if step["key"] not in st.session_state:
            st.session_state[step["key"]] = "idle"

    # Build function map
    fn_map = {}
    if runner_available:
        fn_map = {
            "run_load_mongo":        run_load_mongo,
            "run_extract_pos":       run_extract_pos,
            "run_extract_entities":  run_extract_entities,
            "run_load_neo4j":        run_load_neo4j,
            "run_load_pos_neo4j":    run_load_pos_neo4j,
            "run_embed_documents":   run_embed_documents,
        }

    for step in pipeline_steps:
        with st.container():
            c_label, c_btn = st.columns([4, 1])
            with c_label:
                st.markdown(f"**{step['label']}**")
                st.caption(step["desc"])
            with c_btn:
                state     = st.session_state[step["key"]]
                btn_label = {
                    "idle":    "Run",
                    "running": "Running…",
                    "done":    "✓ Done",
                    "error":   "✗ Error",
                }.get(state, "Run")

                disabled = (not runner_available) or (state == "running")
                if st.button(
                    btn_label,
                    key=f"btn_{step['key']}",
                    disabled=disabled,
                    use_container_width=True,
                ):
                    st.session_state[step["key"]] = "running"
                    st.session_state.ingest_log.append(
                        f"[{time.strftime('%H:%M:%S')}] Starting: {step['label']}"
                    )
                    try:
                        result_msg = fn_map[step["fn"]]()
                        st.session_state[step["key"]] = "done"
                        st.session_state.ingest_log.append(
                            f"[{time.strftime('%H:%M:%S')}] ✓ {step['label']}: {result_msg}"
                        )
                    except Exception as exc:
                        st.session_state[step["key"]] = "error"
                        st.session_state.ingest_log.append(
                            f"[{time.strftime('%H:%M:%S')}] ✗ {step['label']} FAILED: {exc}"
                        )
                    st.rerun()

        st.markdown(
            "<hr style='border:none;border-top:1px solid #E5DDD2;margin:8px 0'>",
            unsafe_allow_html=True,
        )

    col_reset, col_runall = st.columns([1, 1])
    with col_reset:
        if st.button("Reset pipeline", key="btn_reset_pipeline", use_container_width=True):
            for step in pipeline_steps:
                st.session_state[step["key"]] = "idle"
            st.session_state.ingest_log = []
            st.rerun()
    with col_runall:
        if st.button(
            "▶ Run all steps",
            key="btn_run_all",
            type="primary",
            disabled=not runner_available,
            use_container_width=True,
        ):
            for step in pipeline_steps:
                st.session_state[step["key"]] = "running"
                st.session_state.ingest_log.append(
                    f"[{time.strftime('%H:%M:%S')}] Starting: {step['label']}"
                )
                try:
                    result_msg = fn_map[step["fn"]]()
                    st.session_state[step["key"]] = "done"
                    st.session_state.ingest_log.append(
                        f"[{time.strftime('%H:%M:%S')}] ✓ {step['label']}: {result_msg}"
                    )
                except Exception as exc:
                    st.session_state[step["key"]] = "error"
                    st.session_state.ingest_log.append(
                        f"[{time.strftime('%H:%M:%S')}] ✗ {step['label']} FAILED: {exc}"
                    )
            st.rerun()

    if st.session_state.ingest_log:
        st.markdown("### Ingest log")
        st.code("\n".join(st.session_state.ingest_log), language=None)


# ─────────────────────────────────────────────────────────────
# PAGE-WIDE FOOTER
# ─────────────────────────────────────────────────────────────
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
        st.session_state.messages        = []
        st.session_state.last_prediction = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
