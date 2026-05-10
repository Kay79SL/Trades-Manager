"""
ingest/ingest_runner.py
=======================

Wrappers called by the Streamlit Data Upload tab pipeline buttons.
Each function runs the corresponding ingest script as a subprocess,
captures stdout/stderr, and returns a summary string for the log.

Works both locally and on Streamlit Community Cloud — uses sys.executable
so it always picks up the correct Python wherever the app is running.

Location: ingest/ingest_runner.py  (relative to project root)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
# Project root = one level up from this ingest/ folder
PROJECT_ROOT = Path(__file__).parent.parent
INGEST_DIR   = PROJECT_ROOT / "ingest"

# sys.executable = correct Python in any environment:
#   local venv  → F:\Apps\DACARag\venv\Scripts\python.exe
#   Streamlit Cloud → /usr/local/bin/python  (or wherever it lives)
PYTHON = sys.executable


def _run(script_name: str) -> str:
    """
    Run a script in the ingest/ folder using the current Python executable.
    Returns the last 5 lines of output as a summary string.
    Raises RuntimeError on non-zero exit so Streamlit catches and logs it.
    """
    script_path = INGEST_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"{script_name} not found at {script_path}. "
            "Make sure all ingest scripts are committed to GitHub."
        )

    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(PROJECT_ROOT),      # run from project root so .env / secrets are found
        capture_output=True,
        text=True,
        timeout=300,                # 5 min max per step
    )

    output = (result.stdout or "").strip()
    errors = (result.stderr or "").strip()
    full   = "\n".join(filter(None, [output, errors]))

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} exited with code {result.returncode}.\n{full}"
        )

    lines   = [l for l in full.splitlines() if l.strip()]
    summary = "\n".join(lines[-5:]) if lines else "Completed (no output)"
    return summary


# ─────────────────────────────────────────────────────────────
# Step 1 — Load CSVs into MongoDB collections
# ─────────────────────────────────────────────────────────────
def run_load_mongo() -> str:
    """
    Calls load_mongo.py — reads CSVs from GridFS csv_files bucket
    and upserts records into customers, invoices, job_types, items,
    job_items, invoice_items collections.
    """
    return _run("load_mongo.py")


# ─────────────────────────────────────────────────────────────
# Step 2 — Extract PO PDFs → pos collection
# ─────────────────────────────────────────────────────────────
def run_extract_pos() -> str:
    """
    Calls extract_pos_from_pdf.py then load_po_pdfs.py in sequence.
    extract_pos_from_pdf.py — reads PDFs from GridFS po_files bucket,
                               sends to Claude Haiku for field extraction.
    load_po_pdfs.py         — loads the structured results into pos collection.
    """
    extract_summary = _run("extract_pos_from_pdf.py")
    load_summary    = _run("load_po_pdfs.py")
    return f"Extract: {extract_summary} | Load: {load_summary}"


# ─────────────────────────────────────────────────────────────
# Step 3 — Generate embeddings → Vector index
# ─────────────────────────────────────────────────────────────
def run_embed_documents() -> str:
    """
    Calls embed_chunks.py — embeds email bodies, PO descriptions,
    and customer notes into 384-dim vectors for Atlas Vector Search.
    """
    return _run("embed_chunks.py")
