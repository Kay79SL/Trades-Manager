"""
check_pipeline.py
==================
Pipeline health-check + audit log for the DACARag ingestion pipeline.

Audits the state of every email and PO in MongoDB, reports counts to the
console, and PERSISTS the full audit + every error to MongoDB so you have
a queryable history of pipeline health over time.

Two output streams:

  1. CONSOLE  — pretty-printed report + actionable next steps
  2. MONGODB  — written to two collections:
        pipeline_audits           one doc per check_pipeline.py run
        pipeline_errors           one doc per individual error encountered

This means you can answer questions like:
  - "Has email msg_0042 failed extraction more than once this week?"
  - "What was the pipeline state on Sunday at 5pm?"
  - "Show me every PO that's failed extraction in the last 24 hours"

The script never modifies the emails or pos collections themselves — only
writes to its own audit log collections. Safe to run anytime.

Usage:
    python ingest\\check_pipeline.py                 # full report + persist to Mongo
    python ingest\\check_pipeline.py --emails        # emails only
    python ingest\\check_pipeline.py --pos           # POs only
    python ingest\\check_pipeline.py --failures      # show error details for failed docs
    python ingest\\check_pipeline.py --stale 24      # flag docs ingested >24h ago without extraction
    python ingest\\check_pipeline.py --json          # output as JSON for CI / dashboards
    python ingest\\check_pipeline.py --no-persist    # skip MongoDB write (read-only run)
    python ingest\\check_pipeline.py --history 7     # show audit history from last 7 days
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------

def audit_emails(db, stale_hours: float = 24.0) -> dict[str, Any]:
    """Audit email pipeline state."""
    coll = db.emails
    total = coll.count_documents({})

    if total == 0:
        return {"total": 0, "note": "No emails in collection"}

    extracted = coll.count_documents({"extracted": {"$exists": True}})
    pending = coll.count_documents({
        "extracted": {"$exists": False},
        "extraction_failed": {"$exists": False},
    })
    failed = coll.count_documents({"extraction_failed": {"$exists": True}})
    failed_no_retry = coll.count_documents({
        "extracted": {"$exists": False},
        "extraction_failed": {"$exists": True},
    })

    # Confidence distribution
    confidence_dist: dict[str, int] = {}
    for r in coll.aggregate([
        {"$match": {"extracted.confidence": {"$exists": True}}},
        {"$group": {"_id": "$extracted.confidence", "count": {"$sum": 1}}},
    ]):
        confidence_dist[str(r["_id"])] = r["count"]

    # Trade distribution
    trade_dist: dict[str, int] = {}
    for r in coll.aggregate([
        {"$match": {"extracted.trade_needed": {"$exists": True}}},
        {"$group": {"_id": "$extracted.trade_needed", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]):
        trade_dist[str(r["_id"])] = r["count"]

    # Source breakdown (synthetic vs live)
    live = coll.count_documents({"_source": "ms_graph"})
    synthetic = total - live

    # Stale: ingested > N hours ago, still no extraction or failure
    stale_cutoff = time.time() - (stale_hours * 3600)
    stale = coll.count_documents({
        "_ingested_at": {"$lt": stale_cutoff},
        "extracted": {"$exists": False},
        "extraction_failed": {"$exists": False},
    })

    # Returning customers identified
    returning = coll.count_documents({"extracted.is_returning_customer": True})

    return {
        "total":              total,
        "extracted":          extracted,
        "extracted_pct":      f"{extracted / total * 100:.1f}%",
        "pending":            pending,
        "failed":             failed,
        "failed_no_retry":    failed_no_retry,
        "stale":              stale,
        "stale_threshold_h":  stale_hours,
        "synthetic":          synthetic,
        "live":               live,
        "trade_distribution": trade_dist,
        "confidence_distribution": confidence_dist,
        "returning_customers_identified": returning,
    }


def audit_pos(db, stale_hours: float = 24.0) -> dict[str, Any]:
    """Audit purchase order pipeline state."""
    coll = db.pos
    total = coll.count_documents({})

    if total == 0:
        return {"total": 0, "note": "No POs in collection"}

    extracted = coll.count_documents({"extracted": {"$exists": True}})
    pending = coll.count_documents({
        "extracted": {"$exists": False},
        "extraction_failed": {"$exists": False},
    })
    failed = coll.count_documents({"extraction_failed": {"$exists": True}})

    # Linkage to other entities
    linked_customer = coll.count_documents({"matched_customer_id": {"$ne": None}})
    linked_job = coll.count_documents({"matched_job_type_id": {"$ne": None}})
    linked_pdf = coll.count_documents({"pdf_gridfs_filename": {"$exists": True}})

    # PDF binaries in GridFS
    gridfs_count = db["po_files.files"].count_documents({})

    # Complexity distribution (from LLM extraction)
    complexity_dist: dict[str, int] = {}
    for r in coll.aggregate([
        {"$match": {"extracted.complexity": {"$exists": True}}},
        {"$group": {"_id": "$extracted.complexity", "count": {"$sum": 1}}},
    ]):
        complexity_dist[str(r["_id"])] = r["count"]

    # Stale check
    stale_cutoff = time.time() - (stale_hours * 3600)
    stale = coll.count_documents({
        "_parsed_at": {"$lt": stale_cutoff},
        "extracted": {"$exists": False},
        "extraction_failed": {"$exists": False},
    })

    # Risk-flagged POs (from LLM extraction)
    risk_flagged = coll.count_documents({
        "extracted.risk_flags": {"$exists": True, "$ne": []},
    })

    return {
        "total":              total,
        "extracted":          extracted,
        "extracted_pct":      f"{extracted / total * 100:.1f}%",
        "pending":            pending,
        "failed":             failed,
        "stale":              stale,
        "linked_customer":    linked_customer,
        "linked_job_type":    linked_job,
        "linked_pdf":         linked_pdf,
        "pdfs_in_gridfs":     gridfs_count,
        "complexity_distribution": complexity_dist,
        "risk_flagged":       risk_flagged,
    }


def show_failures(db, limit: int = 10) -> None:
    """Show details of recent extraction failures (emails + POs)."""
    print("\n" + "=" * 60)
    print(f"Recent extraction failures (top {limit})")
    print("=" * 60)

    found = 0
    print("\n[ EMAILS ]")
    for doc in db.emails.find(
        {"extraction_failed": {"$exists": True}},
        {"email_id": 1, "extraction_failed": 1, "from_email": 1},
    ).sort("extraction_failed.attempted_at", -1).limit(limit):
        attempted = doc["extraction_failed"].get("attempted_at", 0)
        ts = datetime.fromtimestamp(attempted).strftime("%Y-%m-%d %H:%M") if attempted else "?"
        print(f"  {doc['email_id']:14} attempted: {ts}")
        print(f"    from:     {doc.get('from_email', '')}")
        print(f"    attempts: {doc['extraction_failed'].get('attempts', '?')}")
        print(f"    error:    {doc['extraction_failed'].get('error', '')[:120]}")
        print()
        found += 1
    if found == 0:
        print("  (none)")

    found = 0
    print("\n[ POs ]")
    for doc in db.pos.find(
        {"extraction_failed": {"$exists": True}},
        {"po_number": 1, "extraction_failed": 1, "cust_name": 1},
    ).sort("extraction_failed.attempted_at", -1).limit(limit):
        attempted = doc["extraction_failed"].get("attempted_at", 0)
        ts = datetime.fromtimestamp(attempted).strftime("%Y-%m-%d %H:%M") if attempted else "?"
        print(f"  {doc['po_number']:18} attempted: {ts}")
        print(f"    customer: {doc.get('cust_name', '')}")
        print(f"    attempts: {doc['extraction_failed'].get('attempts', '?')}")
        print(f"    error:    {doc['extraction_failed'].get('error', '')[:120]}")
        print()
        found += 1
    if found == 0:
        print("  (none)")


def show_stale(db, stale_hours: float = 24.0, limit: int = 10) -> None:
    """Show docs that have been pending extraction for too long."""
    cutoff = time.time() - (stale_hours * 3600)
    print("\n" + "=" * 60)
    print(f"Stale documents (>{stale_hours}h pending, no failure record)")
    print("=" * 60)

    print("\n[ EMAILS ]")
    found = 0
    for doc in db.emails.find({
        "_ingested_at": {"$lt": cutoff},
        "extracted": {"$exists": False},
        "extraction_failed": {"$exists": False},
    }, {"email_id": 1, "_ingested_at": 1, "from_email": 1}).limit(limit):
        ts = datetime.fromtimestamp(doc.get("_ingested_at", 0)).strftime("%Y-%m-%d %H:%M")
        print(f"  {doc['email_id']:14} ingested: {ts}  from: {doc.get('from_email', '')}")
        found += 1
    if found == 0:
        print("  (none)")

    print("\n[ POs ]")
    found = 0
    for doc in db.pos.find({
        "_parsed_at": {"$lt": cutoff},
        "extracted": {"$exists": False},
        "extraction_failed": {"$exists": False},
    }, {"po_number": 1, "_parsed_at": 1, "cust_name": 1}).limit(limit):
        ts = datetime.fromtimestamp(doc.get("_parsed_at", 0)).strftime("%Y-%m-%d %H:%M")
        print(f"  {doc['po_number']:18} parsed: {ts}  customer: {doc.get('cust_name', '')}")
        found += 1
    if found == 0:
        print("  (none)")


# ---------------------------------------------------------------------------
# Persistence — write audits and errors to MongoDB
# ---------------------------------------------------------------------------

AUDITS_COLL = "pipeline_audits"
ERRORS_COLL = "pipeline_errors"


def persist_audit(db, email_audit: dict, pos_audit: dict, run_args: dict) -> str:
    """
    Save a snapshot of the pipeline state to the pipeline_audits collection.
    Returns the inserted document's audit_id (an ISO timestamp).
    """
    audit_id = datetime.now().isoformat()
    audit_doc = {
        "audit_id":   audit_id,
        "checked_at": time.time(),
        "checked_at_iso": audit_id,
        "emails":     email_audit,
        "pos":        pos_audit,
        "args":       run_args,
        "host":       os.environ.get("COMPUTERNAME", "unknown"),
    }
    db[AUDITS_COLL].insert_one(audit_doc)

    # Index for fast time-range queries on history
    db[AUDITS_COLL].create_index("checked_at")

    return audit_id


def persist_errors(db, audit_id: str) -> int:
    """
    For every document currently flagged with extraction_failed, write a
    snapshot of that error to the pipeline_errors collection. Returns the
    number of error rows inserted in this run.

    De-duplicates on (collection, doc_id, attempted_at) so re-running the
    audit doesn't create duplicate error rows for the same failure event.
    """
    inserted = 0

    # Email failures
    for doc in db.emails.find(
        {"extraction_failed": {"$exists": True}},
        {"email_id": 1, "extraction_failed": 1, "from_email": 1, "subject": 1},
    ):
        ef = doc.get("extraction_failed", {})
        attempted_at = ef.get("attempted_at", 0)
        existing = db[ERRORS_COLL].find_one({
            "collection":   "emails",
            "doc_id":       doc["email_id"],
            "attempted_at": attempted_at,
        })
        if existing:
            continue
        db[ERRORS_COLL].insert_one({
            "audit_id":     audit_id,
            "collection":   "emails",
            "doc_id":       doc["email_id"],
            "from_email":   doc.get("from_email", ""),
            "subject":      (doc.get("subject", "") or "")[:120],
            "error":        ef.get("error", ""),
            "attempts":     ef.get("attempts", 1),
            "model":        ef.get("model", ""),
            "attempted_at": attempted_at,
            "logged_at":    time.time(),
        })
        inserted += 1

    # PO failures
    for doc in db.pos.find(
        {"extraction_failed": {"$exists": True}},
        {"po_number": 1, "extraction_failed": 1, "cust_name": 1, "job_type": 1},
    ):
        ef = doc.get("extraction_failed", {})
        attempted_at = ef.get("attempted_at", 0)
        existing = db[ERRORS_COLL].find_one({
            "collection":   "pos",
            "doc_id":       doc["po_number"],
            "attempted_at": attempted_at,
        })
        if existing:
            continue
        db[ERRORS_COLL].insert_one({
            "audit_id":     audit_id,
            "collection":   "pos",
            "doc_id":       doc["po_number"],
            "cust_name":    doc.get("cust_name", ""),
            "job_type":     doc.get("job_type", ""),
            "error":        ef.get("error", ""),
            "attempts":     ef.get("attempts", 1),
            "model":        ef.get("model", ""),
            "attempted_at": attempted_at,
            "logged_at":    time.time(),
        })
        inserted += 1

    # Indexes for fast queries
    db[ERRORS_COLL].create_index([("collection", 1), ("doc_id", 1), ("attempted_at", 1)])
    db[ERRORS_COLL].create_index("logged_at")

    return inserted


def show_history(db, days: float = 7.0) -> None:
    """Show the last N days of audit runs."""
    cutoff = time.time() - (days * 86400)
    audits = list(db[AUDITS_COLL].find(
        {"checked_at": {"$gte": cutoff}},
        {"audit_id": 1, "emails": 1, "pos": 1, "args": 1},
    ).sort("checked_at", -1))

    print("\n" + "=" * 60)
    print(f"  Audit history (last {days} day(s)): {len(audits)} run(s)")
    print("=" * 60)

    if not audits:
        print("  (no audits found in this window)")
        return

    print(f"\n  {'When':<19} {'Emails':>10} {'POs':>10} {'Errors':>8}")
    print(f"  {'-'*19} {'-'*10} {'-'*10} {'-'*8}")
    for a in audits:
        ts = a["audit_id"][:19].replace("T", " ")
        em = a.get("emails", {})
        po = a.get("pos", {})
        em_str = f"{em.get('extracted', 0)}/{em.get('total', 0)}"
        po_str = f"{po.get('extracted', 0)}/{po.get('total', 0)}"
        err = (em.get("failed", 0) or 0) + (po.get("failed", 0) or 0)
        print(f"  {ts:<19} {em_str:>10} {po_str:>10} {err:>8}")


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_section(title: str, data: dict, color_code: str = "") -> None:
    print(f"\n{color_code}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {str(k):20} {v}")
        else:
            label = key.replace("_", " ")
            print(f"  {label:30} {value}")


def print_action_items(emails: dict, pos: dict) -> None:
    """Print actionable next steps based on the audit."""
    print("\n" + "=" * 60)
    print("  Actionable next steps")
    print("=" * 60)

    actions = []

    if emails.get("pending", 0) > 0:
        actions.append(
            f"  • {emails['pending']} email(s) pending extraction → "
            f"run: python ingest\\extract_entities.py"
        )
    if emails.get("failed_no_retry", 0) > 0:
        actions.append(
            f"  • {emails['failed_no_retry']} email(s) previously failed without success → "
            f"investigate with --failures, then retry"
        )
    if emails.get("stale", 0) > 0:
        actions.append(
            f"  • {emails['stale']} email(s) stale (>{emails['stale_threshold_h']}h old, no extraction or failure) → "
            f"check with --stale"
        )

    if pos.get("pending", 0) > 0:
        actions.append(
            f"  • {pos['pending']} PO(s) pending enrichment → "
            f"run: python ingest\\extract_pos.py"
        )
    if pos.get("failed", 0) > 0:
        actions.append(
            f"  • {pos['failed']} PO(s) failed enrichment → "
            f"investigate with --failures"
        )
    if pos.get("total", 0) > pos.get("linked_customer", 0):
        unmatched = pos["total"] - pos["linked_customer"]
        actions.append(
            f"  • {unmatched} PO(s) without matched customer → "
            f"check cust_email values in Excel match customers.csv"
        )
    if pos.get("total", 0) > pos.get("linked_job_type", 0):
        unmatched = pos["total"] - pos["linked_job_type"]
        actions.append(
            f"  • {unmatched} PO(s) without matched job_type → "
            f"run: python ingest\\load_pos_to_neo4j.py (re-runs job-type matching)"
        )
    if pos.get("total", 0) > pos.get("pdfs_in_gridfs", 0):
        missing = pos["total"] - pos["pdfs_in_gridfs"]
        actions.append(
            f"  • {missing} PO(s) without PDF in GridFS → "
            f"save Excel sheets as PDFs and run: python ingest\\load_po_pdfs.py"
        )

    if not actions:
        print("\n  ✓ Pipeline is in a clean state. Nothing to do.")
    else:
        print()
        for a in actions:
            print(a)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emails", action="store_true", help="Audit emails only")
    parser.add_argument("--pos", action="store_true", help="Audit POs only")
    parser.add_argument("--failures", action="store_true", help="Show failure details")
    parser.add_argument("--stale", type=float, default=None, metavar="HOURS",
                        help="Stale-document threshold in hours (default 24)")
    parser.add_argument("--show-stale", action="store_true",
                        help="List stale documents")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON (for CI / dashboards)")
    parser.add_argument("--no-persist", action="store_true",
                        help="Skip writing audit + errors to MongoDB")
    parser.add_argument("--history", type=float, default=None, metavar="DAYS",
                        help="Show audit run history from last N days, then exit")
    args = parser.parse_args()

    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    if not mongo_uri:
        print("ERROR: MONGO_URI not set in .env"); sys.exit(1)

    db = MongoClient(mongo_uri)[db_name]

    # History mode — print past runs and exit (no new audit)
    if args.history is not None:
        show_history(db, days=args.history)
        return

    stale_hours = args.stale if args.stale is not None else 24.0

    show_emails = args.emails or not args.pos
    show_pos = args.pos or not args.emails

    email_audit = audit_emails(db, stale_hours) if show_emails else {}
    pos_audit = audit_pos(db, stale_hours) if show_pos else {}

    # Persist to MongoDB unless --no-persist
    audit_id = None
    new_errors = 0
    if not args.no_persist:
        audit_id = persist_audit(
            db, email_audit, pos_audit,
            run_args={
                "emails":      args.emails,
                "pos":         args.pos,
                "stale_hours": stale_hours,
            },
        )
        new_errors = persist_errors(db, audit_id)

    if args.json:
        output = {
            "emails": email_audit,
            "pos": pos_audit,
            "checked_at": datetime.now().isoformat(),
        }
        if audit_id:
            output["audit_id"] = audit_id
            output["new_errors_logged"] = new_errors
        print(json.dumps(output, indent=2, default=str))
        return

    print(f"\n  DACARag Pipeline Health Check")
    print(f"  Database: {db_name}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if audit_id:
        print(f"  Audit ID: {audit_id}")

    if show_emails:
        print_section("EMAILS", email_audit)
    if show_pos:
        print_section("PURCHASE ORDERS", pos_audit)

    if audit_id:
        total_errors_in_log = db[ERRORS_COLL].count_documents({})
        print(f"\n  Persisted to MongoDB:")
        print(f"    Audit log:           {AUDITS_COLL} (audit_id = {audit_id})")
        print(f"    New errors logged:   {new_errors}")
        print(f"    Total errors in log: {total_errors_in_log}")

    if args.failures:
        show_failures(db)

    if args.show_stale:
        show_stale(db, stale_hours)

    if not args.failures and not args.show_stale:
        print_action_items(email_audit, pos_audit)


if __name__ == "__main__":
    main()
