"""
mongo_lookup.py
================
Module 2 of Phase 5 retrieval orchestration.

Direct, exact fetches from MongoDB. No AI involved — just CRUD lookups by
ID, email, customer, etc. Used when the user's intent is "give me this
specific record" rather than "find me something similar".

Design pattern: each function returns either a single document, a list,
or None. None means "not found" (caller decides how to handle).

Used by:
  - orchestrator.py (when router picks "mongo" path)
  - graph_search.py (to enrich graph results with full Mongo documents)
  - predict_quote.py (to fetch invoice statistics)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any
import re

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()


@lru_cache(maxsize=1)
def _get_db():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    return MongoClient(mongo_uri)[db_name]


# ---------------------------------------------------------------------------
# Customer lookups
# ---------------------------------------------------------------------------

def get_customer(customer_id: str) -> dict | None:
    """Fetch a single customer by canonical customer_id."""
    return _get_db().customers.find_one({"customer_id": customer_id}, {"_id": 0})


def get_customer_by_email(email: str) -> dict | None:
    """Find a customer by email (case-insensitive)."""
    if not email:
        return None
    return _get_db().customers.find_one(
        {"email": email.strip().lower()}, {"_id": 0}
    )


def get_customers_by_name(name: str) -> list[dict]:
    """
    Find customers matching a name. May return multiple
    (e.g. 3 Gerards in the dataset). Caller must disambiguate.
    """
    if not name:
        return []
    parts = name.strip().split(maxsplit=1)
    if len(parts) == 2:
        first, last = parts
        return list(_get_db().customers.find(
            {"first_name": first, "last_name": last}, {"_id": 0}
        ))
    # Single token — try first_name OR last_name
    return list(_get_db().customers.find(
        {"$or": [{"first_name": name}, {"last_name": name}]}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# Invoice lookups
# ---------------------------------------------------------------------------

def get_invoice(invoice_id: str) -> dict | None:
    """Fetch a single invoice by invoice_id."""
    return _get_db().invoices.find_one({"invoice_id": invoice_id}, {"_id": 0})


def get_invoices_for_customer(customer_id: str, limit: int = 20) -> list[dict]:
    """All invoices for a customer, most recent first."""
    return list(_get_db().invoices.find(
        {"customer_id": customer_id}, {"_id": 0}
    ).sort("invoice_date", -1).limit(limit))


def get_invoices_for_job_type(job_type_id: str, limit: int = 50) -> list[dict]:
    """All invoices for a particular job type — used for pricing statistics."""
    return list(_get_db().invoices.find(
        {"job_type_id": job_type_id}, {"_id": 0}
    ).limit(limit))


def get_invoice_line_items(invoice_id: str) -> list[dict]:
    """All line items belonging to a particular invoice."""
    return list(_get_db().invoice_items.find(
        {"invoice_id": invoice_id}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# PO lookups
# ---------------------------------------------------------------------------

def get_po(po_number: str) -> dict | None:
    """Fetch a single PO by po_number."""
    return _get_db().pos.find_one({"po_number": po_number}, {"_id": 0})


def get_pos_for_customer(customer_id: str) -> list[dict]:
    """All POs linked to a particular customer."""
    return list(_get_db().pos.find(
        {"matched_customer_id": customer_id}, {"_id": 0}
    ))


def get_pos_for_job_type(job_type_text: str) -> list[dict]:
    """
    Fuzzy match POs by their job_type field (free-form string from PDF).
    Case-insensitive substring match.
    """
    if not job_type_text:
        return []
    pattern = re.escape(job_type_text)
    return list(_get_db().pos.find(
        {"job_type": {"$regex": pattern, "$options": "i"}}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# Email lookups
# ---------------------------------------------------------------------------

def get_email(email_id: str) -> dict | None:
    """Fetch a single email by email_id."""
    return _get_db().emails.find_one({"email_id": email_id}, {"_id": 0})


def get_emails_for_customer(customer_id: str) -> list[dict]:
    """All emails for a customer (matched via the LLM extraction)."""
    return list(_get_db().emails.find(
        {"extracted.matched_customer_id": customer_id}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# Job type and item lookups
# ---------------------------------------------------------------------------

def get_job_type(job_type_id: str) -> dict | None:
    """Fetch a job type by canonical id (e.g. 'pl_01')."""
    return _get_db().job_types.find_one({"job_type_id": job_type_id}, {"_id": 0})


def get_job_type_by_name(job_name: str) -> dict | None:
    """Fetch a job type by display name (e.g. 'Boiler installation')."""
    if not job_name:
        return None
    return _get_db().job_types.find_one(
        {"job_name": {"$regex": f"^{re.escape(job_name)}$", "$options": "i"}},
        {"_id": 0}
    )


def get_item(item_id: str) -> dict | None:
    """Fetch an item by canonical id (e.g. 'it_pl_009')."""
    return _get_db().items.find_one({"item_id": item_id}, {"_id": 0})


def get_items_by_ids(item_ids: list[str]) -> list[dict]:
    """Bulk fetch items by a list of ids."""
    if not item_ids:
        return []
    return list(_get_db().items.find(
        {"item_id": {"$in": item_ids}}, {"_id": 0}
    ))


# ---------------------------------------------------------------------------
# High-level "by intent" dispatcher
# ---------------------------------------------------------------------------

def lookup_by_intent(params: dict) -> dict[str, Any]:
    """
    Dispatcher for the router. Takes a dict of params indicating what to
    look up and returns a structured result. Examples:
        {"po_number": "PO-2026-P0042"}     → fetch single PO
        {"customer_id": "cust_0001"}       → fetch customer + their invoices + POs
        {"customer_email": "x@y.ie"}       → resolve email to customer, then fetch all
        {"invoice_id": "INV-2025-0042"}    → fetch single invoice + line items
    """
    out: dict[str, Any] = {}

    # PO lookup
    if "po_number" in params:
        po = get_po(params["po_number"])
        out["po"] = po
        if po and po.get("matched_customer_id"):
            out["customer"] = get_customer(po["matched_customer_id"])
        return out

    # Invoice lookup
    if "invoice_id" in params:
        inv = get_invoice(params["invoice_id"])
        out["invoice"] = inv
        if inv:
            out["line_items"] = get_invoice_line_items(params["invoice_id"])
            if inv.get("customer_id"):
                out["customer"] = get_customer(inv["customer_id"])
        return out

    # Customer lookup (resolve email or name to customer_id first)
    customer_id = params.get("customer_id")
    if not customer_id and params.get("customer_email"):
        cust = get_customer_by_email(params["customer_email"])
        if cust:
            customer_id = cust["customer_id"]
    if not customer_id and params.get("customer_name"):
        matches = get_customers_by_name(params["customer_name"])
        if len(matches) == 1:
            customer_id = matches[0]["customer_id"]
        elif len(matches) > 1:
            # Ambiguous — return all options for caller to disambiguate
            out["ambiguous_customers"] = matches
            return out

    if customer_id:
        out["customer"] = get_customer(customer_id)
        out["invoices"] = get_invoices_for_customer(customer_id)
        out["pos"] = get_pos_for_customer(customer_id)
        out["emails"] = get_emails_for_customer(customer_id)

    return out


# ---------------------------------------------------------------------------
# CLI for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python mongo_lookup.py <customer_id|po_number|invoice_id>")
        sys.exit(0)

    arg = sys.argv[1]
    if arg.startswith("PO-"):
        result = lookup_by_intent({"po_number": arg})
    elif arg.startswith("INV-"):
        result = lookup_by_intent({"invoice_id": arg})
    elif arg.startswith("cust_"):
        result = lookup_by_intent({"customer_id": arg})
    else:
        result = lookup_by_intent({"customer_name": arg})
    print(json.dumps(result, indent=2, default=str))
