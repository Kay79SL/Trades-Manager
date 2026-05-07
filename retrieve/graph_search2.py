"""
graph_search.py
================
Module 3 of Phase 5 retrieval orchestration.

Cypher query templates for Neo4j. Used when the user's question is
fundamentally about *relationships* — "who's done similar work", "what
materials are typically used for X", "show me past customers like this".

Vector search can't elegantly answer multi-hop relationship queries.
Cypher can. This module wraps a handful of common templates.

Used by:
  - orchestrator.py (when router picks "graph" path)
  - predict_quote.py (uses items_for_job to find the materials recipe)
  - context_assembler.py (to enrich vector results with relationship context)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


@lru_cache(maxsize=1)
# Module-level driver cache (replaces @lru_cache so we can invalidate it)
_DRIVER = {"instance": None}


def _create_driver():
    """Create a fresh Neo4j driver from .env credentials."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(
        uri,
        auth=(user, password),
        # AuraDB Free pauses after idle - keep connections fresh
        max_connection_lifetime=300,  # 5 min
        connection_acquisition_timeout=30,
    )


def _get_driver():
    """
    Get a working Neo4j driver. If the cached driver is dead (AuraDB
    paused), transparently reconnect. End user never sees the error.
    """
    # First call - create driver
    if _DRIVER["instance"] is None:
        _DRIVER["instance"] = _create_driver()
        return _DRIVER["instance"]

    # Subsequent calls - ping to verify driver is alive
    try:
        with _DRIVER["instance"].session() as s:
            s.run("RETURN 1").single()
        return _DRIVER["instance"]
    except Exception:
        # Driver is defunct - close, reconnect
        try:
            _DRIVER["instance"].close()
        except Exception:
            pass
        _DRIVER["instance"] = _create_driver()
        return _DRIVER["instance"]


def close_driver():
    """Close the cached driver (call at app shutdown)."""
    if _DRIVER["instance"] is not None:
        try:
            _DRIVER["instance"].close()
        except Exception:
            pass
        _DRIVER["instance"] = None


# ---------------------------------------------------------------------------
# Material recipe queries (used by predict_quote.py)
# ---------------------------------------------------------------------------

def items_for_job(job_type_id: str) -> list[dict]:
    """
    Find typical materials for a job type via the USES_ITEM graph edge.
    This is the 'recipe' that drives the materials cost prediction.
    """
    cypher = """
    MATCH (j:JobType {job_type_id: $job_type_id})-[r:USES_ITEM]->(i:Item)
    RETURN
      i.item_id            AS item_id,
      i.item_name          AS item_name,
      i.unit_price_ex_vat  AS unit_price,
      i.unit               AS unit,
      r.typical_quantity   AS typical_quantity
    ORDER BY i.item_name
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, job_type_id=job_type_id)]


def items_for_job_by_name(job_name: str) -> list[dict]:
    """Same as items_for_job but lookup by display name."""
    cypher = """
    MATCH (j:JobType)-[r:USES_ITEM]->(i:Item)
    WHERE toLower(j.job_name) = toLower($job_name)
    RETURN
      i.item_id            AS item_id,
      i.item_name          AS item_name,
      i.unit_price_ex_vat  AS unit_price,
      i.unit               AS unit,
      r.typical_quantity   AS typical_quantity
    ORDER BY i.item_name
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, job_name=job_name)]


# ---------------------------------------------------------------------------
# Customer history queries
# ---------------------------------------------------------------------------

def customer_full_history(customer_id: str) -> dict[str, list]:
    """
    Pull everything connected to a customer in the graph: past invoices,
    POs, emails, preferred trade. Returns grouped by relationship type.
    """
    cypher = """
    MATCH (c:Customer {customer_id: $customer_id})
    OPTIONAL MATCH (c)<-[:FOR_CUSTOMER]-(i:Invoice)
    OPTIONAL MATCH (c)<-[:FOR_CUSTOMER]-(po:PO)
    OPTIONAL MATCH (c)<-[:FROM_CUSTOMER]-(e:Email)
    OPTIONAL MATCH (c)-[:PREFERS]->(t:Trade)
    RETURN
      c.customer_id                                  AS customer_id,
      c.first_name                                   AS first_name,
      c.last_name                                    AS last_name,
      collect(DISTINCT i.invoice_id)                 AS invoice_ids,
      collect(DISTINCT po.po_number)                 AS po_numbers,
      collect(DISTINCT e.email_id)                   AS email_ids,
      collect(DISTINCT t.name)                       AS preferred_trades
    """
    with _get_driver().session() as s:
        rec = s.run(cypher, customer_id=customer_id).single()
        if not rec:
            return {}
        return dict(rec)


def similar_customers_by_job(job_type_id: str, limit: int = 5) -> list[dict]:
    """
    Find customers who have had work done in the same job_type.
    Useful for 'have we worked with someone like this before?' queries.
    """
    cypher = """
    MATCH (j:JobType {job_type_id: $job_type_id})<-[:FOR_JOB|ABOUT_JOB]-(rec)
    MATCH (rec)-[:FOR_CUSTOMER|FROM_CUSTOMER]->(c:Customer)
    RETURN
      c.customer_id     AS customer_id,
      c.first_name      AS first_name,
      c.last_name       AS last_name,
      count(DISTINCT rec) AS interaction_count
    ORDER BY interaction_count DESC
    LIMIT $limit
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, job_type_id=job_type_id, limit=limit)]


# ---------------------------------------------------------------------------
# PO and invoice graph queries
# ---------------------------------------------------------------------------

def po_full_context(po_number: str) -> dict:
    """Pull a PO with all its connected entities (customer, job_type, items)."""
    cypher = """
    MATCH (po:PO {po_number: $po_number})
    OPTIONAL MATCH (po)-[:FOR_CUSTOMER]->(c:Customer)
    OPTIONAL MATCH (po)-[:FOR_JOB]->(j:JobType)
    OPTIONAL MATCH (po)-[:CONTAINS_ITEM]->(i:Item)
    RETURN
      po.po_number       AS po_number,
      c.customer_id      AS customer_id,
      c.first_name + ' ' + c.last_name AS customer_name,
      j.job_name         AS job_name,
      collect(DISTINCT i.item_name) AS items
    """
    with _get_driver().session() as s:
        rec = s.run(cypher, po_number=po_number).single()
        return dict(rec) if rec else {}


def similar_pos_by_job_type(job_name: str, limit: int = 5) -> list[dict]:
    """Find past POs for the same job_type — used for price benchmarking."""
    cypher = """
    MATCH (po:PO)-[:FOR_JOB]->(j:JobType)
    WHERE toLower(j.job_name) CONTAINS toLower($job_name)
    RETURN
      po.po_number          AS po_number,
      po.subtotal_ex_vat    AS subtotal,
      po.total_inc_vat      AS total,
      j.job_name            AS job_name
    ORDER BY po.po_number DESC
    LIMIT $limit
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, job_name=job_name, limit=limit)]


# ---------------------------------------------------------------------------
# Trade-based queries
# ---------------------------------------------------------------------------

def jobs_for_trade(trade: str) -> list[dict]:
    """All job types served by a trade (plumber, carpenter, electrician)."""
    cypher = """
    MATCH (j:JobType)-[:OF_TRADE]->(t:Trade {name: $trade})
    RETURN
      j.job_type_id   AS job_type_id,
      j.job_name      AS job_name
    ORDER BY j.job_name
    """
    with _get_driver().session() as s:
        return [dict(r) for r in s.run(cypher, trade=trade.lower())]


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def graph_traversal(params: dict) -> dict[str, Any]:
    """
    Dispatcher for the router. Maps params to the right Cypher template.
    """
    out: dict[str, Any] = {}

    if "job_type_id" in params:
        out["items"] = items_for_job(params["job_type_id"])
        out["similar_customers"] = similar_customers_by_job(params["job_type_id"])

    if "job_name" in params:
        out["items_by_name"] = items_for_job_by_name(params["job_name"])
        out["similar_pos"] = similar_pos_by_job_type(params["job_name"])

    if "customer_id" in params:
        out["customer_history"] = customer_full_history(params["customer_id"])

    if "po_number" in params:
        out["po_context"] = po_full_context(params["po_number"])

    if "trade" in params:
        out["jobs"] = jobs_for_trade(params["trade"])

    return out


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python graph_search.py <customer_id|po_number|job_type_id|trade>")
        sys.exit(0)

    arg = sys.argv[1]
    if arg.startswith("cust_"):
        out = graph_traversal({"customer_id": arg})
    elif arg.startswith("PO-"):
        out = graph_traversal({"po_number": arg})
    elif arg in ("plumber", "carpenter", "electrician"):
        out = graph_traversal({"trade": arg})
    else:
        # Try as job_type_id
        out = graph_traversal({"job_type_id": arg})
    print(json.dumps(out, indent=2, default=str))
