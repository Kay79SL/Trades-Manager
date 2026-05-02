"""
context_assembler.py
=====================
Module 5 of Phase 5 retrieval orchestration.

Takes raw results from one or more retrievers (vector_search, mongo_lookup,
graph_search, predict_quote) and produces a single clean text block to
inject into Claude's prompt. Also tracks source provenance so the final
chatbot response can cite where information came from.

Design goals:
  - Compact: every irrelevant byte costs Claude tokens
  - Structured: clear section headers so Claude can find what it needs
  - Cited: every fact is tagged with its source for transparency
  - Deduped: same chunk appearing in multiple retrievers shows once

Used by:
  - orchestrator.py (final step before Claude API call)
"""

from __future__ import annotations

from typing import Any


# Maximum characters of context to keep things tight (~5000 tokens of context)
MAX_CONTEXT_CHARS = 12_000


def assemble_context(
    results: dict[str, Any],
    user_query: str,
) -> dict[str, Any]:
    """
    Take a dict of retrieval results from multiple modules and produce:
      - context_text: a Markdown-formatted string for Claude's prompt
      - sources: a list of source identifiers for citation in the UI

    `results` shape (any keys can be missing — assemble what's there):
      {
        "vector":  [list of chunk dicts from vector_search],
        "mongo":   {dict of mongo_lookup output},
        "graph":   {dict of graph_search output},
        "predict": {dict of predict_quote output},
      }
    """
    sections: list[str] = []
    sources: list[str] = []

    # 1. Pricing prediction (highest priority - put first if present)
    if "predict" in results and results["predict"]:
        predict_section, predict_sources = _format_predict(results["predict"])
        if predict_section:
            sections.append(predict_section)
            sources.extend(predict_sources)

    # 2. Customer / invoice / PO direct lookups
    if "mongo" in results and results["mongo"]:
        mongo_section, mongo_sources = _format_mongo(results["mongo"])
        if mongo_section:
            sections.append(mongo_section)
            sources.extend(mongo_sources)

    # 3. Graph relationships
    if "graph" in results and results["graph"]:
        graph_section, graph_sources = _format_graph(results["graph"])
        if graph_section:
            sections.append(graph_section)
            sources.extend(graph_sources)

    # 4. Vector search hits — collected last as broad context
    if "vector" in results and results["vector"]:
        seen = set(sources)  # dedupe against earlier sources
        vector_section, vector_sources = _format_vector(results["vector"], seen)
        if vector_section:
            sections.append(vector_section)
            sources.extend(vector_sources)

    if not sections:
        return {
            "context_text": "(No relevant context found in the data stores.)",
            "sources":      [],
        }

    context_text = "\n\n".join(sections)

    # Hard cap to protect against runaway context
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[... truncated ...]"

    return {
        "context_text": context_text,
        "sources":      sources,
    }


# ---------------------------------------------------------------------------
# Per-source formatters
# ---------------------------------------------------------------------------

def _format_predict(predict: dict) -> tuple[str, list[str]]:
    """Format a predict_quote result as a Markdown block."""
    if not predict.get("job_type"):
        return "", []

    sources = []
    lines = ["## PRICING PREDICTION"]
    lines.append(f"**Job type:** {predict['job_type']}")
    lines.append(f"**Confidence:** {predict.get('confidence', 'unknown')}")
    lines.append("")

    # Materials section
    mat = predict.get("materials", {})
    if mat.get("items"):
        lines.append(f"**Materials estimate:** €{mat.get('subtotal', 0):.2f} ex VAT")
        for item in mat["items"][:8]:  # Cap at 8 items for compactness
            lines.append(
                f"  - {item['item_name']:30}  "
                f"qty {item['quantity']} × €{item['unit_price']:.2f} = €{item['line_total']:.2f}"
            )
        if len(mat["items"]) > 8:
            lines.append(f"  - ...and {len(mat['items']) - 8} more items")
        sources.append(f"Neo4j USES_ITEM recipe ({mat['n_recipe_items']} items)")

    # Labour section
    lab = predict.get("labour", {})
    if lab.get("n_invoices", 0) > 0:
        lines.append("")
        lines.append(
            f"**Labour estimate:** €{lab['median_eur']:.2f} (median), "
            f"range €{lab['min_eur']:.2f} - €{lab['max_eur']:.2f} "
            f"(from {lab['n_invoices']} past invoices)"
        )
        sources.append(f"MongoDB invoices ({lab['n_invoices']} records)")

    # Totals
    totals = predict.get("totals", {})
    lines.append("")
    lines.append(f"**Total estimate:** €{totals.get('total_inc_vat', 0):.2f} inc VAT "
                 f"(subtotal €{totals.get('subtotal_ex_vat', 0):.2f} + VAT €{totals.get('vat_23pct', 0):.2f})")

    # Benchmark
    bench = predict.get("benchmark", {})
    if bench.get("n_pos", 0) > 0:
        lines.append("")
        lines.append(
            f"**Benchmark:** average of {bench['n_pos']} similar past PO(s) was "
            f"€{bench.get('avg_total', 0):.2f} inc VAT"
        )
        for po in bench.get("po_numbers", [])[:3]:
            sources.append(f"PO {po}")

    return "\n".join(lines), sources


def _format_mongo(mongo: dict) -> tuple[str, list[str]]:
    """Format mongo_lookup output as Markdown."""
    sources = []
    lines = ["## DIRECT LOOKUPS"]
    has_content = False

    if "ambiguous_customers" in mongo:
        lines.append(f"**Multiple customers match this name:**")
        for c in mongo["ambiguous_customers"]:
            lines.append(f"  - {c['customer_id']}: {c.get('first_name')} {c.get('last_name')} <{c.get('email')}>")
        has_content = True

    if "customer" in mongo and mongo["customer"]:
        c = mongo["customer"]
        lines.append(
            f"**Customer:** {c['customer_id']} - "
            f"{c.get('first_name', '')} {c.get('last_name', '')} "
            f"<{c.get('email', '')}> ({c.get('preferred_trade', '?')})"
        )
        sources.append(f"Customer {c['customer_id']}")
        has_content = True

    if "invoices" in mongo and mongo["invoices"]:
        lines.append(f"**Past invoices ({len(mongo['invoices'])}):**")
        for inv in mongo["invoices"][:5]:
            lines.append(
                f"  - {inv.get('invoice_id')} on {inv.get('invoice_date', '?')}: "
                f"{inv.get('job_type_id', '?')}, total €{inv.get('total_inc_vat', 0):.2f}"
            )
            sources.append(f"Invoice {inv.get('invoice_id')}")
        has_content = True

    if "pos" in mongo and mongo["pos"]:
        lines.append(f"**Active POs ({len(mongo['pos'])}):**")
        for po in mongo["pos"]:
            lines.append(
                f"  - {po.get('po_number')}: {po.get('job_type', '?')}, "
                f"total €{po.get('total_inc_vat', 0):.2f} - {po.get('po_status', '?')}"
            )
            sources.append(f"PO {po.get('po_number')}")
        has_content = True

    if "po" in mongo and mongo["po"]:
        po = mongo["po"]
        lines.append(f"**PO {po.get('po_number')}:**")
        lines.append(f"  - Customer: {po.get('cust_name')} ({po.get('matched_customer_id')})")
        lines.append(f"  - Job: {po.get('job_type')}")
        lines.append(f"  - Status: {po.get('po_status')}")
        lines.append(f"  - Total: €{po.get('total_inc_vat', 0):.2f} inc VAT")
        sources.append(f"PO {po.get('po_number')}")
        has_content = True

    if "invoice" in mongo and mongo["invoice"]:
        inv = mongo["invoice"]
        lines.append(f"**Invoice {inv.get('invoice_id')}:**")
        lines.append(f"  - Date: {inv.get('invoice_date')}")
        lines.append(f"  - Job: {inv.get('job_type_id')}")
        lines.append(f"  - Total: €{inv.get('total_inc_vat', 0):.2f} inc VAT")
        sources.append(f"Invoice {inv.get('invoice_id')}")
        has_content = True

    return ("\n".join(lines), sources) if has_content else ("", [])


def _format_graph(graph: dict) -> tuple[str, list[str]]:
    """Format graph_search output as Markdown."""
    sources = []
    lines = ["## GRAPH RELATIONSHIPS"]
    has_content = False

    if "items" in graph and graph["items"]:
        lines.append(f"**Materials typically used (from graph):**")
        for item in graph["items"][:8]:
            lines.append(
                f"  - {item.get('item_name')} (qty {item.get('typical_quantity', '?')}, "
                f"€{item.get('unit_price', 0):.2f}/{item.get('unit', 'unit')})"
            )
        sources.append("Neo4j USES_ITEM graph")
        has_content = True

    if "items_by_name" in graph and graph["items_by_name"]:
        lines.append(f"**Materials typically used:**")
        for item in graph["items_by_name"][:8]:
            lines.append(f"  - {item.get('item_name')}")
        sources.append("Neo4j USES_ITEM graph")
        has_content = True

    if "similar_customers" in graph and graph["similar_customers"]:
        lines.append(f"**Similar customers (had this job type before):**")
        for c in graph["similar_customers"][:5]:
            lines.append(
                f"  - {c.get('customer_id')}: {c.get('first_name')} {c.get('last_name')} "
                f"({c.get('interaction_count')} interactions)"
            )
        sources.append("Neo4j similar_customers traversal")
        has_content = True

    if "customer_history" in graph and graph["customer_history"]:
        h = graph["customer_history"]
        if h.get("invoice_ids") or h.get("po_numbers") or h.get("email_ids"):
            lines.append(f"**Customer graph context for {h.get('customer_id')}:**")
            if h.get("invoice_ids"):
                lines.append(f"  - Past invoices: {', '.join(h['invoice_ids'][:5])}")
            if h.get("po_numbers"):
                lines.append(f"  - Active POs: {', '.join(h['po_numbers'])}")
            if h.get("preferred_trades"):
                lines.append(f"  - Preferred trades: {', '.join(h['preferred_trades'])}")
            sources.append(f"Neo4j customer history for {h.get('customer_id')}")
            has_content = True

    if "po_context" in graph and graph["po_context"]:
        po = graph["po_context"]
        lines.append(f"**PO graph context:**")
        lines.append(f"  - Customer: {po.get('customer_name')} ({po.get('customer_id')})")
        lines.append(f"  - Job: {po.get('job_name')}")
        if po.get("items"):
            lines.append(f"  - Items: {', '.join(po['items'][:5])}")
        sources.append(f"Neo4j PO graph for {po.get('po_number')}")
        has_content = True

    return ("\n".join(lines), sources) if has_content else ("", [])


def _format_vector(vector: list[dict], seen_sources: set) -> tuple[str, list[str]]:
    """Format vector_search results as Markdown — dedupe against earlier sources."""
    if not vector:
        return "", []

    sources = []
    lines = ["## SEMANTICALLY SIMILAR CONTENT"]

    for r in vector[:6]:  # Cap at top 6 for compactness
        chunk_id = r.get("chunk_id", "?")
        score = r.get("score", 0)
        text = (r.get("text", "") or "")[:200]  # Truncate long chunks
        source_coll = r.get("source_collection", "?")

        source_label = f"{chunk_id} ({source_coll}, similarity {score:.2f})"
        if source_label in seen_sources:
            continue

        lines.append(f"- **[{score:.2f}]** {source_coll}/{chunk_id}")
        lines.append(f"  {text}{'...' if len(r.get('text', '')) > 200 else ''}")
        sources.append(source_label)

    return ("\n".join(lines), sources) if len(lines) > 1 else ("", [])


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Mock-up to verify the formatter works in isolation
    fake_results = {
        "predict": {
            "job_type": "Boiler installation",
            "confidence": "high",
            "materials": {
                "items": [
                    {"item_name": "Gas boiler 24kW", "quantity": 1, "unit_price": 1100.0, "line_total": 1100.0},
                    {"item_name": "Boiler flue kit", "quantity": 1, "unit_price": 85.0, "line_total": 85.0},
                ],
                "subtotal": 1185.0,
                "n_recipe_items": 2,
            },
            "labour": {"median_eur": 450.0, "min_eur": 380.0, "max_eur": 520.0, "n_invoices": 12},
            "totals": {"subtotal_ex_vat": 1635.0, "vat_23pct": 376.05, "total_inc_vat": 2011.05},
            "benchmark": {"n_pos": 3, "avg_total": 2050.0, "po_numbers": ["PO-2026-P0042", "PO-2026-P0058", "PO-2026-P0063"]},
        },
    }
    out = assemble_context(fake_results, "How much for boiler installation?")
    print(out["context_text"])
    print("\n\nSOURCES:", out["sources"])
