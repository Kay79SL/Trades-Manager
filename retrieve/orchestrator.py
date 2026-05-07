"""
orchestrator.py
================
Module 7 of Phase 5 retrieval orchestration — THE GLUE.

Single entry point function `answer_query(user_message)` that ties all
six Phase 5 modules together. Updated to support list/browse queries.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

from . import router
from . import vector_search
from . import mongo_lookup
from . import graph_search
from . import predict_quote
from . import context_assembler


load_dotenv()

ANSWER_MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = """You are a helpful assistant for an Irish trades business
(plumbing, carpentry, electrical). You answer questions about customers,
quotes, jobs, and pricing.

CRITICAL RULES:
- Use ONLY the context provided below. Do not invent customer names, prices,
  invoice numbers, or any other facts.
- If the context contains a list of records, present them clearly and
  comprehensively. Don't say "I can only see one" if the list has more.
- If the context does not contain the answer, say so honestly.
- Quote specific numbers (€ amounts, customer ids, PO numbers, dates) when
  they appear in the context.
- For pricing questions, present the prediction clearly and mention the
  confidence level.
- Keep responses professional and well-formatted (use bullet points for lists).
- No unnecessary preamble."""


@lru_cache(maxsize=1)
def _get_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _handle_list_query(list_target: str) -> dict:
    """Fetch list/summary data based on what the user wants to see."""
    if list_target == "pos":
        return {
            "type": "pos",
            "title": "All Purchase Orders",
            "items": mongo_lookup.list_all_pos(),
        }
    elif list_target == "customers":
        return {
            "type": "customers",
            "title": "Customers",
            "items": mongo_lookup.list_all_customers(),
        }
    elif list_target == "jobs":
        return {
            "type": "jobs",
            "title": "Job Types",
            "items": mongo_lookup.list_all_job_types(),
        }
    elif list_target == "invoices":
        return {
            "type": "invoices",
            "title": "Recent Invoices",
            "items": mongo_lookup.list_all_invoices(),
        }
    elif list_target == "emails":
        return {
            "type": "emails",
            "title": "Recent Emails",
            "items": mongo_lookup.list_recent_emails(),
        }
    elif list_target == "summary":
        return {
            "type": "summary",
            "title": "Database Overview",
            "items": mongo_lookup.get_collection_summary(),
        }
    return {}


def answer_query(user_message: str, verbose: bool = False) -> dict[str, Any]:
    """The single entry point for the chatbot."""
    t_start = time.time()

    # Step 1: Router decides which retrievers to invoke
    routing = router.classify_query(user_message)

    # Step 2: Run the selected retrievers
    results: dict[str, Any] = {}
    paths = routing.get("paths", [])
    params = routing.get("params", {})

    if "vector" in paths:
        results["vector"] = vector_search.vector_search(user_message, top_k=5)

    if "mongo" in paths:
        results["mongo"] = mongo_lookup.lookup_by_intent(params)

    if "graph" in paths:
        # Enrich params with customer_id resolved by mongo lookup
        # so graph_traversal can call customer_full_history()
        graph_params = dict(params)
        mongo_result = results.get("mongo", {})
        if mongo_result.get("customer", {}).get("customer_id"):
            graph_params["customer_id"] = mongo_result["customer"]["customer_id"]
        results["graph"] = graph_search.graph_traversal(graph_params)

    if "predict_quote" in paths:
        job_name = params.get("job_name", "")
        if job_name:
            results["predict"] = predict_quote.predict_quote(
                job_name,
                job_type_id=params.get("job_type_id"),
            )

    # NEW: list/browse path
    if "list" in paths:
        results["list"] = _handle_list_query(params.get("list_target", "summary"))

    # Step 3: Assemble context
    context = context_assembler.assemble_context(results, user_query=user_message)

    # Step 4: Call Claude
    client = _get_client()
    response = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=2000,  # Increased to handle list responses
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"CONTEXT (use this to answer):\n\n"
                    f"{context['context_text']}\n\n"
                    f"---\n\n"
                    f"USER QUESTION: {user_message}"
                ),
            }
        ],
    )
    answer_text = response.content[0].text

    latency_ms = round((time.time() - t_start) * 1000)

    out = {
        "answer":     answer_text,
        "sources":    context["sources"],
        "routing":    routing,
        "latency_ms": latency_ms,
        "tokens_in":  response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }

    if verbose:
        out["raw_results"] = results
        out["context_text"] = context["context_text"]

    return out


if __name__ == "__main__":
    import sys
    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "How much for a boiler installation?",
        "Show me PO-2026-P0042",
        "show me all POs",
    ]
    for q in queries:
        print(f"\n{'=' * 70}")
        print(f"QUESTION: {q}")
        print('=' * 70)
        out = answer_query(q, verbose=False)
        print(f"\n{out['answer']}")
        print(f"\n--- routing: {out['routing']['intent']} via {out['routing']['method']} ---")
        print(f"--- {out['latency_ms']}ms / {out['tokens_in']} tokens in / {out['tokens_out']} tokens out ---")
