"""
orchestrator.py
================
Module 7 of Phase 5 retrieval orchestration — THE GLUE.

Single entry point function `answer_query(user_message)` that ties all
six Phase 5 modules together. The Streamlit UI (Phase 6) only needs to
call this one function — orchestrator handles routing, retrieval,
context assembly, and the final Claude API call.

Flow:
  1. Router classifies the query
  2. Selected retrievers run (vector / mongo / graph / predict)
  3. Context assembler merges results
  4. Claude generates a grounded answer
  5. Return answer + sources + routing trace for the UI

Used by:
  - app/streamlit_app.py (the chatbot UI)
  - test scripts and notebooks for batch evaluation
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
- If the context does not contain the answer, say so honestly.
- Quote specific numbers (€ amounts, customer ids, PO numbers, dates) when
  they appear in the context.
- For pricing questions, present the prediction clearly and mention the
  confidence level.
- Keep responses concise and professional. No unnecessary preamble."""


@lru_cache(maxsize=1)
def _get_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_query(user_message: str, verbose: bool = False) -> dict[str, Any]:
    """
    The single entry point for the chatbot. Orchestrates the entire
    retrieve-then-answer flow.

    Args:
        user_message:  The user's natural-language query.
        verbose:       If True, includes raw retrieval results in the output
                       (useful for debugging and for the UI's "Sources" panel).

    Returns:
        dict with keys:
            - answer:       Claude's natural-language response (str)
            - sources:      List of source identifiers for citation
            - routing:      How the query was classified (intent, paths, method)
            - latency_ms:   Total time for the call
            - raw_results:  Raw retriever outputs (only if verbose=True)
    """
    t_start = time.time()

    # ===== Step 1: Router decides which retrievers to invoke =====
    routing = router.classify_query(user_message)

    # ===== Step 2: Run the selected retrievers =====
    results: dict[str, Any] = {}

    paths = routing.get("paths", [])
    params = routing.get("params", {})

    if "vector" in paths:
        results["vector"] = vector_search.vector_search(user_message, top_k=5)

    if "mongo" in paths:
        results["mongo"] = mongo_lookup.lookup_by_intent(params)

    if "graph" in paths:
        results["graph"] = graph_search.graph_traversal(params)

    if "predict_quote" in paths:
        job_name = params.get("job_name", "")
        if job_name:
            results["predict"] = predict_quote.predict_quote(
                job_name,
                job_type_id=params.get("job_type_id"),
            )

    # ===== Step 3: Assemble context for Claude =====
    context = context_assembler.assemble_context(results, user_query=user_message)

    # ===== Step 4: Call Claude to generate the grounded answer =====
    client = _get_client()
    response = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1500,
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


# ---------------------------------------------------------------------------
# CLI for quick end-to-end testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "How much for a boiler installation?",
        "Show me PO-2026-P0042",
        "What customers have had work done by carpenters?",
    ]
    for q in queries:
        print(f"\n{'=' * 70}")
        print(f"QUESTION: {q}")
        print('=' * 70)
        out = answer_query(q, verbose=False)
        print(f"\n{out['answer']}")
        print(f"\n--- routing: {out['routing']['intent']} via {out['routing']['method']} ---")
        print(f"--- sources ({len(out['sources'])}): {', '.join(out['sources'][:5])} ---")
        print(f"--- {out['latency_ms']}ms / {out['tokens_in']} tokens in / {out['tokens_out']} tokens out ---")
