"""
router.py
==========
Module 6 of Phase 5 retrieval orchestration.

Classifies the user's query into an intent and decides which retrieval
path(s) to invoke. Uses a hybrid strategy:
  1. Pattern matching for high-confidence cases (PO ids, invoice ids,
     "how much" pricing patterns, etc.)
  2. LLM fallback for ambiguous cases (sends a small classification
     prompt to Claude Haiku — cheap and accurate)

Output is a dict the orchestrator uses to dispatch retrievers:
  {
    "paths":     ["mongo", "graph", "vector", "predict_quote"],
    "params":    {parameters extracted from the query},
    "intent":    "pricing_query" | "customer_lookup" | "po_lookup" | "open_question",
    "method":    "pattern" | "llm",
  }

Used by:
  - orchestrator.py (the very first call in the answer flow)
"""

from __future__ import annotations

import os
import re
import json
from functools import lru_cache

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

ROUTER_MODEL = "claude-haiku-4-5-20251001"


@lru_cache(maxsize=1)
def _get_client():
    return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Pattern matchers — high-confidence cases handled without LLM
# ---------------------------------------------------------------------------

PO_ID_RE = re.compile(r"\b(PO-\d{4}-[A-Z]\d{4})\b", re.IGNORECASE)
INVOICE_ID_RE = re.compile(r"\b(INV-\d{4}-\d{4})\b", re.IGNORECASE)
CUSTOMER_ID_RE = re.compile(r"\b(cust_\d{4})\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

# Pricing intent triggers
PRICING_TRIGGERS = re.compile(
    r"\b(how much|cost|price|quote|estimate|charge for|going rate|typical)\b",
    re.IGNORECASE,
)

# Job type keywords (basic dictionary — could be expanded from MongoDB)
JOB_KEYWORDS = {
    "boiler installation":       "Boiler installation",
    "boiler service":            "Boiler service",
    "radiator replacement":      "Radiator replacement",
    "bathroom installation":     "Bathroom suite installation",
    "kitchen installation":      "Kitchen installation",
    "floor sanding":             "Floor sanding",
    "wooden floor laying":       "Wooden floor laying",
    "rewiring":                  "Full house rewire",
    "light fitting":             "Light fitting installation",
    "extra socket":              "Extra socket installation",
}


def _try_pattern_match(query: str) -> dict | None:
    """Returns a routing decision if a high-confidence pattern matches, else None."""
    q = query.strip()

    # Direct ID lookups
    if m := PO_ID_RE.search(q):
        return {
            "paths":  ["mongo", "graph"],
            "params": {"po_number": m.group(1).upper()},
            "intent": "po_lookup",
            "method": "pattern",
        }

    if m := INVOICE_ID_RE.search(q):
        return {
            "paths":  ["mongo"],
            "params": {"invoice_id": m.group(1).upper()},
            "intent": "invoice_lookup",
            "method": "pattern",
        }

    if m := CUSTOMER_ID_RE.search(q):
        return {
            "paths":  ["mongo", "graph"],
            "params": {"customer_id": m.group(1).lower()},
            "intent": "customer_lookup",
            "method": "pattern",
        }

    if m := EMAIL_RE.search(q):
        return {
            "paths":  ["mongo"],
            "params": {"customer_email": m.group(1).lower()},
            "intent": "customer_lookup",
            "method": "pattern",
        }

    # Pricing intent + recognised job type → predict_quote
    if PRICING_TRIGGERS.search(q):
        for keyword, job_name in JOB_KEYWORDS.items():
            if keyword in q.lower():
                return {
                    "paths":  ["predict_quote"],
                    "params": {"job_name": job_name},
                    "intent": "pricing_query",
                    "method": "pattern",
                }

    return None


# ---------------------------------------------------------------------------
# LLM-based classification fallback
# ---------------------------------------------------------------------------

LLM_CLASSIFIER_SYSTEM = """You classify user queries about an Irish trades business
chatbot. Return ONLY a single valid JSON object — no preamble, no markdown.

The JSON must have these fields:
{
  "intent":    "pricing_query" | "customer_lookup" | "po_lookup" |
               "invoice_lookup" | "general_question" | "open_search",
  "paths":     a list including any of: "vector", "mongo", "graph", "predict_quote",
  "params":    a dict with extracted entities (job_name, customer_name, etc.)
}

Rules:
- Pricing questions about a specific job → "predict_quote" path
- Questions about a specific customer → "mongo" + "graph"
- Questions about relationships ("who/which customers") → "graph"
- General "find" or "search" questions → "vector" + optionally "graph"
- If unsure or open-ended → ["vector", "graph"] for broad recall
"""


def _llm_classify(query: str) -> dict:
    """Use Claude Haiku to classify ambiguous queries."""
    client = _get_client()
    response = client.messages.create(
        model=ROUTER_MODEL,
        max_tokens=300,
        system=LLM_CLASSIFIER_SYSTEM,
        messages=[{"role": "user", "content": f"Classify this query: {query}"}],
    )
    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?|\n?```$", "", text).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: broad search
        result = {
            "intent": "open_search",
            "paths":  ["vector", "graph"],
            "params": {},
        }
    result["method"] = "llm"
    return result


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def classify_query(query: str) -> dict:
    """
    Decide which retrieval paths to invoke for this query.
    Tries pattern matching first (fast, free), falls back to LLM (cheap, flexible).
    """
    if not query or not query.strip():
        return {
            "paths":  [],
            "params": {},
            "intent": "empty",
            "method": "skip",
        }

    pattern_result = _try_pattern_match(query)
    if pattern_result:
        return pattern_result

    return _llm_classify(query)


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    test_queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "Show me PO-2026-P0042",
        "What's Gerard Walsh's history?",
        "How much for a boiler installation?",
        "My lights keep flickering",
        "Find emails about wooden floors",
        "Has cust_0001 had work done before?",
        "Tell me about gerardwalsh@icloud.com",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = classify_query(q)
        print(f"  Intent: {result['intent']}  ({result['method']})")
        print(f"  Paths:  {result['paths']}")
        print(f"  Params: {result['params']}")
