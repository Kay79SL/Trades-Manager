"""
vector_search.py
=================
Module 1 of Phase 5 retrieval orchestration.

Reusable wrapper around Atlas Vector Search. Encodes a query string into a
384-dim vector using sentence-transformers, then runs a $vectorSearch
aggregation against the chunks collection. Returns top-k semantically
similar chunks ranked by cosine similarity.

The first call loads the model into memory (~80MB, takes 2-3 sec). All
subsequent calls reuse the loaded model — encoding is then milliseconds.

Used by:
  - orchestrator.py (when router picks "vector" path)
  - context_assembler.py (for "find similar past content" queries)
  - predict_quote.py (to find similar past invoices/POs by job description)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX = "vector_index"
DEFAULT_COLLECTION = "chunks"


# ---------------------------------------------------------------------------
# Connection cache (load model + DB connection once, reuse forever)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load embedding model once, cache for all subsequent calls."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_db():
    """Get MongoDB connection once, cache."""
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    return MongoClient(mongo_uri)[db_name]


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def vector_search(
    query: str,
    top_k: int = 5,
    num_candidates: int = 50,
    source_filter: str | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """
    Find chunks semantically similar to the query.

    Args:
        query:          The natural-language query string.
        top_k:          How many results to return (default 5).
        num_candidates: How many candidates Atlas considers before ranking
                        (more = more accurate but slower; 50 is a good default).
        source_filter:  Optional filter to one source collection
                        (e.g. "emails", "job_types", "items", "pos").
        min_score:      Optional minimum cosine similarity (0-1) to include.

    Returns:
        List of dicts: [{"chunk_id", "source_collection", "source_id",
                         "text", "metadata", "score"}, ...]
    """
    if not query or not query.strip():
        return []

    db = _get_db()
    model = _get_model()

    # Encode query into 384-dim vector (must match what embed_chunks.py used)
    query_vec = model.encode(query, normalize_embeddings=True).tolist()

    # Build the aggregation pipeline
    pipeline: list[dict] = [
        {
            "$vectorSearch": {
                "index":         DEFAULT_INDEX,
                "path":          "embedding",
                "queryVector":   query_vec,
                "numCandidates": num_candidates,
                "limit":         top_k,
            }
        },
        {
            "$project": {
                "_id":               0,
                "chunk_id":          1,
                "source_collection": 1,
                "source_id":         1,
                "text":              1,
                "metadata":          1,
                "score":             {"$meta": "vectorSearchScore"},
            }
        }
    ]

    # Optional source filter (post-vector-search filter)
    if source_filter:
        pipeline.append({"$match": {"source_collection": source_filter}})

    # Optional minimum-score filter
    if min_score is not None:
        pipeline.append({"$match": {"score": {"$gte": min_score}}})

    results = list(db[DEFAULT_COLLECTION].aggregate(pipeline))
    return results


def vector_search_multi_source(
    query: str,
    top_k_per_source: int = 3,
    sources: tuple[str, ...] = ("emails", "job_types", "items", "pos"),
) -> dict[str, list[dict]]:
    """
    Run a vector search across multiple sources separately and return
    grouped results. Useful when you want diverse evidence from each
    source collection rather than a single ranked list.
    """
    grouped: dict[str, list[dict]] = {}
    for source in sources:
        grouped[source] = vector_search(
            query, top_k=top_k_per_source, source_filter=source
        )
    return grouped


# ---------------------------------------------------------------------------
# CLI for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "boiler installation quote"
    print(f"Query: {q}\n")
    for r in vector_search(q, top_k=5):
        print(f"  [{r['score']:.3f}] {r['chunk_id']:25} ({r['source_collection']})")
        print(f"          {r['text'][:90]}...")
