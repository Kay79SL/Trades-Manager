"""Test semantic search via Atlas Vector Search."""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB", "trades_quotes")]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Test queries — each should find semantically related chunks
# even though no exact words match
queries = [
    "my heating is broken",
    "kitchen lights keep flickering",
    "I need a new bathroom",
    "my floor needs to be sanded",
    "boiler installation quote",
]

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print('=' * 60)

    # Encode the query into a vector (same model as embedded chunks)
    query_vec = model.encode(query, normalize_embeddings=True).tolist()

    # Run the vector search aggregation
    results = list(db.chunks.aggregate([
        {
            "$vectorSearch": {
                "index":         "vector_index",
                "path":          "embedding",
                "queryVector":   query_vec,
                "numCandidates": 50,
                "limit":         5,
            }
        },
        {
            "$project": {
                "_id":               0,
                "chunk_id":          1,
                "source_collection": 1,
                "text":              {"$substr": ["$text", 0, 90]},
                "score":             {"$meta": "vectorSearchScore"},
            }
        }
    ]))

    if not results:
        print("  No results — index may still be building, or chunks empty.")
        continue

    for r in results:
        print(f"  [{r['score']:.3f}] {r['chunk_id']:25} ({r['source_collection']})")
        print(f"          {r['text']}...")