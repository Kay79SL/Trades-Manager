"""Test how much faster the second query is when model is already loaded."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import time
from retrieve.orchestrator import answer_query 

print("Query 1 (cold start)...")
t1 = time.time()
result1 = answer_query("Show me PO-2026-P0042")
elapsed1 = time.time() - t1
print(f"  → {elapsed1:.1f}s")
print(f"  Answer: {result1['answer'][:100]}...")

print("\nQuery 2 (warm)...")
t2 = time.time()
result2 = answer_query("How much for boiler installation?")
elapsed2 = time.time() - t2
print(f"  → {elapsed2:.1f}s")
print(f"  Answer: {result2['answer'][:100]}...")

print("\nQuery 3 (warm)...")
t3 = time.time()
result3 = answer_query("Has Gerard Walsh had work done before?")
elapsed3 = time.time() - t3
print(f"  → {elapsed3:.1f}s")
print(f"  Answer: {result3['answer'][:100]}...")

print(f"\nSummary:")
print(f"  Query 1 (cold): {elapsed1:.1f}s  ← includes model load")
print(f"  Query 2 (warm): {elapsed2:.1f}s  ← real production speed")
print(f"  Query 3 (warm): {elapsed3:.1f}s")