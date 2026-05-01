"""Quick spot-check of email extractions in MongoDB."""
import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB", "trades_quotes")]

# ---------- Counts ----------
total = db.emails.count_documents({})
has_field = db.emails.count_documents({"extracted": {"$exists": True}})
real_object = db.emails.count_documents({"extracted": {"$type": "object"}})
is_null = db.emails.count_documents({"extracted": None})
no_field = db.emails.count_documents({"extracted": {"$exists": False}})

print("=" * 60)
print("Extraction status")
print("=" * 60)
print(f"  Total emails:                {total}")
print(f"  extracted field present:     {has_field}")
print(f"  extracted is real object:    {real_object}   <-- this is what we want")
print(f"  extracted is null (broken):  {is_null}")
print(f"  no extracted field:          {no_field}")

if real_object < total:
    print(f"\n  WARNING: {total - real_object} email(s) need re-extraction.")
    print(f"  Run: python test\\fix_null_extractions.py")
    print(f"  Then: python ingest\\extract_entities.py")

# ---------- Sample 3 healthy extractions ----------
print()
print("=" * 60)
print("Sample extractions (3 random, only healthy ones):")
print("=" * 60)

samples = list(db.emails.aggregate([
    {"$match": {"extracted": {"$type": "object"}}},
    {"$sample": {"size": 3}},
]))

if not samples:
    print("\n  No healthy extractions found.")
    print("  All extracted fields are null or missing.")
else:
    for e in samples:
        print(f"\n--- {e['email_id']} ---")
        print(f"From:    {e.get('from_email', '')}")
        print(f"Subject: {(e.get('subject') or '')[:80]}")
        print("Extracted:")
        extr = e.get("extracted") or {}
        for k, v in extr.items():
            if k.startswith("_"):  # skip provenance fields
                continue
            # Truncate long lists/strings for readability
            v_str = str(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "..."
            print(f"  {k:25} {v_str}")