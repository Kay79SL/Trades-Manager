"""Show the literal state of email documents in MongoDB."""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB", "trades_quotes")]

print(f"Total emails: {db.emails.count_documents({})}")
print()

# Look at 5 random emails and show exactly what's there
print("First 5 emails — exact field state:")
print("-" * 60)
for e in db.emails.find({}, {"email_id": 1, "extracted": 1}).limit(5):
    if "extracted" not in e:
        state = "NO 'extracted' field"
    elif e["extracted"] is None:
        state = "extracted is null (None)"
    elif isinstance(e["extracted"], dict):
        keys = list(e["extracted"].keys())
        state = f"extracted is dict with {len(keys)} keys: {keys[:5]}..."
    else:
        state = f"extracted is {type(e['extracted']).__name__}: {e['extracted']!r}"

    print(f"  {e['email_id']:14}  {state}")

# Re-run the count from absolute first principles
print()
print("Counts (re-checked):")
no_field = 0
is_none = 0
is_dict = 0
other = 0
for e in db.emails.find({}, {"extracted": 1}):
    if "extracted" not in e:
        no_field += 1
    elif e["extracted"] is None:
        is_none += 1
    elif isinstance(e["extracted"], dict):
        is_dict += 1
    else:
        other += 1
print(f"  No 'extracted' field:    {no_field}")
print(f"  extracted is None:       {is_none}")
print(f"  extracted is dict:       {is_dict}")
print(f"  extracted is other:      {other}")
print(f"  Total:                   {no_field + is_none + is_dict + other}")