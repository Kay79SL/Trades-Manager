"""Find emails where extracted is null and unset the field so they re-extract."""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB", "trades_quotes")]

broken_count = db.emails.count_documents({"extracted": None})
print(f"Found {broken_count} email(s) with extracted=null")

if broken_count == 0:
    print("Nothing to fix.")
    exit(0)

print(f"\nThis will UNSET 'extracted' on {broken_count} document(s).")
print("Next run of extract_entities.py will then re-process them.")
answer = input("Proceed? (yes/no): ")
if answer.lower() != "yes":
    print("Aborted. No changes made.")
    exit(0)

result = db.emails.update_many(
    {"extracted": None},
    {"$unset": {"extracted": ""}},
)
print(f"\nUnset extracted on {result.modified_count} email(s).")
print("Next step: python ingest\\extract_entities.py")