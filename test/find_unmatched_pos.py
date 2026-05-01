"""Find POs that didn't match a customer and show why."""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB", "trades_quotes")]

# Find unmatched POs
unmatched = list(db.pos.find(
    {"matched_customer_id": None},
    {"po_number": 1, "cust_name": 1, "cust_email": 1, "_id": 0}
))

print(f"Found {len(unmatched)} PO(s) without matched customer:\n")
for po in unmatched:
    po_num = po.get("po_number", "?")
    name = po.get("cust_name", "?")
    email = po.get("cust_email", "?")
    print(f"  {po_num}")
    print(f"    Name in PO:  {name}")
    print(f"    Email in PO: {email}")

    # Try to find the customer by name in customers collection
    if name:
        # Split first/last name
        parts = name.strip().split(maxsplit=1)
        if len(parts) == 2:
            first, last = parts
            cust = db.customers.find_one(
                {"first_name": first, "last_name": last},
                {"customer_id": 1, "email": 1, "_id": 0}
            )
            if cust:
                print(f"    --> Customer EXISTS in DB:")
                print(f"        customer_id: {cust['customer_id']}")
                print(f"        Email in DB: {cust['email']}")
                print(f"    --> EMAIL MISMATCH between PO and DB")
            else:
                print(f"    --> No customer with that name in DB")
    print()