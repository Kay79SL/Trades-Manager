"""Sync any customers added via PO to Neo4j (idempotent)."""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from neo4j import GraphDatabase

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB", "trades_quotes")]
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD")),
)

# Find customers added via PO
new_customers = list(db.customers.find({"_source": "added_from_po"}))
print(f"Customers added via PO in MongoDB: {len(new_customers)}")

if not new_customers:
    print("Nothing to sync.")
    exit(0)

with driver.session() as session:
    for c in new_customers:
        cid = c["customer_id"]

        # Check if already in Neo4j
        result = session.run(
            "MATCH (cust:Customer {customer_id: $cid}) RETURN cust",
            cid=cid,
        )
        if result.single():
            print(f"  {cid} already in Neo4j - skipping")
            continue

        # Create Customer node
        session.run(
            """
            MERGE (cust:Customer {customer_id: $cid})
            SET cust.first_name = $first,
                cust.last_name = $last,
                cust.email = $email,
                cust.phone = $phone,
                cust.county = $county,
                cust.eircode = $eircode,
                cust.preferred_trade = $trade,
                cust.first_contact_date = $first_contact
            """,
            cid=cid,
            first=c.get("first_name", ""),
            last=c.get("last_name", ""),
            email=c.get("email", ""),
            phone=c.get("phone", ""),
            county=c.get("county", ""),
            eircode=c.get("eircode", ""),
            trade=c.get("preferred_trade", ""),
            first_contact=c.get("first_contact_date", ""),
        )

        # Create PREFERS edge (Customer -> Trade)
        if c.get("preferred_trade"):
            session.run(
                """
                MATCH (cust:Customer {customer_id: $cid})
                MATCH (t:Trade {name: $trade})
                MERGE (cust)-[:PREFERS]->(t)
                """,
                cid=cid,
                trade=c.get("preferred_trade", ""),
            )

        # Now find the PO and create FOR_CUSTOMER edge
        for po in db.pos.find({"matched_customer_id": cid}):
            po_num = po["po_number"]
            session.run(
                """
                MATCH (po:PO {po_number: $po_num})
                MATCH (cust:Customer {customer_id: $cid})
                MERGE (po)-[:FOR_CUSTOMER]->(cust)
                """,
                po_num=po_num,
                cid=cid,
            )
            print(f"  {cid}: created node + linked PO {po_num}")

driver.close()
print("\nSync complete.")