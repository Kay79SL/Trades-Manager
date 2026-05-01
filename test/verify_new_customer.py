"""Verify a newly-added customer is properly connected across MongoDB and Neo4j."""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from neo4j import GraphDatabase

load_dotenv()
db = MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB", "trades_quotes")]

# Find the new customer (added via PO)
new_customers = list(db.customers.find(
    {"_source": "added_from_po"},
    {"_id": 0}
))

print(f"Customers added via PO: {len(new_customers)}\n")

if not new_customers:
    print("No customers were added via the PO pipeline.")
    print("Either no unmatched POs, or auto-add was disabled.")
    exit(0)

for c in new_customers:
    print("=" * 60)
    print(f"  Customer: {c['customer_id']} - {c['first_name']} {c['last_name']}")
    print("=" * 60)
    print(f"  Email:           {c['email']}")
    print(f"  Phone:           {c.get('phone', '')}")
    print(f"  County:          {c.get('county', '')}")
    print(f"  Eircode:         {c.get('eircode', '')}")
    print(f"  Trade:           {c.get('preferred_trade', '')}")
    print(f"  Added via PO:    {c.get('_added_via_po', '')}")
    print(f"  First contact:   {c.get('first_contact_date', '')}")

    cust_id = c["customer_id"]

    # Check MongoDB: which POs link to this customer?
    linked_pos = list(db.pos.find(
        {"matched_customer_id": cust_id},
        {"po_number": 1, "job_type": 1, "total_inc_vat": 1, "_id": 0}
    ))
    print(f"\n  MongoDB - POs linked to {cust_id}: {len(linked_pos)}")
    for po in linked_pos:
        total = po.get("total_inc_vat", "?")
        print(f"    {po.get('po_number'):20} {po.get('job_type', ''):30} EUR {total}")

    # Check Neo4j: is there a Customer node and is it linked to PO nodes?
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if neo4j_uri and neo4j_password:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            # Customer node exists?
            result = session.run(
                "MATCH (c:Customer {customer_id: $cid}) RETURN c.first_name AS first, c.last_name AS last",
                cid=cust_id,
            )
            record = result.single()
            if record:
                print(f"\n  Neo4j - Customer node exists: {record['first']} {record['last']}")
            else:
                print(f"\n  Neo4j - WARNING: no Customer node for {cust_id}")
                print(f"    --> Need to add Customer to graph (load_neo4j.py only ran on original 120)")

            # PO -> Customer edges?
            result = session.run(
                """
                MATCH (po:PO)-[:FOR_CUSTOMER]->(c:Customer {customer_id: $cid})
                RETURN po.po_number AS po, po.job_type_text AS job
                """,
                cid=cust_id,
            )
            edges = list(result)
            print(f"\n  Neo4j - PO -[:FOR_CUSTOMER]-> {cust_id} edges: {len(edges)}")
            for e in edges:
                print(f"    {e['po']:20} {e['job']}")

        driver.close()
    print()