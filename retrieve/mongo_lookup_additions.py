"""
Add these functions to your retrieve/mongo_lookup.py file
(at the bottom, before the if __name__ == "__main__": block).

These provide "list/summary" views for open-ended questions like
"show me all POs" that the previous version couldn't handle.
"""

# ---------------------------------------------------------------------------
# List/summary functions for open-ended queries
# ---------------------------------------------------------------------------

def list_all_pos(limit: int = 50) -> list[dict]:
    """All POs with summary fields. Used for 'show me POs' queries."""
    return list(_get_db().pos.find(
        {},
        {
            "_id": 0,
            "po_number": 1,
            "cust_name": 1,
            "matched_customer_id": 1,
            "trade_name": 1,
            "job_type": 1,
            "po_status": 1,
            "total_inc_vat": 1,
            "po_date": 1,
        }
    ).limit(limit))


def list_all_customers(limit: int = 20) -> list[dict]:
    """Sample customers with summary fields."""
    return list(_get_db().customers.find(
        {},
        {
            "_id": 0,
            "customer_id": 1,
            "first_name": 1,
            "last_name": 1,
            "email": 1,
            "preferred_trade": 1,
        }
    ).limit(limit))


def list_all_job_types(trade: str | None = None) -> list[dict]:
    """All job types, optionally filtered by trade."""
    query = {"trade": trade} if trade else {}
    return list(_get_db().job_types.find(
        query,
        {
            "_id": 0,
            "job_type_id": 1,
            "job_name": 1,
            "trade": 1,
        }
    ).sort("trade"))


def list_all_invoices(limit: int = 20) -> list[dict]:
    """Recent invoices with summary fields."""
    return list(_get_db().invoices.find(
        {},
        {
            "_id": 0,
            "invoice_id": 1,
            "customer_id": 1,
            "job_type_id": 1,
            "invoice_date": 1,
            "total_inc_vat": 1,
        }
    ).sort("invoice_date", -1).limit(limit))


def list_recent_emails(limit: int = 10) -> list[dict]:
    """Recent emails with summary."""
    return list(_get_db().emails.find(
        {},
        {
            "_id": 0,
            "email_id": 1,
            "from_email": 1,
            "subject": 1,
            "received_at": 1,
            "extracted.trade_needed": 1,
            "extracted.urgency": 1,
        }
    ).limit(limit))


def get_collection_summary() -> dict:
    """Counts of each collection - useful for 'what data do you have' questions."""
    db = _get_db()
    return {
        "customers":     db.customers.count_documents({}),
        "emails":        db.emails.count_documents({}),
        "invoices":      db.invoices.count_documents({}),
        "invoice_items": db.invoice_items.count_documents({}),
        "items":         db.items.count_documents({}),
        "job_types":     db.job_types.count_documents({}),
        "pos":           db.pos.count_documents({}),
        "chunks":        db.chunks.count_documents({}),
    }
