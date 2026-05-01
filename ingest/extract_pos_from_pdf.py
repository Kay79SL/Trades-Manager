"""
extract_pos_from_pdf.py
========================
Phase 3B: Purchase Order ingestion via PDF + AI.

This is the PRIMARY (and only) PO ingestion path for DACARag. It mirrors the
realistic production scenario: customers send PDF purchase orders, and the
chatbot must extract structured data from them automatically.

Pipeline:
  1. Read each PDF from data/purchase_orders/
  2. Extract text content with pypdf (fast, no OCR needed for digital PDFs)
  3. Send the text to Claude with a structured-extraction prompt
  4. Parse the JSON response
  5. Match cust_email back to a customer_id in the customers collection.
     If no match found, AUTO-ADD a new customer record from the PO data
     (with provenance markers) so referential integrity is maintained.
  6. Save to MongoDB `pos` collection — primary PO storage

Customer matching strategy (in order):
  a) Email match (case-insensitive, whitespace-stripped) on existing customers
  b) First-name + last-name exact match on existing customers
  c) If neither matches: create new customer record with PO data,
     tagged with _source='added_from_po' and _added_via_po=<po_number>

Within-batch deduplication: customers added during this run are visible to
subsequent POs (database is re-queried each time), so multiple POs from
the same new customer all link to a single customer_id.

After this script runs, downstream pipeline steps work unchanged:
  - load_po_pdfs.py     — copies PDF binaries into GridFS bucket po_files
  - load_pos_to_neo4j.py — adds PO nodes to Neo4j with FOR_CUSTOMER, FOR_JOB, CONTAINS_ITEM edges

Cost: ~€0.05 per PO using claude-haiku-4-5. All 15 POs ≈ €0.75.

Usage:
    python ingest\\extract_pos_from_pdf.py --dry-run   # show prompt, no API calls
    python ingest\\extract_pos_from_pdf.py --limit 3   # test on 3 PDFs first
    python ingest\\extract_pos_from_pdf.py             # full run on all 15 PDFs
    python ingest\\extract_pos_from_pdf.py --force     # re-extract all (overwrite)
    python ingest\\extract_pos_from_pdf.py --no-auto-add   # don't auto-add unknown customers
    python ingest\\extract_pos_from_pdf.py --pdf-dir PATH  # custom folder

Dependencies:
    python -m pip install pypdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Conditional imports - tell user what to install if missing
try:
    import pypdf
except ImportError:
    print("ERROR: missing dependency 'pypdf'.")
    print("Install with: python -m pip install pypdf")
    sys.exit(1)

from anthropic import Anthropic
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PDF_DIR = Path("data") / "purchase_orders"
TARGET_COLLECTION = "pos"   # main PO collection (PDF + AI is the only ingestion path)
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 2000   # POs are bigger than emails — more line items, more text

PO_NUMBER_RE = re.compile(r"PO-\d{4}-[A-Z]\d{4}", re.IGNORECASE)


SYSTEM_PROMPT_PDF_PO = """You are a purchase order extraction assistant for a chatbot serving Irish trades.
You read text extracted from PDF purchase orders and produce a structured JSON representation
suitable for downstream storage in MongoDB.

You always respond with a single valid JSON object — no preamble, no commentary, no markdown
code fences. The JSON must conform exactly to the schema provided in the user message.

Be precise:
  - Currency values must be extracted as floats (no euro symbol, no thousands separators).
    Example: "€1,712.60" must become 1712.60.
  - For line_items, capture every line shown in the PDF. Each line MUST include
    item_id, description, quantity, unit, unit_price, and line_total.
  - Dates must be in YYYY-MM-DD format if discernible, otherwise the literal string shown.
  - If a field is not present in the PDF, set it to null. Don't fabricate.
  - For job_type, use the literal text from the PDF's "Job type" field (do NOT try to
    map it to a canonical job_type_id — that's done in a later step by load_pos_to_neo4j.py).
"""


# ---------------------------------------------------------------------------
# PDF reading
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file. Works for digital PDFs (which yours are
    since they were generated from Excel). Would need OCR for scanned PDFs.
    """
    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def detect_po_number(filename: str, pdf_text: str) -> str:
    """
    Try filename first (most reliable), fall back to scanning PDF text.
    """
    match = PO_NUMBER_RE.search(filename)
    if match:
        return match.group(0).upper()
    match = PO_NUMBER_RE.search(pdf_text)
    if match:
        return match.group(0).upper()
    return Path(filename).stem


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def build_prompt(pdf_text: str, po_number_hint: str) -> str:
    """Build the user prompt for one PO."""
    return f"""Extract a structured purchase order from this PDF text.

PO_NUMBER (from filename, treat as authoritative if present): {po_number_hint}

PDF TEXT (extracted by pypdf — formatting may be slightly off):
\"\"\"
{pdf_text}
\"\"\"

REQUIRED OUTPUT SCHEMA (respond with this JSON only):
{{
  "po_number":           "<PO number e.g. PO-2026-P0042>",
  "po_date":             "<YYYY-MM-DD or null>",
  "trade_name":          "<trade business name>",
  "trade_addr":          "<trade address line 1>",
  "trade_city":          "<trade town/city>",
  "trade_eircode":       "<eircode or null>",
  "trade_email":         "<trade email>",
  "trade_phone":         "<trade phone>",
  "trade_vat":           "<trade VAT number or null>",
  "cust_name":           "<customer full name>",
  "cust_addr":           "<customer address>",
  "cust_city":           "<customer town/city>",
  "cust_eircode":        "<eircode or null>",
  "cust_email":          "<customer email>",
  "cust_phone":          "<customer phone>",
  "po_status":           "<status text or null>",
  "job_type":            "<job type as written, e.g. 'Bathroom suite installation'>",
  "job_desc":            "<job description>",
  "delivery_date":       "<YYYY-MM-DD or null>",
  "payment_terms":       "<payment terms>",
  "line_items": [
    {{
      "item_id":            "<e.g. it_pl_009>",
      "description":        "<item description>",
      "quantity":           <number>,
      "unit":               "<e.g. unit, metre, pair>",
      "unit_price_ex_vat":  <number>,
      "line_total_ex_vat":  <number>
    }}
  ],
  "materials_subtotal":  <number>,
  "labour_cost":         <number>,
  "subtotal_ex_vat":     <number>,
  "vat_23pct":           <number>,
  "total_inc_vat":       <number>,
  "extraction_confidence": "high" | "medium" | "low",
  "extraction_notes":    "<any concerns or ambiguities the parser noticed>"
}}
"""


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse Claude's response, tolerating optional markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def extract_one(client: Anthropic, system_prompt: str, user_prompt: str) -> dict:
    """Call Claude and parse the response as JSON."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return parse_json_response(response.content[0].text)


# ---------------------------------------------------------------------------
# Customer matching with auto-add
# ---------------------------------------------------------------------------

def normalize_email(email: str | None) -> str:
    """Lowercase and strip whitespace from an email address."""
    if not email:
        return ""
    return email.strip().lower()


def next_customer_id(db) -> str:
    """Generate the next sequential customer_id (cust_NNNN)."""
    last = db.customers.find_one(
        {"customer_id": {"$regex": r"^cust_\d+$"}},
        sort=[("customer_id", -1)],
    )
    if not last:
        return "cust_0001"
    last_num = int(last["customer_id"].split("_")[1])
    return f"cust_{last_num + 1:04d}"


def infer_trade_from_po(po: dict) -> str:
    """Guess preferred_trade from the trade_name in the PO."""
    name = (po.get("trade_name") or "").lower()
    if "plumb" in name or "heating" in name:
        return "plumber"
    if "carpent" in name or "joiner" in name:
        return "carpenter"
    if "electric" in name:
        return "electrician"
    return "plumber"  # safe default


def match_or_add_customer(db, po: dict, auto_add: bool = True) -> tuple[str | None, str]:
    """
    Try to match a PO's customer to an existing customers record. If no match
    found and auto_add=True, create a new customer record from the PO data.

    Returns (customer_id_or_None, action_taken).
    Possible actions: "matched_email", "matched_name", "added_new", "no_match"
    """
    cust_email = normalize_email(po.get("cust_email"))
    cust_name = (po.get("cust_name") or "").strip()

    # Strategy 1: email match (always re-query so within-batch additions are seen)
    if cust_email:
        existing = db.customers.find_one(
            {"email": cust_email},
            {"customer_id": 1, "_id": 0}
        )
        if existing:
            return existing["customer_id"], "matched_email"

    # Strategy 2: name match (first + last)
    if cust_name:
        parts = cust_name.split(maxsplit=1)
        if len(parts) == 2:
            first, last = parts
            existing = db.customers.find_one(
                {"first_name": first, "last_name": last},
                {"customer_id": 1, "_id": 0}
            )
            if existing:
                return existing["customer_id"], "matched_name"

    # Strategy 3: auto-add (if enabled and we have enough data)
    if not auto_add:
        return None, "no_match"

    if not cust_email and not cust_name:
        # Not enough info to create a customer record
        return None, "no_match"

    # Build new customer record from PO data
    name_parts = cust_name.split(maxsplit=1)
    first = name_parts[0] if len(name_parts) >= 1 else ""
    last = name_parts[1] if len(name_parts) >= 2 else ""

    new_id = next_customer_id(db)
    new_customer = {
        "customer_id":          new_id,
        "first_name":           first,
        "last_name":            last,
        "email":                cust_email,
        "phone":                po.get("cust_phone", ""),
        "address_line_1":       po.get("cust_addr", ""),
        "address_line_2":       "",
        "county":               po.get("cust_city", ""),
        "eircode":              po.get("cust_eircode", ""),
        "preferred_trade":      infer_trade_from_po(po),
        "first_contact_date":   time.strftime("%Y-%m-%d"),
        "_source":              "added_from_po",
        "_added_via_po":        po.get("po_number", ""),
        "_added_at":            time.time(),
    }
    db.customers.insert_one(new_customer)

    return new_id, "added_new"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=str, default=str(DEFAULT_PDF_DIR),
                        help=f"Folder of PO PDFs (default: {DEFAULT_PDF_DIR})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max PDFs to process (0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract PDFs even if already in pos collection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the prompt for the first PDF, no API calls")
    parser.add_argument("--no-auto-add", action="store_true",
                        help="Don't auto-add unknown customers (leave matched_customer_id null)")
    args = parser.parse_args()

    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not mongo_uri:
        print("ERROR: MONGO_URI not set in .env"); sys.exit(1)

    db = MongoClient(mongo_uri)[db_name]

    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set in .env"); sys.exit(1)

    # Show current customer count (we re-query the DB each time during extraction
    # so within-batch additions are picked up on subsequent POs)
    initial_customer_count = db.customers.count_documents({})
    print(f"Current customers in DB: {initial_customer_count}")
    if args.no_auto_add:
        print("Auto-add of unknown customers: DISABLED")
    else:
        print("Auto-add of unknown customers: ENABLED")
    print()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"ERROR: PDF folder not found at {pdf_dir}"); sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: no .pdf files in {pdf_dir}"); sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) in {pdf_dir}\n")

    # Skip already-extracted unless --force
    if not args.force:
        existing = {d["po_number"] for d in db[TARGET_COLLECTION].find(
            {}, {"po_number": 1, "_id": 0}
        )}
        new_pdfs = [
            p for p in pdfs
            if detect_po_number(p.name, "") not in existing
        ]
        if len(new_pdfs) < len(pdfs):
            print(f"  {len(pdfs) - len(new_pdfs)} already extracted (skipping). "
                  f"Use --force to redo all.")
        pdfs = new_pdfs

    if args.limit and len(pdfs) > args.limit:
        pdfs = pdfs[:args.limit]
        print(f"  Limiting to {args.limit} PDF(s)")

    if not pdfs:
        print("Nothing to process. Use --force to re-extract.")
        return

    # Dry run
    if args.dry_run:
        sample_pdf = pdfs[0]
        text = extract_pdf_text(sample_pdf)
        po_num = detect_po_number(sample_pdf.name, text)
        prompt = build_prompt(text, po_num)
        print("=" * 60)
        print(f"DRY RUN — would extract from: {sample_pdf.name}")
        print(f"PDF text length: {len(text)} chars")
        print(f"PO number detected: {po_num}")
        print("=" * 60)
        print("\n--- SYSTEM ---\n" + SYSTEM_PROMPT_PDF_PO)
        print("\n--- USER (truncated to 1500 chars) ---")
        print(prompt[:1500])
        print("\n... (full prompt is " + str(len(prompt)) + " chars)")
        return

    # Real run
    client = Anthropic(api_key=api_key)
    success = 0
    failures: list[tuple[str, str]] = []

    # Track customer matching outcomes
    action_stats = {
        "matched_email":  0,
        "matched_name":   0,
        "added_new":      0,
        "no_match":       0,
    }
    new_customers_log: list[tuple[str, str, str]] = []  # (po_number, customer_id, name)

    for pdf_path in tqdm(pdfs, desc="Extracting"):
        try:
            text = extract_pdf_text(pdf_path)
            if not text.strip():
                raise ValueError("PDF text extraction returned empty")
            po_num = detect_po_number(pdf_path.name, text)
            prompt = build_prompt(text, po_num)

            extracted = extract_one(client, SYSTEM_PROMPT_PDF_PO, prompt)

            # Match or auto-add customer
            customer_id, action = match_or_add_customer(
                db, extracted, auto_add=not args.no_auto_add
            )
            extracted["matched_customer_id"] = customer_id
            extracted["_customer_match_action"] = action
            action_stats[action] += 1

            if action == "added_new":
                new_customers_log.append((
                    extracted.get("po_number", po_num),
                    customer_id,
                    extracted.get("cust_name", ""),
                ))

            extracted["_source_pdf"] = pdf_path.name
            extracted["_extracted_at"] = time.time()
            extracted["_model"] = MODEL
            extracted["_extraction_method"] = "pdf_via_llm"

            db[TARGET_COLLECTION].update_one(
                {"po_number": extracted.get("po_number", po_num)},
                {"$set": extracted},
                upsert=True,
            )
            success += 1
        except Exception as e:
            failures.append((pdf_path.name, str(e)[:200]))

    db[TARGET_COLLECTION].create_index("po_number", unique=True)
    db[TARGET_COLLECTION].create_index("matched_customer_id")
    db[TARGET_COLLECTION].create_index("po_status")

    # Summary
    print("\n" + "=" * 50)
    print(f"Successful extractions: {success}")
    print(f"Failures:               {len(failures)}")
    if failures:
        print("\nFailures:")
        for name, err in failures[:5]:
            print(f"  {name}: {err}")

    # Customer matching summary
    print(f"\nCustomer matching outcomes:")
    print(f"  Matched by email:       {action_stats['matched_email']}")
    print(f"  Matched by name:        {action_stats['matched_name']}")
    print(f"  Added as new customer:  {action_stats['added_new']}")
    print(f"  No match (unresolved):  {action_stats['no_match']}")

    if new_customers_log:
        print(f"\nNew customers added to DB this run:")
        for po_num, cust_id, name in new_customers_log:
            print(f"  {po_num} → {cust_id} ({name})")

    final_customer_count = db.customers.count_documents({})
    if final_customer_count != initial_customer_count:
        print(f"\nCustomer count: {initial_customer_count} → {final_customer_count} "
              f"(+{final_customer_count - initial_customer_count})")

    total = db[TARGET_COLLECTION].count_documents({})
    matched_cust = db[TARGET_COLLECTION].count_documents({"matched_customer_id": {"$ne": None}})
    print(f"\nTotal POs in {TARGET_COLLECTION}: {total}")
    print(f"POs linked to customer:   {matched_cust}/{total}")

    # Confidence breakdown
    print(f"\nConfidence distribution:")
    for r in db[TARGET_COLLECTION].aggregate([
        {"$match": {"extraction_confidence": {"$exists": True}}},
        {"$group": {"_id": "$extraction_confidence", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]):
        print(f"  {str(r['_id']):15} {r['count']}")

    print(f"\nNext steps:")
    print(f"  python ingest\\load_po_pdfs.py        # PDFs to GridFS")
    print(f"  python ingest\\load_pos_to_neo4j.py   # POs into graph")


if __name__ == "__main__":
    main()
