"""
load_po_pdfs.py
================
Phase 3B step: load PO PDF binaries into MongoDB GridFS bucket `po_files`.

Mirrors the .eml → email_files pattern from load_mongo.py — gives the
chatbot a way to retrieve the original PDF for any po_number, while the
parsed structured data already lives in the pos collection.

Looks for PDFs at:  data\\purchase_orders\\PO-*.pdf
Stores in bucket:   po_files (GridFS)
Tagged with:        po_number, content_type, original_filename

Usage:
    python ingest\\load_po_pdfs.py
    python ingest\\load_po_pdfs.py --drop          # drop bucket first (clean reload)
    python ingest\\load_po_pdfs.py --pdf-dir PATH  # custom PDF folder
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import gridfs
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm


DEFAULT_PDF_DIR = Path("data") / "purchase_orders"
BUCKET_NAME = "po_files"

# PO number pattern: PO-2026-P0042, PO-2026-C0019, PO-2026-E0073, etc.
PO_NUMBER_RE = re.compile(r"PO-\d{4}-[A-Z]\d{4}", re.IGNORECASE)


def extract_po_number(filename: str) -> str:
    """Extract the PO number from a filename."""
    match = PO_NUMBER_RE.search(filename)
    if match:
        return match.group(0).upper()
    # Fall back to the whole stem if no canonical PO number found
    return Path(filename).stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=str, default=str(DEFAULT_PDF_DIR),
                        help=f"Folder containing PO PDFs (default: {DEFAULT_PDF_DIR})")
    parser.add_argument("--drop", action="store_true",
                        help="Drop the GridFS bucket before loading")
    args = parser.parse_args()

    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "trades_quotes")
    if not mongo_uri:
        print("ERROR: MONGO_URI not set in .env"); sys.exit(1)

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"ERROR: PDF folder not found at {pdf_dir}"); sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No .pdf files found in {pdf_dir}"); sys.exit(1)

    print(f"Connecting to MongoDB ({db_name})...")
    db = MongoClient(mongo_uri)[db_name]
    fs = gridfs.GridFS(db, collection=BUCKET_NAME)

    if args.drop:
        print(f"Dropping bucket '{BUCKET_NAME}'...")
        db[f"{BUCKET_NAME}.files"].drop()
        db[f"{BUCKET_NAME}.chunks"].drop()

    print(f"\nUploading {len(pdfs)} PDFs to GridFS bucket '{BUCKET_NAME}'...")

    skipped = 0
    uploaded = 0
    for pdf_path in tqdm(pdfs, desc="  uploading"):
        po_number = extract_po_number(pdf_path.name)
        # Skip if already uploaded (idempotent)
        if not args.drop and fs.exists({"po_number": po_number}):
            skipped += 1
            continue

        with pdf_path.open("rb") as f:
            content = f.read()

        fs.put(
            content,
            filename=pdf_path.name,
            po_number=po_number,
            content_type="application/pdf",
            original_filename=pdf_path.name,
            size_bytes=len(content),
        )
        uploaded += 1

    print(f"\n  Uploaded: {uploaded}")
    print(f"  Skipped (already present): {skipped}")

    # Cross-link to the pos collection
    print(f"\nCross-linking to pos collection...")
    pos_count = db.pos.count_documents({})
    matched = 0
    for pdf_doc in db[f"{BUCKET_NAME}.files"].find({}, {"po_number": 1}):
        po_number = pdf_doc.get("po_number")
        if po_number:
            result = db.pos.update_one(
                {"po_number": po_number},
                {"$set": {"pdf_gridfs_filename": pdf_doc.get("filename")}},
            )
            if result.matched_count:
                matched += 1

    print(f"  Cross-linked PDFs to pos documents: {matched}/{pos_count}")

    # Final summary
    files_count = db[f"{BUCKET_NAME}.files"].count_documents({})
    chunks_count = db[f"{BUCKET_NAME}.chunks"].count_documents({})
    print(f"\nFinal state:")
    print(f"  {BUCKET_NAME}.files:  {files_count} files")
    print(f"  {BUCKET_NAME}.chunks: {chunks_count} chunks")


if __name__ == "__main__":
    main()
