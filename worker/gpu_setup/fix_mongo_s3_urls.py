#!/usr/bin/env python3
"""
MongoDB S3 URL Migration Script
================================
Finds every document in the biomes collection that contains the old
S3 bucket name "sparkassets" (ap-south-1) and replaces it with
"sparkassets-us" (us-east-1) — keeping the rest of each string intact.

Replacements performed:
  https://sparkassets.s3.ap-south-1.amazonaws.com/
    → https://sparkassets-us.s3.us-east-1.amazonaws.com/

  s3://sparkassets/
    → s3://sparkassets-us/

  (plain S3 key paths like "images/..." are untouched — they have no bucket prefix)

Usage:
    python fix_mongo_s3_urls.py [--dry-run]

    --dry-run  : print what would change, don't write to MongoDB
"""

import argparse
import os
import re
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI    = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or ""
DB_NAME      = os.getenv("MONGO_DB",  "World_builder")
COLLECTION   = "biomes"

# ── Replacement patterns ──────────────────────────────────────────────────────
REPLACEMENTS = [
    # Full HTTPS URL  (old region → new region + new bucket)
    (
        "https://sparkassets.s3.ap-south-1.amazonaws.com/",
        "https://sparkassets-us.s3.us-east-1.amazonaws.com/",
    ),
    # Also handle if someone used the wrong bucket name spelling
    (
        "https://spackassets.s3.ap-south-1.amazonaws.com/",
        "https://sparkassets-us.s3.us-east-1.amazonaws.com/",
    ),
    # s3:// URI style
    (
        "s3://sparkassets/",
        "s3://sparkassets-us/",
    ),
    (
        "s3://spackassets/",
        "s3://sparkassets-us/",
    ),
]

def fix_string(value: str) -> tuple[str, bool]:
    """Apply all replacements to a string. Returns (new_value, changed)."""
    original = value
    for old, new in REPLACEMENTS:
        value = value.replace(old, new)
    return value, value != original


def fix_value(value):
    """Recursively fix strings inside dicts/lists."""
    if isinstance(value, str):
        fixed, changed = fix_string(value)
        return fixed, changed
    elif isinstance(value, dict):
        new_dict = {}
        any_changed = False
        for k, v in value.items():
            new_v, changed = fix_value(v)
            new_dict[k] = new_v
            if changed:
                any_changed = True
        return new_dict, any_changed
    elif isinstance(value, list):
        new_list = []
        any_changed = False
        for item in value:
            new_item, changed = fix_value(item)
            new_list.append(new_item)
            if changed:
                any_changed = True
        return new_list, any_changed
    else:
        return value, False


def main():
    parser = argparse.ArgumentParser(description="Fix S3 bucket URLs in MongoDB biomes")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    print("=" * 60)
    print(f"MongoDB S3 URL Migration")
    print(f"  URI        : {MONGO_URI.split('@')[-1]}")
    print(f"  DB         : {DB_NAME}.{COLLECTION}")
    print(f"  Mode       : {'DRY RUN' if args.dry_run else 'LIVE UPDATE'}")
    print("=" * 60)

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    db     = client[DB_NAME]
    col    = db[COLLECTION]

    total_docs    = col.count_documents({})
    updated_docs  = 0
    unchanged_docs = 0

    print(f"\nScanning {total_docs} documents in '{COLLECTION}'...")

    for doc in col.find({}):
        doc_id = doc["_id"]
        original_doc = dict(doc)

        # Fix the entire document (recursive)
        new_doc, changed = fix_value(doc)

        if not changed:
            unchanged_docs += 1
            continue

        # Print diff summary
        print(f"\n  Doc: {doc_id}")
        # Find changed fields for readable output
        def find_changes(old, new, path=""):
            if isinstance(old, dict) and isinstance(new, dict):
                for k in old:
                    find_changes(old.get(k), new.get(k), f"{path}.{k}" if path else k)
            elif isinstance(old, list) and isinstance(new, list):
                for i, (o, n) in enumerate(zip(old, new)):
                    find_changes(o, n, f"{path}[{i}]")
            elif isinstance(old, str) and old != new:
                print(f"    [{path}]")
                print(f"      OLD: {old[:100]}")
                print(f"      NEW: {new[:100]}")

        find_changes(original_doc, new_doc)

        if not args.dry_run:
            # Remove _id from the update payload (can't update _id)
            update_payload = {k: v for k, v in new_doc.items() if k != "_id"}
            col.replace_one({"_id": doc_id}, update_payload)
            updated_docs += 1
            print(f"    → UPDATED")
        else:
            updated_docs += 1
            print(f"    → would UPDATE (dry-run)")

    print("\n" + "=" * 60)
    print(f"Results:")
    print(f"  Total docs scanned : {total_docs}")
    print(f"  Docs with changes  : {updated_docs}")
    print(f"  Docs unchanged     : {unchanged_docs}")
    if args.dry_run:
        print("\n  DRY RUN — no writes made. Re-run without --dry-run to apply.")
    else:
        print(f"\n  Done. {updated_docs} documents updated.")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
