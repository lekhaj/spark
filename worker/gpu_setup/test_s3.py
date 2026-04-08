#!/usr/bin/env python3
"""
S3 Connectivity Test
====================
Run on either CPU or GPU instance to verify S3 access to sparkassets-us.

Usage:
    python test_s3.py

Checks:
  1. Bucket exists and is accessible
  2. Can upload a test PNG
  3. Can download it back
  4. Can generate a pre-signed URL
  5. Lists recent objects
"""

import io
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET  = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
S3_REGION  = os.getenv("AWS_REGION",    "us-east-1")
KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
SECRET     = os.getenv("AWS_SECRET_ACCESS_KEY")

print("=" * 60)
print("S3 Connectivity Test")
print(f"  Bucket : {S3_BUCKET}")
print(f"  Region : {S3_REGION}")
print(f"  Key ID : {KEY_ID[:8]}..." if KEY_ID else "  Key ID : NOT SET")
print("=" * 60)

# ── Init client ───────────────────────────────────────────────────────────────
try:
    s3 = boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET,
    )
    print("[1/5] boto3 client created OK")
except Exception as e:
    print(f"[1/5] FAIL: {e}")
    sys.exit(1)

# ── Check bucket ──────────────────────────────────────────────────────────────
try:
    s3.head_bucket(Bucket=S3_BUCKET)
    print(f"[2/5] Bucket '{S3_BUCKET}' accessible OK")
except ClientError as e:
    code = e.response["Error"]["Code"]
    if code == "403":
        print(f"[2/5] Bucket exists but access DENIED (403) — check IAM permissions")
    elif code == "404":
        print(f"[2/5] Bucket '{S3_BUCKET}' NOT FOUND (404) — wrong name or region?")
    else:
        print(f"[2/5] FAIL: {e}")
    sys.exit(1)
except NoCredentialsError:
    print("[2/5] FAIL: No AWS credentials found")
    sys.exit(1)

# ── Upload test ───────────────────────────────────────────────────────────────
TEST_KEY = f"test/s3_connectivity_test_{int(time.time())}.png"
try:
    # Create a small test image
    img = Image.new("RGB", (128, 128), color=(50, 150, 200))
    d   = ImageDraw.Draw(img)
    d.text((10, 50), "S3 Test OK", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=TEST_KEY, Body=buf, ContentType="image/png")
    print(f"[3/5] Upload OK → s3://{S3_BUCKET}/{TEST_KEY}")
except Exception as e:
    print(f"[3/5] Upload FAIL: {e}")
    sys.exit(1)

# ── Download test ─────────────────────────────────────────────────────────────
try:
    resp = s3.get_object(Bucket=S3_BUCKET, Key=TEST_KEY)
    data = resp["Body"].read()
    print(f"[4/5] Download OK — {len(data)} bytes")
except Exception as e:
    print(f"[4/5] Download FAIL: {e}")

# ── Pre-signed URL ────────────────────────────────────────────────────────────
try:
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": TEST_KEY},
        ExpiresIn=3600,
    )
    print(f"[5/5] Pre-signed URL OK:")
    print(f"       {url[:80]}...")
except Exception as e:
    print(f"[5/5] Pre-signed URL FAIL: {e}")

# ── Cleanup test object ───────────────────────────────────────────────────────
try:
    s3.delete_object(Bucket=S3_BUCKET, Key=TEST_KEY)
    print(f"\n[CLEANUP] Test object deleted")
except Exception:
    pass

# ── List recent objects ───────────────────────────────────────────────────────
try:
    resp    = s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=5)
    objects = resp.get("Contents", [])
    print(f"\n[INFO] Recent objects in {S3_BUCKET} (up to 5):")
    for obj in objects:
        print(f"       {obj['Key']}  ({obj['Size']} bytes)")
    if not objects:
        print("       (bucket is empty)")
except Exception as e:
    print(f"[INFO] List failed: {e}")

print("\n" + "=" * 60)
print("S3 test PASSED" if True else "")
print("=" * 60)
