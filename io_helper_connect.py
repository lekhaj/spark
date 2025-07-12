import boto3
import pymongo
import os

# --- S3 Helpers ---
def download_from_s3(bucket_name, s3_key, download_path):
    s3 = boto3.client('s3')
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    s3.download_file(bucket_name, s3_key, download_path)
    print(f"[S3] Downloaded: {s3_key} → {download_path}")
    return download_path

def upload_to_s3(bucket_name, s3_key, file_path):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket_name, s3_key)
    print(f"[S3] Uploaded: {file_path} → s3://{bucket_name}/{s3_key}")
    return f"s3://{bucket_name}/{s3_key}"

def get_latest_glb_from_s3(bucket_name, prefix="incoming_models/"):
    """
    Finds the latest .glb file in a specified S3 folder (prefix).
    """
    s3 = boto3.client("s3")
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        contents = response.get("Contents", [])
        glb_files = [obj for obj in contents if obj["Key"].endswith(".glb")]
        if not glb_files:
            print("[S3] No .glb files found in the prefix.")
            return None
        latest = max(glb_files, key=lambda x: x["LastModified"])
        print(f"[S3] Found latest .glb: {latest['Key']}")
        return latest["Key"]
    except Exception as e:
        print(f"[S3] Error listing objects: {e}")
        return None

# --- MongoDB Helpers ---
def get_mongo_collection(uri, db_name="spark", collection_name="assets"):
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    return db[collection_name]

def update_asset_status(collection, asset_id, status, output_url=None):
    update_fields = {"status": status}
    if output_url:
        update_fields["output_url"] = output_url
    result = collection.update_one({"_id": asset_id}, {"$set": update_fields})
    print(f"[MongoDB] Updated {asset_id} with status '{status}' and output_url: {output_url}")
    return result.modified_count







