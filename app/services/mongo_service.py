from app.config import db

def ping_db():
    return db.command("ping")

def get_data(collection_name: str, limit: int = 5):
    collection = db[collection_name]
    return list(collection.find({}, {"_id": 0}).limit(limit))
