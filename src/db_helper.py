from pymongo import MongoClient
from typing import Dict, List, Any, Optional, Union
from bson.objectid import ObjectId
import os
import os

class MongoDBHelper:
    """
    A helper class to perform CRUD operations and run queries on MongoDB.
    """
    
    def __init__(self, connection_string: str = "mongodb://ec2-15-206-99-66.ap-south-1.compute.amazonaws.com:27017"):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: The MongoDB connection string
        """
        self.client = MongoClient(connection_string)
    
    def list_databases(self) -> List[str]:
        """
        List all databases in the MongoDB server.
        
        Returns:
            List of database names
        """
        return self.client.list_database_names()
    
    def list_collections(self, db_name: str) -> List[str]:
        """
        List all collections in a database.
        
        Args:
            db_name: Name of the database
            
        Returns:
            List of collection names
        """
        db = self.get_database(db_name)
        return db.list_collection_names()
        
    def get_database(self, db_name: str):
        """
        Get a database by name.
        
        Args:
            db_name: Name of the database
            
        Returns:
            The database object
        """
        return self.client[db_name]
    
    def get_collection(self, db_name: str, collection_name: str):
        """
        Get a collection from a database.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            
        Returns:
            The collection object
        """
        db = self.get_database(db_name)
        return db[collection_name]
    
    # Create operations
    def insert_one(self, db_name: str, collection_name: str, document: Dict) -> str:
        """
        Insert a single document into a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            document: Document to insert
            
        Returns:
            The ID of the inserted document
        """
        collection = self.get_collection(db_name, collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def insert_many(self, db_name: str, collection_name: str, documents: List[Dict]) -> List[str]:
        """
        Insert multiple documents into a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            documents: List of documents to insert
            
        Returns:
            List of IDs for the inserted documents
        """
        collection = self.get_collection(db_name, collection_name)
        result = collection.insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    # Read operations
    def find_one(self, db_name: str, collection_name: str, query: Dict = None) -> Optional[Dict]:
        """
        Find a single document in a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            query: Query filter
            
        Returns:
            The found document or None
        """
        collection = self.get_collection(db_name, collection_name)
        return collection.find_one(query or {})
    
    def find_by_id(self, db_name: str, collection_name: str, document_id: str) -> Optional[Dict]:
        """
        Find a document by its ID.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            document_id: ID of the document
            
        Returns:
            The found document or None
        """
        collection = self.get_collection(db_name, collection_name)
        return collection.find_one({"_id": ObjectId(document_id)})
    
    def find_many(self, db_name: str, collection_name: str, query: Dict = None, 
                 sort: List = None, limit: int = 0, skip: int = 0) -> List[Dict]:
        """
        Find multiple documents in a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            query: Query filter
            sort: List of (key, direction) tuples for sorting
            limit: Maximum number of documents to return (0 for all)
            skip: Number of documents to skip
            
        Returns:
            List of found documents
        """
        collection = self.get_collection(db_name, collection_name)
        cursor = collection.find(query or {})
        
        if sort:
            cursor = cursor.sort(sort)
        
        if skip:
            cursor = cursor.skip(skip)
            
        if limit:
            cursor = cursor.limit(limit)
            
        return list(cursor)
    
    def count_documents(self, db_name: str, collection_name: str, query: Dict = None) -> int:
        """
        Count documents in a collection based on query.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            query: Query filter
            
        Returns:
            Number of matching documents
        """
        collection = self.get_collection(db_name, collection_name)
        return collection.count_documents(query or {})
    
    # Update operations
    def update_one(self, db_name: str, collection_name: str, query: Dict, update: Dict) -> int:
        """
        Update a single document in a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            query: Query filter to match the document
            update: Update operations to apply
            
        Returns:
            Number of modified documents
        """
        collection = self.get_collection(db_name, collection_name)
        result = collection.update_one(query, update)
        return result.modified_count
    
    def update_by_id(self, db_name: str, collection_name: str, document_id: str, update: Dict) -> int:
        """
        Update a document by its ID.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            document_id: ID of the document
            update: Update operations to apply
            
        Returns:
            Number of modified documents
        """
        collection = self.get_collection(db_name, collection_name)
        result = collection.update_one({"_id": ObjectId(document_id)}, update)
        return result.modified_count
    
    def update_many(self, db_name: str, collection_name: str, query: Dict, update: Dict) -> int:
        """
        Update multiple documents in a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            query: Query filter to match the documents
            update: Update operations to apply
            
        Returns:
            Number of modified documents
        """
        collection = self.get_collection(db_name, collection_name)
        result = collection.update_many(query, update)
        return result.modified_count
    
    # Delete operations
    def delete_one(self, db_name: str, collection_name: str, query: Dict) -> int:
        """
        Delete a single document from a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            query: Query filter to match the document
            
        Returns:
            Number of deleted documents
        """
        collection = self.get_collection(db_name, collection_name)
        result = collection.delete_one(query)
        return result.deleted_count
    
    def delete_by_id(self, db_name: str, collection_name: str, document_id: str) -> int:
        """
        Delete a document by its ID.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            document_id: ID of the document
            
        Returns:
            Number of deleted documents
        """
        collection = self.get_collection(db_name, collection_name)
        result = collection.delete_one({"_id": ObjectId(document_id)})
        return result.deleted_count
    
    def delete_many(self, db_name: str, collection_name: str, query: Dict) -> int:
        """
        Delete multiple documents from a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            query: Query filter to match the documents
            
        Returns:
            Number of deleted documents
        """
        collection = self.get_collection(db_name, collection_name)
        result = collection.delete_many(query)
        return result.deleted_count
    
    # Aggregation and advanced queries
    def aggregate(self, db_name: str, collection_name: str, pipeline: List[Dict]) -> List[Dict]:
        """
        Perform an aggregation operation.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            pipeline: Aggregation pipeline stages
            
        Returns:
            List of aggregation results
        """
        collection = self.get_collection(db_name, collection_name)
        return list(collection.aggregate(pipeline))
    
    def distinct(self, db_name: str, collection_name: str, field: str, query: Dict = None) -> List:
        """
        Get distinct values for a field.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            field: Field to get distinct values for
            query: Query filter
            
        Returns:
            List of distinct values
        """
        collection = self.get_collection(db_name, collection_name)
        return collection.distinct(field, query or {})
    
    def create_index(self, db_name: str, collection_name: str, keys: Union[str, List], **kwargs):
        """
        Create an index on a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            keys: Keys to index
            **kwargs: Additional arguments for index creation
            
        Returns:
            Name of the created index
        """
        collection = self.get_collection(db_name, collection_name)
        return collection.create_index(keys, **kwargs)
    
    # Methods for finding descriptions
    def find_documents_with_description(self, db_name: str, collection_name: str, limit: int = 0) -> List[Dict]:
        """
        Find all documents that have a 'description' key.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            limit: Maximum number of documents to return (0 for all)
            
        Returns:
            List of documents containing 'description' field
        """
        collection = self.get_collection(db_name, collection_name)
        cursor = collection.find({"description": {"$exists": True}})
        
        if limit:
            cursor = cursor.limit(limit)
            
        return list(cursor)
    
    def get_all_descriptions(self, db_name: str, collection_name: str) -> List[Dict]:
        """
        Get all descriptions from documents in a collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            
        Returns:
            List of dictionaries with document ID and description
        """
        collection = self.get_collection(db_name, collection_name)
        cursor = collection.find(
            {"description": {"$exists": True}},
            {"_id": 1, "description": 1}
        )
        
        return list(cursor)
    
    def search_in_descriptions(self, db_name: str, collection_name: str, search_text: str) -> List[Dict]:
        """
        Search for text in descriptions using text search.
        Note: This requires a text index on the 'description' field.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            search_text: Text to search for in descriptions
            
        Returns:
            List of matching documents
        """
        # Ensure text index exists
        collection = self.get_collection(db_name, collection_name)
        try:
            collection.create_index([("description", "text")])
        except Exception as e:
            print(f"Warning: Could not create text index: {str(e)}")
            
        # Perform text search
        return list(collection.find({"$text": {"$search": search_text}}))
    
    def find_by_image_path(self, db_name: str, collection_name: str, image_path: str) -> Optional[Dict]:
        """
        Find a document by its image_path field with multiple search strategies.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            image_path: Image path to search for
            
        Returns:
            The found document or None
        """
        collection = self.get_collection(db_name, collection_name)
        
        # Extract filename from the path
        filename = os.path.basename(image_path)
        
        # Clean the image path - remove S3 prefixes if present
        clean_path = image_path
        if clean_path.startswith('s3://sparkassets/'):
            clean_path = clean_path.replace('s3://sparkassets/', '')
        elif clean_path.startswith('s3://'):
            clean_path = clean_path.split('/', 2)[2] if '/' in clean_path[5:] else clean_path
        elif clean_path.startswith('https://'):
            from urllib.parse import urlparse
            parsed_url = urlparse(clean_path)
            clean_path = parsed_url.path.lstrip('/')
        
        # If this is a temp path, extract the meaningful part
        # Pattern: mongo_<id>_<timestamp>.png
        if filename.startswith('mongo_') and '_' in filename:
            # Extract the mongo ID from filename
            parts = filename.split('_')
            if len(parts) >= 2:
                mongo_id_part = parts[1]  # The part after 'mongo_'
                # Try to find document by this ID pattern
                try:
                    from bson.objectid import ObjectId
                    if len(mongo_id_part) == 24:  # Standard MongoDB ObjectId length
                        result = collection.find_one({"_id": ObjectId(mongo_id_part)})
                        if result:
                            return result
                except:
                    pass  # Not a valid ObjectId, continue with other searches
        
        # Try multiple search strategies
        search_queries = [
            {"image_path": clean_path},  # Exact match
            {"image_path": image_path},  # Original path
            {"image_path": filename},  # Just filename
            {"image_path": {"$regex": filename.replace('.', '\\.')}},  # Regex search for filename
            {"image_path": {"$regex": filename.split('.')[0]}},  # Search without extension
        ]
        
        # If filename has a pattern like mongo_<id>_timestamp, search for documents with that ID pattern
        if filename.startswith('mongo_') and '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2:
                search_queries.append({"image_path": {"$regex": parts[1]}})  # Search for the ID part
        
        for query in search_queries:
            try:
                result = collection.find_one(query)
                if result:
                    return result
            except Exception:
                continue  # Skip invalid queries
        
        return None

    def debug_image_paths(self, db_name: str, collection_name: str, limit: int = 10) -> List[Dict]:
        """
        Debug method to see what image_path values exist in the collection.
        
        Args:
            db_name: Name of the database
            collection_name: Name of the collection
            limit: Number of documents to sample
            
        Returns:
            List of documents with their image_path and _id
        """
        collection = self.get_collection(db_name, collection_name)
        
        # Get sample documents and their image_path values
        pipeline = [
            {"$match": {"image_path": {"$exists": True}}},
            {"$project": {"_id": 1, "image_path": 1}},
            {"$limit": limit}
        ]
        
        return list(collection.aggregate(pipeline))

    def close(self):
        """Close the MongoDB connection"""
        self.client.close()
