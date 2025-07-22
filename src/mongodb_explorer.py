#!/usr/bin/env python3
# MongoDB Database Explorer

from db_helper import MongoDBHelper
import argparse
import json

def print_json(data):
    """Print data as formatted JSON"""
    print(json.dumps(data, indent=2, default=str))

def main():
    """Main function to interact with MongoDB"""
    parser = argparse.ArgumentParser(description="MongoDB Explorer Tool")
    parser.add_argument("--connection", "-c", default="mongodb://ec2-15-206-99-66.ap-south-1.compute.amazonaws.com:27017",
                        help="MongoDB connection string")
    parser.add_argument("--list-dbs", "-ld", action="store_true", help="List all databases")
    parser.add_argument("--list-collections", "-lc", metavar="DB_NAME", help="List collections in database")
    parser.add_argument("--find-descriptions", "-fd", nargs=2, metavar=("DB_NAME", "COLLECTION_NAME"), 
                       help="Find documents with description field")
    parser.add_argument("--search-descriptions", "-sd", nargs=3, metavar=("DB_NAME", "COLLECTION_NAME", "SEARCH_TEXT"),
                       help="Search for text in descriptions")
    parser.add_argument("--view-db", "-vd", metavar="DB_NAME", help="View all collections and sample data in a database")
    parser.add_argument("--view-collection", "-vc", nargs=2, metavar=("DB_NAME", "COLLECTION_NAME"),
                       help="View all documents in a collection")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Limit the number of results")
    parser.add_argument("--skip", "-s", type=int, default=0, help="Skip number of results (for pagination)")
    parser.add_argument("--all", "-a", action="store_true", help="Show all results without limit")
    
    args = parser.parse_args()
    
    # Initialize MongoDB helper
    mongo = MongoDBHelper(args.connection)
    
    try:
        # List all databases
        if args.list_dbs:
            dbs = mongo.list_databases()
            print("\n--- Available Databases ---")
            for db in dbs:
                print(f"- {db}")
        
        # List collections in a database
        elif args.list_collections:
            collections = mongo.list_collections(args.list_collections)
            print(f"\n--- Collections in {args.list_collections} ---")
            for collection in collections:
                print(f"- {collection}")
        
        # Find documents with description field
        elif args.find_descriptions:
            db_name, collection_name = args.find_descriptions
            docs = mongo.find_documents_with_description(db_name, collection_name, args.limit)
            count = mongo.count_documents(db_name, collection_name, {"description": {"$exists": True}})
            
            print(f"\n--- Documents with Description in {db_name}.{collection_name} ---")
            print(f"Found {count} documents, displaying {min(args.limit, count) if args.limit > 0 else count}:")
            print_json(docs)
        
        # Search in descriptions
        elif args.search_descriptions:
            db_name, collection_name, search_text = args.search_descriptions
            docs = mongo.search_in_descriptions(db_name, collection_name, search_text)
            
            print(f"\n--- Search Results for '{search_text}' in {db_name}.{collection_name} ---")
            print(f"Found {len(docs)} matching documents:")
            
            limit = None if args.all else args.limit
            skip = args.skip
            
            if limit and not args.all:
                docs_slice = docs[skip:skip+limit]
                print(f"Displaying documents {skip+1}-{skip+len(docs_slice)} of {len(docs)}")
                print_json(docs_slice)
            else:
                print_json(docs[skip:])
        
        # View entire database structure and contents
        elif args.view_db:
            db_name = args.view_db
            collections = mongo.list_collections(db_name)
            print(f"\n=== Database Contents: {db_name} ===")
            print(f"Total collections: {len(collections)}")
            
            for collection_name in collections:
                count = mongo.count_documents(db_name, collection_name)
                print(f"\n--- Collection: {collection_name} ({count} documents) ---")
                
                limit = 3 if not args.all and args.limit > 3 else args.limit
                if not args.all and count > limit:
                    print(f"Showing {limit} sample documents (use --view-collection for more):")
                
                # Get sample documents from this collection
                docs = mongo.find_many(db_name, collection_name, limit=limit)
                print_json(docs)
        
        # View all documents in a collection
        elif args.view_collection:
            db_name, collection_name = args.view_collection
            count = mongo.count_documents(db_name, collection_name)
            
            limit = None if args.all else args.limit
            skip = args.skip
            
            print(f"\n--- Collection Contents: {db_name}.{collection_name} ---")
            print(f"Total documents: {count}")
            
            if limit and not args.all:
                docs = mongo.find_many(db_name, collection_name, limit=limit, skip=skip)
                print(f"Displaying documents {skip+1}-{min(skip+limit, count)} of {count}")
            else:
                docs = mongo.find_many(db_name, collection_name, skip=skip)
                print(f"Displaying all documents starting from {skip+1}")
            
            print_json(docs)
        
        # Show usage if no arguments provided
        else:
            parser.print_help()
    
    finally:
        # Close connection when done
        mongo.close()

if __name__ == "__main__":
    main()
