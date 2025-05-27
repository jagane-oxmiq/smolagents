import chromadb
import os
import sys
import json

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} host port [dumpjson]")
    os._exit(255)

# Initialize ChromaDB client
client = chromadb.HttpClient(host=sys.argv[1], port=int(sys.argv[2]))

# Get the collection
collection_name = "oxpython_collection"
collection = client.get_collection(name=collection_name)

# Get all documents from the collection
results = collection.get()

# Print the results
print(f"Collection: {collection_name}")
print(f"Total documents: {collection.count()}")
print("-" * 50)

# Display documents
if results['ids']:
    for i in range(len(results['ids'])):
        print(f"\nDocument {i+1}:")
        print(f"  ID: {results['ids'][i]}")
        
        if results['documents']:
            print(f"  Document:\n{results['documents'][i]}")
        
        if results['metadatas']:
            print(f"  Metadata: {results['metadatas'][i]}")
        
        if 'embeddings' in results and results['embeddings']:
            print(f"  Embedding: {results['embeddings'][i][:5]}...")  # Show first 5 values
    if len(sys.argv) == 4:
        output_data = {
            'collection_name': collection_name,
            'total_documents': collection.count(),
            'documents': []
        }

        for i in range(len(results['ids'])):
            doc = {
                'id': results['ids'][i],
                'document': results['documents'][i] if results['documents'] else None,
                'metadata': results['metadatas'][i] if results['metadatas'] else None,
            }
            output_data['documents'].append(doc)

        with open(f'{collection_name}_dump.json', 'w') as f:
            json.dump(output_data, f, indent=2)
            print(f"\nData saved to {collection_name}_dump.json")
    os._exit(0)
else:
    print("No documents found in the collection.")
    os._exit(255)
