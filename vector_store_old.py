import chromadb
import pandas as pd
from typing import List, Dict, Optional
import os

class VectorStore:
    def __init__(self):
        # Create data directory if it doesn't exist
        self.persist_dir = "./chroma_db"
        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialize ChromaDB client with new configuration
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> bool:
        """Add documents to the vector store"""
        try:
            if metadata is None:
                metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]

            # Generate IDs for documents
            ids = [f"doc_{hash(doc)}" for doc in documents]

            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadata
            )
            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False

    def query_documents(self, query: str, n_results: int = 3) -> List[str]:
        """Query the vector store for relevant documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results['documents'][0]
        except Exception as e:
            print(f"Error querying documents: {str(e)}")
            return []

    def get_document_count(self) -> int:
        """Get the total number of documents in the store"""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error getting document count: {str(e)}")
            return 0