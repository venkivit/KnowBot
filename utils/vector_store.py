import chromadb
import pandas as pd
from chromadb.config import Settings
from typing import List, Dict

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./data"
        ))
        self.collection = self.client.get_or_create_collection("knowledge_base")

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the vector store"""
        try:
            if metadata is None:
                metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]
            
            self.collection.add(
                documents=documents,
                ids=[f"doc_{i}" for i in range(len(documents))],
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
        return self.collection.count()
