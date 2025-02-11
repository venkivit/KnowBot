import chromadb
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os

class VectorStore:
    def __init__(self):
        self.persist_dir = "./chroma_db"
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> bool:
        """Add documents to the vector store"""
        try:
            if metadata is None:
                metadata = [{"source": f"doc_{i}", "timestamp": pd.Timestamp.now().isoformat()} 
                          for i in range(len(documents))]

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

    def get_documents(self, limit: int = 10, offset: int = 0) -> Tuple[List[str], List[Dict]]:
        """Get documents with their metadata"""
        try:
            results = self.collection.get(
                limit=limit,
                offset=offset
            )
            return results['documents'], results['metadatas']
        except Exception as e:
            print(f"Error getting documents: {str(e)}")
            return [], []

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            # Get all documents to find the correct ID
            results = self.collection.get()
            if not results['ids']:
                return False
                
            # Find the document index
            try:
                idx = results['metadatas'].index(next(meta for meta in results['metadatas'] if meta['source'] == doc_id))
                doc_hash_id = results['ids'][idx]
                self.collection.delete(ids=[doc_hash_id])
                return True
            except StopIteration:
                print(f"Document with source {doc_id} not found")
                return False
                
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
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

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": "knowledge_base",
                "persistence_path": self.persist_dir
            }
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}