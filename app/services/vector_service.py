"""
Vector Service
Handles vector storage and retrieval using Pinecone.
"""

import json
import logging
from typing import List,Any,Dict
from app.config import settings
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec

logger = logging.getLogger(f"rag_app.{__name__}")

class VectorService:
    """Service for vector operations using Pinecone."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the vector service with Pinecone.

        Args:
            api_key: Pinecone API key (optional, uses settings if not provided)
        """
        self.api_key = api_key or settings.PINECONE_API_KEY

        if not self.api_key:
            raise ValueError("Pinecone API Key is required. Set API Key in .env file")
        
        self.region = settings.PINECONE_REGION
        self.cloud = settings.PINECONE_CLOUD
        self.index_name = settings.PINECONE_INDEX_NAME

        # Initialize Pinecone client with gRPC for better performance
        self.pc = PineconeGRPC(api_key= self.api_key)
        self.index = None


    def connect_to_index(self):
        """
        Connect to the Pinecone index.
        Creates the index if it doesn't exist.
        """

        try:
            # check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx['name'] for idx in existing_indexes]

            if self.index_name not in index_names:
                # Create index if it doesn't exist
                logger.info(f"Creating PINECONE Index: {self.index_name}")
                self.pc.create_index(
                    name = self.index_name,
                    dimension = 1536,         # OpenAI text-embedding-3-small dimension
                    metric = 'cosine',
                    specs = ServerlessSpec(
                        cloud= self.cloud,
                        region= self.region
                    )       
                )

                logger.info(f"Index: {self.index_name} created successfully")

            # Connect to the index
            index_description = self.pc.describe_index(name=self.index_name)
            self.index = self.pc.Index(host=index_description['host'])
            logger.info(f"Connected to pinecone index: {self.index_name}")

        except Exception as e:
            raise Exception(f"Failed to connect to Pineconne Index: {str(e)}")
        
    def add_documents(
            self,
            chunks: List[Dict[str,Any]],
            embeddings: List[List[float]],
            filename: str,
            namespace: str = 'default'
    ):
        """
        Store document chunks with their embeddings in Pinecone.

        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors corresponding to chunks
            filename: Source filename for metadata
            namespace: Pinecone namespace for organization (default: "default")

        Raises:
            Exception: If upsert fails
        """
        if not self.index:
            self.connect_to_index()

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks, but {len(embeddings)} embeddings")
        
        try:
            # Prepare vectors for upsert
            vectors_to_upsert = []

            for i, (chunk,embedding) in enumerate(zip(chunks,embeddings)):
                # Create unique ID: filename + chunk_index
                vector_id = f"{filename}_{chunk['chunk_id']}"

                metadata = {
                    "filename": filename,
                    "chunk_id": chunk['chunk_id'],
                    "token_count": chunk['tokens'],
                    "text": chunk['text'],           # Limit text size in metadata (Pinecone has limits)
                    "headings": json.dumps(chunk.get("headings", [])),
                    "page_no": json.dumps(chunk.get("pages", [])),
                    "labels": json.dumps(chunk.get("labels", [])),
                    "has_context": len(chunk.get("headings", [])) > 0    # Quick filter for context-aware chunks
                }

                # Create a vector-tuple: (vector_id, embedding, metadata)
                vectors_to_upsert.append((vector_id,embedding,metadata))

            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i: i+batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=namespace
                )

            logger.info(f"Successfully upserted: {len(vectors_to_upsert)} vectors")

        except Exception as e:
            raise Exception(f"Failed to add documents to Pinecone: {str(e)}")
        

    def search(
            self,
            query_embedding: List[float],
            top_k: int = 3,
            namespace: str = 'default',
            filter_dict: Dict[str,Any] | None= None
    ) -> Dict[str,Any]:
        """
        Search for similar vectors in Pinecone.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return (default: 3)
            namespace: Pinecone namespace to search (default: "default")
            filter_dict: Optional metadata filter

        Returns:
            Dictionary with search results:
                - query: The query vector (first 5 dims for reference)
                - chunks: List of matched chunks with metadata and scores
                - total_found: Number of results returned
        """

        if not self.index:
            self.connect_to_index()

        try:
            
            # Query Pineconne
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict,
                namespace= namespace
            )

            # format results
            chunks = []
            for match in results['matches']:
                chunks.append({
                    "id": match['id'],
                    "score": match['score'],
                    "text": match['metadata'].get('text', ''),
                    "metadata": {
                        "filename": match["metadata"].get("filename", ''),
                        "chunk_id": match["metadata"].get("chunk_id", 0),
                        "token_count": match['metadata'].get("token_count", 0),
                        "headings": json.loads(match['metadata'].get("headings", "[]")),
                        "page_no": json.loads(match["metadata"].get("page_no", "[]")),
                        "labels": json.loads(match["metadata"].get("labels", "[]"))
                    }
                })
            return {
                "query_preview": query_embedding[:5],            # Just first 5 dims for reference
                "chunks": chunks,
                "total_found": len(chunks)
            }

        except Exception as e:
            raise Exception(f"failed to search pineconne: {str(e)}")
        
    def get_index_stat(self) -> Dict[str,Any]:
        """
        Get statistics about the Pinecone index.

        Args:
            namespace: Namespace to get stats for

        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            self.connect_to_index()

        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "namespace": stats.get("namespaces", {})
            }
        
        except Exception as e:
            raise Exception(f"Failed to get index from stat: {str(e)}")
        
    
    def delete_by_filename(self, filename: str, namespace: str = 'default'):
        """
        Delete all vectors associated with a filename.

        Args:
            filename: Filename to delete
            namespace: Namespace containing the vectors
        """

        if not self.index:
            self.connect_to_index()

        try:
            # Delete using metadata filter
            self.index.delete(
                filter={
                    "filename": {"$eq": filename}
                },
                namespace=namespace
            )

            logger.info(f"Deleted all vectors for filenam: {filename}")

        except Exception as e:
            raise Exception(f"Failed to delete vectors: {str(e)}")
        

    

        


