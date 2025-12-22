"""
Enhanced RAG System with Chroma Vector Database
Production-ready vector database with proper embeddings
Uses ChromaDB for efficient similarity search and persistence
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    logger.error("ChromaDB not available. Please install: pip install chromadb")
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.error("SentenceTransformers not available. Please install: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EnhancedRAG:
    """
    Enhanced RAG system with ChromaDB vector database
    Production-ready implementation with:
    - ChromaDB for persistent vector storage
    - SentenceTransformer embeddings
    - Smart chunking and similarity search
    - Efficient retrieval and scaling
    """
    
    def __init__(self, collection_name: str = "neurotherapy_knowledge", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Check dependencies
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required but not installed. Run: pip install chromadb")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers is required but not installed. Run: pip install sentence-transformers")
        
        # Create directory if not exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded SentenceTransformer embedding model")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {str(e)}")
            raise
        
        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"✅ ChromaDB client initialized at: {persist_directory}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB: {str(e)}")
            raise
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "DAWOS Agent neurotherapy knowledge base"}
            )
            
            # Get collection stats
            count = self.collection.count()
            logger.info(f"✅ Collection '{collection_name}' ready with {count} documents")
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection: {str(e)}")
            raise
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate semantic embedding for text using SentenceTransformer"""
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {str(e)}")
            raise
    
    def _smart_chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Smart text chunking that preserves context
        Better than simple sentence splitting
        """
        # Split by sentences first
        sentences = text.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                words = current_chunk.split()
                if len(words) > overlap:
                    overlap_text = ' '.join(words[-overlap:])
                    current_chunk = overlap_text + ". " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def ingest_document(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Ingest a document into ChromaDB vector database
        
        Args:
            text: Document text content
            document_id: Unique identifier for the document
            metadata: Optional metadata for the document
            
        Returns:
            Number of chunks created
        """
        try:
            # Smart chunking
            chunks = self._smart_chunk_text(text)
            logger.info(f"📄 Document '{document_id}' split into {len(chunks)} chunks")
            
            # Prepare data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            chunk_embeddings = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                # Create unique ID for each chunk
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Generate embedding
                embedding = self._generate_embedding(chunk)
                
                # Prepare metadata
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    **(metadata or {})
                }
                
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)
                chunk_embeddings.append(embedding)
                chunk_metadatas.append(chunk_metadata)
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=chunk_embeddings,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"✅ Successfully ingested {len(chunks)} chunks from '{document_id}' into ChromaDB")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest document '{document_id}': {str(e)}")
            return 0
    
    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve most relevant chunks using ChromaDB similarity search
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            Combined text from most relevant chunks
        """
        try:
            # Check if collection has documents
            count = self.collection.count()
            if count == 0:
                logger.warning(f"Collection is empty for query: '{query}'")
                return "No relevant information found in the knowledge base."
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Query ChromaDB for similar documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                logger.warning(f"No results found for query: '{query}'")
                return "No relevant information found in the knowledge base."
            
            # Extract and combine results
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            relevant_chunks = []
            for i, doc in enumerate(documents):
                # Log result info
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0
                similarity = 1 - distance  # Convert distance to similarity
                
                logger.info(f"📖 Result {i+1}: similarity={similarity:.3f}, doc={metadata.get('document_id', 'unknown')}")
                relevant_chunks.append(doc)
            
            combined_result = "\n\n".join(relevant_chunks)
            logger.info(f"✅ Retrieved {len(relevant_chunks)} relevant chunks for query: '{query}'")
            
            return combined_result
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed for query '{query}': {str(e)}")
            return f"Error retrieving information: {str(e)}"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_model": "all-MiniLM-L6-v2",
                "database_type": "ChromaDB",
                "status": "ready"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    def clear_collection(self):
        """Clear all data from the ChromaDB collection"""
        try:
            # Delete and recreate collection
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "DAWOS Agent neurotherapy knowledge base"}
            )
            logger.info(f"✅ Cleared ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {str(e)}")