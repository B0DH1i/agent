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
        """Generate semantic embedding for text using SentenceTransformer (single text)"""
        try:
            # For single text embedding (used in queries)
            embedding = self.embedding_model.encode(
                text, 
                normalize_embeddings=True,
                convert_to_tensor=False  # Return numpy array, then convert to list
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {str(e)}")
            raise
    
    def _smart_chunk_text(self, text: str, chunk_size: int = 400, overlap_ratio: float = 0.2) -> List[str]:
        """
        Smart text chunking with consistent character-based measurements
        Fixed unit consistency: both chunk_size and overlap in characters
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum chunk size in characters
            overlap_ratio: Overlap as ratio of chunk_size (0.2 = 20% overlap)
        """
        # Calculate overlap in characters (consistent units)
        overlap_chars = int(chunk_size * overlap_ratio)
        
        # Split by sentences first for better semantic boundaries
        sentences = text.replace('\n', ' ').split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > chunk_size and current_chunk:
                # Current chunk is ready, add it
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap (character-based)
                if len(current_chunk) > overlap_chars:
                    # Take last overlap_chars characters for overlap
                    overlap_text = current_chunk[-overlap_chars:].strip()
                    # Find word boundary to avoid cutting words
                    space_idx = overlap_text.find(' ')
                    if space_idx > 0:
                        overlap_text = overlap_text[space_idx:].strip()
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very small chunks (less than 50 characters)
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        logger.info(f"📝 Text chunking: {len(text)} chars → {len(chunks)} chunks (avg: {sum(len(c) for c in chunks)//len(chunks) if chunks else 0} chars/chunk)")
        
        return chunks
    
    def ingest_document(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Ingest a document into ChromaDB vector database with batch processing optimization
        
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
            
            if not chunks:
                logger.warning(f"No chunks created for document '{document_id}'")
                return 0
            
            # PERFORMANCE FIX: Batch embedding generation instead of individual calls
            logger.info(f"🚀 Generating embeddings for {len(chunks)} chunks in batch...")
            
            # Generate all embeddings in one batch call (10x faster)
            batch_embeddings = self.embedding_model.encode(
                chunks, 
                normalize_embeddings=True,
                batch_size=32,  # Process in batches of 32 for memory efficiency
                show_progress_bar=True if len(chunks) > 10 else False
            )
            
            # Convert to list format for ChromaDB
            batch_embeddings_list = [embedding.tolist() for embedding in batch_embeddings]
            
            # Prepare data for ChromaDB in batch
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, batch_embeddings_list)):
                # Create unique ID for each chunk
                chunk_id = f"{document_id}_chunk_{i}"
                
                # Prepare metadata with source information for academic referencing
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    "source_reference": f"Source: {document_id}",  # For academic citations
                    "chunk_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    **(metadata or {})
                }
                
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)
                chunk_metadatas.append(chunk_metadata)
            
            # BATCH INSERT: Single ChromaDB operation instead of multiple
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=batch_embeddings_list,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"✅ Successfully batch-ingested {len(chunks)} chunks from '{document_id}' into ChromaDB")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest document '{document_id}': {str(e)}")
            return 0
    
    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve most relevant chunks with academic citations and source references
        Optimized for ReAct agent with structured academic formatting
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            Formatted text with academic citations for agent reasoning
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
            
            # Extract and format results with academic citations
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            formatted_chunks = []
            for i, doc in enumerate(documents):
                # Get metadata for citation
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0
                similarity = 1 - distance  # Convert distance to similarity
                
                # Extract source information
                source_ref = metadata.get('source_reference', f'Source: {metadata.get("document_id", "unknown")}')
                chunk_index = metadata.get('chunk_index', i)
                
                # Format chunk with academic citation
                formatted_chunk = f"[{source_ref}, Section {chunk_index+1}]\n{doc.strip()}"
                
                # Add relevance indicator for agent
                if similarity > 0.8:
                    formatted_chunk = f"🎯 HIGH RELEVANCE ({similarity:.2f})\n{formatted_chunk}"
                elif similarity > 0.6:
                    formatted_chunk = f"📊 MODERATE RELEVANCE ({similarity:.2f})\n{formatted_chunk}"
                else:
                    formatted_chunk = f"📝 LOW RELEVANCE ({similarity:.2f})\n{formatted_chunk}"
                
                formatted_chunks.append(formatted_chunk)
                
                logger.info(f"📖 Retrieved: {source_ref} (similarity: {similarity:.3f})")
            
            # Combine results with clear separators for agent parsing
            combined_result = "\n\n" + "="*50 + "\n\n".join(formatted_chunks) + "\n" + "="*50
            
            # Add summary header for agent
            summary_header = f"ACADEMIC RESEARCH RESULTS ({len(formatted_chunks)} sources found for: '{query}'):\n"
            final_result = summary_header + combined_result
            
            # Add instruction for agent citation
            citation_footer = f"\n\nINSTRUCTION FOR AGENT: When referencing this research, cite the source references provided above (e.g., 'According to [Source: neurotherapy_research, Section 2]...')"
            
            logger.info(f"✅ Retrieved {len(formatted_chunks)} relevant chunks with academic citations for query: '{query}'")
            
            return final_result + citation_footer
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed for query '{query}': {str(e)}")
            return f"Error retrieving information: {str(e)}"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the ChromaDB collection"""
        try:
            count = self.collection.count()
            
            # Get sample metadata for analysis
            sample_results = self.collection.get(limit=10, include=["metadatas"])
            
            # Analyze document sources
            sources = set()
            avg_chunk_length = 0
            total_chunks = 0
            
            if sample_results and sample_results.get("metadatas"):
                for metadata in sample_results["metadatas"]:
                    if metadata:
                        sources.add(metadata.get("document_id", "unknown"))
                        chunk_length = metadata.get("chunk_length", 0)
                        if chunk_length > 0:
                            avg_chunk_length += chunk_length
                            total_chunks += 1
            
            avg_chunk_length = avg_chunk_length // total_chunks if total_chunks > 0 else 0
            
            return {
                "total_chunks": count,
                "unique_documents": len(sources),
                "document_sources": list(sources),
                "average_chunk_length": avg_chunk_length,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_model": "all-MiniLM-L6-v2",
                "database_type": "ChromaDB",
                "performance_optimizations": [
                    "Batch embedding generation",
                    "Smart chunking with overlap",
                    "Academic citation formatting",
                    "Relevance scoring"
                ],
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
