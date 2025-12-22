"""
DAWOS Agent Data Loader
Loads public academic research data into RAG system
Document-based knowledge base for neurotherapeutic research
"""

import os
import logging
from typing import List, Dict, Any

# Initialize logger first
logger = logging.getLogger(__name__)

try:
    from .enhanced_rag import EnhancedRAG
    ENHANCED_RAG_AVAILABLE = True
except ImportError:
    ENHANCED_RAG_AVAILABLE = False
    logger.warning("Enhanced RAG not available, using Simple RAG")

# SimpleRAG removed - using EnhancedRAG only


class NeurotherapyDataLoader:
    """
    Loads public neurotherapeutic research data into ChromaDB RAG system
    
    Features:
    - Loads from actual research document files
    - Uses ChromaDB for vector storage
    - Smart document chunking with overlap
    - SentenceTransformer embeddings for semantic search
    """
    
    def __init__(self, data_directory: str = "../data/", use_enhanced: bool = True, force_reload: bool = False):
        self.data_dir = data_directory
        self.use_enhanced = use_enhanced and ENHANCED_RAG_AVAILABLE
        self.force_reload = force_reload
        
        if self.use_enhanced:
            # Use Enhanced RAG with ChromaDB
            self.rag_engine = EnhancedRAG(
                collection_name="neurotherapy_knowledge",
                persist_directory="../chroma_db"
            )
            logger.info("🚀 Using Enhanced RAG with ChromaDB vector database")
        else:
            # Fallback - create minimal RAG
            logger.error("Enhanced RAG not available - this should not happen in production")
            raise ImportError("Enhanced RAG is required for production deployment")
            
        self.loaded_documents = []
        
    def load_all_documents(self) -> EnhancedRAG:
        """
        Load all academic documents from data directory into RAG system
        Checks for existing data to avoid duplicate ingestion
        
        Returns:
            EnhancedRAG: Loaded RAG engine ready for queries
        """
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_dir):
                logger.warning(f"Data directory {self.data_dir} not found. Creating with sample data.")
                self._create_sample_data()
            
            # Check if ChromaDB already has data
            existing_stats = self.rag_engine.get_collection_stats()
            existing_chunks = existing_stats.get("total_chunks", 0)
            
            if existing_chunks > 0 and not self.force_reload:
                logger.info(f"📦 ChromaDB already contains {existing_chunks} chunks")
                logger.info("🔄 Checking if documents need updating...")
                
                # Check if we need to reload (simple check based on file count)
                documents = self._read_all_documents()
                expected_docs = len(documents)
                
                if expected_docs == 0:
                    logger.info("✅ Using existing ChromaDB data")
                    return self.rag_engine
                
                # For now, use existing data unless forced reload
                # TODO: Add file modification time checking for smart updates
                logger.info("✅ Using existing ChromaDB data (use force_reload=True to reload)")
                return self.rag_engine
            elif self.force_reload and existing_chunks > 0:
                logger.info(f"🔄 Force reload requested - clearing {existing_chunks} existing chunks")
                self.rag_engine.clear_collection()
            
            # Load all .txt files from data directory
            documents = self._read_all_documents()
            
            if not documents:
                logger.warning("No documents found. Creating sample knowledge base.")
                self._create_sample_knowledge()
                return self.rag_engine
            
            logger.info(f"📚 Loading {len(documents)} documents into empty ChromaDB...")
            
            # Ingest each document using Enhanced RAG method
            for doc in documents:
                logger.info(f"Loading {doc['filename']} into RAG system...")
                
                # Use Enhanced RAG with ChromaDB
                chunks_count = self.rag_engine.ingest_document(
                    text=doc['content'],
                    document_id=doc['filename'].replace('.txt', ''),
                    metadata={
                        'filename': doc['filename'],
                        'source': 'academic_research',
                        'domain': 'neurotherapy'
                    }
                )
                
                self.loaded_documents.append({
                    'filename': doc['filename'],
                    'size': len(doc['content']),
                    'chunks': chunks_count
                })
            
            logger.info(f"✅ Successfully loaded {len(documents)} documents into knowledge base")
            
            # Get ChromaDB statistics
            stats = self.rag_engine.get_collection_stats()
            logger.info(f"📊 ChromaDB Stats: {stats}")
            
            return self.rag_engine
            
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}")
            # Fallback to sample knowledge
            self._create_sample_knowledge()
            return self.rag_engine
    
    def _read_all_documents(self) -> List[Dict[str, str]]:
        """Read all .txt files from data directory"""
        documents = []
        
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.data_dir, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            
                        if content:  # Only add non-empty files
                            documents.append({
                                'filename': filename,
                                'content': content
                            })
                            logger.info(f"📄 Read {filename}: {len(content)} characters")
                        else:
                            logger.warning(f"⚠️ Empty file skipped: {filename}")
                            
                    except Exception as e:
                        logger.error(f"Failed to read {filename}: {str(e)}")
                        
        except FileNotFoundError:
            logger.warning(f"Data directory {self.data_dir} not found")
            
        return documents
    
    def _create_sample_data(self):
        """Create sample data directory and files for testing"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        sample_files = {
            'neurotherapy_research.txt': """
Neurotherapy research indicates that specific frequencies can influence brainwave patterns.
Alpha waves (8-12 Hz) are associated with relaxed awareness and stress reduction.
Beta waves (13-30 Hz) correlate with focused attention and cognitive processing.
Theta waves (4-8 Hz) are linked to deep meditation and emotional healing.
Delta waves (0.5-4 Hz) are associated with deep sleep and regenerative processes.
Clinical studies show that binaural beats can entrain brainwaves to desired frequencies.
""",
            'binaural_beats_studies.txt': """
Binaural beats are created when two slightly different frequencies are played in each ear.
Research by Heinrich Wilhelm Dove in 1839 first described the binaural beat phenomenon.
Modern studies show binaural beats can reduce anxiety and improve focus.
A 2018 study found that 10 Hz binaural beats reduced anxiety by 26.3%.
Gamma frequency binaural beats (40 Hz) may enhance cognitive performance.
Beta frequency beats (15-20 Hz) can improve attention and concentration.
""",
            'frequency_therapy.txt': """
Solfeggio frequencies are specific tones believed to have healing properties.
396 Hz is associated with liberation from fear and guilt.
417 Hz facilitates change and transformation.
528 Hz is known as the love frequency and DNA repair frequency.
639 Hz promotes harmonious relationships and communication.
741 Hz enhances intuition and problem-solving abilities.
852 Hz is linked to spiritual awakening and higher consciousness.
""",
            'brainwave_analysis.txt': """
EEG studies reveal distinct brainwave patterns during different mental states.
Alpha dominance occurs during relaxed wakefulness and light meditation.
Beta activity increases during focused mental tasks and problem-solving.
Theta rhythms are prominent during REM sleep and deep meditation.
Delta waves dominate during deep, dreamless sleep stages.
Gamma oscillations (30-100 Hz) are associated with consciousness and binding.
Neurofeedback training can help individuals learn to control brainwave patterns.
"""
        }
        
        for filename, content in sample_files.items():
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            logger.info(f"📝 Created sample file: {filename}")
    
    def _create_sample_knowledge(self):
        """Create minimal sample knowledge if no files available"""
        sample_knowledge = """
Alpha waves (8-12 Hz) promote relaxation and reduce stress.
Beta waves (13-30 Hz) enhance focus and cognitive performance.
Theta waves (4-8 Hz) facilitate deep meditation and healing.
Delta waves (0.5-4 Hz) support deep sleep and recovery.
Binaural beats can entrain brainwaves to desired frequencies.
Solfeggio frequencies have specific therapeutic applications.
"""
        
        # Use ChromaDB ingest method
        self.rag_engine.ingest_document(
            text=sample_knowledge,
            document_id="sample_knowledge",
            metadata={'source': 'fallback', 'domain': 'neurotherapy'}
        )
        logger.info("📚 Created minimal sample knowledge base")
    
    def get_loaded_documents_info(self) -> List[Dict[str, Any]]:
        """Get information about loaded documents"""
        return self.loaded_documents
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get ChromaDB vector store statistics"""
        stats = self.rag_engine.get_collection_stats()
        return {
            'total_chunks': stats.get('total_chunks', 0),
            'loaded_documents': len(self.loaded_documents),
            'data_directory': self.data_dir,
            'database_type': 'ChromaDB'
        }


# Global RAG engine instance (will be loaded at startup)
loaded_rag_engine: EnhancedRAG = None


def initialize_knowledge_base(data_dir: str = "../data/", force_reload: bool = False) -> EnhancedRAG:
    """
    Initialize the global knowledge base
    Called at application startup
    
    Args:
        data_dir: Directory containing research documents
        force_reload: If True, clears existing data and reloads from files
    """
    global loaded_rag_engine
    
    logger.info("🚀 Initializing DAWOS Agent Knowledge Base...")
    
    data_loader = NeurotherapyDataLoader(data_dir, force_reload=force_reload)
    loaded_rag_engine = data_loader.load_all_documents()
    
    # Log statistics
    stats = data_loader.get_vector_store_stats()
    logger.info(f"📊 Knowledge Base Stats: {stats}")
    
    return loaded_rag_engine


def get_knowledge_base() -> EnhancedRAG:
    """Get the loaded RAG engine"""
    global loaded_rag_engine
    
    if loaded_rag_engine is None:
        logger.warning("Knowledge base not initialized. Initializing now...")
        loaded_rag_engine = initialize_knowledge_base()
    
    return loaded_rag_engine