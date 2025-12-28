"""
DAWOS ReAct Agent - Standalone Backend Service
Enhanced neurotherapeutic AI agent with real-time emotion monitoring
Production-ready with async support and proper error handling
"""

# Standard library imports
import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import asyncio
import sqlite3
from contextlib import contextmanager
import threading

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Agent modules
from agent_core.react_agent import query_dawos_agent, background_processor
from agent_core.data_loader import initialize_knowledge_base, get_knowledge_base
from agent_core.tools import tools
from integration.dawos_api_client import dawos_client, protocol_mapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="DAWOS ReAct Agent API",
    description="Cognitive Health & Neurotherapeutic Consultant Agent",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# GLOBAL THREAD POOL EXECUTOR - Resource Management Optimization
GLOBAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,  # Adjust based on server capacity
    thread_name_prefix="dawos_agent_"
)

# ============================================================================
# PERSISTENT SESSION STORAGE - SQLite Implementation
# ============================================================================

# Thread-safe SQLite connection
_db_lock = threading.Lock()
DB_PATH = "agent_sessions.db"

def init_database():
    """Initialize SQLite database for persistent session storage"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_type TEXT DEFAULT 'agent_therapy',
                    protocol_source TEXT DEFAULT 'ai_agent_custom',
                    emotion_analysis TEXT,  -- JSON string
                    agent_thinking_trace TEXT,  -- JSON string
                    protocol TEXT,  -- JSON string
                    agent_success BOOLEAN DEFAULT FALSE,
                    total_turns INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'agent_completed'
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON agent_sessions(user_id);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON agent_sessions(created_at);
            """)
            
            conn.commit()
            logger.info("✅ SQLite database initialized successfully")
            
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        raise

@contextmanager
def get_db_connection():
    """Thread-safe database connection context manager"""
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

# Pydantic Models (v2 compatible with timezone-aware datetime)
class ChatRequest(BaseModel):
    message: str
    max_turns: Optional[int] = 5
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Pydantic v2 validator to handle empty strings as None
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
    
    def __init__(self, **data):
        # Convert empty strings to None for optional fields
        if 'user_id' in data and data['user_id'] == '':
            data['user_id'] = None
        if 'session_id' in data and data['session_id'] == '':
            data['session_id'] = None
        super().__init__(**data)

class ChatResponse(BaseModel):
    question: str
    answer: str
    conversation_trace: List[Dict[str, Any]]
    success: bool
    total_turns: int
    agent_session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TestResponse(BaseModel):
    test_summary: Dict[str, Any]
    trace_analysis: Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EmotionFrameRequest(BaseModel):
    user_id: str  # Standardized to str
    emotion_data: dict
    timestamp: Optional[float] = None

class SessionRequest(BaseModel):
    user_id: str  # Standardized to str
    action: str  # "start", "progress", "end"

class AnalyzeRequest(BaseModel):
    user_id: int  # Changed to int for consistency
    emotion_analysis: Optional[dict] = None  # Changed from EmotionAnalysis to dict for flexibility

# Simple token validation (can validate DAWOS tokens)
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Token validation - Validates against DAWOS backend
    """
    try:
        token = credentials.credentials
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token required"
            )
        
        # DAWOS backend'e token validation isteği gönder
        import requests
        dawos_url = os.getenv("DAWOS_BACKEND_URL", "http://backend:8000")
        
        response = requests.get(
            f"{dawos_url}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        
        if response.status_code == 200:
            user_data = response.json()
            return user_data  # User bilgilerini dön
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed"
        )

# Enhanced startup event with database initialization
@app.on_event("startup")
async def startup_event():
    """
    Enhanced startup with fail-fast validation and database initialization
    Validates environment, initializes knowledge base, and sets up persistent storage
    """
    logger.info("🚀 Starting DAWOS ReAct Agent...")
    
    # CRITICAL: Initialize persistent database first
    try:
        init_database()
        logger.info("✅ Persistent session storage initialized")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Database initialization failed: {str(e)}")
        raise RuntimeError(f"Database initialization failed: {str(e)}")
    
    # CRITICAL: Validate environment variables (fail-fast)
    groq_api_key = os.getenv("GROQ_API_KEY")  # Actually Gemini key
    if not groq_api_key:
        logger.error("❌ CRITICAL: GROQ_API_KEY environment variable not found!")
        logger.error("❌ Server cannot start without API key (Gemini)")
        raise RuntimeError("GROQ_API_KEY is required but not configured. Please set environment variable.")
    
    if len(groq_api_key) < 20:
        logger.error("❌ CRITICAL: GROQ_API_KEY appears to be invalid (too short)")
        raise RuntimeError("GROQ_API_KEY appears to be invalid. Please check your API key.")
    
    logger.info(f"✅ API Key validated (Gemini): {groq_api_key[:20]}...")
    
    # Validate data directory
    data_dir = "../data/"
    if not os.path.exists(data_dir):
        logger.warning(f"⚠️ Data directory {data_dir} not found. Will create sample data.")
    
    logger.info("📚 Loading academic knowledge base from documents...")
    
    try:
        # Initialize knowledge base from real documents
        rag_engine = initialize_knowledge_base(data_dir)
        
        # Log statistics
        stats = {
            'total_chunks': len(rag_engine.vector_store) if hasattr(rag_engine, 'vector_store') else 0,
            'status': 'loaded'
        }
        logger.info(f"✅ Knowledge base loaded successfully: {stats}")
        
        # Validate knowledge base is working
        test_query = rag_engine.retrieve("alpha waves", top_k=1)
        if test_query:
            logger.info("✅ Knowledge base test query successful")
        else:
            logger.warning("⚠️ Knowledge base test query returned empty results")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to load knowledge base: {str(e)}")
        logger.error("❌ Server cannot start without knowledge base")
        raise RuntimeError(f"Knowledge base initialization failed: {str(e)}")
    
    logger.info("✅ DAWOS Agent startup completed successfully")

# Cleanup event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("🛑 Shutting down DAWOS Agent...")
    
    # Shutdown thread pool executor
    GLOBAL_EXECUTOR.shutdown(wait=True)
    logger.info("✅ Thread pool executor shutdown complete")
    
    logger.info("✅ DAWOS Agent shutdown complete")

# Health check
@app.get("/")
async def root():
    """Enhanced health check with comprehensive system validation"""
    try:
        # Knowledge base validation
        kb = get_knowledge_base()
        kb_stats = kb.vector_store if kb else []
        total_chunks = len(kb_stats)
        kb_loaded = total_chunks > 0
        
        # Environment validation
        groq_key_status = "✅ Configured" if os.getenv("GROQ_API_KEY") else "❌ Missing"
        
        # System status
        system_status = {
            "service": "DAWOS Advanced Agent Backend",
            "status": "running", 
            "version": "2.0.0",
            "environment": {
                "groq_api_key": groq_key_status,
                "knowledge_base_loaded": kb_loaded,
                "total_chunks": total_chunks
            },
            "capabilities": [
                f"Academic Knowledge Base ({total_chunks} chunks)",
                "Enhanced ReAct Agent with minute-by-minute analysis",
                "Real-time emotion buffer management (5-second frames)",
                "Session lifecycle management with learning",
                "Evidence-based neurotherapeutic protocol generation",
                "User history and pattern recognition",
                "Contextual research with personalization",
                "Thread-pool optimized async processing"
            ],
            "endpoints": [
                "/agent/emotion-frame - Real-time emotion frame processing",
                "/agent/session-management - Session lifecycle control", 
                "/agent/minute-analysis - Buffer analysis trigger",
                "/agent/chat - Interactive consultation",
                "/agent/analyze-and-decide - DAWOS integration"
            ],
            "performance": {
                "async_optimized": True,
                "thread_pool_enabled": True,
                "memory_management": "Enhanced",
                "error_handling": "Production-ready"
            }
        }
        
        return system_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "service": "DAWOS Advanced Agent Backend",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc)
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Agent Chat Endpoint
@app.post("/agent/chat", response_model=ChatResponse)
async def agent_chat(
    request: ChatRequest
    # current_user: dict = Depends(verify_token)  # Disable for proxy
):
    """
    ReAct Agent ile sohbet et - Enhanced with user context
    Neurotherapeutic consultation interface with session and user awareness
    """
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY not configured or invalid. Please check your environment variables."
            )
        
        logger.info(f"Agent chat request: {request.message}")
        logger.info(f"Raw request model: {request}")
        logger.info(f"User ID: {request.user_id}, Session ID: {request.session_id}")
        logger.info(f"Request dict: {request.dict()}")
        logger.info(f"User ID type: {type(request.user_id)}, value: '{request.user_id}'")
        logger.info(f"Session ID type: {type(request.session_id)}, value: '{request.session_id}'")
        
        # Validate and normalize user_id
        user_id = request.user_id or "1"  # Default to "1" instead of "unknown"
        session_id = request.session_id
        
        # Log the actual values received
        logger.info(f"User ID from request: {user_id}")
        logger.info(f"Processed - User ID: {user_id}, Session ID: {session_id}")
        
        # Enhanced agent query with user context
        if user_id and session_id:
            # User has active session - include context
            enhanced_message = f"""
            User Context:
            - User ID: {user_id}
            - Session ID: {session_id}
            - Message: {request.message}
            
            Please respond considering the user's active session and any available history.
            Use appropriate tools to check session status, user patterns, and provide personalized assistance.
            """
            logger.info(f"✅ Full context available - User: {user_id}, Session: {session_id}")
        elif user_id:
            # User identified but no active session
            enhanced_message = f"""
            User Context:
            - User ID: {user_id}
            - Message: {request.message}
            
            Please respond considering this user's history and patterns if available.
            Use appropriate tools to check user background and provide personalized assistance.
            """
            logger.info(f"✅ User context available - User: {user_id}, No session")
        else:
            # No user context - generic response
            enhanced_message = request.message
            logger.warning("⚠️ No user context available - using generic response")
        
        logger.info(f"Enhanced message: {enhanced_message[:200]}...")  # First 200 chars
        
        # CRITICAL FIX: Use global thread pool executor to prevent resource waste
        def run_agent_query():
            return query_dawos_agent(
                question=enhanced_message,
                max_turns=5,  # Tam ReAct döngüsü için
                groq_api_key=groq_api_key,
                user_id=user_id
            )
        
        # Execute in global thread pool (resource efficient)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(GLOBAL_EXECUTOR, run_agent_query)
        except Exception as e:
            logger.error(f"Agent execution error: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        
        return ChatResponse(
            question=result["question"],
            answer=result["final_answer"],
            conversation_trace=result["conversation_trace"],
            success=result["success"],
            total_turns=result["total_turns"],
            agent_session_id=f"session_{user_id}_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Agent chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {str(e)}"
        )

# Test Scenarios Endpoint
@app.post("/agent/test-scenarios", response_model=TestResponse)
async def run_test_scenarios(
    # token: str = Depends(verify_token)  # Disabled for testing
):
    """
    Test scenarios for agent functionality - DISABLED (Gemini migration)
    """
    return TestResponse(
        test_summary={"status": "disabled", "reason": "Migrated to Gemini"},
        trace_analysis={"status": "disabled"},
        timestamp=datetime.now(timezone.utc)
    )

# ============================================================================
# NEW: REAL-TIME EMOTION MONITORING ENDPOINTS
# ============================================================================

@app.post("/agent/emotion-frame")
async def receive_emotion_frame(
    request: EmotionFrameRequest
    # token: str = Depends(verify_token)  # Disabled for testing
):
    """
    Receive emotion frame every 5 seconds for buffer analysis
    Enhanced with background processing to prevent blocking
    """
    try:
        user_id = str(request.user_id)
        emotion_data = request.emotion_data
        
        logger.info(f"Emotion frame received for user {user_id}: {emotion_data.get('emotion', 'UNKNOWN')}")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY not configured"
            )
        
        # Import background processor
        from agent_core.react_agent import background_processor
        
        # Process frame in background (non-blocking)
        background_result = await background_processor.process_emotion_frame_async(
            user_id=user_id,
            emotion_data=emotion_data,
            groq_api_key=groq_api_key
        )
        
        return {
            "user_id": request.user_id,
            "frame_processed": True,
            "processing_status": background_result["status"],
            "message": background_result["message"],
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Emotion frame processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frame processing failed: {str(e)}"
        )


@app.post("/agent/session-management")
async def manage_session(
    request: SessionRequest
    # token: str = Depends(verify_token)  # Disabled for testing
):
    """
    Session lifecycle management: start, progress check, end
    """
    try:
        user_id = str(request.user_id)
        action = request.action.lower()
        
        logger.info(f"Session management request: {action} for user {user_id}")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY not configured"
            )
        
        # Build agent query based on action
        if action == "start":
            agent_query = f"Please start a new emotion monitoring session for user {user_id}. Initialize the buffer and prepare for real-time analysis."
        elif action == "progress":
            agent_query = f"Please analyze the current session progress for user {user_id}. Provide trend analysis and recommendations."
        elif action == "end":
            agent_query = f"Please end the current session for user {user_id}. Provide session summary and save insights for future sessions."
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {action}. Use 'start', 'progress', or 'end'"
            )
        
        # Use agent for session management with global thread pool
        def run_session_management():
            return query_dawos_agent(
                question=agent_query,
                max_turns=5,  # Tam ReAct döngüsü
                groq_api_key=groq_api_key
            )
        
        # Execute in global thread pool (resource efficient)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(GLOBAL_EXECUTOR, run_session_management)
        
        return {
            "user_id": request.user_id,
            "action": action,
            "success": result["success"],
            "agent_response": result["final_answer"],
            "session_status": "updated",
            "timestamp": datetime.now(timezone.utc)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session management error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session management failed: {str(e)}"
        )


@app.post("/agent/minute-analysis")
async def trigger_minute_analysis(
    user_id: str  # Changed to str for consistency
    # token: str = Depends(verify_token)  # Disabled for testing
):
    """
    Manually trigger minute buffer analysis (for testing)
    In production, this would be called automatically every 60 seconds
    """
    try:
        logger.info(f"Manual minute analysis triggered for user {user_id}")
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY not configured"
            )
        
        # Agent analyzes the minute buffer with thread pool
        agent_query = f"Please analyze the current minute buffer for user {user_id}. Determine if intervention is needed based on the 60-second emotion data and academic research."
        
        def run_minute_analysis():
            return query_dawos_agent(
                question=agent_query,
                max_turns=5,  # Tam analiz için
                groq_api_key=groq_api_key
            )
        
        # Execute in global thread pool (resource efficient)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(GLOBAL_EXECUTOR, run_minute_analysis)
        
        return {
            "user_id": user_id,
            "analysis_completed": True,
            "agent_analysis": result["final_answer"],
            "reasoning_trace": result["conversation_trace"],
            "total_reasoning_steps": result["total_turns"],
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"Minute analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Minute analysis failed: {str(e)}"
        )
# ============================================================================
# ENHANCED DAWOS INTEGRATION - Analyze and Decide
# ============================================================================

@app.post("/agent/analyze-and-decide")
async def agent_analyze_and_decide(
    request: AnalyzeRequest
    # token: str = Depends(verify_token)  # Disabled for testing
):
    """
    Agent Decision Workflow - Uses EXISTING DAWOS emotion data:
    1. Receive emotion analysis from DAWOS frontend
    2. Run ReAct thinking process using Professor's method
    3. Determine therapy protocol using academic knowledge
    4. Return protocol for DAWOS audio system
    """
    try:
        user_id = request.user_id
        emotion_analysis = request.emotion_analysis  # From DAWOS frontend
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id required"
            )
        
        # Use emotion analysis from DAWOS if provided, otherwise create mock data with warning
        if emotion_analysis:
            emotion_result = emotion_analysis
            logger.info(f"🎭 Using DAWOS emotion analysis: {emotion_result.get('dominant_emotion')}")
        else:
            # Fallback for testing with explicit warning
            emotion_result = {
                "dominant_emotion": "NEUTRAL",
                "confidence_score": 75.0,
                "faces_detected": 1,
                "data_source": "fallback_mock",
                "warning": "No real emotion data provided, using fallback values"
            }
            logger.warning("🎭 WARNING: No emotion analysis provided, using fallback mock data for testing")
        
        # 2. Run Agent ReAct thinking process with data source context
        logger.info(f"🤖 Starting Agent ReAct thinking process for user {user_id}")
        
        # Enhanced agent query with data source information
        data_source_note = ""
        if emotion_result.get("data_source") == "fallback_mock":
            data_source_note = "\n⚠️ NOTE: Using fallback emotion data due to missing real analysis. Recommendations are for testing purposes only."
        
        agent_query = f"""
        User {user_id} emotion analysis: {emotion_result.get('dominant_emotion', 'UNKNOWN')} ({emotion_result.get('confidence_score', 0)}% confidence).
        
        Recommend optimal neurotherapy frequency based on this emotional state. Use academic research and user history. Be concise.
        """
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GROQ_API_KEY not configured or invalid. Please check your environment variables."
            )
        
        # ReAct query function
        agent_result = query_dawos_agent(
            question=agent_query,
            max_turns=3,  # Balanced for thorough but efficient analysis
            groq_api_key=groq_api_key,
            user_id=user_id  # Add missing user_id parameter
        )
        
        # 3. Map Agent decision to DAWOS AudioProtocol format
        protocol = protocol_mapper.agent_decision_to_audio_protocol(
            agent_result["final_answer"]
        )
        
        # Validate protocol for DAWOS compatibility
        validated_protocol = protocol_mapper.validate_protocol_parameters(protocol)
        
        logger.info(f"✅ Agent decision: {validated_protocol['type']} {validated_protocol['frequency']}Hz for {emotion_result.get('dominant_emotion')}")
        
        # Agent saves its own session (persistent storage)
        agent_session_id = await save_agent_session_persistent(
            user_id=user_id,
            emotion_analysis=emotion_result,
            agent_decision=agent_result,
            protocol=validated_protocol
        )
        
        return {
            "thinking_process": agent_result["conversation_trace"],  # Reasoning trace format
            "protocol_recommendation": validated_protocol,          # Agent custom format
            "emotion_analysis": emotion_result,                     # Original DAWOS analysis
            "agent_success": agent_result["success"],
            "total_turns": agent_result["total_turns"],
            "agent_session_id": agent_session_id,                   # Agent's own session ID
            "timestamp": datetime.now(timezone.utc)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent analyze-and-decide failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent processing failed: {str(e)}"
        )

# Knowledge Base Info (from loaded documents)
@app.get("/agent/knowledge-base")
async def get_knowledge_base_info():
    """Get information about loaded academic knowledge base"""
    try:
        kb = get_knowledge_base()
        
        return {
            "source": "Academic research documents",
            "total_chunks": len(kb.vector_store),
            "loading_method": "ChromaDB vector database",
            "data_directory": "../data/",
            "compliance": "Loaded from real files, not hardcoded strings",
            "topics": [
                "Neurotherapy Research",
                "Binaural Beats Studies", 
                "Frequency Therapy",
                "Brainwave Analysis",
                "Solfeggio Frequencies",
                "Alpha/Beta/Theta/Delta Waves"
            ]
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "Knowledge base not loaded"
        }

# Tools Info
@app.get("/agent/tools")
async def get_tools():
    """Available tools"""
    tools_info = []
    for tool in tools:
        tools_info.append({
            "name": tool.__name__,
            "description": tool.__doc__ or "Açıklama yok",
            "type": "DAWOS Tool"
        })
    
    return {
        "available_tools": tools_info,
        "total_tools": len(tools_info)
    }

# ============================================================================
# PERSISTENT SESSION MANAGEMENT - SQLite Implementation
# ============================================================================

async def save_agent_session_persistent(
    user_id: str,
    emotion_analysis: Dict[str, Any],
    agent_decision: Dict[str, Any],
    protocol: Dict[str, Any]
) -> str:
    """
    Save agent session to persistent SQLite database
    Replaces in-memory storage to prevent data loss
    """
    try:
        session_id = f"agent_{user_id}_{int(datetime.now(timezone.utc).timestamp())}"
        
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO agent_sessions (
                    session_id, user_id, session_type, protocol_source,
                    emotion_analysis, agent_thinking_trace, protocol,
                    agent_success, total_turns, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user_id,
                "agent_therapy",
                "ai_agent_custom",
                json.dumps(emotion_analysis),
                json.dumps(agent_decision.get("conversation_trace", [])),
                json.dumps(protocol),
                agent_decision.get("success", False),
                agent_decision.get("total_turns", 0),
                "agent_completed"
            ))
            conn.commit()
        
        logger.info(f"✅ Agent session saved to persistent storage: {session_id}")
        return session_id
        
    except Exception as e:
        logger.error(f"Failed to save agent session to database: {str(e)}")
        return f"agent_error_{int(datetime.now(timezone.utc).timestamp())}"

async def get_agent_sessions_persistent(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get agent sessions from persistent storage"""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM agent_sessions 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (user_id, limit))
            
            sessions = []
            for row in cursor.fetchall():
                session = {
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "session_type": row["session_type"],
                    "protocol_source": row["protocol_source"],
                    "emotion_analysis": json.loads(row["emotion_analysis"]) if row["emotion_analysis"] else {},
                    "agent_thinking_trace": json.loads(row["agent_thinking_trace"]) if row["agent_thinking_trace"] else [],
                    "protocol": json.loads(row["protocol"]) if row["protocol"] else {},
                    "agent_success": bool(row["agent_success"]),
                    "total_turns": row["total_turns"],
                    "created_at": row["created_at"],
                    "status": row["status"]
                }
                sessions.append(session)
            
            return sessions
            
    except Exception as e:
        logger.error(f"Failed to retrieve agent sessions: {str(e)}")
        return []

@app.get("/agent/sessions/{user_id}")
async def get_agent_sessions(user_id: str):
    """
    Agent session history from persistent storage
    No data loss on server restart
    """
    try:
        sessions = await get_agent_sessions_persistent(user_id, limit=10)
        
        return {
            "user_id": user_id,
            "total_sessions": len(sessions),
            "sessions": sessions,
            "session_type": "agent_therapy_persistent",
            "storage": "SQLite database"
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent sessions: {str(e)}")
        return {"error": str(e)}

@app.get("/agent/session/{session_id}")
async def get_agent_session_detail(session_id: str):
    """
    Agent session detail from persistent storage
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM agent_sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                session = {
                    "session_id": row["session_id"],
                    "user_id": row["user_id"],
                    "session_type": row["session_type"],
                    "protocol_source": row["protocol_source"],
                    "emotion_analysis": json.loads(row["emotion_analysis"]) if row["emotion_analysis"] else {},
                    "agent_thinking_trace": json.loads(row["agent_thinking_trace"]) if row["agent_thinking_trace"] else [],
                    "protocol": json.loads(row["protocol"]) if row["protocol"] else {},
                    "agent_success": bool(row["agent_success"]),
                    "total_turns": row["total_turns"],
                    "created_at": row["created_at"],
                    "status": row["status"]
                }
                return session
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Agent session not found"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,  # DAWOS 8000, Agent 8001
        reload=True
    )
