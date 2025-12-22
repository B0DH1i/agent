"""
DAWOS ReAct Agent Tools
Neurotherapeutic agent tools for academic knowledge retrieval
Uses document-based RAG system for evidence-based recommendations
"""

import json
from typing import Dict, Any
from .data_loader import get_knowledge_base


def calculator(expression: str) -> str:
    """
    Calculates the result of a mathematical expression.
    e.g. calculator: 100 * 0.15
    """
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def search_neurotherapeutic_knowledge(query: str) -> str:
    """
    Searches academic neurotherapeutic knowledge base loaded from research documents.
    Use this tool for questions about binaural beats, frequency therapy, brainwave research,
    AND for protocol recommendations based on emotional states.
    Returns RAW CONTEXT from academic papers for evidence-based analysis.
    
    Examples:
    - search_neurotherapeutic_knowledge: binaural beats definition
    - search_neurotherapeutic_knowledge: stress management alpha waves protocol
    - search_neurotherapeutic_knowledge: SAD depression treatment frequencies
    - search_neurotherapeutic_knowledge: anxiety protocol recommendations
    """
    print(f"\n🧠 [ACADEMIC KNOWLEDGE] Searching for: '{query}'")
    
    # Get the loaded RAG engine (from real documents)
    rag_engine = get_knowledge_base()
    
    # Use ChromaDB retrieve method
    result = rag_engine.retrieve(query, top_k=3)  # Get more context for complex queries
    
    if not result:
        return "No information found in academic knowledge base on this topic."
    
    # Return RAW CONTEXT for evidence-based reasoning
    return f"Academic Research Context: {result}"


def get_user_emotion_analysis(user_id: str, limit: int = 5) -> str:
    """
    Returns user's recent emotion analysis results.
    e.g. get_user_emotion_analysis: user123
    """
    try:
        # Mock data - in real implementation, will call DAWOS API
        mock_results = [
            {"timestamp": "2024-01-15 10:30", "emotion": "HAPPY", "confidence": 85.2},
            {"timestamp": "2024-01-15 10:20", "emotion": "CALM", "confidence": 78.5},
            {"timestamp": "2024-01-15 10:10", "emotion": "FOCUSED", "confidence": 82.1}
        ]
        
        result_text = "Recent Emotion Analysis Results:\n"
        for result in mock_results[:limit]:
            result_text += f"- {result['timestamp']}: {result['emotion']} ({result['confidence']}% confidence)\n"
        
        return result_text
    except Exception as e:
        return f"Could not retrieve emotion analysis data: {e}"


# recommend_therapy_protocol fonksiyonu kaldırıldı
# Agent artık sadece akademik RAG sistemini kullanacak
# Tüm protokol önerileri search_neurotherapeutic_knowledge aracından gelecek


def get_therapy_session_history(user_id: str, limit: int = 3) -> str:
    """
    Returns user's sound therapy session history.
    e.g. get_therapy_session_history: user123
    """
    try:
        # Mock data - in real implementation, will call DAWOS API
        mock_sessions = [
            {"date": "2024-01-15", "protocol": "Alpha waves", "duration": "20 min", "effectiveness": "High"},
            {"date": "2024-01-14", "protocol": "Theta meditation", "duration": "15 min", "effectiveness": "Medium"},
            {"date": "2024-01-13", "protocol": "Beta focus", "duration": "25 min", "effectiveness": "High"}
        ]
        
        result_text = "Recent Therapy Sessions:\n"
        for session in mock_sessions[:limit]:
            result_text += f"- {session['date']}: {session['protocol']} ({session['duration']}) - Effectiveness: {session['effectiveness']}\n"
        
        return result_text
    except Exception as e:
        return f"Could not retrieve therapy history: {e}"


def analyze_emotional_pattern(user_id: str, days: int = 7) -> str:
    """
    Analyzes user's emotional patterns over the last X days.
    e.g. analyze_emotional_pattern: user123
    """
    try:
        # Mock analysis - in real implementation, will call DAWOS API
        analysis = {
            "dominant_emotions": ["HAPPY", "CALM", "FOCUSED"],
            "stress_level": "Low",
            "focus_trend": "Increasing trend",
            "recommendation": "Current positive trend continues, weekly alpha therapy recommended"
        }
        
        result_text = f"Last {days} Days Emotional Analysis:\n"
        result_text += f"- Dominant Emotions: {', '.join(analysis['dominant_emotions'])}\n"
        result_text += f"- Stress Level: {analysis['stress_level']}\n"
        result_text += f"- Focus Trend: {analysis['focus_trend']}\n"
        result_text += f"- Recommendation: {analysis['recommendation']}\n"
        
        return result_text
    except Exception as e:
        return f"Could not analyze emotional patterns: {e}"


# Tool listesi - ReAct agent format
tools = [
    calculator,
    search_neurotherapeutic_knowledge,  # Ana akademik bilgi kaynağı - tüm protokoller buradan
    get_user_emotion_analysis,
    get_therapy_session_history,
    analyze_emotional_pattern
]

# Known actions dictionary - agent tools mapping
known_actions = dict([(tool.__name__, tool) for tool in tools])