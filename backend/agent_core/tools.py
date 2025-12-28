"""
DAWOS ReAct Agent Tools
Enhanced buffer management, session control, and research tools
Security-hardened and production-ready
"""

# Standard library imports
import json
import ast
import operator
from datetime import datetime
from typing import Dict, Any

# Local imports
from .data_loader import get_knowledge_base


# ============================================================================
# BASIC CALCULATION TOOL - SECURE VERSION
# ============================================================================

def calculator(expression: str) -> str:
    """
    Secure mathematical calculator using ast.literal_eval for safety.
    e.g. calculator: 100 * 0.15
    """
    try:
        import ast
        import operator
        
        # Allowed operations for security
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }
        
        def safe_eval(node):
            if isinstance(node, ast.Constant):  # Python 3.8+
                return node.value
            elif isinstance(node, ast.Num):  # Python < 3.8
                return node.n
            elif isinstance(node, ast.BinOp):
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                op = allowed_operators.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
                return op(left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = safe_eval(node.operand)
                op = allowed_operators.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
                return op(operand)
            else:
                raise ValueError(f"Unsupported node type: {type(node).__name__}")
        
        # Parse and evaluate safely
        tree = ast.parse(expression.strip(), mode='eval')
        result = safe_eval(tree.body)
        return str(result)
        
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"Error: Invalid calculation - {str(e)}"
    except Exception as e:
        return f"Error: Calculation failed - {str(e)}"


def search_neurotherapeutic_knowledge(query: str) -> str:
    """
    Searches academic neurotherapeutic knowledge base with enhanced citation formatting.
    Returns structured academic context with source references for evidence-based analysis.
    e.g. search_neurotherapeutic_knowledge: binaural beats anxiety research
    """
    try:
        print(f"\n🧠 [ACADEMIC KNOWLEDGE] Searching for: '{query}'")
        
        # Get the loaded RAG engine (from real documents)
        rag_engine = get_knowledge_base()
        
        # Use enhanced ChromaDB retrieve method with academic citations
        result = rag_engine.retrieve(query, top_k=3)
        
        if not result:
            return "Error: No information found in academic knowledge base on this topic."
        
        # The enhanced retrieve method already provides formatted output with citations
        # No need for additional truncation as it's now properly structured
        
        return f"Academic Research Results:\n{result}"
        
    except Exception as e:
        return f"Error: Knowledge search failed - {str(e)}"


# ============================================================================
# BUFFER MANAGEMENT TOOLS - Real-time Emotion Analysis
# ============================================================================

# Global buffer storage (in production, this would be Redis/Database)
user_emotion_buffers = {}
user_session_data = {}

def add_emotion_frame_to_buffer(user_id: str, emotion_data: str) -> str:
    """
    5 saniyede bir gelen emotion frame'i buffer'a ekler.
    Buffer 12 frame'e ulaştığında analiz tetiklenir.
    e.g. add_emotion_frame_to_buffer: user123 {"emotion": "ANXIETY", "confidence": 0.75, "timestamp": 1640995200}
    """
    try:
        # Parse emotion data
        try:
            frame_data = json.loads(emotion_data)
        except:
            # If string format, parse manually
            frame_data = {"emotion": emotion_data, "confidence": 0.5, "timestamp": int(datetime.now().timestamp())}
        
        # Initialize buffer if not exists
        if user_id not in user_emotion_buffers:
            user_emotion_buffers[user_id] = {
                "current_minute_buffer": [],
                "minute_summaries": [],
                "session_active": True,
                "baseline_established": False,
                "baseline_score": 0.3  # Default baseline
            }
        
        buffer = user_emotion_buffers[user_id]["current_minute_buffer"]
        buffer.append(frame_data)
        
        # If buffer reaches 12 frames (60 seconds), trigger analysis
        if len(buffer) >= 12:
            analysis_result = f"Buffer full for {user_id}. 12 frames collected over 60 seconds. Ready for minute analysis."
            return analysis_result
        
        return f"Frame added to buffer for {user_id}. Buffer size: {len(buffer)}/12 frames."
        
    except Exception as e:
        return f"Error: Frame processing failed - {str(e)}"


def analyze_minute_buffer(user_id: str) -> str:
    """
    60 saniyelik buffer'ı gelişmiş analiz ile değerlendirir:
    - Volatilite (standart sapma) hesaplama
    - Dinamik baseline eşik sistemi
    - Trend analizi ve risk değerlendirmesi
    e.g. analyze_minute_buffer: user123
    """
    try:
        if user_id not in user_emotion_buffers:
            return f"Error: No buffer found for user {user_id}. Start session first."
        
        buffer = user_emotion_buffers[user_id]["current_minute_buffer"]
        if len(buffer) < 12:
            return f"Error: Insufficient data. Buffer has {len(buffer)}/12 frames. Need full minute."
        
        # Emotion mapping for numerical analysis
        emotion_scores = {
            "HAPPY": 0.1, "CALM": 0.2, "NEUTRAL": 0.3, "FOCUSED": 0.2,
            "SLIGHT_ANXIETY": 0.4, "MILD_STRESS": 0.5, "ANXIETY": 0.6, 
            "STRESSED": 0.7, "ANGRY": 0.8, "PANIC": 0.9
        }
        
        # Extract numerical data
        scores = []
        emotions = []
        confidences = []
        
        for frame in buffer:
            emotion = frame.get("emotion", "NEUTRAL")
            confidence = frame.get("confidence", 0.5)
            score = emotion_scores.get(emotion, 0.3)
            
            scores.append(score)
            emotions.append(emotion)
            confidences.append(confidence)
        
        # Advanced statistical analysis
        avg_score = sum(scores) / len(scores)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Volatility calculation (standard deviation)
        variance = sum((x - avg_score) ** 2 for x in scores) / len(scores)
        volatility = variance ** 0.5
        
        # Dynamic baseline system
        user_data = user_emotion_buffers[user_id]
        if not user_data["baseline_established"]:
            # First 2 minutes establish baseline
            minute_count = len(user_data["minute_summaries"])
            if minute_count < 2:
                user_data["baseline_score"] = avg_score
                baseline_threshold = 0.4  # Default for first minutes
            else:
                # Calculate baseline from first 2 minutes
                baseline_scores = []
                for summary in user_data["minute_summaries"][:2]:
                    if "avg_severity" in summary:
                        baseline_scores.append(summary["avg_severity"])
                
                if baseline_scores:
                    user_data["baseline_score"] = sum(baseline_scores) / len(baseline_scores)
                    user_data["baseline_established"] = True
                    baseline_threshold = user_data["baseline_score"] + 0.2  # Dynamic threshold
                else:
                    baseline_threshold = 0.4
        else:
            # Use established baseline
            baseline_threshold = user_data["baseline_score"] + 0.2
        
        # Enhanced trend analysis
        first_quarter = sum(scores[:3]) / 3
        second_quarter = sum(scores[3:6]) / 3
        third_quarter = sum(scores[6:9]) / 3
        fourth_quarter = sum(scores[9:]) / 3
        
        quarters = [first_quarter, second_quarter, third_quarter, fourth_quarter]
        
        # Determine trend pattern
        if quarters[-1] > quarters[0] + 0.15:
            trend = "rapidly_increasing"
        elif quarters[-1] > quarters[0] + 0.05:
            trend = "increasing"
        elif quarters[-1] < quarters[0] - 0.15:
            trend = "rapidly_decreasing"
        elif quarters[-1] < quarters[0] - 0.05:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Risk assessment
        high_volatility = volatility > 0.15
        above_baseline = avg_score > baseline_threshold
        rapid_change = trend in ["rapidly_increasing", "rapidly_decreasing"]
        
        # Decision logic with enhanced criteria
        intervention_needed = (above_baseline and trend in ["increasing", "rapidly_increasing"]) or \
                            (high_volatility and avg_score > 0.5) or \
                            rapid_change
        
        # Determine dominant emotion
        dominant_emotion = max(set(emotions), key=emotions.count)
        
        analysis_result = {
            "minute": len(user_data["minute_summaries"]) + 1,
            "dominant_emotion": dominant_emotion,
            "avg_severity": round(avg_score, 3),
            "volatility": round(volatility, 3),
            "trend": trend,
            "confidence": round(avg_confidence, 3),
            "baseline_threshold": round(baseline_threshold, 3),
            "intervention_needed": intervention_needed,
            "risk_factors": {
                "high_volatility": high_volatility,
                "above_baseline": above_baseline,
                "rapid_change": rapid_change
            }
        }
        
        # Store analysis for baseline calculation
        user_data["minute_summaries"].append(analysis_result)
        
        # Clear buffer for next minute
        user_data["current_minute_buffer"] = []
        
        result_text = f"Minute {analysis_result['minute']} Enhanced Analysis:\n"
        result_text += f"- Dominant Emotion: {dominant_emotion} (severity: {analysis_result['avg_severity']})\n"
        result_text += f"- Volatility: {analysis_result['volatility']} (emotional stability indicator)\n"
        result_text += f"- Trend: {trend} over 60 seconds\n"
        result_text += f"- Baseline Threshold: {analysis_result['baseline_threshold']}\n"
        result_text += f"- Risk Factors: Volatility={high_volatility}, Above_Baseline={above_baseline}, Rapid_Change={rapid_change}\n"
        result_text += f"- Intervention Needed: {'YES' if intervention_needed else 'NO'}"
        
        return result_text
        
    except Exception as e:
        return f"Error: Minute analysis failed - {str(e)}"


def record_minute_summary(user_id: str, decision_data: str) -> str:
    """
    Her dakikanın sonunda özet ve karar kaydeder.
    e.g. record_minute_summary: user123 {"decision": "alpha_protocol", "reasoning": "increasing_anxiety_trend"}
    """
    try:
        # Parse decision data
        try:
            decision = json.loads(decision_data)
        except:
            decision = {"decision": decision_data, "reasoning": "agent_decision"}
        
        if user_id not in user_emotion_buffers:
            return f"Error: No active session for user {user_id}"
        
        minute_num = len(user_emotion_buffers[user_id]["minute_summaries"]) + 1
        
        summary = {
            "minute": minute_num,
            "timestamp": datetime.now().isoformat(),
            "decision": decision.get("decision", "continue_monitoring"),
            "reasoning": decision.get("reasoning", "routine_analysis"),
            "agent_confidence": decision.get("confidence", 0.8)
        }
        
        user_emotion_buffers[user_id]["minute_summaries"].append(summary)
        
        return f"Minute {minute_num} summary recorded: {decision.get('decision')} (Reason: {decision.get('reasoning')})"
        
    except Exception as e:
        return f"Error: Summary recording failed - {str(e)}"


# ============================================================================
# SESSION MANAGEMENT TOOLS
# ============================================================================

def start_emotion_monitoring_session(user_id: str) -> str:
    """
    Yeni emotion monitoring session başlatır.
    e.g. start_emotion_monitoring_session: user123
    """
    try:
        import uuid
        
        session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
        
        # Initialize session data
        user_session_data[user_id] = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "status": "MONITORING",
            "total_minutes": 0,
            "interventions": []
        }
        
        # Initialize emotion buffer with enhanced features
        user_emotion_buffers[user_id] = {
            "current_minute_buffer": [],
            "minute_summaries": [],
            "session_active": True,
            "baseline_established": False,
            "baseline_score": 0.3
        }
        
        return f"Session started for {user_id}. Session ID: {session_id}. Status: MONITORING. Ready for emotion frames (5-second intervals)."
        
    except Exception as e:
        return f"Error: Session start failed - {str(e)}"


def get_session_progress_trend(user_id: str) -> str:
    """
    Mevcut session'ın progress trend'ini analiz eder.
    e.g. get_session_progress_trend: user123
    """
    try:
        if user_id not in user_emotion_buffers:
            return f"No active session for user {user_id}"
        
        summaries = user_emotion_buffers[user_id]["minute_summaries"]
        if len(summaries) < 2:
            return f"Insufficient data. Need at least 2 minutes for trend analysis. Current: {len(summaries)} minutes."
        
        # Analyze trend over minutes
        decisions = [s["decision"] for s in summaries]
        recent_decisions = decisions[-3:] if len(decisions) >= 3 else decisions
        
        intervention_count = sum(1 for d in decisions if "protocol" in d.lower() or "intervention" in d.lower())
        monitoring_count = len(decisions) - intervention_count
        
        # Determine overall trend
        if intervention_count > monitoring_count:
            trend = "INTERVENTION_HEAVY"
        elif intervention_count == 0:
            trend = "STABLE_MONITORING"
        else:
            trend = "MIXED_RESPONSE"
        
        result_text = f"Session Progress Analysis ({len(summaries)} minutes):\n"
        result_text += f"- Total Minutes: {len(summaries)}\n"
        result_text += f"- Interventions: {intervention_count}\n"
        result_text += f"- Monitoring: {monitoring_count}\n"
        result_text += f"- Trend: {trend}\n"
        result_text += f"- Recent Decisions: {', '.join(recent_decisions[-3:])}"
        
        return result_text
        
    except Exception as e:
        return f"Error analyzing session progress: {str(e)}"


def end_session_with_summary(user_id: str) -> str:
    """
    Session'ı sonlandırır ve özet analiz yapar.
    e.g. end_session_with_summary: user123
    """
    try:
        from datetime import datetime
        
        if user_id not in user_emotion_buffers or user_id not in user_session_data:
            return f"No active session found for user {user_id}"
        
        session_data = user_session_data[user_id]
        buffer_data = user_emotion_buffers[user_id]
        
        # Calculate session metrics
        total_minutes = len(buffer_data["minute_summaries"])
        interventions = [s for s in buffer_data["minute_summaries"] if "protocol" in s["decision"].lower()]
        
        # Session effectiveness (simplified)
        effectiveness_score = 0.8 if len(interventions) > 0 else 0.6
        
        # Create session summary
        session_summary = {
            "session_id": session_data["session_id"],
            "user_id": user_id,
            "duration_minutes": total_minutes,
            "total_interventions": len(interventions),
            "effectiveness_score": effectiveness_score,
            "end_time": datetime.now().isoformat(),
            "status": "COMPLETED"
        }
        
        # Save to user history (simplified)
        if "session_history" not in user_session_data:
            user_session_data["session_history"] = {}
        if user_id not in user_session_data["session_history"]:
            user_session_data["session_history"][user_id] = []
        
        user_session_data["session_history"][user_id].append(session_summary)
        
        # Clean up active session
        del user_session_data[user_id]
        del user_emotion_buffers[user_id]
        
        result_text = f"Session completed for {user_id}:\n"
        result_text += f"- Duration: {total_minutes} minutes\n"
        result_text += f"- Interventions: {len(interventions)}\n"
        result_text += f"- Effectiveness: {effectiveness_score:.1f}/1.0\n"
        result_text += f"- Status: COMPLETED"
        
        return result_text
        
    except Exception as e:
        return f"Error ending session: {str(e)}"


# ============================================================================
# USER HISTORY TOOLS
# ============================================================================

def get_user_session_history(user_id: str, limit: int = 5) -> str:
    """
    Kullanıcının session geçmişini getirir.
    e.g. get_user_session_history: user123
    """
    try:
        if "session_history" not in user_session_data or user_id not in user_session_data["session_history"]:
            return f"No session history found for user {user_id}. This appears to be their first session."
        
        history = user_session_data["session_history"][user_id]
        recent_sessions = history[-limit:] if len(history) > limit else history
        
        result_text = f"Session History for {user_id} (Last {len(recent_sessions)} sessions):\n"
        
        for session in recent_sessions:
            result_text += f"- {session['end_time'][:10]}: {session['duration_minutes']}min, "
            result_text += f"{session['total_interventions']} interventions, "
            result_text += f"effectiveness: {session['effectiveness_score']:.1f}\n"
        
        # Calculate averages
        avg_duration = sum(s['duration_minutes'] for s in recent_sessions) / len(recent_sessions)
        avg_effectiveness = sum(s['effectiveness_score'] for s in recent_sessions) / len(recent_sessions)
        
        result_text += f"\nAverages: {avg_duration:.1f}min duration, {avg_effectiveness:.2f} effectiveness"
        
        return result_text
        
    except Exception as e:
        return f"Error retrieving session history: {str(e)}"


def get_user_problem_patterns(user_id: str) -> str:
    """
    Kullanıcının problem pattern'larını analiz eder.
    e.g. get_user_problem_patterns: user123
    """
    try:
        if "session_history" not in user_session_data or user_id not in user_session_data["session_history"]:
            return f"No historical data for pattern analysis. User {user_id} appears to be new."
        
        history = user_session_data["session_history"][user_id]
        
        # Analyze patterns (simplified)
        total_sessions = len(history)
        high_intervention_sessions = sum(1 for s in history if s['total_interventions'] > 2)
        avg_effectiveness = sum(s['effectiveness_score'] for s in history) / total_sessions
        
        # Determine patterns
        patterns = []
        if high_intervention_sessions / total_sessions > 0.6:
            patterns.append("frequent_emotional_fluctuations")
        if avg_effectiveness > 0.7:
            patterns.append("responds_well_to_interventions")
        if avg_effectiveness < 0.5:
            patterns.append("intervention_resistance")
        
        result_text = f"Problem Patterns for {user_id}:\n"
        result_text += f"- Total Sessions: {total_sessions}\n"
        result_text += f"- High-Intervention Sessions: {high_intervention_sessions}/{total_sessions}\n"
        result_text += f"- Average Effectiveness: {avg_effectiveness:.2f}\n"
        result_text += f"- Identified Patterns: {', '.join(patterns) if patterns else 'stable_baseline'}"
        
        return result_text
        
    except Exception as e:
        return f"Error analyzing problem patterns: {str(e)}"


# ============================================================================
# ENHANCED RAG RESEARCH TOOLS
# ============================================================================

def search_contextual_research(query_and_context: str) -> str:
    """
    Kullanıcı bağlamında akademik araştırma yapar.
    e.g. search_contextual_research: {"query": "early anxiety intervention", "user_context": {"history": "responds_well_to_alpha"}}
    """
    try:
        import json
        
        # Parse query and context
        try:
            data = json.loads(query_and_context)
            query = data.get("query", query_and_context)
            user_context = data.get("user_context", {})
        except:
            query = query_and_context
            user_context = {}
        
        # Get the loaded RAG engine (from real documents)
        rag_engine = get_knowledge_base()
        
        # Enhanced query with context
        if user_context:
            context_info = " ".join([f"{k}: {v}" for k, v in user_context.items()])
            enhanced_query = f"{query} considering user context: {context_info}"
        else:
            enhanced_query = query
        
        # Use ChromaDB retrieve method with more results for context
        result = rag_engine.retrieve(enhanced_query, top_k=5)
        
        if not result:
            return f"No contextual research found for: {query}"
        
        return f"Contextual Research Results for '{query}':\n{result}"
        
    except Exception as e:
        return f"Error in contextual research: {str(e)}"


def find_optimal_intervention_research(emotion_and_profile: str) -> str:
    """
    Spesifik emotion pattern için optimal müdahale araştırması.
    e.g. find_optimal_intervention_research: {"emotion_pattern": "increasing_anxiety", "user_profile": {"response_history": "positive_alpha"}}
    """
    try:
        import json
        
        # Parse input
        try:
            data = json.loads(emotion_and_profile)
            emotion_pattern = data.get("emotion_pattern", "anxiety")
            user_profile = data.get("user_profile", {})
        except:
            emotion_pattern = emotion_and_profile
            user_profile = {}
        
        # Get the loaded RAG engine
        rag_engine = get_knowledge_base()
        
        # Build research query
        research_queries = [
            f"{emotion_pattern} intervention protocols",
            f"{emotion_pattern} frequency therapy research",
            f"optimal timing {emotion_pattern} treatment"
        ]
        
        combined_results = []
        for query in research_queries:
            result = rag_engine.retrieve(query, top_k=2)
            if result:
                combined_results.append(result)
        
        if not combined_results:
            return f"No intervention research found for emotion pattern: {emotion_pattern}"
        
        research_text = f"Optimal Intervention Research for '{emotion_pattern}':\n"
        research_text += "\n---\n".join(combined_results)
        
        return research_text
        
    except Exception as e:
        return f"Error finding intervention research: {str(e)}"


# ============================================================================
# UPDATED TOOL LIST
# ============================================================================

# Tool listesi - ReAct agent format
tools = [
    calculator,
    search_neurotherapeutic_knowledge,  # Original RAG search
    
    # NEW: Buffer Management Tools
    add_emotion_frame_to_buffer,
    analyze_minute_buffer,
    record_minute_summary,
    
    # NEW: Session Management Tools  
    start_emotion_monitoring_session,
    get_session_progress_trend,
    end_session_with_summary,
    
    # NEW: User History Tools
    get_user_session_history,
    get_user_problem_patterns,
    
    # NEW: Enhanced Research Tools
    search_contextual_research,
    find_optimal_intervention_research
]

# Known actions dictionary - agent tools mapping
known_actions = dict([(tool.__name__, tool) for tool in tools])
