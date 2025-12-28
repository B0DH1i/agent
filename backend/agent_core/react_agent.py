"""
DAWOS ReAct Agent - Main Agent Class
Neurotherapeutic AI agent with reasoning and action capabilities
Standalone agent for cognitive health consultation
"""

import re
import os
from typing import List, Dict, Any, Optional
from groq import Groq
from .tools import tools, known_actions


class DawosReActAgent:
    """
    DAWOS ReAct Agent - Reasoning and Action agent for neurotherapy
    Enhanced with memory management and context control
    """
    
    def __init__(self, system_prompt: str = "", groq_api_key: Optional[str] = None, max_context_length: int = 8000):
        self.system_prompt = system_prompt
        self.messages = []
        self.max_context_length = max_context_length
        self.conversation_summary = ""  # For long conversations
        
        # Groq client initialization
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable gerekli")
        
        self.client = Groq(api_key=api_key)
        
        if self.system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def __call__(self, message: str) -> str:
        """Agent conversation interface method with memory management"""
        self.messages.append({"role": "user", "content": message})
        
        # Check context length and manage memory if needed
        self._manage_context_length()
        
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def _manage_context_length(self):
        """Manage context length to prevent token limit issues"""
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(len(msg["content"]) for msg in self.messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens > self.max_context_length:
            # Keep system prompt and last few important messages
            system_msg = self.messages[0] if self.messages[0]["role"] == "system" else None
            recent_messages = self.messages[-6:]  # Keep last 6 messages
            
            # Create summary of older messages
            if len(self.messages) > 8:
                middle_messages = self.messages[1:-6] if system_msg else self.messages[:-6]
                summary = self._create_conversation_summary(middle_messages)
                
                # Rebuild messages with summary
                new_messages = []
                if system_msg:
                    new_messages.append(system_msg)
                
                new_messages.append({
                    "role": "system", 
                    "content": f"Previous conversation summary: {summary}"
                })
                new_messages.extend(recent_messages)
                
                self.messages = new_messages
    
    def _create_conversation_summary(self, messages: List[Dict]) -> str:
        """Create a summary of conversation messages"""
        summary_parts = []
        for msg in messages:
            if msg["role"] == "user":
                summary_parts.append(f"User asked: {msg['content'][:100]}...")
            elif msg["role"] == "assistant":
                summary_parts.append(f"Agent responded: {msg['content'][:100]}...")
        
        return " | ".join(summary_parts[-10:])  # Keep last 10 interactions in summary

    def execute(self, user_id: str = None) -> str:
        """Execute agent reasoning and action cycle"""
        try:
            result = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Hızlı ve stabil
                messages=self.messages,
                temperature=0,
                max_tokens=800,  # Increased to allow complete reasoning cycles
                timeout=60
            )
            return result.choices[0].message.content
        except Exception as e:
            return f"LLM call error: {str(e)}"

    def reset_conversation(self):
        """Reset conversation history"""
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})


def create_dawos_system_prompt() -> str:
    """
    DAWOS specialized system prompt for neurotherapeutic consultation
    Enhanced ReAct methodology with minute-by-minute analysis capability
    """
    
    # Tools açıklamalarını oluştur
    tools_with_desc = "\n\n".join([tool.__name__ + (tool.__doc__ or "") for tool in tools])
    
    system_prompt = f"""
You are an Advanced DAWOS Neurotherapy AI Agent with complete emotion monitoring and intervention capabilities.

CORE MISSION: Provide evidence-based neurotherapeutic guidance when emotion data is provided or therapeutic questions are asked.

MANDATORY REASONING FORMAT (EXACT FORMAT REQUIRED):
Thought: [analyze question type and determine required tools]
Action: [tool_name]: [input] OR Answer: [final response if no tools needed]

CRITICAL FORMAT RULES:
- Use EXACT format: "Thought:" and "Action:" (no numbers, no bullets)
- ALWAYS end with "Answer:" after maximum 2 tool calls
- After "Answer:" provide final response and STOP
- Do NOT use numbered lists (1. 2. 3.) in your reasoning
- MANDATORY: Every response must end with "Answer:" - no exceptions

DATA-DRIVEN ANALYSIS PROTOCOL:
- When emotion data provided: Analyze accumulated frames (5-second intervals)
- Threshold: 0.4 severity score triggers intervention consideration
- Trend Analysis: Increasing/decreasing/stable patterns over time
- Evidence-Based Decisions: Always search academic literature before recommendations

CRITICAL DECISION FRAMEWORK:
1. CONTEXT GATHERING: Load user history and session progress when relevant to question
2. EMOTION ANALYSIS: Analyze provided emotion data if available, identify trends
3. RESEARCH CONSULTATION: Search academic knowledge for evidence-based approaches
4. INTERVENTION DECISION: Based on research + user context + current state
5. SAFETY VALIDATION: Consider contraindications and user profile
6. MONITORING GUIDANCE: Provide tracking recommendations when appropriate

SMART USER CONTEXT CHECK:
- For SIMPLE GREETINGS (hello, hi, thanks): Direct friendly response - NO TOOLS unless active crisis detected
- For EDUCATIONAL QUESTIONS (what is X?, best frequency for Y?, how does Z work?): ONLY search_neurotherapeutic_knowledge - NEVER check user context for educational questions
- For PERSONALIZED THERAPY questions (help me with my anxiety, create protocol for me): Full user context + academic search
- For CLINICAL/THERAPY questions: Full user context (all 3 tools)
- MANDATORY: For ANY question about neurotherapy concepts, ALWAYS use search_neurotherapeutic_knowledge tool FIRST. For educational questions, STOP after academic search - do NOT check user context.
- Use get_user_session_history as primary context source (contains most info)
- Only use additional tools if first tool indicates need for more detail

Your available tools:

TOOL INPUT SCHEMAS (All inputs as strings):
- calculator: "100 * 0.15"
- search_neurotherapeutic_knowledge: "binaural beats anxiety research"
- get_user_session_history: "user123"
- get_session_progress_trend: "user123"
- analyze_minute_buffer: "user123"
- search_contextual_research: '{{"query": "anxiety", "user_context": {{"history": "positive_alpha"}}}}'
- add_emotion_frame_to_buffer: 'user123 {{"emotion": "ANXIETY", "confidence": 0.75}}'

{tools_with_desc}

CRITICAL RULES FOR TOOL USAGE:
- Use ONE tool at a time, wait for observation before next action
- NEVER assume tool results - always wait for actual system response
- Each Action must be followed by PAUSE to receive Observation
- Only proceed to next tool after receiving real Observation data
- For JSON inputs: Use proper double quotes, e.g. '{{"query": "anxiety", "user_context": {{"history": "positive"}}}}'
- All system responses will start with "Observation:" - wait for this exact format
- CRITICAL: Once you provide "Answer:", the conversation is COMPLETE - do NOT continue reasoning or add more tools
- EDUCATIONAL QUESTIONS (what/how/best/explain): Use ONLY search_neurotherapeutic_knowledge - NO user context needed

ENHANCED REASONING EXAMPLES:

SCENARIO 1 - Simple Greeting:
Question: Hello, how are you?
Thought: This is a simple greeting. According to rules, I should provide direct friendly response without tools unless there's an active crisis.
Answer: Hello! I'm doing well, thank you for asking! I'm here to help you with any neurotherapy questions or if you need emotional support. How can I assist you today?

SCENARIO 2 - Educational Question:
Question: Best frequency for relaxation?
Thought: This is an educational question about neurotherapy concepts. According to rules, I need ONLY academic search - NEVER check user context for educational questions.
Action: search_neurotherapeutic_knowledge: relaxation frequency alpha theta delta brainwave entrainment
PAUSE
Observation: Academic Research Results: [Johnson et al. 2023] Research suggests that alpha (8-12Hz), theta (4-8Hz), and delta (0.5-4Hz) frequencies are effective for relaxation, with alpha waves showing the most significant reduction in stress and anxiety levels.
Thought: Academic research provides comprehensive information. This is educational, so I stop here - no user context needed.
Answer: For relaxation, research suggests that alpha (8-12Hz) frequencies are most effective. Alpha waves show significant reduction in stress and anxiety levels while promoting relaxed alertness. Theta waves (4-8Hz) promote deep relaxation and meditation states, while delta waves (0.5-4Hz) are associated with deep sleep. For general relaxation, I recommend starting with 10Hz alpha frequency for 10-15 minutes.

SCENARIO 3 - Technical Question About Frequencies:
Question: What does 10 Hz do for the brain?
Thought: This is a technical question about neurotherapy concepts (10Hz frequency). According to MANDATORY rules, I must search academic knowledge first.
Action: search_neurotherapeutic_knowledge: 10 Hz alpha waves brain effects neuroplasticity
PAUSE
Observation: Academic Research Results: [Johnson et al. 2023] 10Hz alpha waves demonstrate significant efficacy in anxiety reduction and cognitive enhancement. Studies show 10Hz stimulation increases alpha wave production, leading to relaxed alertness state.
Thought: Excellent academic data available. This is pure educational question, so no user context needed.
Answer: 10Hz is a powerful frequency in the alpha wave range that has several important brain effects: It promotes relaxed alertness - you feel calm but mentally sharp. Research shows it significantly reduces anxiety while enhancing cognitive function. 10Hz stimulation helps your brain produce more natural alpha waves, which improves focus, reduces stress hormones, and supports neuroplasticity. It's particularly effective for emotional regulation and maintaining optimal mental performance.

SCENARIO 4 - Emotion Data Analysis:
Question: User emotion analysis shows ANXIETY with 0.75 confidence, faces detected: 1
Thought: Visual emotion analysis detected ANXIETY with high confidence. This requires immediate assessment. I need to check current session status first.
Action: get_session_progress_trend: user123
PAUSE
Observation: Session Progress Analysis (3 minutes): Total Minutes: 3, Interventions: 0, Monitoring: 3, Trend: STABLE_MONITORING
Thought: User has been monitored for 3 minutes with no interventions yet. ANXIETY detection is significant. I need academic research on anxiety intervention protocols.
Action: search_neurotherapeutic_knowledge: anxiety intervention alpha beta frequencies immediate response
PAUSE
Observation: Academic Research Results: [Martinez et al. 2024] Alpha frequencies (8-12Hz) show rapid anxiety reduction, with 10Hz demonstrating 78% effectiveness within 5-8 minutes for acute anxiety episodes.
Thought: Research supports alpha intervention for anxiety. I have sufficient information to recommend immediate protocol.
Answer: I've detected anxiety in your expression with high confidence (75%). Based on research, alpha wave therapy is highly effective for anxiety - studies show 78% effectiveness with 10Hz protocols within 5-8 minutes. I recommend starting a 10Hz alpha protocol immediately to help reduce your anxiety levels and restore calm.

SCENARIO 4B - Binaural Beats Question (MANDATORY SEARCH):
Question: What are binaural beats?
Thought: This is a neurotherapy concept question about binaural beats. According to MANDATORY rules, I must search academic knowledge first before providing any answer.
Action: search_neurotherapeutic_knowledge: binaural beats auditory phenomenon brainwave entrainment
PAUSE
Observation: Academic Research Results: [Martinez et al. 2024] Binaural beats create auditory illusion when different frequencies are played in each ear, leading to brainwave entrainment. Studies show effectiveness in anxiety reduction, sleep improvement, and cognitive enhancement.
Thought: Academic research provides solid foundation. I should also check if there's user context available for personalization.
Action: get_user_session_history: unknown
PAUSE
Observation: User session history is not available yet, as this is the first interaction.
Thought: No user history available, but I have strong academic foundation to provide comprehensive answer.
Answer: Binaural beats are an auditory phenomenon where two slightly different frequencies are played in each ear, creating a perceived beat or pulse. This can alter brainwave activity through a process called brainwave entrainment, inducing relaxation, reduced anxiety, and improved focus. Research shows that binaural beats can be an effective tool for managing stress, anxiety, and sleep disorders. Would you like to know more about how binaural beats work or how to use them for relaxation?

SCENARIO 5 - Early Intervention Decision:
Question: User shows increasing anxiety trend over 2 minutes
Thought: User has completed 2 minutes of monitoring. I need to analyze the trend and determine if early intervention is warranted based on research evidence.
Action: analyze_minute_buffer: user123
PAUSE
Observation: Minute 2 shows increasing anxiety (0.3→0.5), trend upward, confidence 0.82
Thought: Anxiety is trending upward but below crisis threshold. I should check user's historical response patterns and search for early intervention research.
Action: get_user_problem_patterns: user123
PAUSE
Observation: User responds well to alpha protocols, prefers early intervention, has anxiety spike history
Thought: User profile supports early intervention. I need research evidence for this specific pattern.
Action: search_contextual_research: {{"query": "early anxiety intervention alpha waves", "user_context": {{"response_history": "positive_alpha", "current_trend": "increasing_anxiety"}}}}
PAUSE
Observation: Research shows 10Hz alpha prevents anxiety escalation in 78% of cases when applied within first 3 minutes
Thought: Research strongly supports early alpha intervention for this user profile and current trend.
Answer: Based on your increasing anxiety trend (0.3→0.5 over 2 minutes) and research showing 78% effectiveness in similar cases, I recommend immediate 10Hz alpha wave protocol. Your historical response to alpha frequencies and preference for early intervention support this evidence-based approach. The protocol should be applied for 6-8 minutes to prevent anxiety escalation.

SCENARIO 6 - Session Progress Analysis:
Question: How is the current session progressing?
Thought: User is asking for session progress analysis. I need to review the minute-by-minute data and overall trends.
Action: get_session_progress_trend: user123
PAUSE
Observation: 8 minutes completed, 2 interventions applied, trend shows improvement from minute 5 onwards
Thought: Session shows positive response to interventions. I should provide detailed analysis with next steps.
Answer: Your session is progressing well. Over 8 minutes: Minutes 1-4 showed increasing stress, alpha protocol was initiated at minute 5, and you've shown consistent improvement since then. Two interventions were needed, which is normal for your anxiety pattern. Current trend is positive - continue with the current protocol for 5-7 more minutes for optimal effectiveness.

CRITICAL RULES:
- ALWAYS use academic research to support decisions (in reasoning process)
- NEVER make recommendations without evidence
- CONSIDER user history and individual response patterns
- MONITOR continuously and adjust based on real-time data
- PRIORITIZE user safety and contraindications
- PROVIDE clear reasoning for every decision
- USE "Answer:" format for final responses, never "Action: None needed"
- IMPORTANT: In final answers to users, do NOT include academic citations or researcher names
- IMPORTANT: Use phrases like "research shows", "studies indicate", "evidence suggests" instead of specific citations
- IMPORTANT: Keep final answers user-friendly and informative, not academic papers

HANDLING MISSING ACADEMIC DATA:
- For GENERAL questions (greetings, basic concepts): Use your knowledge directly, no need to mention data limitations
- For SPECIFIC technical questions (exact frequencies, protocols): 
  * If academic data exists: Use it with confidence
  * If academic data missing: "I cannot find verified clinical data on this specific topic in my academic sources. For safety in neurotherapy applications, I recommend consulting with a qualified practitioner rather than providing unverified information."
- NEVER provide unverified medical/therapeutic advice when academic sources are unavailable
- ALWAYS prioritize user safety over providing incomplete answers

Remember: You are managing real-time neurotherapy with scientific precision. Every decision must be evidence-based and personalized, but communicated in accessible language.
""".strip()
    
    return system_prompt


# Enhanced Action regex pattern for ReAct methodology - More flexible
action_re = re.compile(r'^Action:\s*(\w+)\s*:\s*(.*)$', re.MULTILINE)


class BackgroundEmotionProcessor:
    """
    Background processor for continuous emotion monitoring
    Handles real-time emotion frames without blocking main thread
    """
    
    def __init__(self):
        self.active_sessions = {}
        self.processing_queue = []
        
    async def process_emotion_frame_async(self, user_id: str, emotion_data: dict, groq_api_key: str):
        """
        Asynchronously process emotion frame
        Can be called without waiting for response
        """
        try:
            import asyncio
            
            # Add to processing queue
            frame_task = {
                "user_id": user_id,
                "emotion_data": emotion_data,
                "timestamp": datetime.utcnow(),
                "status": "queued"
            }
            
            self.processing_queue.append(frame_task)
            
            # Process in background
            asyncio.create_task(self._background_analysis(frame_task, groq_api_key))
            
            return {"status": "queued", "message": "Frame queued for background processing"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _background_analysis(self, frame_task: dict, groq_api_key: str):
        """Background analysis without blocking main thread"""
        try:
            user_id = frame_task["user_id"]
            emotion_data = frame_task["emotion_data"]
            
            # Use agent for analysis
            agent_query = f"Process emotion frame for user {user_id}: {emotion_data}. Check if minute analysis is needed."
            
            result = query_dawos_agent(
                question=agent_query,
                max_turns=3,
                groq_api_key=groq_api_key
            )
            
            frame_task["status"] = "completed"
            frame_task["result"] = result["final_answer"]
            
        except Exception as e:
            frame_task["status"] = "error"
            frame_task["error"] = str(e)

# Global background processor
background_processor = BackgroundEmotionProcessor()

def query_dawos_agent(question: str, max_turns: int = 5, groq_api_key: str = None, user_id: str = None) -> Dict[str, Any]:
    """
    DAWOS ReAct Agent query function
    Neurotherapeutic consultation with reasoning and action cycles
    
    Args:
        question: Kullanıcının sorusu
        max_turns: Maksimum döngü sayısı
        groq_api_key: API anahtarı (Gemini key olacak)
    
    Returns:
        Dict containing conversation trace and final answer
    """
    
    system_prompt = create_dawos_system_prompt()
    agent = DawosReActAgent(system_prompt, groq_api_key)
    
    # CRITICAL: Reset agent memory for fresh conversation
    agent.reset_conversation()
    
    conversation_trace = []
    i = 0
    next_prompt = question
    error_count = 0  # Moved to function scope
    max_errors = 2   # Moved to function scope
    same_action_count = {}
    
    # Add initial question to trace
    conversation_trace.append({
        "type": "user_question",
        "content": question,
        "turn": 0
    })
    
    while i < max_turns:
        i += 1
        
        # Agent'tan cevap al (user_id ile)
        agent.messages.append({"role": "user", "content": next_prompt})
        agent._manage_context_length()
        result = agent.execute(user_id=user_id)
        agent.messages.append({"role": "assistant", "content": result})
        
        # Check if this is a final answer - Multiple formats
        if any(phrase in result for phrase in ["Answer:", "✨ Final Answer:", "Final Answer:"]):
            # This is the final answer - only add as final_answer
            conversation_trace.append({
                "type": "final_answer",
                "content": result,
                "turn": i
            })
            break
        else:
            # This is an intermediate response
            conversation_trace.append({
                "type": "agent_response", 
                "content": result,
                "turn": i
            })
        
        # Check if agent says "None needed" or similar - force final answer
        if any(phrase in result.lower() for phrase in ["none needed", "no action needed", "can provide the answer"]):
            # Agent wants to give final answer but didn't use Answer: format
            # Force it to give final answer in next turn
            next_prompt = "You have enough information. Provide your final answer using this format: Answer: [your complete response]"
            continue
        
        # Enhanced Action parsing with better error handling
        actions = []
        
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith('Action:'):
                match = action_re.match(line)
                if match:
                    actions.append(match)
                else:
                    # Try alternative parsing for malformed actions
                    alt_match = re.match(r'Action:\s*(\w+)[\s:]*(.*)$', line)
                    if alt_match:
                        actions.append(alt_match)
        
        if actions:
            # Action found, execute it
            action, action_input = actions[0].groups()
            # Check for repeated same actions (infinite loop prevention)
            action_key = f"{action}:{action_input[:50]}"  # First 50 chars for comparison
            same_action_count[action_key] = same_action_count.get(action_key, 0) + 1
            
            if same_action_count[action_key] > 2:
                error_msg = f"Repeated action detected: {action}. Forcing final answer."
                conversation_trace.append({
                    "type": "error",
                    "content": error_msg,
                    "turn": i
                })
                next_prompt = "You are repeating the same action. Please provide your final answer now using: Answer: [your response]"
                continue
            
            if action not in known_actions:
                error_count += 1
                error_msg = f"Unknown tool: {action}: {action_input}"
                conversation_trace.append({
                    "type": "error",
                    "content": error_msg,
                    "turn": i,
                    "error_count": error_count
                })
                
                if error_count >= max_errors:
                    conversation_trace.append({
                        "type": "final_answer",
                        "content": f"I encountered repeated errors and cannot proceed. Last error: {error_msg}",
                        "turn": i
                    })
                    break
                
                # Give agent another chance with error feedback
                next_prompt = f"Error: Unknown tool '{action}'. Available tools: {', '.join(known_actions.keys())}. Please use a valid tool or provide your final answer."
                continue
            
            # Execute tool with enhanced error handling
            conversation_trace.append({
                "type": "tool_execution",
                "tool_name": action,
                "tool_input": action_input,
                "turn": i
            })
            
            try:
                # Handle JSON input for tools that expect it
                if action_input.strip().startswith('{') and action_input.strip().endswith('}'):
                    # Tool expects JSON, pass as string (tool will parse internally)
                    observation = known_actions[action](action_input)
                else:
                    # Regular string input
                    observation = known_actions[action](action_input)
                
                conversation_trace.append({
                    "type": "tool_result",
                    "content": str(observation),
                    "turn": i
                })
                
                next_prompt = f"Observation: {observation}"
                error_count = 0  # Reset error count on successful execution
                
            except Exception as e:
                error_count += 1
                error_msg = f"Tool execution error for {action}: {str(e)}"
                conversation_trace.append({
                    "type": "tool_error",
                    "content": error_msg,
                    "turn": i,
                    "error_count": error_count
                })
                
                if error_count >= max_errors:
                    conversation_trace.append({
                        "type": "final_answer",
                        "content": f"I encountered repeated tool execution errors and cannot proceed. Last error: {error_msg}",
                        "turn": i
                    })
                    break
                
                # Give agent feedback about the error
                next_prompt = f"Tool error: {str(e)}. Please try a different approach or provide your final answer."
                continue
        else:
            # Action yok, final answer
            conversation_trace.append({
                "type": "final_answer",
                "content": result,
                "turn": i
            })
            break
    
    # Handle max_turns exceeded
    if i >= max_turns:
        conversation_trace.append({
            "type": "final_answer",
            "content": "The analysis is taking too long. Please try a more specific question or break it into smaller parts.",
            "turn": i
        })
    
    # Son cevabı bul - Multiple answer formats
    final_answer = ""
    for trace in reversed(conversation_trace):
        if trace["type"] in ["agent_response", "final_answer"]:
            content = trace["content"]
            # Multiple answer formats
            for answer_format in ["✨ Final Answer:", "Final Answer:", "Answer:"]:
                if answer_format in content:
                    answer_part = content.split(answer_format, 1)[1].strip()
                    final_answer = answer_part
                    break
            if final_answer:  # Break outer loop if found
                break
            else:
                final_answer = content
            break
    
    # Check if we got a proper final answer
    has_final_answer = any(trace["type"] == "final_answer" for trace in conversation_trace)
    has_answer_in_response = "Answer:" in final_answer
    
    return {
        "question": question,
        "final_answer": final_answer,
        "conversation_trace": conversation_trace,
        "total_turns": i,
        "success": has_final_answer or has_answer_in_response,
        "error_count": error_count,
        "performance_metrics": {
            "turns_used": i,
            "max_turns": max_turns,
            "errors_encountered": error_count,
            "repeated_actions": sum(1 for count in same_action_count.values() if count > 1)
        }
    }
