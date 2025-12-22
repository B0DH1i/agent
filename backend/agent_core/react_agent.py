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
    Specialized for cognitive health and neurotherapeutic consultation
    """
    
    def __init__(self, system_prompt: str = "", groq_api_key: Optional[str] = None):
        self.system_prompt = system_prompt
        self.messages = []
        
        # Groq client initialization
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable gerekli")
        
        self.client = Groq(api_key=api_key)
        
        if self.system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def __call__(self, message: str) -> str:
        """Agent conversation interface method"""
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self) -> str:
        """Execute agent reasoning and action cycle"""
        try:
            result = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=self.messages,
                temperature=0,
                stop=["PAUSE", "Observation:", "<|"]
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
    ReAct methodology adapted for cognitive health applications
    """
    
    # Tools açıklamalarını oluştur
    tools_with_desc = "\n\n".join([tool.__name__ + (tool.__doc__ or "") for tool in tools])
    
    system_prompt = f"""
You are a DAWOS Neurotherapy AI Agent. You MUST follow this EXACT format:

MANDATORY FORMAT:
1. Thought: [your reasoning]
2. Action: [tool_name]: [input] OR Answer: [final response]

CRITICAL RULES:
- After using 1-2 tools, you MUST respond with "Answer: [complete response]"
- NEVER say "Action: None needed" - use "Answer:" instead
- Answer must be comprehensive and helpful
- Do NOT continue with more actions after you have enough information

IMPORTANT: For protocol recommendations, use search_neurotherapeutic_knowledge with queries like:
- "stress management protocol alpha waves"
- "SAD depression treatment frequencies protocol"
- "anxiety protocol recommendations binaural beats"
All protocol information comes from academic research documents.

Your available actions are:

{tools_with_desc}

CORRECT EXAMPLE:

Question: What protocol do you recommend for stress?
Thought: User needs stress management protocol. I should search academic knowledge base for stress protocols.
Action: search_neurotherapeutic_knowledge: stress management protocol alpha waves recommendations
PAUSE
Observation: Alpha wave entrainment (8-12 Hz) reduced cortisol levels by 23%. 10 Hz alpha most effective (87% success rate). Clinical Protocol: 10 Hz alpha with 432 Hz carrier, 20 minutes.
Thought: I have sufficient academic research data about stress protocols. I will now provide a complete answer.
Answer: For stress management, I recommend 10 Hz Alpha waves based on academic research. Studies show alpha wave entrainment reduces cortisol levels by 23% within 20 minutes, with 87% success rate. Clinical protocol: 10 Hz alpha binaural beats with 432 Hz carrier frequency for 20 minutes. This is the most evidence-based approach for acute stress management.

REMEMBER: Use "Answer:" not "Action: None needed"!
""".strip()
    
    return system_prompt


# Action regex pattern for ReAct methodology
action_re = re.compile(r'^Action: (\w+): (.*)$')


def query_dawos_agent(question: str, max_turns: int = 5, groq_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    DAWOS ReAct Agent query function
    Neurotherapeutic consultation with reasoning and action cycles
    
    Args:
        question: Kullanıcının sorusu
        max_turns: Maksimum döngü sayısı
        groq_api_key: Groq API anahtarı
    
    Returns:
        Dict containing conversation trace and final answer
    """
    
    system_prompt = create_dawos_system_prompt()
    agent = DawosReActAgent(system_prompt, groq_api_key)
    
    conversation_trace = []
    i = 0
    next_prompt = question
    
    # Add initial question to trace
    conversation_trace.append({
        "type": "user_question",
        "content": question,
        "turn": 0
    })
    
    while i < max_turns:
        i += 1
        
        # Agent'tan cevap al
        result = agent(next_prompt)
        
        conversation_trace.append({
            "type": "agent_response", 
            "content": result,
            "turn": i
        })
        
        # Check if this is a final answer
        if "Answer:" in result:
            # This is the final answer
            conversation_trace.append({
                "type": "final_answer",
                "content": result,
                "turn": i
            })
            break
        
        # Check if agent says "None needed" or similar - force final answer
        if any(phrase in result.lower() for phrase in ["none needed", "no action needed", "can provide the answer"]):
            # Agent wants to give final answer but didn't use Answer: format
            # Force it to give final answer in next turn
            next_prompt = "You have enough information. Provide your final answer using this format: Answer: [your complete response]"
            continue
        
        # Action parsing with regex pattern
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        
        if actions:
            # Action found, execute it
            action, action_input = actions[0].groups()
            
            if action not in known_actions:
                error_msg = f"Unknown tool: {action}: {action_input}"
                conversation_trace.append({
                    "type": "error",
                    "content": error_msg,
                    "turn": i
                })
                break
            
            # Execute tool
            conversation_trace.append({
                "type": "tool_execution",
                "tool_name": action,
                "tool_input": action_input,
                "turn": i
            })
            
            try:
                observation = known_actions[action](action_input)
                conversation_trace.append({
                    "type": "tool_result",
                    "content": str(observation),
                    "turn": i
                })
                
                next_prompt = f"Observation: {observation}"
                
            except Exception as e:
                error_msg = f"Tool execution error: {str(e)}"
                conversation_trace.append({
                    "type": "tool_error",
                    "content": error_msg,
                    "turn": i
                })
                break
        else:
            # Action yok, final answer
            conversation_trace.append({
                "type": "final_answer",
                "content": result,
                "turn": i
            })
            break
    
    # Son cevabı bul - Answer: kısmını extract et
    final_answer = ""
    for trace in reversed(conversation_trace):
        if trace["type"] in ["agent_response", "final_answer"]:
            content = trace["content"]
            # Answer: kısmını bul
            if "Answer:" in content:
                answer_part = content.split("Answer:", 1)[1].strip()
                final_answer = answer_part
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
        "success": has_final_answer or has_answer_in_response
    }