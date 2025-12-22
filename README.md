# DAWOS ReAct Agent - Neurotherapy AI Assistant

## 🎯 Project Overview

This project transforms a static Large Language Model into an autonomous **ReAct Agent** that can think, reason, and use external tools to solve problems. The agent specializes in **neurotherapy and frequency therapy**, providing evidence-based therapeutic protocol recommendations using academic research data.

## 🧠 ReAct Architecture

### Traditional LLM vs ReAct Agent
```
Traditional LLM:
User Question → LLM → Static Answer

ReAct Agent:
User Question → Thought → Action → Observation → Thought → Final Answer
```

### The ReAct Loop
1. **Thought**: Agent analyzes the situation and plans next steps
2. **Action**: Agent selects and uses appropriate tools
3. **Observation**: Agent processes tool results
4. **Repeat**: Until sufficient information is gathered
5. **Answer**: Agent provides comprehensive, evidence-based response

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   User Query    │───▶│   FastAPI       │───▶│   JSON          │  │
│  │ "I feel stressed"│    │   Server        │    │   Response      │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      REACT AGENT BRAIN                              │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │    THOUGHT      │───▶│     ACTION      │───▶│   OBSERVATION   │  │
│  │ "Need research  │    │ search_neuro... │    │ "10Hz reduces   │  │
│  │  on stress"     │    │ ("stress mgmt") │    │  cortisol 23%"  │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│           ▲                                             │           │
│           │              ┌─────────────────┐            │           │
│           └──────────────│  FINAL ANSWER   │◀───────────┘           │
│                          │ "Recommend 10Hz │                        │
│                          │  alpha waves"   │                        │
│                          └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         TOOL LAYER                                  │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ Academic RAG    │    │   Calculator    │    │  Emotion        │  │
│  │ Search Tool     │    │     Tool        │    │  Analysis       │  │
│  │                 │    │                 │    │                 │  │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │  │
│  │ │Query: "stress│ │    │ │ 440-444=4Hz │ │    │ │ User: SAD   │ │  │
│  │ │management"   │ │    │ │ calculation │ │    │ │ Confidence: │ │  │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ │    78%      │ │  │
│  └─────────────────┘    └─────────────────┘    │ └─────────────┘ │  │
│           │                       │            └─────────────────┘  │
│           ▼                       ▼                       │         │
│  ┌─────────────────┐    ┌─────────────────┐              ▼         │
│  │ ChromaDB Vector │    │ Math Operations │    ┌─────────────────┐  │
│  │ Similarity      │    │ Frequency Calc  │    │ Mock Emotion    │  │
│  │ Search          │    │                 │    │ Data            │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   ChromaDB      │    │  Academic       │    │  Research       │  │
│  │ Vector Store    │    │  Documents      │    │  Papers         │  │
│  │                 │    │                 │    │                 │  │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │  │
│  │ │768-dim      │ │    │ │neurotherapy │ │    │ │Johnson 2023 │ │  │
│  │ │embeddings   │ │    │ │_research.txt│ │    │ │Martinez 2023│ │  │
│  │ │vectors      │ │    │ │binaural_    │ │    │ │Williams 2024│ │  │
│  │ │             │ │    │ │beats.txt    │ │    │ │Thompson 2024│ │  │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 Complete Agent Workflow

### Step-by-Step Process Flow
```
1. USER INPUT
   ┌─────────────────────────────────────────────────────────────┐
   │ "I'm feeling stressed. What frequency should I use?"        │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
2. AGENT INITIALIZATION
   ┌─────────────────────────────────────────────────────────────┐
   │ • Load system prompt with ReAct instructions               │
   │ • Initialize conversation history                          │
   │ • Connect to GROQ Llama-3.3-70b model                    │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
3. REASONING PHASE (Turn 1)
   ┌─────────────────────────────────────────────────────────────┐
   │ THOUGHT: "User is stressed and needs therapeutic protocol.  │
   │          I should search academic research for stress       │
   │          management frequencies and their effectiveness."   │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
4. TOOL SELECTION & EXECUTION
   ┌─────────────────────────────────────────────────────────────┐
   │ ACTION: search_neurotherapeutic_knowledge                   │
   │ INPUT: "stress management alpha waves cortisol research"    │
   │                                                             │
   │ TOOL EXECUTION:                                             │
   │ • Generate query embedding (768 dimensions)                │
   │ • Search ChromaDB vector store                             │
   │ • Retrieve top-3 most relevant chunks                      │
   │ • Return raw academic context                              │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
5. OBSERVATION PROCESSING
   ┌─────────────────────────────────────────────────────────────┐
   │ OBSERVATION: "Alpha wave entrainment (8-12 Hz) reduced     │
   │ cortisol levels by 23% within 20 minutes. 10 Hz alpha     │
   │ most effective (87% success rate) for stress management.   │
   │ Clinical studies show significant improvement..."           │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
6. DECISION MAKING (Turn 2)
   ┌─────────────────────────────────────────────────────────────┐
   │ THOUGHT: "Research clearly shows 10 Hz alpha waves are     │
   │          most effective for stress with 87% success rate.  │
   │          I have sufficient evidence to provide specific    │
   │          protocol recommendation."                          │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
7. FINAL RESPONSE GENERATION
   ┌─────────────────────────────────────────────────────────────┐
   │ ANSWER: "For stress management, I recommend 10 Hz alpha    │
   │ waves based on clinical research. Studies show alpha wave  │
   │ entrainment reduces cortisol by 23% within 20 minutes,     │
   │ with 87% success rate. Protocol: 10 Hz binaural beats      │
   │ with 432 Hz carrier for 20 minutes."                       │
   └─────────────────────────────────────────────────────────────┘
```

## 📋 Test Scenarios

### Scenario 1: Single-Step Query
**Input**: "What are binaural beats?"

**Agent Process**:
```
Turn 1:
├─ Thought: "Need to search academic knowledge about binaural beats"
├─ Action: search_neurotherapeutic_knowledge("binaural beats definition")
├─ Observation: "Binaural beats discovered by Heinrich Wilhelm Dove (1839)..."
└─ Answer: "Binaural beats are auditory phenomena created when..."
```

### Scenario 2: Multi-Step Query
**Input**: "I'm feeling stressed. What frequency would you recommend?"

**Agent Process**:
```
Turn 1:
├─ Thought: "User needs stress management protocol, search research"
├─ Action: search_neurotherapeutic_knowledge("stress alpha waves research")
├─ Observation: "10 Hz alpha reduces cortisol by 23%, 87% success rate..."

Turn 2:
├─ Thought: "Have sufficient evidence, can provide specific recommendation"
└─ Answer: "For stress: 10 Hz alpha waves, 20 minutes, 87% effective..."
```

## 🛠️ Technical Implementation

### Core Components
- **LLM**: GROQ Llama-3.3-70b-versatile
- **Vector DB**: ChromaDB with 768-dim embeddings
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **API**: FastAPI for production deployment

### Knowledge Base
- **4 Academic Documents**: Neurotherapy research, binaural beats, frequency therapy, brainwave analysis
- **Smart Chunking**: 200-character chunks with 50-character overlap
- **Semantic Search**: Top-3 relevant chunks per query

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set API key
export GROQ_API_KEY="your_key"

# Run agent
cd backend && python main.py
```

## 📊 Project Achievements

### ✅ ReAct Implementation
- Complete Thought→Action→Observation loops
- Tool-based architecture (no hardcoded responses)
- Multi-step reasoning capabilities
- Academic research integration

### ✅ Production Ready
- ChromaDB vector database
- FastAPI web interface
- Error handling and safety limits
- Evidence-based recommendations