# 🎓 Bates Technical College - Multi-Agent Student Advisor System

A production-grade, multi-agent AI system that provides comprehensive student advising for Bates Technical College using RAG, parallel agents, custom tools, memory, and full observability.

## 🎯 What This System Does

- **Multi-Agent Architecture**: Router coordinates 3 specialist agents
- **Intelligent Routing**: Questions automatically routed to the right expert (with caching)
- **Parallel Execution**: Multiple agents run simultaneously for complex queries
- **Session Memory**: Remembers student context across conversations
- **Custom Tools**: Course search and program finder
- **Full Observability**: Logging, tracing, and metrics for every interaction
- **RAG-Powered**: Searches through 459-page course catalog intelligently
- **Adaptive Retrieval**: Smart context retrieval based on query complexity
- **Health Monitoring**: Built-in health check script with reporting

## 📁 Project Structure

```
bates_advisor_agent/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── main.py                       # Main entry point
├── agent.py                      # Agent interface module
├── health_check.py               # System health diagnostics
├── .env                          # Environment variables
├── data/                         # PDF storage
│   └── BatesTech2025-26Catalog.pdf
├── src/
│   ├── ingest_pdf.py            # Loads PDF into vector database
│   ├── chat.py                  # Interactive chat interface
│   ├── orchestrator.py          # Main system orchestrator
│   ├── agents/
│   │   ├── base_rag_agent.py    # Base class for RAG agents
│   │   ├── router_agent.py      # Routes to specialist agents
│   │   ├── program_agent.py     # Program/course specialist
│   │   ├── admissions_agent.py  # Admissions specialist
│   │   └── financial_agent.py   # Financial aid specialist
│   ├── tools/
│   │   ├── course_search.py     # Custom course search tool
│   │   └── program_finder.py    # Custom program finder tool
│   ├── memory/
│   │   └── session_manager.py   # Session & memory management
│   └── observability/
│       └── logger.py            # Logging, tracing, metrics
├── chroma_db/                   # Vector database (auto-generated)
├── memory/                      # Session storage (auto-generated)
├── health_reports/              # Health check reports (auto-generated)
└── logs/                        # Agent interaction logs
```

## 🚀 Setup Instructions

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
```

**Get your free API key at:** https://aistudio.google.com/apikey

### Step 3: Place Your PDF

Copy the Bates catalog PDF into the `data/` folder:
```bash
cp /path/to/BatesTech2025-26Catalog.pdf data/
```

### Step 4: Ingest the PDF (One-Time Setup)

This creates the vector database:
```bash
python src/ingest_pdf.py
```

This takes 2-5 minutes and only needs to be done once!

### Step 5: Start Chatting!

```bash
python main.py
```

## 💡 How It Works (Multi-Agent Architecture)

### Agent Flow
```
Student Question → Router Agent → [Analyzes Intent] 
                                         ↓
                    ┌───────────────────────────────────┐
                    ↓                ↓                  ↓
            Program Agent    Admissions Agent    Financial Agent
                    ↓                ↓                  ↓
            [RAG Retrieval]  [RAG Retrieval]   [RAG Retrieval]
                    ↓                ↓                  ↓
            [Custom Tools]   [Custom Tools]    [Custom Tools]
                    ↓                ↓                  ↓
                    └───────────────┬───────────────────┘
                                    ↓
                        Synthesized Response
                                    ↓
                            Session Memory Updated
```

### System Components

1. **Router Agent**: 
   - Analyzes question intent using LLM
   - Routes to appropriate specialist agent(s)
   - Can trigger parallel agents for complex queries

2. **Specialist Agents**:
   - **Program Agent**: Courses, curriculum, program requirements
   - **Admissions Agent**: Enrollment, deadlines, prerequisites
   - **Financial Agent**: Tuition, fees, financial aid, scholarships

3. **RAG Pipeline**:
   - PDF chunked and vectorized
   - Similarity search finds relevant content
   - LLM generates contextualized answers

4. **Custom Tools**:
   - `course_search`: Find courses by code/name/topic
   - `program_finder`: Search programs by field

5. **Memory System**:
   - Tracks student context (major, year, interests)
   - Maintains conversation history
   - Enables personalized recommendations

6. **Observability**:
   - Every agent interaction logged
   - Traces routing decisions
   - Metrics on response time and accuracy

## 🎓 Advanced Concepts Demonstrated

### 1. Multi-Agent Systems ✅
- **Router Agent**: Intelligent intent classification and routing
- **Specialist Agents**: Domain-specific experts working in parallel
- **Sequential + Parallel**: Router → Specialists → Synthesis

### 2. Custom Tools ✅
- **course_search**: Structured search through course catalog
- **program_finder**: Find programs by field/keyword
- Tool result caching for performance

### 3. Sessions & Memory ✅
- **Session Management**: Track conversation state
- **Student Context Memory**: Remember major, interests, goals
- **Conversation History**: Build on previous questions
- **Persistent Memory**: Store user preferences

### 4. Observability ✅
- **Structured Logging**: JSON logs for every interaction
- **Agent Tracing**: Track routing decisions and tool usage
- **Performance Metrics**: Response time, accuracy tracking
- **Error Handling**: Graceful failures with detailed logs

### 5. RAG (Retrieval-Augmented Generation)
- Vector similarity search
- Chunk retrieval and ranking
- Context-aware generation

### 6. Prompt Engineering
- Role-specific prompts for each agent
- Few-shot examples for routing
- Chain-of-thought reasoning

## 🏥 Health Check

Run the health check script to verify system status:

```bash
python health_check.py
```

This performs 8 diagnostic checks:
- Environment configuration
- Dependencies verification
- Vector database connectivity
- Agent initialization
- Router agent functionality
- Specialist agents status
- Memory system
- End-to-end query test

Reports are saved to `health_reports/` with timestamps.

## 🐳 Docker Deployment

### Build and Run with Docker Compose (Recommended)

```bash
# Build the image
docker-compose build

# Run the advisor (interactive mode)
docker-compose run --rm bates-advisor

# Run health check
docker-compose --profile health run --rm health-check
```

### Build and Run with Docker

```bash
# Build the image
docker build -t bates-advisor .

# Run the container
docker run -it --rm \
  -e GOOGLE_API_KEY=your_api_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db:ro \
  -v $(pwd)/logs:/app/logs \
  bates-advisor

# Run health check
docker run --rm \
  -e GOOGLE_API_KEY=your_api_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db:ro \
  bates-advisor python health_check.py
```

**Note:** The vector database (`chroma_db/`) must be created locally first by running `python src/ingest_pdf.py` before containerizing.

## 🔧 Customization Ideas

- Add more documents (student handbook, schedules, etc.)
- Connect to a web interface with Streamlit
- Deploy to cloud (AWS Lambda, Azure Functions)
- Create specialized agents (enrollment advisor, career counselor, etc.)

## 📝 Example Questions to Try

- "What programs does Bates Tech offer in healthcare?"
- "Tell me about the Carpentry program requirements"
- "What are the admission requirements?"
- "How much does tuition cost?"
- "What courses are in the Welding program?"

---

**Ready for production!** Run `python main.py` to start, or use Docker for containerized deployment. 🚀
