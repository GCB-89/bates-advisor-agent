"""
Agent Module for Bates Multi-Agent Advisor System

This file provides a simple interface to the multi-agent orchestrator
that can be used by various frontends (CLI, web, ADK, etc.).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Global variables
orchestrator = None
GLOBAL_SESSION = None

def initialize_orchestrator():
    """Initialize the orchestrator if not already done."""
    global orchestrator, GLOBAL_SESSION
    
    if orchestrator is not None:
        return orchestrator
    
    try:
        from orchestrator import MultiAgentOrchestrator
    except ImportError as e:
        print(f"❌ Error importing orchestrator: {e}")
        print("Make sure you're in the correct directory and dependencies are installed.")
        raise
    
    # Initialize the orchestrator
    print("🚀 Initializing multi-agent system...")
    chroma_db_path = str(Path(__file__).parent / "chroma_db")
    
    try:
        orchestrator = MultiAgentOrchestrator(chroma_db_path=chroma_db_path)
        print("✅ Multi-agent orchestrator ready!")
    except FileNotFoundError:
        print("❌ Vector database not found!")
        print("Please run: python src/ingest_pdf.py")
        raise
    except Exception as e:
        print(f"❌ Error initializing orchestrator: {e}")
        raise
    
    # Create a persistent session
    GLOBAL_SESSION = orchestrator.create_session()
    print(f"📝 Session created: {GLOBAL_SESSION}")
    
    return orchestrator

# Initialize on import
try:
    initialize_orchestrator()
except Exception as e:
    print(f"⚠️ Failed to initialize on import: {e}")
    print("Will try to initialize on first use.")


def answer_student_question(question: str) -> str:
    """
    Answer student questions using the multi-agent system.
    
    This function routes questions through the orchestrator which:
    1. Uses Router Agent to classify intent
    2. Routes to appropriate specialist(s) (Program/Admissions/Financial)
    3. Each specialist uses RAG with ChromaDB for accurate answers
    4. Synthesizes multi-agent responses
    5. Updates session memory
    
    Args:
        question: Student's question about Bates Technical College
        
    Returns:
        Comprehensive answer with metadata about which agents responded
    """
    try:
        # Ensure orchestrator is initialized
        if orchestrator is None:
            initialize_orchestrator()
        
        # Process through the orchestrator
        result = orchestrator.process_query(GLOBAL_SESSION, question)
        
        # Format the response
        response = result["response"]
        
        # Add metadata footer (agents used, speed, sources)
        agents_used = ", ".join(result["agents_used"])
        time_ms = int(result["execution_time"] * 1000)
        num_sources = len(result["sources"])
        
        # Include source pages
        if num_sources > 0:
            pages = sorted(set(s["page"] for s in result["sources"]))
            pages_str = ", ".join(map(str, pages[:10]))  # Show first 10 pages
            if len(pages) > 10:
                pages_str += f" ... ({len(pages)} total)"
            
            response += f"\n\n---\n*📚 Sources: Catalog pages {pages_str}*\n"
            response += f"*🤖 Answered by: {agents_used}*\n"
            response += f"*⚡ Response time: {time_ms}ms*"
        else:
            response += f"\n\n---\n*🤖 Answered by: {agents_used} • ⚡ {time_ms}ms*"
        
        return response
        
    except Exception as e:
        error_msg = f"❌ Error processing your question: {str(e)}\n\n"
        error_msg += "Please try rephrasing your question or contact support if the issue persists."
        return error_msg


def get_system_metrics() -> str:
    """
    Get performance metrics from the multi-agent system.
    
    Returns:
        Formatted metrics summary
    """
    try:
        metrics = orchestrator.get_metrics()
        
        summary = "📊 **System Performance Metrics**\n\n"
        summary += f"**Query Statistics:**\n"
        summary += f"• Total queries processed: {metrics.get('total_queries', 0)}\n"
        summary += f"• Total agent calls: {metrics.get('total_agent_calls', 0)}\n"
        summary += f"• Total tool invocations: {metrics.get('total_tool_calls', 0)}\n\n"
        
        summary += f"**Performance:**\n"
        avg_time = metrics.get('average_response_time', 0)
        summary += f"• Average response time: {int(avg_time * 1000)}ms\n\n"
        
        if 'agent_usage' in metrics:
            summary += f"**Agent Usage:**\n"
            for agent, count in metrics['agent_usage'].items():
                summary += f"• {agent}: {count} calls\n"
        
        return summary
        
    except Exception as e:
        return f"Error retrieving metrics: {str(e)}"


def process_query(question: str) -> dict:
    """
    Process a query through the multi-agent system.
    
    Args:
        question: The question to process
        
    Returns:
        Dictionary containing response and metadata
    """
    return orchestrator.process_query(GLOBAL_SESSION, question)


# Export main functions for easy import
__all__ = ['answer_student_question', 'get_system_metrics', 'process_query', 'orchestrator', 'GLOBAL_SESSION']

if __name__ == "__main__":
    # If run directly, start a simple test mode
    print("🧪 Agent module test mode")
    print("Type a question to test the system, or 'quit' to exit:")
    
    while True:
        try:
            question = input("\nYou: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if question.lower() == 'metrics':
                print("\n" + get_system_metrics())
            elif question:
                response = answer_student_question(question)
                print(f"\nAgent: {response}")
        except KeyboardInterrupt:
            break
    
    print("\n👋 Test session ended")
