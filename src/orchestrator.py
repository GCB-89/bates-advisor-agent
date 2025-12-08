"""
Multi-Agent System Orchestrator

Coordinates the entire student advisor system:
1. Router Agent - Routes queries to specialists
2. Specialist Agents - Execute in parallel/sequential
3. Session Manager - Tracks context and memory
4. Observability Logger - Logs everything
5. Response Synthesis - Combines multi-agent responses
"""

import time
import sys
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(str(Path(__file__).parent))

from agents.router_agent import RouterAgent
from agents.program_agent import ProgramAdvisorAgent
from agents.admissions_agent import AdmissionsAdvisorAgent
from agents.financial_agent import FinancialAdvisorAgent
from agents.base_rag_agent import get_shared_embeddings
from memory.session_manager import SessionManager
from observability.logger import get_logger


class MultiAgentOrchestrator:
    """
    Orchestrates the multi-agent student advisor system.
    
    Architecture:
    User Query → Router → Specialist Agent(s) → Response Synthesis → User
                   ↓
            Memory & Observability (tracked throughout)
    
    Features:
    - Intelligent routing
    - Parallel agent execution
    - Session memory
    - Full observability
    - Response synthesis
    """
    
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """Initialize the orchestrator with all components."""
        print("🚀 Initializing Multi-Agent Orchestrator...")
        
        # Initialize router
        print("  → Loading Router Agent...")
        self.router = RouterAgent()
        
        # Initialize shared embeddings (reused across all agents)
        print("  → Initializing shared embeddings...")
        shared_embeddings = get_shared_embeddings()
        
        # Initialize specialist agents with shared embeddings
        print("  → Loading Specialist Agents...")
        self.agents = {
            "program": ProgramAdvisorAgent(chroma_db_path, embeddings=shared_embeddings),
            "admissions": AdmissionsAdvisorAgent(chroma_db_path, embeddings=shared_embeddings),
            "financial": FinancialAdvisorAgent(chroma_db_path, embeddings=shared_embeddings)
        }
        
        # Initialize memory and observability
        print("  → Loading Session Manager...")
        self.session_manager = SessionManager()
        
        print("  → Loading Observability Logger...")
        self.logger = get_logger()
        
        print("✅ Orchestrator ready!\n")
    
    def process_query(
        self,
        session_id: str,
        query: str
    ) -> Dict:
        """
        Main query processing pipeline.
        
        Steps:
        1. Log query
        2. Route to agent(s)
        3. Execute agent(s)
        4. Synthesize responses
        5. Update memory
        6. Log results
        
        Args:
            session_id: Session identifier
            query: User question
            
        Returns:
            Dict with response and metadata
        """
        start_time = time.time()
        
        # Step 1: Log incoming query
        self.logger.log_query(session_id, query)
        
        # Get session context
        context = self.session_manager.get_context_for_agent(session_id)
        
        # Extract any context hints from query
        self.session_manager.extract_context_from_query(session_id, query)
        
        # Step 2: Route query
        routing_result = self.router.route(query)
        selected_agents = routing_result["selected_agents"]
        
        self.logger.log_routing(
            session_id,
            query,
            selected_agents,
            routing_result["reasoning"]
        )
        
        # Handle general/greeting queries that don't need RAG agents
        if "general" in selected_agents and len(selected_agents) == 1:
            general_response = self._handle_general_query(query)
            self.session_manager.add_message(session_id, "user", query)
            self.session_manager.add_message(session_id, "assistant", general_response)
            end_time = time.time()
            return {
                "response": general_response,
                "sources": [],
                "agents_used": ["General Advisor"],
                "routing_reasoning": routing_result["reasoning"],
                "execution_time": end_time - start_time
            }
        
        # Filter out "general" from agents if mixed with other agents
        selected_agents = [a for a in selected_agents if a != "general"]
        if not selected_agents:
            selected_agents = ["program"]  # Default fallback
        
        # Step 3: Execute agents in PARALLEL for better performance
        agent_responses = []
        valid_agents = [(agent_id, self.agents[agent_id]) for agent_id in selected_agents if agent_id in self.agents]
        
        if len(valid_agents) == 1:
            # Single agent - run directly
            agent_id, agent = valid_agents[0]
            agent_start = time.time()
            result = agent.process_query(
                query=query,
                student_context=context["student_context"],
                conversation_history=context["conversation_context"]
            )
            agent_end = time.time()
            
            tools_used = result.get("tools_used", [])
            self.logger.log_agent_execution(session_id, agent.name, agent_start, agent_end, result["response"], tools_used)
            for tool in tools_used:
                self.logger.log_tool_usage(session_id, tool, {"query": query}, "Tool executed")
            agent_responses.append(result)
        else:
            # Multiple agents - run in parallel
            def run_agent(agent_tuple):
                agent_id, agent = agent_tuple
                agent_start = time.time()
                result = agent.process_query(
                    query=query,
                    student_context=context["student_context"],
                    conversation_history=context["conversation_context"]
                )
                agent_end = time.time()
                return (agent_id, agent, agent_start, agent_end, result)
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(run_agent, agent_tuple): agent_tuple[0] for agent_tuple in valid_agents}
                
                for future in as_completed(futures):
                    agent_id, agent, agent_start, agent_end, result = future.result()
                    tools_used = result.get("tools_used", [])
                    self.logger.log_agent_execution(session_id, agent.name, agent_start, agent_end, result["response"], tools_used)
                    for tool in tools_used:
                        self.logger.log_tool_usage(session_id, tool, {"query": query}, "Tool executed")
                    agent_responses.append(result)
        
        # Step 4: Synthesize responses (if multiple agents)
        if len(agent_responses) == 1:
            final_response = agent_responses[0]["response"]
            sources = agent_responses[0]["sources"]
        else:
            final_response, sources = self._synthesize_responses(agent_responses)
        
        # Step 5: Update memory
        self.session_manager.add_message(session_id, "user", query)
        self.session_manager.add_message(session_id, "assistant", final_response)
        
        self.logger.log_memory_update(
            session_id,
            "conversation_history",
            {"messages_added": 2}
        )
        
        # Step 6: Log final response
        end_time = time.time()
        total_time = end_time - start_time
        
        self.logger.log_response(
            session_id,
            final_response,
            total_time,
            len(sources)
        )
        
        return {
            "response": final_response,
            "sources": sources,
            "agents_used": [r["agent"] for r in agent_responses],
            "routing_reasoning": routing_result["reasoning"],
            "execution_time": total_time
        }
    
    def _handle_general_query(self, query: str) -> str:
        """
        Handle general/greeting queries that don't need RAG.
        
        Args:
            query: User's general question
            
        Returns:
            Friendly response guiding user to ask specific questions
        """
        query_lower = query.lower()
        
        # Check for different types of general queries
        if any(word in query_lower for word in ["hello", "hi", "hey", "greetings"]):
            return """Hello! Welcome to the Bates Technical College Student Advisor. I'm here to help you with:

* **Programs & Courses** - Learn about our certificates, degrees, and training programs
* **Admissions** - Application process, requirements, and enrollment steps
* **Financial Aid** - Tuition costs, FAFSA, scholarships, and payment options

What would you like to know about? Feel free to ask me anything!"""
        
        elif any(phrase in query_lower for phrase in ["can you help", "help me", "assist", "what can you do"]):
            return """Of course! I can definitely help. To best assist you, could you please tell me what you need help with? For example, are you interested in:

* Learning about a specific program at Bates Tech?
* Understanding the admissions or enrollment process?
* Exploring financial aid, scholarships, or tuition costs?

Just ask your question and I'll do my best to provide helpful information!"""
        
        elif any(phrase in query_lower for phrase in ["thank", "thanks", "appreciate"]):
            return """You're welcome! If you have any more questions about Bates Technical College, feel free to ask. I'm happy to help with programs, admissions, or financial aid questions."""
        
        elif any(phrase in query_lower for phrase in ["who are you", "what are you"]):
            return """I'm the Bates Technical College Student Advisor, an AI assistant designed to help prospective and current students with questions about:

* **Programs** - Over 30 professional-technical programs
* **Admissions** - How to apply and enroll
* **Financial Aid** - Funding your education

How can I help you today?"""
        
        else:
            return """I'd be happy to help! I'm the Bates Technical College Student Advisor. I can answer questions about:

* Programs and courses offered at Bates Tech
* Admissions requirements and the application process
* Financial aid, scholarships, and tuition

What would you like to know?"""
    
    def _synthesize_responses(self, responses: List[Dict]) -> tuple:
        """
        Synthesize multiple agent responses into one coherent answer.
        
        Args:
            responses: List of agent response dicts
            
        Returns:
            Tuple of (synthesized_response, combined_sources)
        """
        # Simple synthesis: combine responses with agent attribution
        parts = []
        all_sources = []
        
        for response in responses:
            parts.append(f"**{response['agent']}:**\n{response['response']}")
            all_sources.extend(response['sources'])
        
        synthesized = "\n\n".join(parts)
        
        # Remove duplicate sources
        unique_sources = []
        seen_pages = set()
        for source in all_sources:
            page = source['page']
            if page not in seen_pages:
                unique_sources.append(source)
                seen_pages.add(page)
        
        return synthesized, unique_sources
    
    def create_session(self) -> str:
        """Create a new session."""
        return self.session_manager.create_session()
    
    def get_session(self, session_id: str) -> Dict:
        """Get session data."""
        return self.session_manager.get_session(session_id)
    
    def save_session(self, session_id: str):
        """Save session to disk."""
        self.session_manager.save_session(session_id)
    
    def get_metrics(self) -> Dict:
        """Get system metrics."""
        return self.logger.get_metrics()
    
    def print_metrics(self):
        """Print metrics summary."""
        self.logger.print_metrics_summary()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("🎓 Multi-Agent Student Advisor System - Test Run")
    print("=" * 70)
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Create a session
    session_id = orchestrator.create_session()
    print(f"\n📝 Session Created: {session_id}\n")
    
    # Test questions
    test_questions = [
        "What programs does Bates Tech offer in healthcare?",
        "How do I apply and what does it cost?",
        "Tell me about the Carpentry program courses"
    ]
    
    # Process each question
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}/{len(test_questions)}")
        print(f"{'='*70}\n")
        
        result = orchestrator.process_query(session_id, question)
        
        print(f"\n💬 Response:\n{result['response']}")
        print(f"\n📊 Metadata:")
        print(f"  • Agents used: {', '.join(result['agents_used'])}")
        print(f"  • Execution time: {result['execution_time']*1000:.0f}ms")
        print(f"  • Sources: {len(result['sources'])}")
    
    # Print final metrics
    print(f"\n{'='*70}")
    orchestrator.print_metrics()
    
    # Save session
    orchestrator.save_session(session_id)
    print(f"💾 Session saved: {session_id}")
