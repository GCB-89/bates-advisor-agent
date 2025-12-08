"""
Router Agent - Multi-Agent Coordinator

Routes student questions to appropriate specialist agents:
- Program Advisor: Courses, programs, curriculum
- Admissions Advisor: Requirements, enrollment, applications
- Financial Advisor: Costs, financial aid, scholarships

Can route to multiple agents for complex queries.
"""

import os
from typing import List, Dict
from functools import lru_cache
import hashlib
import google.generativeai as genai


# Cache for routing decisions (normalized query -> result)
_routing_cache = {}

def _normalize_query(query: str) -> str:
    """Normalize query for caching (lowercase, strip, remove extra spaces)."""
    return ' '.join(query.lower().strip().split())


class RouterAgent:
    """
    Router that analyzes queries and routes to specialist agents.
    
    Capabilities:
    - Intent classification
    - Single or multi-agent routing
    - Parallel agent execution for complex queries
    - Response synthesis
    """
    
    def __init__(self):
        """Initialize router with Gemini model."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.model = genai.GenerativeModel(model_name)
        
        # Available agents and their expertise
        self.agent_registry = {
            "program": {
                "name": "Program Advisor",
                "expertise": [
                    "courses", "classes", "curriculum", "degree", "certificate",
                    "program requirements", "prerequisites", "course descriptions",
                    "credits", "course codes", "training", "pathway"
                ]
            },
            "admissions": {
                "name": "Admissions Advisor",
                "expertise": [
                    "admission", "apply", "application", "enrollment", "enroll",
                    "requirements", "deadline", "placement test", "acceptance",
                    "registration", "qualify", "eligibility"
                ]
            },
            "financial": {
                "name": "Financial Aid Advisor",
                "expertise": [
                    "cost", "tuition", "fees", "price", "financial aid",
                    "scholarship", "grant", "loan", "FAFSA", "payment",
                    "afford", "expensive", "funding", "money"
                ]
            },
            "general": {
                "name": "General Advisor",
                "expertise": [
                    "help", "hello", "hi", "hey", "thanks", "thank you",
                    "who are you", "what can you do", "assist", "support"
                ]
            }
        }
    
    def analyze_intent(self, query: str) -> Dict:
        """
        Analyze query intent using LLM-based classification.
        
        Args:
            query: Student question
            
        Returns:
            Dict with selected agents and reasoning
        """
        # Build classification prompt
        prompt = f"""Analyze this student question and determine which advisor(s) should handle it.

Available Advisors:
1. Program Advisor - Courses, programs, curriculum, requirements, degrees
2. Admissions Advisor - Admission requirements, applications, enrollment, deadlines
3. Financial Aid Advisor - Tuition, costs, financial aid, scholarships, payments
4. General Advisor - Greetings, general help, unclear questions, "who are you", "can you help me"

Student Question: "{query}"

Rules:
- Select 1-3 advisors based on question content
- Use Program Advisor for course/program questions
- Use Admissions Advisor for application/enrollment questions
- Use Financial Advisor for cost/aid questions
- Use General Advisor for greetings, unclear questions, or "can you help me" type questions
- Complex questions may need multiple advisors

Respond in this exact format:
AGENTS: [agent1, agent2, ...]
REASONING: Brief explanation

Examples:
Q: "What courses are in the nursing program?"
AGENTS: [program]
REASONING: Question about program courses and curriculum

Q: "How do I apply to the welding program and how much does it cost?"
AGENTS: [admissions, financial]
REASONING: Question covers both application process and costs

Q: "What are the admission requirements for dental hygiene?"
AGENTS: [admissions]
REASONING: Question about admission requirements

Q: "Can you help me?"
AGENTS: [general]
REASONING: General greeting asking for assistance

Q: "Hello, what can you do?"
AGENTS: [general]
REASONING: Greeting and general inquiry about capabilities

Now analyze the student's question:"""

        # Get LLM classification
        response = self.model.generate_content(prompt)
        response_text = response.text
        
        # Parse response
        selected_agents = []
        reasoning = ""
        
        for line in response_text.split('\n'):
            if line.startswith('AGENTS:'):
                # Extract agent names
                agents_str = line.split('AGENTS:')[1].strip()
                agents_str = agents_str.strip('[]')
                selected_agents = [a.strip() for a in agents_str.split(',')]
            elif line.startswith('REASONING:'):
                reasoning = line.split('REASONING:')[1].strip()
        
        # Fallback to keyword matching if parsing fails
        if not selected_agents:
            selected_agents = self._keyword_based_routing(query)
            reasoning = "Fallback keyword-based routing"
        
        return {
            "agents": selected_agents,
            "reasoning": reasoning
        }
    
    def _keyword_based_routing(self, query: str) -> List[str]:
        """
        Fallback: Simple keyword-based routing.
        
        Used if LLM classification fails.
        """
        query_lower = query.lower()
        selected = []
        
        for agent_id, info in self.agent_registry.items():
            # Check if any expertise keywords match
            if any(keyword in query_lower for keyword in info["expertise"]):
                selected.append(agent_id)
        
        # Default to program advisor if no matches
        if not selected:
            selected = ["program"]
        
        return selected
    
    def route(self, query: str) -> Dict:
        """
        Main routing function with caching.
        
        Args:
            query: Student question
            
        Returns:
            Routing decision with agents and reasoning
        """
        # Check cache first
        normalized = _normalize_query(query)
        if normalized in _routing_cache:
            cached = _routing_cache[normalized]
            return {
                "query": query,
                "selected_agents": cached["agents"],
                "reasoning": cached["reasoning"] + " (cached)"
            }
        
        # Analyze and cache result
        result = self.analyze_intent(query)
        _routing_cache[normalized] = result
        
        # Keep cache size manageable (max 100 entries)
        if len(_routing_cache) > 100:
            # Remove oldest entry
            oldest_key = next(iter(_routing_cache))
            del _routing_cache[oldest_key]
        
        return {
            "query": query,
            "selected_agents": result["agents"],
            "reasoning": result["reasoning"]
        }


# Example usage
if __name__ == "__main__":
    print("🎯 Testing Router Agent\n")
    print("=" * 60)
    
    router = RouterAgent()
    
    test_questions = [
        "What courses are in the Carpentry program?",
        "How do I apply to Bates Tech?",
        "What programs do you offer and how much do they cost?",
        "I'm interested in nursing - what are the requirements and how do I enroll?",
        "Tell me about financial aid options"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 60)
        
        result = router.route(question)
        
        print(f"🎯 Selected Agents: {result['selected_agents']}")
        print(f"💭 Reasoning: {result['reasoning']}")
        print("=" * 60)
