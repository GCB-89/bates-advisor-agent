"""
Admissions Advisor Agent

Specializes in:
- Admission requirements
- Application processes
- Enrollment procedures
- Deadlines and schedules
- Placement tests
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_rag_agent import BaseRAGAgent


class AdmissionsAdvisorAgent(BaseRAGAgent):
    """
    Specialist agent for admissions and enrollment.
    
    Expertise:
    - Admission requirements for programs
    - Application processes and deadlines
    - Placement testing
    - Enrollment procedures
    - Prerequisites and qualifications
    """
    
    def __init__(self, chroma_db_path: str = "./chroma_db", embeddings=None):
        super().__init__(
            name="Admissions Advisor",
            role="Admissions, Enrollment, and Requirements Expert",
            chroma_db_path=chroma_db_path,
            embeddings=embeddings
        )
        
        self.system_prompt = """You are the Admissions Advisor for Bates Technical College, specializing in admissions, enrollment, and requirements.

Your Expertise:
- Admission requirements for all programs
- Application processes and procedures
- Important deadlines and schedules
- Placement testing and assessment
- Prerequisites and program qualifications
- Transfer credit policies
- International student admissions

Guidelines:
- ALWAYS answer questions using the information provided in the context/catalog data
- Provide clear, step-by-step guidance for applications based on the catalog
- Explain requirements thoroughly with specific details from the data
- Always mention important deadlines when available
- Be encouraging, supportive, and CONFIDENT in your answers
- Reference specific catalog pages for detailed requirements
- NEVER mention internal tools or technical systems to students
- ONLY recommend contacting the school if the specific information is truly NOT in the catalog data provided
- If you must redirect, use:
  * Website: www.batestech.edu
  * Phone: (253) 680-7000
  * Email: admissions@batestech.edu

Key Information to Cover:
- What documents are needed
- When to apply
- How to submit applications
- What tests may be required
- Program-specific requirements

Remember: You have access to the college catalog - USE IT to answer questions! Be confident and helpful."""
    
    def process_query(
        self,
        query: str,
        student_context: str = "",
        conversation_history: str = ""
    ) -> dict:
        """Process admissions-related queries."""
        # Enhance query for admissions-specific context
        query_lower = query.lower()
        
        # Add context keywords for better retrieval
        enhanced_query = query
        if "apply" in query_lower or "application" in query_lower:
            enhanced_query += " application process requirements deadlines"
        elif "requirement" in query_lower:
            enhanced_query += " admission prerequisite qualification"
        elif "deadline" in query_lower:
            enhanced_query += " schedule timeline dates"
        
        # Retrieve and process
        context_chunks = self.retrieve_context(enhanced_query, k=5)
        context = self.format_context(context_chunks)
        
        response = self.generate_response(
            query=query,
            context=context,
            student_context=student_context,
            conversation_history=conversation_history,
            system_prompt=self.system_prompt
        )
        
        return {
            "agent": self.name,
            "response": response,
            "sources": [
                {"page": chunk["page"], "preview": chunk["content"][:150]}
                for chunk in context_chunks
            ]
        }


if __name__ == "__main__":
    print("📋 Testing Admissions Advisor Agent\n")
    print("=" * 60)
    
    agent = AdmissionsAdvisorAgent()
    
    test_questions = [
        "What are the admission requirements?",
        "How do I apply to Bates Tech?",
        "When are application deadlines?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 60)
        
        result = agent.process_query(question)
        
        print(f"💬 Response:\n{result['response']}")
        print(f"📄 Sources: {len(result['sources'])}")
        print("=" * 60)
