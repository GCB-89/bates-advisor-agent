"""
Program Advisor Agent

Specializes in:
- Course information and descriptions
- Program requirements and curriculum
- Degree and certificate pathways
- Course prerequisites
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_rag_agent import BaseRAGAgent
from tools.course_search import create_course_search_tool
from tools.program_finder import create_program_finder_tool


class ProgramAdvisorAgent(BaseRAGAgent):
    """
    Specialist agent for programs and courses.
    
    This agent has deep knowledge of:
    - All programs offered at Bates Tech
    - Course catalogs and descriptions
    - Program requirements
    - Prerequisites and sequencing
    """
    
    def __init__(self, chroma_db_path: str = "./chroma_db", embeddings=None):
        super().__init__(
            name="Program Advisor",
            role="Programs, Courses, and Curriculum Expert",
            chroma_db_path=chroma_db_path,
            embeddings=embeddings
        )
        
        # Initialize custom tools
        self.course_search_tool = create_course_search_tool(chroma_db_path)
        self.program_finder_tool = create_program_finder_tool(chroma_db_path)
        
        # Specialized system prompt
        self.system_prompt = """You are the Program Advisor for Bates Technical College, specializing in programs, courses, and curriculum planning.

Your Expertise:
- Detailed knowledge of all academic programs (degrees, certificates, training)
- Course descriptions, credits, and prerequisites
- Program requirements and pathways
- Course sequencing and scheduling
- Transfer credits and articulation agreements

Guidelines:
- ALWAYS answer questions using the information provided in the context/catalog data
- Provide accurate course codes, credits, and requirements from the catalog
- Explain program pathways clearly with specific details
- Suggest courses based on student goals
- Reference specific pages when possible
- Be confident in your answers when the information is in the catalog
- NEVER mention internal tools like 'course_search', 'program_finder', or 'RAG retrieval' to students
- ONLY recommend contacting the school if the specific information is truly NOT in the catalog data provided
- If you must redirect, use:
  * Website: www.batestech.edu
  * Phone: (253) 680-7000
  * Email: info@batestech.edu

Remember: You have access to the college catalog - USE IT to answer questions! Only redirect when absolutely necessary."""
    
    def process_query(
        self,
        query: str,
        student_context: str = "",
        conversation_history: str = ""
    ) -> dict:
        """
        Process query with tool usage detection.
        
        Automatically determines if custom tools should be used.
        """
        query_lower = query.lower()
        
        # Detect if we should use tools
        tool_results = []
        
        # Use course search tool if query mentions specific courses
        if any(keyword in query_lower for keyword in ["course", "class", "credit", "prerequisite"]):
            # Extract potential course code or topic
            import re
            course_match = re.search(r'[A-Z]{3,5}\s*\d{3}', query.upper())
            
            if course_match:
                # Specific course code mentioned
                search_query = course_match.group(0)
            else:
                # General course topic
                words = query.split()
                search_query = " ".join(w for w in words if len(w) > 3)[:50]
            
            if search_query:
                tool_result = self.course_search_tool(search_query, limit=3)
                tool_results.append(f"Course Search Results:\n{tool_result}")
        
        # Use program finder for program-related queries
        if any(keyword in query_lower for keyword in ["program", "major", "degree", "certificate"]):
            # Extract field
            fields = ["healthcare", "nursing", "dental", "medical", "carpentry", 
                     "construction", "welding", "automotive", "culinary", "business"]
            
            field = next((f for f in fields if f in query_lower), None)
            
            if field:
                tool_result = self.program_finder_tool(field, limit=3)
                tool_results.append(f"Program Search Results:\n{tool_result}")
        
        # Combine tool results with RAG context
        context_chunks = self.retrieve_context(query)
        context = self.format_context(context_chunks)
        
        if tool_results:
            combined_context = "\n\n".join(tool_results) + "\n\n" + context
        else:
            combined_context = context
        
        # Generate response
        response = self.generate_response(
            query=query,
            context=combined_context,
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
            ],
            "tools_used": ["course_search", "program_finder"] if tool_results else []
        }


# Example usage
if __name__ == "__main__":
    print("🎓 Testing Program Advisor Agent\n")
    print("=" * 60)
    
    agent = ProgramAdvisorAgent()
    
    test_questions = [
        "What courses are in the Carpentry program?",
        "Tell me about CARPT 111",
        "What healthcare programs do you offer?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 60)
        
        result = agent.process_query(question)
        
        print(f"💬 Response:\n{result['response']}")
        print(f"\n📊 Tools used: {result['tools_used']}")
        print(f"📄 Sources: {len(result['sources'])}")
        print("=" * 60)
