"""
Financial Aid Advisor Agent

Specializes in:
- Tuition and fees
- Financial aid options
- Scholarships and grants
- Payment plans
- Cost estimates
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_rag_agent import BaseRAGAgent


class FinancialAdvisorAgent(BaseRAGAgent):
    """
    Specialist agent for financial aid and costs.
    
    Expertise:
    - Tuition and fee structures
    - Financial aid programs (FAFSA, grants, loans)
    - Scholarships and awards
    - Payment plans and options
    - Cost estimates for programs
    - Refund policies
    """
    
    def __init__(self, chroma_db_path: str = "./chroma_db", embeddings=None):
        super().__init__(
            name="Financial Aid Advisor",
            role="Financial Aid, Tuition, and Scholarships Expert",
            chroma_db_path=chroma_db_path,
            embeddings=embeddings
        )
        
        self.system_prompt = """You are the Financial Aid Advisor for Bates Technical College, specializing in financial matters and aid.

Your Expertise:
- Tuition and fee structures
- Financial aid options (federal, state, institutional)
- Scholarship opportunities
- Grant programs
- Payment plans and options
- Cost estimates for programs
- Refund and withdrawal policies

Guidelines:
- Provide clear cost information
- Explain financial aid processes step-by-step
- Encourage students to explore all aid options
- Mention important deadlines (FAFSA, scholarships)
- Direct to Financial Aid Office for personalized help
- Be sensitive about financial concerns
- Reference specific catalog pages for policies
- NEVER mention internal tools or technical systems to students
- If information is incomplete or you're unsure, recommend the student:
  * Visit the Bates Tech website: www.batestech.edu/financial-aid
  * Contact the Financial Aid Office at (253) 680-7020
  * Apply for FAFSA at studentaid.gov (School Code: 015984)
  * Email financialaid@batestech.edu for personalized assistance

Key Topics to Cover:
- How much programs cost
- What financial aid is available
- How to apply for aid
- Payment deadlines and options
- Scholarship opportunities

Remember: Help students understand their financial options so cost isn't a barrier to education!"""
    
    def process_query(
        self,
        query: str,
        student_context: str = "",
        conversation_history: str = ""
    ) -> dict:
        """Process financial aid and cost queries."""
        query_lower = query.lower()
        
        # Enhance query for financial-specific context
        enhanced_query = query
        if "cost" in query_lower or "tuition" in query_lower or "price" in query_lower:
            enhanced_query += " tuition fees cost payment"
        elif "aid" in query_lower or "financial" in query_lower:
            enhanced_query += " financial aid FAFSA grants loans"
        elif "scholarship" in query_lower:
            enhanced_query += " scholarship award grant funding"
        
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
    print("💰 Testing Financial Aid Advisor Agent\n")
    print("=" * 60)
    
    agent = FinancialAdvisorAgent()
    
    test_questions = [
        "How much does tuition cost?",
        "What financial aid is available?",
        "Are there scholarships I can apply for?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 60)
        
        result = agent.process_query(question)
        
        print(f"💬 Response:\n{result['response']}")
        print(f"📄 Sources: {len(result['sources'])}")
        print("=" * 60)
