"""
Base RAG Agent for Google ADK

Provides RAG (Retrieval-Augmented Generation) capabilities to all specialist agents.
"""

import os
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai


# Shared embeddings instance (singleton pattern)
_shared_embeddings = None

def get_shared_embeddings():
    """Get or create shared embeddings instance."""
    global _shared_embeddings
    if _shared_embeddings is None:
        _shared_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return _shared_embeddings


class BaseRAGAgent:
    """
    Base class for all RAG-powered agents.
    
    Provides:
    - Vector database access
    - Context retrieval
    - Response generation
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        chroma_db_path: str = "./chroma_db",
        model_name: str = None,
        embeddings = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name
            role: Agent's specialty/role
            chroma_db_path: Path to vector database
            model_name: Gemini model to use
            embeddings: Shared embeddings instance (optional)
        """
        self.name = name
        self.role = role
        
        # Use shared embeddings if provided, otherwise create new
        if embeddings is None:
            embeddings = get_shared_embeddings()
        
        self.vectorstore = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings,
            collection_name="bates_catalog"
    )
        
        # Initialize Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.model = genai.GenerativeModel(self.model_name)
    
    def retrieve_context(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve relevant context from vector database.
        Uses adaptive retrieval: fewer chunks for simple queries, more for complex ones.
        
        Args:
            query: User query
            k: Number of chunks to retrieve (auto-determined if None)
            
        Returns:
            List of relevant document chunks
        """
        # Adaptive retrieval based on query complexity
        if k is None:
            word_count = len(query.split())
            has_multiple_questions = '?' in query and query.count('?') > 1
            has_complex_keywords = any(kw in query.lower() for kw in ['compare', 'difference', 'all', 'list', 'everything'])
            
            if word_count < 8 and not has_multiple_questions:
                k = 2  # Simple query
            elif has_complex_keywords or has_multiple_questions or word_count > 20:
                k = 5  # Complex query
            else:
                k = 3  # Medium query
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", "Bates Catalog")
            }
            for doc in results
        ]
    
    def format_context(self, context_chunks: List[Dict]) -> str:
        """Format retrieved context for prompt."""
        if not context_chunks:
            return "No relevant context found."
        
        formatted = []
        for i, chunk in enumerate(context_chunks, 1):
            formatted.append(f"[Source {i} - Page {chunk['page']}]")
            formatted.append(chunk['content'])
            formatted.append("")
        
        return "\n".join(formatted)
    
    def generate_response(
        self,
        query: str,
        context: str,
        student_context: str = "",
        conversation_history: str = "",
        system_prompt: str = None
    ) -> str:
        """
        Generate response using Gemini with RAG context.
        
        Args:
            query: User question
            context: Retrieved context from vector DB
            student_context: Student background/preferences
            conversation_history: Recent conversation
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Generated response
        """
        # Build prompt
        if system_prompt is None:
            system_prompt = f"""You are {self.name}, a specialized {self.role} advisor for Bates Technical College.

Your expertise: {self.role}

Use the context provided to answer student questions accurately and helpfully."""
        
        prompt_parts = [system_prompt]
        
        if student_context:
            prompt_parts.append(f"\nStudent Context:\n{student_context}")
        
        if conversation_history:
            prompt_parts.append(f"\nRecent Conversation:\n{conversation_history}")
        
        prompt_parts.append(f"\nRelevant Information from Bates Catalog:\n{context}")
        prompt_parts.append(f"\nStudent Question: {query}")
        prompt_parts.append("\nProvide a helpful, accurate answer based on the information above:")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Generate response
        response = self.model.generate_content(full_prompt)
        
        return response.text
    
    def process_query(
        self,
        query: str,
        student_context: str = "",
        conversation_history: str = ""
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve → generate → return.
        
        Args:
            query: User question
            student_context: Student background
            conversation_history: Recent conversation
            
        Returns:
            Dict with response and sources
        """
        # Retrieve relevant context
        context_chunks = self.retrieve_context(query)
        context = self.format_context(context_chunks)
        
        # Generate response
        response = self.generate_response(
            query=query,
            context=context,
            student_context=student_context,
            conversation_history=conversation_history
        )
        
        return {
            "agent": self.name,
            "response": response,
            "sources": [
                {"page": chunk["page"], "preview": chunk["content"][:150]}
                for chunk in context_chunks
            ]
        }


# Example usage
if __name__ == "__main__":
    # Test the base agent
    agent = BaseRAGAgent(
        name="Test Agent",
        role="testing specialist"
    )
    
    test_query = "What programs does Bates Tech offer in healthcare?"
    
    print(f"Testing BaseRAGAgent with query: '{test_query}'")
    print("=" * 60)
    
    result = agent.process_query(test_query)
    
    print(f"\nAgent: {result['agent']}")
    print(f"Response: {result['response']}")
    print(f"\nSources used: {len(result['sources'])}")
