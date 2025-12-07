"""
Custom Tool: Program Finder

Searches for academic programs at Bates Technical College by field or keyword.
"""

from typing import Dict, List
import re
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings



class ProgramFinderTool:
    """Tool for finding academic programs at Bates Tech."""
    
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """Initialize with vector database."""
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings,
            collection_name="bates_catalog"
        )
    
    def search(self, field: str, limit: int = 5) -> Dict:
        """
        Search for programs in a specific field.
        
        Args:
            field: Field of study (e.g., "healthcare", "construction", "automotive")
            limit: Max number of results
            
        Returns:
            Dict with programs found
        """
        # Enhance query with program-specific terms
        search_query = f"{field} program degree certificate training"
        
        # Search vector database
        results = self.vectorstore.similarity_search(search_query, k=limit * 3)
        
        # Extract programs
        programs = self._extract_programs(results, limit)
        
        return {
            "field": field,
            "found": len(programs),
            "programs": programs
        }
    
    def _extract_programs(self, results: List, limit: int) -> List[Dict]:
        """Extract program information from results."""
        programs = []
        seen_programs = set()
        
        # Keywords that indicate program descriptions
        program_keywords = ["program", "degree", "certificate", "training", "diploma"]
        
        for doc in results:
            content = doc.page_content
            
            # Look for program names (often capitalized and followed by keywords)
            # Example: "Carpentry Program", "Practical Nursing", etc.
            lines = content.split('\n')
            
            for line in lines:
                # Check if line contains program keywords
                if any(keyword in line.lower() for keyword in program_keywords):
                    # Extract potential program name
                    program_name = line.strip()
                    
                    # Clean up
                    program_name = re.sub(r'\s+', ' ', program_name)
                    
                    if len(program_name) > 10 and len(program_name) < 100:
                        if program_name not in seen_programs:
                            seen_programs.add(program_name)
                            
                            # Get context around this line
                            start_idx = max(0, content.find(line) - 200)
                            end_idx = min(len(content), content.find(line) + 300)
                            description = content[start_idx:end_idx].strip()
                            
                            programs.append({
                                "name": program_name,
                                "description": description,
                                "page": doc.metadata.get("page", "N/A")
                            })
                            
                            if len(programs) >= limit:
                                return programs
        
        return programs
    
    def format_results(self, results: Dict) -> str:
        """Format search results."""
        if results["found"] == 0:
            return f"No programs found in '{results['field']}' field"
        
        output = [f"Found {results['found']} programs in {results['field']}:\n"]
        
        for i, program in enumerate(results["programs"], 1):
            output.append(f"{i}. {program['name']}")
            output.append(f"   {program['description'][:200]}...")
            output.append(f"   (Page {program['page']})\n")
        
        return "\n".join(output)


def create_program_finder_tool(chroma_db_path: str = "./chroma_db"):
    """
    Factory function to create a program finder tool.
    
    Returns a function that can be used as a tool by agents.
    """
    tool = ProgramFinderTool(chroma_db_path)
    
    def program_finder(field: str, limit: int = 5) -> str:
        """
        Find academic programs in a specific field at Bates Technical College.
        
        Args:
            field: Field of study (healthcare, construction, automotive, etc.)
            limit: Maximum number of results
            
        Returns:
            Formatted string with program information
        """
        results = tool.search(field, limit)
        return tool.format_results(results)
    
    return program_finder
