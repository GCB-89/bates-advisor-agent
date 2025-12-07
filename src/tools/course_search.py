"""
Custom Tool: Course Search

A tool that allows agents to search for specific courses by:
- Course code (e.g., "NURS 101")
- Course name (e.g., "Introduction to Nursing")
- Topic/keyword (e.g., "anatomy")
"""

from typing import Dict, List
import re
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class CourseSearchTool:
    """
    Custom tool for searching courses in the Bates catalog.
    
    Compatible with both LangChain and Google ADK patterns.
    """
    
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        """Initialize the course search tool with vector database."""
        # Load vector database
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings,
            collection_name="bates_catalog"
        )
    
    def search(self, query: str, limit: int = 5) -> Dict:
        """
        Execute the course search.
        
        Args:
            query: Search query
            limit: Max number of results
            
        Returns:
            Dict with courses and metadata
        """
        # Enhance query for better course matching
        search_query = self._enhance_query(query)
        
        # Perform similarity search
        results = self.vectorstore.similarity_search(
            search_query,
            k=limit * 2  # Get more results to filter
        )
        
        # Extract and format courses
        courses = self._extract_courses(results, limit)
        
        return {
            "query": query,
            "found": len(courses),
            "courses": courses
        }
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with course-specific terms."""
        # If query looks like a course code, prioritize exact matching
        if re.match(r'^[A-Z]{3,5}\s*\d{3}', query.upper()):
            return query.upper()
        
        return f"{query} course description"
    
    def _extract_courses(self, results: List, limit: int) -> List[Dict]:
        """Extract course information from search results."""
        courses = []
        seen_courses = set()
        
        for doc in results:
            # Extract course code if present
            content = doc.page_content
            course_match = re.search(r'([A-Z]{3,5})\s*(\d{3})\s*-\s*([^\(]+)', content)
            
            if course_match:
                code = f"{course_match.group(1)} {course_match.group(2)}"
                name = course_match.group(3).strip()
                
                # Avoid duplicates
                if code in seen_courses:
                    continue
                
                seen_courses.add(code)
                
                # Extract credits
                credit_match = re.search(r'\((\d+)\)', content)
                credits = credit_match.group(1) if credit_match else "N/A"
                
                # Extract description
                desc_start = content.find(name) + len(name)
                description = content[desc_start:desc_start + 200].strip()
                
                courses.append({
                    "code": code,
                    "name": name,
                    "credits": credits,
                    "description": description,
                    "page": doc.metadata.get("page", "N/A")
                })
                
                if len(courses) >= limit:
                    break
        
        return courses
    
    def format_results(self, results: Dict) -> str:
        """Format search results as a readable string."""
        if results["found"] == 0:
            return f"No courses found matching '{results['query']}'"
        
        output = [f"Found {results['found']} courses:\n"]
        
        for i, course in enumerate(results["courses"], 1):
            output.append(f"{i}. {course['code']} - {course['name']}")
            output.append(f"   Credits: {course['credits']}")
            output.append(f"   {course['description'][:150]}...")
            output.append(f"   (Page {course['page']})\n")
        
        return "\n".join(output)


def create_course_search_tool(chroma_db_path: str = "./chroma_db"):
    """
    Factory function to create a course search tool.
    
    Returns a function that can be used as a tool by agents.
    """
    tool = CourseSearchTool(chroma_db_path)
    
    def course_search(query: str, limit: int = 5) -> str:
        """
        Search for courses in the Bates Technical College catalog.
        
        Args:
            query: Course code, name, or topic to search for
            limit: Maximum number of results (default: 5)
            
        Returns:
            Formatted string with course information
        """
        results = tool.search(query, limit)
        return tool.format_results(results)
    
    return course_search
