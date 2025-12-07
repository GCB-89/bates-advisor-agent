#!/usr/bin/env python3
"""
Agent Health Check Script

Performs comprehensive health checks on the Bates Technical College Multi-Agent System:
- Environment configuration
- Dependencies verification
- Database connectivity
- Agent initialization
- End-to-end query test
- Performance metrics

Saves results to health_reports/ folder with timestamp.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class HealthChecker:
    """Performs health checks on the multi-agent system."""
    
    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.start_time = datetime.now()
        self.reports_dir = Path("health_reports")
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_all_checks(self) -> Dict:
        """Run all health checks and return results."""
        print("=" * 60)
        print("🏥 AGENT HEALTH CHECK")
        print(f"📅 {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        checks = [
            ("Environment", self.check_environment),
            ("Dependencies", self.check_dependencies),
            ("Vector Database", self.check_vector_database),
            ("Agent Initialization", self.check_agent_initialization),
            ("Router Agent", self.check_router_agent),
            ("Specialist Agents", self.check_specialist_agents),
            ("Memory System", self.check_memory_system),
            ("End-to-End Query", self.check_end_to_end),
        ]
        
        passed = 0
        failed = 0
        warnings = 0
        
        for name, check_func in checks:
            print(f"\n🔍 Checking {name}...")
            try:
                result = check_func()
                self.results[name] = result
                
                if result["status"] == "pass":
                    print(f"   ✅ {name}: PASSED")
                    passed += 1
                elif result["status"] == "warning":
                    print(f"   ⚠️  {name}: WARNING - {result.get('message', '')}")
                    warnings += 1
                else:
                    print(f"   ❌ {name}: FAILED - {result.get('message', '')}")
                    failed += 1
                    
            except Exception as e:
                self.results[name] = {"status": "error", "message": str(e)}
                print(f"   ❌ {name}: ERROR - {e}")
                failed += 1
        
        # Summary
        summary = {
            "timestamp": self.start_time.isoformat(),
            "total_checks": len(checks),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "overall_status": "healthy" if failed == 0 else "unhealthy",
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "checks": self.results
        }
        
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print(f"   Total Checks: {len(checks)}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ⚠️  Warnings: {warnings}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️  Duration: {summary['duration_seconds']:.2f}s")
        print(f"\n   Overall Status: {'🟢 HEALTHY' if failed == 0 else '🔴 UNHEALTHY'}")
        print("=" * 60)
        
        # Save report
        self.save_report(summary)
        
        return summary
    
    def check_environment(self) -> Dict:
        """Check environment variables and configuration."""
        from dotenv import load_dotenv
        load_dotenv()
        
        issues = []
        
        # Check API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"status": "fail", "message": "GOOGLE_API_KEY not set"}
        
        if len(api_key) < 20:
            issues.append("API key seems too short")
        
        # Check model config
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        
        return {
            "status": "pass" if not issues else "warning",
            "message": "; ".join(issues) if issues else "Environment configured correctly",
            "details": {
                "api_key_set": bool(api_key),
                "api_key_length": len(api_key) if api_key else 0,
                "model": model
            }
        }
    
    def check_dependencies(self) -> Dict:
        """Check if all required packages are installed."""
        required = [
            ("google.generativeai", "google-generativeai"),
            ("langchain", "langchain"),
            ("langchain_chroma", "langchain-chroma"),
            ("chromadb", "chromadb"),
            ("rich", "rich"),
            ("dotenv", "python-dotenv"),
            ("structlog", "structlog"),
        ]
        
        missing = []
        installed = []
        
        for module, package in required:
            try:
                __import__(module.split('.')[0])
                installed.append(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            return {
                "status": "fail",
                "message": f"Missing packages: {', '.join(missing)}",
                "details": {"missing": missing, "installed": installed}
            }
        
        return {
            "status": "pass",
            "message": f"All {len(installed)} required packages installed",
            "details": {"installed": installed}
        }
    
    def check_vector_database(self) -> Dict:
        """Check if vector database is accessible and has data."""
        chroma_path = Path("chroma_db")
        
        if not chroma_path.exists():
            return {"status": "fail", "message": "Vector database not found. Run: python src/ingest_pdf.py"}
        
        # Try to load and query
        try:
            from langchain_chroma import Chroma
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = Chroma(
                persist_directory=str(chroma_path),
                embedding_function=embeddings,
                collection_name="bates_catalog"
            )
            
            # Test query
            results = vectorstore.similarity_search("test", k=1)
            doc_count = len(results)
            
            return {
                "status": "pass",
                "message": f"Vector database accessible with data",
                "details": {
                    "path": str(chroma_path),
                    "test_query_results": doc_count
                }
            }
        except Exception as e:
            return {"status": "fail", "message": f"Database error: {str(e)}"}
    
    def check_agent_initialization(self) -> Dict:
        """Check if orchestrator can be initialized."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            start = time.time()
            from orchestrator import MultiAgentOrchestrator
            orch = MultiAgentOrchestrator()
            init_time = time.time() - start
            
            return {
                "status": "pass",
                "message": f"Orchestrator initialized in {init_time:.2f}s",
                "details": {
                    "init_time_seconds": round(init_time, 2),
                    "agents_loaded": list(orch.agents.keys())
                }
            }
        except Exception as e:
            return {"status": "fail", "message": str(e)}
    
    def check_router_agent(self) -> Dict:
        """Check if router agent works correctly."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from agents.router_agent import RouterAgent
            
            router = RouterAgent()
            
            # Test routing
            test_cases = [
                ("What programs are available?", ["program"]),
                ("How do I apply?", ["admissions"]),
                ("What is the tuition cost?", ["financial"]),
            ]
            
            passed = 0
            for query, expected in test_cases:
                result = router.route(query)
                # Check if at least one expected agent is selected
                if any(agent in result["selected_agents"] for agent in expected):
                    passed += 1
            
            if passed == len(test_cases):
                return {
                    "status": "pass",
                    "message": f"Router correctly classified {passed}/{len(test_cases)} test queries"
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Router classified {passed}/{len(test_cases)} correctly"
                }
        except Exception as e:
            return {"status": "fail", "message": str(e)}
    
    def check_specialist_agents(self) -> Dict:
        """Check if all specialist agents can be initialized."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            from agents.program_agent import ProgramAdvisorAgent
            from agents.admissions_agent import AdmissionsAdvisorAgent
            from agents.financial_agent import FinancialAdvisorAgent
            
            agents = {}
            for name, AgentClass in [
                ("Program", ProgramAdvisorAgent),
                ("Admissions", AdmissionsAdvisorAgent),
                ("Financial", FinancialAdvisorAgent)
            ]:
                agent = AgentClass()
                agents[name] = agent.name
            
            return {
                "status": "pass",
                "message": f"All {len(agents)} specialist agents initialized",
                "details": {"agents": agents}
            }
        except Exception as e:
            return {"status": "fail", "message": str(e)}
    
    def check_memory_system(self) -> Dict:
        """Check if memory system works correctly."""
        try:
            from memory.session_manager import SessionManager
            
            manager = SessionManager()
            
            # Test session creation
            session_id = manager.create_session()
            
            # Test message addition
            manager.add_message(session_id, "user", "Test question")
            manager.add_message(session_id, "assistant", "Test response")
            
            # Test retrieval
            session = manager.get_session(session_id)
            
            if session and len(session["conversation_history"].messages) == 2:
                return {
                    "status": "pass",
                    "message": "Memory system working correctly",
                    "details": {"test_session_id": session_id}
                }
            else:
                return {"status": "warning", "message": "Session created but messages not stored correctly"}
                
        except Exception as e:
            return {"status": "fail", "message": str(e)}
    
    def check_end_to_end(self) -> Dict:
        """Perform an end-to-end query test."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            from orchestrator import MultiAgentOrchestrator
            
            orch = MultiAgentOrchestrator()
            session = orch.create_session()
            
            start = time.time()
            result = orch.process_query(session, "What programs are offered?")
            query_time = time.time() - start
            
            # Validate response
            if not result.get("response"):
                return {"status": "fail", "message": "Empty response received"}
            
            if len(result["response"]) < 50:
                return {"status": "warning", "message": "Response seems too short"}
            
            return {
                "status": "pass",
                "message": f"Query completed in {query_time:.2f}s",
                "details": {
                    "query_time_seconds": round(query_time, 2),
                    "response_length": len(result["response"]),
                    "agents_used": result["agents_used"],
                    "sources_count": len(result["sources"])
                }
            }
        except Exception as e:
            return {"status": "fail", "message": str(e)}
    
    def save_report(self, summary: Dict):
        """Save health check report to file."""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"health_report_{timestamp}.json"
        filepath = self.reports_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n💾 Report saved: {filepath}")
        
        # Also save a latest.json for easy access
        latest_path = self.reports_dir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)


def main():
    """Run health check."""
    checker = HealthChecker()
    summary = checker.run_all_checks()
    
    # Exit with appropriate code
    sys.exit(0 if summary["overall_status"] == "healthy" else 1)


if __name__ == "__main__":
    main()
