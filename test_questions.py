"""Test script for running multiple questions through the advisor"""
from dotenv import load_dotenv
load_dotenv()

from src.orchestrator import MultiAgentOrchestrator

orch = MultiAgentOrchestrator()
session = orch.session_manager.create_session()

questions = [
    "Does the Welding program include hands-on lab experience?",
    "What are the prerequisites for the Nursing program?",
    "How much does tuition cost?",
    "What financial aid is available for veterans?",
    "Can I transfer credits from another college?",
    "What are the admission requirements?",
    "Do you have any IT or computer programs?",
    "How long is the Dental Hygiene program?",
]

for q in questions:
    print("\n" + "=" * 70)
    print(f"Q: {q}")
    print("=" * 70)
    result = orch.process_query(session_id=session, query=q)
    agents = result["agents_used"]
    sources = result["sources"]
    resp = result["response"]
    print(f"AGENTS: {agents}")
    print(f"SOURCES: {len(sources)}")
    print(f"RESPONSE ({len(resp)} chars):")
    if len(resp) > 800:
        print(resp[:800] + "\n...[truncated]")
    else:
        print(resp)
