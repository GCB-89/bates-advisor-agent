"""
Observability Module - Structured Logging, Tracing, and Metrics

This module provides comprehensive observability for the multi-agent system:
- Structured JSON logging
- Agent tracing (track routing and execution)
- Performance metrics
- Error tracking
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
from rich.console import Console

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Rich console for pretty terminal output
console = Console()


class ObservabilityLogger:
    """
    Centralized logging and tracing for the multi-agent system.
    
    Features:
    - Structured JSON logs
    - Agent execution tracing
    - Performance metrics
    - Error tracking
    """
    
    def __init__(self, log_file: str = "agent_interactions.jsonl"):
        """
        Initialize the logger.
        
        Args:
            log_file: Name of the log file (JSON Lines format)
        """
        self.log_file = LOGS_DIR / log_file
        
        # Configure structlog for JSON logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
        
        # Metrics storage
        self.metrics = {
            "total_queries": 0,
            "total_agent_calls": 0,
            "total_tool_calls": 0,
            "average_response_time": 0.0,
            "agent_usage": {},
            "tool_usage": {}
        }
    
    def log_query(self, session_id: str, query: str, metadata: Optional[Dict] = None):
        """Log an incoming user query."""
        log_entry = {
            "event": "user_query",
            "session_id": session_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self._write_log(log_entry)
        self.metrics["total_queries"] += 1
        
        console.print(f"[bold cyan]📝 Query:[/bold cyan] {query}")
    
    def log_routing(self, session_id: str, query: str, selected_agents: list, reasoning: str):
        """Log router decision."""
        log_entry = {
            "event": "routing_decision",
            "session_id": session_id,
            "query": query,
            "selected_agents": selected_agents,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_log(log_entry)
        console.print(f"[bold yellow]🎯 Routing:[/bold yellow] → {', '.join(selected_agents)}")
        console.print(f"[dim]Reason: {reasoning}[/dim]")
    
    def log_agent_execution(
        self, 
        session_id: str, 
        agent_name: str, 
        start_time: float,
        end_time: float,
        result: str,
        tools_used: list = None
    ):
        """Log agent execution details."""
        execution_time = end_time - start_time
        
        log_entry = {
            "event": "agent_execution",
            "session_id": session_id,
            "agent_name": agent_name,
            "execution_time_ms": round(execution_time * 1000, 2),
            "tools_used": tools_used or [],
            "result_length": len(result),
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_log(log_entry)
        
        # Update metrics
        self.metrics["total_agent_calls"] += 1
        self.metrics["agent_usage"][agent_name] = self.metrics["agent_usage"].get(agent_name, 0) + 1
        
        console.print(f"[bold green]✅ {agent_name}:[/bold green] Completed in {execution_time*1000:.0f}ms")
    
    def log_tool_usage(self, session_id: str, tool_name: str, tool_input: Dict, tool_output: Any):
        """Log custom tool usage."""
        log_entry = {
            "event": "tool_execution",
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output_preview": str(tool_output)[:200],
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_log(log_entry)
        
        # Update metrics
        self.metrics["total_tool_calls"] += 1
        self.metrics["tool_usage"][tool_name] = self.metrics["tool_usage"].get(tool_name, 0) + 1
        
        console.print(f"[bold magenta]🔧 Tool:[/bold magenta] {tool_name}")
    
    def log_response(
        self, 
        session_id: str, 
        response: str, 
        total_time: float,
        sources_used: int = 0
    ):
        """Log final response to user."""
        log_entry = {
            "event": "system_response",
            "session_id": session_id,
            "response_length": len(response),
            "total_time_ms": round(total_time * 1000, 2),
            "sources_used": sources_used,
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_log(log_entry)
        
        # Update average response time
        n = self.metrics["total_queries"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = ((current_avg * (n - 1)) + total_time) / n
        
        console.print(f"[bold blue]💬 Response:[/bold blue] Generated in {total_time*1000:.0f}ms")
        console.print(f"[dim]Sources: {sources_used} | Length: {len(response)} chars[/dim]")
    
    def log_error(self, session_id: str, error: Exception, context: Dict = None):
        """Log errors with full context."""
        log_entry = {
            "event": "error",
            "session_id": session_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_log(log_entry)
        console.print(f"[bold red]❌ Error:[/bold red] {error}")
    
    def log_memory_update(self, session_id: str, memory_type: str, update: Dict):
        """Log memory system updates."""
        log_entry = {
            "event": "memory_update",
            "session_id": session_id,
            "memory_type": memory_type,
            "update": update,
            "timestamp": datetime.now().isoformat()
        }
        
        self._write_log(log_entry)
        console.print(f"[bold purple]💾 Memory Updated:[/bold purple] {memory_type}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return current system metrics."""
        return self.metrics.copy()
    
    def print_metrics_summary(self):
        """Print a formatted summary of metrics."""
        console.print("\n[bold cyan]📊 System Metrics Summary[/bold cyan]")
        console.print("=" * 50)
        console.print(f"Total Queries: {self.metrics['total_queries']}")
        console.print(f"Total Agent Calls: {self.metrics['total_agent_calls']}")
        console.print(f"Total Tool Calls: {self.metrics['total_tool_calls']}")
        console.print(f"Avg Response Time: {self.metrics['average_response_time']*1000:.0f}ms")
        
        if self.metrics['agent_usage']:
            console.print("\n[bold]Agent Usage:[/bold]")
            for agent, count in self.metrics['agent_usage'].items():
                console.print(f"  • {agent}: {count} calls")
        
        if self.metrics['tool_usage']:
            console.print("\n[bold]Tool Usage:[/bold]")
            for tool, count in self.metrics['tool_usage'].items():
                console.print(f"  • {tool}: {count} calls")
        
        console.print("=" * 50 + "\n")
    
    def _write_log(self, log_entry: Dict):
        """Write log entry to file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


# Global logger instance
_logger = None

def get_logger() -> ObservabilityLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = ObservabilityLogger()
    return _logger
