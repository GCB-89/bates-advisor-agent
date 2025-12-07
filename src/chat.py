import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from orchestrator import MultiAgentOrchestrator
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()


def print_welcome():
    """Print welcome message."""
    welcome_text = """
# 🎓 Bates Technical College Student Advisor

Welcome! I'm your multi-agent AI advisor powered by Google's Gemini.

**What I can help with:**
- 📚 Programs and courses (Program Advisor)
- 📋 Admissions and enrollment (Admissions Advisor)
- 💰 Financial aid and costs (Financial Advisor)

**How it works:**
1. Ask any question about Bates Tech
2. Your question is routed to the right specialist(s)
3. Get accurate answers from the 459-page catalog!

**Commands:**
- Type your question naturally
- Type 'metrics' to see system stats
- Type 'context' to see your student profile
- Type 'quit' or 'exit' to leave

Let's get started! 🚀
    """
    
    console.print(Panel(Markdown(welcome_text), style="bold cyan"))


def print_response(result: dict):
    """Print agent response nicely formatted."""
    # Response
    console.print("\n[bold green]💬 Response:[/bold green]")
    console.print(Panel(result["response"], style="green"))
    
    # Metadata
    metadata = f"\n🤖 Agents: {', '.join(result['agents_used'])}\n⏱️  Time: {result['execution_time']*1000:.0f}ms\n📄 Sources: {len(result['sources'])} pages\n"
    console.print(metadata, style="dim")


def main():
    """Main chat loop."""
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        console.print("[bold red]❌ Error: GOOGLE_API_KEY not found![/bold red]")
        console.print("\nPlease:")
        console.print("1. Get your API key from: https://aistudio.google.com/apikey")
        console.print("2. Create a .env file (copy from .env.example)")
        console.print("3. Add your key: GOOGLE_API_KEY=your_key_here")
        return
    
    # Print welcome
    print_welcome()
    
    # Initialize orchestrator
    try:
        console.print("[yellow]⏳ Initializing agents...[/yellow]\n")
        orchestrator = MultiAgentOrchestrator()
    except FileNotFoundError:
        console.print("[bold red]❌ Error: Vector database not found![/bold red]")
        console.print("\nPlease run: [bold cyan]python src/ingest_pdf.py[/bold cyan]")
        console.print("This creates the vector database from the PDF.")
        return
    except Exception as e:
        console.print(f"[bold red]❌ Error initializing: {e}[/bold red]")
        return
    
    # Create session
    session_id = orchestrator.create_session()
    console.print(f"[dim]Session ID: {session_id}[/dim]\n")
    
    # Main chat loop
    while True:
        try:
            # Get user input
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            # Check for commands
            if question.lower() in ['quit', 'exit', 'bye']:
                console.print("\n[yellow]👋 Goodbye! Your session has been saved.[/yellow]")
                orchestrator.save_session(session_id)
                break
            
            elif question.lower() == 'metrics':
                orchestrator.print_metrics()
                continue
            
            elif question.lower() == 'context':
                session = orchestrator.get_session(session_id)
                if session:
                    context = session["student_context"]
                    console.print(f"\n[bold]Your Student Profile:[/bold]")
                    console.print(f"Major: {context.major or 'Not set'}")
                    console.print(f"Year: {context.year or 'Not set'}")
                    console.print(f"Interests: {', '.join(context.interests) or 'None yet'}")
                continue
            
            elif not question.strip():
                continue
            
            # Process query
            console.print(f"\n[yellow]🔍 Analyzing your question...[/yellow]")
            
            result = orchestrator.process_query(session_id, question)
            
            # Print response
            print_response(result)
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]👋 Interrupted. Saving session...[/yellow]")
            orchestrator.save_session(session_id)
            break
        
        except Exception as e:
            # FIXED: Use regular print() to avoid Rich markup parsing errors
            print(f"\n❌ Error: {e}")
            print("Please try rephrasing your question.\n")
    
    # Final metrics
    console.print("\n[bold cyan]📊 Session Summary:[/bold cyan]")
    orchestrator.print_metrics()


if __name__ == "__main__":
    main()