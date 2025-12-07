#!/usr/bin/env python3
"""
Main Entry Point for Bates Technical College Student Advisor

This file serves as the main entry point that can be launched from VS Code
or command line.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def start_chat_interface():
    """Start the CLI chat interface."""
    try:
        from chat import main as chat_main
        chat_main()
    except Exception as e:
        print(f"❌ Error starting chat interface: {e}")
        print("\nTry running: python src/chat.py")
        sys.exit(1)


def main():
    """Main entry point - starts the CLI chat interface."""
    try:
        start_chat_interface()
    except KeyboardInterrupt:
        print("\n👋 Agent stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()