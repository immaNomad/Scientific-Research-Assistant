#!/usr/bin/env python3
"""
RAG Research Assistant - GUI Launcher
Simple script to launch the desktop GUI application
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Launch the RAG Desktop GUI"""
    print("🚀 Launching RAG Research Assistant Desktop GUI...")
    
    # Get the script directory
    script_dir = Path(__file__).parent
    gui_script = script_dir / "rag_desktop_gui.py"
    
    if not gui_script.exists():
        print(f"❌ GUI script not found: {gui_script}")
        return 1
    
    try:
        # Launch the GUI application
        subprocess.run([sys.executable, str(gui_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch GUI: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 GUI launcher interrupted")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 