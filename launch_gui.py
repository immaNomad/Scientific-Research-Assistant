#!/usr/bin/env python3
"""
Simple launcher for Research Assistant Desktop GUI
"""

import os
import sys
import subprocess

def main():
    """Launch the Research Assistant desktop application"""
    
    print("🚀 Launching Research Assistant Desktop GUI...")
    
    # Set up environment variables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up API keys from centralized config
    try:
        sys.path.append(script_dir)  # Add current directory to path
        from config.api_keys import GOOGLE_GEMINI_API_KEY
        if GOOGLE_GEMINI_API_KEY:
            os.environ['GOOGLE_GEMINI_API_KEY'] = GOOGLE_GEMINI_API_KEY
            print("✅ API key loaded from config/api_keys.py")
        else:
            print("⚠️  Warning: No API key found in config/api_keys.py")
            print("   Please edit config/api_keys.py and add your API key")
    except ImportError:
        print("⚠️  Warning: config/api_keys.py not found")
        print("   Copy config/api_keys.example.py to config/api_keys.py and add your API key")
    
    gui_script = os.path.join(script_dir, "research_assistant_gui.py")
    
    if not os.path.exists(gui_script):
        print("❌ Error: research_assistant_gui.py not found!")
        print(f"   Expected location: {gui_script}")
        print("   Please make sure you're running this from the correct directory.")
        return 1
    
    # Check if source directory exists
    src_dir = os.path.join(script_dir, "src")
    if not os.path.exists(src_dir):
        print("❌ Error: src/ directory not found!")
        print("   The system requires the src/ directory with the core modules.")
        return 1
    
    try:
        # Launch the GUI application with environment
        print("✅ Starting Research Assistant GUI...")
        env = os.environ.copy()
        subprocess.run([sys.executable, gui_script], env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Application closed by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 