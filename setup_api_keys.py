#!/usr/bin/env python3
"""
API Key Setup Helper
Simple script to help set up your API keys
"""

import os
import shutil

def setup_api_keys():
    """Set up API keys configuration"""
    
    print("🔑 API Key Setup Helper")
    print("=" * 50)
    
    # Check if api_keys.py already exists
    if os.path.exists("config/api_keys.py"):
        print("✅ config/api_keys.py already exists")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Copy template
    if os.path.exists("config/api_keys.example.py"):
        shutil.copy("config/api_keys.example.py", "config/api_keys.py")
        print("✅ Created config/api_keys.py from template")
    else:
        print("❌ Template file config/api_keys.example.py not found")
        return
    
    # Get API key from user
    print("\n🎯 Google Gemini API Key Setup")
    print("1. Visit: https://aistudio.google.com/app/apikey")
    print("2. Create a new API key")
    print("3. Copy your API key\n")
    
    api_key = input("Paste your Google Gemini API key here: ").strip()
    
    if api_key:
        # Update the file
        with open("config/api_keys.py", "r") as f:
            content = f.read()
        
        # Replace the placeholder
        content = content.replace(
            'GOOGLE_GEMINI_API_KEY = "your_gemini_api_key_here"',
            f'GOOGLE_GEMINI_API_KEY = "{api_key}"'
        )
        
        with open("config/api_keys.py", "w") as f:
            f.write(content)
        
        print("✅ API key saved to config/api_keys.py")
        print("\n🚀 Setup complete! You can now run:")
        print("   python launch_gui.py")
    else:
        print("⚠️  No API key provided. You can edit config/api_keys.py manually later.")
    
    print("\n📝 Note: config/api_keys.py is excluded from git for security")

if __name__ == "__main__":
    setup_api_keys() 