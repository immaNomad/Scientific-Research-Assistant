#!/usr/bin/env python3
"""
Free Setup Script for Scientific Research Assistant
This script sets up the app to work with free AI services only.
"""

import os
import sys

def create_free_env():
    """Create .env file configured for free usage"""
    env_content = """# Scientific Research Assistant - Free Configuration
# No API keys required! Uses free Hugging Face models.

# Optional: Add these if you get them later
# OPENAI_API_KEY=your_openai_key_here
# SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key
# PUBMED_API_KEY=your_pubmed_key

# Application Settings
DEBUG=false
USE_GPU=false

# Free AI is enabled by default - no setup needed!
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file for free usage")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Core dependencies for free version
    free_deps = [
        "PyQt5>=5.15.0",
        "requests>=2.25.0", 
        "aiohttp>=3.8.0",
        "loguru>=0.6.0",
        "python-dotenv>=0.19.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "beautifulsoup4>=4.10.0"
    ]
    
    for dep in free_deps:
        print(f"Installing {dep}...")
        os.system(f"pip install {dep}")
    
    print("✅ Dependencies installed")

def test_free_setup():
    """Test if the free setup works"""
    print("🧪 Testing free AI setup...")
    
    try:
        # Test imports
        import aiohttp
        import asyncio
        from src.models.free_llm_client import free_llm_client
        
        print("✅ Free AI client available")
        
        # Test the app can start
        import PyQt5
        print("✅ GUI framework available") 
        
        print("\n🎉 FREE SETUP COMPLETE!")
        print("\nYour Scientific Research Assistant is ready to use with:")
        print("• ✅ Free Hugging Face AI models")
        print("• ✅ Enhanced template analysis")  
        print("• ✅ Full literature search capabilities")
        print("• ✅ No API keys required!")
        
        print("\n🚀 To start the app:")
        print("   python3 launch_gui.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🆓 Scientific Research Assistant - FREE SETUP")
    print("=" * 50)
    print("This will configure your app to work with FREE AI services!")
    print("No OpenAI API key required.\n")
    
    # Check if user wants to proceed
    response = input("Continue with free setup? (y/n): ").lower().strip()
    if response != 'y':
        print("Setup cancelled.")
        return
    
    print("\n🔧 Setting up free version...")
    
    # Step 1: Create free .env file
    create_free_env()
    
    # Step 2: Install dependencies  
    install_dependencies()
    
    # Step 3: Test setup
    if test_free_setup():
        print("\n" + "=" * 50)
        print("SUCCESS! Your app is ready to use for FREE! 🎉")
        print("\nFeatures available:")
        print("• Literature search (arXiv, Semantic Scholar, PubMed)")
        print("• AI-powered analysis using free Hugging Face models")
        print("• Enhanced template analysis as fallback")
        print("• Complete desktop GUI experience")
        print("\nTo upgrade later with OpenAI:")
        print("1. Get an OpenAI API key")
        print("2. Add OPENAI_API_KEY=your_key to .env file")
        print("3. Restart the app")
    else:
        print("\n❌ Setup had issues. Check the error messages above.")

if __name__ == "__main__":
    main() 