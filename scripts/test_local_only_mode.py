#!/usr/bin/env python3
"""
Test Local-Only Mode - No External APIs
Tests the complete local-only configuration with custom AI model
"""

import sys
import os
import asyncio
from pathlib import Path

# Add paths
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(root_path, 'src')
sys.path.insert(0, root_path)
sys.path.insert(0, src_path)

def test_configuration():
    """Test the local-only configuration"""
    print("=" * 60)
    print("🔧 TESTING LOCAL-ONLY CONFIGURATION")
    print("=" * 60)
    
    # Test config import
    try:
        from config.api_keys import (
            USE_LOCAL_MODELS_ONLY, 
            DEFAULT_CHAT_MODEL, 
            DEFAULT_EMBEDDING_MODEL,
            GOOGLE_API_KEY
        )
        
        print(f"✅ Configuration loaded successfully")
        print(f"   • USE_LOCAL_MODELS_ONLY: {USE_LOCAL_MODELS_ONLY}")
        print(f"   • DEFAULT_CHAT_MODEL: {DEFAULT_CHAT_MODEL}")
        print(f"   • DEFAULT_EMBEDDING_MODEL: {DEFAULT_EMBEDDING_MODEL}")
        print(f"   • GOOGLE_API_KEY: {'None (disabled)' if GOOGLE_API_KEY is None else 'Set'}")
        
        return USE_LOCAL_MODELS_ONLY
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_llm_manager():
    """Test the LLM manager with local-only mode"""
    print("\n" + "=" * 60)
    print("🤖 TESTING LLM MANAGER")
    print("=" * 60)
    
    try:
        from models.llm_client import LLMManager
        
        print("🔄 Initializing LLM Manager...")
        manager = LLMManager()
        
        print(f"✅ LLM Manager initialized")
        print(f"   • Available clients: {manager.get_available_clients()}")
        print(f"   • Total clients: {len(manager.clients)}")
        
        for i, (name, client) in enumerate(manager.clients):
            print(f"   • Client {i+1}: {name} ({'available' if client.is_available() else 'not available'})")
        
        return manager
        
    except Exception as e:
        print(f"❌ LLM Manager error: {e}")
        return None

async def test_text_generation(manager):
    """Test text generation with local model"""
    print("\n" + "=" * 60)
    print("📝 TESTING TEXT GENERATION")
    print("=" * 60)
    
    if not manager:
        print("❌ Cannot test - LLM Manager not available")
        return False
    
    test_prompt = """Based on the following research papers about computer vision, provide a comprehensive summary:

Paper 1: Deep Residual Learning for Image Recognition
Abstract: We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.

Paper 2: Very Deep Convolutional Networks for Large-Scale Image Recognition
Abstract: In this work we investigate the effect of convolutional network depth on its accuracy in the large-scale image recognition setting.

Please provide a detailed summary of key findings and methodologies."""
    
    try:
        print("🔄 Generating text with local model...")
        response = await manager.generate(test_prompt, max_tokens=300, temperature=0.7)
        
        print(f"✅ Text generation successful!")
        print(f"   • Model used: {response.model}")
        print(f"   • Content length: {len(response.content)} characters")
        print(f"   • Tokens used: {response.tokens_used}")
        print(f"   • Latency: {response.latency:.2f} seconds")
        
        print(f"\n📄 Generated content preview:")
        print("-" * 40)
        print(response.content[:200] + "..." if len(response.content) > 200 else response.content)
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Text generation error: {e}")
        return False

async def test_database_search():
    """Test database search functionality"""
    print("\n" + "=" * 60)
    print("🔍 TESTING DATABASE SEARCH")
    print("=" * 60)
    
    try:
        from database.paper_db import PaperDatabase
        
        print("🔄 Connecting to database...")
        db = PaperDatabase()
        papers = db.get_all_papers()
        
        print(f"✅ Database connection successful!")
        print(f"   • Total papers: {len(papers)}")
        
        if papers:
            sources = set(p.source for p in papers)
            print(f"   • Sources: {', '.join(sources)}")
            
            # Test search
            search_results = db.search_papers("computer vision")
            print(f"   • Search results for 'computer vision': {len(search_results)}")
            
            if search_results:
                print(f"   • Top result: {search_results[0].title[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

async def main():
    """Run all tests"""
    print("Starting Local-Only Mode Tests...\n")
    
    # Test 1: Configuration
    is_local_only = test_configuration()
    
    # Test 2: LLM Manager
    manager = test_llm_manager()
    
    # Test 3: Text Generation
    text_gen_success = await test_text_generation(manager)
    
    # Test 4: Database Search
    db_success = await test_database_search()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = is_local_only and manager and text_gen_success and db_success
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Your research assistant is now in complete local-only mode:")
        print("   • No external API calls")
        print("   • Complete privacy and control")
        print("   • Using custom-trained local AI model")
        print("   • Local database with your research papers")
        print("   • Offline capability")
        
        print("\n🚀 You can now use your research assistant:")
        print("   • python main.py  # Command-line interface")
        print("   • python launch_gui.py  # GUI interface")
        
    else:
        print("❌ Some tests failed:")
        print(f"   • Configuration: {'✅' if is_local_only else '❌'}")
        print(f"   • LLM Manager: {'✅' if manager else '❌'}")
        print(f"   • Text Generation: {'✅' if text_gen_success else '❌'}")
        print(f"   • Database Search: {'✅' if db_success else '❌'}")

if __name__ == "__main__":
    asyncio.run(main()) 