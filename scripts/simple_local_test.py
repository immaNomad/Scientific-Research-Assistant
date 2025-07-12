#!/usr/bin/env python3
"""
Simple Local AI Test - Demonstrates your local model working without API keys
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Also add root directory to path for config
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

def print_banner():
    print("=" * 70)
    print("🔒 LOCAL AI MODEL TEST - NO API KEYS REQUIRED")
    print("=" * 70)
    print("✅ Complete privacy and control")
    print("✅ Uses your custom-trained local AI model")
    print("✅ No external API calls")
    print("✅ Works offline")
    print("=" * 70)

async def test_local_ai_model():
    """Test the local AI model directly"""
    print_banner()
    
    # Check if model exists
    model_path = Path("data/models/local_ai/local_ai_model.pt")
    if not model_path.exists():
        print("❌ Local AI model not found at:", model_path)
        return False
    
    print(f"✅ Local AI model found: {model_path}")
    print(f"📊 Model size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Import and load the local model
        from models.local_ai_model import LocalAIInference
        
        print("\n🔄 Loading local AI model...")
        model = LocalAIInference(str(model_path))
        print("✅ Local AI model loaded successfully!")
        
        # Test queries
        test_cases = [
            ("machine learning", "This paper presents a novel approach to machine learning algorithms for classification tasks."),
            ("neural networks", "Deep neural networks have shown remarkable performance in computer vision applications."),
            ("computer vision", "The proposed method achieves state-of-the-art results on image recognition benchmarks.")
        ]
        
        print("\n🧪 Testing Local AI Model Performance:")
        print("-" * 50)
        
        for query, context in test_cases:
            print(f"\n📝 Query: '{query}'")
            print(f"📄 Context: {context[:60]}...")
            
            # Test relevance scoring
            relevance = model.calculate_relevance(query, context)
            print(f"🎯 Relevance Score: {relevance:.2f}/5.0")
            
            # Test summary generation
            summary = model.generate_summary(query, context)
            print(f"📋 Summary: {summary[:80]}...")
            
            # Test keyword extraction
            keywords = model.extract_keywords(context)
            print(f"🔑 Keywords: {', '.join(keywords)}")
        
        print("\n🎉 SUCCESS: Your local AI model is working perfectly!")
        print("\n🔒 PRIVACY BENEFITS:")
        print("   • No data sent to external servers")
        print("   • No API keys required")
        print("   • Complete offline capability")
        print("   • Full control over your AI model")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing local AI model: {e}")
        return False

async def test_database():
    """Test the local database"""
    print("\n📚 Testing Local Database:")
    print("-" * 30)
    
    try:
        from database.paper_db import PaperDatabase
        
        db = PaperDatabase()
        papers = db.get_all_papers()
        
        print(f"✅ Database connected successfully")
        print(f"📊 Total papers: {len(papers)}")
        
        if papers:
            sources = set(p.source for p in papers)
            print(f"📦 Sources: {', '.join(sources)}")
            
            # Show sample papers
            print(f"\n📄 Sample papers:")
            for i, paper in enumerate(papers[:3]):
                print(f"   {i+1}. {paper.title[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing database: {e}")
        return False

async def main():
    """Run all tests"""
    print("Starting Local AI System Tests...\n")
    
    # Test 1: Local AI Model
    model_success = await test_local_ai_model()
    
    # Test 2: Local Database
    db_success = await test_database()
    
    # Summary
    print("\n" + "=" * 70)
    print("🏁 TEST RESULTS SUMMARY:")
    print("=" * 70)
    
    if model_success and db_success:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Your research assistant is ready to use in complete local mode:")
        print("   • No API keys needed")
        print("   • Complete privacy")
        print("   • Offline capability")
        print("   • Custom-trained AI model")
        print("   • Local database with your papers")
        
        print("\n🚀 You can now use your research assistant with:")
        print("   python main.py  # Start the research assistant")
        print("   python launch_gui.py  # Start the GUI version")
        
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if not model_success:
            print("   • Train your local model: python scripts/train_local_model.py")
        if not db_success:
            print("   • Check your database: python scripts/setup_local_database.py")

if __name__ == "__main__":
    asyncio.run(main()) 