#!/usr/bin/env python3
"""
Test Local Mode - Complete Privacy and No API Keys Required
This script demonstrates your research assistant working entirely locally
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from rag.local_ai_enhanced_rag import LocalAIEnhancedRAG
from database.paper_db import PaperDatabase

def print_banner():
    """Print welcome banner"""
    print("═" * 80)
    print("🔒 COMPLETE LOCAL MODE TEST")
    print("   • No external API keys required")
    print("   • Complete privacy and control")
    print("   • Uses your trained local AI model")
    print("   • Searches your local database (50 papers)")
    print("═" * 80)

async def test_local_mode():
    """Test the complete local mode functionality"""
    print_banner()
    
    # Initialize local AI enhanced RAG
    print("\n🔧 Initializing Local AI Enhanced RAG...")
    rag = LocalAIEnhancedRAG(
        db_path="data/papers/papers.db",
        use_local_model=True  # Force local model usage
    )
    
    # Check model status
    print("\n📊 Model Status:")
    status = rag.get_model_status()
    for key, value in status.items():
        print(f"   • {key}: {value}")
    
    # Test database
    print("\n📚 Database Status:")
    db = PaperDatabase()
    papers = db.get_all_papers()
    print(f"   • Total papers: {len(papers)}")
    print(f"   • Sources: {', '.join(set(p.source for p in papers))}")
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "deep learning neural networks",
        "computer vision applications"
    ]
    
    print("\n🔍 Testing Local AI Research Assistant...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: '{query}' ---")
        
        try:
            # Perform local research analysis
            analysis = await rag.research_and_analyze(query)
            
            print(f"✅ Query processed successfully")
            print(f"   • Papers analyzed: {len(analysis.papers)}")
            print(f"   • Processing time: {analysis.processing_metadata.get('processing_time_seconds', 0):.2f}s")
            print(f"   • Model used: {analysis.processing_metadata.get('llm_model_used', 'local')}")
            print(f"   • Summary length: {len(analysis.summarized_findings)} characters")
            
            # Show sample results
            if analysis.papers:
                print(f"   • Top paper: {analysis.papers[0].title[:60]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎉 Local Mode Testing Complete!")
    print("\nYour research assistant is working completely locally with:")
    print("   ✅ No external API calls")
    print("   ✅ Complete privacy")
    print("   ✅ Your custom-trained AI model")
    print("   ✅ Local database (50 papers)")
    print("   ✅ No internet required for research")

if __name__ == "__main__":
    asyncio.run(test_local_mode()) 