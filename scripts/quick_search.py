#!/usr/bin/env python3
"""Quick search test script"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def main():
    from database.enhanced_search import EnhancedSearchEngine
    
    search_engine = EnhancedSearchEngine()
    
    queries = [
        "machine learning",
        "deep learning",
        "neural networks",
        "computer vision",
        "natural language processing"
    ]
    
    print("🔍 Quick Search Test")
    print("=" * 30)
    
    for query in queries:
        print(f"\nSearching: {query}")
        try:
            results = await search_engine.search_literature(query, max_results_per_source=3)
            print(f"  Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result.title[:50]}...")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Show cache stats
    cache_stats = search_engine.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"   Size: {cache_stats['cache_size']}")
    print(f"   Hit rate: {cache_stats['cache_hits']}/{cache_stats['cache_size']}")

if __name__ == "__main__":
    asyncio.run(main())
