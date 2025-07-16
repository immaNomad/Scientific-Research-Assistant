#!/usr/bin/env python3
"""
Database Population Script
Automatically populates the database with AI/ML papers from arXiv and Semantic Scholar
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from loguru import logger

# Add src to path
# Add src and project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

from database.paper_db import PaperDatabase
from database.paper_collector import PaperCollector, collect_papers
from database.local_search import LocalSearchEngine
from database.enhanced_search import EnhancedSearchEngine

def setup_logging():
    """Configure logging for the script"""
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    logger.add("logs/database_population.log", rotation="10 MB", retention="7 days")

async def main():
    """Main function to populate database"""
    setup_logging()
    
    print("🚀 AI/ML Research Database Population Script")
    print("=" * 50)
    
    # Configuration
    TARGET_PAPERS = 50
    DB_PATH = "data/papers/papers.db"
    
    # Initialize database
    logger.info("Initializing database...")
    db = PaperDatabase(DB_PATH)
    
    # Check current status
    stats = db.get_stats()
    current_count = stats['total_papers']
    
    print(f"📊 Current database status:")
    print(f"   Papers: {current_count}")
    print(f"   PDFs: {stats['papers_with_pdf']}")
    print(f"   Sources: {stats['source_distribution']}")
    print(f"   Target: {TARGET_PAPERS}")
    
    if current_count >= TARGET_PAPERS:
        print(f"✅ Database already has {current_count} papers (target: {TARGET_PAPERS})")
        
        # Still run search engine setup
        print("\n🔧 Setting up enhanced search engine...")
        search_engine = EnhancedSearchEngine(DB_PATH)
        await test_search_engine(search_engine)
        
        print("\n🎉 Database is ready for use!")
        return
    
    # Calculate papers needed
    needed = TARGET_PAPERS - current_count
    print(f"\n🔍 Need to collect {needed} papers")
    
    # Start collection process
    print("\n🌐 Starting paper collection...")
    logger.info(f"Starting collection of {needed} papers")
    
    start_time = time.time()
    
    try:
        # Collect papers
        collected_ids = await collect_papers(needed)
        
        collection_time = time.time() - start_time
        
        print(f"\n✅ Collection completed in {collection_time:.1f} seconds")
        print(f"   Successfully collected: {len(collected_ids)} papers")
        
        # Update stats
        final_stats = db.get_stats()
        print(f"\n📊 Final database status:")
        print(f"   Total papers: {final_stats['total_papers']}")
        print(f"   Papers with PDFs: {final_stats['papers_with_pdf']}")
        print(f"   Source distribution: {final_stats['source_distribution']}")
        
        # Setup enhanced search engine
        print("\n🔧 Setting up enhanced search engine...")
        search_engine = EnhancedSearchEngine(DB_PATH)
        await test_search_engine(search_engine)
        
        print("\n🎉 Database population completed successfully!")
        print("💡 You can now use the research assistant with your local database.")
        
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        print(f"❌ Collection failed: {e}")
        return 1

async def test_search_engine(search_engine: EnhancedSearchEngine):
    """Test the search engine with sample queries"""
    print("\n🧪 Testing search functionality...")
    
    test_queries = [
        "machine learning",
        "neural networks", 
        "deep learning",
        "computer vision",
        "natural language processing"
    ]
    
    for query in test_queries:
        try:
            results = await search_engine.search_literature(query, max_results_per_source=5)
            print(f"   '{query}': {len(results)} results")
        except Exception as e:
            print(f"   '{query}': Error - {e}")
    
    # Display enhanced search stats
    cache_stats = search_engine.get_cache_stats()
    index_stats = search_engine.get_index_stats()
    
    print(f"\n📈 Enhanced search engine statistics:")
    print(f"   Cache size: {cache_stats['cache_size']}")
    print(f"   Index size: {index_stats['total_papers_indexed']} papers")
    print(f"   Keywords indexed: {index_stats['keyword_index_size']}")
    print(f"   Authors indexed: {index_stats['author_index_size']}")

def interactive_mode():
    """Interactive mode for database management"""
    print("\n🔧 Interactive Database Management")
    print("Available commands:")
    print("  populate - Populate database with papers")
    print("  status - Show database status")
    print("  search <query> - Test search functionality")
    print("  clear-cache - Clear search cache")
    print("  exit - Exit interactive mode")
    
    db = PaperDatabase()
    search_engine = EnhancedSearchEngine()
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "exit":
                break
            elif command == "populate":
                asyncio.run(main())
            elif command == "status":
                stats = db.get_stats()
                print(f"Papers: {stats['total_papers']}")
                print(f"PDFs: {stats['papers_with_pdf']}")
                print(f"Sources: {stats['source_distribution']}")
            elif command.startswith("search "):
                query = command[7:]
                if query:
                    results = asyncio.run(search_engine.search_literature(query, max_results_per_source=5))
                    print(f"Found {len(results)} results for '{query}'")
                    for i, result in enumerate(results[:3], 1):
                        print(f"  {i}. {result.title[:60]}...")
            elif command == "clear-cache":
                search_engine.clear_cache()
                print("Search cache cleared")
            else:
                print("Unknown command. Type 'exit' to quit.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        asyncio.run(main()) 