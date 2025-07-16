#!/usr/bin/env python3
"""
Setup Local Database Script
Downloads and stores 50+ AI/ML papers in local database with PDFs
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add src and project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from database.paper_db import PaperDatabase
from database.paper_collector import PaperCollector, collect_papers
from database.local_search import LocalSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def setup_database():
    """Set up the local paper database and collect AI/ML papers"""
    
    print("🚀 Setting up local AI/ML paper database...")
    print("=" * 50)
    
    # Initialize database
    db = PaperDatabase()
    print(f"✅ Database initialized at: {db.db_path}")
    print(f"📁 Papers directory: {db.papers_dir}")
    
    # Check current status
    stats = db.get_stats()
    print(f"\n📊 Current database status:")
    print(f"   Total papers: {stats['total_papers']}")
    print(f"   Papers with PDFs: {stats['papers_with_pdf']}")
    print(f"   Source distribution: {stats['source_distribution']}")
    
    # Collect papers if needed
    target_count = 50
    current_count = stats['total_papers']
    
    if current_count < target_count:
        needed = target_count - current_count
        print(f"\n🔍 Need to collect {needed} more papers to reach target of {target_count}")
        
        try:
            # Start paper collection
            print("🌐 Starting paper collection from arXiv and Semantic Scholar...")
            collected_ids = await collect_papers(needed)
            
            print(f"\n✅ Successfully collected {len(collected_ids)} papers!")
            
            # Show updated stats
            new_stats = db.get_stats()
            print(f"\n📊 Updated database status:")
            print(f"   Total papers: {new_stats['total_papers']}")
            print(f"   Papers with PDFs: {new_stats['papers_with_pdf']}")
            print(f"   Source distribution: {new_stats['source_distribution']}")
            
        except Exception as e:
            logger.error(f"Error during paper collection: {e}")
            print(f"❌ Error collecting papers: {e}")
    else:
        print(f"✅ Database already has {current_count} papers (target: {target_count})")
    
    print("\n🔧 Setting up local search engine...")
    search_engine = LocalSearchEngine()
    
    # Test the search engine
    print("\n🧪 Testing search functionality...")
    test_queries = [
        "machine learning",
        "neural networks", 
        "artificial intelligence",
        "deep learning",
        "computer vision"
    ]
    
    for query in test_queries:
        results = await search_engine.search_literature(query, max_results_per_source=3)
        print(f"   '{query}': {len(results)} results")
    
    # Display search engine stats
    search_stats = search_engine.get_database_stats()
    print(f"\n📈 Search engine statistics:")
    print(f"   Available sources: {search_engine.get_available_sources()}")
    if 'domain_distribution' in search_stats:
        print(f"   Domain distribution: {search_stats['domain_distribution']}")
    
    print("\n🎉 Database setup complete!")
    print("=" * 50)
    
    return db, search_engine

def display_sample_papers(db: PaperDatabase, limit: int = 5):
    """Display a few sample papers from the database"""
    print(f"\n📄 Sample papers in database (showing {limit}):")
    print("-" * 60)
    
    papers = db.get_all_papers(limit)
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title[:60]}...")
        print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"   Source: {paper.source}")
        print(f"   PDF: {'✅' if paper.pdf_path and os.path.exists(paper.pdf_path) else '❌'}")
        print(f"   Categories: {paper.categories[:3] if paper.categories else 'None'}")

def main():
    """Main setup function"""
    print("🤖 AI/ML Research Paper Database Setup")
    print("This script will collect 50+ AI/ML papers and set up a local database\n")
    
    try:
        # Run async setup
        db, search_engine = asyncio.run(setup_database())
        
        # Display sample papers
        display_sample_papers(db)
        
        print("\n💡 Next steps:")
        print("1. The database is ready to use")
        print("2. Papers are stored in individual folders with PDFs")
        print("3. You can now run the research assistant with local papers")
        print("4. Use 'python launch_gui.py' to start the GUI")
        
        print(f"\n📁 Database location: {db.db_path}")
        print(f"📁 Papers folder: {db.papers_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 