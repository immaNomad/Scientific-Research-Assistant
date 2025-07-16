#!/usr/bin/env python3
"""
Add AI, ML, and Cybersecurity Papers to Database
Enhanced script to collect 100 high-quality papers focused on AI, ML, and cybersecurity
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from loguru import logger
import aiohttp
import json
from typing import List, Dict, Optional
import random

# Add src and config to path
# Add src and project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

from database.paper_db import PaperDatabase, Paper
from database.paper_collector import PaperCollector
from database.enhanced_search import EnhancedSearchEngine

# Create simplified clients to avoid config import issues
class SimpleArxivClient:
    """Simplified arXiv client without config dependency"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit = 3.0  # seconds between requests
        
    async def search(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search arXiv for papers matching the query"""
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.base_url}?{query_string}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        import xmltodict
                        xml_content = await response.text()
                        parsed = xmltodict.parse(xml_content)
                        
                        feed = parsed.get("feed", {})
                        entries = feed.get("entry", [])
                        
                        if isinstance(entries, dict):
                            entries = [entries]
                        
                        papers = []
                        for entry in entries:
                            try:
                                authors = entry.get("author", [])
                                if isinstance(authors, dict):
                                    authors = [authors]
                                author_names = [author.get("name", "") for author in authors]
                                
                                categories = entry.get("category", [])
                                if isinstance(categories, dict):
                                    categories = [categories]
                                category_list = [cat.get("@term", "") for cat in categories]
                                
                                doi = None
                                if "arxiv:doi" in entry:
                                    doi = entry["arxiv:doi"]["#text"]
                                
                                paper = {
                                    'id': entry["id"].split("/")[-1],
                                    'title': entry["title"].strip(),
                                    'authors': author_names,
                                    'abstract': entry["summary"].strip(),
                                    'published': entry["published"],
                                    'categories': category_list,
                                    'url': entry["id"],
                                    'doi': doi
                                }
                                papers.append(paper)
                            except Exception as e:
                                logger.error(f"Error parsing arXiv entry: {e}")
                                continue
                        
                        logger.info(f"Retrieved {len(papers)} papers from arXiv")
                        return papers
                    else:
                        logger.error(f"arXiv API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error querying arXiv: {e}")
            return []

class SimpleSemanticScholarClient:
    """Simplified Semantic Scholar client"""
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.rate_limit = 1.0  # seconds between requests
        
    async def search_papers(self, query: str, limit: int = 20) -> List[Dict]:
        """Search Semantic Scholar for papers"""
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,citationCount,venue,year,url,externalIds"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        papers = []
                        
                        for paper_data in data.get("data", []):
                            try:
                                authors = []
                                if paper_data.get("authors"):
                                    authors = [author.get("name", "") for author in paper_data["authors"]]
                                
                                # Get DOI from external IDs
                                doi = None
                                if paper_data.get("externalIds") and paper_data["externalIds"].get("DOI"):
                                    doi = paper_data["externalIds"]["DOI"]
                                
                                paper = {
                                    'title': paper_data.get("title", ""),
                                    'authors': authors,
                                    'abstract': paper_data.get("abstract", ""),
                                    'published_date': str(paper_data.get("year", "")),
                                    'citation_count': paper_data.get("citationCount", 0),
                                    'venue': paper_data.get("venue", ""),
                                    'url': paper_data.get("url", ""),
                                    'doi': doi
                                }
                                papers.append(paper)
                            except Exception as e:
                                logger.error(f"Error parsing Semantic Scholar entry: {e}")
                                continue
                        
                        logger.info(f"Retrieved {len(papers)} papers from Semantic Scholar")
                        return papers
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error querying Semantic Scholar: {e}")
            return []

class AIMLCyberSecurityCollector:
    """Specialized collector for AI, ML, and cybersecurity papers"""
    
    def __init__(self, db_path: str = "data/papers/papers.db"):
        self.db = PaperDatabase(db_path)
        self.arxiv_client = SimpleArxivClient()
        self.semantic_client = SimpleSemanticScholarClient()
        
        # Enhanced search queries for AI, ML, and cybersecurity
        self.search_queries = {
            'ai_core': [
                "artificial intelligence machine learning",
                "deep learning neural networks",
                "reinforcement learning algorithms",
                "generative artificial intelligence",
                "large language models LLM",
                "transformer models attention mechanism",
                "computer vision image recognition",
                "natural language processing NLP",
                "AI optimization algorithms",
                "explainable AI interpretable machine learning"
            ],
            'ml_techniques': [
                "machine learning classification regression",
                "supervised learning unsupervised learning",
                "ensemble learning random forest",
                "support vector machines SVM",
                "convolutional neural networks CNN",
                "recurrent neural networks RNN LSTM",
                "graph neural networks GNN",
                "transfer learning domain adaptation",
                "few-shot learning meta learning",
                "federated learning distributed ML"
            ],
            'cybersecurity_ai': [
                "AI cybersecurity threat detection",
                "machine learning intrusion detection",
                "deep learning malware detection",
                "AI network security anomaly detection",
                "artificial intelligence fraud detection",
                "ML phishing detection email security",
                "AI vulnerability assessment",
                "machine learning cyber threat intelligence",
                "deep learning security analytics",
                "AI-powered incident response"
            ],
            'cybersecurity_core': [
                "cybersecurity network security",
                "cryptography encryption algorithms",
                "blockchain security consensus",
                "IoT security internet of things",
                "cloud security infrastructure",
                "mobile security android iOS",
                "web application security OWASP",
                "penetration testing ethical hacking",
                "digital forensics incident response",
                "privacy preserving technologies"
            ],
            'ai_security': [
                "adversarial machine learning attacks",
                "AI model security robustness",
                "machine learning privacy attacks",
                "federated learning security",
                "AI fairness bias mitigation",
                "ML model interpretability",
                "AI ethics algorithmic bias",
                "machine learning differential privacy",
                "AI system security vulnerabilities",
                "trustworthy AI reliable systems"
            ],
            'emerging_topics': [
                "quantum machine learning",
                "AI edge computing mobile",
                "neuromorphic computing AI",
                "AI hardware accelerators",
                "autonomous systems AI",
                "AI robotics control systems",
                "AI healthcare medical diagnosis",
                "AI finance algorithmic trading",
                "AI recommendation systems",
                "AI natural language generation"
            ]
        }
    
    async def collect_papers(self, target_count: int = 100) -> List[str]:
        """Collect target number of papers from all categories"""
        logger.info(f"🎯 Starting collection of {target_count} AI/ML/Cybersecurity papers...")
        
        collected_papers = []
        
        # Calculate papers per category
        categories = list(self.search_queries.keys())
        papers_per_category = target_count // len(categories)
        
        print(f"📊 Collection Plan:")
        print(f"   Total target: {target_count} papers")
        print(f"   Categories: {len(categories)}")
        print(f"   Papers per category: ~{papers_per_category}")
        print(f"   Sources: arXiv + Semantic Scholar")
        
        # Collect from each category
        for category, queries in self.search_queries.items():
            if len(collected_papers) >= target_count:
                break
            
            remaining = target_count - len(collected_papers)
            category_target = min(papers_per_category, remaining)
            
            print(f"\n🔍 Collecting from category: {category}")
            print(f"   Target: {category_target} papers")
            
            category_papers = await self._collect_from_category(category, queries, category_target)
            collected_papers.extend(category_papers)
            
            print(f"   ✅ Collected: {len(category_papers)} papers")
            print(f"   📈 Total so far: {len(collected_papers)}")
        
        print(f"\n🎉 Collection completed!")
        print(f"   Successfully collected: {len(collected_papers)} papers")
        print(f"   Target achieved: {len(collected_papers)}/{target_count} ({len(collected_papers)/target_count*100:.1f}%)")
        
        return collected_papers
    
    async def _collect_from_category(self, category: str, queries: List[str], target_count: int) -> List[str]:
        """Collect papers from a specific category"""
        collected_papers = []
        papers_per_source = target_count // 2  # Split between arXiv and Semantic Scholar
        
        # Collect from arXiv
        arxiv_papers = await self._collect_from_arxiv(queries, papers_per_source)
        collected_papers.extend(arxiv_papers)
        
        # Collect from Semantic Scholar
        semantic_papers = await self._collect_from_semantic_scholar(queries, papers_per_source)
        collected_papers.extend(semantic_papers)
        
        # If we need more papers, collect additional from arXiv
        remaining = target_count - len(collected_papers)
        if remaining > 0:
            additional_papers = await self._collect_from_arxiv(queries, remaining)
            collected_papers.extend(additional_papers)
        
        return collected_papers
    
    async def _collect_from_arxiv(self, queries: List[str], target_count: int) -> List[str]:
        """Collect papers from arXiv"""
        collected_papers = []
        
        # Shuffle queries for variety
        shuffled_queries = queries.copy()
        random.shuffle(shuffled_queries)
        
        for query in shuffled_queries:
            if len(collected_papers) >= target_count:
                break
            
            try:
                papers_needed = min(10, target_count - len(collected_papers))
                papers = await self.arxiv_client.search(query, max_results=papers_needed)
                
                for paper in papers:
                    if len(collected_papers) >= target_count:
                        break
                    
                    # Convert to our Paper format
                    db_paper = Paper(
                        id="",  # Will be generated
                        title=paper['title'],
                        authors=paper['authors'],
                        abstract=paper['abstract'],
                        source='arxiv',
                        url=paper['url'],
                        pdf_path="",
                        folder_path="",
                        published_date=paper['published'],
                        arxiv_id=paper['id'],
                        categories=paper['categories'],
                        doi=paper.get('doi')
                    )
                    
                    # Add to database
                    if self.db.add_paper(db_paper):
                        collected_papers.append(db_paper.id)
                        logger.info(f"✅ Added arXiv paper: {db_paper.title[:60]}...")
                
                # Rate limiting
                await asyncio.sleep(self.arxiv_client.rate_limit)
                
            except Exception as e:
                logger.error(f"Error collecting from arXiv with query '{query}': {e}")
                continue
        
        return collected_papers
    
    async def _collect_from_semantic_scholar(self, queries: List[str], target_count: int) -> List[str]:
        """Collect papers from Semantic Scholar"""
        collected_papers = []
        
        # Shuffle queries for variety
        shuffled_queries = queries.copy()
        random.shuffle(shuffled_queries)
        
        for query in shuffled_queries:
            if len(collected_papers) >= target_count:
                break
            
            try:
                papers_needed = min(10, target_count - len(collected_papers))
                papers = await self.semantic_client.search_papers(query, limit=papers_needed)
                
                for paper in papers:
                    if len(collected_papers) >= target_count:
                        break
                    
                    # Convert to our Paper format
                    db_paper = Paper(
                        id="",  # Will be generated
                        title=paper['title'],
                        authors=paper['authors'],
                        abstract=paper['abstract'],
                        source='semantic_scholar',
                        url=paper['url'],
                        pdf_path="",
                        folder_path="",
                        published_date=paper['published_date'],
                        citation_count=paper['citation_count'],
                        venue=paper['venue'],
                        doi=paper.get('doi')
                    )
                    
                    # Add to database
                    if self.db.add_paper(db_paper):
                        collected_papers.append(db_paper.id)
                        logger.info(f"✅ Added Semantic Scholar paper: {db_paper.title[:60]}...")
                
                # Rate limiting
                await asyncio.sleep(self.semantic_client.rate_limit)
                
            except Exception as e:
                logger.error(f"Error collecting from Semantic Scholar with query '{query}': {e}")
                continue
        
        return collected_papers
    
    def get_database_stats(self) -> Dict:
        """Get current database statistics"""
        return self.db.get_stats()

async def main():
    """Main function to add AI/ML/Cybersecurity papers"""
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    logger.add("logs/ai_ml_cybersecurity_collection.log", rotation="10 MB", retention="7 days")
    
    print("🚀 AI/ML/Cybersecurity Paper Collection Script")
    print("=" * 60)
    
    # Initialize collector
    collector = AIMLCyberSecurityCollector()
    
    # Check current database status
    stats = collector.get_database_stats()
    current_count = stats['total_papers']
    
    print(f"📊 Current database status:")
    print(f"   Total papers: {current_count}")
    print(f"   Papers with PDFs: {stats['papers_with_pdf']}")
    print(f"   Source distribution: {stats['source_distribution']}")
    
    # Set target
    TARGET_PAPERS = 100
    
    print(f"\n🎯 Collection target: {TARGET_PAPERS} new papers")
    print(f"   Focus areas: AI, Machine Learning, Cybersecurity")
    print(f"   Sources: arXiv + Semantic Scholar")
    
    # Confirm before proceeding
    response = input(f"\n❓ Proceed with collecting {TARGET_PAPERS} papers? (y/n): ")
    if response.lower() != 'y':
        print("❌ Collection cancelled")
        return
    
    # Start collection
    start_time = time.time()
    
    try:
        collected_ids = await collector.collect_papers(TARGET_PAPERS)
        
        collection_time = time.time() - start_time
        
        print(f"\n✅ Collection completed in {collection_time:.1f} seconds")
        print(f"   Successfully collected: {len(collected_ids)} papers")
        
        # Final database stats
        final_stats = collector.get_database_stats()
        print(f"\n📊 Final database statistics:")
        print(f"   Total papers: {final_stats['total_papers']}")
        print(f"   Papers with PDFs: {final_stats['papers_with_pdf']}")
        print(f"   Source distribution: {final_stats['source_distribution']}")
        
        # Setup enhanced search engine
        print(f"\n🔧 Setting up enhanced search engine...")
        try:
            search_engine = EnhancedSearchEngine()
            
            # Test search functionality
            print(f"\n🧪 Testing search functionality...")
            test_queries = [
                "machine learning cybersecurity",
                "AI threat detection",
                "deep learning security",
                "artificial intelligence",
                "neural networks"
            ]
            
            for query in test_queries:
                try:
                    results = await search_engine.search_literature(query, max_results_per_source=5)
                    print(f"   '{query}': {len(results)} results")
                except Exception as e:
                    print(f"   '{query}': Error - {e}")
        except Exception as e:
            print(f"   Search engine setup failed: {e}")
        
        print(f"\n🎉 Database enhancement completed successfully!")
        print(f"💡 Your research assistant now has {final_stats['total_papers']} papers!")
        print(f"🔍 Try searching for: 'AI cybersecurity', 'machine learning', 'deep learning'")
        
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        print(f"❌ Collection failed: {e}")
        return 1

if __name__ == "__main__":
    asyncio.run(main()) 