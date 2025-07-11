"""
Paper Collection System
Downloads AI/ML papers from arXiv and Semantic Scholar, stores PDFs in separate folders
"""

import asyncio
import aiohttp
import aiofiles
import os
import sys
import time
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import json
import logging
from urllib.parse import urljoin, urlparse
import feedparser
import re

from .paper_db import PaperDatabase, Paper

logger = logging.getLogger(__name__)

@dataclass
class PaperSource:
    """Configuration for paper sources"""
    name: str
    base_url: str
    rate_limit: float  # seconds between requests
    max_results: int = 100

class PaperCollector:
    """Collects and downloads AI/ML papers from arXiv and Semantic Scholar"""
    
    def __init__(self, db: PaperDatabase):
        self.db = db
        self.session = None
        
        # Configure sources
        self.sources = {
            'arxiv': PaperSource(
                name='arxiv',
                base_url='http://export.arxiv.org/api/query',
                rate_limit=3.0,  # 3 seconds between requests
                max_results=100
            ),
            'semantic_scholar': PaperSource(
                name='semantic_scholar', 
                base_url='https://api.semanticscholar.org/graph/v1',
                rate_limit=1.0,  # 1 second between requests
                max_results=100
            )
        }
        
        # AI/ML search queries for different domains
        self.ai_ml_queries = [
            # Core AI/ML
            "artificial intelligence machine learning",
            "deep learning neural networks",
            "computer vision image recognition", 
            "natural language processing nlp",
            "reinforcement learning",
            "generative AI large language models",
            "transformer models attention mechanism",
            "convolutional neural networks CNN",
            "recurrent neural networks RNN LSTM",
            "graph neural networks",
            
            # Specific AI Applications
            "AI robotics autonomous systems",
            "machine learning healthcare medical",
            "AI computer security cybersecurity",
            "recommendation systems collaborative filtering",
            "speech recognition voice processing",
            "AI optimization algorithms",
            "federated learning distributed AI",
            "AI edge computing mobile",
            "explainable AI interpretable machine learning",
            "AI ethics fairness bias",
            
            # Advanced ML Techniques
            "transfer learning domain adaptation",
            "few-shot learning meta learning",
            "unsupervised learning clustering",
            "semi-supervised learning",
            "active learning query strategies",
            "ensemble learning boosting bagging",
            "gaussian processes bayesian optimization",
            "generative adversarial networks GAN",
            "variational autoencoders VAE",
            "diffusion models text-to-image",
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Research-Assistant/1.0 (Educational Purpose)'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def collect_ai_ml_papers(self, target_count: int = 50) -> List[str]:
        """
        Collect target number of AI/ML papers from both sources
        Returns list of paper IDs that were successfully added
        """
        logger.info(f"Starting collection of {target_count} AI/ML papers...")
        
        collected_papers = []
        papers_per_source = target_count // 2  # Split between arXiv and Semantic Scholar
        
        try:
            # Collect from arXiv
            logger.info("Collecting papers from arXiv...")
            arxiv_papers = await self._collect_from_arxiv(papers_per_source)
            collected_papers.extend(arxiv_papers)
            
            # Collect from Semantic Scholar  
            logger.info("Collecting papers from Semantic Scholar...")
            semantic_papers = await self._collect_from_semantic_scholar(papers_per_source)
            collected_papers.extend(semantic_papers)
            
            # If we need more papers, collect additional ones
            remaining = target_count - len(collected_papers)
            if remaining > 0:
                logger.info(f"Collecting {remaining} additional papers...")
                additional_arxiv = await self._collect_from_arxiv(remaining)
                collected_papers.extend(additional_arxiv)
            
            logger.info(f"Successfully collected {len(collected_papers)} papers")
            return collected_papers
            
        except Exception as e:
            logger.error(f"Error during paper collection: {e}")
            return collected_papers
    
    async def _collect_from_arxiv(self, target_count: int) -> List[str]:
        """Collect papers from arXiv"""
        collected_papers = []
        queries_used = 0
        
        for query in self.ai_ml_queries:
            if len(collected_papers) >= target_count:
                break
                
            try:
                logger.info(f"Searching arXiv for: {query}")
                papers = await self._search_arxiv(query, min(20, target_count - len(collected_papers)))
                
                for paper_data in papers:
                    if len(collected_papers) >= target_count:
                        break
                    
                    # Create Paper object
                    paper = Paper(
                        id="",  # Will be generated
                        title=paper_data['title'],
                        authors=paper_data['authors'],
                        abstract=paper_data['abstract'],
                        source='arxiv',
                        url=paper_data['url'],
                        pdf_path="",  # Will be set when PDF is downloaded
                        folder_path="",  # Will be set by database
                        published_date=paper_data.get('published_date'),
                        arxiv_id=paper_data.get('arxiv_id'),
                        categories=paper_data.get('categories', [])
                    )
                    
                    # Add to database
                    if self.db.add_paper(paper):
                        # Download PDF
                        if await self._download_arxiv_pdf(paper):
                            collected_papers.append(paper.id)
                            logger.info(f"Added arXiv paper: {paper.title[:50]}...")
                        else:
                            logger.warning(f"Failed to download PDF for: {paper.title[:50]}...")
                
                queries_used += 1
                
                # Rate limiting
                await asyncio.sleep(self.sources['arxiv'].rate_limit)
                
            except Exception as e:
                logger.error(f"Error searching arXiv with query '{query}': {e}")
                continue
        
        logger.info(f"Collected {len(collected_papers)} papers from arXiv using {queries_used} queries")
        return collected_papers
    
    async def _collect_from_semantic_scholar(self, target_count: int) -> List[str]:
        """Collect papers from Semantic Scholar"""
        collected_papers = []
        queries_used = 0
        
        for query in self.ai_ml_queries:
            if len(collected_papers) >= target_count:
                break
                
            try:
                logger.info(f"Searching Semantic Scholar for: {query}")
                papers = await self._search_semantic_scholar(query, min(20, target_count - len(collected_papers)))
                
                for paper_data in papers:
                    if len(collected_papers) >= target_count:
                        break
                    
                    # Create Paper object
                    paper = Paper(
                        id="",  # Will be generated
                        title=paper_data['title'],
                        authors=paper_data['authors'],
                        abstract=paper_data['abstract'],
                        source='semantic_scholar',
                        url=paper_data['url'],
                        pdf_path="",  # Will be set when PDF is downloaded
                        folder_path="",  # Will be set by database
                        published_date=paper_data.get('published_date'),
                        semantic_scholar_id=paper_data.get('semantic_scholar_id'),
                        citation_count=paper_data.get('citation_count'),
                        venue=paper_data.get('venue'),
                        categories=paper_data.get('categories', [])
                    )
                    
                    # Add to database
                    if self.db.add_paper(paper):
                        # Try to download PDF (Semantic Scholar may not always have direct PDF links)
                        if await self._download_semantic_scholar_pdf(paper):
                            collected_papers.append(paper.id)
                            logger.info(f"Added Semantic Scholar paper: {paper.title[:50]}...")
                        else:
                            # Still count it as collected even without PDF
                            collected_papers.append(paper.id)
                            logger.info(f"Added Semantic Scholar paper (no PDF): {paper.title[:50]}...")
                
                queries_used += 1
                
                # Rate limiting
                await asyncio.sleep(self.sources['semantic_scholar'].rate_limit)
                
            except Exception as e:
                logger.error(f"Error searching Semantic Scholar with query '{query}': {e}")
                continue
        
        logger.info(f"Collected {len(collected_papers)} papers from Semantic Scholar using {queries_used} queries")
        return collected_papers
    
    async def _search_arxiv(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search arXiv for papers"""
        # Focus on AI/ML categories
        categories = "cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL OR cat:cs.NE OR cat:stat.ML"
        search_query = f"({query}) AND ({categories})"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        url = self.sources['arxiv'].base_url
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    xml_content = await response.text()
                    return self._parse_arxiv_response(xml_content)
                else:
                    logger.error(f"arXiv API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error querying arXiv: {e}")
            return []
    
    async def _search_semantic_scholar(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search Semantic Scholar for papers"""
        url = f"{self.sources['semantic_scholar'].base_url}/paper/search"
        
        params = {
            'query': query,
            'limit': max_results,
            'fields': 'paperId,title,abstract,authors,year,venue,url,citationCount,fieldsOfStudy,externalIds'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_semantic_scholar_response(data)
                else:
                    logger.error(f"Semantic Scholar API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error querying Semantic Scholar: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict]:
        """Parse arXiv XML response"""
        papers = []
        
        try:
            # Parse using feedparser for better handling
            feed = feedparser.parse(xml_content)
            
            for entry in feed.entries:
                # Extract arXiv ID
                arxiv_id = entry.id.split('/')[-1]
                
                # Clean title and abstract
                title = re.sub(r'\s+', ' ', entry.title).strip()
                abstract = re.sub(r'\s+', ' ', entry.summary).strip()
                
                # Extract authors
                authors = [author.name for author in entry.authors] if hasattr(entry, 'authors') else ['Unknown']
                
                # Extract categories
                categories = [tag.term for tag in entry.tags] if hasattr(entry, 'tags') else []
                
                # PDF URL
                pdf_url = None
                for link in entry.links:
                    if link.type == 'application/pdf':
                        pdf_url = link.href
                        break
                
                paper_data = {
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'url': entry.link,
                    'pdf_url': pdf_url,
                    'published_date': entry.published if hasattr(entry, 'published') else None,
                    'arxiv_id': arxiv_id,
                    'categories': categories
                }
                
                papers.append(paper_data)
                
        except Exception as e:
            logger.error(f"Error parsing arXiv response: {e}")
        
        return papers
    
    def _parse_semantic_scholar_response(self, data: Dict) -> List[Dict]:
        """Parse Semantic Scholar JSON response"""
        papers = []
        
        try:
            for paper_data in data.get('data', []):
                # Extract authors
                authors = [author.get('name', 'Unknown') for author in paper_data.get('authors', [])]
                
                # Extract fields of study as categories
                categories = paper_data.get('fieldsOfStudy', [])
                
                paper = {
                    'title': paper_data.get('title', ''),
                    'authors': authors,
                    'abstract': paper_data.get('abstract', ''),
                    'url': paper_data.get('url', ''),
                    'semantic_scholar_id': paper_data.get('paperId'),
                    'citation_count': paper_data.get('citationCount', 0),
                    'venue': paper_data.get('venue'),
                    'published_date': str(paper_data.get('year')) if paper_data.get('year') else None,
                    'categories': categories
                }
                
                papers.append(paper)
                
        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar response: {e}")
        
        return papers
    
    async def _download_arxiv_pdf(self, paper: Paper) -> bool:
        """Download PDF for arXiv paper"""
        if not paper.arxiv_id:
            return False
        
        # Construct PDF URL
        pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        
        # Set up file paths
        pdf_filename = f"{paper.arxiv_id}.pdf"
        pdf_path = os.path.join(paper.folder_path, pdf_filename)
        
        try:
            async with self.session.get(pdf_url) as response:
                if response.status == 200:
                    async with aiofiles.open(pdf_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    # Update paper with PDF path
                    paper.pdf_path = pdf_path
                    self.db.update_paper(paper)
                    
                    logger.info(f"Downloaded arXiv PDF: {pdf_filename}")
                    return True
                else:
                    logger.warning(f"Failed to download arXiv PDF: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error downloading arXiv PDF: {e}")
            return False
    
    async def _download_semantic_scholar_pdf(self, paper: Paper) -> bool:
        """Download PDF for Semantic Scholar paper (if available)"""
        # Semantic Scholar doesn't always provide direct PDF links
        # This is a placeholder - in practice, you might need to use other methods
        # to obtain PDFs for Semantic Scholar papers
        
        # For now, we'll just create a placeholder file
        placeholder_path = os.path.join(paper.folder_path, "paper_info.json")
        
        try:
            paper_info = {
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'url': paper.url,
                'semantic_scholar_id': paper.semantic_scholar_id,
                'source': 'semantic_scholar'
            }
            
            async with aiofiles.open(placeholder_path, 'w') as f:
                await f.write(json.dumps(paper_info, indent=2))
            
            # Update paper with info file path (not actual PDF)
            paper.pdf_path = placeholder_path
            self.db.update_paper(paper)
            
            return True
        except Exception as e:
            logger.error(f"Error creating Semantic Scholar paper info: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the paper collection"""
        stats = self.db.get_stats()
        
        # Add collection-specific stats
        papers = self.db.get_all_papers()
        
        pdf_count = 0
        source_files = {'arxiv': 0, 'semantic_scholar': 0}
        
        for paper in papers:
            if paper.pdf_path and os.path.exists(paper.pdf_path):
                if paper.source == 'arxiv' and paper.pdf_path.endswith('.pdf'):
                    pdf_count += 1
                source_files[paper.source] += 1
        
        stats.update({
            'actual_pdfs': pdf_count,
            'source_files': source_files,
            'collection_queries': len(self.ai_ml_queries)
        })
        
        return stats

# Convenience function for easy paper collection
async def collect_papers(target_count: int = 50) -> List[str]:
    """
    Convenience function to collect AI/ML papers
    Returns list of collected paper IDs
    """
    db = PaperDatabase()
    
    async with PaperCollector(db) as collector:
        return await collector.collect_ai_ml_papers(target_count) 