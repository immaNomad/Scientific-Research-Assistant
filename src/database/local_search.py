"""
Local Search System
Replaces external API calls with local database searches
Maintains the same interface as the original API system
"""

import asyncio
import os
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import re
from datetime import datetime

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from .paper_db import PaperDatabase, Paper

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Unified search result format - maintains compatibility with original system"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    source: str
    url: str
    published_date: Optional[str] = None
    categories: Optional[List[str]] = None
    doi: Optional[str] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    relevance_score: float = 0.0

class LocalSearchEngine:
    """
    Local search engine that replaces external API calls
    Provides the same interface as the original RAG pipeline
    """
    
    def __init__(self, db_path: str = "data/papers/papers.db"):
        self.db = PaperDatabase(db_path)
        
        # Domain-specific keywords for intelligent routing
        self.domain_keywords = {
            'healthcare': [
                'medical', 'healthcare', 'clinical', 'diagnosis', 'treatment', 
                'patient', 'disease', 'therapy', 'drug', 'medicine', 'health',
                'cancer', 'diabetes', 'cardiovascular', 'neurological', 'radiology',
                'epidemiology', 'genomics', 'biomarker', 'pharmaceutical'
            ],
            'computer_science': [
                'algorithm', 'software', 'programming', 'database', 'network',
                'security', 'cybersecurity', 'encryption', 'protocol', 'system',
                'computing', 'compiler', 'architecture', 'distributed', 'parallel'
            ],
            'ai_ml': [
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'AI', 'ML', 'model', 'training', 'prediction',
                'classification', 'regression', 'clustering', 'optimization',
                'transformer', 'attention', 'generative', 'supervised', 'unsupervised'
            ],
            'physics': [
                'quantum', 'particle', 'energy', 'momentum', 'wave', 'field',
                'electromagnetic', 'thermodynamics', 'mechanics', 'relativity',
                'cosmology', 'astrophysics', 'condensed matter', 'optics'
            ],
            'mathematics': [
                'theorem', 'proof', 'equation', 'function', 'matrix', 'vector',
                'calculus', 'algebra', 'geometry', 'topology', 'statistics',
                'probability', 'optimization', 'numerical', 'analysis'
            ]
        }
    
    async def search_literature(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results_per_source: int = 20,
                               domain: str = 'general') -> List[SearchResult]:
        """
        Search local database - maintains same interface as original search_literature
        
        Args:
            query: Search query
            sources: List of sources (ignored - we use local database)
            max_results_per_source: Maximum results to return
            domain: Domain for relevance filtering
            
        Returns:
            List of unified search results
        """
        logger.info(f"Searching local database for: '{query}'")
        
        try:
            # Determine preferred source based on domain
            preferred_source = self._get_preferred_source(query, domain)
            
            # Search in database
            papers = self.db.search_papers(
                query=query, 
                source=preferred_source,
                limit=max_results_per_source
            )
            
            # Convert to SearchResult format
            search_results = []
            for paper in papers:
                result = self._paper_to_search_result(paper, query)
                search_results.append(result)
            
            # Sort by relevance score
            search_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            logger.info(f"Found {len(search_results)} papers in local database")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching local database: {e}")
            return []
    
    def _get_preferred_source(self, query: str, domain: str) -> Optional[str]:
        """
        Determine preferred source based on query and domain
        Mimics the domain-aware routing from the original system
        """
        query_lower = query.lower()
        
        # Check for healthcare-related queries
        if domain == 'healthcare' or any(keyword in query_lower for keyword in self.domain_keywords['healthcare']):
            # In the original system, healthcare queries would go to PubMed
            # Since we don't have PubMed papers, we'll prefer Semantic Scholar for academic papers
            return 'semantic_scholar'
        
        # Check for computer science/AI queries  
        if (domain in ['computer_science', 'ai_ml'] or 
            any(keyword in query_lower for keyword in self.domain_keywords['ai_ml']) or
            any(keyword in query_lower for keyword in self.domain_keywords['computer_science'])):
            # Computer Science queries can use either source, prefer arXiv for recent research
            return 'arxiv'
        
        # Check for physics/math queries
        if (domain in ['physics', 'mathematics'] or
            any(keyword in query_lower for keyword in self.domain_keywords['physics']) or
            any(keyword in query_lower for keyword in self.domain_keywords['mathematics'])):
            return 'arxiv'  # arXiv is strong in physics/math
        
        # Default: no preference, search all sources
        return None
    
    def _paper_to_search_result(self, paper: Paper, query: str) -> SearchResult:
        """Convert Paper object to SearchResult with relevance scoring"""
        
        # Calculate relevance score based on query matching
        relevance_score = self._calculate_relevance(paper, query)
        
        return SearchResult(
            id=paper.id,
            title=paper.title,
            authors=paper.authors,
            abstract=paper.abstract,
            source=paper.source,
            url=paper.url,
            published_date=paper.published_date,
            categories=paper.categories,
            doi=paper.doi,
            citation_count=paper.citation_count,
            venue=paper.venue,
            relevance_score=relevance_score
        )
    
    def _calculate_relevance(self, paper: Paper, query: str) -> float:
        """Calculate relevance score between paper and query"""
        score = 0.0
        query_terms = set(query.lower().split())
        
        # Title matching (highest weight)
        title_terms = set(paper.title.lower().split())
        title_matches = len(query_terms.intersection(title_terms))
        score += title_matches * 3.0
        
        # Abstract matching (medium weight)
        abstract_terms = set(paper.abstract.lower().split())
        abstract_matches = len(query_terms.intersection(abstract_terms))
        score += abstract_matches * 1.5
        
        # Category matching (medium weight)
        if paper.categories:
            category_text = ' '.join(paper.categories).lower()
            category_terms = set(category_text.split())
            category_matches = len(query_terms.intersection(category_terms))
            score += category_matches * 2.0
        
        # Keywords matching (medium weight)
        if paper.keywords:
            keyword_text = ' '.join(paper.keywords).lower()
            keyword_terms = set(keyword_text.split())
            keyword_matches = len(query_terms.intersection(keyword_terms))
            score += keyword_matches * 2.0
        
        # Boost score for exact phrase matches in title
        if query.lower() in paper.title.lower():
            score += 5.0
        
        # Boost score for exact phrase matches in abstract
        if query.lower() in paper.abstract.lower():
            score += 3.0
        
        # Normalize by query length
        if len(query_terms) > 0:
            score = score / len(query_terms)
        
        return score
    
    async def search_by_category(self, category: str, limit: int = 20) -> List[SearchResult]:
        """Search papers by category"""
        papers = self.db.get_papers_by_category(category, limit)
        return [self._paper_to_search_result(paper, category) for paper in papers]
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[SearchResult]:
        """Get a specific paper by ID"""
        paper = self.db.get_paper(paper_id)
        if paper:
            return self._paper_to_search_result(paper, "")
        return None
    
    async def get_recent_papers(self, source: str = None, limit: int = 20) -> List[SearchResult]:
        """Get recent papers from database"""
        papers = self.db.get_all_papers(limit)
        
        if source:
            papers = [p for p in papers if p.source == source]
        
        return [self._paper_to_search_result(paper, "") for paper in papers]
    
    def get_available_sources(self) -> List[str]:
        """Get list of available sources in the database"""
        stats = self.db.get_stats()
        return list(stats['source_distribution'].keys())
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        stats = self.db.get_stats()
        
        # Add search-specific statistics
        papers = self.db.get_all_papers()
        
        # Count papers by domain (based on categories)
        domain_counts = {}
        for paper in papers:
            if paper.categories:
                for category in paper.categories:
                    domain = self._categorize_domain(category)
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        stats['domain_distribution'] = domain_counts
        stats['search_engine'] = 'LocalSearchEngine'
        
        return stats
    
    def _categorize_domain(self, category: str) -> str:
        """Categorize a paper category into broader domains"""
        category_lower = category.lower()
        
        # AI/ML categories
        if any(term in category_lower for term in ['ai', 'ml', 'learning', 'neural', 'intelligence']):
            return 'AI/ML'
        
        # Computer Science categories
        if any(term in category_lower for term in ['cs', 'computer', 'software', 'algorithm']):
            return 'Computer Science'
        
        # Physics categories  
        if any(term in category_lower for term in ['physics', 'quantum', 'particle', 'astro']):
            return 'Physics'
        
        # Math categories
        if any(term in category_lower for term in ['math', 'stat', 'probability']):
            return 'Mathematics'
        
        return 'Other'

class LocalRAGPipeline:
    """
    Local RAG Pipeline that uses the local search engine
    Maintains compatibility with the original RAGPipeline interface
    """
    
    def __init__(self, db_path: str = "data/papers/papers.db"):
        self.search_engine = LocalSearchEngine(db_path)
        self.max_retrieved_docs = 10  # Configurable
    
    async def search_literature(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results_per_source: int = 20,
                               domain: str = 'general') -> List[SearchResult]:
        """Main search interface - delegates to local search engine"""
        return await self.search_engine.search_literature(
            query=query,
            sources=sources,
            max_results_per_source=max_results_per_source,
            domain=domain
        )
    
    def add_papers_to_vector_store(self, papers: List[SearchResult], domain: str = 'general'):
        """
        Placeholder for vector store functionality
        In a full implementation, this would add papers to a vector database
        """
        logger.info(f"Would add {len(papers)} papers to vector store for domain: {domain}")
        pass
    
    async def process_query(self, 
                           query: str, 
                           sources: List[str] = None,
                           domain: str = 'general',
                           generate_hypotheses: bool = True) -> Dict:
        """
        Process a complete research query through the local RAG pipeline
        Returns a simplified response structure
        """
        logger.info(f"Processing query through local RAG: '{query}'")
        start_time = datetime.now()
        
        try:
            # Search literature using local database
            papers = await self.search_literature(
                query=query,
                sources=sources,
                domain=domain,
                max_results_per_source=self.max_retrieved_docs
            )
            
            # Create response metadata
            metadata = {
                'num_papers_retrieved': len(papers),
                'sources_used': sources or self.search_engine.get_available_sources(),
                'domain': domain,
                'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                'average_relevance_score': sum(p.relevance_score for p in papers) / len(papers) if papers else 0,
                'search_engine': 'local_database'
            }
            
            # Simple response structure
            response = {
                'query': query,
                'retrieved_papers': papers,
                'summary': f"Found {len(papers)} relevant papers in local database",
                'hypotheses': [] if not generate_hypotheses else [f"Research direction based on {len(papers)} papers"],
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Local query processed successfully in {metadata['processing_time_seconds']:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error processing local query: {e}")
            return {
                'query': query,
                'retrieved_papers': [],
                'summary': f"Error processing query: {str(e)}",
                'hypotheses': [],
                'metadata': {'error': str(e)},
                'timestamp': datetime.now().isoformat()
            } 