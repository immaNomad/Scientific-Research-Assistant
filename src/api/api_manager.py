"""
API Manager to coordinate different literature search APIs
"""

import asyncio
from typing import List, Dict, Optional
from loguru import logger

from .arxiv_client import ArxivClient
from .semantic_scholar_client import SemanticScholarClient
from .pubmed_client import PubMedClient

class APIManager:
    """Manages and coordinates different API clients"""
    
    def __init__(self):
        self.arxiv_client = ArxivClient()
        self.semantic_scholar_client = SemanticScholarClient()
        self.pubmed_client = PubMedClient()
        
        # Track API health and rate limits
        self.api_status = {
            'arxiv': {'available': True, 'last_error': None},
            'semantic_scholar': {'available': True, 'last_error': None},
            'pubmed': {'available': True, 'last_error': None}
        }
    
    async def search_all_sources(self, 
                                query: str, 
                                sources: List[str] = None,
                                max_results_per_source: int = 20) -> Dict:
        """
        Search all available sources in parallel
        
        Args:
            query: Search query
            sources: List of sources to search (default: all)
            max_results_per_source: Maximum results per source
            
        Returns:
            Dictionary with results from each source
        """
        if sources is None:
            sources = ['arxiv', 'semantic_scholar']  # Default sources
        
        tasks = []
        
        if 'arxiv' in sources and self.api_status['arxiv']['available']:
            tasks.append(self._search_arxiv_safe(query, max_results_per_source))
        
        if 'semantic_scholar' in sources and self.api_status['semantic_scholar']['available']:
            tasks.append(self._search_semantic_scholar_safe(query, max_results_per_source))
        
        if 'pubmed' in sources and self.api_status['pubmed']['available']:
            tasks.append(self._search_pubmed_safe(query, max_results_per_source))
        
        # Execute searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined_results = {
            'arxiv': [],
            'semantic_scholar': [],
            'pubmed': []
        }
        
        source_index = 0
        for source in sources:
            if self.api_status[source]['available'] and source_index < len(results):
                if isinstance(results[source_index], Exception):
                    logger.error(f"Error in {source}: {results[source_index]}")
                    self.api_status[source]['available'] = False
                    self.api_status[source]['last_error'] = str(results[source_index])
                else:
                    combined_results[source] = results[source_index]
                source_index += 1
        
        return combined_results
    
    async def _search_arxiv_safe(self, query: str, max_results: int) -> List:
        """Safe arXiv search with error handling"""
        try:
            return await self.arxiv_client.search(query, max_results)
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            self.api_status['arxiv']['available'] = False
            self.api_status['arxiv']['last_error'] = str(e)
            return []
    
    async def _search_semantic_scholar_safe(self, query: str, max_results: int) -> List:
        """Safe Semantic Scholar search with error handling"""
        try:
            return await self.semantic_scholar_client.search(query, max_results)
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            self.api_status['semantic_scholar']['available'] = False
            self.api_status['semantic_scholar']['last_error'] = str(e)
            return []
    
    async def _search_pubmed_safe(self, query: str, max_results: int) -> List:
        """Safe PubMed search with error handling"""
        try:
            return await self.pubmed_client.search(query, max_results)
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            self.api_status['pubmed']['available'] = False
            self.api_status['pubmed']['last_error'] = str(e)
            return []
    
    def get_api_status(self) -> Dict:
        """Get current status of all APIs"""
        return self.api_status.copy()
    
    def reset_api_status(self, source: str = None):
        """Reset API status (mark as available again)"""
        if source:
            if source in self.api_status:
                self.api_status[source]['available'] = True
                self.api_status[source]['last_error'] = None
        else:
            # Reset all
            for api in self.api_status:
                self.api_status[api]['available'] = True
                self.api_status[api]['last_error'] = None 