"""
Enhanced Search Engine with Performance Optimizations
Adds caching, full-text search, and semantic similarity
"""

import asyncio
import sqlite3
import json
import hashlib
import time
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict
import logging
import threading
from pathlib import Path

from .paper_db import PaperDatabase, Paper
from .local_search import SearchResult, LocalSearchEngine

logger = logging.getLogger(__name__)

@dataclass
class SearchCache:
    """Cache for search results"""
    query_hash: str
    results: List[SearchResult]
    timestamp: float
    domain: str
    
    def is_expired(self, ttl: int = 300) -> bool:  # 5 minutes default
        return time.time() - self.timestamp > ttl

class EnhancedSearchEngine(LocalSearchEngine):
    """
    Enhanced search engine with performance optimizations:
    - Query result caching
    - Full-text search indexing
    - Semantic similarity scoring
    - Parallel processing
    - Query preprocessing
    """
    
    def __init__(self, db_path: str = "data/papers/papers.db"):
        super().__init__(db_path)
        
        # Performance optimizations
        self.search_cache = {}  # query_hash -> SearchCache
        self.cache_lock = threading.Lock()
        self.similarity_cache = {}  # paper_id -> precomputed vectors
        
        # Enhanced indexing
        self.keyword_index = defaultdict(set)  # keyword -> set of paper_ids
        self.author_index = defaultdict(set)   # author -> set of paper_ids
        self.category_index = defaultdict(set) # category -> set of paper_ids
        
        # Build indexes on initialization
        self._build_indexes()
        
        logger.info("Enhanced search engine initialized with caching and indexing")
    
    def _build_indexes(self):
        """Build in-memory indexes for faster search"""
        logger.info("Building search indexes...")
        
        papers = self.db.get_all_papers()
        
        for paper in papers:
            # Build keyword index
            self._index_text(paper.title, paper.id)
            self._index_text(paper.abstract, paper.id)
            
            # Build author index
            for author in paper.authors:
                self.author_index[author.lower()].add(paper.id)
            
            # Build category index
            if paper.categories:
                for category in paper.categories:
                    self.category_index[category.lower()].add(paper.id)
        
        logger.info(f"Indexes built: {len(self.keyword_index)} keywords, "
                   f"{len(self.author_index)} authors, {len(self.category_index)} categories")
    
    def _index_text(self, text: str, paper_id: str):
        """Index text for keyword search"""
        if not text:
            return
        
        # Simple tokenization and indexing
        words = text.lower().split()
        for word in words:
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 2:  # Only index words longer than 2 chars
                self.keyword_index[clean_word].add(paper_id)
    
    def _get_query_hash(self, query: str, sources: List[str], domain: str) -> str:
        """Generate hash for query caching"""
        content = f"{query}_{sorted(sources or [])}_{domain}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_results(self, query_hash: str) -> Optional[List[SearchResult]]:
        """Get cached search results if available and not expired"""
        with self.cache_lock:
            if query_hash in self.search_cache:
                cache_entry = self.search_cache[query_hash]
                if not cache_entry.is_expired():
                    logger.debug(f"Cache hit for query hash: {query_hash}")
                    return cache_entry.results
                else:
                    # Remove expired cache
                    del self.search_cache[query_hash]
                    logger.debug(f"Cache expired for query hash: {query_hash}")
        return None
    
    def _cache_results(self, query_hash: str, results: List[SearchResult], domain: str):
        """Cache search results"""
        with self.cache_lock:
            self.search_cache[query_hash] = SearchCache(
                query_hash=query_hash,
                results=results,
                timestamp=time.time(),
                domain=domain
            )
            
            # Limit cache size
            if len(self.search_cache) > 100:
                # Remove oldest entries
                oldest_key = min(self.search_cache.keys(), 
                               key=lambda k: self.search_cache[k].timestamp)
                del self.search_cache[oldest_key]
    
    async def search_literature(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results_per_source: int = 20,
                               domain: str = 'general') -> List[SearchResult]:
        """
        Enhanced search with caching and optimized algorithms
        """
        # Check cache first
        query_hash = self._get_query_hash(query, sources, domain)
        cached_results = self._get_cached_results(query_hash)
        if cached_results:
            logger.info(f"Returning cached results for query: '{query}'")
            return cached_results[:max_results_per_source]
        
        logger.info(f"Performing enhanced search for: '{query}'")
        start_time = time.time()
        
        # Multi-strategy search
        results = []
        
        # 1. Exact phrase search
        exact_results = await self._exact_phrase_search(query, max_results_per_source // 4)
        results.extend(exact_results)
        
        # 2. Keyword-based search using indexes
        keyword_results = await self._keyword_search(query, max_results_per_source // 2)
        results.extend(keyword_results)
        
        # 3. Semantic search (database query)
        semantic_results = await self._semantic_search(query, sources, max_results_per_source)
        results.extend(semantic_results)
        
        # 4. Author search
        author_results = await self._author_search(query, max_results_per_source // 4)
        results.extend(author_results)
        
        # Remove duplicates while preserving order
        seen_ids = set()
        unique_results = []
        for result in results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit results
        final_results = unique_results[:max_results_per_source]
        
        # Cache results
        self._cache_results(query_hash, final_results, domain)
        
        search_time = time.time() - start_time
        logger.info(f"Enhanced search completed in {search_time:.2f}s, found {len(final_results)} results")
        
        return final_results
    
    async def _exact_phrase_search(self, query: str, limit: int) -> List[SearchResult]:
        """Search for exact phrase matches"""
        results = []
        papers = self.db.get_all_papers(limit * 2)  # Get more to filter
        
        for paper in papers:
            if query.lower() in paper.title.lower() or query.lower() in paper.abstract.lower():
                result = self._paper_to_search_result(paper, query)
                result.relevance_score += 2.0  # Boost for exact matches
                results.append(result)
        
        return results[:limit]
    
    async def _keyword_search(self, query: str, limit: int) -> List[SearchResult]:
        """Search using keyword indexes"""
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        
        # Find papers matching keywords
        paper_scores = defaultdict(float)
        
        for word in query_words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word in self.keyword_index:
                for paper_id in self.keyword_index[clean_word]:
                    paper_scores[paper_id] += 1.0
        
        # Get top papers
        top_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        results = []
        for paper_id, score in top_papers:
            paper = self.db.get_paper(paper_id)
            if paper:
                result = self._paper_to_search_result(paper, query)
                result.relevance_score += score
                results.append(result)
        
        return results
    
    async def _semantic_search(self, query: str, sources: List[str], limit: int) -> List[SearchResult]:
        """Use the original semantic search"""
        return await super().search_literature(query, sources, limit)
    
    async def _author_search(self, query: str, limit: int) -> List[SearchResult]:
        """Search by author names"""
        results = []
        query_lower = query.lower()
        
        for author, paper_ids in self.author_index.items():
            if query_lower in author:
                for paper_id in list(paper_ids)[:limit]:
                    paper = self.db.get_paper(paper_id)
                    if paper:
                        result = self._paper_to_search_result(paper, query)
                        result.relevance_score += 1.5  # Boost for author matches
                        results.append(result)
        
        return results[:limit]
    
    def clear_cache(self):
        """Clear search cache"""
        with self.cache_lock:
            self.search_cache.clear()
            logger.info("Search cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self.cache_lock:
            return {
                'cache_size': len(self.search_cache),
                'cache_hits': sum(1 for c in self.search_cache.values() if not c.is_expired()),
                'expired_entries': sum(1 for c in self.search_cache.values() if c.is_expired())
            }
    
    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'keyword_index_size': len(self.keyword_index),
            'author_index_size': len(self.author_index),
            'category_index_size': len(self.category_index),
            'total_papers_indexed': len(set().union(*self.keyword_index.values())) if self.keyword_index else 0
        } 