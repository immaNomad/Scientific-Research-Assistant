"""
Inclusive Search System
More flexible and inclusive search that finds papers that are slightly related
"""

import asyncio
import sqlite3
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import difflib
from pathlib import Path
import os
import sys

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from .local_search import LocalSearchEngine, SearchResult
from .paper_db import PaperDatabase, Paper

class InclusiveSearchEngine(LocalSearchEngine):
    """
    More inclusive search engine that finds papers that are slightly related
    Features:
    - Fuzzy matching
    - Partial word matching
    - Synonym matching
    - Lower relevance thresholds
    - Broader search strategies
    """
    
    def __init__(self, db_path: str = "data/papers/papers.db"):
        super().__init__(db_path)
        
        # Initialize keyword index if not already done by parent
        if not hasattr(self, 'keyword_index'):
            self.keyword_index = defaultdict(set)
        
        # Expanded domain keywords for better matching
        self.expanded_keywords = {
            'learning': ['learn', 'training', 'teach', 'education', 'knowledge', 'acquisition'],
            'network': ['net', 'connection', 'graph', 'topology', 'architecture'],
            'neural': ['neuron', 'brain', 'cognitive', 'artificial', 'synaptic'],
            'deep': ['depth', 'layer', 'hierarchical', 'multilayer', 'stacked'],
            'machine': ['automated', 'automatic', 'computational', 'algorithmic'],
            'computer': ['computing', 'computational', 'digital', 'electronic'],
            'vision': ['visual', 'image', 'picture', 'sight', 'perception'],
            'language': ['linguistic', 'text', 'natural', 'speech', 'communication'],
            'model': ['modeling', 'framework', 'approach', 'method', 'technique'],
            'algorithm': ['algorithmic', 'computation', 'procedure', 'method'],
            'optimization': ['optimize', 'optimizing', 'optimal', 'minimize', 'maximize'],
            'classification': ['classify', 'categorize', 'categorization', 'class'],
            'recognition': ['recognize', 'identification', 'identify', 'detect'],
            'detection': ['detect', 'identification', 'discovery', 'recognition'],
            'segmentation': ['segment', 'partition', 'divide', 'separation'],
            'processing': ['process', 'processing', 'computation', 'analysis'],
            'analysis': ['analyze', 'examination', 'study', 'investigation'],
            'system': ['systems', 'framework', 'platform', 'infrastructure'],
            'data': ['dataset', 'information', 'database', 'knowledge'],
            'medical': ['medicine', 'clinical', 'healthcare', 'diagnosis', 'treatment'],
            'image': ['images', 'picture', 'visual', 'photo', 'imaging'],
            'security': ['secure', 'safety', 'protection', 'privacy', 'cybersecurity'],
            'performance': ['efficiency', 'speed', 'accuracy', 'quality', 'improvement']
        }
        
        # Build expanded keyword index
        self._build_expanded_index()
    
    def _build_expanded_index(self):
        """Build expanded keyword index with synonyms and variants"""
        papers = self.db.get_all_papers()
        
        # Enhanced keyword index with fuzzy matching
        for paper in papers:
            # Index title and abstract with expanded keywords
            self._index_text_expanded(paper.title, paper.id)
            self._index_text_expanded(paper.abstract, paper.id)
            
            # Index partial words (substrings)
            self._index_partial_words(paper.title, paper.id)
            self._index_partial_words(paper.abstract, paper.id)
    
    def _index_text_expanded(self, text: str, paper_id: str):
        """Index text with expanded keywords and synonyms"""
        if not text:
            return
        
        text_lower = text.lower()
        words = text_lower.split()
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 2:
                # Add to keyword index
                self.keyword_index[clean_word].add(paper_id)
                
                # Add synonyms and variants
                for base_word, variants in self.expanded_keywords.items():
                    if base_word in clean_word or clean_word in base_word:
                        for variant in variants:
                            self.keyword_index[variant].add(paper_id)
                    
                    # Check if current word is a variant
                    if clean_word in variants:
                        self.keyword_index[base_word].add(paper_id)
                        for other_variant in variants:
                            self.keyword_index[other_variant].add(paper_id)
    
    def _index_partial_words(self, text: str, paper_id: str):
        """Index partial words for fuzzy matching"""
        if not text:
            return
        
        text_lower = text.lower()
        words = text_lower.split()
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 4:  # Only for longer words
                # Add substrings
                for i in range(len(clean_word) - 2):
                    for j in range(i + 3, len(clean_word) + 1):
                        substring = clean_word[i:j]
                        if len(substring) >= 3:
                            self.keyword_index[f"partial_{substring}"].add(paper_id)
    
    def _calculate_relevance_inclusive(self, paper: Paper, query: str) -> float:
        """More inclusive relevance calculation"""
        score = 0.0
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # 1. Exact phrase matching (highest weight)
        if query_lower in paper.title.lower():
            score += 10.0
        if query_lower in paper.abstract.lower():
            score += 8.0
        
        # 2. Title word matching
        title_lower = paper.title.lower()
        title_words = set(title_lower.split())
        
        for query_term in query_terms:
            # Exact word match
            if query_term in title_words:
                score += 5.0
            
            # Partial word match
            for title_word in title_words:
                if query_term in title_word or title_word in query_term:
                    score += 3.0
                
                # Fuzzy matching
                similarity = difflib.SequenceMatcher(None, query_term, title_word).ratio()
                if similarity > 0.7:  # 70% similarity
                    score += 2.0 * similarity
        
        # 3. Abstract word matching
        abstract_lower = paper.abstract.lower()
        abstract_words = set(abstract_lower.split())
        
        for query_term in query_terms:
            # Exact word match
            if query_term in abstract_words:
                score += 2.0
            
            # Partial word match
            for abstract_word in abstract_words:
                if query_term in abstract_word or abstract_word in query_term:
                    score += 1.0
                
                # Fuzzy matching
                similarity = difflib.SequenceMatcher(None, query_term, abstract_word).ratio()
                if similarity > 0.7:
                    score += 1.0 * similarity
        
        # 4. Synonym and variant matching
        for query_term in query_terms:
            if query_term in self.expanded_keywords:
                for variant in self.expanded_keywords[query_term]:
                    if variant in title_lower:
                        score += 3.0
                    if variant in abstract_lower:
                        score += 1.5
        
        # 5. Category matching
        if paper.categories:
            category_text = ' '.join(paper.categories).lower()
            for query_term in query_terms:
                if query_term in category_text:
                    score += 2.0
        
        # 6. Keyword matching
        if paper.keywords:
            keyword_text = ' '.join(paper.keywords).lower()
            for query_term in query_terms:
                if query_term in keyword_text:
                    score += 2.0
        
        # 7. Substring matching (for very inclusive search)
        for query_term in query_terms:
            if len(query_term) > 3:
                # Check if query term is a substring of title/abstract
                if query_term in title_lower:
                    score += 1.0
                if query_term in abstract_lower:
                    score += 0.5
        
        # 8. Length-based normalization (longer queries need higher scores)
        if len(query_terms) > 1:
            score = score / len(query_terms)
        
        # 9. Minimum score boost (ensure all papers get some score)
        if score > 0:
            score += 0.1  # Small boost for any match
        
        return score
    
    async def search_literature_inclusive(self, 
                                        query: str, 
                                        sources: List[str] = None,
                                        max_results_per_source: int = 20,
                                        domain: str = 'general',
                                        min_relevance: float = 0.1) -> List[SearchResult]:
        """
        More inclusive search that finds papers that are slightly related
        """
        # Get all papers
        papers = self.db.get_all_papers()
        
        # Filter by source if specified
        if sources:
            papers = [p for p in papers if p.source in sources]
        
        # Calculate relevance for all papers
        results = []
        for paper in papers:
            relevance = self._calculate_relevance_inclusive(paper, query)
            
            # Apply much lower minimum relevance threshold
            if relevance >= min_relevance:
                result = SearchResult(
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
                    relevance_score=relevance
                )
                results.append(result)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Return top results
        return results[:max_results_per_source]
    
    async def search_literature(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results_per_source: int = 20,
                               domain: str = 'general') -> List[SearchResult]:
        """
        Override the original search to use inclusive search
        """
        return await self.search_literature_inclusive(
            query=query,
            sources=sources,
            max_results_per_source=max_results_per_source,
            domain=domain,
            min_relevance=0.1  # Very low threshold
        )
    
    async def search_with_all_papers(self, 
                                   query: str, 
                                   sources: List[str] = None,
                                   max_results_per_source: int = 20) -> List[SearchResult]:
        """
        Search that guarantees to check all papers in the database
        """
        return await self.search_literature_inclusive(
            query=query,
            sources=sources,
            max_results_per_source=max_results_per_source,
            min_relevance=0.0  # No minimum threshold - all papers considered
        )
    
    def get_search_stats(self) -> Dict:
        """Get statistics about the search system"""
        papers = self.db.get_all_papers()
        
        return {
            'total_papers': len(papers),
            'sources': list(set(p.source for p in papers)),
            'expanded_keyword_count': len(self.expanded_keywords),
            'keyword_index_size': len(self.keyword_index),
            'search_type': 'inclusive',
            'min_relevance_threshold': 0.1,
            'fuzzy_matching': True,
            'partial_matching': True,
            'synonym_matching': True
        } 