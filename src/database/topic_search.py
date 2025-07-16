"""
Topic-Focused Search System
Finds research papers that are specifically helpful to research topics
"""

import asyncio
import sqlite3
import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import difflib
from pathlib import Path
import os
import sys
import math

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from .inclusive_search import InclusiveSearchEngine, SearchResult
from .paper_db import PaperDatabase, Paper

@dataclass
class TopicRelevance:
    """Relevance score breakdown for a paper to a topic"""
    paper_id: str
    total_score: float
    title_relevance: float
    abstract_relevance: float
    methodology_relevance: float
    application_relevance: float
    novelty_score: float
    citation_boost: float
    doi_available: bool

class TopicSearchEngine(InclusiveSearchEngine):
    """
    Topic-focused search engine that finds papers specifically helpful to research topics
    Features:
    - Topic-specific relevance scoring
    - Methodology and application matching
    - Research impact assessment
    - Comprehensive DOI handling
    """
    
    def __init__(self, db_path: str = "data/papers/papers.db"):
        super().__init__(db_path)
        
        # Topic-specific keywords and their importance weights
        self.topic_keywords = {
            # Core research areas
            'machine_learning': {
                'core': ['machine learning', 'ml', 'artificial intelligence', 'ai', 'learning algorithm'],
                'methods': ['supervised', 'unsupervised', 'reinforcement', 'deep learning', 'neural network'],
                'applications': ['classification', 'regression', 'clustering', 'prediction', 'recognition'],
                'weight': 1.0
            },
            'deep_learning': {
                'core': ['deep learning', 'neural network', 'neural networks', 'deep neural', 'artificial neural'],
                'methods': ['convolutional', 'recurrent', 'transformer', 'attention', 'backpropagation'],
                'applications': ['image recognition', 'natural language', 'speech recognition', 'computer vision'],
                'weight': 1.2
            },
            'computer_vision': {
                'core': ['computer vision', 'image processing', 'visual recognition', 'image analysis'],
                'methods': ['convolution', 'feature extraction', 'object detection', 'segmentation'],
                'applications': ['medical imaging', 'autonomous vehicles', 'surveillance', 'robotics'],
                'weight': 1.1
            },
            'natural_language': {
                'core': ['natural language processing', 'nlp', 'text processing', 'language model'],
                'methods': ['tokenization', 'parsing', 'sentiment analysis', 'named entity'],
                'applications': ['translation', 'summarization', 'question answering', 'chatbot'],
                'weight': 1.1
            },
            'medical_ai': {
                'core': ['medical', 'healthcare', 'clinical', 'biomedical', 'medical imaging'],
                'methods': ['diagnosis', 'prognosis', 'treatment', 'pathology', 'radiology'],
                'applications': ['patient care', 'drug discovery', 'clinical decision', 'telemedicine'],
                'weight': 1.3
            },
            'optimization': {
                'core': ['optimization', 'optimize', 'optimizing', 'optimal', 'minimize', 'maximize'],
                'methods': ['gradient descent', 'genetic algorithm', 'simulated annealing', 'particle swarm'],
                'applications': ['hyperparameter tuning', 'neural architecture search', 'model compression'],
                'weight': 1.0
            },
            'security': {
                'core': ['security', 'cybersecurity', 'privacy', 'encryption', 'vulnerability'],
                'methods': ['authentication', 'authorization', 'cryptography', 'blockchain'],
                'applications': ['network security', 'data protection', 'malware detection', 'fraud'],
                'weight': 1.2
            }
        }
        
        # Research quality indicators
        self.quality_indicators = {
            'methodology': ['method', 'approach', 'algorithm', 'technique', 'framework', 'model'],
            'evaluation': ['evaluation', 'experiment', 'benchmark', 'dataset', 'metric', 'performance'],
            'novelty': ['novel', 'new', 'innovative', 'original', 'first', 'introduce'],
            'comparison': ['compare', 'comparison', 'baseline', 'state-of-the-art', 'previous work'],
            'practical': ['practical', 'real-world', 'application', 'implementation', 'deployment']
        }
        
        # Build topic-specific indexes
        self._build_topic_indexes()
    
    def _build_topic_indexes(self):
        """Build topic-specific indexes for faster search"""
        papers = self.db.get_all_papers()
        
        # Topic relevance index
        self.topic_relevance_index = defaultdict(lambda: defaultdict(float))
        
        # DOI availability index
        self.doi_index = {}
        
        for paper in papers:
            # Index DOI availability
            self.doi_index[paper.id] = bool(paper.doi and paper.doi.strip())
            
            # Calculate topic relevance for each paper
            for topic, keywords in self.topic_keywords.items():
                relevance = self._calculate_topic_relevance(paper, topic, keywords)
                if relevance > 0:
                    self.topic_relevance_index[topic][paper.id] = relevance
    
    def _calculate_topic_relevance(self, paper: Paper, topic: str, keywords: Dict) -> float:
        """Calculate how relevant a paper is to a specific topic"""
        score = 0.0
        
        text = f"{paper.title} {paper.abstract}".lower()
        
        # Core topic keywords (highest weight)
        for keyword in keywords['core']:
            if keyword in text:
                score += 3.0 * keywords['weight']
        
        # Method keywords (medium weight)
        for keyword in keywords['methods']:
            if keyword in text:
                score += 2.0 * keywords['weight']
        
        # Application keywords (medium weight)
        for keyword in keywords['applications']:
            if keyword in text:
                score += 2.0 * keywords['weight']
        
        return score
    
    def _identify_paper_topic(self, paper: Paper) -> str:
        """Identify the primary topic of a paper"""
        best_topic = 'general'
        best_score = 0.0
        
        for topic, keywords in self.topic_keywords.items():
            score = self._calculate_topic_relevance(paper, topic, keywords)
            if score > best_score:
                best_score = score
                best_topic = topic
        
        return best_topic
    
    def _calculate_research_helpfulness(self, paper: Paper, query: str) -> TopicRelevance:
        """Calculate how helpful a paper is for research on a specific topic"""
        
        # Base relevance from inclusive search
        base_score = self._calculate_relevance_inclusive(paper, query)
        
        # Title relevance (papers with topic in title are more helpful)
        title_relevance = self._calculate_title_relevance(paper, query)
        
        # Abstract relevance (comprehensive coverage in abstract)
        abstract_relevance = self._calculate_abstract_relevance(paper, query)
        
        # Methodology relevance (papers that introduce/compare methods)
        methodology_relevance = self._calculate_methodology_relevance(paper, query)
        
        # Application relevance (papers that show practical applications)
        application_relevance = self._calculate_application_relevance(paper, query)
        
        # Novelty score (papers that introduce new concepts)
        novelty_score = self._calculate_novelty_score(paper)
        
        # Citation boost (more cited papers are generally more helpful)
        citation_boost = self._calculate_citation_boost(paper)
        
        # DOI availability (papers with DOI are more accessible)
        doi_available = self.doi_index.get(paper.id, False)
        doi_boost = 1.0 if doi_available else 0.8
        
        # Combined score
        total_score = (
            base_score * 0.3 +
            title_relevance * 0.25 +
            abstract_relevance * 0.15 +
            methodology_relevance * 0.1 +
            application_relevance * 0.1 +
            novelty_score * 0.05 +
            citation_boost * 0.05
        ) * doi_boost
        
        return TopicRelevance(
            paper_id=paper.id,
            total_score=total_score,
            title_relevance=title_relevance,
            abstract_relevance=abstract_relevance,
            methodology_relevance=methodology_relevance,
            application_relevance=application_relevance,
            novelty_score=novelty_score,
            citation_boost=citation_boost,
            doi_available=doi_available
        )
    
    def _calculate_title_relevance(self, paper: Paper, query: str) -> float:
        """Calculate relevance based on title match"""
        query_lower = query.lower()
        title_lower = paper.title.lower()
        
        score = 0.0
        
        # Exact phrase match in title
        if query_lower in title_lower:
            score += 15.0
        
        # Word matches in title
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        
        common_words = query_words.intersection(title_words)
        score += len(common_words) * 5.0
        
        # Partial word matches
        for q_word in query_words:
            for t_word in title_words:
                if q_word in t_word or t_word in q_word:
                    score += 3.0
        
        return score
    
    def _calculate_abstract_relevance(self, paper: Paper, query: str) -> float:
        """Calculate relevance based on abstract coverage"""
        query_lower = query.lower()
        abstract_lower = paper.abstract.lower()
        
        score = 0.0
        
        # Exact phrase match in abstract
        if query_lower in abstract_lower:
            score += 8.0
        
        # Word frequency in abstract
        query_words = query_lower.split()
        abstract_words = abstract_lower.split()
        
        for q_word in query_words:
            count = abstract_words.count(q_word)
            score += count * 2.0
        
        # Context richness (abstract length and query coverage)
        if len(abstract_words) > 100:  # Comprehensive abstract
            score += 2.0
        
        return score
    
    def _calculate_methodology_relevance(self, paper: Paper, query: str) -> float:
        """Calculate relevance based on methodology contribution"""
        text = f"{paper.title} {paper.abstract}".lower()
        score = 0.0
        
        # Check for methodology indicators
        for indicator in self.quality_indicators['methodology']:
            if indicator in text:
                score += 2.0
        
        # Check for evaluation indicators
        for indicator in self.quality_indicators['evaluation']:
            if indicator in text:
                score += 1.5
        
        # Check for comparison indicators
        for indicator in self.quality_indicators['comparison']:
            if indicator in text:
                score += 1.0
        
        return score
    
    def _calculate_application_relevance(self, paper: Paper, query: str) -> float:
        """Calculate relevance based on practical applications"""
        text = f"{paper.title} {paper.abstract}".lower()
        score = 0.0
        
        # Check for practical indicators
        for indicator in self.quality_indicators['practical']:
            if indicator in text:
                score += 2.0
        
        # Real-world application keywords
        practical_keywords = ['case study', 'implementation', 'deployment', 'real-world', 'practical']
        for keyword in practical_keywords:
            if keyword in text:
                score += 1.5
        
        return score
    
    def _calculate_novelty_score(self, paper: Paper) -> float:
        """Calculate novelty/innovation score"""
        text = f"{paper.title} {paper.abstract}".lower()
        score = 0.0
        
        # Check for novelty indicators
        for indicator in self.quality_indicators['novelty']:
            if indicator in text:
                score += 1.0
        
        # Innovation keywords
        innovation_keywords = ['breakthrough', 'pioneering', 'groundbreaking', 'cutting-edge']
        for keyword in innovation_keywords:
            if keyword in text:
                score += 2.0
        
        return score
    
    def _calculate_citation_boost(self, paper: Paper) -> float:
        """Calculate boost based on citation count"""
        if not paper.citation_count:
            return 0.0
        
        # Logarithmic scaling for citation count
        if paper.citation_count > 0:
            return math.log(paper.citation_count + 1) * 0.5
        
        return 0.0
    
    async def search_helpful_papers(self, 
                                  query: str, 
                                  sources: List[str] = None,
                                  max_results: int = 20,
                                  min_helpfulness: float = 0.5) -> List[Tuple[SearchResult, TopicRelevance]]:
        """
        Search for papers that are specifically helpful to a research topic
        """
        # Get all papers
        papers = self.db.get_all_papers()
        
        # Filter by source if specified
        if sources:
            papers = [p for p in papers if p.source in sources]
        
        # Calculate helpfulness for all papers
        helpful_papers = []
        for paper in papers:
            helpfulness = self._calculate_research_helpfulness(paper, query)
            
            if helpfulness.total_score >= min_helpfulness:
                # Convert to SearchResult
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
                    relevance_score=helpfulness.total_score
                )
                helpful_papers.append((result, helpfulness))
        
        # Sort by helpfulness score
        helpful_papers.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Return top results
        return helpful_papers[:max_results]
    
    async def search_literature(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results_per_source: int = 20,
                               domain: str = 'general') -> List[SearchResult]:
        """
        Override to use topic-focused search
        """
        helpful_papers = await self.search_helpful_papers(
            query=query,
            sources=sources,
            max_results=max_results_per_source,
            min_helpfulness=0.1  # Very low threshold for inclusivity
        )
        
        # Return just the SearchResult objects
        return [paper for paper, _ in helpful_papers]
    
    def get_topic_stats(self) -> Dict:
        """Get statistics about topic coverage"""
        stats = {}
        
        for topic in self.topic_keywords.keys():
            relevant_papers = len(self.topic_relevance_index[topic])
            stats[topic] = {
                'relevant_papers': relevant_papers,
                'keywords': len(self.topic_keywords[topic]['core'] + 
                             self.topic_keywords[topic]['methods'] + 
                             self.topic_keywords[topic]['applications'])
            }
        
        # DOI coverage
        doi_count = sum(1 for has_doi in self.doi_index.values() if has_doi)
        total_papers = len(self.doi_index)
        
        stats['doi_coverage'] = {
            'papers_with_doi': doi_count,
            'total_papers': total_papers,
            'coverage_percentage': (doi_count / total_papers * 100) if total_papers > 0 else 0
        }
        
        return stats
    
    def explain_relevance(self, paper_id: str, query: str) -> Dict:
        """Explain why a paper is relevant to a query"""
        paper = self.db.get_paper(paper_id)
        if not paper:
            return {}
        
        helpfulness = self._calculate_research_helpfulness(paper, query)
        
        return {
            'paper_title': paper.title,
            'query': query,
            'total_score': helpfulness.total_score,
            'breakdown': {
                'title_relevance': helpfulness.title_relevance,
                'abstract_relevance': helpfulness.abstract_relevance,
                'methodology_relevance': helpfulness.methodology_relevance,
                'application_relevance': helpfulness.application_relevance,
                'novelty_score': helpfulness.novelty_score,
                'citation_boost': helpfulness.citation_boost
            },
            'doi_available': helpfulness.doi_available,
            'doi': paper.doi if paper.doi else 'Not available',
            'primary_topic': self._identify_paper_topic(paper)
        } 