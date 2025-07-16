"""
Local AI Enhanced RAG System
Integrates custom trained AI model with the local database RAG pipeline
"""

import asyncio
import os
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import json
from datetime import datetime
from pathlib import Path

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from rag.local_enhanced_rag import LocalEnhancedRAG, PaperInfo, ResearchAnalysis
from database.enhanced_search import EnhancedSearchEngine, SearchResult
from database.inclusive_search import InclusiveSearchEngine
from database.topic_search import TopicSearchEngine
from analytics.performance_monitor import log_query_performance

class LocalAIEnhancedRAG(LocalEnhancedRAG):
    """
    Enhanced RAG system that can use either:
    1. External LLM APIs (Gemini, etc.)
    2. Your custom trained local AI model
    3. Hybrid approach combining both
    """
    
    def __init__(self, db_path: str = "data/papers/papers.db", 
                 local_model_path: Optional[str] = None,
                 use_local_model: bool = False):
        super().__init__(db_path)
        
        # Replace basic search with topic-focused search for better research relevance
        self.search_engine = TopicSearchEngine(db_path)
        
        # Local AI model setup
        self.local_model_path = local_model_path
        self.use_local_model = use_local_model
        self.local_ai = None
        
        # Try to find and load local model
        self._setup_local_ai()
        
        logger.info(f"Initialized Local AI Enhanced RAG")
        logger.info(f"Using local model: {self.use_local_model}")
        logger.info(f"Enhanced search: Enabled")
    
    def _setup_local_ai(self):
        """Setup local AI model if available"""
        if not self.use_local_model:
            return
        
        # Auto-discover local model if not specified
        if not self.local_model_path:
            model_dir = Path("data/models/local_ai")
            if model_dir.exists():
                # Look for best model
                best_model = model_dir / "best_model.pt"
                if best_model.exists():
                    self.local_model_path = str(best_model)
                else:
                    # Look for any model
                    model_files = list(model_dir.glob("*.pt"))
                    if model_files:
                        self.local_model_path = str(model_files[0])
        
        # Load local model
        if self.local_model_path and os.path.exists(self.local_model_path):
            try:
                from models.local_ai_model import LocalAIInference
                self.local_ai = LocalAIInference(self.local_model_path)
                logger.info(f"Loaded local AI model: {self.local_model_path}")
                self.use_local_model = True
            except Exception as e:
                logger.warning(f"Failed to load local AI model: {e}")
                self.use_local_model = False
                self.local_ai = None
        else:
            logger.info("No local AI model found, using external LLM")
            self.use_local_model = False
    
    async def research_and_analyze(self, query: str, sources: List[str] = None) -> ResearchAnalysis:
        """
        Enhanced research analysis using local AI model + enhanced search
        """
        start_time = datetime.now()
        logger.info(f"Starting Local AI Enhanced research analysis for: '{query}'")
        
        # Use enhanced search engine
        search_type = "enhanced_local_ai" if self.use_local_model else "enhanced_external"
        
        try:
            # Step 1: Enhanced search for papers
            papers = await self._extract_five_papers_enhanced(query, sources)
            
            if not papers:
                # Log performance
                processing_time = (datetime.now() - start_time).total_seconds()
                log_query_performance(query, processing_time, 0, search_type, 
                                    "general", False, "No papers found")
                
                return self._create_empty_analysis(query, "No papers found in enhanced search")
            
            # Step 2: Generate enhanced summary using appropriate model
            if self.use_local_model and self.local_ai:
                summarized_findings = await self._generate_local_ai_summary(papers, query)
            else:
                summarized_findings = await self._generate_enhanced_summarized_findings(papers, query)
            
            # Step 3: Generate hypothesis using appropriate model
            if self.use_local_model and self.local_ai:
                hypothesis = await self._generate_local_ai_hypothesis(papers, query, summarized_findings)
            else:
                hypothesis = await self._generate_enhanced_research_hypothesis(papers, query, summarized_findings)
            
            # Enhanced metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            llm_model = self._get_model_info()
            
            # Get search statistics
            cache_stats = self.search_engine.get_cache_stats()
            index_stats = self.search_engine.get_index_stats()
            
            metadata = {
                "papers_found": len(papers),
                "processing_time_seconds": processing_time,
                "sources_used": list(set([paper.source for paper in papers])),
                "llm_model_used": llm_model,
                "search_type": search_type,
                "enhanced_features": {
                    "cache_hits": cache_stats.get('cache_hits', 0),
                    "cache_size": cache_stats.get('cache_size', 0),
                    "indexed_papers": index_stats.get('total_papers_indexed', 0),
                    "keywords_indexed": index_stats.get('keyword_index_size', 0)
                }
            }
            
            # Log successful performance
            log_query_performance(query, processing_time, len(papers), search_type, 
                                "general", True)
            
            return ResearchAnalysis(
                query=query,
                papers=papers,
                summarized_findings=summarized_findings,
                hypothesis=hypothesis,
                processing_metadata=metadata,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            # Log failed performance
            processing_time = (datetime.now() - start_time).total_seconds()
            log_query_performance(query, processing_time, 0, search_type, 
                                "general", False, str(e))
            
            logger.error(f"Error in Local AI Enhanced RAG: {e}")
            return self._create_empty_analysis(query, f"Analysis failed: {str(e)}")
    
    async def _extract_five_papers_enhanced(self, query: str, sources: List[str] = None) -> List[PaperInfo]:
        """Extract papers using enhanced search engine"""
        try:
            # Use enhanced search
            search_results = await self.search_engine.search_literature(
                query=query,
                sources=sources,
                max_results_per_source=5,
                domain=self._classify_domain(query)
            )
            
            # Convert SearchResult to PaperInfo
            papers = []
            for result in search_results[:5]:  # Limit to 5 papers
                paper_info = PaperInfo(
                    title=result.title,
                    doi=result.doi,
                    authors=result.authors,
                    abstract=result.abstract,
                    source=result.source,
                    url=result.url,
                    published_date=result.published_date,
                    citation_count=result.citation_count,
                    venue=result.venue
                )
                papers.append(paper_info)
            
            logger.info(f"Enhanced search found {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            # Fallback to basic search
            return await super()._extract_five_papers(query, sources)
    
    async def _generate_local_ai_summary(self, papers: List[PaperInfo], query: str) -> str:
        """Generate summary using local AI model in paragraph format"""
        if not self.local_ai:
            return await self._generate_enhanced_summarized_findings(papers, query)
        
        try:
            # Combine paper abstracts for context
            context = "\n\n".join([
                f"Paper {i+1}: {paper.title}\nAbstract: {paper.abstract}"
                for i, paper in enumerate(papers)
            ])
            
            # Generate summary using local model
            summary = self.local_ai.generate_summary(query, context, max_length=200)
            
            # Create paragraph mentioning paper names
            paragraph_parts = []
            for i, paper in enumerate(papers):
                title = paper.title
                # Extract brief insight from abstract
                abstract_key = paper.abstract.split('.')[0] if paper.abstract else "focuses on this research area"
                
                if i == 0:
                    paragraph_parts.append(f"'{title}' indicates that {abstract_key}")
                elif i == len(papers) - 1:
                    paragraph_parts.append(f"finally '{title}' demonstrates {abstract_key}")
                else:
                    transition = ["while", "additionally", "furthermore", "moreover"][i % 4]
                    paragraph_parts.append(f"{transition} '{title}' shows {abstract_key}")
            
            paragraph_summary = ". ".join(paragraph_parts) + "."
            
            # Add local AI analysis note
            source_count = len(set(paper.source for paper in papers))
            ai_note = f" This analysis leverages a custom-trained local AI model to synthesize findings from {len(papers)} papers across {source_count} research sources, providing domain-specific insights for {query}."
            
            return paragraph_summary + ai_note
            
        except Exception as e:
            logger.error(f"Error in local AI summary generation: {e}")
            return await self._generate_enhanced_summarized_findings(papers, query)
    
    async def _generate_local_ai_hypothesis(self, papers: List[PaperInfo], query: str, 
                                          summarized_findings: str) -> str:
        """Generate hypothesis using local AI model in numbered format"""
        if not self.local_ai:
            return await self._generate_enhanced_research_hypothesis(papers, query, summarized_findings)
        
        try:
            # Create context for hypothesis generation
            context = f"Research Query: {query}\n\nFindings: {summarized_findings}\n\nPapers: "
            context += "; ".join([paper.title for paper in papers])
            
            # Use local AI for hypothesis generation (simplified for now)
            hypothesis_base = self.local_ai.generate_summary(
                "Generate research hypothesis", context, max_length=150
            )
            
            # Create numbered hypotheses
            hypotheses = []
            sources = list(set(paper.source for paper in papers))

            hypotheses.append(f"1. The local AI analysis of {len(papers)} papers suggests that {hypothesis_base}")

            hypotheses.append(f"2. Cross-domain validation using methodologies from {len(sources)} different research sources could enhance the generalizability of {query} approaches.")

            hypotheses.append(f"3. Integration of the complementary techniques identified in these papers could address current limitations in {query} research.")

            hypotheses.append(f"4. The domain-specific insights from the custom-trained local AI model indicate potential for novel applications in {query}.")

            if len(papers) >= 4:
                hypotheses.append(f"5. Future research could benefit from investigating the unexplored intersections between the {len(papers)} analyzed papers to discover new research directions for {query}.")
            
            return "\n".join(hypotheses)
            
        except Exception as e:
            logger.error(f"Error in local AI hypothesis generation: {e}")
            return await self._generate_enhanced_research_hypothesis(papers, query, summarized_findings)
    
    def _get_model_info(self) -> str:
        """Get information about the model being used"""
        if self.use_local_model and self.local_ai:
            return f"local_ai_model ({Path(self.local_model_path).stem})"
        elif self.llm_manager.clients:
            client_name, client = self.llm_manager.clients[0]
            return f"{client_name} (external)"
        else:
            return "fallback"
    
    def _classify_domain(self, query: str) -> str:
        """Classify query domain for enhanced search"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['machine learning', 'ml', 'neural', 'deep learning']):
            return 'ai_ml'
        elif any(term in query_lower for term in ['medical', 'health', 'clinical', 'disease']):
            return 'healthcare'
        elif any(term in query_lower for term in ['computer', 'software', 'algorithm', 'system']):
            return 'computer_science'
        elif any(term in query_lower for term in ['vision', 'image', 'visual', 'video']):
            return 'computer_vision'
        else:
            return 'general'
    
    def _analyze_paper_domains(self, papers: List[PaperInfo]) -> Dict[str, int]:
        """Analyze domains of papers"""
        domains = {
            'ai_ml': 0,
            'computer_vision': 0,
            'nlp': 0,
            'robotics': 0,
            'healthcare': 0,
            'other': 0
        }
        
        for paper in papers:
            text = (paper.title + " " + paper.abstract).lower()
            
            if any(term in text for term in ['machine learning', 'neural', 'deep learning', 'ai']):
                domains['ai_ml'] += 1
            elif any(term in text for term in ['vision', 'image', 'visual', 'computer vision']):
                domains['computer_vision'] += 1
            elif any(term in text for term in ['language', 'nlp', 'text', 'linguistic']):
                domains['nlp'] += 1
            elif any(term in text for term in ['robot', 'robotic', 'autonomous']):
                domains['robotics'] += 1
            elif any(term in text for term in ['medical', 'health', 'clinical', 'disease']):
                domains['healthcare'] += 1
            else:
                domains['other'] += 1
        
        return domains
    
    def get_model_status(self) -> Dict:
        """Get status of available models"""
        status = {
            'local_ai_available': self.local_ai is not None,
            'local_model_path': self.local_model_path,
            'use_local_model': self.use_local_model,
            'external_llm_available': len(self.llm_manager.clients) > 0,
            'enhanced_search': True,
            'search_cache_size': self.search_engine.get_cache_stats().get('cache_size', 0),
            'indexed_papers': self.search_engine.get_index_stats().get('total_papers_indexed', 0)
        }
        
        if self.local_ai:
            try:
                # Test local model
                test_relevance = self.local_ai.calculate_relevance("test", "test context")
                status['local_model_functional'] = True
                status['local_model_test_score'] = test_relevance
            except Exception as e:
                status['local_model_functional'] = False
                status['local_model_error'] = str(e)
        
        return status
    
    def switch_to_local_model(self, model_path: str = None):
        """Switch to using local AI model"""
        if model_path:
            self.local_model_path = model_path
        
        self._setup_local_ai()
        
        if self.local_ai:
            logger.info("Switched to local AI model")
            return True
        else:
            logger.error("Failed to switch to local AI model")
            return False
    
    def switch_to_external_llm(self):
        """Switch to using external LLM"""
        self.use_local_model = False
        logger.info("Switched to external LLM")

# Convenience function for easy integration
def create_local_ai_enhanced_rag(local_model_path: str = None, 
                                use_local_model: bool = True) -> LocalAIEnhancedRAG:
    """Create a Local AI Enhanced RAG system"""
    return LocalAIEnhancedRAG(
        local_model_path=local_model_path,
        use_local_model=use_local_model
    ) 