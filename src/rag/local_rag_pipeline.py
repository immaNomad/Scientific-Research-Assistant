"""
Local RAG Pipeline
Uses local database instead of external APIs for paper retrieval
Maintains same interface as original RAGPipeline for compatibility
"""

import asyncio
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from loguru import logger
import json
from datetime import datetime
import os
import sys

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from database.local_search import LocalSearchEngine, SearchResult
from models.embeddings import EmbeddingManager, VectorStore, DocumentChunker
from models.llm_client import llm_manager, LLMResponse
from config.config import config

@dataclass
class RAGResponse:
    """Response from RAG pipeline"""
    query: str
    retrieved_papers: List[SearchResult]
    summary: str
    hypotheses: List[str]
    metadata: Dict
    timestamp: str

class LocalRAGPipeline:
    """
    Local RAG pipeline using database instead of external APIs
    Provides enhanced content generation from local paper dataset
    """
    
    def __init__(self, db_path: str = "data/papers/papers.db"):
        # Initialize local search engine
        self.search_engine = LocalSearchEngine(db_path)
        
        # Initialize models for enhanced content generation
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.document_chunker = DocumentChunker()
        
        # Configuration
        self.max_retrieved_docs = getattr(config.rag, 'MAX_RETRIEVED_DOCS', 10)
        
        logger.info("Initialized Local RAG Pipeline with database")
    
    async def search_literature(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results_per_source: int = 20,
                               domain: str = 'general') -> List[SearchResult]:
        """
        Search local database - maintains same interface as original
        
        Args:
            query: Search query
            sources: List of sources (optional filtering)
            max_results_per_source: Maximum results to return
            domain: Domain for relevance filtering
            
        Returns:
            List of search results from local database
        """
        logger.info(f"Searching local database for: '{query}' in domain: {domain}")
        
        # Use the local search engine
        results = await self.search_engine.search_literature(
            query=query,
            sources=sources,
            max_results_per_source=max_results_per_source,
            domain=domain
        )
        
        # Enhanced ranking for local results
        ranked_results = await self._rank_results(results, query, domain)
        
        return ranked_results[:self.max_retrieved_docs]
    
    async def _rank_results(self, 
                           results: List[SearchResult], 
                           query: str, 
                           domain: str) -> List[SearchResult]:
        """Enhanced ranking for local search results"""
        # Already have relevance scores from local search
        # Sort by relevance score (descending)
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    async def generate_summary(self, 
                              papers: List[SearchResult], 
                              query: str,
                              max_length: int = None) -> str:
        """
        Generate enhanced summary using LLM with local paper content
        """
        if not papers:
            return "No papers found in local database for this query."
        
        logger.info(f"Generating summary for {len(papers)} papers")
        
        try:
            # Prepare context from papers
            papers_context = self._prepare_papers_context(papers)
            
            # Create enhanced prompt for local content
            prompt = f"""Based on the following research papers from our local database, provide a comprehensive summary addressing the query: "{query}"

Papers Context:
{papers_context}

Please provide:
1. Key findings and insights
2. Common themes and patterns
3. Research methodologies used
4. Potential implications and applications
5. Areas where the research converges or diverges

Generate a well-structured summary that synthesizes the information from these {len(papers)} papers:"""
            
            # Try to use LLM for enhanced summary
            response = await llm_manager.generate(
                prompt=prompt,
                max_tokens=max_length or 1000,
                temperature=0.7
            )
            
            if response and response.content:
                return response.content
            else:
                # Fallback to template-based summary
                return self._generate_enhanced_template_summary(papers, query)
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._generate_enhanced_template_summary(papers, query)
    
    def _prepare_papers_context(self, papers: List[SearchResult]) -> str:
        """Prepare context from papers for LLM processing"""
        context_parts = []
        
        for i, paper in enumerate(papers, 1):
            paper_info = f"""
Paper {i}: {paper.title}
Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}
Source: {paper.source}
Abstract: {paper.abstract[:500]}{'...' if len(paper.abstract) > 500 else ''}
{"Categories: " + ', '.join(paper.categories[:3]) if paper.categories else ""}
"""
            context_parts.append(paper_info.strip())
        
        return "\n\n".join(context_parts)
    
    def _generate_enhanced_template_summary(self, papers: List[SearchResult], query: str) -> str:
        """Enhanced template-based summary for local papers"""
        
        # Analyze sources
        source_counts = {}
        total_citations = 0
        categories_set = set()
        
        for paper in papers:
            source_counts[paper.source] = source_counts.get(paper.source, 0) + 1
            if paper.citation_count:
                total_citations += paper.citation_count
            if paper.categories:
                categories_set.update(paper.categories)
        
        # Generate enhanced summary
        summary_parts = [
            f"## Research Summary for: '{query}'",
            f"**Papers analyzed:** {len(papers)} papers from local database",
            f"**Sources:** {', '.join(f'{source}: {count}' for source, count in source_counts.items())}",
        ]
        
        if total_citations > 0:
            summary_parts.append(f"**Total citations:** {total_citations}")
        
        if categories_set:
            top_categories = list(categories_set)[:5]
            summary_parts.append(f"**Key areas:** {', '.join(top_categories)}")
        
        # Key findings
        summary_parts.extend([
            "\n### Key Findings:",
            "• " + "\n• ".join([
                f"Research in {paper.source}: {paper.title[:100]}..." 
                for paper in papers[:3]
            ])
        ])
        
        # Research themes
        if len(papers) >= 3:
            summary_parts.extend([
                "\n### Research Themes:",
                f"• The {len(papers)} papers span multiple approaches to {query}",
                f"• Common methodologies include experimental validation and theoretical analysis",
                f"• Research shows both foundational work and applied implementations"
            ])
        
        # Implications
        summary_parts.extend([
            "\n### Implications:",
            f"• This local collection provides {len(papers)} perspectives on {query}",
            f"• Research spans from {min(source_counts.values())} to {max(source_counts.values())} papers per source",
            f"• Results suggest active research interest in this area"
        ])
        
        return "\n".join(summary_parts)
    
    async def generate_hypotheses(self, 
                                 papers: List[SearchResult], 
                                 query: str,
                                 num_hypotheses: int = 3) -> List[str]:
        """
        Generate research hypotheses based on local paper content
        """
        if not papers:
            return []
        
        logger.info(f"Generating {num_hypotheses} hypotheses from {len(papers)} papers")
        
        try:
            # Prepare context for hypothesis generation
            papers_context = self._prepare_papers_context(papers)
            
            prompt = f"""Based on the following research papers about "{query}", generate {num_hypotheses} novel research hypotheses or directions that could advance this field:

Papers Context:
{papers_context}

Generate {num_hypotheses} distinct research hypotheses that:
1. Build upon the existing work shown in these papers
2. Identify gaps or opportunities for future research
3. Suggest novel approaches or methodologies
4. Could lead to practical applications or theoretical advances

Format each hypothesis as a clear, testable statement:"""

            response = await llm_manager.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.8
            )
            
            if response and response.content:
                hypotheses = self._parse_hypotheses(response.content)
                return hypotheses[:num_hypotheses]
            else:
                return self._generate_enhanced_template_hypotheses(papers, query, num_hypotheses)
                
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            return self._generate_enhanced_template_hypotheses(papers, query, num_hypotheses)
    
    def _parse_hypotheses(self, content: str) -> List[str]:
        """Parse hypotheses from LLM response"""
        hypotheses = []
        
        # Split by numbered lists or bullet points
        lines = content.split('\n')
        current_hypothesis = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_hypothesis:
                    hypotheses.append(current_hypothesis.strip())
                    current_hypothesis = ""
                continue
            
            # Check if it's a new hypothesis (numbered or bulleted)
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '•', '-', '*']):
                if current_hypothesis:
                    hypotheses.append(current_hypothesis.strip())
                current_hypothesis = line
            else:
                current_hypothesis += " " + line
        
        # Add the last hypothesis
        if current_hypothesis:
            hypotheses.append(current_hypothesis.strip())
        
        # Clean up hypotheses
        cleaned_hypotheses = []
        for hyp in hypotheses:
            # Remove numbering and bullet points
            cleaned = hyp.lstrip('123456789.•-* ').strip()
            if len(cleaned) > 20:  # Filter out very short entries
                cleaned_hypotheses.append(cleaned)
        
        return cleaned_hypotheses
    
    def _generate_enhanced_template_hypotheses(self, papers: List[SearchResult], query: str, num_hypotheses: int) -> List[str]:
        """Generate template-based hypotheses enhanced for local dataset"""
        
        sources = list(set([paper.source for paper in papers]))
        categories = []
        for paper in papers:
            if paper.categories:
                categories.extend(paper.categories)
        
        unique_categories = list(set(categories))[:3]
        
        hypotheses = [
            f"Future research could integrate findings from {len(papers)} papers in our local database to develop more comprehensive approaches to {query}",
            f"Cross-source analysis between {' and '.join(sources)} papers suggests opportunities for methodological improvements in {query}",
            f"The {len(papers)} papers in our collection indicate potential for novel applications of {query} in domains like {', '.join(unique_categories[:2]) if unique_categories else 'emerging fields'}"
        ]
        
        # Add domain-specific hypotheses
        if 'arxiv' in sources and 'semantic_scholar' in sources:
            hypotheses.append(f"Combining theoretical frameworks from arXiv with empirical studies from Semantic Scholar could advance {query} research")
        
        if len(papers) >= 5:
            hypotheses.append(f"Meta-analysis of the {len(papers)} papers could reveal systematic patterns in {query} methodologies")
        
        return hypotheses[:num_hypotheses]
    
    async def process_query(self, 
                           query: str, 
                           sources: List[str] = None,
                           domain: str = 'general',
                           generate_hypotheses: bool = True) -> RAGResponse:
        """
        Process a complete research query through the local RAG pipeline
        Enhanced to work with local paper dataset
        """
        logger.info(f"Processing query through Local RAG: '{query}'")
        start_time = datetime.now()
        
        try:
            # Step 1: Search local literature
            papers = await self.search_literature(
                query=query,
                sources=sources,
                domain=domain
            )
            
            # Step 2: Generate enhanced summary
            summary = await self.generate_summary(papers, query)
            
            # Step 3: Generate hypotheses (if requested)
            hypotheses = []
            if generate_hypotheses and papers:
                hypotheses = await self.generate_hypotheses(papers, query)
            
            # Step 4: Prepare metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metadata = {
                'num_papers_retrieved': len(papers),
                'sources_used': sources or self.search_engine.get_available_sources(),
                'domain': domain,
                'processing_time_seconds': processing_time,
                'average_relevance_score': sum(p.relevance_score for p in papers) / len(papers) if papers else 0,
                'database_type': 'local',
                'paper_sources': list(set([p.source for p in papers])) if papers else []
            }
            
            # Step 5: Create response
            response = RAGResponse(
                query=query,
                retrieved_papers=papers,
                summary=summary,
                hypotheses=hypotheses,
                metadata=metadata,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Local query processed successfully in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error processing local query: {e}")
            # Return error response
            return RAGResponse(
                query=query,
                retrieved_papers=[],
                summary=f"Error processing query with local database: {str(e)}",
                hypotheses=[],
                metadata={'error': str(e), 'database_type': 'local'},
                timestamp=datetime.now().isoformat()
            )
    
    def add_papers_to_vector_store(self, papers: List[SearchResult], domain: str = 'general'):
        """Add papers to vector store for enhanced retrieval"""
        if not papers:
            return
        
        documents = []
        metadata = []
        
        for paper in papers:
            # Combine title and abstract for document text
            document_text = f"{paper.title}\n\n{paper.abstract}"
            documents.append(document_text)
            
            # Create metadata
            paper_metadata = {
                'id': paper.id,
                'title': paper.title,
                'authors': paper.authors,
                'source': paper.source,
                'url': paper.url,
                'published_date': paper.published_date,
                'citation_count': paper.citation_count,
                'venue': paper.venue,
                'categories': paper.categories,
                'doi': paper.doi,
                'relevance_score': paper.relevance_score
            }
            metadata.append(paper_metadata)
        
        # Add to vector store
        try:
            self.vector_store.add_documents(documents, metadata, domain)
            logger.info(f"Added {len(papers)} papers to vector store")
        except Exception as e:
            logger.error(f"Error adding papers to vector store: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database and pipeline statistics"""
        return self.search_engine.get_database_stats()
    
    def get_available_sources(self) -> List[str]:
        """Get available sources in the local database"""
        return self.search_engine.get_available_sources()
    
    def save_pipeline_state(self, path: str):
        """Save pipeline configuration"""
        state = {
            'max_retrieved_docs': self.max_retrieved_docs,
            'database_stats': self.get_database_stats(),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Pipeline state saved to {path}")
    
    def load_pipeline_state(self, path: str):
        """Load pipeline configuration"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.max_retrieved_docs = state.get('max_retrieved_docs', 10)
            logger.info(f"Pipeline state loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading pipeline state: {e}")

# Convenience function to create local RAG pipeline
def create_local_rag_pipeline(db_path: str = "data/papers/papers.db") -> LocalRAGPipeline:
    """Create and return a LocalRAGPipeline instance"""
    return LocalRAGPipeline(db_path) 