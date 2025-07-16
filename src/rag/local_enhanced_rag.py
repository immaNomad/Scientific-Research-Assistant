"""
Local Enhanced RAG System
Uses local database instead of external APIs for enhanced research analysis
Provides content generation, summarization, and hypothesis generation
"""

import asyncio
from typing import List, Dict, Optional, Tuple
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
from database.inclusive_search import InclusiveSearchEngine
from database.topic_search import TopicSearchEngine
from models.llm_client import LLMManager, LLMResponse
from rag.local_rag_pipeline import LocalRAGPipeline

@dataclass
class PaperInfo:
    """Structured information for a research paper"""
    title: str
    doi: Optional[str]
    authors: List[str]
    abstract: str
    source: str
    url: str
    published_date: Optional[str] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None

@dataclass
class ResearchAnalysis:
    """Complete research analysis output with enhanced content generation"""
    query: str
    papers: List[PaperInfo]
    summarized_findings: str
    hypothesis: str
    processing_metadata: Dict
    timestamp: str

class LocalEnhancedRAG:
    """
    Enhanced RAG system using local database
    Provides advanced content generation from local AI/ML paper dataset
    """
    
    def __init__(self, db_path: str = "data/papers/papers.db"):
        self.local_rag = LocalRAGPipeline(db_path)
        self.search_engine = TopicSearchEngine(db_path)  # Use topic-focused search
        self.llm_manager = LLMManager()
        self.target_paper_count = 5
        
        logger.info("Initialized Local Enhanced RAG with topic-focused search database")
        
    async def research_and_analyze(self, query: str, sources: List[str] = None) -> ResearchAnalysis:
        """
        Complete research analysis pipeline using local database:
        1. Extract 5 relevant papers from local database
        2. Generate enhanced LLM-based summary of findings
        3. Generate novel research hypothesis
        
        Args:
            query: Research query
            sources: List of sources to search (optional)
            
        Returns:
            Complete research analysis with enhanced content generation
        """
        start_time = datetime.now()
        logger.info(f"Starting local enhanced research analysis for: '{query}'")
        
        # Step 1: Extract exactly 5 papers from local database
        papers = await self._extract_five_papers(query, sources)
        
        if not papers:
            return self._create_empty_analysis(query, "No papers found in local database")
        
        # Step 2: Generate enhanced summarized findings using LLM
        summarized_findings = await self._generate_enhanced_summarized_findings(papers, query)
        
        # Step 3: Generate novel research hypothesis using LLM
        hypothesis = await self._generate_enhanced_research_hypothesis(papers, query, summarized_findings)
        
        # Prepare enhanced metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        llm_model = 'fallback'
        if self.llm_manager.clients:
            client_name, client = self.llm_manager.clients[0]
            llm_model = getattr(client, 'model', client_name)
        
        metadata = {
            "papers_found": len(papers),
            "processing_time_seconds": processing_time,
            "sources_used": list(set([paper.source for paper in papers])),
            "llm_model_used": llm_model,
            "database_type": "local",
            "total_papers_in_db": (await self._get_db_stats())['total_papers'],
            "query_relevance_scores": [getattr(p, 'relevance_score', 0.0) for p in papers]
        }
        
        return ResearchAnalysis(
            query=query,
            papers=papers,
            summarized_findings=summarized_findings,
            hypothesis=hypothesis,
            processing_metadata=metadata,
            timestamp=datetime.now().isoformat()
        )
    
    async def _extract_five_papers(self, query: str, sources: List[str] = None) -> List[PaperInfo]:
        """Extract exactly 5 most relevant papers from local database"""
        logger.info("Extracting 5 most relevant papers from local database...")
        
        # Use local search to get papers
        search_results = await self.search_engine.search_literature(
            query=query,
            sources=sources,
            max_results_per_source=15  # Get more to ensure diversity
        )
        
        if not search_results:
            logger.warning("No papers found in local database")
            return []
        
        # Convert to PaperInfo format with source diversity
        papers = []
        
        # Ensure source diversity: try to get papers from each available source
        papers_by_source = {}
        for result in search_results:
            source = result.source
            if source not in papers_by_source:
                papers_by_source[source] = []
            papers_by_source[source].append(result)
        
        # Select papers with diversity: aim for balanced representation
        max_per_source = max(1, self.target_paper_count // len(papers_by_source)) if papers_by_source else 1
        selected_papers = []
        
        # First pass: take top papers from each source
        for source, source_papers in papers_by_source.items():
            for paper in source_papers[:max_per_source]:
                if len(selected_papers) < self.target_paper_count:
                    selected_papers.append(paper)
        
        # Second pass: fill remaining slots with highest-ranked papers
        remaining_slots = self.target_paper_count - len(selected_papers)
        if remaining_slots > 0:
            remaining_papers = [p for p in search_results if p not in selected_papers]
            # Sort by relevance score
            remaining_papers.sort(key=lambda x: x.relevance_score, reverse=True)
            selected_papers.extend(remaining_papers[:remaining_slots])
        
        # Convert to PaperInfo format
        for result in selected_papers:
            paper = PaperInfo(
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
            papers.append(paper)
        
        logger.info(f"Successfully extracted {len(papers)} papers from local database")
        return papers
    
    async def _generate_enhanced_summarized_findings(self, papers: List[PaperInfo], query: str) -> str:
        """Generate enhanced LLM-based summary of findings from local papers in paragraph format"""
        logger.info("Generating enhanced summarized findings using LLM...")
        
        # Prepare enhanced context for LLM
        papers_context = self._prepare_enhanced_papers_context(papers)
        
        # Get database statistics for context
        db_stats = await self._get_db_stats()
        
        # Generate concise paragraph summary mentioning paper names
        paragraph_prompt = f"""
You are a research analyst examining {len(papers)} papers from a database of {db_stats.get('total_papers', 'many')} papers on "{query}".

Papers analyzed:
{papers_context}

Create a concise paragraph summary (200-300 words) that mentions each paper by name and their key findings. Use the format:
"'Paper Title 1' indicates that [key finding]... while 'Paper Title 2' shows [finding]... additionally 'Paper Title 3' demonstrates [finding]..."

Focus on:
- Mention each paper's title in single quotes
- Connect findings with transitions like "while", "additionally", "furthermore", "moreover"
- Keep it flowing as one cohesive paragraph
- Highlight the most important findings from each paper

Summarized Findings:
"""
        
        try:
            response = await self.llm_manager.generate(paragraph_prompt, max_tokens=400, temperature=0.6)
            if response and hasattr(response, 'content') and response.content:
                content = response.content.strip()
                # Check if the response follows the expected format (mentions paper titles in quotes)
                if any(f"'{paper.title}'" in content for paper in papers):
                    return content
                else:
                    logger.warning("LLM response doesn't follow expected paragraph format, using fallback")
                    return self._generate_paragraph_fallback(papers, query)
            else:
                logger.warning("Empty response for summarized findings")
                return self._generate_paragraph_fallback(papers, query)
        except Exception as e:
            logger.error(f"Error generating summarized findings: {e}")
            return self._generate_paragraph_fallback(papers, query)
    
    def _generate_paragraph_fallback(self, papers: List[PaperInfo], query: str) -> str:
        """Generate fallback paragraph summary mentioning paper names"""
        if not papers:
            return f"No papers found for analysis of '{query}'."
        
        # Create paragraph mentioning each paper
        findings_parts = []
        
        for i, paper in enumerate(papers):
            title = paper.title
            # Extract key info from abstract
            abstract_sentences = paper.abstract.split('.')[:2]  # First 2 sentences
            key_finding = '. '.join(abstract_sentences).strip()
            
            if i == 0:
                findings_parts.append(f"'{title}' indicates that {key_finding}")
            elif i == len(papers) - 1:
                findings_parts.append(f"finally '{title}' demonstrates {key_finding}")
            else:
                transition = ["while", "additionally", "furthermore", "moreover"][i % 4]
                findings_parts.append(f"{transition} '{title}' shows {key_finding}")
        
        paragraph = ". ".join(findings_parts) + "."
        
        # Add metadata
        source_count = len(set(paper.source for paper in papers))
        citation_total = sum(paper.citation_count or 0 for paper in papers)
        
        metadata = f" This analysis draws from {len(papers)} papers across {source_count} research sources with a combined citation count of {citation_total}, providing comprehensive insights into {query}."
        
        return paragraph + metadata
    
    async def _generate_enhanced_research_hypothesis(self, papers: List[PaperInfo], query: str, summary: str) -> str:
        """Generate enhanced research hypothesis based on local papers and summary in numbered format"""
        logger.info("Generating enhanced research hypothesis using LLM...")
        
        papers_context = self._prepare_enhanced_papers_context(papers)
        db_stats = await self._get_db_stats()
        
        # Generate numbered hypothesis list
        hypothesis_prompt = f"""
Based on analysis of {len(papers)} research papers on "{query}" from a database of {db_stats.get('total_papers', 'many')} papers:

Research Summary:
{summary[:600]}...

Papers Context:
{papers_context}

Generate 3-5 numbered research hypotheses that:
1. Are specific and testable
2. Build on existing work from the analyzed papers
3. Identify clear research gaps
4. Propose novel approaches or improvements
5. Are feasible for future research

Format as a numbered list:
1. [First hypothesis]
2. [Second hypothesis]
3. [Third hypothesis]
etc.

Each hypothesis should be 1-2 sentences and focus on a specific aspect of {query}.

Research Hypotheses:
"""
        
        try:
            response = await self.llm_manager.generate(hypothesis_prompt, max_tokens=400, temperature=0.7)
            if response and hasattr(response, 'content') and response.content:
                content = response.content.strip()
                # Check if the response follows the expected format (numbered list)
                if any(f"{i}." in content for i in range(1, 6)):
                    return content
                else:
                    logger.warning("LLM response doesn't follow expected numbered format, using fallback")
                    return self._generate_numbered_hypothesis_fallback(papers, query)
            else:
                logger.warning("Empty response for hypothesis")
                return self._generate_numbered_hypothesis_fallback(papers, query)
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return self._generate_numbered_hypothesis_fallback(papers, query)
    
    def _generate_numbered_hypothesis_fallback(self, papers: List[PaperInfo], query: str) -> str:
        """Generate fallback numbered hypothesis"""
        if not papers:
            return f"1. No hypotheses can be generated without relevant papers for '{query}'."
        
        # Generate numbered hypotheses based on papers
        hypotheses = []
        
        # Extract key themes from papers
        sources = list(set(paper.source for paper in papers))
        citation_total = sum(paper.citation_count or 0 for paper in papers)
        
        # Generate specific numbered hypotheses
        hypotheses.append(f"1. Integrating methodologies from the analyzed {len(papers)} papers on {query} could yield a hybrid approach that outperforms individual techniques by combining their complementary strengths.")
        
        hypotheses.append(f"2. The gaps identified in current {query} research suggest that cross-domain validation using datasets from multiple sources ({', '.join(sources)}) would significantly improve generalization capabilities.")
        
        hypotheses.append(f"3. Building upon the findings from these high-impact papers (combined {citation_total} citations), a novel framework could be developed to address the scalability limitations mentioned across multiple studies.")
        
        hypotheses.append(f"4. The convergence of themes across different research sources indicates that a unified theoretical model for {query} could provide better explanatory power than current fragmented approaches.")
        
        if len(papers) >= 4:
            hypotheses.append(f"5. Future research could benefit from investigating the unexplored intersections between the methodologies presented in these {len(papers)} papers to discover novel applications for {query}.")
        
        return "\n".join(hypotheses)
    
    def _prepare_enhanced_papers_context(self, papers: List[PaperInfo]) -> str:
        """Prepare enhanced context from papers for LLM processing"""
        context_parts = []
        
        for i, paper in enumerate(papers, 1):
            # Calculate additional metrics
            abstract_word_count = len(paper.abstract.split())
            
            paper_info = f"""
Paper {i}: {paper.title}
Authors: {', '.join(paper.authors[:4])}{'...' if len(paper.authors) > 4 else ''}
Source: {paper.source} | Published: {paper.published_date or 'Unknown'}
Venue: {paper.venue or 'Unknown'} | Citations: {paper.citation_count or 'N/A'}
Abstract ({abstract_word_count} words): {paper.abstract}
"""
            if paper.doi:
                paper_info += f"DOI: {paper.doi}\n"
            
            context_parts.append(paper_info.strip())
        
        return "\n" + "="*80 + "\n".join(context_parts) + "\n" + "="*80
    
    async def _get_db_stats(self) -> Dict:
        """Get database statistics"""
        try:
            return self.search_engine.get_database_stats()
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'total_papers': 'Unknown', 'source_distribution': {}}
    
    def _create_empty_analysis(self, query: str, reason: str) -> ResearchAnalysis:
        """Create empty analysis when no papers found"""
        return ResearchAnalysis(
            query=query,
            papers=[],
            summarized_findings=f"No research papers found in local database for query: '{query}'. {reason}",
            hypothesis="Cannot generate hypothesis without relevant papers in local database.",
            processing_metadata={
                "papers_found": 0,
                "error": reason,
                "database_type": "local"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def format_analysis_output(self, analysis: ResearchAnalysis) -> str:
        """Format the complete analysis output for display"""
        
        output_parts = [
            "=" * 80,
            f"ENHANCED RESEARCH ANALYSIS: {analysis.query}",
            "=" * 80,
            f"Generated: {analysis.timestamp}",
            f"Papers Analyzed: {len(analysis.papers)}",
            f"Processing Time: {analysis.processing_metadata.get('processing_time_seconds', 'Unknown'):.2f}s",
            f"Database Type: {analysis.processing_metadata.get('database_type', 'Unknown')}",
            "",
            "RESEARCH PAPERS:",
            "-" * 40
        ]
        
        # Add paper information
        for i, paper in enumerate(analysis.papers, 1):
            output_parts.extend([
                f"{i}. {paper.title}",
                f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}",
                f"   Source: {paper.source} | Citations: {paper.citation_count or 'N/A'}",
                f"   URL: {paper.url}",
                ""
            ])
        
        # Add summary and hypothesis
        output_parts.extend([
            "SUMMARIZED FINDINGS:",
            "-" * 40,
            analysis.summarized_findings,
            "",
            "RESEARCH HYPOTHESIS:",
            "-" * 40,
            analysis.hypothesis,
            "",
            "=" * 80
        ])
        
        return "\n".join(output_parts)

# Convenience function to create local enhanced RAG
def create_local_enhanced_rag(db_path: str = "data/papers/papers.db") -> LocalEnhancedRAG:
    """Create and return a LocalEnhancedRAG instance"""
    return LocalEnhancedRAG(db_path) 