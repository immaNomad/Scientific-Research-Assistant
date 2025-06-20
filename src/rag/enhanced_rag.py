"""
Enhanced RAG System for Scientific Research
Designed for extracting 5 papers, generating summaries and hypotheses using LLM
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import json
from datetime import datetime

from src.api import ArxivClient, SemanticScholarClient
from src.models.llm_client import LLMManager, LLMResponse
from src.rag.rag_pipeline import SearchResult, RAGPipeline

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
    """Complete research analysis output"""
    query: str
    papers: List[PaperInfo]
    summarized_findings: str
    hypothesis: str
    processing_metadata: Dict
    timestamp: str

class EnhancedRAG:
    """Enhanced RAG system for scientific research with LLM integration"""
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.llm_manager = LLMManager()
        self.target_paper_count = 5
        
    async def research_and_analyze(self, query: str, sources: List[str] = None) -> ResearchAnalysis:
        """
        Complete research analysis pipeline:
        1. Extract 5 relevant papers
        2. Generate LLM-based summary of findings
        3. Generate research hypothesis
        
        Args:
            query: Research query
            sources: List of sources to search
            
        Returns:
            Complete research analysis
        """
        start_time = datetime.now()
        logger.info(f"Starting enhanced research analysis for: '{query}'")
        
        # Step 1: Extract exactly 5 papers
        papers = await self._extract_five_papers(query, sources)
        
        # Step 2: Generate summarized findings using LLM
        summarized_findings = await self._generate_summarized_findings(papers, query)
        
        # Step 3: Generate hypothesis using LLM
        hypothesis = await self._generate_research_hypothesis(papers, query, summarized_findings)
        
        # Prepare metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        llm_model = 'fallback'
        if self.llm_manager.clients:
            client_name, client = self.llm_manager.clients[0]
            llm_model = getattr(client, 'model', client_name)
        
        metadata = {
            "papers_found": len(papers),
            "processing_time_seconds": processing_time,
            "sources_used": list(set([paper.source for paper in papers])),
            "llm_model_used": llm_model
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
        """Extract exactly 5 most relevant papers"""
        logger.info("Extracting 5 most relevant papers...")
        
        # Use existing RAG pipeline to get papers
        search_results = await self.rag_pipeline.search_literature(
            query=query,
            sources=sources or ['arxiv', 'semantic_scholar'],
            max_results_per_source=15  # Get more to ensure we have 5 good ones
        )
        
        # Convert to PaperInfo format and take top 5
        papers = []
        for result in search_results[:self.target_paper_count]:
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
        
        logger.info(f"Successfully extracted {len(papers)} papers")
        return papers
    
    async def _generate_summarized_findings(self, papers: List[PaperInfo], query: str) -> str:
        """Generate LLM-based summary of findings from the papers"""
        logger.info("Generating summarized findings using LLM...")
        
        # Prepare context for LLM
        papers_context = self._prepare_papers_context_for_llm(papers)
        
        prompt = f"""
You are a scientific research expert. Based on the following research papers related to the query "{query}", provide a comprehensive summary of the key findings.

Research Papers:
{papers_context}

Task: Analyze these papers and provide a well-structured summary that includes:
1. Main research themes and approaches
2. Key findings and results
3. Current state of the field
4. Gaps or limitations identified
5. Emerging trends and technologies

Please provide a detailed summary (300-500 words) that synthesizes the findings across all papers.

Summary:
"""
        
        try:
            response = await self.llm_manager.generate(prompt, max_tokens=600, temperature=0.7)
            if response and hasattr(response, 'content') and response.content:
                return response.content.strip()
            else:
                logger.warning("Empty or invalid LLM response for summary")
                return self._generate_fallback_summary(papers, query)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._generate_fallback_summary(papers, query)
    
    async def _generate_research_hypothesis(self, papers: List[PaperInfo], query: str, summary: str) -> str:
        """Generate research hypothesis based on papers and summary"""
        logger.info("Generating research hypothesis using LLM...")
        
        papers_context = self._prepare_papers_context_for_llm(papers)
        
        prompt = f"""
You are a scientific research expert tasked with generating a novel research hypothesis. Based on the following information:

Query: "{query}"

Research Summary:
{summary}

Research Papers Context:
{papers_context}

Task: Generate a specific, testable research hypothesis that:
1. Addresses gaps identified in the current research
2. Builds upon existing findings
3. Proposes a novel approach or investigation
4. Is feasible and well-motivated
5. Could advance the field significantly

Please provide:
1. A clear hypothesis statement
2. Brief justification (2-3 sentences)
3. Potential methodology approach (1-2 sentences)

Hypothesis:
"""
        
        try:
            response = await self.llm_manager.generate(prompt, max_tokens=400, temperature=0.8)
            if response and hasattr(response, 'content') and response.content:
                return response.content.strip()
            else:
                logger.warning("Empty or invalid LLM response for hypothesis")
                return self._generate_fallback_hypothesis(papers, query)
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return self._generate_fallback_hypothesis(papers, query)
    
    def _prepare_papers_context_for_llm(self, papers: List[PaperInfo]) -> str:
        """Prepare paper context for LLM prompts"""
        context = ""
        for i, paper in enumerate(papers, 1):
            context += f"\nPaper {i}:\n"
            context += f"Title: {paper.title}\n"
            context += f"Authors: {', '.join(paper.authors[:3])}"
            if len(paper.authors) > 3:
                context += f" et al."
            context += f"\nDOI: {paper.doi or 'N/A'}\n"
            context += f"Source: {paper.source.upper()}\n"
            if paper.citation_count:
                context += f"Citations: {paper.citation_count}\n"
            
            # Handle potential None abstracts
            abstract_text = paper.abstract if paper.abstract else "No abstract available"
            context += f"Abstract: {abstract_text[:300]}...\n"
            context += "-" * 50
        
        return context
    
    def _generate_fallback_summary(self, papers: List[PaperInfo], query: str) -> str:
        """Generate fallback summary when LLM is not available"""
        summary = f"Analysis of {len(papers)} research papers related to '{query}':\n\n"
        
        # Extract key themes - handle None abstracts
        all_abstracts = " ".join([paper.abstract for paper in papers if paper.abstract])
        summary += "Key research areas covered:\n"
        
        # Simple keyword extraction
        keywords = self._extract_keywords(all_abstracts)
        for keyword in keywords[:5]:
            summary += f"- {keyword}\n"
        
        summary += f"\nThe papers span multiple sources including {', '.join(set([paper.source for paper in papers]))}. "
        summary += f"Publication dates range from recent works to established research. "
        summary += f"This collection provides a comprehensive overview of current approaches and findings in the field."
        
        return summary
    
    def _generate_fallback_hypothesis(self, papers: List[PaperInfo], query: str) -> str:
        """Generate fallback hypothesis when LLM is not available"""
        return f"""Based on the analysis of recent literature on '{query}', I hypothesize that:

Integrating multiple approaches identified in the reviewed papers could lead to significant improvements in performance and applicability. Specifically, combining the methodologies from the top-cited papers with emerging techniques could address current limitations.

This hypothesis is supported by the diverse range of approaches found in the literature and the identified gaps between different research directions. Future work should focus on systematic integration and empirical validation of combined approaches."""
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction for fallback summary"""
        # Basic keyword extraction - in practice, you might want to use more sophisticated NLP
        words = text.lower().split()
        
        # Common research terms
        research_terms = []
        key_phrases = ['machine learning', 'deep learning', 'neural network', 'algorithm', 'model', 
                      'performance', 'accuracy', 'method', 'approach', 'framework', 'system',
                      'analysis', 'evaluation', 'optimization', 'classification', 'prediction']
        
        for phrase in key_phrases:
            if phrase in text.lower():
                research_terms.append(phrase.title())
        
        return research_terms[:10]
    
    def format_analysis_output(self, analysis: ResearchAnalysis) -> str:
        """Format the complete analysis for display"""
        output = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          RESEARCH ANALYSIS REPORT                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

🔍 QUERY: {analysis.query}

📚 EXTRACTED PAPERS (Top 5):
{'='*80}
"""
        
        for i, paper in enumerate(analysis.papers, 1):
            output += f"""
{i}. {paper.title}
   Authors: {', '.join(paper.authors[:3])}{'et al.' if len(paper.authors) > 3 else ''}
   DOI: {paper.doi or 'Not available'}
   Source: {paper.source.upper()}
   URL: {paper.url}
   {"Citations: " + str(paper.citation_count) if paper.citation_count else ""}
   {"Published: " + str(paper.published_date)[:10] if paper.published_date else ""}

"""
        
        output += f"""
📝 SUMMARIZED FINDINGS:
{'-'*40}
{analysis.summarized_findings}

💡 RESEARCH HYPOTHESIS:
{'-'*40}
{analysis.hypothesis}

📊 METADATA:
{'-'*40}
Papers analyzed: {analysis.processing_metadata['papers_found']}
Sources used: {', '.join(analysis.processing_metadata['sources_used'])}
LLM model: {analysis.processing_metadata['llm_model_used']}
Processing time: {analysis.processing_metadata['processing_time_seconds']:.2f} seconds
Timestamp: {analysis.timestamp}

"""
        return output 