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
        self.search_engine = LocalSearchEngine(db_path)
        self.llm_manager = LLMManager()
        self.target_paper_count = 5
        
        logger.info("Initialized Local Enhanced RAG with database")
        
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
        """Generate enhanced LLM-based summary of findings from local papers"""
        logger.info("Generating enhanced summarized findings using LLM...")
        
        # Prepare enhanced context for LLM
        papers_context = self._prepare_enhanced_papers_context(papers)
        
        # Get database statistics for context
        db_stats = await self._get_db_stats()
        
        prompt = f"""
You are an AI research expert analyzing papers from a curated local database. Based on the following {len(papers)} high-quality research papers related to "{query}", provide a comprehensive and insightful summary.

Database Context:
- Total papers in database: {db_stats.get('total_papers', 'Unknown')}
- Sources: {', '.join(db_stats.get('source_distribution', {}).keys())}
- Selected {len(papers)} most relevant papers for analysis

Research Papers:
{papers_context}

Task: Analyze these papers and provide a detailed summary that includes:

1. **Core Research Themes**: What are the main approaches and methodologies?
2. **Key Scientific Findings**: What significant results and discoveries are reported?
3. **Methodological Insights**: What research methods and techniques are being used?
4. **Current State Assessment**: Where does the field stand based on these papers?
5. **Research Gaps & Opportunities**: What limitations or future directions are identified?
6. **Cross-Paper Synthesis**: How do findings across papers relate and build upon each other?

Provide a comprehensive summary (400-600 words) that demonstrates deep understanding of the research landscape.

Enhanced Research Summary:
"""
        
        try:
            response = await self.llm_manager.generate(prompt, max_tokens=800, temperature=0.7)
            if response and hasattr(response, 'content') and response.content:
                return response.content.strip()
            else:
                logger.warning("Empty or invalid LLM response for enhanced summary")
                return self._generate_enhanced_fallback_summary(papers, query)
        except Exception as e:
            logger.error(f"Error generating enhanced summary: {e}")
            return self._generate_enhanced_fallback_summary(papers, query)
    
    async def _generate_enhanced_research_hypothesis(self, papers: List[PaperInfo], query: str, summary: str) -> str:
        """Generate enhanced research hypothesis based on local papers and summary"""
        logger.info("Generating enhanced research hypothesis using LLM...")
        
        papers_context = self._prepare_enhanced_papers_context(papers)
        db_stats = await self._get_db_stats()
        
        prompt = f"""
You are a leading AI research scientist tasked with generating a novel, impactful research hypothesis. You have access to a curated database of {db_stats.get('total_papers', 'many')} research papers and have analyzed the {len(papers)} most relevant papers for the query: "{query}"

Research Context:
{summary}

Detailed Papers Analysis:
{papers_context}

Based on this comprehensive analysis of local research papers, generate a novel research hypothesis that:

1. **Builds on Existing Work**: Leverages insights from the analyzed papers
2. **Identifies Clear Gaps**: Addresses limitations or unexplored areas
3. **Proposes Innovation**: Suggests new methodologies, applications, or theoretical frameworks
4. **Has Practical Impact**: Could advance the field or solve real-world problems
5. **Is Testable**: Can be validated through research methodologies
6. **Synthesizes Knowledge**: Combines insights from multiple papers in novel ways

Generate a well-structured research hypothesis that includes:
- The core hypothesis statement
- Scientific rationale based on the papers
- Potential research methodology
- Expected contributions to the field

Novel Research Hypothesis:
"""
        
        try:
            response = await self.llm_manager.generate(prompt, max_tokens=600, temperature=0.8)
            if response and hasattr(response, 'content') and response.content:
                return response.content.strip()
            else:
                logger.warning("Empty or invalid LLM response for enhanced hypothesis")
                return self._generate_enhanced_fallback_hypothesis(papers, query)
        except Exception as e:
            logger.error(f"Error generating enhanced hypothesis: {e}")
            return self._generate_enhanced_fallback_hypothesis(papers, query)
    
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
    
    def _generate_enhanced_fallback_summary(self, papers: List[PaperInfo], query: str) -> str:
        """Generate enhanced template-based summary for local papers"""
        
        # Analyze paper characteristics
        source_counts = {}
        total_citations = 0
        venues = []
        authors_count = 0
        
        for paper in papers:
            source_counts[paper.source] = source_counts.get(paper.source, 0) + 1
            if paper.citation_count:
                total_citations += paper.citation_count
            if paper.venue:
                venues.append(paper.venue)
            authors_count += len(paper.authors)
        
        unique_venues = list(set(venues))[:3]
        avg_citations = total_citations / len(papers) if papers else 0
        
        # Generate enhanced summary
        summary_parts = [
            f"# Enhanced Research Analysis: {query}",
            f"*Analysis of {len(papers)} curated papers from local database*\n",
            
            "## Dataset Overview",
            f"**Papers Analyzed:** {len(papers)} high-quality research papers",
            f"**Source Distribution:** {', '.join(f'{source}: {count}' for source, count in source_counts.items())}",
            f"**Research Impact:** {total_citations} total citations (avg: {avg_citations:.1f} per paper)",
            f"**Author Diversity:** {authors_count} total authors across papers",
        ]
        
        if unique_venues:
            summary_parts.append(f"**Key Venues:** {', '.join(unique_venues)}")
        
        # Research themes analysis
        summary_parts.extend([
            "\n## Core Research Themes",
            f"The {len(papers)} papers in our local database reveal several key research directions in {query}:",
        ])
        
        # Add paper-specific insights
        for i, paper in enumerate(papers[:3], 1):
            source_note = f"[{paper.source}]"
            summary_parts.append(f"• **Theme {i}:** {paper.title[:80]}... {source_note}")
        
        # Methodological insights
        summary_parts.extend([
            "\n## Key Findings & Methodologies",
            f"• **Diverse Approaches:** Papers span from {min(source_counts.values())} to {max(source_counts.values())} per source, indicating methodological diversity",
            f"• **Research Quality:** Average citation count of {avg_citations:.1f} suggests impactful research",
            f"• **Current Focus:** The {len(papers)} papers represent current state-of-the-art in {query}",
        ])
        
        # Cross-paper synthesis
        if len(papers) >= 3:
            summary_parts.extend([
                "\n## Research Landscape Analysis",
                f"• **Interdisciplinary Nature:** {len(source_counts)} different sources indicate cross-disciplinary relevance",
                f"• **Evolution:** Research spans multiple publication venues showing field maturity",
                f"• **Knowledge Gaps:** Analysis of {len(papers)} papers reveals opportunities for future research",
            ])
        
        # Future directions
        summary_parts.extend([
            "\n## Implications & Future Directions",
            f"• **Field Advancement:** These {len(papers)} papers provide solid foundation for future research in {query}",
            f"• **Methodological Innovation:** Cross-source analysis suggests opportunities for novel approaches",
            f"• **Practical Applications:** Research indicates potential for real-world impact and implementation",
        ])
        
        return "\n".join(summary_parts)
    
    def _generate_enhanced_fallback_hypothesis(self, papers: List[PaperInfo], query: str) -> str:
        """Generate enhanced template-based hypothesis for local papers"""
        
        sources = list(set([paper.source for paper in papers]))
        total_citations = sum([paper.citation_count or 0 for paper in papers])
        venues = list(set([paper.venue for paper in papers if paper.venue]))
        
        hypothesis_parts = [
            f"# Novel Research Hypothesis for {query}",
            "*Based on analysis of local database papers*\n",
            
            "## Core Hypothesis Statement",
            f"Building upon the insights from {len(papers)} high-quality research papers in our local database, "
            f"we hypothesize that future advancement in {query} could be achieved through a novel "
            f"synthesis of methodologies identified across {', '.join(sources)} sources.\n",
            
            "## Scientific Rationale",
            f"**Evidence Base:** Analysis of {len(papers)} papers with {total_citations} total citations",
            f"**Cross-Source Insights:** Integration of findings from {len(sources)} different research sources",
            f"**Publication Diversity:** Research spanning {len(venues)} distinct venues indicates broad applicability\n",
            
            "## Proposed Research Direction",
            "The hypothesis suggests investigating:",
            f"• **Methodological Synthesis:** Combining approaches from {' and '.join(sources)} research",
            f"• **Gap Addressing:** Filling identified limitations in current {query} research",
            f"• **Novel Applications:** Extending findings to unexplored domains within {query}",
            f"• **Enhanced Performance:** Improving upon current state-of-the-art through innovative integration\n",
            
            "## Expected Contributions",
            f"• **Theoretical Advancement:** New frameworks for understanding {query}",
            f"• **Methodological Innovation:** Novel approaches synthesized from {len(papers)} paper analysis",
            f"• **Practical Impact:** Solutions addressing real-world challenges in {query}",
            f"• **Field Unification:** Bridging insights across {len(sources)} research communities\n",
            
            "## Research Methodology",
            "Proposed validation through:",
            f"• **Experimental Validation:** Testing hypotheses derived from {len(papers)} paper analysis",
            f"• **Comparative Studies:** Benchmarking against methods identified in local database",
            f"• **Cross-Domain Testing:** Applying insights across {query} subfields",
            f"• **Collaborative Research:** Engaging with authors from analyzed papers for validation"
        ]
        
        return "\n".join(hypothesis_parts)
    
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