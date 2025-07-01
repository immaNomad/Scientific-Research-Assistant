import asyncio
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from loguru import logger
import json
from datetime import datetime

from src.api import ArxivClient, SemanticScholarClient, PubMedClient
from src.models.embeddings import EmbeddingManager, VectorStore, DocumentChunker
from src.models.llm_client import llm_manager, LLMResponse
from config.config import config

@dataclass
class SearchResult:
    """Unified search result from multiple sources"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    source: str  # 'arxiv', 'pubmed', 'semantic_scholar'
    url: str
    published_date: Optional[str] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    categories: Optional[List[str]] = None
    doi: Optional[str] = None
    relevance_score: float = 0.0

@dataclass
class RAGResponse:
    """Response from RAG pipeline"""
    query: str
    retrieved_papers: List[SearchResult]
    summary: str
    hypotheses: List[str]
    metadata: Dict
    timestamp: str

class RAGPipeline:
    """Main RAG pipeline for scientific research assistance"""
    
    def __init__(self):
        # Initialize API clients
        self.arxiv_client = ArxivClient()
        self.semantic_scholar_client = SemanticScholarClient()
        self.pubmed_client = PubMedClient()
        
        # Initialize models
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.document_chunker = DocumentChunker()
        
        # Configuration
        self.max_retrieved_docs = config.rag.MAX_RETRIEVED_DOCS
        
    async def search_literature(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results_per_source: int = 20,
                               domain: str = 'general') -> List[SearchResult]:
        """
        Search across multiple literature sources
        
        Args:
            query: Search query
            sources: List of sources to search ('arxiv', 'pubmed', 'semantic_scholar')
            max_results_per_source: Maximum results per source
            domain: Domain for relevance filtering
            
        Returns:
            List of unified search results
        """
        if sources is None:
            sources = ['arxiv', 'pubmed', 'semantic_scholar']  # Default sources
        
        logger.info(f"Searching literature for query: '{query}' in sources: {sources}")
        
        # Create async tasks for each source
        tasks = []
        
        if 'arxiv' in sources:
            tasks.append(self._search_arxiv(query, max_results_per_source))
        
        if 'pubmed' in sources:
            tasks.append(self._search_pubmed(query, max_results_per_source))
        
        if 'semantic_scholar' in sources:
            tasks.append(self._search_semantic_scholar(query, max_results_per_source))
        
        # Execute searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in source {sources[i] if i < len(sources) else i}: {result}")
                continue
            if isinstance(result, list):
                all_results.extend(result)
        
        # Remove duplicates and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = await self._rank_results(unique_results, query, domain)
        
        # Return top results
        return ranked_results[:self.max_retrieved_docs]
    
    async def _search_arxiv(self, query: str, max_results: int) -> List[SearchResult]:
        """Search arXiv and convert to unified format"""
        try:
            papers = await self.arxiv_client.search(query, max_results)
            results = []
            
            for paper in papers:
                result = SearchResult(
                    id=paper.id,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    source='arxiv',
                    url=paper.url,
                    published_date=paper.published,
                    categories=paper.categories,
                    doi=paper.doi
                )
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    async def _search_pubmed(self, query: str, max_results: int) -> List[SearchResult]:
        """Search PubMed and convert to unified format"""
        try:
            papers = await self.pubmed_client.search(query, max_results)
            results = []
            
            for paper in papers:
                result = SearchResult(
                    id=paper.id,
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    source='pubmed',
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/" if paper.pmid else "",
                    published_date=paper.published_date,
                    venue=paper.journal,
                    doi=paper.doi
                )
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []

    async def _search_semantic_scholar(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Semantic Scholar and convert to unified format"""
        try:
            papers = await self.semantic_scholar_client.search(query, max_results)
            results = []
            
            for paper in papers:
                result = SearchResult(
                    id=paper.id,
                    title=paper.title,
                    authors=[author['name'] for author in paper.authors],
                    abstract=paper.abstract,
                    source='semantic_scholar',
                    url=paper.url,
                    published_date=str(paper.year) if paper.year else None,
                    citation_count=paper.citation_count,
                    venue=paper.venue,
                    categories=paper.fields_of_study,
                    doi=paper.doi
                )
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate papers from different sources"""
        seen_titles = set()
        seen_dois = set()
        unique_results = []
        
        for result in results:
            # Check for duplicates by title (normalized)
            normalized_title = result.title.lower().strip()
            if normalized_title in seen_titles:
                continue
            
            # Check for duplicates by DOI
            if result.doi and result.doi in seen_dois:
                continue
            
            seen_titles.add(normalized_title)
            if result.doi:
                seen_dois.add(result.doi)
            
            unique_results.append(result)
        
        return unique_results
    
    async def _rank_results(self, 
                           results: List[SearchResult], 
                           query: str, 
                           domain: str) -> List[SearchResult]:
        """Rank results by relevance using embeddings"""
        if not results:
            return []
        
        try:
            # Extract text for embedding
            texts = [f"{result.title} {result.abstract}" for result in results]
            
            # Get embeddings for documents and query
            doc_embeddings = self.embedding_manager.encode_documents(texts, domain)
            query_embedding = self.embedding_manager.encode_query(query, domain)
            
            # Calculate similarity scores
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embeddings
            )[0]
            
            # Assign relevance scores and sort
            for i, result in enumerate(results):
                result.relevance_score = float(similarities[i])
            
            # Sort by relevance score (descending)
            ranked_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
            
            return ranked_results
        
        except Exception as e:
            logger.error(f"Error ranking results: {e}")
            # Return results with default scoring
            return results
    
    async def generate_summary(self, 
                              papers: List[SearchResult], 
                              query: str,
                              max_length: int = None) -> str:
        """Generate a summary of retrieved papers using LLM"""
        if not papers:
            return "No papers found for the given query."
        
        max_length = max_length or config.rag.MAX_SUMMARY_LENGTH
        
        try:
            # Prepare context from top papers
            context_papers = papers[:5]  # Top 5 most relevant
            context = self._prepare_papers_context(context_papers)
            
            # Create LLM prompt for summary
            prompt = f"""Based on the following research papers about "{query}", provide a comprehensive academic summary:

{context}

Please provide a summary that includes:
1. **Main Research Themes**: Key topics and approaches across the papers
2. **Methodological Insights**: Common methodologies and techniques used
3. **Key Findings**: Important results and contributions
4. **Research Trends**: Patterns and directions in the field
5. **Notable Papers**: Highlight 2-3 most significant papers

Keep the summary academic, concise, and focused on the most important insights from these {len(context_papers)} papers.

Summary:"""

            # Generate using LLM
            response = await llm_manager.generate(
                prompt=prompt,
                max_tokens=max_length,
                temperature=config.rag.TEMPERATURE
            )
            
            logger.info(f"Summary generated using {response.model}")
            return response.content
            
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            # Fallback to template-based summary
            return self._generate_template_summary(papers, query)
    
    def _prepare_papers_context(self, papers: List[SearchResult]) -> str:
        """Prepare paper context for LLM prompts"""
        context_parts = []
        
        for i, paper in enumerate(papers, 1):
            # Truncate abstract if too long
            abstract = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
            
            paper_context = f"""Paper {i}:
Title: {paper.title}
Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}
Source: {paper.source}
Abstract: {abstract}"""
            
            if paper.citation_count:
                paper_context += f"\nCitations: {paper.citation_count}"
            
            context_parts.append(paper_context)
        
        return "\n\n".join(context_parts)
    
    def _generate_template_summary(self, papers: List[SearchResult], query: str) -> str:
        """Fallback template-based summary"""
        summary_parts = [
            f"Found {len(papers)} relevant papers for query: '{query}'",
            "",
            "Key findings:"
        ]
        
        for i, paper in enumerate(papers[:5]):  # Top 5 papers
            summary_parts.append(
                f"{i+1}. {paper.title} ({paper.source})"
            )
            if paper.citation_count:
                summary_parts.append(f"   Citations: {paper.citation_count}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    async def generate_hypotheses(self, 
                                 papers: List[SearchResult], 
                                 query: str,
                                 num_hypotheses: int = 3) -> List[str]:
        """Generate research hypotheses based on retrieved papers using LLM"""
        if not papers:
            return []
        
        try:
            # Prepare context from top papers
            context_papers = papers[:5]  # Top 5 most relevant
            context = self._prepare_papers_context(context_papers)
            
            # Create LLM prompt for hypothesis generation
            prompt = f"""Based on the following research papers about "{query}", generate {num_hypotheses} specific and actionable research hypotheses:

{context}

Generate hypotheses that:
1. **Address Research Gaps**: Identify unexplored areas or limitations in current work
2. **Combine Insights**: Integrate methodologies or findings from multiple papers
3. **Extend Applications**: Suggest novel applications or domains for existing techniques
4. **Improve Methods**: Propose enhancements to current approaches

Each hypothesis should be:
- Specific and testable
- Grounded in the literature
- Novel and innovative
- Feasible for future research

Please format as a numbered list of {num_hypotheses} hypotheses:

Research Hypotheses:"""

            # Generate using LLM
            response = await llm_manager.generate(
                prompt=prompt,
                max_tokens=config.rag.MAX_HYPOTHESIS_LENGTH,
                temperature=config.rag.TEMPERATURE + 0.1  # Slightly higher for creativity
            )
            
            logger.info(f"Hypotheses generated using {response.model}")
            
            # Parse hypotheses from response
            hypotheses = self._parse_hypotheses(response.content)
            return hypotheses[:num_hypotheses]
            
        except Exception as e:
            logger.error(f"LLM hypothesis generation failed: {e}")
            # Fallback to template-based hypotheses
            return self._generate_template_hypotheses(papers, query, num_hypotheses)
    
    def _parse_hypotheses(self, content: str) -> List[str]:
        """Parse hypotheses from LLM response"""
        hypotheses = []
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered items (1., 2., etc.) or bullet points
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                # Remove numbering and clean up
                hypothesis = line
                # Remove common prefixes
                for prefix in ['1.', '2.', '3.', '4.', '5.', '•', '-', '*']:
                    if hypothesis.startswith(prefix):
                        hypothesis = hypothesis[len(prefix):].strip()
                        break
                
                if hypothesis and len(hypothesis) > 20:  # Ensure it's substantial
                    hypotheses.append(hypothesis)
        
        # If parsing failed, try splitting by double newlines
        if not hypotheses:
            parts = content.split('\n\n')
            for part in parts:
                part = part.strip()
                if part and len(part) > 20:
                    hypotheses.append(part)
        
        return hypotheses
    
    def _generate_template_hypotheses(self, papers: List[SearchResult], query: str, num_hypotheses: int) -> List[str]:
        """Fallback template-based hypotheses"""
        hypotheses = [
            f"Based on the literature, there may be opportunities to combine methods from {papers[0].title if papers else 'various sources'}",
            f"Future research could explore the intersection of themes found in {len(papers)} retrieved papers",
            f"Novel approaches might emerge from addressing gaps identified in the current literature on '{query}'"
        ]
        
        return hypotheses[:num_hypotheses]
    
    async def process_query(self, 
                           query: str, 
                           sources: List[str] = None,
                           domain: str = 'general',
                           generate_hypotheses: bool = True) -> RAGResponse:
        """
        Process a complete research query through the RAG pipeline
        
        Args:
            query: Research query
            sources: Literature sources to search
            domain: Research domain
            generate_hypotheses: Whether to generate hypotheses
            
        Returns:
            Complete RAG response
        """
        logger.info(f"Processing query: '{query}'")
        start_time = datetime.now()
        
        try:
            # Step 1: Search literature
            papers = await self.search_literature(
                query=query,
                sources=sources,
                domain=domain
            )
            
            # Step 2: Generate summary
            summary = await self.generate_summary(papers, query)
            
            # Step 3: Generate hypotheses (if requested)
            hypotheses = []
            if generate_hypotheses:
                hypotheses = await self.generate_hypotheses(papers, query)
            
            # Step 4: Prepare metadata
            metadata = {
                'num_papers_retrieved': len(papers),
                'sources_used': sources or ['arxiv', 'semantic_scholar'],
                'domain': domain,
                'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                'average_relevance_score': sum(p.relevance_score for p in papers) / len(papers) if papers else 0
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
            
            logger.info(f"Query processed successfully in {metadata['processing_time_seconds']:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Return error response
            return RAGResponse(
                query=query,
                retrieved_papers=[],
                summary=f"Error processing query: {str(e)}",
                hypotheses=[],
                metadata={'error': str(e)},
                timestamp=datetime.now().isoformat()
            )
    
    def add_papers_to_vector_store(self, papers: List[SearchResult], domain: str = 'general'):
        """Add papers to vector store for future retrieval"""
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
                'doi': paper.doi
            }
            metadata.append(paper_metadata)
        
        # Add to vector store
        self.vector_store.add_documents(documents, metadata, domain)
        logger.info(f"Added {len(papers)} papers to vector store")
    
    def search_vector_store(self, 
                           query: str, 
                           k: int = 10, 
                           domain: str = 'general') -> List[SearchResult]:
        """Search the local vector store"""
        results = self.vector_store.search(query, k, domain)
        
        # Convert back to SearchResult format
        search_results = []
        for result in results:
            metadata = result['metadata']
            search_result = SearchResult(
                id=metadata.get('id', ''),
                title=metadata.get('title', ''),
                authors=metadata.get('authors', []),
                abstract=result['document'].split('\n\n', 1)[1] if '\n\n' in result['document'] else '',
                source=metadata.get('source', 'vector_store'),
                url=metadata.get('url', ''),
                published_date=metadata.get('published_date'),
                citation_count=metadata.get('citation_count'),
                venue=metadata.get('venue'),
                categories=metadata.get('categories'),
                doi=metadata.get('doi'),
                relevance_score=result['score']
            )
            search_results.append(search_result)
        
        return search_results
    
    def save_pipeline_state(self, path: str):
        """Save the pipeline state to disk"""
        self.vector_store.save(path)
        logger.info(f"Pipeline state saved to {path}")
    
    def load_pipeline_state(self, path: str):
        """Load the pipeline state from disk"""
        success = self.vector_store.load(path)
        if success:
            logger.info(f"Pipeline state loaded from {path}")
        return success 
