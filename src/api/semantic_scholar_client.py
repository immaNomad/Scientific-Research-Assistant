import asyncio
import aiohttp
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from config.config import config

@dataclass
class SemanticScholarPaper:
    """Data class for Semantic Scholar paper information"""
    id: str
    title: str
    authors: List[Dict[str, str]]  # {"name": str, "authorId": str}
    abstract: str
    year: Optional[int]
    venue: Optional[str]
    url: str
    citation_count: int
    reference_count: int
    influential_citation_count: int
    fields_of_study: List[str]
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pubmed_id: Optional[str] = None
    citations: Optional[List[str]] = None  # List of citing paper IDs
    references: Optional[List[str]] = None  # List of referenced paper IDs

class SemanticScholarClient:
    """Client for Semantic Scholar API with advanced citation analysis"""
    
    def __init__(self):
        self.base_url = config.api.SEMANTIC_SCHOLAR_BASE_URL
        self.api_key = config.api.SEMANTIC_SCHOLAR_API_KEY
        self.rate_limit = config.api.SEMANTIC_SCHOLAR_RATE_LIMIT
        self.last_request_time = 0
        
        # Headers for requests
        self.headers = {
            "User-Agent": "RAG-Research-Assistant/1.0",
        }
        if self.api_key:
            self.headers["x-api-key"] = self.api_key
    
    async def _rate_limit_wait(self):
        """Ensure rate limiting compliance"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _parse_paper(self, paper_data: Dict) -> SemanticScholarPaper:
        """Parse paper data from Semantic Scholar API response"""
        try:
            # Extract author information
            authors = []
            for author in paper_data.get("authors", []):
                authors.append({
                    "name": author.get("name", ""),
                    "authorId": author.get("authorId", "")
                })
            
            # Extract external IDs
            external_ids = paper_data.get("externalIds", {})
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")
            pubmed_id = external_ids.get("PubMed")
            
            # Extract fields of study
            fields = paper_data.get("fieldsOfStudy", [])
            if not isinstance(fields, list):
                fields = []
            
            return SemanticScholarPaper(
                id=paper_data.get("paperId", ""),
                title=paper_data.get("title", ""),
                authors=authors,
                abstract=paper_data.get("abstract", ""),
                year=paper_data.get("year"),
                venue=paper_data.get("venue"),
                url=paper_data.get("url", ""),
                citation_count=paper_data.get("citationCount", 0),
                reference_count=paper_data.get("referenceCount", 0),
                influential_citation_count=paper_data.get("influentialCitationCount", 0),
                fields_of_study=fields,
                doi=doi,
                arxiv_id=arxiv_id,
                pubmed_id=pubmed_id
            )
        except Exception as e:
            logger.error(f"Error parsing Semantic Scholar paper: {e}")
            return None
    
    async def search(self, 
                     query: str, 
                     max_results: int = 20,
                     fields: List[str] = None,
                     year_range: tuple = None) -> List[SemanticScholarPaper]:
        """
        Search Semantic Scholar for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            fields: List of fields to return
            year_range: Tuple of (start_year, end_year)
            
        Returns:
            List of SemanticScholarPaper objects
        """
        await self._rate_limit_wait()
        
        # Default fields to retrieve
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "authors", "year", "venue",
                "url", "citationCount", "referenceCount", "influentialCitationCount",
                "fieldsOfStudy", "externalIds"
            ]
        
        params = {
            "query": query,
            "limit": min(max_results, 100),  # API limit
            "fields": ",".join(fields)
        }
        
        # Add year filter if specified
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        url = f"{self.base_url}/paper/search"
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        papers = []
                        
                        for paper_data in data.get("data", []):
                            paper = self._parse_paper(paper_data)
                            if paper:
                                papers.append(paper)
                        
                        logger.info(f"Retrieved {len(papers)} papers from Semantic Scholar")
                        return papers
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error querying Semantic Scholar: {e}")
            return []
    
    async def get_paper_details(self, paper_id: str) -> Optional[SemanticScholarPaper]:
        """
        Get detailed information about a specific paper
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            SemanticScholarPaper object or None
        """
        await self._rate_limit_wait()
        
        fields = [
            "paperId", "title", "abstract", "authors", "year", "venue",
            "url", "citationCount", "referenceCount", "influentialCitationCount",
            "fieldsOfStudy", "externalIds", "citations", "references"
        ]
        
        params = {"fields": ",".join(fields)}
        url = f"{self.base_url}/paper/{paper_id}"
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        paper = self._parse_paper(data)
                        
                        # Add citation and reference data
                        if paper:
                            paper.citations = [c["paperId"] for c in data.get("citations", [])]
                            paper.references = [r["paperId"] for r in data.get("references", [])]
                        
                        return paper
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting paper details: {e}")
            return None
    
    async def get_citations(self, paper_id: str, max_results: int = 50) -> List[SemanticScholarPaper]:
        """
        Get papers that cite the given paper
        
        Args:
            paper_id: Semantic Scholar paper ID
            max_results: Maximum number of citing papers to return
            
        Returns:
            List of citing papers
        """
        await self._rate_limit_wait()
        
        fields = [
            "paperId", "title", "abstract", "authors", "year", "venue",
            "citationCount", "influentialCitationCount", "fieldsOfStudy"
        ]
        
        params = {
            "fields": ",".join(fields),
            "limit": min(max_results, 1000)
        }
        
        url = f"{self.base_url}/paper/{paper_id}/citations"
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        papers = []
                        
                        for citation in data.get("data", []):
                            citing_paper = citation.get("citingPaper", {})
                            paper = self._parse_paper(citing_paper)
                            if paper:
                                papers.append(paper)
                        
                        logger.info(f"Retrieved {len(papers)} citing papers")
                        return papers
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error getting citations: {e}")
            return []
    
    async def get_author_papers(self, author_id: str, max_results: int = 50) -> List[SemanticScholarPaper]:
        """
        Get papers by a specific author
        
        Args:
            author_id: Semantic Scholar author ID
            max_results: Maximum number of papers to return
            
        Returns:
            List of author's papers
        """
        await self._rate_limit_wait()
        
        fields = [
            "paperId", "title", "abstract", "year", "venue",
            "citationCount", "influentialCitationCount", "fieldsOfStudy"
        ]
        
        params = {
            "fields": ",".join(fields),
            "limit": min(max_results, 1000)
        }
        
        url = f"{self.base_url}/author/{author_id}/papers"
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        papers = []
                        
                        for paper_data in data.get("data", []):
                            paper = self._parse_paper(paper_data)
                            if paper:
                                papers.append(paper)
                        
                        logger.info(f"Retrieved {len(papers)} papers by author")
                        return papers
                    else:
                        logger.error(f"Semantic Scholar API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error getting author papers: {e}")
            return []
    
    async def get_trending_papers(self, field: str = "Computer Science", days: int = 7) -> List[SemanticScholarPaper]:
        """
        Get trending papers in a specific field
        
        Args:
            field: Field of study
            days: Number of days to look back
            
        Returns:
            List of trending papers
        """
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Search for recent papers with high citation velocity
        query = f"fieldsOfStudy:{field}"
        year_range = (start_date.year, end_date.year)
        
        papers = await self.search(query, max_results=100, year_range=year_range)
        
        # Sort by influential citation count and recency
        trending = sorted(papers, 
                         key=lambda p: (p.influential_citation_count, p.year or 0), 
                         reverse=True)
        
        return trending[:20] 