import asyncio
import aiohttp
import time
import xmltodict
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from config.config import config

@dataclass
class ArxivPaper:
    """Data class for arXiv paper information"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    updated: str
    categories: List[str]
    url: str
    pdf_url: str
    doi: Optional[str] = None

class ArxivClient:
    """Client for arXiv API with rate limiting and async support"""
    
    def __init__(self):
        self.base_url = config.api.ARXIV_BASE_URL
        self.rate_limit = config.api.ARXIV_RATE_LIMIT
        self.last_request_time = 0
        
    async def _rate_limit_wait(self):
        """Ensure rate limiting compliance"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _build_query(self, query: str, max_results: int = 20, sort_by: str = "relevance", 
                     sort_order: str = "descending", start: int = 0) -> str:
        """Build arXiv API query URL"""
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.base_url}?{query_string}"
    
    def _parse_entry(self, entry: Dict) -> ArxivPaper:
        """Parse a single arXiv entry from XML response"""
        try:
            authors = entry.get("author", [])
            if isinstance(authors, dict):
                authors = [authors]
            author_names = [author.get("name", "") for author in authors]
            
            categories = entry.get("category", [])
            if isinstance(categories, dict):
                categories = [categories]
            category_list = [cat.get("@term", "") for cat in categories]
            
            doi = None
            if "arxiv:doi" in entry:
                doi = entry["arxiv:doi"]["#text"]
            
            return ArxivPaper(
                id=entry["id"].split("/")[-1],
                title=entry["title"].strip(),
                authors=author_names,
                abstract=entry["summary"].strip(),
                published=entry["published"],
                updated=entry["updated"],
                categories=category_list,
                url=entry["id"],
                pdf_url=entry["id"].replace("abs", "pdf") + ".pdf",
                doi=doi
            )
        except Exception as e:
            logger.error(f"Error parsing arXiv entry: {e}")
            return None
    
    async def search(self, query: str, max_results: int = 20, sort_by: str = "relevance") -> List[ArxivPaper]:
        """Search arXiv for papers matching the query"""
        await self._rate_limit_wait()
        
        url = self._build_query(query, max_results, sort_by)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        parsed = xmltodict.parse(xml_content)
                        
                        feed = parsed.get("feed", {})
                        entries = feed.get("entry", [])
                        
                        if isinstance(entries, dict):
                            entries = [entries]
                        
                        papers = []
                        for entry in entries:
                            paper = self._parse_entry(entry)
                            if paper:
                                papers.append(paper)
                        
                        logger.info(f"Retrieved {len(papers)} papers from arXiv")
                        return papers
                    else:
                        logger.error(f"arXiv API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error querying arXiv: {e}")
            return []
    
    async def search_by_categories(self, categories: List[str], keywords: str = "", max_results: int = 20) -> List[ArxivPaper]:
        """Search arXiv by specific categories"""
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        
        if keywords:
            query = f"({cat_query}) AND ({keywords})"
        else:
            query = cat_query
            
        return await self.search(query, max_results)
    
    async def get_recent_papers(self, category: str = "cs.AI", days: int = 7, max_results: int = 20) -> List[ArxivPaper]:
        """Get recent papers from a specific category"""
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        date_str = start_date.strftime("%Y%m%d")
        query = f"cat:{category} AND submittedDate:[{date_str}* TO *]"
        
        return await self.search(query, max_results, sort_by="submittedDate") 