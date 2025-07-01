import asyncio
import aiohttp
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger
import xml.etree.ElementTree as ET

from config.config import config

@dataclass
class PubMedPaper:
    """Data class for PubMed paper information"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    journal: str
    published_date: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    keywords: Optional[List[str]] = None
    mesh_terms: Optional[List[str]] = None

class PubMedClient:
    """Client for PubMed API (E-utilities) with rate limiting"""
    
    def __init__(self):
        self.base_url = config.api.PUBMED_BASE_URL
        self.api_key = config.api.PUBMED_API_KEY
        self.rate_limit = config.api.PUBMED_RATE_LIMIT
        self.last_request_time = 0
        
        self.headers = {
            "User-Agent": "RAG-Research-Assistant/1.0 (mailto:your-email@example.com)",
        }
    
    async def _rate_limit_wait(self):
        """Ensure rate limiting compliance"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def _esearch(self, query: str, max_results: int = 20) -> List[str]:
        """Search PubMed and get PMIDs"""
        await self._rate_limit_wait()
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "sort": "relevance"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.base_url}/esearch.fcgi"
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        root = ET.fromstring(xml_content)
                        
                        pmids = []
                        id_list = root.find(".//IdList")
                        if id_list is not None:
                            for id_elem in id_list.findall("Id"):
                                pmids.append(id_elem.text)
                        
                        return pmids
                    else:
                        logger.error(f"PubMed eSearch error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error in PubMed eSearch: {e}")
            return []
    
    async def _efetch(self, pmids: List[str]) -> List[PubMedPaper]:
        """Fetch full records for given PMIDs"""
        if not pmids:
            return []
        
        await self._rate_limit_wait()
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.base_url}/efetch.fcgi"
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        return self._parse_efetch_response(xml_content)
                    else:
                        logger.error(f"PubMed eFetch error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error in PubMed eFetch: {e}")
            return []
    
    def _parse_efetch_response(self, xml_content: str) -> List[PubMedPaper]:
        """Parse eFetch XML response"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
        
        except Exception as e:
            logger.error(f"Error parsing PubMed XML: {e}")
        
        return papers
    
    def _parse_article(self, article) -> Optional[PubMedPaper]:
        """Parse a single PubMed article"""
        try:
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            medline_citation = article.find(".//MedlineCitation")
            if medline_citation is None:
                return None
            
            citation_article = medline_citation.find(".//Article")
            if citation_article is None:
                return None
            
            title_elem = citation_article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            abstract_parts = []
            abstract_elem = citation_article.find(".//Abstract")
            if abstract_elem is not None:
                for abstract_text in abstract_elem.findall(".//AbstractText"):
                    if abstract_text.text:
                        abstract_parts.append(abstract_text.text)
            abstract = " ".join(abstract_parts)
            
            authors = []
            author_list = citation_article.find(".//AuthorList")
            if author_list is not None:
                for author in author_list.findall(".//Author"):
                    last_name = author.find(".//LastName")
                    first_name = author.find(".//ForeName")
                    
                    if last_name is not None:
                        name = last_name.text
                        if first_name is not None:
                            name = f"{first_name.text} {name}"
                        authors.append(name)
            
            journal_elem = citation_article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            pub_date = ""
            date_elem = citation_article.find(".//PubDate")
            if date_elem is not None:
                year = date_elem.find(".//Year")
                month = date_elem.find(".//Month")
                day = date_elem.find(".//Day")
                
                if year is not None:
                    pub_date = year.text
                    if month is not None:
                        pub_date += f"-{month.text}"
                        if day is not None:
                            pub_date += f"-{day.text}"
            
            doi = None
            article_ids = article.findall(".//ArticleId")
            for article_id in article_ids:
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            mesh_terms = []
            mesh_list = medline_citation.find(".//MeshHeadingList")
            if mesh_list is not None:
                for mesh_heading in mesh_list.findall(".//MeshHeading"):
                    descriptor = mesh_heading.find(".//DescriptorName")
                    if descriptor is not None:
                        mesh_terms.append(descriptor.text)
            
            keywords = []
            keyword_list = medline_citation.find(".//KeywordList")
            if keyword_list is not None:
                for keyword in keyword_list.findall(".//Keyword"):
                    if keyword.text:
                        keywords.append(keyword.text)
            
            return PubMedPaper(
                id=pmid,
                title=title,
                authors=authors,
                abstract=abstract,
                journal=journal,
                published_date=pub_date,
                doi=doi,
                pmid=pmid,
                keywords=keywords,
                mesh_terms=mesh_terms
            )
        
        except Exception as e:
            logger.error(f"Error parsing PubMed article: {e}")
            return None
    
    async def search(self, query: str, max_results: int = 20, sort_by: str = "relevance") -> List[PubMedPaper]:
        """Search PubMed for papers"""
        logger.info(f"Searching PubMed for: {query}")
        
        pmids = await self._esearch(query, max_results)
        
        if not pmids:
            logger.info("No PMIDs found")
            return []
        
        papers = await self._efetch(pmids)
        
        logger.info(f"Retrieved {len(papers)} papers from PubMed")
        return papers
    
    async def search_by_mesh(self, mesh_terms: List[str], max_results: int = 20) -> List[PubMedPaper]:
        """Search by MeSH terms"""
        mesh_query = " AND ".join(f'"{term}"[MeSH Terms]' for term in mesh_terms)
        return await self.search(mesh_query, max_results)
    
    async def get_recent_papers(self, query: str = "", days: int = 7, max_results: int = 20) -> List[PubMedPaper]:
        """Get recent papers from PubMed"""
        from datetime import datetime, timedelta
        
        date_query = f'"last {days} days"[PDat]'
        
        if query:
            full_query = f"({query}) AND {date_query}"
        else:
            full_query = date_query
        
        return await self.search(full_query, max_results, sort_by="pub_date")
    
    async def get_related_papers(self, pmid: str, max_results: int = 20) -> List[PubMedPaper]:
        """Get papers related to a specific PMID"""
        await self._rate_limit_wait()
        
        params = {
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": pmid,
            "retmax": max_results
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.base_url}/elink.fcgi"
        
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        root = ET.fromstring(xml_content)
                        
                        pmids = []
                        for link_set in root.findall(".//LinkSet"):
                            for link_set_db in link_set.findall(".//LinkSetDb"):
                                for link in link_set_db.findall(".//Link/Id"):
                                    pmids.append(link.text)
                        
                        if pmid in pmids:
                            pmids.remove(pmid)
                        
                        return await self._efetch(pmids[:max_results])
                    else:
                        logger.error(f"PubMed eLink error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error getting related papers: {e}")
            return [] 