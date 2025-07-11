"""
Local Paper Database Management System
Handles storage, retrieval, and management of AI/ML research papers
"""

import sqlite3
import os
import json
import hashlib
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Data class for research paper information"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    source: str  # 'arxiv' or 'semantic_scholar'
    url: str
    pdf_path: str  # Path to local PDF file
    folder_path: str  # Path to paper's folder
    published_date: Optional[str] = None
    doi: Optional[str] = None
    categories: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    arxiv_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class PaperDatabase:
    """Local database for managing AI/ML research papers"""
    
    def __init__(self, db_path: str = "data/papers/papers.db", papers_dir: str = "data/papers"):
        self.db_path = db_path
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        
        # Create database if it doesn't exist
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database with proper schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,  -- JSON array
                    abstract TEXT NOT NULL,
                    source TEXT NOT NULL CHECK (source IN ('arxiv', 'semantic_scholar')),
                    url TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    folder_path TEXT NOT NULL,
                    published_date TEXT,
                    doi TEXT,
                    categories TEXT,  -- JSON array
                    keywords TEXT,    -- JSON array
                    citation_count INTEGER,
                    venue TEXT,
                    arxiv_id TEXT,
                    semantic_scholar_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for faster searches
            conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON papers(title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON papers(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_authors ON papers(authors)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_categories ON papers(categories)")
            conn.commit()
            
    def add_paper(self, paper: Paper) -> bool:
        """Add a new paper to the database"""
        try:
            # Set timestamps
            now = datetime.now().isoformat()
            paper.created_at = now
            paper.updated_at = now
            
            # Generate unique ID if not provided
            if not paper.id:
                paper.id = self._generate_paper_id(paper.title, paper.authors)
            
            # Create paper folder
            paper_folder = self.papers_dir / paper.id
            paper_folder.mkdir(exist_ok=True)
            paper.folder_path = str(paper_folder)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO papers (
                        id, title, authors, abstract, source, url, pdf_path, folder_path,
                        published_date, doi, categories, keywords, citation_count, venue,
                        arxiv_id, semantic_scholar_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper.id,
                    paper.title,
                    json.dumps(paper.authors),
                    paper.abstract,
                    paper.source,
                    paper.url,
                    paper.pdf_path,
                    paper.folder_path,
                    paper.published_date,
                    paper.doi,
                    json.dumps(paper.categories) if paper.categories else None,
                    json.dumps(paper.keywords) if paper.keywords else None,
                    paper.citation_count,
                    paper.venue,
                    paper.arxiv_id,
                    paper.semantic_scholar_id,
                    paper.created_at,
                    paper.updated_at
                ))
                conn.commit()
                
            logger.info(f"Added paper: {paper.title}")
            return True
            
        except sqlite3.IntegrityError:
            logger.warning(f"Paper already exists: {paper.title}")
            return False
        except Exception as e:
            logger.error(f"Error adding paper: {e}")
            return False
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Retrieve a paper by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_paper(row)
            return None
    
    def search_papers(self, query: str, source: Optional[str] = None, limit: int = 20) -> List[Paper]:
        """Search papers by title, abstract, or authors"""
        search_query = f"%{query.lower()}%"
        
        base_sql = """
            SELECT * FROM papers 
            WHERE (LOWER(title) LIKE ? OR LOWER(abstract) LIKE ? OR LOWER(authors) LIKE ?)
        """
        params = [search_query, search_query, search_query]
        
        if source:
            base_sql += " AND source = ?"
            params.append(source)
            
        base_sql += " ORDER BY title LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(base_sql, params)
            rows = cursor.fetchall()
            
            return [self._row_to_paper(row) for row in rows]
    
    def get_papers_by_category(self, category: str, limit: int = 20) -> List[Paper]:
        """Get papers by category"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM papers 
                WHERE categories LIKE ? 
                ORDER BY title LIMIT ?
            """, (f"%{category}%", limit))
            rows = cursor.fetchall()
            
            return [self._row_to_paper(row) for row in rows]
    
    def get_all_papers(self, limit: Optional[int] = None) -> List[Paper]:
        """Get all papers from database"""
        sql = "SELECT * FROM papers ORDER BY created_at DESC"
        params = []
        
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            
            return [self._row_to_paper(row) for row in rows]
    
    def update_paper(self, paper: Paper) -> bool:
        """Update an existing paper"""
        try:
            paper.updated_at = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE papers SET
                        title=?, authors=?, abstract=?, source=?, url=?, pdf_path=?,
                        folder_path=?, published_date=?, doi=?, categories=?, keywords=?,
                        citation_count=?, venue=?, arxiv_id=?, semantic_scholar_id=?, updated_at=?
                    WHERE id=?
                """, (
                    paper.title,
                    json.dumps(paper.authors),
                    paper.abstract,
                    paper.source,
                    paper.url,
                    paper.pdf_path,
                    paper.folder_path,
                    paper.published_date,
                    paper.doi,
                    json.dumps(paper.categories) if paper.categories else None,
                    json.dumps(paper.keywords) if paper.keywords else None,
                    paper.citation_count,
                    paper.venue,
                    paper.arxiv_id,
                    paper.semantic_scholar_id,
                    paper.updated_at,
                    paper.id
                ))
                conn.commit()
                
            logger.info(f"Updated paper: {paper.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating paper: {e}")
            return False
    
    def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper and its files"""
        try:
            # Get paper info first
            paper = self.get_paper(paper_id)
            if not paper:
                return False
            
            # Delete files
            if os.path.exists(paper.folder_path):
                shutil.rmtree(paper.folder_path)
            
            # Delete from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
                conn.commit()
                
            logger.info(f"Deleted paper: {paper.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting paper: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT source, COUNT(*) FROM papers GROUP BY source")
            source_counts = dict(cursor.fetchall())
            
            cursor = conn.execute("SELECT COUNT(*) FROM papers WHERE pdf_path IS NOT NULL AND pdf_path != ''")
            papers_with_pdf = cursor.fetchone()[0]
            
            return {
                'total_papers': total_papers,
                'papers_with_pdf': papers_with_pdf,
                'source_distribution': source_counts,
                'database_path': self.db_path,
                'papers_directory': str(self.papers_dir)
            }
    
    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        """Convert database row to Paper object"""
        return Paper(
            id=row['id'],
            title=row['title'],
            authors=json.loads(row['authors']),
            abstract=row['abstract'],
            source=row['source'],
            url=row['url'],
            pdf_path=row['pdf_path'],
            folder_path=row['folder_path'],
            published_date=row['published_date'],
            doi=row['doi'],
            categories=json.loads(row['categories']) if row['categories'] else None,
            keywords=json.loads(row['keywords']) if row['keywords'] else None,
            citation_count=row['citation_count'],
            venue=row['venue'],
            arxiv_id=row['arxiv_id'],
            semantic_scholar_id=row['semantic_scholar_id'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )
    
    def _generate_paper_id(self, title: str, authors: List[str]) -> str:
        """Generate unique paper ID from title and authors"""
        content = f"{title}_{'-'.join(authors[:3])}"  # Use first 3 authors
        return hashlib.md5(content.encode()).hexdigest()[:12] 