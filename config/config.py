import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class APIConfig:
    """Configuration for external APIs"""
    # arXiv API settings
    ARXIV_BASE_URL: str = "http://export.arxiv.org/api/query"
    ARXIV_RATE_LIMIT: int = 1  # requests per second
    
    # PubMed API settings
    PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    PUBMED_RATE_LIMIT: int = 3  # requests per second
    PUBMED_API_KEY: Optional[str] = os.getenv("PUBMED_API_KEY")
    
    # Semantic Scholar API settings
    SEMANTIC_SCHOLAR_BASE_URL: str = "https://api.semanticscholar.org/graph/v1"
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    SEMANTIC_SCHOLAR_RATE_LIMIT: int = 100  # requests per second (default)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    # Embedding models
    SCIBERT_MODEL: str = "allenai/scibert_scivocab_uncased"
    BIOBERT_MODEL: str = "dmis-lab/biobert-base-cased-v1.1"
    
    # Generation models
    GENERATION_MODEL: str = "microsoft/DialoGPT-medium"  # Can be upgraded to larger models
    
    # Vector search settings
    EMBEDDING_DIMENSION: int = 768
    VECTOR_INDEX_TYPE: str = "HNSW"  # or "IVF"
    
    # Device settings
    DEVICE: str = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    # Retrieval settings
    MAX_RETRIEVED_DOCS: int = 20
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Generation settings
    MAX_SUMMARY_LENGTH: int = 500
    MAX_HYPOTHESIS_LENGTH: int = 300
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning"""
    # Reward function weights
    RELEVANCE_WEIGHT: float = 0.4  # α
    NOVELTY_WEIGHT: float = 0.3    # β
    CLARITY_WEIGHT: float = 0.2    # γ
    USER_FEEDBACK_WEIGHT: float = 0.1  # δ
    
    # Training settings
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    EPISODES: int = 1000
    
    # Action space
    API_SOURCES: List[str] = None
    QUERY_PARAMS: Dict = None
    
    def __post_init__(self):
        if self.API_SOURCES is None:
            self.API_SOURCES = ["arxiv", "pubmed", "semantic_scholar"]
        if self.QUERY_PARAMS is None:
            self.QUERY_PARAMS = {
                "max_results": [10, 20, 50, 100],
                "sort_by": ["relevance", "date", "citations"],
                "date_range": ["1y", "2y", "5y", "all"]
            }

@dataclass
class WebConfig:
    """Configuration for web application"""
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "5000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

@dataclass
class DatabaseConfig:
    """Configuration for data storage"""
    # Vector database
    VECTOR_DB_PATH: str = "./data/embeddings"
    CACHE_DB_PATH: str = "./data/cache"
    
    # Redis (optional)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # File storage
    PAPER_CACHE_DIR: str = "./data/papers"
    LOG_DIR: str = "./logs"

class Config:
    """Main configuration class"""
    def __init__(self):
        self.api = APIConfig()
        self.model = ModelConfig()
        self.rag = RAGConfig()
        self.rl = RLConfig()
        self.web = WebConfig()
        self.database = DatabaseConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        import os
        directories = [
            self.database.VECTOR_DB_PATH,
            self.database.CACHE_DB_PATH,
            self.database.PAPER_CACHE_DIR,
            self.database.LOG_DIR,
            "./data/models",
            "./data/feedback"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config() 