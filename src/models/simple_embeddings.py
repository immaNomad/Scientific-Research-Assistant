"""
Simple embedding functionality without heavy ML dependencies
For basic functionality when full ML stack is not available
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

@dataclass 
class SimpleEmbeddingManager:
    """Simple embedding manager using TF-IDF"""
    
    def __init__(self):
        self.vectorizers = {}
        self.document_embeddings = {}
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def encode_documents(self, texts: List[str], domain: str = 'general') -> np.ndarray:
        """Encode documents using TF-IDF"""
        if domain not in self.vectorizers:
            self.vectorizers[domain] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Fit or transform based on whether vectorizer has been fitted
        try:
            embeddings = self.vectorizers[domain].transform(processed_texts)
        except:
            # First time - fit and transform
            embeddings = self.vectorizers[domain].fit_transform(processed_texts)
        
        return embeddings.toarray()
    
    def encode_query(self, query: str, domain: str = 'general') -> np.ndarray:
        """Encode query using existing vectorizer"""
        if domain not in self.vectorizers:
            # If no vectorizer exists, create a basic one
            self.vectorizers[domain] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            # For a new vectorizer, return zeros (no training data)
            return np.zeros((1, 1000))
        
        processed_query = self._preprocess_text(query)
        
        try:
            query_embedding = self.vectorizers[domain].transform([processed_query])
            return query_embedding.toarray()
        except:
            # If transform fails, return zeros
            return np.zeros((1, 1000))

@dataclass
class SimpleVectorStore:
    """Simple vector store implementation"""
    
    def __init__(self):
        self.documents = {}
        self.embeddings = {}
        self.metadata = {}
    
    def add_documents(self, 
                     documents: List[str], 
                     embeddings: np.ndarray,
                     metadata: List[Dict] = None,
                     domain: str = 'general'):
        """Add documents to the store"""
        if domain not in self.documents:
            self.documents[domain] = []
            self.embeddings[domain] = []
            self.metadata[domain] = []
        
        self.documents[domain].extend(documents)
        self.embeddings[domain].append(embeddings)
        
        if metadata:
            self.metadata[domain].extend(metadata)
        else:
            self.metadata[domain].extend([{}] * len(documents))
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 10, 
               domain: str = 'general') -> List[Dict]:
        """Search for similar documents"""
        if domain not in self.embeddings or not self.embeddings[domain]:
            return []
        
        # Combine all embeddings for this domain
        all_embeddings = np.vstack(self.embeddings[domain])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, all_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents[domain]):
                results.append({
                    'document': self.documents[domain][idx],
                    'metadata': self.metadata[domain][idx],
                    'similarity': similarities[idx]
                })
        
        return results

@dataclass 
class SimpleDocumentChunker:
    """Simple document chunking"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
            # Break if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks if chunks else [text]
    
    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunk multiple documents"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
        return all_chunks

# For backwards compatibility, provide aliases
EmbeddingManager = SimpleEmbeddingManager
VectorStore = SimpleVectorStore
DocumentChunker = SimpleDocumentChunker 