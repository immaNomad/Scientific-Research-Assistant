# Fallback to simple implementation if ML libraries not available
try:
    import torch
    import numpy as np
    from typing import List, Union, Optional
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import faiss
    from loguru import logger
    import pickle
    import os
    
    from config.config import config
    ML_AVAILABLE = True
    
    class EmbeddingManager:
        """Manages different embedding models for scientific documents"""
        
        def __init__(self):
            self.device = config.model.DEVICE
            self.models = {}
            self.tokenizers = {}
            self.sentence_transformers = {}
            
            # Initialize models lazily
            self._initialized = False
        
        def _initialize_models(self):
            """Initialize embedding models on first use"""
            if self._initialized:
                return
                
            logger.info("Initializing embedding models...")
            
            try:
                # Load SciBERT for general scientific documents
                logger.info("Loading SciBERT model...")
                self.models['scibert'] = AutoModel.from_pretrained(
                    config.model.SCIBERT_MODEL,
                    trust_remote_code=True
                ).to(self.device)
                self.tokenizers['scibert'] = AutoTokenizer.from_pretrained(
                    config.model.SCIBERT_MODEL,
                    trust_remote_code=True
                )
                
                # Load BioBERT for biomedical documents
                logger.info("Loading BioBERT model...")
                self.models['biobert'] = AutoModel.from_pretrained(
                    config.model.BIOBERT_MODEL,
                    trust_remote_code=True
                ).to(self.device)
                self.tokenizers['biobert'] = AutoTokenizer.from_pretrained(
                    config.model.BIOBERT_MODEL,
                    trust_remote_code=True
                )
                
                # Load sentence transformers for faster inference
                logger.info("Loading sentence transformers...")
                self.sentence_transformers['scibert'] = SentenceTransformer(
                    'allenai/specter2_base'  # Scientific document embeddings
                )
                
                self._initialized = True
                logger.info("All embedding models initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing models: {e}")
                raise
        
        def _get_transformer_embeddings(self, 
                                       texts: List[str], 
                                       model_name: str = 'scibert',
                                       max_length: int = 512) -> np.ndarray:
            """Get embeddings using transformer models"""
            if not self._initialized:
                self._initialize_models()
            
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            embeddings = []
            
            for text in texts:
                # Tokenize and encode
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use CLS token embedding or mean pooling
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embedding = outputs.pooler_output.squeeze()
                    else:
                        # Mean pooling
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    
                    embeddings.append(embedding.cpu().numpy())
            
            return np.array(embeddings)
        
        def _get_sentence_transformer_embeddings(self, 
                                               texts: List[str], 
                                               model_name: str = 'scibert') -> np.ndarray:
            """Get embeddings using sentence transformers (faster)"""
            if not self._initialized:
                self._initialize_models()
            
            model = self.sentence_transformers[model_name]
            return model.encode(texts, convert_to_numpy=True)
        
        def encode_documents(self, 
                            documents: List[str], 
                            domain: str = 'general',
                            use_fast: bool = True) -> np.ndarray:
            """
            Encode documents into embeddings
            
            Args:
                documents: List of document texts
                domain: 'general', 'biomedical', or 'cs' for domain-specific models
                use_fast: Whether to use faster sentence transformers
                
            Returns:
                Array of embeddings
            """
            if not documents:
                return np.array([])
            
            # Choose appropriate model based on domain
            if domain == 'biomedical':
                model_name = 'biobert'
            else:
                model_name = 'scibert'
            
            logger.info(f"Encoding {len(documents)} documents with {model_name}")
            
            if use_fast and model_name in self.sentence_transformers:
                return self._get_sentence_transformer_embeddings(documents, model_name)
            else:
                return self._get_transformer_embeddings(documents, model_name)
        
        def encode_query(self, 
                        query: str, 
                        domain: str = 'general',
                        use_fast: bool = True) -> np.ndarray:
            """
            Encode a single query into embedding
            
            Args:
                query: Query text
                domain: Domain for model selection
                use_fast: Whether to use faster sentence transformers
                
            Returns:
                Query embedding
            """
            return self.encode_documents([query], domain, use_fast)[0]

    class VectorStore:
        """FAISS-based vector store for semantic search"""
        
        def __init__(self, dimension: int = None):
            self.dimension = dimension or config.model.EMBEDDING_DIMENSION
            self.index = None
            self.documents = []
            self.metadata = []
            self.embedding_manager = EmbeddingManager()
            
        def _create_index(self, embeddings: np.ndarray):
            """Create FAISS index from embeddings"""
            if config.model.VECTOR_INDEX_TYPE == "HNSW":
                # HNSW index for better quality
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 40
                self.index.hnsw.efSearch = 16
            else:
                # IVF index for faster search
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                
            self.index.add(embeddings)
            
            if hasattr(self.index, 'train'):
                self.index.train(embeddings)
        
        def add_documents(self, 
                         documents: List[str], 
                         metadata: List[dict] = None,
                         domain: str = 'general'):
            """
            Add documents to the vector store
            
            Args:
                documents: List of document texts
                metadata: List of metadata dictionaries
                domain: Domain for embedding model selection
            """
            if not documents:
                return
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Generate embeddings
            embeddings = self.embedding_manager.encode_documents(documents, domain)
            
            # Create or update index
            if self.index is None:
                self._create_index(embeddings)
            else:
                self.index.add(embeddings)
            
            # Store documents and metadata
            self.documents.extend(documents)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * len(documents))
        
        def search(self, 
                   query: str, 
                   k: int = 10,
                   domain: str = 'general',
                   score_threshold: float = 0.0) -> List[dict]:
            """
            Search for similar documents
            
            Args:
                query: Query text
                k: Number of results to return
                domain: Domain for embedding model selection
                score_threshold: Minimum similarity score
                
            Returns:
                List of search results with documents and scores
            """
            if self.index is None or len(self.documents) == 0:
                return []
            
            # Encode query
            query_embedding = self.embedding_manager.encode_query(query, domain)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and score >= score_threshold:
                    results.append({
                        'document': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            
            return results
        
        def save(self, path: str):
            """Save vector store to disk"""
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, f"{path}.index")
            
            # Save documents and metadata
            data = {
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension
            }
            
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Vector store saved to {path}")
        
        def load(self, path: str):
            """Load vector store from disk"""
            try:
                # Load FAISS index
                if os.path.exists(f"{path}.index"):
                    self.index = faiss.read_index(f"{path}.index")
                
                # Load documents and metadata
                if os.path.exists(f"{path}.pkl"):
                    with open(f"{path}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                    self.dimension = data['dimension']
                    
                    logger.info(f"Vector store loaded from {path}")
                    return True
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                return False
            
            return False

    class DocumentChunker:
        """Utility class for chunking documents for better retrieval"""
        
        def __init__(self, 
                     chunk_size: int = None, 
                     chunk_overlap: int = None):
            self.chunk_size = chunk_size or config.rag.CHUNK_SIZE
            self.chunk_overlap = chunk_overlap or config.rag.CHUNK_OVERLAP
        
        def chunk_text(self, text: str, metadata: dict = None) -> List[dict]:
            """
            Chunk text into smaller pieces with overlap
            
            Args:
                text: Text to chunk
                metadata: Metadata to attach to each chunk
                
            Returns:
                List of chunk dictionaries
            """
            if not text:
                return []
            
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    'chunk_index': len(chunks),
                    'chunk_start': i,
                    'chunk_end': i + len(chunk_words)
                })
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
            
            return chunks
        
        def chunk_documents(self, documents: List[dict]) -> List[dict]:
            """
            Chunk multiple documents
            
            Args:
                documents: List of document dictionaries with 'text' and 'metadata'
                
            Returns:
                List of chunk dictionaries
            """
            all_chunks = []
            
            for doc in documents:
                chunks = self.chunk_text(doc['text'], doc.get('metadata', {}))
                all_chunks.extend(chunks)
            
            return all_chunks

except ImportError as e:
    from loguru import logger
    logger.info(f"ML libraries not available ({e}), using simple implementation")
    ML_AVAILABLE = False
    
    # Import simple fallback implementations
    from .simple_embeddings import SimpleEmbeddingManager as EmbeddingManager
    from .simple_embeddings import SimpleVectorStore as VectorStore  
    from .simple_embeddings import SimpleDocumentChunker as DocumentChunker 