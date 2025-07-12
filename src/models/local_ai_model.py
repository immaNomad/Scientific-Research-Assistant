"""
Local AI Model for Research Paper Analysis
Custom transformer model trained on local research database
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)
import numpy as np
import json
import os
import sys
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import time
from datetime import datetime

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from database.paper_db import PaperDatabase, Paper

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the local AI model"""
    # Model architecture
    model_name: str = "distilbert-base-uncased"  # Base model for fine-tuning
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    max_sequence_length: int = 512
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Data parameters
    train_test_split: float = 0.8
    max_papers_for_training: int = 1000
    
    # Output
    num_labels: int = 5  # For relevance scoring
    output_dir: str = "data/models/local_ai"

class ResearchPaperDataset(Dataset):
    """Dataset for research paper training"""
    
    def __init__(self, papers: List[Paper], tokenizer, max_length: int = 512, mode: str = "train"):
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # Create training data
        self.data = self._prepare_training_data()
        
    def _prepare_training_data(self) -> List[Dict]:
        """Prepare training data from papers"""
        training_data = []
        
        for paper in self.papers:
            # Create different types of training examples
            
            # 1. Title-Abstract relevance
            title_abstract = {
                'input_text': paper.title,
                'target_text': paper.abstract,
                'task': 'summarization',
                'relevance_score': 5.0  # High relevance
            }
            training_data.append(title_abstract)
            
            # 2. Abstract-Keywords relevance (if available)
            if paper.keywords:
                abstract_keywords = {
                    'input_text': paper.abstract,
                    'target_text': ', '.join(paper.keywords),
                    'task': 'keyword_extraction',
                    'relevance_score': 4.0
                }
                training_data.append(abstract_keywords)
            
            # 3. Query-Paper relevance (synthetic queries)
            synthetic_queries = self._generate_synthetic_queries(paper)
            for query in synthetic_queries:
                query_paper = {
                    'input_text': query,
                    'target_text': paper.abstract,
                    'task': 'query_matching',
                    'relevance_score': 3.0
                }
                training_data.append(query_paper)
        
        return training_data
    
    def _generate_synthetic_queries(self, paper: Paper) -> List[str]:
        """Generate synthetic queries from paper content"""
        queries = []
        
        # Extract key terms from title
        title_words = paper.title.lower().split()
        important_words = [w for w in title_words if len(w) > 4 and w not in ['using', 'with', 'based', 'approach']]
        
        if len(important_words) >= 2:
            queries.append(' '.join(important_words[:2]))
            queries.append(' '.join(important_words[-2:]))
        
        # Use categories as queries
        if paper.categories:
            for category in paper.categories[:2]:
                if len(category) > 3:
                    queries.append(category.lower())
        
        # Create domain-specific queries
        domain_terms = {
            'machine learning': ['ml', 'learning', 'algorithm'],
            'deep learning': ['neural', 'deep', 'network'],
            'computer vision': ['vision', 'image', 'visual'],
            'nlp': ['language', 'text', 'linguistic'],
            'ai': ['artificial', 'intelligence', 'intelligent']
        }
        
        abstract_lower = paper.abstract.lower()
        for domain, terms in domain_terms.items():
            if any(term in abstract_lower for term in terms):
                queries.append(domain)
        
        return queries[:3]  # Limit to 3 synthetic queries per paper
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input and target
        input_encoding = self.tokenizer(
            item['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            item['target_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'target_ids': target_encoding['input_ids'].flatten(),
            'target_attention_mask': target_encoding['attention_mask'].flatten(),
            'relevance_score': torch.tensor(item['relevance_score'], dtype=torch.float),
            'task': item['task']
        }

class LocalResearchAI(nn.Module):
    """Custom AI model for research paper analysis"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load base transformer model
        self.base_model = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Custom layers for research tasks
        self.relevance_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.summarization_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.keyword_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        )
        
        # Task classification head
        self.task_classifier = nn.Linear(config.hidden_size, 3)  # 3 tasks
        
    def forward(self, input_ids, attention_mask, task=None):
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT doesn't have pooler_output, so we use last_hidden_state[:, 0, :] (CLS token)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        
        # Task-specific outputs
        relevance_score = self.relevance_head(pooled_output)
        summarization_features = self.summarization_head(pooled_output)
        keyword_features = self.keyword_head(pooled_output)
        task_logits = self.task_classifier(pooled_output)
        
        return {
            'relevance_score': relevance_score,
            'summarization_features': summarization_features,
            'keyword_features': keyword_features,
            'task_logits': task_logits,
            'hidden_states': pooled_output
        }

class LocalAITrainer:
    """Trainer for the local AI model"""
    
    def __init__(self, config: ModelConfig, db_path: str = "data/papers/papers.db"):
        self.config = config
        self.db = PaperDatabase(db_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model = LocalResearchAI(config)
        self.tokenizer = self.model.tokenizer
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"Initialized LocalAITrainer on device: {self.device}")
    
    def prepare_datasets(self) -> Tuple[ResearchPaperDataset, ResearchPaperDataset]:
        """Prepare training and validation datasets"""
        logger.info("Preparing datasets from local paper database...")
        
        # Get papers from database
        papers = self.db.get_all_papers()
        if len(papers) > self.config.max_papers_for_training:
            papers = papers[:self.config.max_papers_for_training]
        
        logger.info(f"Using {len(papers)} papers for training")
        
        # Split into train/test
        split_idx = int(len(papers) * self.config.train_test_split)
        train_papers = papers[:split_idx]
        val_papers = papers[split_idx:]
        
        # Create datasets
        train_dataset = ResearchPaperDataset(train_papers, self.tokenizer, 
                                           self.config.max_sequence_length, mode="train")
        val_dataset = ResearchPaperDataset(val_papers, self.tokenizer, 
                                         self.config.max_sequence_length, mode="val")
        
        logger.info(f"Created train dataset: {len(train_dataset)} examples")
        logger.info(f"Created validation dataset: {len(val_dataset)} examples")
        
        return train_dataset, val_dataset
    
    def train(self):
        """Train the local AI model"""
        logger.info("Starting model training...")
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Calculate loss (multi-task learning)
                relevance_loss = F.mse_loss(
                    outputs['relevance_score'].squeeze(), 
                    batch['relevance_score'] / 5.0  # Normalize to [0,1]
                )
                
                # Task classification loss
                task_labels = torch.zeros(batch['input_ids'].size(0), dtype=torch.long, device=self.device)
                for i, task in enumerate(batch['task']):
                    if task == 'summarization':
                        task_labels[i] = 0
                    elif task == 'keyword_extraction':
                        task_labels[i] = 1
                    else:  # query_matching
                        task_labels[i] = 2
                
                task_loss = F.cross_entropy(outputs['task_logits'], task_labels)
                
                # Combined loss
                total_loss = relevance_loss + 0.5 * task_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                if num_batches % 10 == 0:
                    logger.info(f"Batch {num_batches}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Validation
            val_loss = self._validate(val_loader)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model")
                logger.info("Saved new best model")
        
        logger.info("Training completed!")
        return self.model
    
    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                relevance_loss = F.mse_loss(
                    outputs['relevance_score'].squeeze(), 
                    batch['relevance_score'] / 5.0
                )
                
                total_loss += relevance_loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_model(self, name: str = "local_ai_model"):
        """Save the trained model"""
        model_path = Path(self.config.output_dir) / f"{name}.pt"
        config_path = Path(self.config.output_dir) / f"{name}_config.json"
        tokenizer_path = Path(self.config.output_dir) / "tokenizer"
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump({
                'model_name': self.config.model_name,
                'hidden_size': self.config.hidden_size,
                'num_attention_heads': self.config.num_attention_heads,
                'num_hidden_layers': self.config.num_hidden_layers,
                'max_sequence_length': self.config.max_sequence_length,
                'num_labels': self.config.num_labels
            }, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path

class LocalAIInference:
    """Inference engine for the trained local AI model"""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load model and config
        self._load_model()
        
    def _load_model(self):
        """Load the trained model"""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = LocalResearchAI(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        tokenizer_path = Path(self.model_path).parent / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        logger.info(f"Loaded local AI model from {self.model_path}")
    
    def generate_summary(self, query: str, context: str, max_length: int = 150) -> str:
        """Generate summary using local model"""
        input_text = f"Query: {query} Context: {context}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use summarization features to generate response
        features = outputs['summarization_features']
        
        # Simple template-based generation (can be enhanced with actual generation)
        if 'machine learning' in query.lower():
            return "Based on the machine learning research analyzed, key findings include methodological advances and practical applications in the domain."
        elif 'deep learning' in query.lower():
            return "The deep learning research demonstrates novel architectures and training approaches with applications across multiple domains."
        else:
            return "The research analysis reveals significant contributions to the field with practical implications and future research directions."
    
    def calculate_relevance(self, query: str, paper_text: str) -> float:
        """Calculate relevance score between query and paper"""
        input_text = f"Query: {query} Paper: {paper_text}"
        
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            relevance_score = outputs['relevance_score'].squeeze().item()
        
        return relevance_score * 5.0  # Scale back to [0,5]
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract keywords from text"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            keyword_features = outputs['keyword_features']
        
        # Simple keyword extraction (can be enhanced)
        words = text.lower().split()
        important_words = [w for w in words if len(w) > 4]
        return important_words[:top_k]

# Convenience functions
def train_local_model(db_path: str = "data/papers/papers.db", config: ModelConfig = None) -> str:
    """Train a local AI model on the research database"""
    if config is None:
        config = ModelConfig()
    
    trainer = LocalAITrainer(config, db_path)
    model = trainer.train()
    model_path = trainer.save_model()
    
    return str(model_path)

def load_local_model(model_path: str) -> LocalAIInference:
    """Load a trained local AI model for inference"""
    return LocalAIInference(model_path) 