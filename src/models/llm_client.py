"""
LLM Client for RAG Pipeline
Supports multiple LLM providers: OpenAI, local models, and fallbacks
"""

import os
import asyncio
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers torch")

from config.config import config

@dataclass
class LLMResponse:
    """Response from LLM generation"""
    content: str
    model: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> LLMResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if self.api_key and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and self.api_key is not None
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> LLMResponse:
        """Generate text using OpenAI API"""
        if not self.is_available():
            raise Exception("OpenAI client not available")
        
        try:
            import time
            start_time = time.time()
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            # Rough cost estimation (GPT-3.5-turbo pricing)
            cost = (tokens_used * 0.002 / 1000) if tokens_used else None
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

class LocalLLMClient(BaseLLMClient):
    """Local transformer model client"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the local model"""
        try:
            logger.info(f"Loading local model: {self.model_name}")
            
            # Use text generation pipeline for simplicity
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() and config.model.DEVICE == "cuda" else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                do_sample=True,
                pad_token_id=50256  # GPT-2 style padding
            )
            
            logger.info(f"Local model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            self.pipeline = None
    
    def is_available(self) -> bool:
        return TRANSFORMERS_AVAILABLE and self.pipeline is not None
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> LLMResponse:
        """Generate text using local model"""
        if not self.is_available():
            raise Exception("Local LLM not available")
        
        try:
            import time
            start_time = time.time()
            
            # Generate text
            result = await asyncio.to_thread(
                self.pipeline,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                return_full_text=False
            )
            
            latency = time.time() - start_time
            content = result[0]['generated_text'].strip()
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                tokens_used=len(content.split()),  # Rough estimate
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"Local LLM generation error: {e}")
            raise

class FallbackLLMClient(BaseLLMClient):
    """Fallback client using templates (current implementation)"""
    
    def is_available(self) -> bool:
        return True
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> LLMResponse:
        """Generate using template-based approach"""
        import time
        start_time = time.time()
        
        # Simple template-based generation
        if "summary" in prompt.lower():
            content = self._generate_template_summary(prompt)
        elif "hypothesis" in prompt.lower() or "hypotheses" in prompt.lower():
            content = self._generate_template_hypotheses(prompt)
        else:
            content = "Based on the provided context, here are the key insights from the literature."
        
        latency = time.time() - start_time
        
        return LLMResponse(
            content=content,
            model="template-fallback",
            latency=latency
        )
    
    def _generate_template_summary(self, prompt: str) -> str:
        """Generate template-based summary"""
        return """Based on the retrieved literature, several key themes emerge:

1. **Methodological Approaches**: The papers demonstrate diverse methodological frameworks
2. **Current Trends**: Recent research shows increasing focus on practical applications
3. **Research Gaps**: Opportunities exist for cross-disciplinary collaboration
4. **Future Directions**: The field is moving toward more integrated approaches

The literature provides a solid foundation for understanding current research directions."""
    
    def _generate_template_hypotheses(self, prompt: str) -> str:
        """Generate template-based hypotheses"""
        return """1. Integration of multiple methodologies from the reviewed papers could yield novel approaches to the research problem.

2. Cross-domain applications of the identified techniques may reveal new insights and improve performance in related fields.

3. Addressing the research gaps highlighted in the literature could lead to significant theoretical and practical advances."""

class LLMManager:
    """Manages multiple LLM clients with fallback strategy"""
    
    def __init__(self):
        self.clients = []
        self.current_client = None
        
        # Initialize clients in order of preference
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients in order of preference"""
        
        # 1. Try OpenAI first (best quality)
        openai_client = OpenAIClient()
        if openai_client.is_available():
            self.clients.append(("openai", openai_client))
            logger.info("✅ OpenAI client available")
        
        # 2. Try local model (good quality, privacy)
        local_client = LocalLLMClient()
        if local_client.is_available():
            self.clients.append(("local", local_client))
            logger.info("✅ Local LLM client available")
        
        # 3. Always have fallback (reliable)
        fallback_client = FallbackLLMClient()
        self.clients.append(("fallback", fallback_client))
        logger.info("✅ Fallback client available")
        
        # Set current client to the best available
        if self.clients:
            self.current_client = self.clients[0][1]
            logger.info(f"Using LLM client: {self.clients[0][0]}")
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> LLMResponse:
        """Generate text using the best available client"""
        
        for client_name, client in self.clients:
            try:
                logger.debug(f"Attempting generation with {client_name}")
                response = await client.generate(prompt, max_tokens, temperature)
                logger.info(f"✅ Generation successful with {client_name}")
                return response
                
            except Exception as e:
                logger.warning(f"❌ {client_name} failed: {e}")
                continue
        
        raise Exception("All LLM clients failed")
    
    def get_available_clients(self) -> List[str]:
        """Get list of available client names"""
        return [name for name, client in self.clients if client.is_available()]
    
    def switch_client(self, client_name: str) -> bool:
        """Switch to a specific client"""
        for name, client in self.clients:
            if name == client_name and client.is_available():
                self.current_client = client
                logger.info(f"Switched to {client_name} client")
                return True
        return False

# Global LLM manager instance
llm_manager = LLMManager() 