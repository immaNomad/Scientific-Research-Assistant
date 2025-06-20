"""
LLM Client for RAG Pipeline
Supports Google Gemini, local models, and fallbacks
"""

import os
import asyncio
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")

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

class GeminiClient(BaseLLMClient):
    """Google Gemini client"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        self.model = model
        
        if self.api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        return GEMINI_AVAILABLE and self.api_key is not None
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> LLMResponse:
        """Generate text using Google Gemini API"""
        if not self.is_available():
            raise Exception("Gemini client not available")
        
        try:
            import time
            start_time = time.time()
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            
            # Generate response
            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            latency = time.time() - start_time
            
            # Safely extract content from response
            content = None
            
            # Try multiple ways to extract content from Gemini response
            try:
                if hasattr(response, 'text') and response.text:
                    content = response.text
                elif hasattr(response, 'candidates') and response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts and len(candidate.content.parts) > 0:
                            content = candidate.content.parts[0].text
                        elif hasattr(candidate.content, 'text'):
                            content = candidate.content.text
                elif hasattr(response, 'parts') and response.parts and len(response.parts) > 0:
                    content = response.parts[0].text
                
                # If still no content, try different structure
                if not content and hasattr(response, '_result') and response._result:
                    if hasattr(response._result, 'candidates') and response._result.candidates:
                        content = response._result.candidates[0].content.parts[0].text
                        
            except (AttributeError, IndexError, TypeError) as extraction_error:
                logger.warning(f"Content extraction error: {extraction_error}")
                content = None
            
            # Validate content
            if not content or not content.strip():
                logger.error(f"Gemini response structure: {type(response)} - {dir(response) if hasattr(response, '__dict__') else 'No attributes'}")
                raise Exception("Empty or invalid content from Gemini response")
            
            # Estimate tokens (rough approximation)
            tokens_used = len(content.split()) + len(prompt.split())
            
            # Gemini is free for reasonable usage
            cost = 0.0
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                cost=cost,
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
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
    """Fallback client using templates"""
    
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
        
        # 1. Try Gemini first (Google's LLM API - primary choice)
        gemini_client = GeminiClient()
        if gemini_client.is_available():
            self.clients.append(("gemini", gemini_client))
            logger.info("✅ Gemini client available")
        
        # 2. Try local model (privacy option)
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
        
        last_error = None
        for client_name, client in self.clients:
            try:
                logger.debug(f"Attempting generation with {client_name}")
                response = await client.generate(prompt, max_tokens, temperature)
                
                # Validate response
                if response and hasattr(response, 'content') and response.content and response.content.strip():
                    logger.info(f"✅ Generation successful with {client_name}")
                    return response
                else:
                    logger.warning(f"❌ {client_name} returned empty response")
                    continue
                
            except Exception as e:
                logger.warning(f"❌ {client_name} failed: {e}")
                last_error = e
                continue
        
        # If all clients failed, return a basic fallback response
        logger.error("🔄 All LLM clients failed, using emergency fallback")
        return LLMResponse(
            content="Based on the research context provided, this analysis covers the current state of research in this domain. The papers identified represent recent work and established findings in the field.",
            model="emergency-fallback",
            tokens_used=20,
            latency=0.0
        )
    
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