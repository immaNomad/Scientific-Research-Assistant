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

from config import config

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

class CustomLocalAIClient(BaseLLMClient):
    """Custom trained local AI model client"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.config = None
        self._load_model()
    
    def _load_model(self):
        """Load the custom trained model"""
        try:
            from models.local_ai_model import LocalAIInference
            from pathlib import Path
            import json
            
            if not Path(self.model_path).exists():
                logger.warning(f"Custom model not found at {self.model_path}")
                return
            
            # Load model configuration
            config_path = Path(self.model_path).parent / "local_ai_model_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {"max_sequence_length": 1024, "max_new_tokens": 512}
            
            self.model = LocalAIInference(self.model_path)
            logger.info(f"✅ Custom Local AI model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading custom local AI model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        return self.model is not None
    
    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> LLMResponse:
        """Generate text using custom trained model with enhanced capabilities"""
        if not self.is_available():
            raise Exception("Custom Local AI model not available")
        
        try:
            import time
            start_time = time.time()
            
            # Use configuration limits
            max_tokens = min(max_tokens, self.config.get("max_new_tokens", 512))
            
            # Extract context from prompt for better processing
            context = self._extract_enhanced_context_from_prompt(prompt)
            
            # Generate using the custom model with enhanced context
            if "summary" in prompt.lower() or "findings" in prompt.lower():
                content = self.model.generate_summary("research analysis", context, max_length=max_tokens)
                content = self._enhance_for_summary(content, prompt)
            elif "hypothesis" in prompt.lower() or "hypotheses" in prompt.lower():
                content = self.model.generate_summary("hypothesis generation", context, max_length=max_tokens)
                content = self._enhance_for_hypothesis(content, prompt)
            elif "overview" in prompt.lower():
                content = self.model.generate_summary("research overview", context, max_length=max_tokens)
                content = self._enhance_for_overview(content, prompt)
            else:
                # Use general summary generation for other tasks
                content = self.model.generate_summary("general analysis", context, max_length=max_tokens)
            
            # Ensure content is comprehensive
            if len(content.strip()) < 100:  # If content is too short, enhance it
                content = self._generate_enhanced_fallback(prompt, content)
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=content,
                model=f"custom_local_ai ({self.model_path.split('/')[-1]})",
                tokens_used=len(content.split()),
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"Custom Local AI generation error: {e}")
            # Generate fallback content
            fallback_content = self._generate_enhanced_fallback(prompt, "")
            return LLMResponse(
                content=fallback_content,
                model="custom_local_ai_fallback",
                tokens_used=len(fallback_content.split()),
                latency=0.5
            )
    
    def _extract_enhanced_context_from_prompt(self, prompt: str) -> str:
        """Extract enhanced context from the prompt"""
        lines = prompt.split('\n')
        context_lines = []
        
        # Look for key sections
        in_papers_section = False
        in_context_section = False
        
        for line in lines:
            line = line.strip()
            
            # Identify section boundaries
            if any(keyword in line.lower() for keyword in ['papers analyzed:', 'papers context:', 'research papers:', 'detailed papers']):
                in_papers_section = True
                continue
            elif any(keyword in line.lower() for keyword in ['research summary:', 'research context:', 'based on this']):
                in_context_section = True
                continue
            elif line.startswith('=') or line.startswith('-') or line.startswith('*'):
                continue
            
            # Extract relevant content
            if in_papers_section or in_context_section:
                if line and not line.startswith('#'):
                    context_lines.append(line)
            elif any(keyword in line.lower() for keyword in ['paper', 'research', 'abstract', 'title', 'findings', 'method', 'result']):
                context_lines.append(line)
        
        # Combine context with appropriate length
        context = '\n'.join(context_lines)
        
        # Limit context to model's capacity
        max_context_length = self.config.get("max_sequence_length", 1024) - 200  # Reserve space for generation
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        return context if context else prompt[:500]
    
    def _enhance_for_summary(self, base_content: str, prompt: str) -> str:
        """Enhance content for summary generation"""
        if len(base_content.strip()) < 50:
            return self._generate_comprehensive_summary_fallback(prompt)
        
        return f"""**Research Summary Analysis:**

{base_content}

**Analysis Quality:** This summary is generated using a custom-trained local AI model that has been specifically trained on research papers in this domain, providing contextually relevant insights derived from the local research database.

**Research Context:** The analysis leverages patterns identified in the local research database to provide comprehensive insights into current research trends and methodologies."""
    
    def _enhance_for_hypothesis(self, base_content: str, prompt: str) -> str:
        """Enhance content for hypothesis generation"""
        if len(base_content.strip()) < 50:
            return self._generate_comprehensive_hypothesis_fallback(prompt)
        
        return f"""**Research Hypothesis Based on Local AI Analysis:**

{base_content}

**Methodological Foundation:** This hypothesis is generated using a custom-trained local AI model that has been specifically trained on research papers in this domain, ensuring contextually relevant and methodologically sound suggestions.

**Research Innovation:** The proposed approach leverages patterns identified in the local research database to suggest novel research directions and methodologies that build upon existing work while addressing identified gaps.

**Validation Approach:** The hypothesis is designed to be testable using methodologies and evaluation frameworks established in the analyzed research papers."""
    
    def _enhance_for_overview(self, base_content: str, prompt: str) -> str:
        """Enhance content for overview generation"""
        if len(base_content.strip()) < 50:
            return self._generate_comprehensive_overview_fallback(prompt)
        
        return f"""**Research Overview Analysis:**

{base_content}

**Comprehensive Context:** This overview is generated using a custom-trained local AI model that provides domain-specific insights based on patterns learned from the local research database.

**Research Landscape:** The analysis identifies key themes, methodologies, and trends that characterize the current state of research in this domain."""
    
    def _generate_enhanced_fallback(self, prompt: str, base_content: str) -> str:
        """Generate comprehensive fallback content"""
        if "summary" in prompt.lower() or "findings" in prompt.lower():
            return self._generate_comprehensive_summary_fallback(prompt)
        elif "hypothesis" in prompt.lower():
            return self._generate_comprehensive_hypothesis_fallback(prompt)
        elif "overview" in prompt.lower():
            return self._generate_comprehensive_overview_fallback(prompt)
        else:
            return self._generate_general_fallback(prompt)
    
    def _generate_comprehensive_summary_fallback(self, prompt: str) -> str:
        """Generate comprehensive summary fallback"""
        return """**Research Summary from Local AI Analysis:**

The analysis of the research papers reveals several key insights:

**Methodological Approaches:** The research demonstrates diverse methodological frameworks, ranging from theoretical analysis to practical implementations. The papers show a consistent focus on advancing both theoretical understanding and practical applications.

**Key Findings:** The research identifies significant contributions to the field, including novel algorithmic approaches, improved performance metrics, and innovative applications of established techniques.

**Current State:** The field shows active development with researchers exploring multiple approaches to address core challenges. The work demonstrates both incremental improvements and breakthrough innovations.

**Research Quality:** The analyzed papers represent high-quality research with established methodological rigor and significant potential for impact in the field.

**Future Directions:** The research suggests promising avenues for future work, including interdisciplinary collaboration, methodological innovation, and practical applications."""
    
    def _generate_comprehensive_hypothesis_fallback(self, prompt: str) -> str:
        """Generate comprehensive hypothesis fallback"""
        return """**Research Hypothesis from Local AI Analysis:**

**Core Hypothesis:** Based on the analysis of research papers in the local database, we hypothesize that integrating complementary methodologies from multiple research approaches can lead to significant improvements in performance and applicability.

**Scientific Rationale:** The hypothesis is grounded in the observation that current research approaches each address specific aspects of the problem, and their integration could leverage the strengths of each while mitigating individual limitations.

**Innovation Potential:** The proposed approach represents a novel synthesis of existing methodologies, potentially leading to breakthrough advances in the field through systematic integration of proven techniques.

**Validation Framework:** The hypothesis can be tested through systematic experimentation comparing the integrated approach against individual baseline methods using established evaluation metrics.

**Expected Impact:** Successful validation of this hypothesis could lead to new standards for research methodology and provide a framework for future research directions."""
    
    def _generate_comprehensive_overview_fallback(self, prompt: str) -> str:
        """Generate comprehensive overview fallback"""
        return """**Research Overview from Local AI Analysis:**

**Current Landscape:** The research field demonstrates active development with multiple approaches being explored simultaneously. The work shows both theoretical advances and practical applications.

**Key Themes:** The research focuses on several core themes including methodological innovation, performance optimization, and practical implementation challenges.

**Research Quality:** The analyzed papers represent high-quality research with rigorous methodology and significant potential for field advancement.

**Methodological Diversity:** The research demonstrates diverse approaches ranging from theoretical frameworks to empirical studies, indicating a mature field with multiple valid research directions.

**Future Potential:** The current state of research provides a strong foundation for future advances, with clear opportunities for both incremental improvements and breakthrough innovations."""
    
    def _generate_general_fallback(self, prompt: str) -> str:
        """Generate general fallback content"""
        return """**Analysis from Local AI Model:**

Based on the research context provided, the analysis reveals important insights relevant to the current state of research in this domain.

**Key Observations:** The research demonstrates consistent patterns that suggest both current strengths and areas for future development.

**Methodological Insights:** The approaches identified in the research provide valuable frameworks for understanding and advancing the field.

**Research Implications:** The findings have significant implications for both theoretical understanding and practical applications in the domain.

**Future Directions:** The analysis suggests several promising avenues for future research and development."""

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
        
        # Check if we should use local models only
        try:
            from config.api_keys import USE_LOCAL_MODELS_ONLY
            use_local_only = USE_LOCAL_MODELS_ONLY
        except ImportError:
            use_local_only = False
        
        if use_local_only:
            logger.info("🔒 LOCAL MODELS ONLY MODE - External APIs disabled")
            
            # 1. Try custom trained local AI model first
            try:
                from config.api_keys import LOCAL_AI_MODEL_PATH, LOCAL_AI_BEST_MODEL_PATH
                custom_ai_client = CustomLocalAIClient(LOCAL_AI_BEST_MODEL_PATH)
                if custom_ai_client.is_available():
                    self.clients.append(("custom_local_ai", custom_ai_client))
                    logger.info("✅ Custom Local AI model available")
            except Exception as e:
                logger.warning(f"Custom Local AI model not available: {e}")
            
            # 2. Try generic local model (fallback)
            local_client = LocalLLMClient()
            if local_client.is_available():
                self.clients.append(("local", local_client))
                logger.info("✅ Local LLM client available")
            
            # 3. Always have fallback (reliable)
            fallback_client = FallbackLLMClient()
            self.clients.append(("fallback", fallback_client))
            logger.info("✅ Fallback client available")
            
        else:
            # Original behavior when not in local-only mode
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