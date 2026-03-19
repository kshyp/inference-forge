"""Base async LLM client interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import asyncio
import time
import warnings

# Suppress Google SDK deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')
warnings.filterwarnings('ignore', category=FutureWarning, module='google.generativeai')


@dataclass
class Prompt:
    """Structured prompt for LLM."""
    system: str
    user: str
    # Optional: additional context that might be provider-specific
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    provider: str
    
    # Usage stats
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Timing
    latency_ms: float = 0.0
    
    # Raw for debugging
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)


class LLMClient(ABC):
    """Abstract async LLM client."""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
    
    @abstractmethod
    async def complete(self, prompt: Prompt, deterministic: bool = False) -> LLMResponse:
        """
        Send prompt to LLM and return standardized response.
        
        Args:
            prompt: The prompt to send
            deterministic: If True, use temperature=0 for reproducible results
        
        Returns:
            LLMResponse with content that should be valid JSON
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return provider capabilities."""
        pass
    
    async def complete_with_retry(
        self, 
        prompt: Prompt,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> LLMResponse:
        """Complete with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return await self.complete(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
        raise RuntimeError("Should not reach here")


class AnthropicClient(LLMClient):
    """Anthropic Claude client."""
    
    PROVIDER = "anthropic"
    
    # Model quality tiers (for consensus weighting)
    MODEL_QUALITY = {
        "claude-opus-4-1-20250805": 1.0,
        "claude-opus-4-20250514": 0.98,
        "claude-opus-4-5-20251101": 0.97,
        "claude-opus-4-6": 0.96,
        "claude-sonnet-4-20250514": 0.95,
        "claude-sonnet-4-5-20250929": 0.94,
        "claude-sonnet-4-6": 0.93,
        "claude-haiku-4-5-20251001": 0.85,
    }
    
    async def complete(self, prompt: Prompt, deterministic: bool = False) -> LLMResponse:
        """Call Claude API."""
        # Lazy import to avoid dependency if not used
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")
        
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        start_time = time.time()
        
        # Build request kwargs
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.user}],
        }
        
        # Use temperature=0 for deterministic results
        if deterministic:
            kwargs["temperature"] = 0.0
        
        # Add response format parameters
        kwargs.update(self._get_response_format())
        
        response = await client.messages.create(**kwargs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        content = response.content[0].text if response.content else ""
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.PROVIDER,
            prompt_tokens=response.usage.input_tokens if response.usage else None,
            completion_tokens=response.usage.output_tokens if response.usage else None,
            total_tokens=(response.usage.input_tokens + response.usage.output_tokens) 
                        if response.usage else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
    
    def _get_response_format(self) -> Dict[str, Any]:
        """Get response format parameters for this model."""
        # Claude 3.7+ supports structured output via beta header
        # For now, rely on system prompt for JSON
        return {}
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "max_tokens": 8192,
            "supports_json_mode": False,  # Via system prompt only
            "supports_vision": "claude-3" in self.model,
            "quality_tier": self.MODEL_QUALITY.get(self.model, 0.8),
        }


class OpenAIClient(LLMClient):
    """OpenAI GPT client."""
    
    PROVIDER = "openai"
    
    MODEL_QUALITY = {
        "gpt-4": 1.0,
        "gpt-4-turbo-preview": 1.0,
        "gpt-4o": 1.0,
        "gpt-4o-mini": 0.9,
        "gpt-3.5-turbo": 0.8,
    }
    
    async def complete(self, prompt: Prompt) -> LLMResponse:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user}
        ]
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        choice = response.choices[0]
        content = choice.message.content if choice.message else ""
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.PROVIDER,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            total_tokens=response.usage.total_tokens if response.usage else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "max_tokens": 4096,
            "supports_json_mode": True,
            "supports_vision": "vision" in self.model or "gpt-4o" in self.model,
            "quality_tier": self.MODEL_QUALITY.get(self.model, 0.8),
        }


class MoonshotClient(LLMClient):
    """Moonshot AI (Kimi) client."""
    
    PROVIDER = "moonshot"
    DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
    
    MODEL_QUALITY = {
        "kimi-latest": 0.9,
        "kimi-k2": 0.9,
        "kimi-k1.5": 0.85,
    }
    
    async def complete(self, prompt: Prompt, deterministic: bool = False) -> LLMResponse:
        try:
            import openai  # Moonshot is OpenAI-compatible
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url or self.DEFAULT_BASE_URL
        )
        
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }
        
        # Use temperature=0 for deterministic results
        # Note: kimi-k2.5 only supports temperature=1, so skip for that model
        if deterministic and "k2.5" not in self.model:
            kwargs["temperature"] = 0.0
            kwargs["seed"] = 42  # For reproducibility
        
        response = await client.chat.completions.create(**kwargs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        choice = response.choices[0]
        content = choice.message.content if choice.message else ""
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.PROVIDER,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            total_tokens=response.usage.total_tokens if response.usage else None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "max_tokens": 4096,
            "supports_json_mode": False,  # Check if supported
            "supports_vision": False,
            "quality_tier": self.MODEL_QUALITY.get(self.model, 0.85),
        }


class VLLMLocalClient(LLMClient):
    """Local vLLM-hosted model client."""
    
    PROVIDER = "vllm_local"
    
    # Quality depends on what model is loaded - assume decent
    MODEL_QUALITY = {
        "default": 0.75,
    }
    
    async def complete(self, prompt: Prompt) -> LLMResponse:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        
        if not self.base_url:
            raise ValueError("base_url required for vllm_local (e.g., http://localhost:8000/v1)")
        
        client = openai.AsyncOpenAI(
            api_key=self.api_key or "dummy",  # vLLM doesn't check API key locally
            base_url=self.base_url
        )
        
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user}
        ]
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        choice = response.choices[0]
        content = choice.message.content if choice.message else ""
        
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.PROVIDER,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            total_tokens=response.usage.total_tokens if response.usage else None,
            latency_ms=latency_ms,
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "max_tokens": 4096,
            "supports_json_mode": False,
            "supports_vision": False,
            "quality_tier": self.MODEL_QUALITY.get("default", 0.75),
            "local": True,
        }


class GeminiClient(LLMClient):
    """Google Gemini client."""
    
    PROVIDER = "gemini"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    MODEL_QUALITY = {
        "gemini-2.5-pro": 0.95,
        "gemini-2.5-flash": 0.95,
        "gemini-2.0-flash": 0.90,
        "gemini-2.0-flash-lite": 0.85,
    }
    
    async def complete(self, prompt: Prompt, deterministic: bool = False) -> LLMResponse:
        """Call Gemini API."""
        # Try newer google-genai first, fall back to older google-generativeai
        try:
            from google import genai
            from google.genai import types
            
            # New SDK style (google-genai)
            client = genai.Client(api_key=self.api_key)
            
            # Construct the prompt with system instruction
            full_prompt = f"{prompt.system}\n\n{prompt.user}"
            
            start_time = time.time()
            
            # Build generation config
            gen_config = types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=0.0 if deterministic else 0.2,
            )
            
            # Generate response
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=gen_config,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = response.text if response.text else ""
            
            # Extract usage info if available
            usage = response.usage_metadata
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER,
                prompt_tokens=usage.prompt_token_count if usage else None,
                completion_tokens=usage.candidates_token_count if usage else None,
                total_tokens=usage.total_token_count if usage else None,
                latency_ms=latency_ms,
            )
            
        except ImportError:
            # Fall back to older google-generativeai
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "Google AI SDK required. Install either:\n"
                    "  pip install google-genai  (recommended)\n"
                    "  pip install google-generativeai  (legacy)"
                )
            
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Create model instance
            model = genai.GenerativeModel(self.model)
            
            # Construct the prompt with system instruction
            full_prompt = f"{prompt.system}\n\n{prompt.user}"
            
            start_time = time.time()
            
            # Generate response
            generation_config = {
                "max_output_tokens": 4096,
                "temperature": 0.0 if deterministic else 0.2,
            }
            
            response = await model.generate_content_async(
                full_prompt,
                generation_config=generation_config
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = response.text if response.text else ""
            
            # Extract usage info if available
            usage_metadata = getattr(response, 'usage_metadata', None)
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            
            if usage_metadata:
                prompt_tokens = getattr(usage_metadata, 'prompt_token_count', None)
                completion_tokens = getattr(usage_metadata, 'candidates_token_count', None)
                total_tokens = getattr(usage_metadata, 'total_token_count', None)
            
            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.PROVIDER,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "max_tokens": 8192,
            "supports_json_mode": False,  # Via prompt engineering
            "supports_vision": "vision" in self.model or "pro" in self.model,
            "quality_tier": self.MODEL_QUALITY.get(self.model, 0.85),
        }
