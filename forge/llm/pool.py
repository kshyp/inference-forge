"""Intelligence pool for auto-discovering and managing LLM providers."""

import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type

from .base import LLMClient, Prompt, LLMResponse
from .providers import AnthropicClient, OpenAIClient, MoonshotClient, VLLMLocalClient, GeminiClient

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceSource:
    """A configured LLM provider available for use."""
    provider: str           # "anthropic", "openai", "moonshot", etc.
    model: str              # Specific model name
    client: LLMClient       # Initialized client
    capabilities: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        quality = self.capabilities.get("quality_tier", "?")
        return f"IntelligenceSource({self.provider}/{self.model}, quality={quality})"


@dataclass
class CallResult:
    """Result of a single LLM call."""
    source: IntelligenceSource
    response: Optional[LLMResponse] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.response is not None and self.error is None


class IntelligencePool:
    """
    Auto-discovers available LLM providers from environment variables.
    Manages parallel invocation across all available intelligence sources.
    """
    
    # Provider configuration: env var -> client class -> default models
    PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {
        "anthropic": {
            "env_key": "ANTHROPIC_API_KEY",
            "client_class": AnthropicClient,
            "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
            "base_url_env": None,
        },
        "openai": {
            "env_key": "OPENAI_API_KEY",
            "client_class": OpenAIClient,
            "models": ["gpt-4", "gpt-4o", "gpt-4o-mini"],
            "base_url_env": "OPENAI_BASE_URL",
        },
        "moonshot": {
            "env_key": "MOONSHOT_API_KEY",
            "client_class": MoonshotClient,
            "models": ["kimi-k2-0711-preview", "kimi-k2-turbo-preview", "kimi-k2.5"],
            "base_url_env": "MOONSHOT_BASE_URL",
        },
        "vllm_local": {
            "env_key": "VLLM_LOCAL_URL",  # Different: this is the URL itself
            "client_class": VLLMLocalClient,
            "models": ["local-model"],  # vLLM can list models, but we use default
            "base_url_env": "VLLM_LOCAL_URL",
        },
        "gemini": {
            "env_key": "GEMINI_API_KEY",
            "client_class": GeminiClient,
            "models": ["gemini-2.5-flash"],  # Only model currently working
            "base_url_env": None,  # Uses default
        },
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pool and auto-discover sources.
        
        Args:
            config: Optional overrides for provider registry
        """
        self.sources: List[IntelligenceSource] = []
        self.config = config or {}
        self._discover_sources()
    
    def _discover_sources(self) -> None:
        """Auto-discover available intelligence from environment variables."""
        print("[IntelligencePool] Discovering LLM sources from environment...")
        
        for provider_name, cfg in self.PROVIDER_REGISTRY.items():
            env_key = cfg["env_key"]
            env_value = os.environ.get(env_key)
            
            if not env_value:
                logger.debug(f"Skipping {provider_name}: {env_key} not set")
                continue
            
            # Get base URL if applicable
            base_url = None
            if cfg["base_url_env"]:
                base_url = os.environ.get(cfg["base_url_env"])
            elif provider_name == "vllm_local":
                base_url = env_value  # For vllm_local, the env var IS the URL
            
            # Initialize each model for this provider
            for model in cfg["models"]:
                try:
                    client_class: Type[LLMClient] = cfg["client_class"]
                    client = client_class(
                        api_key=env_value if provider_name != "vllm_local" else "dummy",
                        model=model,
                        base_url=base_url
                    )
                    
                    capabilities = client.get_capabilities()
                    
                    source = IntelligenceSource(
                        provider=provider_name,
                        model=model,
                        client=client,
                        capabilities=capabilities
                    )
                    
                    self.sources.append(source)
                    print(f"   ✓ Discovered: {provider_name}/{model} (quality={capabilities.get('quality_tier', '?')})")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider_name}/{model}: {e}")
                    print(f"   ✗ Failed to initialize {provider_name}/{model}: {e}")
        
        if not self.sources:
            logger.error(
                "No intelligence sources available! "
                "Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "MOONSHOT_API_KEY, GEMINI_API_KEY, or VLLM_LOCAL_URL"
            )
            print("   ⚠️ No LLM sources available! Set at least one API key.")
        else:
            logger.info(f"Intelligence pool ready with {len(self.sources)} sources")
            print(f"   → Total: {len(self.sources)} intelligence source(s) ready")
    
    def get_sources(
        self,
        providers: Optional[List[str]] = None,
        min_quality: Optional[float] = None,
    ) -> List[IntelligenceSource]:
        """
        Get filtered list of sources.
        
        Args:
            providers: Only include these providers (e.g., ["anthropic", "openai"])
            min_quality: Minimum quality tier (0.0-1.0)
        
        Returns:
            Filtered list of sources
        """
        result = self.sources
        
        if providers:
            result = [s for s in result if s.provider in providers]
        
        if min_quality is not None:
            result = [
                s for s in result 
                if s.capabilities.get("quality_tier", 0) >= min_quality
            ]
        
        return result
    
    async def call_all(
        self,
        prompt: Prompt,
        providers: Optional[List[str]] = None,
        timeout_seconds: float = 60.0,
        deterministic: bool = False
    ) -> List[CallResult]:
        """
        Call all matching sources in parallel.
        
        Args:
            prompt: The prompt to send
            providers: Optional filter for specific providers
            timeout_seconds: Max time to wait for each call
            deterministic: If True, use temperature=0 for reproducible results
        
        Returns:
            List of CallResult (one per source, may include errors)
        """
        sources = self.get_sources(providers=providers)
        
        if not sources:
            logger.warning("No intelligence sources available for call_all")
            return []
        
        # Create tasks for parallel execution
        tasks = [
            self._call_single(source, prompt, timeout_seconds, deterministic)
            for source in sources
        ]
        
        # Gather all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        call_results = []
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                call_results.append(CallResult(
                    source=source,
                    error=str(result)
                ))
            else:
                call_results.append(result)
        
        # Log summary
        successes = sum(1 for r in call_results if r.success)
        logger.info(
            f"Parallel LLM call complete: {successes}/{len(call_results)} succeeded, "
            f"avg latency: {self._avg_latency(call_results):.0f}ms"
        )
        
        return call_results
    
    async def _call_single(
        self,
        source: IntelligenceSource,
        prompt: Prompt,
        timeout_seconds: float,
        deterministic: bool = False
    ) -> CallResult:
        """Call a single source with timeout."""
        try:
            response = await asyncio.wait_for(
                source.client.complete(prompt, deterministic=deterministic),
                timeout=timeout_seconds
            )
            return CallResult(source=source, response=response)
        except asyncio.TimeoutError:
            return CallResult(
                source=source,
                error=f"Timeout after {timeout_seconds}s"
            )
        except Exception as e:
            return CallResult(
                source=source,
                error=f"{type(e).__name__}: {str(e)}"
            )
    
    def _avg_latency(self, results: List[CallResult]) -> float:
        """Calculate average latency of successful calls."""
        latencies = [
            r.response.latency_ms 
            for r in results 
            if r.success and r.response
        ]
        return sum(latencies) / len(latencies) if latencies else 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get pool status for health checks."""
        return {
            "total_sources": len(self.sources),
            "sources": [
                {
                    "provider": s.provider,
                    "model": s.model,
                    "quality_tier": s.capabilities.get("quality_tier"),
                    "supports_json": s.capabilities.get("supports_json_mode"),
                }
                for s in self.sources
            ]
        }


# Global singleton pool
_global_pool: Optional[IntelligencePool] = None


def get_global_intelligence_pool() -> IntelligencePool:
    """Get or create global intelligence pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = IntelligencePool()
    return _global_pool


def reset_global_pool() -> None:
    """Reset global pool (useful for testing)."""
    global _global_pool
    _global_pool = None
