"""LLM multi-provider support with consensus for Inference Forge."""

from .base import LLMClient, Prompt, LLMResponse
from .pool import (
    IntelligencePool, 
    IntelligenceSource, 
    get_global_intelligence_pool,
    reset_global_pool,
    CallResult,
)
from .consensus import (
    ConsensusEngine,
    ConsensusResult,
    ConsensusSuggestion,
    ModelResponse,
    ConsensusConfig,
)

__all__ = [
    "LLMClient",
    "Prompt",
    "LLMResponse",
    "IntelligencePool",
    "IntelligenceSource",
    "get_global_intelligence_pool",
    "reset_global_pool",
    "CallResult",
    "ConsensusEngine",
    "ConsensusResult",
    "ConsensusSuggestion",
    "ModelResponse",
    "ConsensusConfig",
]
