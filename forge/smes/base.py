"""Base classes for Subject Matter Expert (SME) agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class DataRequirement:
    """Data required by an SME from the profiler."""
    data_type: str       # "nsys_report", "ncu_report", "vllm_logs", etc.
    required: bool       # True = must have, False = optional
    extractors: List[str] = field(default_factory=list)  # Specific metrics to extract


@dataclass
class ExperimentSuggestion:
    """A suggested configuration experiment."""
    config_changes: Dict[str, Any]
    expected_improvement: str
    confidence: float
    rationale: str


@dataclass
class SMEResponse:
    """Return value from SME.analyze() - findings and suggestions."""
    findings: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[ExperimentSuggestion] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RegistrationInfo:
    """Returned when SME successfully registers with Coordinator."""
    sme_id: str
    triggers: List[str]                   # Signals this SME handles
    data_requirements: List[DataRequirement]  # What data it needs


class BaseSME(ABC):
    """
    Base class for Subject Matter Expert agents.
    
    Each SME:
    1. Registers with platform-specific data requirements
    2. Analyzes profiling data using AI (multi-LLM consensus) and returns 
       findings + experiment suggestions
    """
    
    @abstractmethod
    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Register with Coordinator for this platform.
        
        Args:
            platform_info: Dict with keys like 'type', 'gpu_model', 'gpu_count', 
                          'cuda_version', 'driver_version'
        
        Returns:
            RegistrationInfo if SME can work on this platform, None otherwise.
        """
        pass
    
    @abstractmethod
    async def analyze(self, profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze profiling data and suggest experiments.
        
        This method is async to enable parallel LLM calls for multi-model consensus.
        
        Args:
            profiling_data: Dict keyed by data_type with profiler outputs
            benchmark_metrics: Benchmark results (TTFT, TPOT, throughput, etc.)
        
        Returns:
            SMEResponse with findings and experiment suggestions
        """
        pass
