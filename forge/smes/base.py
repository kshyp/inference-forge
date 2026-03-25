"""Base classes for Subject Matter Expert (SME) agents."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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
    # Is this SME relevant to the current data?
    is_relevant: bool = True
    relevance_score: float = 0.0
    relevance_reason: str = ""


@dataclass
class RegistrationInfo:
    """Returned when SME successfully registers with Coordinator."""
    sme_id: str
    # Data types this SME can analyze (for profiler planning)
    data_requirements: List[DataRequirement]
    # Human-readable description of this SME's expertise
    description: str = ""


class BaseSME(ABC):
    """
    Base class for Subject Matter Expert agents.
    
    Each SME:
    1. Registers with platform-specific data requirements
    2. Scans profiling data to determine relevance (is this data relevant to my expertise?)
    3. Analyzes relevant profiling data and returns findings + experiment suggestions
    
    The Coordinator NO LONGER decides which SME gets triggered. Instead:
    - All profiling data is available in the profile directory
    - Each SME scans the data to determine if it reveals bottlenecks in their domain
    - SMEs return empty suggestions if data is not relevant to their expertise
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
    async def analyze(self, 
                      profile_dir: Path,
                      profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze profiling data and suggest experiments.
        
        The SME should:
        1. First scan the data to determine if it's relevant (use scan_data())
        2. If not relevant, return SMEResponse with is_relevant=False and empty suggestions
        3. If relevant, perform full analysis and return suggestions
        
        This method is async to enable parallel LLM calls for multi-model consensus.
        
        Args:
            profile_dir: Path to directory containing all profiling data files
                         (e.g., experiments/data/profile/{experiment_id}/)
            profiling_data: Pre-loaded profiling data (vllm_logs, nsys_report, etc.)
            benchmark_metrics: Benchmark results (TTFT, TPOT, throughput, etc.)
        
        Returns:
            SMEResponse with findings, suggestions, and relevance information
        """
        pass
    
    def scan_data(self, 
                  profile_dir: Path,
                  profiling_data: Dict[str, Any],
                  benchmark_metrics: Dict[str, Any]) -> tuple[bool, float, str]:
        """
        Scan profiling data to determine if this SME is relevant.
        
        Each SME implements its own logic to determine relevance:
        - QuantizationSME: Look for memory pressure, bandwidth issues, FP16 models
        - SchedulingSME: Look for queue buildup, low GPU utilization, high TTFT
        - SpeculativeSME: Look for high TPOT, decode phase slowness
        - etc.
        
        Args:
            profile_dir: Path to directory containing profiling data files
            profiling_data: Pre-loaded profiling data
            benchmark_metrics: Benchmark results
            
        Returns:
            Tuple of (is_relevant, relevance_score, reason)
            - is_relevant: True if this SME should analyze this data
            - relevance_score: 0.0-1.0 indicating confidence in relevance
            - reason: Human-readable explanation of why SME is/isn't relevant
        """
        # Default implementation: always relevant
        # Subclasses should override with domain-specific logic
        return True, 0.5, "Default relevance - SME did not implement scan_data()"
