"""Base profiler interface for the Profile Executor Agent.

All profilers (NCU, NSys, vLLM logs, etc.) implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ProfilingContext:
    """Context passed to profilers during execution.
    
    Contains information about the vLLM instance being profiled
    and where to save outputs.
    """
    # vLLM process info
    vllm_pid: Optional[int] = None
    vllm_log_path: Optional[Path] = None
    vllm_port: int = 8000
    
    # Profiling parameters
    duration_seconds: int = 60
    warmup_seconds: int = 5
    
    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("./data/profiles"))
    experiment_id: str = ""
    
    # vLLM configuration
    config: Dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    
    # Benchmark context (for steady-state profiling)
    saturation_rate: Optional[float] = None


@dataclass
class RawProfilerOutput:
    """Raw output from a profiler run.
    
    This is the unprocessed output that will be passed to extract().
    """
    report_path: Optional[Path] = None
    stdout: str = ""
    stderr: str = ""
    log_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_mock: bool = False


@dataclass
class ProfilerResult:
    """Structured result after extraction.
    
    Contains the extracted metrics plus metadata about the run.
    """
    data_type: str  # "ncu_report", "vllm_logs", etc.
    extractors: Dict[str, Any]  # Extracted metrics
    raw_output: RawProfilerOutput
    success: bool = True
    error_message: Optional[str] = None


class ProfilerError(Exception):
    """Base profiler error."""
    pass


class ProfilerNotAvailableError(ProfilerError):
    """Profiler binary not found or permissions issue."""
    pass


class ProfilerTimeoutError(ProfilerError):
    """Profiler took too long."""
    pass


class ExtractorError(ProfilerError):
    """Failed to extract specific metric."""
    pass


class DataQualityError(ProfilerError):
    """Data collected but seems invalid (e.g., all zeros)."""
    pass


class UnsupportedDataTypeError(ProfilerError):
    """No profiler available for the requested data type on this platform."""
    pass


class BaseProfiler(ABC):
    """Abstract base class for all profilers (NCU, NSys, vLLM logs, etc.).
    
    Each profiler implements:
    - run(): Execute the profiler tool
    - extract(): Parse output and extract metrics
    - is_available(): Check if this profiler can run on this system
    
    Example:
        class MyProfiler(BaseProfiler):
            DEPENDENCIES = ["vllm_running"]
            ESTIMATED_DURATION_SECONDS = 60
            
            async def run(self, context: ProfilingContext) -> RawProfilerOutput:
                # Run profiler tool
                return RawProfilerOutput(report_path=path)
            
            async def extract(self, raw_output, extractors):
                # Parse and extract metrics
                return {"metric": value}
            
            def is_available(self) -> bool:
                return shutil.which("my_tool") is not None
    """
    
    # Dependencies that must be satisfied before this profiler runs
    # e.g., ["vllm_running"] means vLLM must be started first
    DEPENDENCIES: List[str] = []
    
    # Estimated runtime (for progress reporting and timeout)
    ESTIMATED_DURATION_SECONDS: int = 60
    
    # Data type this profiler produces
    DATA_TYPE: str = ""
    
    @abstractmethod
    async def run(self, context: ProfilingContext) -> RawProfilerOutput:
        """Execute the profiler.
        
        Args:
            context: Profiling context with vLLM info and output paths
            
        Returns:
            RawProfilerOutput with report paths and raw data
            
        Raises:
            ProfilerError: If the profiler fails to run
        """
        pass
    
    @abstractmethod
    async def extract(
        self,
        raw_output: RawProfilerOutput,
        extractors: List[str]
    ) -> Dict[str, Any]:
        """Parse raw output and extract requested metrics.
        
        Args:
            raw_output: Output from run()
            extractors: List of metrics to extract (e.g., ["gpu_utilization", "memory_bw"])
            
        Returns:
            Dict mapping extractor name to extracted value
            
        Raises:
            ExtractorError: If extraction fails for a specific metric
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this profiler can run on this system.
        
        Returns:
            True if the profiler binary/tool is available
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get profiler metadata.
        
        Returns:
            Dict with profiler information
        """
        return {
            "profiler_id": self.__class__.__name__,
            "data_type": self.DATA_TYPE,
            "dependencies": self.DEPENDENCIES,
            "estimated_duration_seconds": self.ESTIMATED_DURATION_SECONDS,
            "available": self.is_available(),
        }
