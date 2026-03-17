"""Report Verifier - Validates data quality from profilers.

Ensures collected profiling data is valid and useful before passing
to the Coordinator for SME analysis.
"""

from typing import Any, Dict, List, Optional

from .profilers.base import DataQualityError


class ReportVerifier:
    """Verifies quality of profiling data.
    
    Checks:
    - Data is not empty/all None
    - Required extractors have values
    - Values are in reasonable ranges
    - No obvious corruption
    
    Usage:
        verifier = ReportVerifier()
        try:
            verifier.verify(result, required_extractors)
        except DataQualityError as e:
            # Handle invalid data
    """
    
    # Reasonable ranges for common metrics
    # Values outside these ranges trigger warnings
    REASONABLE_RANGES = {
        "gpu_utilization_percent": (0.0, 100.0),
        "memory_bandwidth_utilization_percent": (0.0, 100.0),
        "compute_utilization_percent": (0.0, 100.0),
        "sm_utilization_percent": (0.0, 100.0),
        "occupancy_percent": (0.0, 100.0),
        "kv_cache_usage_percent": (0.0, 100.0),
        "memory_throughput_gbps": (0.0, 2000.0),
        "max_num_seqs": (1, 10000),
        "tensor_parallel_size": (1, 16),
        "pipeline_parallel_size": (1, 16),
    }
    
    def __init__(self, strict: bool = False):
        """Initialize verifier.
        
        Args:
            strict: If True, warnings become errors
        """
        self.strict = strict
        self.warnings: List[str] = []
    
    def verify(
        self,
        data: Dict[str, Any],
        required_extractors: List[str],
        data_type: str = ""
    ) -> bool:
        """Verify profiling data quality.
        
        Args:
            data: Extracted data from profiler
            required_extractors: List of extractors that must have values
            data_type: Type of data (for error messages)
            
        Returns:
            True if verification passes
            
        Raises:
            DataQualityError: If verification fails and strict=True
                             or critical issues found
        """
        self.warnings = []
        prefix = f"[{data_type}] " if data_type else ""
        
        # Check 1: Data not empty
        if not data:
            raise DataQualityError(f"{prefix}No data extracted")
        
        # Check 2: All required extractors present
        missing = self._check_required_extractors(data, required_extractors)
        if missing:
            msg = f"{prefix}Missing required extractors: {', '.join(missing)}"
            if self.strict:
                raise DataQualityError(msg)
            self.warnings.append(msg)
        
        # Check 3: No all-None values for required extractors
        none_values = self._check_none_values(data, required_extractors)
        if none_values:
            msg = f"{prefix}Required extractors have None values: {', '.join(none_values)}"
            if self.strict:
                raise DataQualityError(msg)
            self.warnings.append(msg)
        
        # Check 4: Values in reasonable ranges
        out_of_range = self._check_ranges(data)
        for extractor, value, (min_val, max_val) in out_of_range:
            msg = f"{prefix}Value {value} for '{extractor}' outside typical range [{min_val}, {max_val}]"
            if self.strict:
                raise DataQualityError(msg)
            self.warnings.append(msg)
        
        # Check 5: No suspicious patterns
        suspicious = self._check_suspicious_patterns(data)
        for pattern in suspicious:
            msg = f"{prefix}Suspicious pattern: {pattern}"
            if self.strict:
                raise DataQualityError(msg)
            self.warnings.append(msg)
        
        return len(self.warnings) == 0 or not self.strict
    
    def _check_required_extractors(
        self,
        data: Dict[str, Any],
        required: List[str]
    ) -> List[str]:
        """Check if all required extractors are present.
        
        Args:
            data: Extracted data
            required: Required extractor names
            
        Returns:
            List of missing extractor names
        """
        return [e for e in required if e not in data]
    
    def _check_none_values(
        self,
        data: Dict[str, Any],
        required: List[str]
    ) -> List[str]:
        """Check for None values in required extractors.
        
        Args:
            data: Extracted data
            required: Required extractor names
            
        Returns:
            List of extractor names with None values
        """
        return [e for e in required if e in data and data[e] is None]
    
    def _check_ranges(
        self,
        data: Dict[str, Any]
    ) -> List[tuple]:
        """Check if values are in reasonable ranges.
        
        Args:
            data: Extracted data
            
        Returns:
            List of (extractor, value, expected_range) tuples for outliers
        """
        out_of_range = []
        
        for extractor, value in data.items():
            if extractor not in self.REASONABLE_RANGES:
                continue
            
            if value is None:
                continue
            
            try:
                num_value = float(value)
            except (TypeError, ValueError):
                continue
            
            min_val, max_val = self.REASONABLE_RANGES[extractor]
            if num_value < min_val or num_value > max_val:
                out_of_range.append((extractor, num_value, (min_val, max_val)))
        
        return out_of_range
    
    def _check_suspicious_patterns(
        self,
        data: Dict[str, Any]
    ) -> List[str]:
        """Check for suspicious patterns in data.
        
        Args:
            data: Extracted data
            
        Returns:
            List of suspicious pattern descriptions
        """
        patterns = []
        
        # Check for all zeros
        numeric_values = []
        for k, v in data.items():
            if isinstance(v, (int, float)) and not k.endswith("_error"):
                numeric_values.append(v)
        
        if numeric_values and all(v == 0 for v in numeric_values):
            patterns.append("All numeric values are zero")
        
        # Check for all percentages being exactly the same
        percent_values = [
            v for k, v in data.items()
            if isinstance(v, (int, float)) and "percent" in k
        ]
        if len(percent_values) > 1 and len(set(percent_values)) == 1:
            patterns.append(f"All percentage values are identical ({percent_values[0]})")
        
        # Check for negative values where they shouldn't be
        for k, v in data.items():
            if isinstance(v, (int, float)) and v < 0:
                if any(x in k for x in ["percent", "count", "time", "bytes"]):
                    patterns.append(f"Negative value for '{k}': {v}")
        
        return patterns
    
    def verify_completeness(
        self,
        collected_data: Dict[str, Dict[str, Any]],
        requirements: Dict[str, Dict[str, Any]]
    ) -> tuple:
        """Verify all required data types were collected.
        
        Args:
            collected_data: Dict mapping data_type -> extracted data
            requirements: From SMERegistry.get_profiler_requirements()
            
        Returns:
            Tuple of (is_complete, missing_required_list)
        """
        missing = []
        
        for data_type, req_info in requirements.items():
            is_required = req_info.get("required", False)
            
            if not is_required:
                continue
            
            if data_type not in collected_data:
                missing.append(data_type)
                continue
            
            data = collected_data[data_type]
            if not data:
                missing.append(data_type)
                continue
        
        return len(missing) == 0, missing
    
    def get_warnings(self) -> List[str]:
        """Get list of verification warnings."""
        return self.warnings.copy()
    
    def has_warnings(self) -> bool:
        """Check if any warnings were generated."""
        return len(self.warnings) > 0
